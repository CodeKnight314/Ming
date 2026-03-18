from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import html
import logging
from queue import Queue
import re
from typing import Any, Dict, List

from ming.core.config import create_queries_store_from_config, create_redis_from_config
from ming.core.outline_parser import extract_outline_block, outline_to_sections
from ming.core.redis import QueryStoreConfig, RedisDatabaseConfig
from ming.scout import ScoutSubagent
from ming.subagent import ResearchSubagent
from ming.extraction.kg_module import KGRedisStore, ERConfig
from ming.tools.kg_query_tool import KGQueryTool
from ming.writer_agent import WriterAgent, WriterAgentConfig
from ming.extraction.ner_re_pipeline import NERREPipeline
from ming.models import OpenRouterModel, OpenRouterModelConfig
from ming.core.prompts import PLANNING_PROMPT, OUTLINE_PROMPT
import xml.etree.ElementTree
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class MingDeepResearchConfig:
    redis_config: RedisDatabaseConfig
    scout_config: dict[str, Any]
    subagent_config: dict[str, Any]
    queries_redis_config: QueryStoreConfig | dict[str, Any] | None = None
    kg_redis_config: RedisDatabaseConfig | dict[str, Any] | None = None
    writer_model: OpenRouterModelConfig | None = None
    writer_fallback_model: OpenRouterModelConfig | None = None
    draft_output_path: str | None = None
    num_research_subagents: int = 3
    outline_max_context_ids: int = 180
    outline_context_token_budget: int = 128_000
    outline_source_char_limit: int = 2500
    source_min_tokens: int = 400
    research_source_budget: int = 250
    max_chunks_per_source: int = 8
    source_score_cutoff: float = 4.5


class MingDeepResearch:
    _CHARS_PER_TOKEN_ESTIMATE = 4

    def __init__(self, config: MingDeepResearchConfig):
        self.config = config
        redis_cfg = {
            "redis": {
                "hostname": config.redis_config.hostname,
                "port": config.redis_config.port,
                "db": config.redis_config.db,
            }
        }
        self.context_redis = create_redis_from_config(redis_cfg)
        queries_cfg = {
            "redis": redis_cfg["redis"],
            "queries_redis": (
                {
                    "hostname": config.queries_redis_config.hostname,
                    "port": config.queries_redis_config.port,
                    "db": config.queries_redis_config.db,
                }
                if isinstance(config.queries_redis_config, QueryStoreConfig)
                else config.queries_redis_config
            ),
        }
        self.queries_redis = create_queries_store_from_config(queries_cfg)
        
        kg_cfg = config.kg_redis_config if hasattr(config, "kg_redis_config") and config.kg_redis_config else redis_cfg
        self.kg_redis = create_redis_from_config(kg_cfg)

        self.kg_store = KGRedisStore(self.kg_redis, ERConfig(threshold=0.5, num_perm=128))
        self.kg_query_tool = KGQueryTool(self.kg_store)
        self.nerre_pipeline = NERREPipeline(
            re_config=OpenRouterModelConfig(model_name="google/gemma-3-4b-it", max_tokens=1024),
            kg_store=self.kg_store,
            max_workers=32,
            source_score_cutoff=config.source_score_cutoff,
        )

        self.scout = ScoutSubagent(
            config.scout_config
        )

        subagent_config = dict(config.subagent_config)
        subagent_config["source_min_tokens"] = config.source_min_tokens
        normalized_tool_configs = []
        for tool_config in subagent_config.get("tool_configs", []) or []:
            if isinstance(tool_config, dict):
                normalized = dict(tool_config)
                if normalized.get("type") == "open_url_tool":
                    normalized.setdefault("min_tokens", config.source_min_tokens)
                normalized_tool_configs.append(normalized)
            else:
                normalized_tool_configs.append(tool_config)
        subagent_config["tool_configs"] = normalized_tool_configs
        self.subagent_config = subagent_config

        self.research_subagents = [ResearchSubagent(
            config=self.subagent_config,
            database=self.context_redis,
            query_store=self.queries_redis,
        ) for _ in range(config.num_research_subagents)]

        self.planning_model = OpenRouterModel(
            OpenRouterModelConfig(
                model_name="qwen/qwen3.5-plus-02-15",
                temperature=0.2,
                max_tokens=1024,
            )
        )

        self.outline_model = OpenRouterModel(
            OpenRouterModelConfig(
                model_name="qwen/qwen3.5-plus-02-15",
                temperature=0.2,
                max_tokens=8192,
            )
        )

        self.writer_agent = WriterAgent(
            WriterAgentConfig(
                model=config.writer_model
                or OpenRouterModelConfig(
                    model_name="qwen/qwen3.5-plus-02-15",
                    temperature=0.2,
                    max_tokens=4096,
                ),
                fallback_model=config.writer_fallback_model,
                kg_query_tool=self.kg_query_tool,
                draft_output_path=config.draft_output_path,
            )
        )

    @staticmethod
    def _generate_with_system_prompt(
        model: OpenRouterModel,
        system_prompt: str,
        user_input: str,
    ) -> str:
        prompt = "\n".join(
            [
                system_prompt.strip(),
                "",
                "User:",
                user_input.strip(),
                "",
                "Assistant:",
            ]
        )
        return model.generate(prompt).strip()

    @staticmethod
    def _strip_markdown_fences(raw_output: str) -> str:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned

    @staticmethod
    def _escape_xml_text_nodes(xml_text: str) -> str:
        def _escape_text_segment(segment: str) -> str:
            return html.escape(html.unescape(segment), quote=False)

        parts: list[str] = []
        last_end = 0
        for match in re.finditer(r"<[^>]+>", xml_text):
            text_segment = xml_text[last_end:match.start()]
            if text_segment:
                parts.append(_escape_text_segment(text_segment))
            parts.append(match.group(0))
            last_end = match.end()

        trailing = xml_text[last_end:]
        if trailing:
            parts.append(_escape_text_segment(trailing))

        return "".join(parts)

    @staticmethod
    def _parse_planning_result(planning_result: str) -> Dict[str, Any]:
        # Parse the planning result using xml.etree.ElementTree
        raw_output = planning_result
        cleaned = MingDeepResearch._strip_markdown_fences(raw_output)

        try:
            root = xml.etree.ElementTree.fromstring(cleaned)
        except xml.etree.ElementTree.ParseError as first_error:
            sanitized = MingDeepResearch._escape_xml_text_nodes(cleaned)
            try:
                root = xml.etree.ElementTree.fromstring(sanitized)
                logger.warning(
                    "Planning XML required sanitization before parsing: %s",
                    first_error,
                )
            except xml.etree.ElementTree.ParseError as second_error:
                logger.error(
                    "Failed to parse planning result XML after sanitization: %s\nResult: %s",
                    second_error,
                    raw_output,
                )
                return {"research_angles": [], "constraints": ""}

        try:
            research_angles = root.findall("research_angles/research_angle")
            constraints_el = root.find("constraints")
            constraints = constraints_el.text if constraints_el is not None and constraints_el.text else ""

            return {
                "research_angles": [
                    {
                        "topic": angle.find("topic").text if angle.find("topic") is not None else "Unknown Topic",
                        "success_criteria": angle.find("success_criteria").text if angle.find("success_criteria") is not None else "",
                    }
                    for angle in research_angles
                ],
                "constraints": constraints,
            }
        except Exception as e:
            logger.error(f"Failed to extract planning fields: {e}\nResult: {raw_output}")
            return {"research_angles": [], "constraints": ""}

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 1200) -> str:
        cleaned = (text or "").strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "\n...[truncated]..."

    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        if not text:
            return 0
        return max(1, (len(text) + cls._CHARS_PER_TOKEN_ESTIMATE - 1) // cls._CHARS_PER_TOKEN_ESTIMATE)

    def _build_outline_context(self, context_ids: List[str], scout_brief: str) -> str:
        budget_tokens = max(1, self.config.outline_context_token_budget)
        budget_chars = budget_tokens * self._CHARS_PER_TOKEN_ESTIMATE
        max_context_ids = max(0, self.config.outline_max_context_ids)
        source_char_limit = max(256, self.config.outline_source_char_limit)

        parts = ["---Truncated context---\n\n"]
        used_chars = len(parts[0])

        scout_section = f"Scout brief:\n{(scout_brief or '').strip()}"
        if scout_section.strip():
            available_chars = max(0, budget_chars - used_chars)
            if available_chars > 0:
                if len(scout_section) > available_chars:
                    scout_section = self._truncate_text(scout_section, available_chars)
                parts.append(scout_section)
                used_chars += len(scout_section)

        evidence_header = "\n\nResearch evidence:\n\n"
        if used_chars + len(evidence_header) <= budget_chars:
            parts.append(evidence_header)
            used_chars += len(evidence_header)

        included_sources = 0
        total_candidates = min(len(context_ids), max_context_ids)
        for cid in context_ids[:max_context_ids]:
            entry = self.context_redis.get_entry(cid)
            if not entry or not entry.get("raw_content"):
                continue

            content = (
                "Title: "
                + entry.get("title", "Untitled")
                + "\nContent: "
                + entry.get("raw_content", "")
            )
            if len(content) > source_char_limit:
                content = content[:source_char_limit] + "\n---[Truncated for length]---\n\n"
            content += "\n\n"

            if used_chars + len(content) > budget_chars:
                break

            parts.append(content)
            used_chars += len(content)
            included_sources += 1

        logger.info(
            "Outline context assembled candidates=%d included=%d budget_tokens=%d estimated_tokens=%d estimated_chars=%d",
            total_candidates,
            included_sources,
            budget_tokens,
            self._estimate_tokens("".join(parts)),
            used_chars,
        )
        return "".join(parts)


    def run(self, query: str) -> str:
        # 1. Scout the web for information
        import time
        logger.info(f"Scouting landscape for: {query}")
        start_time = time.time()
        scout_result = self.scout.run(query)
        end_time = time.time()
        logger.info(f"Scouting landscape for: {query} took {round(end_time - start_time, 2)} seconds")
        
        # 2. Plan the research
        logger.info("Planning research angles...")
        start_time = time.time()
        planning_result = self._generate_with_system_prompt(
            self.planning_model,
            PLANNING_PROMPT,
            scout_result["landscape_brief"],
        )
        research_plan = self._parse_planning_result(planning_result)
        end_time = time.time()
        logger.info(f"Planning research angles took {round(end_time - start_time, 2)} seconds")
        
        # 3. Parallel Execution of Research Subagents
        logger.info(f"Executing {len(research_plan['research_angles'])} research angles in parallel...")
        start_time = time.time()
        results = []
        subagent_pool: Queue[ResearchSubagent] = Queue()
        for subagent in self.research_subagents:
            subagent_pool.put(subagent)

        def _run_research_angle(topic_prompt: str, scout_brief: str) -> Dict[str, Any]:
            subagent = subagent_pool.get()
            try:
                return subagent.run(topic_prompt, scout_brief)
            finally:
                subagent_pool.put(subagent)
        
        with ThreadPoolExecutor(max_workers=self.config.num_research_subagents) as executor:
            future_to_angle = {}
            for angle in research_plan["research_angles"]:
                topic_prompt = (
                    f"Topic: {angle['topic']}\n"
                    f"Success Criteria: {angle['success_criteria']}\n"
                    f"Constraints: {research_plan['constraints']}"
                )

                future = executor.submit(
                    _run_research_angle,
                    topic_prompt,
                    scout_result["landscape_brief"],
                )
                future_to_angle[future] = angle
            
            for future in as_completed(future_to_angle):
                angle = future_to_angle[future]
                try:
                    res = future.result()
                    res["_angle_topic"] = angle["topic"]
                    res["_success_criteria"] = angle["success_criteria"]
                    results.append(res)
                    logger.info(f"Completed research for angle: {angle['topic']}")
                except Exception as e:
                    logger.error(f"Subagent failed for angle '{angle['topic']}': {e}")
        end_time = time.time()
        logger.info(f"Executing research angles took {round(end_time - start_time, 2)} seconds")
        
        total_sources = 0
        for res in results:
            total_sources += len(res.get("context_ids", []))
        logger.info(f"Total number of sources to process: {total_sources}")
        logger.info("Starting batch NER/RE extraction into Knowledge Graph...")
        start_time = time.time()
        all_context_ids: List[str] = []
        seen_context_ids = set()
        all_kg_entities = []
        for res in results:
            for cid in res.get("context_ids", []):
                if cid in seen_context_ids:
                    continue
                seen_context_ids.add(cid)
                all_context_ids.append(cid)

        source_inputs = []
        for cid in tqdm(all_context_ids):
            entry = self.context_redis.get_entry(cid)
            if entry and entry.get("raw_content"):
                source_inputs.append((entry["raw_content"], entry.get("url", ""), cid))

        collected_sources = self.nerre_pipeline.collect_sources(source_inputs)
        retained_sources = self.nerre_pipeline.filter_sources(
            collected_sources,
            min_tokens=self.config.source_min_tokens,
            source_budget=self.config.research_source_budget,
        )
        retained_context_ids = [source.context_id for source in retained_sources if source.context_id]
        logger.info(
            "Retained %d/%d sources after minimum-length filtering, score cutoff, and budget pruning",
            len(retained_sources),
            len(collected_sources),
        )
        optimized_chunks = self.nerre_pipeline.select_chunks_for_re(
            retained_sources,
            max_chunks_per_source=self.config.max_chunks_per_source,
        )
        logger.info(
            "Selected %d chunks across retained sources (max %d per source) for KG extraction",
            len(optimized_chunks),
            self.config.max_chunks_per_source,
        )

        try:
            all_kg_entities.extend(self.nerre_pipeline.run_re_on_chunks(optimized_chunks))
        except Exception as e:
            logger.warning(f"Batch NER/RE failed: {e}")
        end_time = time.time()
        logger.info(f"NER/RE extraction took {round(end_time - start_time, 2)} seconds")

        logger.info("Starting batch entity resolution...")
        start_time = time.time()
        self.kg_store.perform_entity_resolution(all_kg_entities)
        end_time = time.time()
        logger.info(f"Entity resolution took {round(end_time - start_time, 2)} seconds")
        logger.info(f"Number of relationships active in the knowledge graph: {len(self.kg_store.get_relationships())}")
        logger.info(f"Number of chunks active in the knowledge graph: {len(self.kg_store.get_chunks())}")
        logger.info(f"Number of canonical entities active in the knowledge graph: {len(self.kg_store.get_canonical_entities())}")

        # final report
        logger.info("Generating final report...")
        initial_context = self._build_outline_context(
            context_ids=retained_context_ids,
            scout_brief=scout_result["landscape_brief"],
        )

        # 4. Generate Outline
        logger.info("Generating report outline...")
        start_time = time.time()

        max_iterations = 3
        report_title = ""
        constraints_paragraph = ""
        sections = []
        for attempt in range(1, max_iterations + 1):
            outline_result = self._generate_with_system_prompt(
                self.outline_model,
                OUTLINE_PROMPT,
                initial_context,
            )
            try:
                outline_xml = extract_outline_block(outline_result)
                report_title, constraints_paragraph, sections = outline_to_sections(
                    outline_xml
                )
                break
            except Exception as e:
                logger.error(
                    "Failed to generate parseable outline on attempt %d/%d: %s",
                    attempt,
                    max_iterations,
                    e,
                )

        if len(sections) == 0:
            logger.error(
                "Failed to generate report outline after %d iterations",
                max_iterations,
            )
            return ""

        if not report_title:
            report_title = query
        end_time = time.time()
        logger.info(f"Generating report outline took {round(end_time - start_time, 2)} seconds")

        logger.info("Writing iterative markdown report...")
        start_time = time.time()
        report_markdown = self.writer_agent.run(
            report_title=report_title,
            constraints_paragraph=constraints_paragraph,
            sections=sections,
            draft_output_path=self.config.draft_output_path,
        )
        end_time = time.time()
        logger.info(f"Writing report took {round(end_time - start_time, 2)} seconds")

        return report_markdown

if __name__ == "__main__":
    from ming.core.config import create_ming_deep_research_config
    logging.basicConfig(level=logging.INFO)
    config = create_ming_deep_research_config()
    orchestrator = MingDeepResearch(config)
    import time
    start_time = time.time()
    result = orchestrator.run("I need to dynamically adjust Kubernetes (K8S) cluster node counts based on fluctuating business request volumes, ensuring resources are scaled up proactively before peak loads and scaled down promptly during troughs. The standard Cluster Autoscaler (CA) isn't suitable as it relies on pending pods and might not fit non-elastic node group scenarios. What are effective implementation strategies, best practices, or existing projects that address predictive or scheduled autoscaling for K8S nodes?")
    end_time = time.time()
    print(f"\n\nTime taken: {round(end_time - start_time, 2)/60} minutes")
