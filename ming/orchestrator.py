from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List

from ming.core.config import create_queries_store_from_config, create_redis_from_config
from ming.core.outline_parser import extract_outline_block, outline_to_sections
from ming.core.redis import QueryStoreConfig, RedisDatabaseConfig
from ming.scout import ScoutSubagent
from ming.subagent import ResearchSubagent, Agent, AgentConfig, AgentResult
from ming.extraction.kg_module import KGRedisStore, ERConfig
from ming.tools.kg_query_tool import KGQueryTool
from ming.writer_agent import WriterAgent, WriterAgentConfig
from ming.extraction.ner_re_pipeline import NERREPipeline
from ming.models import OpenRouterModelConfig
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
    draft_output_path: str | None = None
    num_research_subagents: int = 3
    outline_max_context_ids: int = 180
    outline_context_token_budget: int = 128_000
    outline_source_char_limit: int = 2500


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
            re_config=OpenRouterModelConfig(model_name="google/gemini-2.5-flash-lite"),
            kg_store=self.kg_store,
            ner_model_name="en_core_web_sm",
            max_workers=32,
        )

        self.scout = ScoutSubagent(
            config.scout_config
        )

        self.research_subagents = [ResearchSubagent(
            config=config.subagent_config,
            database=self.context_redis,
            query_store=self.queries_redis,
        ) for _ in range(config.num_research_subagents)]

        self.planning_agent = Agent(
            config=AgentConfig(
                model=OpenRouterModelConfig(
                    model_name="qwen/qwen3.5-plus-02-15",
                    temperature=0.2,
                    max_tokens=1024,
                ),
                system_prompt=PLANNING_PROMPT
            )
        )

        self.outline_agent = Agent(
            config=AgentConfig(
                model=OpenRouterModelConfig(
                    model_name="qwen/qwen3.5-plus-02-15",
                    temperature=0.2,
                    max_tokens=8192,
                ),
                system_prompt=OUTLINE_PROMPT
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
                kg_query_tool=self.kg_query_tool,
                draft_output_path=config.draft_output_path,
            )
        )

    @staticmethod
    def _extract_agent_output(result: str | AgentResult) -> str:
        if isinstance(result, AgentResult):
            return result.output
        return result

    @staticmethod
    def _parse_planning_result(planning_result: str | AgentResult) -> Dict[str, Any]:
        # Parse the planning result using xml.etree.ElementTree
        raw_output = MingDeepResearch._extract_agent_output(planning_result)
        try:
            # Handle potential markdown wrapping or extra whitespace
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                # Remove markdown fences if present
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            
            root = xml.etree.ElementTree.fromstring(cleaned)
        except xml.etree.ElementTree.ParseError as e:
            logger.error(f"Failed to parse planning result XML: {e}\nResult: {raw_output}")
            # Return a minimal valid structure to prevent total failure
            return {"research_angles": [], "constraints": ""}

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
            "constraints": constraints
        }

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
        planning_result = self.planning_agent.run(scout_result["landscape_brief"])
        research_plan = self._parse_planning_result(planning_result)
        end_time = time.time()
        logger.info(f"Planning research angles took {round(end_time - start_time, 2)} seconds")
        
        # 3. Parallel Execution of Research Subagents
        logger.info(f"Executing {len(research_plan['research_angles'])} research angles in parallel...")
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.num_research_subagents) as executor:
            future_to_angle = {}
            for angle in research_plan["research_angles"]:
                topic_prompt = (
                    f"Topic: {angle['topic']}\n"
                    f"Success Criteria: {angle['success_criteria']}\n"
                    f"Constraints: {research_plan['constraints']}"
                )
                
                subagent = ResearchSubagent(
                    config=self.config.subagent_config,
                    database=self.context_redis,
                    query_store=self.queries_redis,
                )
                
                future = executor.submit(subagent.run, topic_prompt, scout_result["landscape_brief"])
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
                source_inputs.append((entry["raw_content"], entry.get("url", "")))

        try:
            all_kg_entities.extend(self.nerre_pipeline.run_batch(source_inputs))
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
            context_ids=all_context_ids,
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
            outline_result = self.outline_agent.run(initial_context)
            try:
                outline_xml = extract_outline_block(self._extract_agent_output(outline_result))
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
    result = orchestrator.run("Acting as an expert in K-12 education research and an experienced frontline teacher, research and analyze global case studies on the practical application of AIGC (AI-Generated Content) in primary and secondary school classrooms. Identify, categorize, and analyze various application approaches and their corresponding examples. The final report should present an overall framework, detailed category discussions, practical implementation methods, future trends, and recommendations for educators.")
    end_time = time.time()
    print(f"\n\nTime taken: {round(end_time - start_time, 2)/60} minutes")