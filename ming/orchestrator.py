from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import html
import json
import logging
from queue import Queue
import re
from typing import Any, Dict, List

from ming.core.config import create_queries_store_from_config, create_redis_from_config
from ming.core.outline_parser import SectionPlan, extract_outline_block, outline_to_sections
from ming.core.redis import QueryStoreConfig, RedisDatabaseConfig
from ming.scout import ScoutSubagent
from ming.subagent import ResearchSubagent
from ming.extraction.kg_module import KGRedisStore, ERConfig
from ming.tools.kg_query_tool import KGQueryTool
from ming.writer_agent import WriterAgent, WriterAgentConfig
from ming.extraction.ner_re_pipeline import NERREPipeline
from ming.models import OpenRouterModel, OpenRouterModelConfig
from ming.core.prompts import OUTLINE_PROMPT, PLANNING_PROMPT
import xml.etree.ElementTree
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_RESEARCH_PRIMARY = "nvidia/nemotron-3-super-120b-a12b"
_DEFAULT_FLASH_FALLBACK = "qwen/qwen3.5-flash-02-23"
_DEFAULT_WRITER_PRIMARY = "qwen/qwen3.5-plus-02-15"


@dataclass
class MingDeepResearchConfig:
    redis_config: RedisDatabaseConfig
    scout_config: dict[str, Any]
    subagent_config: dict[str, Any]
    queries_redis_config: QueryStoreConfig | dict[str, Any] | None = None
    kg_redis_config: RedisDatabaseConfig | dict[str, Any] | None = None
    writer_model: OpenRouterModelConfig | None = None
    writer_fallback_model: OpenRouterModelConfig | None = None
    writer_polish_model: OpenRouterModelConfig | None = None
    outline_model: OpenRouterModelConfig | None = None
    outline_fallback_model: OpenRouterModelConfig | None = None
    draft_output_path: str | None = None
    num_research_subagents: int = 3
    outline_max_context_ids: int = 180
    outline_context_token_budget: int = 128_000
    outline_source_char_limit: int = 2500
    source_min_tokens: int = 400
    research_source_budget: int = 250
    max_sources_threshold: int = 160
    max_chunks_per_source: int = 8
    source_score_cutoff: float = 4.5
    writer_num_parallel_sections: int = 8


class MingDeepResearch:
    _CHARS_PER_TOKEN_ESTIMATE = 4

    def __init__(self, config: MingDeepResearchConfig, runtime_observer: Any | None = None):
        self.config = config
        self.runtime_observer = runtime_observer
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
            re_config=OpenRouterModelConfig(
                model_name="google/gemma-3n-e4b-it", max_new_tokens=1024
            ),
            kg_store=self.kg_store,
            max_workers=16,
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
                model_name=_DEFAULT_RESEARCH_PRIMARY,
                temperature=0.2,
                max_new_tokens=8192,
            )
        )

        outline_primary_cfg = config.outline_model or OpenRouterModelConfig(
            model_name=_DEFAULT_RESEARCH_PRIMARY,
            temperature=0.2,
            max_new_tokens=8192,
        )
        outline_fallback_cfg = config.outline_fallback_model
        if outline_fallback_cfg is None:
            outline_fallback_cfg = OpenRouterModelConfig(
                model_name=_DEFAULT_FLASH_FALLBACK,
                temperature=outline_primary_cfg.temperature,
                max_new_tokens=outline_primary_cfg.max_new_tokens,
                site_url=outline_primary_cfg.site_url,
                site_name=outline_primary_cfg.site_name,
                model_kwargs=deepcopy(outline_primary_cfg.model_kwargs)
                if outline_primary_cfg.model_kwargs
                else None,
            )
        self.outline_model = OpenRouterModel(outline_primary_cfg)
        self.outline_fallback_model = OpenRouterModel(outline_fallback_cfg)

        writer_primary_cfg = config.writer_model or OpenRouterModelConfig(
            model_name=_DEFAULT_WRITER_PRIMARY,
            temperature=0.2,
            max_new_tokens=4096,
        )
        writer_fallback_cfg = config.writer_fallback_model
        if writer_fallback_cfg is None:
            writer_fallback_cfg = OpenRouterModelConfig(
                model_name=_DEFAULT_FLASH_FALLBACK,
                temperature=writer_primary_cfg.temperature,
                max_new_tokens=writer_primary_cfg.max_new_tokens,
                site_url=writer_primary_cfg.site_url,
                site_name=writer_primary_cfg.site_name,
                model_kwargs=deepcopy(writer_primary_cfg.model_kwargs)
                if writer_primary_cfg.model_kwargs
                else None,
            )

        self.writer_agent = WriterAgent(
            WriterAgentConfig(
                model=writer_primary_cfg,
                fallback_model=writer_fallback_cfg,
                polish_model=None,
                kg_query_tool=self.kg_query_tool,
                draft_output_path=config.draft_output_path,
                num_parallel_sections=config.writer_num_parallel_sections,
            )
        )

    def _stage_started(
        self,
        stage: str,
        message: str,
        *,
        metrics: dict[str, Any] | None = None,
        active_angle_count: int | None = None,
        completed_angle_count: int | None = None,
    ) -> None:
        if self.runtime_observer is None:
            return
        self.runtime_observer.stage_transition(
            component="orchestrator",
            stage=stage,
            status="started",
            message=message,
            metrics=metrics,
            active_angle_count=active_angle_count,
            completed_angle_count=completed_angle_count,
        )

    def _stage_completed(
        self,
        stage: str,
        message: str,
        *,
        metrics: dict[str, Any] | None = None,
        active_angle_count: int | None = None,
        completed_angle_count: int | None = None,
    ) -> None:
        if self.runtime_observer is None:
            return
        self.runtime_observer.stage_transition(
            component="orchestrator",
            stage=stage,
            status="completed",
            message=message,
            metrics=metrics,
            active_angle_count=active_angle_count,
            completed_angle_count=completed_angle_count,
        )

    def _stage_progress(
        self,
        stage: str,
        message: str,
        *,
        processed: int,
        total: int,
        elapsed_seconds: float,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        if self.runtime_observer is None:
            return
        metrics = {
            f"{stage}_processed": processed,
            f"{stage}_total": total,
            f"{stage}_elapsed_seconds": round(elapsed_seconds, 3),
        }
        if extra_metrics:
            metrics.update(extra_metrics)
        self.runtime_observer.update_run(metrics=metrics)
        self.runtime_observer.emit_event(
            kind="metric_update",
            component="orchestrator",
            status="running",
            message=message,
            stage=stage,
            metrics=metrics,
        )

    def _register_angle(self, angle_id: str, topic: str, success_criteria: str) -> None:
        if self.runtime_observer is None:
            return
        self.runtime_observer.register_angle(
            angle_id=angle_id,
            topic=topic,
            success_criteria=success_criteria,
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

    def _generate_outline_with_system_prompt(self, system_prompt: str, user_input: str) -> str:
        """Outline LLM call: primary model first, then flash fallback on any failure."""
        try:
            return self._generate_with_system_prompt(
                self.outline_model, system_prompt, user_input
            )
        except Exception as exc:
            fb_name = self.outline_fallback_model.config.model_name
            logger.warning(
                "Primary outline model failed; retrying with fallback model=%s: %s",
                fb_name,
                exc,
            )
            return self._generate_with_system_prompt(
                self.outline_fallback_model, system_prompt, user_input
            )

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
    def _extract_research_plan_xml(fragment: str) -> str:
        """If the model added chatter before/after, pull the research_plan element only."""
        cleaned = MingDeepResearch._strip_markdown_fences(fragment.strip())
        match = re.search(
            r"<research_plan\b[^>]*>.*?</research_plan>",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return match.group(0).strip() if match else cleaned

    @staticmethod
    def _parse_planning_result(planning_result: str) -> Dict[str, Any]:
        # Parse the planning result using xml.etree.ElementTree
        raw_output = planning_result
        cleaned = MingDeepResearch._extract_research_plan_xml(raw_output)

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

    def _build_outline_context(self, context_ids: List[str], scout_brief: str, user_query: str = "") -> str:
        budget_tokens = max(1, self.config.outline_context_token_budget)
        budget_chars = budget_tokens * self._CHARS_PER_TOKEN_ESTIMATE
        max_context_ids = max(0, self.config.outline_max_context_ids)
        source_char_limit = max(256, self.config.outline_source_char_limit)

        parts = ["---Truncated context---\n\n"]
        used_chars = len(parts[0])

        if user_query.strip():
            query_section = f"Original user query:\n{user_query.strip()}\n\n"
            parts.append(query_section)
            used_chars += len(query_section)

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

    def _plan_research(self, query: str, scout_result: Dict[str, Any]) -> Dict[str, Any]:
        brief = (scout_result.get("landscape_brief") or "").strip()
        user_block = (
            f"Original user topic (match this language in your angle topics):\n{query.strip()}\n\n"
            f"Scout brief (landscape and evidence orientation):\n{brief}\n"
        )
        planning_result = self._generate_with_system_prompt(
            self.planning_model,
            PLANNING_PROMPT,
            user_block,
        )
        parsed = self._parse_planning_result(planning_result)
        if not parsed.get("research_angles"):
            fb_name = self.outline_fallback_model.config.model_name
            logger.warning(
                "Planning produced no parseable angles; retrying with outline fallback model=%s",
                fb_name,
            )
            planning_result = self._generate_with_system_prompt(
                self.outline_fallback_model,
                PLANNING_PROMPT,
                user_block,
            )
            parsed = self._parse_planning_result(planning_result)
        return parsed

    def _run_planning_stage(self, query: str, scout_result: Dict[str, Any]) -> Dict[str, Any]:
        self._stage_started("planning", "Planning stage started.")
        logger.info("Planning research angles...")
        import time

        start_time = time.time()
        research_plan = self._plan_research(query, scout_result)
        end_time = time.time()
        logger.info(f"Planning research angles took {round(end_time - start_time, 2)} seconds")
        self._stage_completed(
            "planning",
            "Planning stage completed.",
            metrics={
                "elapsed_seconds": round(end_time - start_time, 3),
                "research_angle_count": len(research_plan.get("research_angles", [])),
            },
        )
        return research_plan

    def _outline_system_prompt(self) -> str:
        return OUTLINE_PROMPT

    def _finalize_report(
        self,
        query: str,
        report_title: str,
        constraints_paragraph: str,
        sections: List[SectionPlan],
        draft_output_path: str | None = None,
    ) -> str:
        path = (
            draft_output_path
            if draft_output_path is not None
            else self.config.draft_output_path
        )
        return self.writer_agent.run(
            report_title=report_title,
            constraints_paragraph=constraints_paragraph,
            sections=sections,
            draft_output_path=path,
            runtime_observer=self.runtime_observer,
            user_query=query,
        )

    def run(self, query: str, *, draft_output_path: str | None = None) -> str:
        # 1. Scout the web for information
        import time
        self._stage_started("scout", "Scout stage started.")
        logger.info(f"Scouting landscape for: {query}")
        start_time = time.time()
        scout_result = self.scout.run(query, observer=self.runtime_observer, query_store=self.queries_redis)
        end_time = time.time()
        logger.info(f"Scouting landscape for: {query} took {round(end_time - start_time, 2)} seconds")
        self._stage_completed(
            "scout",
            "Scout stage completed.",
            metrics={
                "elapsed_seconds": round(end_time - start_time, 3),
                "query_count": len(scout_result.get("queries", [])),
                "search_result_count": len(scout_result.get("search_results", [])),
            },
        )
        
        # 2. Plan the research (subclasses may skip the LLM planner, e.g. Crescent)
        research_plan = self._run_planning_stage(query, scout_result)

        # 3. Parallel Execution of Research Subagents
        for index, angle in enumerate(research_plan["research_angles"], start=1):
            angle["_angle_id"] = f"angle_{index}"
            self._register_angle(angle["_angle_id"], angle["topic"], angle["success_criteria"])
        self._stage_started(
            "research_parallel",
            "Parallel research stage started.",
            active_angle_count=len(research_plan["research_angles"]),
            completed_angle_count=0,
            metrics={"research_angle_count": len(research_plan["research_angles"])},
        )
        logger.info(f"Executing {len(research_plan['research_angles'])} research angles in parallel...")
        start_time = time.time()
        results = []
        subagent_pool: Queue[ResearchSubagent] = Queue()
        for subagent in self.research_subagents:
            subagent_pool.put(subagent)

        def _run_research_angle(
            topic_prompt: str,
            scout_brief: str,
            angle_id: str,
            angle_topic: str,
            success_criteria: str,
            project_topic: str,
        ) -> Dict[str, Any]:
            subagent = subagent_pool.get()
            try:
                return subagent.run(
                    topic_prompt,
                    scout_brief,
                    observer=self.runtime_observer,
                    angle_id=angle_id,
                    angle_topic=angle_topic,
                    success_criteria=success_criteria,
                    project_topic=project_topic,
                )
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
                    angle["_angle_id"],
                    angle["topic"],
                    angle["success_criteria"],
                    query,
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
                    if self.runtime_observer is not None:
                        self.runtime_observer.update_angle(
                            angle["_angle_id"],
                            status="completed",
                            stage="decide",
                            iteration=int(res.get("iteration", 0)),
                            queries_total=len(res.get("all_queries", []) or []),
                            context_ids_total=len(res.get("context_ids", []) or []),
                            statistics=res.get("statistics", {}) or {},
                            emit_event=True,
                            message=f"Completed research for angle: {angle['topic']}",
                        )
                except Exception as e:
                    logger.error(f"Subagent failed for angle '{angle['topic']}': {e}")
                    if self.runtime_observer is not None:
                        self.runtime_observer.update_angle(
                            angle["_angle_id"],
                            status="failed",
                            stage="research_parallel",
                            error=f"{type(e).__name__}: {e}",
                            emit_event=True,
                            message=f"Research angle failed: {angle['topic']}",
                        )
        end_time = time.time()
        logger.info(f"Executing research angles took {round(end_time - start_time, 2)} seconds")
        self._stage_completed(
            "research_parallel",
            "Parallel research stage completed.",
            active_angle_count=0,
            completed_angle_count=len(results),
            metrics={
                "elapsed_seconds": round(end_time - start_time, 3),
                "completed_angle_count": len(results),
            },
        )
        
        total_sources = 0
        for res in results:
            total_sources += len(res.get("context_ids", []))
        logger.info(f"Total number of sources to process: {total_sources}")
        logger.info("Starting batch NER/RE extraction into Knowledge Graph...")
        start_time = time.time()
        self._stage_started("kg_collect", "KG source collection started.")
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
        total_context_ids = len(all_context_ids)
        for index, cid in enumerate(tqdm(all_context_ids), start=1):
            entry = self.context_redis.get_entry(cid)
            if entry and entry.get("raw_content"):
                source_inputs.append((entry["raw_content"], entry.get("url", ""), cid))
            self._stage_progress(
                "kg_collect",
                "Preparing source inputs for KG collection.",
                processed=index,
                total=total_context_ids,
                elapsed_seconds=time.time() - start_time,
                extra_metrics={
                    "kg_collect_source_inputs_total": total_context_ids,
                    "kg_collect_source_inputs_ready": len(source_inputs),
                },
            )

        collected_sources = self.nerre_pipeline.collect_sources(
            source_inputs,
            progress_callback=lambda processed, total: self._stage_progress(
                "kg_collect",
                "Collecting NER chunks from sources.",
                processed=processed,
                total=total,
                elapsed_seconds=time.time() - start_time,
                extra_metrics={
                    "kg_collect_source_inputs_total": len(source_inputs),
                    "kg_collect_source_inputs_ready": len(source_inputs),
                },
            ),
        )
        self._stage_completed(
            "kg_collect",
            "KG source collection completed.",
            metrics={
                "elapsed_seconds": round(time.time() - start_time, 3),
                "source_input_count": len(source_inputs),
                "collected_source_count": len(collected_sources),
                "kg_collect_processed": len(collected_sources),
                "kg_collect_total": len(source_inputs),
            },
        )
        self._stage_started("kg_filter", "KG source filtering started.")
        retained_sources = self.nerre_pipeline.filter_sources(
            collected_sources,
            min_tokens=self.config.source_min_tokens,
            source_budget=self.config.research_source_budget,
            max_sources_threshold=self.config.max_sources_threshold,
        )
        retained_context_ids = [source.context_id for source in retained_sources if source.context_id]
        logger.info(
            "Retained %d/%d sources after minimum-length filtering, score cutoff, and budget pruning",
            len(retained_sources),
            len(collected_sources),
        )
        self._stage_progress(
            "kg_filter",
            "Filtering collected sources for KG processing.",
            processed=len(collected_sources),
            total=len(collected_sources),
            elapsed_seconds=time.time() - start_time,
            extra_metrics={
                "kg_filter_retained": len(retained_sources),
            },
        )
        self._stage_completed(
            "kg_filter",
            "KG source filtering completed.",
            metrics={
                "retained_source_count": len(retained_sources),
                "collected_source_count": len(collected_sources),
                "kg_filter_processed": len(collected_sources),
                "kg_filter_total": len(collected_sources),
                "kg_filter_retained": len(retained_sources),
            },
        )
        self._stage_started("kg_re", "KG relationship extraction started.")
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
            all_kg_entities.extend(
                self.nerre_pipeline.run_re_on_chunks(
                    optimized_chunks,
                    progress_callback=lambda processed, total: self._stage_progress(
                        "kg_re",
                        "Running RE extraction on optimized chunks.",
                        processed=processed,
                        total=total,
                        elapsed_seconds=time.time() - start_time,
                        extra_metrics={
                            "kg_re_chunk_total": len(optimized_chunks),
                        },
                    ),
                )
            )
        except Exception as e:
            logger.warning(f"Batch NER/RE failed: {e}")
        end_time = time.time()
        logger.info(f"NER/RE extraction took {round(end_time - start_time, 2)} seconds")
        self._stage_completed(
            "kg_re",
            "KG relationship extraction completed.",
            metrics={
                "elapsed_seconds": round(end_time - start_time, 3),
                "selected_chunk_count": len(optimized_chunks),
                "kg_entity_count": len(all_kg_entities),
                "kg_re_processed": len(optimized_chunks),
                "kg_re_total": len(optimized_chunks),
            },
        )

        logger.info("Starting batch entity resolution...")
        start_time = time.time()
        self._stage_started("entity_resolution", "Entity resolution started.")
        self.kg_store.perform_entity_resolution(all_kg_entities)
        self._stage_progress(
            "entity_resolution",
            "Resolving entities into canonical KG nodes.",
            processed=len(all_kg_entities),
            total=len(all_kg_entities),
            elapsed_seconds=time.time() - start_time,
        )
        end_time = time.time()
        logger.info(f"Entity resolution took {round(end_time - start_time, 2)} seconds")
        logger.info(f"Number of relationships active in the knowledge graph: {len(self.kg_store.get_relationships())}")
        logger.info(f"Number of chunks active in the knowledge graph: {len(self.kg_store.get_chunks())}")
        logger.info(f"Number of canonical entities active in the knowledge graph: {len(self.kg_store.get_canonical_entities())}")
        self._stage_completed(
            "entity_resolution",
            "Entity resolution completed.",
            metrics={
                "elapsed_seconds": round(end_time - start_time, 3),
                "relationship_count": len(self.kg_store.get_relationships()),
                "chunk_count": len(self.kg_store.get_chunks()),
                "canonical_entity_count": len(self.kg_store.get_canonical_entities()),
            },
        )

        # final report
        logger.info("Generating final report...")
        initial_context = self._build_outline_context(
            context_ids=retained_context_ids,
            scout_brief=scout_result["landscape_brief"],
            user_query=query,
        )

        # 4. Generate Outline
        self._stage_started("outline", "Outline generation started.")
        logger.info("Generating report outline...")
        start_time = time.time()

        max_iterations = 3
        report_title = ""
        constraints_paragraph = ""
        sections = []
        for attempt in range(1, max_iterations + 1):
            outline_result = self._generate_outline_with_system_prompt(
                self._outline_system_prompt(),
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
                print(outline_result)

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
        self._stage_completed(
            "outline",
            "Outline generation completed.",
            metrics={
                "elapsed_seconds": round(end_time - start_time, 3),
                "section_count": len(sections),
                "outline_section_titles_json": json.dumps(
                    [s.title for s in sections], ensure_ascii=False
                ),
            },
        )

        self._stage_started("write", "Report writing started.")
        # Seed writer metrics on the run snapshot immediately so the TUI sees
        # section count and rows before the first parallel section completes (and
        # even if an older writer_agent omitted runtime_observer).
        if self.runtime_observer is not None and sections:
            self.runtime_observer.update_run(
                metrics={
                    "write_sections_completed": 0,
                    "write_sections_total": len(sections),
                    "write_sections_progress_json": json.dumps(
                        [{"title": s.title, "status": "running"} for s in sections],
                        ensure_ascii=False,
                    ),
                }
            )
        logger.info("Writing iterative markdown report...")
        start_time = time.time()
        report_markdown = self._finalize_report(
            query,
            report_title,
            constraints_paragraph,
            sections,
            draft_output_path=draft_output_path,
        )
        end_time = time.time()
        logger.info(f"Writing report took {round(end_time - start_time, 2)} seconds")
        self._stage_completed(
            "write",
            "Report writing completed.",
            metrics={
                "elapsed_seconds": round(end_time - start_time, 3),
                "report_length": len(report_markdown or ""),
            },
        )

        return report_markdown


def _docker_redis_container_name(worker_id: int, role: str) -> str:
    return f"ming-worker-{worker_id}-{role}"


def _worker_redis_ports(worker_id: int, base_port: int = 6390) -> Dict[str, int]:
    """Return (context_port, queries_port, kg_port) for a worker."""
    offset = worker_id * 3
    return {
        "context": base_port + offset,
        "queries": base_port + offset + 1,
        "kg": base_port + offset + 2,
    }


def _start_worker_redis(worker_id: int, base_port: int = 6390) -> Dict[str, int]:
    """Spin up 3 isolated Redis containers for a single worker. Returns port dict."""
    import subprocess

    ports = _worker_redis_ports(worker_id, base_port)
    for role, port in ports.items():
        name = _docker_redis_container_name(worker_id, role)
        # Remove if leftover from a previous run
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
        )
        subprocess.run(
            ["docker", "run", "-d", "-p", f"{port}:6379", "--name", name, "redis:latest"],
            capture_output=True,
            check=True,
        )
    return ports


def _stop_worker_redis(worker_id: int) -> None:
    """Tear down all Redis containers for a worker."""
    import subprocess

    for role in ("context", "queries", "kg"):
        name = _docker_redis_container_name(worker_id, role)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _build_worker_config(
    raw_config: Dict[str, Any],
    ports: Dict[str, int],
) -> "MingDeepResearchConfig":
    """Clone the base config but point Redis at the worker's isolated ports."""
    from copy import deepcopy
    worker_cfg = deepcopy(raw_config)

    worker_cfg["redis"] = {
        "hostname": "localhost",
        "port": ports["context"],
        "db": 0,
    }
    worker_cfg["queries_redis"] = {
        "hostname": "localhost",
        "port": ports["queries"],
        "db": 0,
    }
    worker_cfg["kg_redis"] = {
        "hostname": "localhost",
        "port": ports["kg"],
        "db": 0,
    }
    return worker_cfg


def _flush_worker_redis(worker_cfg: Dict[str, Any]) -> None:
    """FLUSHDB on all three worker Redis instances."""
    import redis as _redis

    for key in ("redis", "queries_redis", "kg_redis"):
        cfg = worker_cfg.get(key)
        if not cfg:
            continue
        cl = _redis.Redis(
            host=str(cfg.get("hostname", "localhost")),
            port=int(cfg.get("port", 6379)),
            db=int(cfg.get("db", 0)),
            decode_responses=True,
        )
        try:
            cl.flushdb()
        finally:
            cl.close()


def _worker_loop(
    worker_id: int,
    work_queue: "queue.Queue[Dict[str, Any] | None]",
    submission_dir: "Path",
    raw_config: Dict[str, Any],
    base_port: int,
    counters: Dict[str, Any],
    counter_lock: "threading.Lock",
) -> None:
    """Long-running worker: own Redis triplet, pulls jobs from shared queue."""
    import time as _time
    from ming.core.config import create_ming_deep_research_config

    ports = _start_worker_redis(worker_id, base_port)
    worker_cfg = _build_worker_config(raw_config, ports)
    ming_config = create_ming_deep_research_config(worker_cfg)
    orchestrator = MingDeepResearch(ming_config)

    try:
        while True:
            row = work_queue.get()
            if row is None:
                break  # Poison pill

            qid = row["id"]
            prompt = str(row["prompt"])
            out_path = submission_dir / f"id_{qid}.md"

            if out_path.exists():
                with counter_lock:
                    counters["skipped"] += 1
                tqdm.write(f"[worker {worker_id}] Skip id={qid} (exists)")
                continue

            _flush_worker_redis(worker_cfg)
            start = _time.time()
            try:
                result = orchestrator.run(
                    prompt, draft_output_path=str(out_path.resolve())
                )
            except Exception as exc:
                with counter_lock:
                    counters["failed"] += 1
                tqdm.write(f"[worker {worker_id}] FAILED id={qid}: {exc}")
                continue

            elapsed_min = round(_time.time() - start, 2) / 60

            # Point draft_output_path at out_path so WriterAgent persists here (not only
            # config's reports/draft.md). Sync return value when non-empty.
            if result:
                out_path.write_text(result, encoding="utf-8")
            if out_path.exists():
                with counter_lock:
                    counters["written"] += 1
                tqdm.write(
                    f"[worker {worker_id}] Wrote id={qid} -> {out_path} ({elapsed_min:.2f} min)"
                )
            else:
                with counter_lock:
                    counters["failed"] += 1
                tqdm.write(
                    f"[worker {worker_id}] Empty report for id={qid} ({elapsed_min:.2f} min)"
                )
    finally:
        _stop_worker_redis(worker_id)


if __name__ == "__main__":
    import argparse
    import queue
    import threading
    import time
    from pathlib import Path

    from ming.core.config import create_ming_deep_research_config, load_config
    from ming.core.redis_flush import flush_research_redis_for_new_run

    parser = argparse.ArgumentParser(description="Ming deep research CLI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--jsonl",
        type=str,
        help="Path to query JSONL (fields: id, prompt, ...). Writes id_<id>.md per row.",
    )
    group.add_argument("--query", type=str, help="Run a single query (stdout timing only).")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config JSON path (used for Redis flush targets).",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="deepresearch-bench/submission/",
        help="Output directory for id_<id>.md with --jsonl (default: <jsonl>/../submission).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for --jsonl batch mode (default: 1, sequential).",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=6390,
        help="Starting port for parallel worker Redis containers (default: 6390).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    raw_config = load_config(args.config)

    if args.query:
        config = create_ming_deep_research_config(raw_config)
        orchestrator = MingDeepResearch(config)
        flush_research_redis_for_new_run(raw_config)
        start_time = time.time()
        orchestrator.run(args.query)
        print(f"\n\nTime taken: {round(time.time() - start_time, 2) / 60} minutes")
    else:
        jsonl_path = Path(args.jsonl).expanduser().resolve()
        submission_dir = (
            Path(args.submission_dir).expanduser().resolve()
            if args.submission_dir
            else jsonl_path.parent.parent / "submission"
        )
        submission_dir.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    tqdm.write(f"Skipping line {line_no}: invalid JSON ({e})")
                    continue
                if "id" not in row or "prompt" not in row:
                    tqdm.write(f"Skipping line {line_no}: missing id or prompt")
                    continue
                entries.append(row)

        num_workers = max(1, args.workers)

        if num_workers == 1:
            # Sequential mode: use the original Redis from config (no extra containers)
            config = create_ming_deep_research_config(raw_config)
            orchestrator = MingDeepResearch(config)

            n_skipped = n_written = n_failed = 0
            pbar = tqdm(entries, desc="JSONL queries", unit="query", dynamic_ncols=True)
            for row in pbar:
                qid = row["id"]
                prompt = str(row["prompt"])
                out_path = submission_dir / f"id_{qid}.md"
                pbar.set_postfix(
                    skipped=n_skipped, written=n_written, failed=n_failed, id=qid, refresh=False
                )
                if out_path.exists():
                    n_skipped += 1
                    pbar.set_postfix(
                        skipped=n_skipped, written=n_written, failed=n_failed, id=qid
                    )
                    tqdm.write(f"Skip id={qid} (exists: {out_path})")
                    continue

                flush_research_redis_for_new_run(raw_config)
                start_time = time.time()
                result = orchestrator.run(
                    prompt, draft_output_path=str(out_path.resolve())
                )
                elapsed_min = round(time.time() - start_time, 2) / 60

                if result:
                    out_path.write_text(result, encoding="utf-8")
                if out_path.exists():
                    n_written += 1
                    pbar.set_postfix(
                        skipped=n_skipped, written=n_written, failed=n_failed, id=qid
                    )
                    tqdm.write(f"Wrote id={qid} -> {out_path} ({elapsed_min:.2f} min)")
                else:
                    n_failed += 1
                    pbar.set_postfix(
                        skipped=n_skipped, written=n_written, failed=n_failed, id=qid
                    )
                    tqdm.write(
                        f"Empty report for id={qid}; not writing {out_path} ({elapsed_min:.2f} min)"
                    )
        else:
            # Parallel mode: each worker gets its own Redis container triplet
            print(f"Starting {num_workers} parallel workers (base port {args.base_port})...")

            work_q: queue.Queue[Dict[str, Any] | None] = queue.Queue()
            for entry in entries:
                work_q.put(entry)
            # Poison pills to signal workers to exit
            for _ in range(num_workers):
                work_q.put(None)

            counters: Dict[str, int] = {"skipped": 0, "written": 0, "failed": 0}
            counter_lock = threading.Lock()

            threads: List[threading.Thread] = []
            for wid in range(num_workers):
                t = threading.Thread(
                    target=_worker_loop,
                    args=(
                        wid,
                        work_q,
                        submission_dir,
                        raw_config,
                        args.base_port,
                        counters,
                        counter_lock,
                    ),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            print(
                f"\nDone. written={counters['written']}  "
                f"skipped={counters['skipped']}  failed={counters['failed']}"
            )