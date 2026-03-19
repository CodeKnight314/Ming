from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import logging
from math import ceil
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Union

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict
from dataclasses import asdict, dataclass

from ming.models import BaseModel, create_model_from_spec, OpenRouterModelConfig
from ming.tools import BaseTool, ToolConfig, create_tool_from_spec
from ming.core.prompts import (
    DECISION_PROMPT,
    GENERATE_QUERIES_PROMPT,
    THINK_PROMPT,
)
from ming.core.redis import QueryStore, RedisDatabase
from ming.core.text_metrics import count_language_aware_tokens

logger = logging.getLogger(__name__)


def _parse_criteria_from_topic(topic: str) -> str:
    """Extract the success criteria block from a formatted topic string."""
    match = re.search(r"Success Criteria:\s*(.*?)(?=\nConstraints:|\Z)", topic, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_gaps_from_history(history: List[str]) -> List[str]:
    """Extract unresolved GAP lines from the most recent criteria assessment in history.

    Scans history in reverse to find the last THINK synthesis that contains a
    ``## Criteria Assessment`` block, then returns all GAP values that are not
    ``None`` or ``N/A``.  These are injected into GENERATE_QUERIES_PROMPT so the
    next query round targets identified evidence gaps rather than re-exploring
    already-covered ground.
    """
    for entry in reversed(history):
        if "## Criteria Assessment" not in entry:
            continue
        gaps = []
        for match in re.finditer(r"GAP:\s*(.+?)(?:\n|$)", entry):
            gap_text = match.group(1).strip()
            if gap_text.lower() not in ("none", "n/a", ""):
                gaps.append(gap_text)
        return gaps
    return []


def _count_criteria(criteria_text: str) -> int:
    """Count the number of distinct success criteria in a criteria block."""
    if not criteria_text:
        return 1
    items = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.):])\s+", criteria_text)
    return max(1, len(items)) if items else 1


def _compute_think_tokens(num_sources: int, num_criteria: int) -> int:
    """Compute a dynamic token budget for the think synthesis.

    Formula: 300 base + 40 per source + 120 per criterion, clamped to [768, 2048].
    """
    return 8192


class ResearchSubagentConfig(TypedDict, total=False):
    max_context_len: int
    min_query_count: int
    max_query_count: int
    max_total_queries: int
    max_open_urls_per_iteration: int
    max_url_wait_seconds: float
    model: dict[str, Any]
    fallback_model: dict[str, Any]
    tool_configs: List[dict[str, Any]]
    max_iterations: Optional[int]
    source_min_tokens: int

class ResearchSubagentState(TypedDict, total=False):
    topic: str
    context_ids: List[str]
    history: List[str]
    iteration: int
    scout_report: str
    queries: List[str]
    all_queries: List[str]
    decision: str


class ResearchSubagent:
    def __init__(
        self,
        config: ResearchSubagentConfig,
        database: RedisDatabase,
        query_store: QueryStore | None = None,
    ):
        self.model = create_model_from_spec(config.get("model"))
        fallback_spec = config.get("fallback_model")
        self.fallback_model: BaseModel | None = (
            create_model_from_spec(fallback_spec) if fallback_spec else None
        )
        self.config = config
        self.database = database
        self.query_store = query_store

        self._tool_map = {}
        for tool_config in (config.get("tool_configs") or []):
            try:
                tool = create_tool_from_spec(tool_config)
                if tool.preflight_check():
                    name = tool.get_name()
                    self._tool_map[name] = tool
            except Exception as e:
                logger.warning("Skipping tool %s: %s", tool_config, e)

        self.max_iterations = config.get("max_iterations", 3)
        self.source_min_tokens = max(1, int(config.get("source_min_tokens", 400)))
        self.graph = self._build_graph()
        self.statistics = self._empty_statistics()
        self._runtime_observer: Any | None = None
        self._angle_id: str | None = None
        self._angle_topic: str = ""
        self._angle_success_criteria: str = ""

    def _generate_with_fallback(self, prompt: str, **generation_kwargs: Any) -> str:
        """Call the primary model, falling back to a secondary model on failure.

        This protects individual research angles from being dropped entirely when
        the primary engine (e.g. qwen3.5 flash) has transient OpenRouter/runtime issues.
        """
        try:
            return self.model.generate(prompt, **self._generation_kwargs(**generation_kwargs))
        except Exception as exc:
            if not self.fallback_model:
                raise
            logger.warning(
                "Primary subagent model failed; falling back to secondary model: %s",
                exc,
            )
            return self.fallback_model.generate(
                prompt,
                **self._generation_kwargs(**generation_kwargs),
            )

    def _empty_statistics(self) -> Dict[str, int]:
        return {
            "total_searches": 0,
            "unique_urls": 0,
            "total_queries": 0,
            "discarded_search_results": 0,
            "discarded_open_results": 0,
            "ranked_out_urls": 0,
            "successful_open_results": 0,
            "skipped_cached_urls": 0,
            "tavily_depth_fallbacks": 0,
        }

    def _generation_kwargs(self, **overrides: Any):
        base = {
            "max_new_tokens": self.config.get("max_new_tokens", 1024),
            "temperature": self.config.get("temperature", 0.7),
            "do_sample": self.config.get("do_sample", True),
            "use_cache": self.config.get("use_cache", True),
        }
        return {**base, **overrides}

    def _parse_queries_from_response(self, response: str) -> List[str]:
        tagged = re.findall(
            r"<query>(.*?)</query>", response, re.IGNORECASE | re.DOTALL
        )
        candidates = tagged if tagged else response.split("\n")
        queries = []
        seen = set()
        for candidate in candidates:
            query = re.sub(r"\s+", " ", candidate).strip()
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(query)
        return queries

    def _current_iteration(self, state: ResearchSubagentState) -> int:
        return int(state.get("iteration", 0)) + 1

    def _max_total_queries(self) -> int:
        raw_limit = self.config.get("max_new_queries", 10)
        return max(1, int(raw_limit))

    def _ensure_angle_registered(self) -> None:
        if self._runtime_observer is None or not self._angle_id:
            return
        try:
            self._runtime_observer.register_angle(
                angle_id=self._angle_id,
                topic=self._angle_topic or self._angle_id,
                success_criteria=self._angle_success_criteria,
            )
        except Exception:
            return

    def _update_observer(
        self,
        state: ResearchSubagentState,
        *,
        stage: str,
        status: str,
        message: str,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if self._runtime_observer is None or not self._angle_id:
            return
        self._runtime_observer.update_angle(
            self._angle_id,
            status=status,
            stage=stage,
            iteration=self._current_iteration(state),
            queries_total=len(state.get("all_queries") or []),
            context_ids_total=len(state.get("context_ids") or []),
            statistics=dict(self.statistics),
            error=error,
            emit_event=True,
            message=message,
            metrics=metrics,
        )

    def _generate_queries_node(self, state: ResearchSubagentState) -> Dict[str, Any]:
        self._update_observer(
            state,
            stage="generate_queries",
            status="running",
            message="Generating research queries.",
        )
        topic = state["topic"]
        history = state.get("history") or []
        all_queries = state.get("all_queries") or []
        scout_report = (state.get("scout_report") or "").strip()
        remaining_budget = max(0, self._max_total_queries() - len(all_queries))
        max_q = min(int(self.config.get("max_query_count", 5)), remaining_budget)
        min_q = min(int(self.config.get("min_query_count", 3)), max_q)

        if remaining_budget <= 0 or max_q <= 0:
            next_state = {
                "queries": [],
                "all_queries": all_queries,
                "history": history + ["Query budget exhausted; no additional queries generated."],
            }
            self._update_observer(
                {
                    **state,
                    **next_state,
                },
                stage="generate_queries",
                status="running",
                message="Query budget exhausted; no additional queries generated.",
                metrics={"generated_query_count": 0, "remaining_query_budget": 0},
            )
            return next_state

        if scout_report:
            scout_section = "Scout brief:\n" + scout_report + "\n\n"
        else:
            scout_section = ""

        previous_queries_list = list(all_queries)
        if self.query_store:
            stored = self.query_store.get_queries(topic)
            seen = set(q.lower() for q in previous_queries_list)
            for q in stored:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    previous_queries_list.append(q)

        if previous_queries_list:
            previous_queries_section = (
                "Queries already run (avoid repeating; fill gaps or explore new angles):\n"
                + "\n".join(f"- {q}" for q in previous_queries_list)
                + "\n\n"
            )
        else:
            previous_queries_section = ""

        if history:
            history_section = "Prior research and synthesis:\n" + "\n\n".join(history)
            guidance = "Focus on filling gaps, verifying claims, or exploring new angles based on prior findings."
        else:
            history_section = ""
            guidance = "Cast a broad net to find relevant sources."

        gaps = _extract_gaps_from_history(history) if history else []
        if gaps:
            gaps_section = (
                "Unresolved gaps from prior synthesis that MUST be addressed in this round:\n"
                + "\n".join(f"- {g}" for g in gaps)
                + "\n"
            )
        else:
            gaps_section = ""

        prompt = GENERATE_QUERIES_PROMPT.format(
            topic=topic,
            scout_section=scout_section,
            previous_queries_section=previous_queries_section,
            history_section=history_section,
            gaps_section=gaps_section,
            min_queries=min_q,
            max_queries=max_q,
            guidance=guidance,
        )
        response = self._generate_with_fallback(prompt)
        queries = self._parse_queries_from_response(response)[:remaining_budget]
        next_state = {
            "queries": queries,
            "all_queries": all_queries + queries,
            "history": history + ["Generated queries: " + ", ".join(queries)],
        }
        self._update_observer(
            {
                **state,
                **next_state,
            },
            stage="generate_queries",
            status="running",
            message="Generated research queries.",
            metrics={
                "generated_query_count": len(queries),
                "remaining_query_budget": max(
                    0, self._max_total_queries() - len(next_state["all_queries"])
                ),
            },
        )

        return next_state

    def _search_web(self, query: str) -> Dict[str, Any]:
        tool = self._tool_map.get("web_search_tool")
        if tool is None:
            raise ValueError("Web search tool not found.")
        return tool.run(query)

    def _fetch_website_content(self, url: str) -> Dict[str, Any]:
        tool = self._tool_map.get("open_url_tool")
        if tool is None:
            raise ValueError("Open URL tool not found.")
        return tool.run(url)

    def _is_cacheable_context_content(self, content: Optional[str]) -> bool:
        if content is None:
            return False
        cleaned = content.strip()
        if not cleaned:
            return False

        lowered = cleaned.lower()
        failure_markers = (
            "failed to fetch url",
            "could not extract text content from page",
            "exception occurred while processing url",
        )
        return not any(marker in lowered for marker in failure_markers)

    def _meets_source_length_threshold(self, content: Optional[str]) -> bool:
        if content is None:
            return False
        cleaned = content.strip()
        if not cleaned:
            return False
        return count_language_aware_tokens(cleaned) >= self.source_min_tokens

    @staticmethod
    def _score_web_result(result: Dict[str, Any], index: int) -> tuple[int, float, int, int]:
        raw_score = result.get("score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = float("-inf")
        snippet_len = len((result.get("content") or "").strip())
        return (
            int(raw_score is not None),
            score,
            snippet_len,
            -index,
        )

    def _select_open_candidates(
        self, web_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not web_results:
            return []

        best_by_url: Dict[str, tuple[tuple[int, float, int, int], Dict[str, Any]]] = {}
        for index, result in enumerate(web_results):
            url = (result.get("url") or "").strip()
            if not url:
                continue
            if (
                result.get("discard")
                or result.get("discard_candidate")
                or url.endswith(".pdf")
                or url.endswith(".docx")
            ):
                self.statistics["discarded_search_results"] += 1
                continue

            score_key = self._score_web_result(result, index)
            current = best_by_url.get(url)
            if current is None or score_key > current[0]:
                best_by_url[url] = (score_key, result)

        ranked_results = [
            item[1]
            for item in sorted(
                best_by_url.values(),
                key=lambda item: item[0],
                reverse=True,
            )
        ]

        max_open_urls = max(1, int(self.config.get("max_open_urls_per_iteration", 12)))
        if len(ranked_results) > max_open_urls:
            self.statistics["ranked_out_urls"] += len(ranked_results) - max_open_urls
        return ranked_results[:max_open_urls]

    def _get_or_fetch_url_context_id(
        self, url: str, web_result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get context ID for a URL, using cache when available.
        Handles race when multiple subagents fetch the same URL concurrently:
        - First to acquire lock fetches and inserts.
        - Others wait (poll) until the URL is indexed, then use cached result.
        """
        if not url or url.endswith(".pdf") or url.endswith(".docx"):
            return None

        if web_result.get("discard") or web_result.get("discard_candidate"):
            self.statistics["discarded_search_results"] += 1
            return None

        # 1. Check cache first
        cached_id = self.database.get_context_id_by_url(url)
        if cached_id is not None:
            self.statistics["skipped_cached_urls"] += 1
            return cached_id

        # 2. Try to be the writer
        acquired = self.database.try_acquire_url_fetch_lock(url)
        if acquired:
            try:
                raw_text = (web_result.get("content") or "").strip()
                tavily_tokens = count_language_aware_tokens(raw_text)

                fallback_min_tokens = int(
                    self.config.get("tavily_depth_fallback_min_tokens", 600)
                )
                fallback_score_floor = float(
                    self.config.get("tavily_depth_fallback_score_threshold", 0.7)
                )
                tavily_score = float(web_result.get("score") or 0.0)

                if (
                    tavily_tokens < fallback_min_tokens
                    and tavily_score >= fallback_score_floor
                    and "open_url_tool" in self._tool_map
                ):
                    self.statistics["tavily_depth_fallbacks"] += 1
                    try:
                        fetched = self._fetch_website_content(url)
                        fetched_content = fetched.get("content")
                        if self._is_cacheable_context_content(fetched_content):
                            fetched_tokens = count_language_aware_tokens(
                                fetched.get("content", "")
                            )
                            if fetched_tokens > tavily_tokens:
                                raw_text = fetched["content"].strip()
                    except Exception as fetch_exc:
                        logger.debug(
                            "Trafilatura depth fallback failed for %s: %s", url, fetch_exc
                        )

                entry = {
                    "url": url,
                    "title": web_result.get("title", "Untitled"),
                    "content": raw_text,
                    "raw_content": raw_text,
                    "token_count": count_language_aware_tokens(raw_text),
                    "below_source_min_tokens": not self._meets_source_length_threshold(raw_text),
                    "content_missing": not bool(raw_text),
                    "preserved_from_search": True,
                    "retrieval_status": web_result.get("status", "success"),
                    "retrieval_notes": web_result.get("retrieval_notes", ""),
                }
                if not self._is_cacheable_context_content(raw_text):
                    self.statistics["discarded_open_results"] += 1
                context_id = self.database.create_entry(entry)
                self.database.set_url_index(url, context_id)
                self.statistics["successful_open_results"] += 1
                return context_id
            except Exception as e:
                logger.warning("Failed to fetch and cache URL %s: %s", url, e)
                self.statistics["discarded_open_results"] += 1
                return None
            finally:
                self.database.release_url_fetch_lock(url)

        poll_interval = 0.2
        max_wait = max(0.0, float(self.config.get("max_url_wait_seconds", 8.0)))
        waited = 0.0
        while waited < max_wait:
            time.sleep(poll_interval)
            waited += poll_interval
            cached_id = self.database.get_context_id_by_url(url)
            if cached_id is not None:
                return cached_id
            if not self.database.is_url_fetch_locked(url):
                logger.info(
                    "Skipping URL %s after peer fetch completed without caching it",
                    url,
                )
                return None
        logger.warning(
            "Timeout waiting %.1fs for URL %s to be cached by another subagent",
            max_wait,
            url,
        )
        return None

    def _retrieve(self, state: ResearchSubagentState) -> Dict[str, Any]:
        self._update_observer(
            state,
            stage="retrieve",
            status="running",
            message="Retrieving sources for research angle.",
        )
        queries = state.get("queries") or []
        if not queries:
            return {"history": state["history"] + ["No queries generated."]}

        web_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, len(queries))) as executor:
            futures = [executor.submit(self._search_web, q) for q in queries]
            for future in as_completed(futures):
                web_results.extend(future.result())

        self.statistics["total_searches"] += len(queries)
        self.statistics["total_queries"] += len(queries)

        if self.query_store:
            topic = state.get("topic", "")
            if topic:
                self.query_store.add_queries(topic, queries)

        candidate_results = self._select_open_candidates(web_results)

        new_context_ids = []
        if candidate_results:
            with ThreadPoolExecutor(max_workers=min(8, len(candidate_results))) as executor:
                futures = [
                    executor.submit(
                        self._get_or_fetch_url_context_id,
                        r.get("url", ""),
                        r,
                    )
                    for r in candidate_results
                ]
                for future in as_completed(futures):
                    context_id = future.result()
                    if context_id is not None:
                        new_context_ids.append(context_id)

        existing_ids = state.get("context_ids") or []
        context_ids = existing_ids + new_context_ids
        self.statistics["unique_urls"] = len(set(context_ids))
        history_entry = (
            "cached web content from Tavily for queries: "
            + ", ".join(queries)
            + f" (cached {len(new_context_ids)} of {len(web_results)} search results)"
        )
        next_state = {
            "context_ids": context_ids,
            "history": state["history"] + [history_entry],
        }
        self._update_observer(
            {
                **state,
                **next_state,
            },
            stage="retrieve",
            status="running",
            message="Retrieved sources for research angle.",
            metrics={
                "query_count": len(queries),
                "search_result_count": len(web_results),
                "candidate_result_count": len(candidate_results),
                "new_context_id_count": len(new_context_ids),
            },
        )
        return next_state
    
    def _think(self, state: ResearchSubagentState) -> Dict[str, Any]:
        self._update_observer(
            state,
            stage="think",
            status="running",
            message="Synthesizing retrieved context.",
        )
        context_ids = state.get("context_ids") or []
        all_queries = state.get("all_queries") or []
        topic = state.get("topic", "")
        max_context_len = self.config.get("max_context_len", 4096)

        if not context_ids:
            return {"history": state["history"] + ["No context retrieved to synthesize."]}

        criteria_text = _parse_criteria_from_topic(topic)
        num_criteria = _count_criteria(criteria_text)
        think_tokens = _compute_think_tokens(len(context_ids), num_criteria)

        def _truncate_context(content: str, max_len: int) -> str:
            return content[:max_len] if content else ""

        context_parts = []

        if criteria_text:
            context_parts.append(f"Research success criteria:\n{criteria_text}")

        queries_for_context = list(all_queries)
        if self.query_store and topic:
            stored = self.query_store.get_queries(topic)
            seen = set(q.lower() for q in queries_for_context)
            for q in stored:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    queries_for_context.append(q)

        if queries_for_context:
            context_parts.append(
                "Queries run so far (including prior sessions):\n"
                + "\n".join(f"- {q}" for q in queries_for_context)
            )

        with ThreadPoolExecutor(max_workers=min(16, len(context_ids))) as executor:
            futures = [executor.submit(self.database.get_entry, cid) for cid in context_ids]
            for future in as_completed(futures):
                entry = future.result()
                if not entry:
                    continue
                content = _truncate_context(
                    entry.get("content", "") or "", max_context_len
                ) + "---[Truncated for length]---"
                title = entry.get("title", "Untitled")
                context_parts.append(f"Title: {title}\nContent: {content}")

        context_str = "\n\n---\n\n".join(context_parts)

        prompt = THINK_PROMPT.format(context=context_str)
        response = self._generate_with_fallback(
            prompt,
            max_new_tokens=think_tokens,
        )

        next_state = {"history": state["history"] + [response]}
        self._update_observer(
            {
                **state,
                **next_state,
            },
            stage="think",
            status="running",
            message="Completed synthesis for current iteration.",
            metrics={"think_token_budget": think_tokens},
        )
        return next_state

    def _decide_node(self, state: ResearchSubagentState) -> Dict[str, Any]:
        self._update_observer(
            state,
            stage="decide",
            status="running",
            message="Evaluating whether to continue research.",
        )
        iteration = state.get("iteration", 0)
        max_iterations = self.config.get("max_iterations", 8)
        max_queries = self._max_total_queries()
        
        if iteration >= max_iterations or len(state.get("all_queries", [])) >= max_queries:
            stop_reason = (
                "Max iterations reached; stopping research."
                if iteration >= max_iterations
                else "Max query budget reached; stopping research."
            )
            next_state = {
                "decision": "stop",
                "iteration": iteration + 1,
                "history": state["history"] + [stop_reason],
            }
            self._update_observer(
                {
                    **state,
                    **next_state,
                },
                stage="decide",
                status="completed",
                message=stop_reason,
                metrics={"decision": "stop", "max_total_queries": max_queries},
            )
            return next_state

        history = state.get("history") or []
        history_str = "\n\n".join(history) if history else "No prior research yet."
        prompt = DECISION_PROMPT.format(history=history_str)
        response = self._generate_with_fallback(
            prompt,
            max_new_tokens=32,
        )

        decision = "continue" if "continue" in response.lower() else "stop"
        history_entry = (
            "Continuing research." if decision == "continue" else "Stopping research."
        )
        next_state = {
            "decision": decision,
            "iteration": iteration + 1,
            "history": state["history"] + [history_entry],
        }
        self._update_observer(
            {
                **state,
                **next_state,
            },
            stage="decide",
            status="completed",
            message=history_entry,
            metrics={"decision": decision},
        )
        return next_state

    def _route_after_decide(self, state: ResearchSubagentState) -> str:
        return state.get("decision", "stop")

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(ResearchSubagentState)
        workflow.add_node("generate_queries", self._generate_queries_node)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("think", self._think)
        workflow.add_node("decide", self._decide_node)

        workflow.add_edge(START, "generate_queries")
        workflow.add_edge("generate_queries", "retrieve")
        workflow.add_edge("retrieve", "think")
        workflow.add_edge("think", "decide")
        workflow.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {"continue": "generate_queries", "stop": END},
        )

        return workflow.compile()

    def _export_history(self, state: ResearchSubagentState) -> Dict[str, Any]:
        return {
            "history": state["history"],
        }

    def run(
        self,
        topic: str,
        scout_report: str = "",
        observer: Any | None = None,
        angle_id: str | None = None,
        angle_topic: str | None = None,
        success_criteria: str = "",
    ) -> Dict[str, Any]:
        self.statistics = self._empty_statistics()
        self._runtime_observer = observer
        self._angle_id = angle_id
        self._angle_topic = angle_topic or topic
        self._angle_success_criteria = success_criteria
        self._ensure_angle_registered()
        initial_state: ResearchSubagentState = {
            "topic": topic,
            "context_ids": [],
            "history": [],
            "iteration": 0,
            "scout_report": scout_report,
            "all_queries": [],
        }

        recursion_limit = int(self.config.get("recursion_limit", 40))
        try:
            results = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
            results["statistics"] = dict(self.statistics)
            self._update_observer(
                results,
                stage="decide",
                status="completed",
                message="Research angle completed.",
                metrics={"final_decision": results.get("decision")},
            )
            return results
        except Exception as exc:
            self._update_observer(
                initial_state,
                stage="research_parallel",
                status="failed",
                message=f"Research angle failed: {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            for tool in self._tool_map.values():
                if hasattr(tool, "close"):
                    tool.close()
            self._runtime_observer = None
            self._angle_id = None
            self._angle_topic = ""
            self._angle_success_criteria = ""

    def get_statistics(self) -> Dict[str, Any]:
        return self.statistics


@dataclass
class AgentConfig:
    model: dict[str, Any]
    system_prompt: str
    fallback_model: dict[str, Any] | OpenRouterModelConfig | None = None
    tools: List[ToolConfig] | None = None
    max_iterations: int = 15
    max_tool_calls_per_turn: int = 8


class AgentMessage(TypedDict):
    role: str
    content: str


class AgentState(TypedDict, total=False):
    messages: List[AgentMessage]
    output: str
    iteration: int


@dataclass
class AgentResult:
    output: str
    messages: List[AgentMessage]
    iteration: int


class Agent:
    # Match only well-formed tool calls whose payload is a JSON object.
    # This avoids treating nested stray <tool_call> tags as the payload.
    _TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def __init__(self, config: AgentConfig):
        model_spec = config.model
        if isinstance(model_spec, OpenRouterModelConfig):
            model_spec = asdict(model_spec)
            model_spec["provider"] = "openrouter"
        self.model = create_model_from_spec(model_spec)
        fallback_spec = config.fallback_model
        if isinstance(fallback_spec, OpenRouterModelConfig):
            fallback_spec = asdict(fallback_spec)
            fallback_spec["provider"] = "openrouter"
        self.fallback_model: BaseModel | None = (
            create_model_from_spec(fallback_spec) if fallback_spec else None
        )
        self._tool_map: Dict[str, BaseTool] = {}
        for tool_config in (config.tools or []):
            try:
                if isinstance(tool_config, BaseTool):
                    tool = tool_config
                else:
                    tool = create_tool_from_spec(tool_config)
                if tool.preflight_check():
                    self._tool_map[tool.get_name()] = tool
            except Exception as e:
                logger.warning("Skipping tool %s: %s", tool_config, e)
        self.system_prompt = config.system_prompt
        self.max_iterations = config.max_iterations
        self.max_tool_calls_per_turn = max(1, config.max_tool_calls_per_turn)
        self.graph = self._build_graph()

    def _generate_with_fallback(self, prompt: str, **generation_kwargs: Any) -> str:
        try:
            return self.model.generate(prompt, **generation_kwargs)
        except Exception as exc:
            if not self.fallback_model:
                raise
            logger.warning(
                "Primary agent model failed; falling back to secondary model: %s",
                exc,
            )
            return self.fallback_model.generate(prompt, **generation_kwargs)

    def _format_tools_for_prompt(self) -> str:
        if not self._tool_map:
            return ""
        sections = [tool.format_for_prompt() for tool in self._tool_map.values()]
        return (
            "# Available Tools\n\n"
            + "\n\n".join(sections)
            + "\n\n"
            "To use a tool, include in your response:\n"
            '<tool_call>{"name": "tool_name", "parameters": {"param": "value"}}</tool_call>\n\n'
            "You may invoke multiple tools in one response.\n"
            "When you have enough information, respond without any <tool_call> tags."
        )

    def _build_system_message(self) -> str:
        parts = [self.system_prompt]
        tools_section = self._format_tools_for_prompt()
        if tools_section:
            parts.append(tools_section)
        return "\n\n".join(parts)

    def _messages_to_prompt(self, messages: List[AgentMessage]) -> str:
        parts: List[str] = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"\nUser:\n{content}")
            elif role == "assistant":
                parts.append(f"\nAssistant:\n{content}")
            elif role == "tool_result":
                parts.append(f"\nTool Results:\n{content}")
        parts.append("\nAssistant:\n")
        return "\n".join(parts)

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for match in self._TOOL_CALL_RE.finditer(text):
            try:
                payload = json.loads(match.group(1).strip())
                if isinstance(payload, dict) and "name" in payload:
                    calls.append(payload)
            except json.JSONDecodeError:
                continue
        return calls

    def _strip_tool_calls(self, text: str) -> str:
        return self._TOOL_CALL_RE.sub("", text).strip()

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        prompt = self._messages_to_prompt(state["messages"])
        response = self._generate_with_fallback(prompt)
        messages = state["messages"] + [{"role": "assistant", "content": response}]
        return {"messages": messages, "output": response}

    def _execute_tools(self, state: AgentState) -> Dict[str, Any]:
        last_response = state.get("output", "")
        tool_calls = self._parse_tool_calls(last_response)

        results: List[str] = []
        executed = 0
        for call in tool_calls:
            name = call.get("name", "")
            params = call.get("parameters", {})
            tool = self._tool_map.get(name)
            if tool is None:
                continue
            valid, _ = tool.validate_parameters(params)
            if not valid:
                continue
            try:
                result = tool.run(**params)
                results.append(f"[{name}]\n{result}")
                executed += 1
                if executed >= self.max_tool_calls_per_turn:
                    break
            except Exception as exc:
                results.append(f"[{name}] Error: {exc}")

        tool_content = "\n\n".join(results) if results else "No valid tool calls executed."
        messages = state["messages"] + [{"role": "tool_result", "content": tool_content}]
        return {"messages": messages, "iteration": state.get("iteration", 0) + 1}

    def _route_after_model(self, state: AgentState) -> str:
        if state.get("iteration", 0) >= self.max_iterations:
            return "end"
        if self._parse_tool_calls(state.get("output", "")):
            return "continue"
        return "end"

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("call_model", self._call_model)
        workflow.add_node("execute_tools", self._execute_tools)

        workflow.add_edge(START, "call_model")
        workflow.add_conditional_edges(
            "call_model",
            self._route_after_model,
            {"continue": "execute_tools", "end": END},
        )
        workflow.add_edge("execute_tools", "call_model")

        return workflow.compile()

    def run(self, user_input: str) -> AgentResult:
        system_msg = self._build_system_message()
        initial: AgentState = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_input},
            ],
            "output": "",
            "iteration": 0,
        }

        if not self._tool_map:
            prompt = self._messages_to_prompt(initial["messages"])
            output = self._generate_with_fallback(prompt)
            return AgentResult(
                output=output,
                messages=initial["messages"] + [{"role": "assistant", "content": output}],
                iteration=0
            )

        result = self.graph.invoke(initial)
        return AgentResult(
            output=self._strip_tool_calls(result.get("output", "")),
            messages=result.get("messages", initial["messages"]),
            iteration=result.get("iteration", 0),
        )

if __name__ == "__main__":
    from ming.core.config import create_subagent_from_config
    subagent = create_subagent_from_config()
    topic = "Sino-Soviet relations in the 1960s"
    result = subagent.run(topic)
    print(result)
    print(subagent.get_statistics())
    report = subagent.save_url_word_count_distribution(
        context_ids=result.get("context_ids") or [],
        topic=topic,
    )
    if report is None:
        print("No URL word-count distribution was generated.")
    else:
        print(report)
    text_export = subagent.export_contexts_as_text_files(
        context_ids=result.get("context_ids") or [],
        topic=topic,
    )
    if text_export is None:
        print("No URL text files were exported.")
    else:
        print(text_export)
