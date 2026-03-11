from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import logging
from math import ceil
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Union

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from ming.models import BaseModel, create_model_from_spec
from ming.tools import BaseTool, ToolConfig, create_tool_from_spec
from ming.core.prompts import (
    DECISION_PROMPT,
    GENERATE_QUERIES_PROMPT,
    THINK_PROMPT,
)
from ming.core.redis import QueryStore, RedisDatabase

logger = logging.getLogger(__name__)

class ResearchSubagentConfig(TypedDict, total=False):
    max_context_len: int
    min_query_count: int
    max_query_count: int
    model: dict[str, Any]
    tool_configs: List[dict[str, Any]]
    max_iterations: Optional[int]

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
    _MIN_CONTEXT_CHARS = 200

    def __init__(
        self,
        config: ResearchSubagentConfig,
        database: RedisDatabase,
        query_store: QueryStore | None = None,
    ):
        self.model = create_model_from_spec(config.get("model"))
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
                    if name == "think_tool":
                        tool.model = self.model
            except Exception as e:
                logger.warning("Skipping tool %s: %s", tool_config, e)

        self.max_iterations = config.get("max_iterations")
        self.graph = self._build_graph()
        self.statistics = self._empty_statistics()

    def _empty_statistics(self) -> Dict[str, int]:
        return {
            "total_searches": 0,
            "unique_urls": 0,
            "total_queries": 0,
            "discarded_search_results": 0,
            "discarded_open_results": 0,
            "successful_open_results": 0,
            "skipped_cached_urls": 0,
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

    def _generate_queries_node(self, state: ResearchSubagentState) -> Dict[str, Any]:
        topic = state["topic"]
        history = state.get("history") or []
        all_queries = state.get("all_queries") or []
        scout_report = (state.get("scout_report") or "").strip()
        min_q = self.config.get("min_query_count", 3)
        max_q = self.config.get("max_query_count", 5)

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

        prompt = GENERATE_QUERIES_PROMPT.format(
            topic=topic,
            scout_section=scout_section,
            previous_queries_section=previous_queries_section,
            history_section=history_section,
            min_queries=min_q,
            max_queries=max_q,
            guidance=guidance,
        )
        response = self.model.generate(prompt, **self._generation_kwargs())
        queries = self._parse_queries_from_response(response)

        return {
            "queries": queries,
            "all_queries": all_queries + queries,
            "history": history + ["Generated queries: " + ", ".join(queries)],
        }

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

    def _is_valid_context_content(self, content: Optional[str]) -> bool:
        if content is None:
            return False
        cleaned = content.strip()
        if len(cleaned) < self._MIN_CONTEXT_CHARS:
            return False

        lowered = cleaned.lower()
        failure_markers = (
            "failed to fetch url",
            "could not extract text content from page",
            "exception occurred while processing url",
        )
        return not any(marker in lowered for marker in failure_markers)

    def _get_or_fetch_url_context_id(
        self, url: str, web_result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get context ID for a URL, using cache when available.
        Handles race when multiple subagents fetch the same URL concurrently:
        - First to acquire lock fetches and inserts.
        - Others wait (poll) until the URL is indexed, then use cached result.
        """
        if not url:
            return None

        if web_result.get("discard") or web_result.get("discard_candidate"):
            self.statistics["discarded_search_results"] += 1
            return None

        # 1. Check cache first
        cached_id = self.database.get_context_id_by_url(url)
        if cached_id is not None:
            self.statistics["skipped_cached_urls"] += 1
            return cached_id

        # 2. Try to be the fetcher
        acquired = self.database.try_acquire_url_fetch_lock(url)
        if acquired:
            try:
                wb_raw = self._fetch_website_content(url)
                if (
                    wb_raw.get("discard", True)
                    or wb_raw.get("status") != "success"
                    or not self._is_valid_context_content(wb_raw.get("content"))
                ):
                    self.statistics["discarded_open_results"] += 1
                    return None
                entry = {
                    "url": url,
                    "title": web_result.get("title", "Untitled"),
                    "content": web_result.get("content", ""),
                    "raw_content": wb_raw.get("content", ""),
                }
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
        max_wait = 90
        waited = 0.0
        while waited < max_wait:
            time.sleep(poll_interval)
            waited += poll_interval
            cached_id = self.database.get_context_id_by_url(url)
            if cached_id is not None:
                return cached_id
        logger.warning("Timeout waiting for URL %s to be cached by another subagent", url)
        return None

    def _retrieve(self, state: ResearchSubagentState) -> Dict[str, Any]:
        queries = state.get("queries") or []
        if not queries:
            return {"history": state["history"] + ["No queries generated."]}

        web_results = []
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

        new_context_ids = []
        if web_results:
            with ThreadPoolExecutor(max_workers=max(1, len(web_results))) as executor:
                futures = [
                    executor.submit(
                        self._get_or_fetch_url_context_id,
                        r.get("url", ""),
                        r,
                    )
                    for r in web_results
                ]
                for future in as_completed(futures):
                    context_id = future.result()
                    if context_id is not None:
                        new_context_ids.append(context_id)

        existing_ids = state.get("context_ids") or []
        context_ids = existing_ids + new_context_ids
        self.statistics["unique_urls"] = len(set(context_ids))
        history_entry = "retrieved web content for queries: " + ", ".join(queries)
        return {
            "context_ids": context_ids,
            "history": state["history"] + [history_entry],
        }
    
    def _think(self, state: ResearchSubagentState) -> Dict[str, Any]:
        context_ids = state.get("context_ids") or []
        all_queries = state.get("all_queries") or []
        topic = state.get("topic", "")
        max_context_len = self.config.get("max_context_len", 4096)

        if not context_ids:
            return {"history": state["history"] + ["No context retrieved to synthesize."]}

        def _truncate_context(content: str, max_len: int) -> str:
            return content[:max_len] if content else ""

        context_parts = []

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

        with ThreadPoolExecutor(max_workers=max(1, len(context_ids))) as executor:
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

        think_tool = self._tool_map.get("think_tool")
        if think_tool is not None:
            response = think_tool.run(context_str)
        else:
            prompt = THINK_PROMPT.format(context=context_str)
            response = self.model.generate(
                prompt,
                **self._generation_kwargs(),
            )

        return {"history": state["history"] + [response]}

    def _decide_node(self, state: ResearchSubagentState) -> Dict[str, Any]:
        iteration = state.get("iteration", 0)
        max_iterations = self.config.get("max_iterations", 8)

        if iteration >= max_iterations:
            return {
                "decision": "stop",
                "iteration": iteration + 1,
                "history": state["history"] + ["Max iterations reached; stopping research."],
            }

        history = state.get("history") or []
        history_str = "\n\n".join(history) if history else "No prior research yet."
        prompt = DECISION_PROMPT.format(history=history_str)
        response = self.model.generate(
            prompt,
            **self._generation_kwargs(max_new_tokens=32),
        )

        decision = "continue" if "continue" in response.lower() else "stop"
        history_entry = (
            "Continuing research." if decision == "continue" else "Stopping research."
        )
        return {
            "decision": decision,
            "iteration": iteration + 1,
            "history": state["history"] + [history_entry],
        }

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

    def _count_words(self, text: str) -> int:
        return len(re.findall(r"\b\w+\b", text or ""))

    def get_url_word_counts(self, context_ids: List[str]) -> List[Dict[str, Union[str, int]]]:
        if not context_ids:
            return []

        entries: List[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, len(context_ids))) as executor:
            futures = [executor.submit(self.database.get_entry, cid) for cid in context_ids]
            for future in as_completed(futures):
                entry = future.result()
                if entry:
                    entries.append(entry)

        word_counts: List[Dict[str, Union[str, int]]] = []
        seen_urls = set()
        for entry in entries:
            url = (entry.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            content = (entry.get("raw_content") or entry.get("content") or "").strip()
            word_counts.append(
                {
                    "url": url,
                    "title": entry.get("title", "Untitled"),
                    "word_count": self._count_words(content),
                }
            )

        return sorted(word_counts, key=lambda item: int(item["word_count"]))

    def save_url_word_count_distribution(
        self,
        context_ids: List[str],
        topic: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, Any]]:
        word_counts = self.get_url_word_counts(context_ids)
        if not word_counts:
            return None

        base_dir = Path(output_dir or Path.cwd() / "artifacts")
        base_dir.mkdir(parents=True, exist_ok=True)

        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_") or "topic"
        csv_path = base_dir / f"{slug}_url_word_counts.csv"
        svg_path = base_dir / f"{slug}_url_word_count_distribution.svg"

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["url", "title", "word_count"])
            writer.writeheader()
            writer.writerows(word_counts)

        values = [int(item["word_count"]) for item in word_counts]
        min_value = min(values)
        max_value = max(values)
        bin_count = min(20, max(8, ceil(len(values) ** 0.75)))
        span = max(1, max_value - min_value)
        bin_width = max(1, ceil((span + 1) / bin_count))

        bins = []
        for index in range(bin_count):
            start = min_value + index * bin_width
            end = start + bin_width - 1
            bins.append({"start": start, "end": end, "count": 0})

        for value in values:
            bin_index = min((value - min_value) // bin_width, bin_count - 1)
            bins[int(bin_index)]["count"] += 1

        width = 1000
        height = 600
        margin_left = 70
        margin_right = 30
        margin_top = 50
        margin_bottom = 120
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        bar_gap = 12
        bar_width = max(20, (plot_width - bar_gap * (bin_count - 1)) / max(1, bin_count))
        max_bin_count = max(bin["count"] for bin in bins) or 1

        bars = []
        labels = []
        for index, bin_data in enumerate(bins):
            bar_height = 0 if max_bin_count == 0 else (bin_data["count"] / max_bin_count) * plot_height
            x = margin_left + index * (bar_width + bar_gap)
            y = margin_top + (plot_height - bar_height)
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2f6db2" />'
            )
            label = f'{bin_data["start"]}-{bin_data["end"]}'
            labels.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{height - 80}" font-size="12" text-anchor="end" transform="rotate(-35 {x + bar_width / 2:.1f},{height - 80})">{label}</text>'
            )
            labels.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{max(margin_top + 14, y - 8):.1f}" font-size="12" text-anchor="middle">{bin_data["count"]}</text>'
            )

        y_ticks = []
        for tick in range(5):
            tick_value = round(max_bin_count * tick / 4) if max_bin_count > 1 else tick
            tick_y = margin_top + plot_height - (tick / 4) * plot_height
            y_ticks.append(
                f'<line x1="{margin_left}" y1="{tick_y:.1f}" x2="{width - margin_right}" y2="{tick_y:.1f}" stroke="#d0d7de" stroke-width="1" />'
            )
            y_ticks.append(
                f'<text x="{margin_left - 10}" y="{tick_y + 4:.1f}" font-size="12" text-anchor="end">{tick_value}</text>'
            )

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{width / 2}" y="28" font-size="22" text-anchor="middle" font-family="Arial, sans-serif">Word Count Distribution per URL</text>
<text x="{width / 2}" y="48" font-size="13" text-anchor="middle" fill="#555" font-family="Arial, sans-serif">{topic}</text>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2" />
<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2" />
{''.join(y_ticks)}
{''.join(bars)}
{''.join(labels)}
<text x="{width / 2}" y="{height - 24}" font-size="14" text-anchor="middle" font-family="Arial, sans-serif">Word-count bins</text>
<text x="22" y="{margin_top + plot_height / 2}" font-size="14" text-anchor="middle" transform="rotate(-90 22,{margin_top + plot_height / 2})" font-family="Arial, sans-serif">URLs in bin</text>
</svg>
"""
        svg_path.write_text(svg, encoding="utf-8")

        return {
            "csv_path": str(csv_path),
            "plot_path": str(svg_path),
            "url_count": len(word_counts),
            "min_word_count": min_value,
            "max_word_count": max_value,
        }

    def export_contexts_as_text_files(
        self,
        context_ids: List[str],
        topic: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, Any]]:
        word_counts = self.get_url_word_counts(context_ids)
        if not word_counts:
            return None

        base_dir = Path(output_dir or Path.cwd() / "artifacts")
        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_") or "topic"
        text_dir = base_dir / f"{slug}_url_texts"
        text_dir.mkdir(parents=True, exist_ok=True)

        exported_files = []
        for index, item in enumerate(word_counts, start=1):
            entry_id = self.database.get_context_id_by_url(str(item["url"]))
            if entry_id is None:
                continue
            entry = self.database.get_entry(entry_id)
            if not entry:
                continue

            file_slug = re.sub(r"[^a-z0-9]+", "_", str(item["title"]).lower()).strip("_")
            if not file_slug:
                file_slug = f"url_{index:03d}"
            file_path = text_dir / f"{index:03d}_{file_slug[:80]}.txt"

            body = (entry.get("raw_content") or entry.get("content") or "").strip()
            text = (
                f"URL: {item['url']}\n"
                f"Title: {item['title']}\n"
                f"Word Count: {item['word_count']}\n\n"
                f"{body}\n"
            )
            file_path.write_text(text, encoding="utf-8")
            exported_files.append(str(file_path))

        return {
            "text_dir": str(text_dir),
            "file_count": len(exported_files),
            "files": exported_files,
        }

    def run(self, topic: str, scout_report: str = "") -> Dict[str, Any]:
        self.statistics = self._empty_statistics()
        initial_state: ResearchSubagentState = {
            "topic": topic,
            "context_ids": [],
            "history": [],
            "iteration": 0,
            "scout_report": scout_report,
            "all_queries": [],
        }

        return self.graph.invoke(initial_state)

    def get_statistics(self) -> Dict[str, Any]:
        return self.statistics

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
