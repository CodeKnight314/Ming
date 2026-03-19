from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from ming.core.prompts import SCOUT_QUERY_PROMPT, SCOUT_SUMMARY_PROMPT
from ming.core.redis import QueryStore
from ming.models import create_model_from_spec
from ming.tools import create_tool_from_spec

logger = logging.getLogger(__name__)


class ScoutSubagentConfig(TypedDict, total=False):
    min_query_count: int
    max_query_count: int
    max_results_per_query: int
    max_landscape_results: int
    model: dict[str, Any]
    tool_configs: List[dict[str, Any]]
    max_new_tokens: int
    temperature: float
    do_sample: bool
    use_cache: bool


class ScoutResult(TypedDict, total=False):
    topic: str
    queries: List[str]
    search_results: List[Dict[str, Any]]
    landscape_brief: str


class ScoutSubagent:
    def __init__(
        self,
        config: ScoutSubagentConfig,
    ):
        self.config = config
        self.model = create_model_from_spec(config.get("model"))
        fallback_spec = config.get("fallback_model")
        self.fallback_model = (
            create_model_from_spec(fallback_spec) if fallback_spec else None
        )

        self._tool_map = {}
        for tool_config in (config.get("tool_configs") or []):
            try:
                tool = create_tool_from_spec(tool_config)
                if tool.preflight_check():
                    self._tool_map[tool.get_name()] = tool
            except Exception as exc:
                logger.warning("Skipping scout tool %s: %s", tool_config, exc)

    def _generate_with_fallback(self, prompt: str, **generation_kwargs: Any) -> str:
        try:
            return self.model.generate(prompt, **self._generation_kwargs(**generation_kwargs))
        except Exception as exc:
            if not self.fallback_model:
                raise
            logger.warning(
                "Primary scout model failed; falling back to secondary model: %s",
                exc,
            )
            return self.fallback_model.generate(
                prompt,
                **self._generation_kwargs(**generation_kwargs),
            )

    def _generation_kwargs(self, **overrides: Any) -> Dict[str, Any]:
        base = {
            "max_new_tokens": self.config.get("max_new_tokens", 384),
            "temperature": self.config.get("temperature", 0.2),
            "do_sample": self.config.get("do_sample", False),
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

    def _search_web(self, query: str) -> List[Dict[str, Any]]:
        tool = self._tool_map.get("web_search_tool")
        if tool is None:
            raise ValueError("ScoutSubagent requires web_search_tool.")
        return tool.run(query)

    def _generate_queries(self, topic: str) -> List[str]:
        min_q = max(1, int(self.config.get("min_query_count", 3)))
        max_q = max(min_q, int(self.config.get("max_query_count", 5)))

        previous_queries_section = ""


        prompt = SCOUT_QUERY_PROMPT.format(
            topic=topic,
            previous_queries_section=previous_queries_section,
            query_count=max_q,
        )
        response = self._generate_with_fallback(
            prompt,
            max_new_tokens=256,
        )
        queries = self._parse_queries_from_response(response)
        queries = queries[:max_q]

        if len(queries) < min_q:
            normalized_topic = re.sub(r"\s+", " ", topic).strip()
            if normalized_topic and normalized_topic.lower() not in {
                query.lower() for query in queries
            }:
                queries.append(normalized_topic)

        return queries[:max_q]

    def _dedupe_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        max_results = max(1, int(self.config.get("max_landscape_results", 10)))
        deduped = []
        seen_urls = set()
        for item in results:
            url = (item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(item)
            if len(deduped) >= max_results:
                break
        return deduped

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No results were found during the scout search burst."

        lines = []
        for index, item in enumerate(results, start=1):
            title = (item.get("title") or "Untitled").strip()
            url = (item.get("url") or "").strip()
            snippet = re.sub(r"\s+", " ", (item.get("content") or "").strip())
            if len(snippet) > 320:
                snippet = snippet[:320].rstrip() + "..."
            published_date = (item.get("published_date") or "").strip()

            lines.append(f"[Result {index}]")
            lines.append(f"Title: {title}")
            if published_date:
                lines.append(f"Published: {published_date}")
            if url:
                lines.append(f"URL: {url}")
            if snippet:
                lines.append(f"Snippet: {snippet}")
            lines.append("")
        return "\n".join(lines).strip()

    def run(self, topic: str, observer: Any | None = None) -> ScoutResult:
        queries = self._generate_queries(topic)
        if observer is not None:
            observer.emit_event(
                kind="metric_update",
                component="scout",
                status="running",
                message="Scout generated queries.",
                stage="scout",
                metrics={"query_count": len(queries)},
            )
        max_results_per_query = max(1, int(self.config.get("max_results_per_query", 3)))

        search_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, len(queries))) as executor:
            futures = [executor.submit(self._search_web, query) for query in queries]
            for future in as_completed(futures):
                results = future.result() or []
                search_results.extend(results[:max_results_per_query])

        search_results = self._dedupe_results(search_results)
        if observer is not None:
            observer.emit_event(
                kind="metric_update",
                component="scout",
                status="running",
                message="Scout search burst completed.",
                stage="scout",
                metrics={"search_result_count": len(search_results)},
            )
        evidence_block = self._format_search_results(search_results)
        landscape_brief = self._generate_with_fallback(
            SCOUT_SUMMARY_PROMPT.format(
                topic=topic,
                scout_results=evidence_block,
            ),
            max_new_tokens=512,
        ).strip()
        if observer is not None:
            observer.emit_event(
                kind="metric_update",
                component="scout",
                status="completed",
                message="Scout landscape brief generated.",
                stage="scout",
                metrics={"landscape_brief_chars": len(landscape_brief)},
            )

        return {
            "topic": topic,
            "queries": queries,
            "search_results": search_results,
            "landscape_brief": landscape_brief,
        }
