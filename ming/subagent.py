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

        self.max_iterations = config.get("max_iterations", 3)
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


@dataclass
class AgentConfig:
    model: dict[str, Any]
    system_prompt: str
    tools: List[ToolConfig] | None = None
    max_iterations: int = 15


class AgentMessage(TypedDict):
    role: str
    content: str


class AgentState(TypedDict, total=False):
    messages: List[AgentMessage]
    output: str
    iteration: int


class Agent:
    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def __init__(self, config: AgentConfig):
        model_spec = config.model
        if isinstance(model_spec, OpenRouterModelConfig):
            model_spec = asdict(model_spec)
            model_spec["provider"] = "openrouter"
        self.model = create_model_from_spec(model_spec)
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
        self.graph = self._build_graph()

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
                logger.warning("Malformed tool_call JSON: %s", match.group(1)[:120])
        return calls

    def _strip_tool_calls(self, text: str) -> str:
        return self._TOOL_CALL_RE.sub("", text).strip()

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        prompt = self._messages_to_prompt(state["messages"])
        response = self.model.generate(prompt)
        messages = state["messages"] + [{"role": "assistant", "content": response}]
        return {"messages": messages, "output": response}

    def _execute_tools(self, state: AgentState) -> Dict[str, Any]:
        last_response = state.get("output", "")
        tool_calls = self._parse_tool_calls(last_response)

        results: List[str] = []
        for call in tool_calls:
            name = call.get("name", "")
            params = call.get("parameters", {})
            tool = self._tool_map.get(name)
            if tool is None:
                results.append(f"[{name}] Error: unknown tool.")
                continue
            valid, err = tool.validate_parameters(params)
            if not valid:
                results.append(f"[{name}] Validation error: {err}")
                continue
            try:
                result = tool.run(**params)
                results.append(f"[{name}]\n{result}")
            except Exception as exc:
                results.append(f"[{name}] Error: {exc}")

        tool_content = "\n\n".join(results)
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

    def run(self, user_input: str) -> str:
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
            return self.model.generate(prompt)

        result = self.graph.invoke(initial)
        return self._strip_tool_calls(result.get("output", ""))

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
