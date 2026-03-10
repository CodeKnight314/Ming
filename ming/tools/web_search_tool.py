from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import requests
from urllib.parse import urlparse

from ming.tools.base_tools import BaseTool, ToolSchema

@dataclass
class WebSearchToolConfig:
    api_key: Optional[str] = None
    max_results: int = 30
    search_depth: str = "basic"
    topic: str = "general"
    include_raw_content: bool = False


class WebSearchTool(BaseTool):
    _UNSUPPORTED_DOMAINS = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "vimeo.com",
        "www.vimeo.com",
        "tiktok.com",
        "www.tiktok.com",
        "instagram.com",
        "www.instagram.com",
        "x.com",
        "www.x.com",
        "twitter.com",
        "www.twitter.com",
        "facebook.com",
        "www.facebook.com",
    }

    def __init__(self, config: WebSearchToolConfig, name: str = "web_search_tool"):
        super().__init__(name)
        self.config = config
        self.api_key = (config.api_key or os.environ.get("TAVILY_API_KEY", "")).strip()
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from tavily import TavilyClient
        except Exception as exc:
            raise RuntimeError(
                "tavily-python is not installed. Install with `pip install tavily-python`."
            ) from exc
        self._client = TavilyClient(self.api_key)

    def get_parameters(self) -> ToolSchema:
        return {
            "description": "Searches the web for fresh external information and returns grounded evidence with source URLs. Use this tool FIRST for any query that needs external information.",
            "when_to_use": "Use FIRST for any query requiring external information. Run web search before open_url_tool. Use when RAG results are insufficient, or when the question needs external/public standards, recent news, live data, or information not in the internal knowledge base.",
            "parameters": [
                {
                    "name": "query",
                    "type": "string",
                    "description": "Search query for the web.",
                    "required": True,
                },
            ],
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        query = parameters.get("query")

        if query is None:
            return False, "Missing required parameter 'query'."
        if not isinstance(query, str):
            return False, f"Parameter 'query' must be a string, got {type(query).__name__}."
        if not query.strip():
            return False, "Parameter 'query' cannot be empty."

        return True, ""

    def preflight_check(self) -> bool:
        # Avoid network call in preflight: only verify credentials exist.
        return bool(self.api_key)

    def _clean_text(self, text: str, max_chars: int = 1200) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars] + "..."

    def _post_search(self, query: str, max_results: int) -> Dict[str, Any]:
        self._ensure_client()
        search_kwargs: Dict[str, Any] = {
            "query": query,
            "include_answer": False,
            "search_depth": self.config.search_depth,
            "max_results": max_results,
            "include_images": False,
            "topic": self.config.topic,
            "include_raw_content": self.config.include_raw_content,
        }

        data = self._client.search(**search_kwargs)
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected Tavily response format.")
        return data

    def _classify_source(self, url: str, title: str) -> Dict[str, Any]:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        title_lower = (title or "").lower()

        source_type = "webpage"
        discard_candidate = False
        discard_reason = None

        if domain in self._UNSUPPORTED_DOMAINS:
            source_type = "video"
            discard_candidate = True
            discard_reason = "unsupported_source_type"
        elif path.endswith(".pdf"):
            source_type = "pdf"
        elif "video" in title_lower:
            source_type = "video"
            discard_candidate = True
            discard_reason = "unsupported_source_type"

        retrieval_notes = (
            f"source_type={source_type}; discard_candidate={discard_candidate}"
        )
        if discard_reason:
            retrieval_notes += f"; discard_reason={discard_reason}"

        return {
            "source_type": source_type,
            "discard_candidate": discard_candidate,
            "discard_reason": discard_reason,
            "status": "unsupported" if discard_candidate else "success",
            "discard": bool(discard_candidate),
            "retrieval_notes": retrieval_notes,
        }

    def _normalize_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_results, list):
            return []

        normalized: List[Dict[str, Any]] = []
        seen_urls = set()

        for item in raw_results:
            if not isinstance(item, dict):
                continue

            url = str(item.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = str(item.get("title", "")).strip() or "Untitled"
            source_info = self._classify_source(url, title)

            normalized.append(
                {
                    "title": title,
                    "url": url,
                    "score": item.get("score"),
                    "published_date": item.get("published_date"),
                    "content": str(item.get("raw_content") or item.get("content") or "").strip(),
                    **source_info,
                }
            )

        return normalized

    def run(self, query: str) -> Dict[str, Any]:
        is_valid, error = self.validate_parameters({"query": query})
        if not is_valid:
            return []

        if not self.api_key:
            return []

        try:
            response = self._post_search(query=query, max_results=self.config.max_results)
            results = self._normalize_results(response.get("results"))
            return results
        except Exception as exc:
            return []

    def check_api_usage(self) -> int:
        response = requests.get(
            "https://api.tavily.com/usage",
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        if response.status_code != 200:
            return {}
        else: 
            data = response.json()
            limits = {
                "total_requests": data["key"]["usage"],
                "limit": data["key"]["limit"],
                "current_plan": data["account"]["current_plan"], 
                "plan_usage": data["account"]["plan_usage"],
                "plan_limit": data["account"]["plan_limit"],
                "paygo_usage": data["account"]["paygo_usage"],
                "paygo_limit": data["account"]["paygo_limit"],
            }
            return limits
