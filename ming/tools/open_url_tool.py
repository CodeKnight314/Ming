from ming.tools.base_tools import BaseTool, ToolSchema
import httpx
from trafilatura import fetch_url, extract
from typing import Dict, Any, Tuple
from urllib.parse import urlparse

class OpenUrlTool(BaseTool):
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

    _MIN_CONTENT_CHARS = 200

    def __init__(self, name: str = "open_url_tool"):
        super().__init__(name)

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        url = parameters.get("url")
        if url is None:
            return False, "Missing required parameter 'url'."
        if not isinstance(url, str):
            return False, f"Parameter 'url' must be a string, got {type(url).__name__}."
        if not url.strip():
            return False, "Parameter 'url' cannot be empty."
        return True, ""

    def preflight_check(self) -> bool:
        try:
            response = httpx.get("https://www.google.com", timeout=10.0)
            return response.status_code == 200
        except Exception:
            return False

    def run(self, url: str) -> Dict[str, Any]:
        """
        Fetch a URL and extract its main text content.
        Returns structured retrieval status and extracted content when available.
        """
        result = {
            "url": url,
            "content": None,
            "status": "fetch_failed",
            "discard": True,
            "discard_reason": None,
            "source_type": "webpage",
            "retrieval_notes": "",
        }
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        if domain in self._UNSUPPORTED_DOMAINS:
            result.update(
                {
                    "status": "unsupported",
                    "discard_reason": "unsupported_source_type",
                    "source_type": "video",
                    "retrieval_notes": f"Skipped unsupported domain: {domain}",
                }
            )
            return result

        if path.endswith(".pdf"):
            result["source_type"] = "pdf"

        try:
            html = fetch_url(url)
            if not html:
                result.update(
                    {
                        "status": "fetch_failed",
                        "discard_reason": "fetch_failed",
                        "retrieval_notes": "fetch_url returned no content",
                    }
                )
                return result

            content = extract(html)
            if not content:
                result.update(
                    {
                        "status": "extract_failed",
                        "discard_reason": "extract_failed",
                        "retrieval_notes": "trafilatura.extract returned no content",
                    }
                )
                return result

            cleaned = content.strip()
            if len(cleaned) < self._MIN_CONTENT_CHARS:
                result.update(
                    {
                        "content": cleaned,
                        "status": "low_content",
                        "discard_reason": "low_content",
                        "retrieval_notes": (
                            f"Extracted content below minimum threshold: "
                            f"{len(cleaned)} < {self._MIN_CONTENT_CHARS}"
                        ),
                    }
                )
                return result

            result.update(
                {
                    "content": cleaned,
                    "status": "success",
                    "discard": False,
                    "discard_reason": None,
                    "retrieval_notes": f"Extracted {len(cleaned)} characters",
                }
            )
            return result
        except Exception as e:
            error_text = str(e).lower()
            status = "blocked" if any(token in error_text for token in ("403", "401", "429", "forbidden")) else "fetch_failed"
            result.update(
                {
                    "status": status,
                    "discard_reason": status,
                    "retrieval_notes": f"Exception occurred while processing URL: {e}",
                }
            )
            return result

    def get_parameters(self) -> ToolSchema:
        return {
            "description": "Fetches and extracts the full text content from a URL. Returns the original page content as-is without summarization. Use AFTER web_search_tool to retrieve full content from URLs found in search results.",
            "when_to_use": "Use AFTER web_search_tool when you need the full page content from specific URLs. Pass URLs from web search results. Preserve and return the extracted content verbatim—do not summarize or prepare for final response.",
            "parameters": [
                {
                    "name": "url",
                    "type": "string",
                    "description": "The URL to open.",
                    "required": True,
                }
            ],
        }
