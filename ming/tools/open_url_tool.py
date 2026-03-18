import logging
import asyncio
import threading
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx
from trafilatura import extract

from ming.core.text_metrics import count_language_aware_tokens

from ming.tools.base_tools import BaseTool, ToolSchema

logger = logging.getLogger(__name__)

class OpenUrlTool(BaseTool):
    _REQUEST_TIMEOUT = httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0)
    _REQUEST_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }
    _MAX_RESPONSE_BYTES = 2_000_000
    _STREAM_CHUNK_SIZE = 65_536
    _HTML_CONTENT_TYPES = (
        "text/html",
        "application/xhtml+xml",
    )
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

    def __init__(self, name: str = "open_url_tool", min_tokens: int = 400):
        super().__init__(name)
        self.min_tokens = max(1, int(min_tokens))
        # NOTE: This tool is invoked from threadpools (ResearchSubagent) and must be
        # safe to call from arbitrary threads. httpx.AsyncClient is bound to an
        # asyncio loop and is not safe to share across threads/loops.
        #
        # We therefore run all async work on a dedicated background loop thread.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._client: httpx.AsyncClient | None = None
        self._thread_lock = threading.Lock()

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
        # Avoid a live network probe during tool construction.
        return True

    def _ensure_loop_thread(self) -> None:
        with self._thread_lock:
            if (
                self._loop_thread is not None
                and self._loop is not None
                and self._loop_thread.is_alive()
                and not self._loop.is_closed()
            ):
                return

            loop = asyncio.new_event_loop()

            def _runner() -> None:
                asyncio.set_event_loop(loop)
                try:
                    loop.run_forever()
                finally:
                    # Best-effort cleanup if the loop is stopped unexpectedly.
                    try:
                        if self._client is not None and not self._client.is_closed:
                            loop.run_until_complete(self._client.aclose())
                    except Exception:
                        pass
                    try:
                        loop.close()
                    except Exception:
                        pass

            thread = threading.Thread(
                target=_runner,
                name="OpenUrlToolLoop",
                daemon=True,
            )
            self._loop = loop
            self._loop_thread = thread
            self._client = None
            thread.start()

    async def _ensure_client_async(self) -> None:
        if self._client is not None and not self._client.is_closed:
            return
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            timeout=self._REQUEST_TIMEOUT,
            headers=self._REQUEST_HEADERS,
            follow_redirects=True,
        )

    def _run_on_loop(self, coro: "asyncio.Future[Dict[str, Any]]") -> Dict[str, Any]:
        self._ensure_loop_thread()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _fetch_html(self, url: str) -> Tuple[Optional[str], str]:
        await self._ensure_client_async()
        assert self._client is not None
        async with self._client.stream("GET", url) as response:
            status_code = response.status_code
            if status_code in {401, 403, 429}:
                raise httpx.HTTPStatusError(
                    f"Blocked with status {status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if content_type and not any(
                html_type in content_type for html_type in self._HTML_CONTENT_TYPES
            ):
                return None, f"unsupported content-type: {content_type}"

            body = bytearray()
            async for chunk in response.aiter_bytes(self._STREAM_CHUNK_SIZE):
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) > self._MAX_RESPONSE_BYTES:
                    return (
                        None,
                        (
                            f"response exceeded byte limit: "
                            f"{len(body)} > {self._MAX_RESPONSE_BYTES}"
                        ),
                    )

            html = bytes(body).decode(response.encoding or "utf-8", errors="ignore").strip()
            if not html:
                return None, "empty response body"
            return html, f"downloaded {len(body)} bytes"

    async def _run_async(self, url: str) -> Dict[str, Any]:
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
            html, fetch_notes = await self._fetch_html(url)
            if not html:
                result.update(
                    {
                        "status": "fetch_failed",
                        "discard_reason": "fetch_failed",
                        "retrieval_notes": fetch_notes,
                    }
                )
                return result

            content = extract(
                html,
                url=url,
                fast=True,
                no_fallback=True,
                include_comments=False,
                include_images=False,
                include_links=False,
            )
            if not content:
                result.update(
                    {
                        "status": "extract_failed",
                        "discard_reason": "extract_failed",
                        "retrieval_notes": f"{fetch_notes}; trafilatura.extract returned no content",
                    }
                )
                return result

            cleaned = content.strip()
            token_count = count_language_aware_tokens(cleaned)
            if token_count < self.min_tokens:
                result.update(
                    {
                        "content": cleaned,
                        "status": "low_content",
                        "discard_reason": "low_content",
                        "retrieval_notes": (
                            f"{fetch_notes}; "
                            f"Extracted content below minimum threshold: "
                            f"{token_count} < {self.min_tokens} language-aware tokens"
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
                    "retrieval_notes": (
                        f"{fetch_notes}; "
                        f"Extracted {len(cleaned)} characters and {token_count} language-aware tokens"
                    ),
                }
            )
            return result
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response is not None else None
            status = "blocked" if status_code in {401, 403, 429} else "fetch_failed"
            result.update(
                {
                    "status": status,
                    "discard_reason": status,
                    "retrieval_notes": f"HTTP error while fetching URL: {e}",
                }
            )
            return result
        except httpx.TimeoutException as e:
            result.update(
                {
                    "status": "fetch_failed",
                    "discard_reason": "fetch_failed",
                    "retrieval_notes": f"Timed out while fetching URL: {e}",
                }
            )
            return result
        except httpx.RequestError as e:
            result.update(
                {
                    "status": "fetch_failed",
                    "discard_reason": "fetch_failed",
                    "retrieval_notes": f"HTTP request failed while fetching URL: {e}",
                }
            )
            return result
        except Exception as e:
            error_text = str(e).lower()
            logger.warning("Unexpected open_url_tool failure for %s: %s", url, e)
            status = (
                "blocked"
                if any(token in error_text for token in ("403", "401", "429", "forbidden"))
                else "fetch_failed"
            )
            result.update(
                {
                    "status": status,
                    "discard_reason": status,
                    "retrieval_notes": f"Exception occurred while processing URL: {e}",
                }
            )
            return result

    def run(self, url: str) -> Dict[str, Any]:
        """Synchronous wrapper around the async implementation.

        The rest of the codebase expects tools to be synchronous (`BaseTool.run`).
        """
        return self._run_on_loop(self._run_async(url))

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

    def close(self) -> None:
        with self._thread_lock:
            loop = self._loop
            thread = self._loop_thread

        if loop is None or thread is None:
            return

        try:
            async def _shutdown() -> None:
                try:
                    if self._client is not None and not self._client.is_closed:
                        await self._client.aclose()
                finally:
                    asyncio.get_running_loop().stop()

            asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(timeout=10)
        except Exception:
            # If the loop is already closed/stopped, just ignore.
            pass

        with self._thread_lock:
            self._client = None
            self._loop = None
            self._loop_thread = None
