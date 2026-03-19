import base64
import json
import logging
import mimetypes
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter

from ming.models.base_model import BaseModel

logger = logging.getLogger(__name__)

_MAX_INVOKE_ATTEMPTS = 5
_INITIAL_RETRY_DELAY_SECONDS = 1.0


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _is_retryable_openrouter_error(exc: BaseException) -> bool:
    retryable_type_names = {
        "responsevalidationerror",
        "timeout",
        "readtimeout",
        "connecttimeout",
        "remoteprotocolerror",
        "apiconnectionerror",
    }
    retryable_message_fragments = (
        "eof while parsing",
        "expecting value",
        "connection reset",
        "connection aborted",
        "remote end closed connection",
        "server disconnected without sending a response",
        "temporarily unavailable",
        "timed out",
        "code': 5",  # OpenRouter occasionally returns {'error': {'code': 5xx, ...}}
    )

    for error in _iter_exception_chain(exc):
        if isinstance(error, (TimeoutError, ConnectionError, json.JSONDecodeError)):
            return True
        error_type_name = type(error).__name__.lower()
        error_message = str(error).lower()
        if error_type_name in retryable_type_names:
            return True
        if any(fragment in error_message for fragment in retryable_message_fragments):
            return True
    return False


def _summarize_openrouter_error_payload_from_text(text: str) -> str | None:
    """Best-effort extraction of OpenRouter {'error': ...} payload from exception strings.

    langchain_openrouter sometimes raises ResponseValidationError when the HTTP response body
    is an OpenRouter error payload (e.g. code=502) rather than a normal chat completion.
    We can't reliably access the raw body, so we parse the exception text conservatively.
    """
    lower = text.lower()
    if "input_value" not in lower or "{'error':" not in text:
        return None

    # Example fragment (often truncated):
    # input_value={'error': {'message': 'Up...content.', 'code': 502}}, input_type=dict
    code_match = re.search(r"'code'\s*:\s*(\d{3})", text)
    message_match = re.search(r"'message'\s*:\s*'([^']+)'", text)
    if not code_match and not message_match:
        return "OpenRouter error payload (unparsed)"

    code = code_match.group(1) if code_match else "unknown"
    message = message_match.group(1) if message_match else "unknown"
    return f"OpenRouter upstream error code={code} message={message}"


def _exception_chain_summary(exc: BaseException) -> str:
    parts: list[str] = []
    for error in _iter_exception_chain(exc):
        text = str(error)
        parsed = _summarize_openrouter_error_payload_from_text(text)
        if parsed:
            parts.append(parsed)
            continue
        parts.append(f"{type(error).__name__}: {error}")
    return " <- ".join(parts)


@dataclass
class OpenRouterModelConfig:
    model_name: str = "openai/gpt-4o"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    site_url: Optional[str] = None
    site_name: Optional[str] = None
    model_kwargs: Optional[dict[str, Any]] = None


def _build_chat_openrouter(
    model: str,
    api_key: str,
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    site_url: Optional[str] = None,
    site_name: Optional[str] = None,
) -> Any:
    """Build ChatOpenRouter client matching OpenRouter usage pattern.
    ChatOpenRouter requires 'reasoning' as a top-level param, not inside model_kwargs."""
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if site_url:
        kwargs["app_url"] = site_url
    if site_name:
        kwargs["app_title"] = site_name

    # Extract reasoning from model_kwargs and pass as top-level (required by ChatOpenRouter)
    if model_kwargs:
        model_kwargs = dict(model_kwargs)
        if "reasoning" in model_kwargs:
            kwargs["reasoning"] = model_kwargs.pop("reasoning")
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

    return ChatOpenRouter(**kwargs)


class OpenRouterModel(BaseModel):
    def __init__(self, config: Optional[OpenRouterModelConfig] = None):
        # Prefer repo-local .env over any pre-exported shell variables.
        load_dotenv(override=True)
        self.config = config or OpenRouterModelConfig()
        self.prompt = ""

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required. Set it in .env or environment variables."
            )

        self.client = _build_chat_openrouter(
            model=self.config.model_name,
            api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            model_kwargs=self.config.model_kwargs,
            site_url=self.config.site_url,
            site_name=self.config.site_name,
        )

    def _image_part(self, image: str) -> Optional[dict]:
        """Build image part for OpenRouter multimodal format. Returns None if image cannot be loaded."""
        if image.startswith("data:") and ";base64," in image:
            header, data = image.split(";base64,", 1)
            mime_type = header.replace("data:", "", 1) or "image/png"
            return {"type": "image", "base64": data, "mime_type": mime_type}
        if image.startswith("http://") or image.startswith("https://"):
            return {"type": "image", "url": image}

        path = Path(image)
        if not path.exists():
            logger.warning("Image path does not exist: %s", image)
            return None
        try:
            mime_type, _ = mimetypes.guess_type(path.name)
            mime_type = mime_type or "application/octet-stream"
            data = base64.b64encode(path.read_bytes()).decode("utf-8")
            return {"type": "image", "base64": data, "mime_type": mime_type}
        except Exception as e:
            logger.warning("Failed to read image from path %s: %s", image, e)
            return None

    def _build_message_content(self, prompt: str, images: Optional[List[str]] = None) -> list:
        content: list = [{"type": "text", "text": prompt}]
        if images:
            for image in images:
                part = self._image_part(image)
                if part is not None:
                    content.append(part)
        return content

    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        **generation_kwargs: Any,
    ) -> str:
        response_text, _ = self.generate_with_metadata(prompt, images, **generation_kwargs)
        return response_text

    def generate_with_metadata(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        **generation_kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        if images:
            images = self._validate_images(images)
        content = self._build_message_content(prompt, images)
        logger.info(
            "Calling OpenRouter model=%s prompt_chars=%d images=%d",
            self.config.model_name,
            len(prompt),
            len(images or []),
        )
        start_time = time.time()
        last_exception: Exception | None = None
        for attempt in range(1, _MAX_INVOKE_ATTEMPTS + 1):
            try:
                response = self.client.invoke([HumanMessage(content=content)])
                break
            except Exception as exc:
                last_exception = exc
                if attempt >= _MAX_INVOKE_ATTEMPTS or not _is_retryable_openrouter_error(exc):
                    logger.exception(
                        "OpenRouter request failed for model=%s after attempt %d/%d (%s)",
                        self.config.model_name,
                        attempt,
                        _MAX_INVOKE_ATTEMPTS,
                        _exception_chain_summary(exc),
                    )
                    raise

                sleep_seconds = _INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                # Add a small jitter to avoid synchronized retries when running batches.
                sleep_seconds = sleep_seconds * (0.85 + 0.3 * random.random())
                logger.warning(
                    "Retrying OpenRouter request for model=%s after attempt %d/%d in %.1fs (%s)",
                    self.config.model_name,
                    attempt,
                    _MAX_INVOKE_ATTEMPTS,
                    sleep_seconds,
                    _exception_chain_summary(exc),
                )
                time.sleep(sleep_seconds)
        else:
            raise RuntimeError(
                f"OpenRouter request failed unexpectedly for model={self.config.model_name}"
            ) from last_exception

        elapsed = time.time() - start_time
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        metadata = {
            "usage_metadata": getattr(response, "usage_metadata", None),
            "response_metadata": getattr(response, "response_metadata", None) or {},
        }
        logger.info(
            "OpenRouter response received model=%s elapsed=%.2fs usage=%s",
            self.config.model_name,
            elapsed,
            metadata["usage_metadata"],
        )
        return response_text, metadata

    def __call__(self, prompt: str, images: Optional[List[str]] = None, **generation_kwargs: Any) -> str:
        return self.generate(prompt, images, **generation_kwargs)

    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    def get_prompt(self) -> str:
        return self.prompt
