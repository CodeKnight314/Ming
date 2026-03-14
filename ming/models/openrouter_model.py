import base64
import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter

from ming.models.base_model import BaseModel

logger = logging.getLogger(__name__)


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
        load_dotenv()
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
        try:
            response = self.client.invoke([HumanMessage(content=content)])
        except Exception:
            logger.exception("OpenRouter request failed for model=%s", self.config.model_name)
            raise
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
