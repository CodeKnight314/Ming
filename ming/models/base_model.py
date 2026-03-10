import base64
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import httpx
import requests

logger = logging.getLogger(__name__)


def url_to_base64_data(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch image from URL and return as data URI. Returns None on failure."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/png").split(";")[0].strip()
        b64 = base64.b64encode(resp.content).decode("utf-8")
        return f"data:{content_type};base64,{b64}"
    except Exception as e:
        logger.warning("Failed to fetch image from URL %s: %s", url[:80], e)
        return None


class BaseModel(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        **generation_kwargs: Any,
    ) -> str:
        """Generate text from a prompt and optional images."""
        raise NotImplementedError

    @abstractmethod
    def set_prompt(self, prompt: str) -> None:
        """Store an internal prompt value."""
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self) -> str:
        """Return the internally stored prompt value."""
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        **generation_kwargs: Any,
    ) -> str:
        """Convenience callable wrapper around generation."""
        raise NotImplementedError

    def _validate_images(self, images: List[str]) -> List[str]:
        """Validate images and return a list of valid images."""
        import os

        valid_images = []

        for img in images:
            if img.startswith("http://") or img.startswith("https://"):
                try:
                    with httpx.Client() as client:
                        response = client.head(img)
                        if response.status_code == 200:
                            valid_images.append(img)
                except httpx.RequestError:
                    continue
            elif img.startswith("data:image/") and ";base64," in img:
                valid_images.append(img)
            elif os.path.exists(img):
                valid_images.append(img)

        return valid_images
