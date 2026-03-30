"""Model router for creating models from spec dicts."""

from typing import Any

from ming.models.base_model import BaseModel
from ming.models.openrouter_model import OpenRouterModel, OpenRouterModelConfig


def _resolve_max_new_tokens_from_spec(spec: dict[str, Any]) -> int | None:
    """Prefer max_new_tokens; accept legacy max_tokens on spec or generation_config."""
    gen = spec.get("generation_config") or {}
    if not isinstance(gen, dict):
        gen = {}
    for bucket, key in (
        (spec, "max_new_tokens"),
        (gen, "max_new_tokens"),
        (spec, "max_tokens"),
        (gen, "max_tokens"),
    ):
        if key in bucket and bucket[key] is not None:
            return int(bucket[key])
    return None


def create_model_from_spec(spec: dict[str, Any]) -> BaseModel:
    """Create a model from a spec dictionary.
    Supports provider: openrouter (default for ming).
    Spec keys: provider, model_name (or model), temperature, max_new_tokens, site_url, site_name.
    """
    if not isinstance(spec, dict):
        raise ValueError("Model spec must be a dictionary.")

    provider = str(spec.get("provider", "openrouter")).strip().lower()
    model_name = str(spec.get("model_name") or spec.get("model") or "").strip()
    if not model_name:
        raise ValueError("Model spec must include 'model_name'.")

    generation_config = spec.get("generation_config", {})
    if not isinstance(generation_config, dict):
        generation_config = {}

    temperature = float(
        generation_config.get("temperature", spec.get("temperature", 0.0))
    )
    max_new_tokens = _resolve_max_new_tokens_from_spec(spec)

    if provider == "openrouter":
        return OpenRouterModel(
            OpenRouterModelConfig(
                model_name=model_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                site_url=spec.get("site_url"),
                site_name=spec.get("site_name"),
                model_kwargs=spec.get("model_kwargs"),
            )
        )
    else:
        raise ValueError(f"Unsupported provider '{provider}'.")