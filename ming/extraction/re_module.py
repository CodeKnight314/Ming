import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

from ming.models import create_model_from_spec, OpenRouterModelConfig

logger = logging.getLogger(__name__)

RE_JSON_ERROR_DIR = Path("re_json_errors")


@dataclass
class Relationship:
    subject: str
    predicate: str
    object: str
    object_type: str
    confidence: float


@dataclass
class REUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


@dataclass
class RERunResult:
    relationships: List[Relationship]
    usage: REUsage


def _config_to_spec(config: Union[dict, OpenRouterModelConfig]) -> dict:
    """Convert config to the dict format expected by create_model_from_spec."""
    if isinstance(config, dict):
        return config
    spec = {
        "provider": "openrouter",
        "model_name": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "site_url": config.site_url,
        "site_name": config.site_name,
    }
    if getattr(config, "model_kwargs", None):
        spec["model_kwargs"] = config.model_kwargs
    return spec


class REModule:
    def __init__(self, config: Union[dict, OpenRouterModelConfig]):
        spec = _config_to_spec(config)
        self.model = create_model_from_spec(spec)

        self.prompt_template = """You are a relationship extraction system. Given a text passage and one or more target entities, extract factual relationships each target entity has in the passage. The object of a relationship can be another named entity, a concept, or a descriptive attribute.

        Return ONLY a JSON array with this schema:
        [
            {{
                "subject": "target entity name",
                "predicate": "relationship verb or type",
                "object": "related entity, concept, or attribute",
                "object_type": "entity" | "concept" | "attribute",
                "confidence": 0.0-1.0
            }}
        ]

        Rules:
        - Extract relationships ONLY for the provided target entities, not other entities in the passage.
        - Each target entity may have zero or more relationships.
        - If no meaningful relationships exist for any target entity, return [].

        No markdown fences, no preamble, no explanation.

        Text passage:
        {text}

        Target entities: {target_entities}
        """

    def _write_json_error(
        self, input_prompt: str, raw_output: str, error: json.JSONDecodeError
    ) -> None:
        """Write failed JSON parse to its own error file with input and raw output."""
        RE_JSON_ERROR_DIR.mkdir(parents=True, exist_ok=True)
        path = RE_JSON_ERROR_DIR / f"re_json_error_{uuid.uuid4().hex[:12]}.txt"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# JSONDecodeError: {error}\n\n")
                f.write("# --- Input (prompt) ---\n")
                f.write(input_prompt)
                f.write("\n\n# --- Raw output ---\n")
                f.write(raw_output)
            logger.info("Wrote failed JSON response to %s", path)
        except OSError as e:
            logger.warning("Could not write JSON error file: %s", e)

    def _parse_json_response(self, response: str, input_prompt: str = "") -> List[dict]:
        """Parse JSON from model response, stripping markdown fences if present.
        Handles truncated or malformed output by attempting recovery or returning []."""
        text = response.strip()
        fence_match = re.search(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            raw = json.loads(text)
            return raw if isinstance(raw, list) else []
        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed (%s), attempting recovery", e)
            recovered = self._try_recover_truncated_json(text, e)
            if recovered is not None:
                return recovered
            logger.warning("JSON recovery failed, returning empty list")
            self._write_json_error(input_prompt, response, e)
            return []

    def _try_recover_truncated_json(self, text: str, error: json.JSONDecodeError) -> List[dict] | None:
        """Try to salvage partial results from truncated JSON (e.g. hit max_tokens)."""
        # Find last complete object boundary: "},{" indicates two complete objects
        last_boundary = text.rfind("},{")
        if last_boundary >= 0:
            # Truncate after the first complete object, close the array
            truncated = text[: last_boundary + 1] + "]"
            try:
                raw = json.loads(truncated)
                return raw if isinstance(raw, list) else []
            except json.JSONDecodeError:
                pass
        # Try extracting just the first complete object
        first_obj_end = text.find("}")
        if first_obj_end >= 0 and text.strip().startswith("["):
            truncated = text[: first_obj_end + 1] + "]"
            try:
                raw = json.loads(truncated)
                return raw if isinstance(raw, list) else []
            except json.JSONDecodeError:
                pass
        return None

    def _extract_usage(self, metadata: dict[str, Any] | None) -> REUsage:
        if not metadata:
            return REUsage()

        usage_metadata = metadata.get("usage_metadata") or {}
        response_metadata = metadata.get("response_metadata") or {}
        cost = response_metadata.get("cost", 0.0) or 0.0

        return REUsage(
            input_tokens=int(usage_metadata.get("input_tokens", 0) or 0),
            output_tokens=int(usage_metadata.get("output_tokens", 0) or 0),
            total_tokens=int(usage_metadata.get("total_tokens", 0) or 0),
            cost=float(cost),
        )

    def run_with_metadata(
        self, text: str, target_entities: List[str]
    ) -> RERunResult:
        if not target_entities:
            return RERunResult(relationships=[], usage=REUsage())

        prompt = self.prompt_template.format(
            text=text,
            target_entities=", ".join(target_entities),
        )

        metadata: dict[str, Any] | None = None
        if hasattr(self.model, "generate_with_metadata"):
            response, metadata = self.model.generate_with_metadata(prompt)
        else:
            response = self.model.generate(prompt)

        raw = self._parse_json_response(response, prompt)

        if not isinstance(raw, list):
            return RERunResult(relationships=[], usage=self._extract_usage(metadata))

        relationships = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                rel = Relationship(
                    subject=str(item.get("subject", "")),
                    predicate=str(item.get("predicate", "")),
                    object=str(item.get("object", "")),
                    object_type=str(item.get("object_type", "attribute")),
                    confidence=float(item.get("confidence", 0.0)),
                )
                relationships.append(rel)
            except (TypeError, ValueError):
                continue

        return RERunResult(
            relationships=relationships,
            usage=self._extract_usage(metadata),
        )

    def run(
        self, text: str, target_entities: List[str]
    ) -> List[Relationship]:
        return self.run_with_metadata(text, target_entities).relationships
