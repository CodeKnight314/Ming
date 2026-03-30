import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any, List, Union

from json_repair import repair_json
from ming.models import OpenRouterModelConfig, create_model_from_spec
from ming.extraction.kg_schema import Relationship

logger = logging.getLogger(__name__)

RE_JSON_ERROR_DIR = Path("re_json_errors")


def _config_to_spec(config: Union[dict, OpenRouterModelConfig]) -> dict:
    """Convert config to the dict format expected by create_model_from_spec."""
    if isinstance(config, dict):
        return config
    spec = {
        "provider": "openrouter",
        "model_name": config.model_name,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
        "site_url": config.site_url,
        "site_name": config.site_name,
    }
    if getattr(config, "model_kwargs", None):
        spec["model_kwargs"] = config.model_kwargs
    return spec


class REModule:
    def __init__(
        self,
        config: Union[dict, OpenRouterModelConfig],
    ):
        spec = _config_to_spec(config)
        self.model = create_model_from_spec(spec)

        self.prompt_template_zh = """你是一个关系抽取系统。给定一段文本和一个或多个目标实体，提取该文本中目标实体的所有事实关系。关系的宾语可以是另一个命名实体、概念或描述性属性。

        仅返回符合此模式的 JSON 数组：
        [
            {{
                "subject": "目标实体名称",
                "predicate": "关系动词或类型",
                "object": "相关的实体、概念或属性",
                "object_type": "entity" | "concept" | "attribute",
                "confidence": "high" | "medium" | "low"
            }}
        ]

        规则：
        - 仅为提供的目标实体提取关系，不要提取文中其他实体的关系。
        - 每个目标实体可以有零个或多个关系。
        - 如果目标实体没有任何有意义的关系，返回 []。
        - 置信度用离散档位，不要输出 0-1 小数：
          - high：文本中有明确、直接的陈述或强证据支持
          - medium：有合理暗示但不够直接，或表述较弱/有条件
          - low：可能为推断、模糊、或证据不足（尽量少输出这类关系）

        不要使用 markdown 代码块，不要有前言，不要有解释。

        文本片段：
        {text}

        目标实体：{target_entities}
        """
        self.prompt_template_en = """You are a relationship extraction system. Given a text passage and one or more target entities, extract factual relationships each target entity has in the passage. The object of a relationship can be another named entity, a concept, or a descriptive attribute.

        Return ONLY a JSON array with this schema:
        [
            {{
                "subject": "target entity name",
                "predicate": "relationship verb or type",
                "object": "related entity, concept, or attribute",
                "object_type": "entity" | "concept" | "attribute",
                "confidence": "high" | "medium" | "low"
            }}
        ]

        Rules:
        - Extract relationships ONLY for the provided target entities, not other entities in the passage.
        - Each target entity may have zero or more relationships.
        - If no meaningful relationships exist for any target entity, return [].
        - Confidence MUST be a discrete bucket, not a 0-1 float:
          - high: explicitly stated / directly supported in the passage
          - medium: plausible but weaker / indirect / hedged
          - low: speculative or weakly supported (avoid emitting these when possible)

        No markdown fences, no preamble, no explanation.

        Text passage:
        {text}

        Target entities: {target_entities}
        """

    def _is_chinese(self, text: str) -> bool:
        return any('\u4e00' <= char <= '\u9fff' for char in text)

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
        Uses json-repair for malformed output; returns [] if repair fails."""
        text = response.strip()
        fence_match = re.search(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            raw = json.loads(text)
            return raw if isinstance(raw, list) else []
        except json.JSONDecodeError as e:
            salvaged = self._salvage_truncated_array(response)
            if salvaged is not None:
                return salvaged

            try:
                raw = repair_json(text, return_objects=True)
                return raw if isinstance(raw, list) else []
            except (json.JSONDecodeError, Exception) as repair_err:
                logger.warning("json-repair failed (%s), returning empty list", repair_err)
                self._write_json_error(input_prompt, response, e)
                return []

    def _salvage_truncated_array(self, response: str) -> list[dict] | None:
        if not response:
            return None

        start = response.find("[")
        if start == -1:
            return None

        array_text = response[start:]

        last_good: list[dict] | None = None
        for idx, ch in enumerate(array_text):
            if ch != "}":
                continue
            prefix = array_text[: idx + 1]
            candidate = prefix + "]"
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    last_good = parsed
            except json.JSONDecodeError:
                continue

        return last_good

    def run(
        self, text: str, target_entities: List[str]
    ) -> List[Relationship]:
        if not target_entities:
            return []

        is_chinese = self._is_chinese(text)
        prompt_template = self.prompt_template_zh if is_chinese else self.prompt_template_en

        prompt = prompt_template.format(
            text=text,
            target_entities=", ".join(target_entities),
        )

        if hasattr(self.model, "generate_with_metadata"):
            response, _ = self.model.generate_with_metadata(prompt)
        else:
            response = self.model.generate(prompt)

        raw = self._parse_json_response(response, prompt)

        if not isinstance(raw, list):
            return []

        bucket_to_confidence = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
        }

        relationships: List[Relationship] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                raw_conf = item.get("confidence", 0.0)
                confidence: float
                if isinstance(raw_conf, str):
                    confidence = bucket_to_confidence.get(raw_conf.strip().lower(), 0.0)
                else:
                    confidence = float(raw_conf or 0.0)
                    # Backwards-compat: if older prompts emit 0-1 floats, clamp safely.
                    if confidence != confidence:  # NaN guard
                        confidence = 0.0
                    confidence = max(0.0, min(1.0, confidence))

                rel = Relationship(
                    relationship_id=uuid.uuid4().hex,
                    subject=str(item.get("subject", "")),
                    predicate=str(item.get("predicate", "")),
                    object=str(item.get("object", "")),
                    object_type=str(item.get("object_type", "attribute")),
                    confidence=confidence,
                )
                relationships.append(rel)
            except (TypeError, ValueError):
                continue

        return relationships
