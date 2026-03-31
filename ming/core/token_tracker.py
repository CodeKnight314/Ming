from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "call_count": self.call_count,
        }


class TokenTracker:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: Dict[str, ModelUsage] = {}
        self._total_web_queries: int = 0

    def record_llm_usage(
        self,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        with self._lock:
            usage = self._models.setdefault(model_name, ModelUsage())
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens
            usage.call_count += 1

    def record_web_queries(self, count: int = 1) -> None:
        with self._lock:
            self._total_web_queries += count

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total_input = 0
            total_output = 0
            total_calls = 0
            models = {}
            for name, usage in sorted(self._models.items()):
                models[name] = usage.to_dict()
                total_input += usage.input_tokens
                total_output += usage.output_tokens
                total_calls += usage.call_count
            return {
                "models": models,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_llm_calls": total_calls,
                "total_web_queries": self._total_web_queries,
            }
