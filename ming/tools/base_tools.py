from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional, TypedDict


class ModelConfig(TypedDict):
    model_name: str
    max_new_tokens: Optional[int]
    temperature: Optional[float]
    do_sample: Optional[bool]
    use_cache: Optional[bool]


class ToolParameterSchema(TypedDict):
    """Schema for a single tool parameter, for LLM-readable tool descriptions."""
    name: str
    type: str
    description: str
    required: bool


class ToolSchema(TypedDict):
    """Full tool schema for model decision-making: description, when to use, parameters."""

    description: str
    when_to_use: str
    parameters: List[ToolParameterSchema]


class BaseTool(ABC):
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    @abstractmethod
    def get_parameters(self) -> ToolSchema:
        """Return a rich schema (description, when_to_use, parameters) for model decision-making."""
        raise NotImplementedError

    def format_for_prompt(self) -> str:
        """Format tool schema as human-readable text for inclusion in prompts."""
        schema = self.get_parameters()
        lines = [
            f"## {self.name}",
            schema["description"],
            "",
            f"Use when: {schema['when_to_use']}",
            "",
            "Parameters:",
        ]
        for p in schema["parameters"]:
            req = "required" if p["required"] else "optional"
            lines.append(f"- {p['name']} ({req}): {p['description']}")
        return "\n".join(lines)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def preflight_check(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameters and return (is_valid, error_message)."""
        raise NotImplementedError
