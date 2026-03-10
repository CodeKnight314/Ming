from ming.tools.base_tools import BaseTool, ToolSchema
from ming.models import BaseModel, create_model_from_spec
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ThinkToolConfig:
    """Config for ThinkTool. model_config is used when model is not passed."""
    model_config: dict[str, Any]
    max_new_tokens: int = 384
    temperature: float = 0.7
    do_sample: bool = False
    use_cache: bool = True


class ThinkTool(BaseTool):
    def __init__(
        self,
        config: ThinkToolConfig,
        model: BaseModel | None = None,
        name: str = "think_tool",
    ):
        super().__init__(name)
        self.config = config
        self.model = model or create_model_from_spec(config.model_config)
        self.generation_params = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "do_sample": config.do_sample,
            "use_cache": config.use_cache,
        }

        self.system_prompt = """
        Stop and think. Do not summarize or report. Reason over what you have done so far in relation to the query.

        Before planning or acting, take a moment to genuinely reflect:
        - What is the core question or goal?
        - What assumptions am I making? Are they valid?
        - What alternative approaches or angles have I not considered?
        - If I'm stuck, what would help me get unstuck?

        Think step by step. Challenge your own reasoning. Only after this reflection should you decide what to do next.

        Current context: {context}
        """

    def get_parameters(self) -> ToolSchema:
        return {
            "description": "Pause to reason and reflect before planning or acting. Use when you need to think through the problem, challenge assumptions, or consider alternatives—not to summarize.",
            "when_to_use": "When you need to stop and think before deciding: when stuck, when facing ambiguity, when about to make a significant decision, or when you want to verify your reasoning before proceeding.",
            "parameters": [
                {
                    "name": "context",
                    "type": "string",
                    "description": "The current context and the results of the previous tool calls.",
                    "required": True,
                }
            ],
        }

    def run(self, context: str) -> str:
        prompt = self.system_prompt.format(context=context)
        response = self.model.generate(
            prompt,
            **self.generation_params,
        )
        return response
    
    def preflight_check(self) -> bool:
        if self.model is None:
            return False
        if not isinstance(self.model, BaseModel):
            return False
        return True

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        context = parameters.get("context")
        if context is None:
            return False, "Missing required parameter 'context'."
        if not isinstance(context, str):
            return False, "Context must be a string"
        return True, ""

