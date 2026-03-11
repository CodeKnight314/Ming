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
        Synthesize the retrieved content below into a coherent research synthesis.

        Reflect on:
        - What is the core question or goal?
        - Where do findings agree or disagree? Reconcile conflicts with methodology/recency/authority.
        - What mechanisms or causes explain the facts?
        - What patterns emerge across sources? What gaps remain?

        Output guidelines:
        - Cite sources inline as [1], [2], [3]. End with ## Sources listing [N] Title: URL.
        - Explain significance: why numbers matter, what trends imply.
        - Write in flowing paragraphs. No self-referential language. Match the topic language.
        - Produce synthesis that informs whether more research is needed.

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

