from typing import Any, Dict, List, Tuple

from ming.tools.base_tools import BaseTool, ToolSchema, ToolParameterSchema


class ThinkTool(BaseTool):
    def __init__(self) -> None:
        super().__init__("think_tool")

    def get_parameters(self) -> ToolSchema:
        return ToolSchema(
            description=(
                "Use this tool to pause and reason about your research progress. "
                "It does not fetch new information — it is a structured reflection step. "
                "Pass your reasoning as the 'reasoning' parameter and the tool will "
                "echo it back so you can reference it in later steps."
            ),
            when_to_use=(
                "Call this tool AFTER every search_evidence call and BEFORE you "
                "start writing. Assess: What key facts did the last query surface? "
                "Which subsections now have sufficient evidence? What gaps remain? "
                "Should you query again or begin writing?"
            ),
            parameters=[
                ToolParameterSchema(
                    name="reasoning",
                    type="string",
                    description="Your assessment of current evidence and plan for next steps.",
                    required=True,
                ),
            ],
        )

    def run(self, reasoning: str = "", **kwargs: Any) -> str:
        return reasoning

    def preflight_check(self) -> bool:
        return True

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        if "reasoning" not in parameters or not isinstance(parameters.get("reasoning"), str):
            return False, "Parameter 'reasoning' (string) is required."
        return True, ""
