from typing import Any, Union

from ming.tools.base_tools import BaseTool
from ming.tools.web_search_tool import WebSearchTool, WebSearchToolConfig
from ming.tools.open_url_tool import OpenUrlTool

ToolConfig = Union[WebSearchToolConfig, dict[str, Any]]


def create_tool_from_spec(spec: ToolConfig) -> BaseTool:
    """Create a tool from a spec dictionary or config object.

    Dict spec keys: type (required), plus type-specific config.
    Supported types: web_search_tool, open_url_tool.
    """
    if isinstance(spec, WebSearchToolConfig):
        return WebSearchTool(config=spec)

    if not isinstance(spec, dict):
        raise ValueError("Tool spec must be a dictionary or config object.")

    tool_type = str(spec.get("type", "")).strip().lower()
    if not tool_type:
        raise ValueError("Tool spec must include 'type'.")

    if tool_type == "web_search_tool":
        config = WebSearchToolConfig(
            api_key=spec.get("api_key"),
            max_results=int(spec.get("max_results", 30)),
            search_depth=str(spec.get("search_depth", "basic")),
            topic=str(spec.get("topic", "general")),
            include_raw_content=spec.get("include_raw_content", "text"),
            score_cutoff=float(spec.get("score_cutoff", 0.5)),
        )
        return WebSearchTool(config=config)

    if tool_type == "open_url_tool":
        return OpenUrlTool(
            name=spec.get("name", "open_url_tool"),
            min_tokens=int(spec.get("min_tokens", 400)),
        )

    raise ValueError(f"Unsupported tool type '{tool_type}'.")
