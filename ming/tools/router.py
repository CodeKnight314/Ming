from typing import Any, Union

from ming.tools.base_tools import BaseTool
from ming.tools.think_tool import ThinkTool, ThinkToolConfig
from ming.tools.web_search_tool import WebSearchTool, WebSearchToolConfig
from ming.tools.open_url_tool import OpenUrlTool

ToolConfig = Union[ThinkToolConfig, WebSearchToolConfig, dict[str, Any]]


def create_tool_from_spec(spec: ToolConfig) -> BaseTool:
    """Create a tool from a spec dictionary or config object.

    Dict spec keys: type (required), plus type-specific config.
    Supported types: think_tool, web_search_tool, open_url_tool.
    """
    if isinstance(spec, ThinkToolConfig):
        return ThinkTool(config=spec)
    if isinstance(spec, WebSearchToolConfig):
        return WebSearchTool(config=spec)

    if not isinstance(spec, dict):
        raise ValueError("Tool spec must be a dictionary or config object.")

    tool_type = str(spec.get("type", "")).strip().lower()
    if not tool_type:
        raise ValueError("Tool spec must include 'type'.")

    if tool_type == "think_tool":
        model_config = spec.get("model_config")
        if not model_config:
            raise ValueError("think_tool spec requires 'model_config'.")
        config = ThinkToolConfig(
            model_config=model_config,
            max_new_tokens=int(spec.get("max_new_tokens", 384)),
            temperature=float(spec.get("temperature", 0.7)),
            do_sample=bool(spec.get("do_sample", False)),
            use_cache=bool(spec.get("use_cache", True)),
        )
        return ThinkTool(config=config)

    if tool_type == "web_search_tool":
        config = WebSearchToolConfig(
            api_key=spec.get("api_key"),
            max_results=int(spec.get("max_results", 30)),
            search_depth=str(spec.get("search_depth", "basic")),
            topic=str(spec.get("topic", "general")),
            include_raw_content=bool(spec.get("include_raw_content", False)),
        )
        return WebSearchTool(config=config)

    if tool_type == "open_url_tool":
        return OpenUrlTool(
            name=spec.get("name", "open_url_tool"),
            min_tokens=int(spec.get("min_tokens", 400)),
        )

    raise ValueError(f"Unsupported tool type '{tool_type}'.")
