"""Load config.json for Ming."""

import json
from pathlib import Path
from typing import Any, Dict

from ming.core.redis import (
    QueryStore,
    QueryStoreConfig,
    RedisDatabase,
    RedisDatabaseConfig,
)
from ming.scout import ScoutSubagent, ScoutSubagentConfig
from ming.subagent import ResearchSubagent, ResearchSubagentConfig


def load_config(path: str | Path = "config.json") -> dict[str, Any]:
    """Load config from JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p) as f:
        return json.load(f)


def create_redis_from_config(
    config: dict[str, Any] | RedisDatabaseConfig | None = None,
) -> RedisDatabase:
    """Create RedisDatabase from a config dict or RedisDatabaseConfig."""
    if config is None:
        config = load_config()
    if isinstance(config, RedisDatabaseConfig):
        db_config = config
    else:
        redis_cfg = config.get("redis", {})
        db_config = RedisDatabaseConfig(
            hostname=str(redis_cfg.get("hostname", "localhost")),
            port=int(redis_cfg.get("port", 6379)),
            db=int(redis_cfg.get("db", 0)),
        )
    return RedisDatabase(db_config)


def create_queries_store_from_config(
    config: dict[str, Any] | None = None,
) -> QueryStore:
    """Create QueryStore from config. Uses queries_redis if present, else redis with db=1."""
    if config is None:
        config = load_config()
    queries_cfg = config.get("queries_redis")
    if queries_cfg:
        store_config = QueryStoreConfig(
            hostname=str(queries_cfg.get("hostname", "localhost")),
            port=int(queries_cfg.get("port", 6379)),
            db=int(queries_cfg.get("db", 1)),
        )
    else:
        redis_cfg = config.get("redis", {})
        store_config = QueryStoreConfig(
            hostname=str(redis_cfg.get("hostname", "localhost")),
            port=int(redis_cfg.get("port", 6379)),
            db=1,
        )
    return QueryStore(store_config)


def create_subagent_from_config(
    config: dict[str, Any] | None = None,
    database: RedisDatabase | None = None,
    query_store: QueryStore | None = None,
) -> ResearchSubagent:
    """Create ResearchSubagent from config. Uses config.json if config is None."""
    if config is None:
        config = load_config()
    if database is None:
        database = create_redis_from_config(config)
    if query_store is None:
        query_store = create_queries_store_from_config(config)

    subagent_cfg = dict(config.get("subagent", {}))
    if "source_min_tokens" in config:
        subagent_cfg["source_min_tokens"] = int(config["source_min_tokens"])
    tool_configs = []
    for tool_config in subagent_cfg.get("tool_configs", []) or []:
        if isinstance(tool_config, dict):
            normalized = dict(tool_config)
            if normalized.get("type") == "open_url_tool" and "source_min_tokens" in config:
                normalized.setdefault("min_tokens", int(config["source_min_tokens"]))
            tool_configs.append(normalized)
        else:
            tool_configs.append(tool_config)
    if tool_configs:
        subagent_cfg["tool_configs"] = tool_configs
    subagent_config: ResearchSubagentConfig = {
        k: v for k, v in subagent_cfg.items() if v is not None
    }

    return ResearchSubagent(
        config=subagent_config,
        database=database,
        query_store=query_store,
    )


def create_research_config_from_config(
    config: dict[str, Any] | None = None,
) -> ResearchSubagentConfig:
    """Build ResearchSubagentConfig from config dict. For use with ResearchSubagent."""
    from ming.core.redis import RedisDatabaseConfig

    if config is None:
        config = load_config()
    redis_cfg = config.get("redis", {})
    return ResearchSubagentConfig(
        redis_config=RedisDatabaseConfig(
            hostname=str(redis_cfg.get("hostname", "localhost")),
            port=int(redis_cfg.get("port", 6379)),
            db=int(redis_cfg.get("db", 0)),
        ),
        scout_config=config.get("scout", {}),
        subagent_config=config.get("subagent", {}),
        queries_redis_config=config.get("queries_redis"),
    )


def create_ming_deep_research_config(
    config: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build MingDeepResearchConfig from config dict."""
    from ming.models.openrouter_model import OpenRouterModelConfig
    from ming.core.redis import QueryStoreConfig, RedisDatabaseConfig
    from ming.orchestrator import MingDeepResearchConfig

    if config is None:
        config = load_config()

    redis_cfg = config.get("redis", {})
    redis_config = RedisDatabaseConfig(
        hostname=str(redis_cfg.get("hostname", "localhost")),
        port=int(redis_cfg.get("port", 6379)),
        db=int(redis_cfg.get("db", 0)),
    )

    queries_cfg = config.get("queries_redis")
    if queries_cfg:
        queries_redis_config = QueryStoreConfig(
            hostname=str(queries_cfg.get("hostname", "localhost")),
            port=int(queries_cfg.get("port", 6379)),
            db=int(queries_cfg.get("db", 1)),
        )
    else:
        queries_redis_config = None

    writer_model_cfg = config.get("writer_model")
    if writer_model_cfg:
        writer_model = OpenRouterModelConfig(
            model_name=writer_model_cfg.get("model_name", "qwen/qwen3.5-plus-02-15"),
            temperature=float(writer_model_cfg.get("temperature", 0.2)),
            max_tokens=int(writer_model_cfg.get("max_tokens", 4096)),
            site_url=writer_model_cfg.get("site_url"),
            site_name=writer_model_cfg.get("site_name"),
            model_kwargs=writer_model_cfg.get("model_kwargs"),
        )
    else:
        writer_model = None

    writer_fallback_cfg = config.get("writer_fallback_model")
    if writer_fallback_cfg:
        writer_fallback_model = OpenRouterModelConfig(
            model_name=writer_fallback_cfg.get("model_name", "qwen/qwen3.5-flash-02-23"),
            temperature=float(writer_fallback_cfg.get("temperature", 0.2)),
            max_tokens=int(writer_fallback_cfg.get("max_tokens", 4096)),
            site_url=writer_fallback_cfg.get("site_url"),
            site_name=writer_fallback_cfg.get("site_name"),
            model_kwargs=writer_fallback_cfg.get("model_kwargs"),
        )
    else:
        writer_fallback_model = None

    outline_model_cfg = config.get("outline_model")
    if outline_model_cfg:
        outline_model = OpenRouterModelConfig(
            model_name=outline_model_cfg.get("model_name", "qwen/qwen3.5-plus-02-15"),
            temperature=float(outline_model_cfg.get("temperature", 0.2)),
            max_tokens=int(outline_model_cfg.get("max_tokens", 8192)),
            site_url=outline_model_cfg.get("site_url"),
            site_name=outline_model_cfg.get("site_name"),
            model_kwargs=outline_model_cfg.get("model_kwargs"),
        )
    else:
        outline_model = None

    outline_fallback_cfg = config.get("outline_fallback_model")
    if outline_fallback_cfg:
        outline_fallback_model = OpenRouterModelConfig(
            model_name=outline_fallback_cfg.get("model_name", "qwen/qwen3.5-flash-02-23"),
            temperature=float(outline_fallback_cfg.get("temperature", 0.2)),
            max_tokens=int(outline_fallback_cfg.get("max_tokens", 8192)),
            site_url=outline_fallback_cfg.get("site_url"),
            site_name=outline_fallback_cfg.get("site_name"),
            model_kwargs=outline_fallback_cfg.get("model_kwargs"),
        )
    else:
        outline_fallback_model = None

    kg_cfg = config.get("kg_redis")
    if kg_cfg:
        kg_redis_config = RedisDatabaseConfig(
            hostname=str(kg_cfg.get("hostname", "localhost")),
            port=int(kg_cfg.get("port", 6379)),
            db=int(kg_cfg.get("db", 0)),
        )
    else:
        kg_redis_config = None

    return MingDeepResearchConfig(
        redis_config=redis_config,
        scout_config=config.get("scout", {}),
        subagent_config=config.get("subagent", {}),
        queries_redis_config=queries_redis_config,
        kg_redis_config=kg_redis_config,
        writer_model=writer_model,
        writer_fallback_model=writer_fallback_model,
        outline_model=outline_model,
        outline_fallback_model=outline_fallback_model,
        draft_output_path=config.get("draft_output_path"),
        num_research_subagents=int(config.get("num_research_subagents", 3)),
        outline_max_context_ids=int(config.get("outline_max_context_ids", 64)),
        outline_context_token_budget=int(
            config.get("outline_context_token_budget", 128_000)
        ),
        outline_source_char_limit=int(config.get("outline_source_char_limit", 3000)),
        source_min_tokens=int(config.get("source_min_tokens", 400)),
        research_source_budget=int(config.get("research_source_budget", 250)),
        max_sources_threshold=int(config.get("max_sources_threshold", 160)),
        max_chunks_per_source=int(config.get("max_chunks_per_source", 8)),
        source_score_cutoff=float(config.get("source_score_cutoff", 4.5)),
        writer_num_parallel_sections=int(config.get("writer_num_parallel_sections", 8)),
    )


def create_scout_from_config(
    config: dict[str, Any] | None = None,
    query_store: QueryStore | None = None,
) -> ScoutSubagent:
    """Create ScoutSubagent from config. Uses config.json if config is None."""
    if config is None:
        config = load_config()
    if query_store is None:
        query_store = create_queries_store_from_config(config)

    scout_cfg = config.get("scout", {})
    scout_config: ScoutSubagentConfig = {
        k: v for k, v in scout_cfg.items() if v is not None
    }

    return ScoutSubagent(config=scout_config, query_store=query_store)
