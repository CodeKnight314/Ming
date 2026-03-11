"""Load config.json for Ming."""

import json
from pathlib import Path
from typing import Any

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


def create_redis_from_config(config: dict[str, Any] | None = None) -> RedisDatabase:
    """Create RedisDatabase from config. Uses config.json if config is None."""
    if config is None:
        config = load_config()
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

    subagent_cfg = config.get("subagent", {})
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
) -> "ResearchConfig":
    """Build ResearchConfig from config dict. For use with ResearchOrchestrator."""
    from ming.core.redis import RedisDatabaseConfig
    from ming.orchestrator import ResearchConfig

    if config is None:
        config = load_config()
    redis_cfg = config.get("redis", {})
    return ResearchConfig(
        redis_config=RedisDatabaseConfig(
            hostname=str(redis_cfg.get("hostname", "localhost")),
            port=int(redis_cfg.get("port", 6379)),
            db=int(redis_cfg.get("db", 0)),
        ),
        scout_config=config.get("scout", {}),
        subagent_config=config.get("subagent", {}),
        queries_redis_config=config.get("queries_redis"),
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
