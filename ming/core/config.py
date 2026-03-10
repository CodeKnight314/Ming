"""Load config.json for Ming."""

import json
from pathlib import Path
from typing import Any

from ming.core.redis import RedisDatabase, RedisDatabaseConfig
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
    )
    return RedisDatabase(db_config)


def create_subagent_from_config(
    config: dict[str, Any] | None = None,
    database: RedisDatabase | None = None,
) -> ResearchSubagent:
    """Create ResearchSubagent from config. Uses config.json if config is None."""
    if config is None:
        config = load_config()
    if database is None:
        database = create_redis_from_config(config)

    subagent_cfg = config.get("subagent", {})
    subagent_config: ResearchSubagentConfig = {
        k: v for k, v in subagent_cfg.items() if v is not None
    }

    return ResearchSubagent(config=subagent_config, database=database)


def create_scout_from_config(
    config: dict[str, Any] | None = None,
) -> ScoutSubagent:
    """Create ScoutSubagent from config. Uses config.json if config is None."""
    if config is None:
        config = load_config()

    scout_cfg = config.get("scout", {})
    scout_config: ScoutSubagentConfig = {
        k: v for k, v in scout_cfg.items() if v is not None
    }

    return ScoutSubagent(config=scout_config)
