from dataclasses import dataclass
from typing import Any, Dict

from ming.core.config import (
    create_queries_store_from_config,
    create_redis_from_config,
)
from ming.core.redis import RedisDatabaseConfig
from ming.scout import ScoutSubagent
from ming.subagent import ResearchSubagent


@dataclass
class ResearchConfig:
    redis_config: RedisDatabaseConfig
    scout_config: dict[str, Any]
    subagent_config: dict[str, Any]
    queries_redis_config: dict[str, Any] | None = None
    num_research_subagents: int = 3


class ResearchOrchestrator:
    def __init__(self, config: ResearchConfig):
        self.config = config
        redis_cfg = {
            "redis": {
                "hostname": config.redis_config.hostname,
                "port": config.redis_config.port,
            }
        }
        self.redis = create_redis_from_config(redis_cfg)
        full_config = {**redis_cfg}
        if config.queries_redis_config:
            full_config["queries_redis"] = config.queries_redis_config
        self.query_store = create_queries_store_from_config(full_config)
        self.scout = ScoutSubagent(
            config.scout_config,
            query_store=self.query_store,
        )
        self.research_subagents = [ResearchSubagent(
            config=config.subagent_config,
            database=self.redis,
            query_store=self.query_store,
        ) for _ in range(config.num_research_subagents)]

    def run(self, query: str): 
        pass
