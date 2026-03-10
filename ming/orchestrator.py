from dataclasses import dataclass
from typing import Any, Dict

from ming.core.config import create_redis_from_config
from ming.core.redis import RedisDatabaseConfig
from ming.scout import ScoutSubagent
from ming.subagent import ResearchSubagent


@dataclass
class ResearchConfig:
    redis_config: RedisDatabaseConfig
    scout_config: dict[str, Any]
    subagent_config: dict[str, Any]


class ResearchOrchestrator:
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.redis = create_redis_from_config(
            {
                "redis": {
                    "hostname": config.redis_config.hostname,
                    "port": config.redis_config.port,
                }
            }
        )
        self.scout = ScoutSubagent(config.scout_config)
        self.research_subagent = ResearchSubagent(
            config=config.subagent_config,
            database=self.redis,
        )

    def scout_topic(self, query: str) -> Dict[str, Any]:
        return self.scout.run(query)

    def run(self, query: str) -> Dict[str, Any]:
        scout_result = self.scout.run(query)
        research_result = self.research_subagent.run(
            topic=query,
            scout_report=scout_result.get("landscape_brief", ""),
        )
        return {
            "query": query,
            "scout": scout_result,
            "research": research_result,
        }
