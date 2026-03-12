from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from typing import Any, Dict

from ming.core.config import create_queries_store_from_config, create_redis_from_config
from ming.core.redis import QueryStoreConfig, RedisDatabaseConfig
from ming.scout import ScoutSubagent, ScoutSubagentConfig
from ming.subagent import ResearchSubagent, Agent, AgentConfig, ResearchSubagentConfig
from ming.extraction.kg_module import KGRedisStore, ERConfig
from ming.tools.kg_query_tool import KGQueryTool
from ming.extraction.ner_re_pipeline import NERREPipeline
from ming.models import OpenRouterModelConfig
from ming.core.prompts import PLANNING_PROMPT, FINAL_REPORT_PROMPT
import xml.etree.ElementTree
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class ResearchConfig:
    redis_config: RedisDatabaseConfig
    scout_config: dict[str, Any]
    subagent_config: dict[str, Any]
    queries_redis_config: QueryStoreConfig | dict[str, Any] | None = None
    num_research_subagents: int = 3


class ResearchOrchestrator:
    def __init__(self, config: ResearchConfig):
        self.config = config
        redis_cfg = {
            "redis": {
                "hostname": config.redis_config.hostname,
                "port": config.redis_config.port,
                "db": config.redis_config.db,
            }
        }
        self.context_redis = create_redis_from_config(redis_cfg)
        queries_cfg = {
            "redis": redis_cfg["redis"],
            "queries_redis": (
                {
                    "hostname": config.queries_redis_config.hostname,
                    "port": config.queries_redis_config.port,
                    "db": config.queries_redis_config.db,
                }
                if isinstance(config.queries_redis_config, QueryStoreConfig)
                else config.queries_redis_config
            ),
        }
        self.queries_redis = create_queries_store_from_config(queries_cfg)
        self.kg_redis = create_redis_from_config(redis_cfg)

        self.kg_store = KGRedisStore(self.kg_redis, ERConfig(threshold=0.5, num_perm=128))
        self.kg_query_tool = KGQueryTool(self.kg_store)
        self.nerre_pipeline = NERREPipeline(
            re_config=OpenRouterModelConfig(model_name="google/gemini-2.5-flash-lite"),
            kg_store=self.kg_store,
            ner_model_name="en_core_web_sm",
            max_workers=8,
        )

        self.scout = ScoutSubagent(
            config.scout_config
        )

        self.research_subagents = [ResearchSubagent(
            config=config.subagent_config,
            database=self.context_redis,
            query_store=self.queries_redis,
        ) for _ in range(config.num_research_subagents)]

        self.planning_agent = Agent(
            config=AgentConfig(
                model=OpenRouterModelConfig(
                    model_name="qwen/qwen3.5-plus-02-15",
                    temperature=0.2,
                    max_tokens=1024,
                ),
                system_prompt=PLANNING_PROMPT,
                tools=[self.kg_query_tool],
            )
        )

        self.final_report_agent = Agent(
            config=AgentConfig(
                model=OpenRouterModelConfig(
                    model_name="qwen/qwen3.5-plus-02-15",
                    temperature=0.2,
                    max_tokens=2048,
                ),
                system_prompt=FINAL_REPORT_PROMPT,
                tools=[self.kg_query_tool],
            )
        )

    @staticmethod
    def _parse_planning_result(planning_result: str) -> Dict[str, Any]:
        # Parse the planning result using xml.etree.ElementTree
        try:
            # Handle potential markdown wrapping or extra whitespace
            cleaned = planning_result.strip()
            if cleaned.startswith("```"):
                # Remove markdown fences if present
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            
            root = xml.etree.ElementTree.fromstring(cleaned)
        except xml.etree.ElementTree.ParseError as e:
            logger.error(f"Failed to parse planning result XML: {e}\nResult: {planning_result}")
            # Return a minimal valid structure to prevent total failure
            return {"research_angles": [], "constraints": ""}

        research_angles = root.findall("research_angles/research_angle")
        constraints_el = root.find("constraints")
        constraints = constraints_el.text if constraints_el is not None and constraints_el.text else ""
        
        return {
            "research_angles": [
                {
                    "topic": angle.find("topic").text if angle.find("topic") is not None else "Unknown Topic",
                    "success_criteria": angle.find("success_criteria").text if angle.find("success_criteria") is not None else "",
                }
                for angle in research_angles
            ], 
            "constraints": constraints
        }


    def run(self, query: str) -> str:
        # 1. Scout the web for information
        import time
        logger.info(f"Scouting landscape for: {query}")
        start_time = time.time()
        scout_result = self.scout.run(query)
        end_time = time.time()
        logger.info(f"Scouting landscape for: {query} took {round(end_time - start_time, 2)} seconds")
        
        # 2. Plan the research
        logger.info("Planning research angles...")
        start_time = time.time()
        planning_result = self.planning_agent.run(scout_result["landscape_brief"])
        research_plan = self._parse_planning_result(planning_result)
        end_time = time.time()
        logger.info(f"Planning research angles took {round(end_time - start_time, 2)} seconds")
        
        # 3. Parallel Execution of Research Subagents
        logger.info(f"Executing {len(research_plan['research_angles'])} research angles in parallel...")
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.num_research_subagents) as executor:
            future_to_angle = {}
            for angle in research_plan["research_angles"]:
                topic_prompt = (
                    f"Topic: {angle['topic']}\n"
                    f"Success Criteria: {angle['success_criteria']}\n"
                    f"Constraints: {research_plan['constraints']}"
                )
                
                subagent = ResearchSubagent(
                    config=self.config.subagent_config,
                    database=self.context_redis,
                    query_store=self.queries_redis,
                )
                
                future = executor.submit(subagent.run, topic_prompt, scout_result["landscape_brief"])
                future_to_angle[future] = angle
            
            for future in as_completed(future_to_angle):
                angle = future_to_angle[future]
                try:
                    res = future.result()
                    results.append(res)
                    logger.info(f"Completed research for angle: {angle['topic']}")
                except Exception as e:
                    logger.error(f"Subagent failed for angle '{angle['topic']}': {e}")
        end_time = time.time()
        logger.info(f"Executing research angles took {round(end_time - start_time, 2)} seconds")
        
        #how many total sources are there?
        total_sources = 0
        for res in results:
            total_sources += len(res.get("context_ids", []))
        logger.info(f"Total number of sources to process: {total_sources}")
        logger.info("Starting sequential NER/RE extraction into Knowledge Graph...")
        start_time = time.time()
        all_context_ids = set()
        for res in results:
            all_context_ids.update(res.get("context_ids", []))
            
        for cid in tqdm(all_context_ids):
            entry = self.context_redis.get_entry(cid)
            if entry and entry.get("raw_content"):
                try:
                    self.nerre_pipeline.run(entry["raw_content"], entry.get("url", ""))
                except Exception as e:
                    logger.warning(f"NER/RE failed for context {cid}: {e}")
        end_time = time.time()
        logger.info(f"NER/RE extraction took {round(end_time - start_time, 2)} seconds")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = ResearchConfig(
        redis_config=RedisDatabaseConfig(hostname="localhost", port=6379),
        scout_config=ScoutSubagentConfig(model={
            "provider": "openrouter",
            "model_name": "qwen/qwen3.5-flash-02-23",
            "temperature": 0.2,
            "max_tokens": 512,
        },
        tool_configs=[
            {
                "type": "web_search_tool",
                "max_results": 5,
                "search_depth": "basic",
                "topic": "general",
            }
        ]),
        subagent_config=ResearchSubagentConfig(model={
            "provider": "openrouter",
            "model_name": "qwen/qwen3.5-flash-02-23",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        tool_configs=[
            {"type": "web_search_tool", "max_results": 20},
            {"type": "open_url_tool"},
            {"type": "think_tool"},
        ]),
        queries_redis_config=QueryStoreConfig(hostname="localhost", port=6379),
        num_research_subagents=3,
    )
    orchestrator = ResearchOrchestrator(config)
    orchestrator.run("Write a research report on the history of the United States and its relationship with the Soviet Union.")
    # print(result)
