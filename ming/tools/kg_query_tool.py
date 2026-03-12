from ming.tools.base_tools import BaseTool
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
from ming.core.redis import RedisDatabase
from ming.extraction.kg_module import KGRedisStore, ERConfig

@dataclass
class KGQueryToolConfig:
    pass

class KGQueryTool(BaseTool):
    def __init__(self, config: KGQueryToolConfig, kg_store: KGRedisStore, name: str = "kg_query_tool"):
        super().__init__(name)
        self.config = config
        self.kg_store = kg_store

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        action = parameters.get("action")
        if action is None:
            return False, "Missing required parameter 'action'."
        if action not in ["get_neighbors", "find_connection"]:
            return False, "Invalid action. Must be 'get_neighbors' or 'find_connection'."
        if action == "get_neighbors":
            subject = parameters.get("subject")
            if subject is None:
                return False, "Missing required parameter 'subject'."
            if not isinstance(subject, str):
                return False, "Subject must be a string."
            return True, ""
        if action == "find_connection":
            subject = parameters.get("subject")
            object = parameters.get("object")
            if subject is None or object is None:
                return False, "Missing required parameters 'subject' and 'object'."
            if not isinstance(subject, str) or not isinstance(object, str):
                return False, "Subject and object must be strings."
            return True, ""
        return False, "Invalid action."

    def preflight_check(self) -> bool:
        if self.kg_store is None:
            return False
        if not isinstance(self.kg_store, KGRedisStore):
            return False
        return True

    def get_parameters(self):
        return {
            "description": "Query the knowledge graph for information.",
            "when_to_use": "When you need to query the knowledge graph for information.",
            "parameters": [
                {
                    "name": "action",
                    "type": "string",
                    "description": "The type of query: 'get_neighbors' (all facts for an entity), 'find_connection' (path between two entities)",
                    "required": True,
                },
                {
                    "name": "subject",
                    "type": "string",
                    "description": "The entity to query the knowledge graph for information. Required for 'get_neighbors' and 'find_connection' queries.",
                    "required": True,
                },
                {
                    "name": "object",
                    "type": "string",
                    "description": "The object to query the knowledge graph for information. Required for 'find_connection' queries.",
                    "required": False,
                }
            ]
        }

    def run(self, parameters: Dict[str, Any]) -> List[str]:
        action = parameters["action"]
        if action == "get_neighbors":
            return self._get_neighbors(parameters["subject"])
        if action == "find_connection":
            return self._find_connection(parameters["subject"], parameters["object"])
        return []

    def _get_neighbors(self, subject: str) -> List[str]:
        return self.kg_store.get_neighbors(subject)

    def _find_connection(self, subject: str, object: str) -> List[str]:
        return self.kg_store.find_connection(subject, object)