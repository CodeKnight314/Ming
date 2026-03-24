from ming.tools.base_tools import BaseTool
from typing import Any, Dict, Tuple, List

from ming.extraction.kg_module import KGRedisStore

class KGQueryTool(BaseTool):
    def __init__(self, kg_store: KGRedisStore, name: str = "kg_query_tool"):
        super().__init__(name)
        self.kg_store = kg_store

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        action = parameters.get("action")
        if action is None:
            return False, "Missing required parameter 'action'."
        valid_actions = [
            "get_neighbors", "find_connection", "search_evidence", "list_entities",
        ]
        if action not in valid_actions:
            return False, f"Invalid action. Must be one of: {', '.join(valid_actions)}."
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
        if action == "search_evidence":
            query = parameters.get("query")
            if query is None:
                return False, "Missing required parameter 'query'."
            if not isinstance(query, str):
                return False, "Query must be a string."
            limit = parameters.get("limit", 10)
            if not isinstance(limit, int) or limit <= 0:
                return False, "Limit must be a positive integer."
            diversify_by_url = parameters.get("diversify_by_url", True)
            if not isinstance(diversify_by_url, bool):
                return False, "diversify_by_url must be a boolean."
            return True, ""
        if action == "list_entities":
            keyword = parameters.get("keyword", "")
            if not isinstance(keyword, str):
                return False, "keyword must be a string."
            label = parameters.get("label", "")
            if not isinstance(label, str):
                return False, "label must be a string."
            limit = parameters.get("limit", 30)
            if not isinstance(limit, int) or limit <= 0:
                return False, "Limit must be a positive integer."
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
            "description": "Query the knowledge graph for information, ranked evidence cards, and entity discovery.",
            "when_to_use": (
                "Use `search_evidence` first to surface ranked facts and supporting URLs for a topic. "
                "Use `list_entities` to discover what entities exist in the KG matching a keyword or label — "
                "this helps you find valid names for `get_neighbors` and `find_connection`. "
                "Use `get_neighbors` to explore all facts about a specific entity. "
                "Use `find_connection` to trace relationships between two entities."
            ),
            "parameters": [
                {
                    "name": "action",
                    "type": "string",
                    "description": (
                        "The type of query: "
                        "'search_evidence' (ranked evidence cards for a query), "
                        "'list_entities' (discover entities in the KG by keyword/label filter — returns entity names, labels, fact counts, and top predicates), "
                        "'get_neighbors' (all facts for an entity), "
                        "'find_connection' (path between two entities)"
                    ),
                    "required": True,
                },
                {
                    "name": "subject",
                    "type": "string",
                    "description": "The entity to query the knowledge graph for information. You can search for entities in both English and Chinese. Required for 'get_neighbors' and 'find_connection' queries.",
                    "required": True,
                },
                {
                    "name": "object",
                    "type": "string",
                    "description": "The object to query the knowledge graph for information. You can search for entities in both English and Chinese. IMPORTANT: The 'subject' and 'object' must be in the same language. Required for 'find_connection' queries.",
                    "required": False,
                },
                {
                    "name": "query",
                    "type": "string",
                    "description": "The search query used for ranked evidence retrieval. Required for 'search_evidence'.",
                    "required": False,
                },
                {
                    "name": "keyword",
                    "type": "string",
                    "description": "Substring filter for entity names. Used with 'list_entities' to find entities containing this keyword (case-insensitive). Optional — omit or pass empty string to list top entities by fact count.",
                    "required": False,
                },
                {
                    "name": "label",
                    "type": "string",
                    "description": "Entity type filter for 'list_entities'. Common labels: ORG, PERSON, GPE, PRODUCT, EVENT, LOC, FAC, WORK_OF_ART, LAW. Case-insensitive. Optional.",
                    "required": False,
                },
                {
                    "name": "limit",
                    "type": "integer",
                    "description": "Maximum number of results to return. Used by 'search_evidence' and 'list_entities'.",
                    "required": False,
                },
                {
                    "name": "diversify_by_url",
                    "type": "boolean",
                    "description": "Whether to diversify top-ranked evidence cards by dominant supporting URL.",
                    "required": False,
                }
            ]
        }

    def run(
        self,
        action: str | Dict[str, Any],
        subject: str | None = None,
        object: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(action, dict):
            parameters = dict(action)
        else:
            parameters = {"action": action, "subject": subject, "object": object, **kwargs}

        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            raise ValueError(error)

        action = parameters["action"]
        if action == "get_neighbors":
            return self._get_neighbors(parameters["subject"])
        if action == "find_connection":
            return self._find_connection(parameters["subject"], parameters["object"])
        if action == "search_evidence":
            return self.search_evidence(
                parameters["query"],
                limit=parameters.get("limit", 10),
                diversify_by_url=parameters.get("diversify_by_url", True),
            )
        if action == "list_entities":
            return self._list_entities(
                keyword=parameters.get("keyword", ""),
                label=parameters.get("label", ""),
                limit=parameters.get("limit", 30),
            )
        return []

    def _get_neighbors(self, subject: str) -> List[str]:
        return self.kg_store.get_neighbors(subject)

    def _find_connection(self, subject: str, object: str) -> List[str]:
        return self.kg_store.find_connection(subject, object)

    def _list_entities(
        self,
        keyword: str = "",
        label: str = "",
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        return self.kg_store.list_entities(keyword=keyword, label=label, limit=limit)

    def search_evidence(
        self,
        query: str,
        limit: int = 10,
        diversify_by_url: bool = True,
    ) -> Dict[str, Any]:
        return self.kg_store.search_evidence(
            query=query,
            limit=limit,
            diversify_by_url=diversify_by_url,
        )
