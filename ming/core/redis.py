import json
import redis
from dataclasses import dataclass
from uuid import uuid4
from typing import Any, List, Optional

_URL_INDEX_PREFIX = "url:"
_URL_LOCK_PREFIX = "url:lock:"
_URL_LOCK_TTL = 120  # seconds
_QUERIES_PREFIX = "queries:"


@dataclass
class RedisDatabaseConfig:
    hostname: str
    port: int
    db: int = 0


class RedisDatabase:
    def __init__(self, config: RedisDatabaseConfig):
        self.client = redis.Redis(
            host=config.hostname,
            port=config.port,
            db=config.db,
            decode_responses=True,
        )
        self.config = config

    def create_entry(self, entry: dict[str, Any]) -> str:
        uuid = str(uuid4())
        self.client.hset(uuid, mapping=self._serialize_entry(entry))
        return uuid

    def get_entry(self, uuid: str) -> dict[str, Any]:
        return self.client.hgetall(uuid)

    def delete_entry(self, uuid: str):
        self.client.delete(uuid)

    def update_entry(self, uuid: str, entry: dict[str, Any]):
        self.client.hset(uuid, mapping=self._serialize_entry(entry))

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize non-scalar values to Redis-compatible strings."""
        if value is None:
            return ""
        if isinstance(value, (str, bytes, int, float)):
            return value
        if isinstance(value, set):
            return json.dumps(sorted(value))
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value)
        return str(value)

    @classmethod
    def _serialize_entry(cls, entry: dict[str, Any]) -> dict[str, Any]:
        return {key: cls._serialize_value(value) for key, value in entry.items()}

    def get_context_id_by_url(self, url: str) -> Optional[str]:
        """Return the context ID (UUID) for a cached URL, or None if not cached."""
        return self.client.get(_URL_INDEX_PREFIX + url)

    def set_url_index(self, url: str, context_id: str) -> None:
        """Associate a URL with its context ID in the cache index."""
        self.client.set(_URL_INDEX_PREFIX + url, context_id)

    def try_acquire_url_fetch_lock(self, url: str, ttl: int = _URL_LOCK_TTL) -> bool:
        """Acquire an exclusive lock for fetching this URL. Returns True if acquired."""
        return bool(self.client.set(_URL_LOCK_PREFIX + url, "1", nx=True, ex=ttl))

    def release_url_fetch_lock(self, url: str) -> None:
        """Release the fetch lock for a URL."""
        self.client.delete(_URL_LOCK_PREFIX + url)

    def close(self):
        self.client.close()


@dataclass
class QueryStoreConfig:
    """Config for QueryStore. Uses a separate Redis DB or instance to store search queries."""
    hostname: str
    port: int
    db: int = 1


class QueryStore:
    """Stores search queries per topic. Used to avoid repeating queries and to provide context for generation."""

    def __init__(self, config: QueryStoreConfig):
        self.client = redis.Redis(
            host=config.hostname,
            port=config.port,
            db=config.db,
            decode_responses=True,
        )
        self.config = config

    def _key(self, topic: str) -> str:
        return _QUERIES_PREFIX + topic.strip().lower()

    def add_queries(self, topic: str, queries: List[str]) -> None:
        """Add queries to the store for a topic. Deduplicates automatically."""
        if not topic or not queries:
            return
        key = self._key(topic)
        for q in queries:
            q_clean = (q or "").strip()
            if q_clean:
                self.client.sadd(key, q_clean)

    def get_queries(self, topic: str) -> List[str]:
        """Return all stored queries for a topic."""
        if not topic:
            return []
        key = self._key(topic)
        members = self.client.smembers(key)
        return sorted(members) if members else []

    def close(self) -> None:
        self.client.close()