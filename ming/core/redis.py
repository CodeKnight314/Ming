import redis
from dataclasses import dataclass
from uuid import uuid4
from typing import Any, Optional

_URL_INDEX_PREFIX = "url:"
_URL_LOCK_PREFIX = "url:lock:"
_URL_LOCK_TTL = 120  # seconds


@dataclass
class RedisDatabaseConfig:
    hostname: str
    port: int


class RedisDatabase:
    def __init__(self, config: RedisDatabaseConfig):
        self.client = redis.Redis(
            host=config.hostname,
            port=config.port,
            decode_responses=True,
        )
        self.config = config

    def create_entry(self, entry: dict[str, Any]) -> str:
        uuid = str(uuid4())
        self.client.hset(uuid, mapping=entry)
        return uuid

    def get_entry(self, uuid: str) -> dict[str, Any]:
        return self.client.hgetall(uuid)

    def delete_entry(self, uuid: str):
        self.client.delete(uuid)

    def update_entry(self, uuid: str, entry: dict[str, Any]):
        self.client.hset(uuid, mapping=entry)

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