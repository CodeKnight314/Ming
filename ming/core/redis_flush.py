"""Clear research Redis data before each orchestrator run.

startup.sh starts three containers: context (6379), queries (6380), kg (6381).
The runtime service also uses ``runtime:*`` keys on the context instance/db;
we must not FLUSHDB there or the control plane is wiped mid-job.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import redis

from ming.runtime.contracts import DEFAULT_RUNTIME_NAMESPACE

logger = logging.getLogger(__name__)


def _client(cfg: Mapping[str, Any], *, default_port: int) -> redis.Redis:
    return redis.Redis(
        host=str(cfg.get("hostname", "localhost")),
        port=int(cfg.get("port", default_port)),
        db=int(cfg.get("db", 0)),
        decode_responses=True,
    )


def _delete_keys_outside_namespace(client: redis.Redis, protected_prefix: str) -> int:
    """Delete all keys that do not start with ``protected_prefix`` (e.g. ``runtime:``)."""
    deleted = 0
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, count=500)
        for key in keys:
            if key.startswith(protected_prefix):
                continue
            client.delete(key)
            deleted += 1
        if cursor == 0:
            break
    return deleted


def _delete_keys_by_prefix(client: redis.Redis, prefix: str, protected_prefix: str = "") -> int:
    """Delete all keys matching ``prefix*``, optionally skipping ``protected_prefix``."""
    deleted = 0
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match=prefix + "*", count=500)
        for key in keys:
            if protected_prefix and key.startswith(protected_prefix):
                continue
            client.delete(key)
            deleted += 1
        if cursor == 0:
            break
    return deleted


def flush_research_redis_for_new_run(
    config: dict[str, Any],
    *,
    runtime_namespace: str = DEFAULT_RUNTIME_NAMESPACE,
    key_prefix: str = "",
) -> None:
    """
    Before each new research query/job:

    - **Context redis** (``config["redis"]``): remove every key except ``{namespace}:*``
      so runtime streams/snapshots survive on the shared 6379 instance.
    - **Queries redis** (``config["queries_redis"]``): ``FLUSHDB`` on that DB.
    - **KG redis** (``config["kg_redis"]``): ``FLUSHDB`` on that DB.

    When ``key_prefix`` is non-empty, only keys starting with that prefix are
    removed (prefix-scoped flush for concurrent workers sharing the same Redis
    instances).
    """
    protected = f"{runtime_namespace}:"

    redis_cfg = config.get("redis")
    if isinstance(redis_cfg, dict) and redis_cfg:
        ctx = _client(redis_cfg, default_port=6379)
        try:
            if key_prefix:
                n = _delete_keys_by_prefix(ctx, key_prefix, protected)
            else:
                n = _delete_keys_outside_namespace(ctx, protected)
            logger.info(
                "Context Redis %s:%s db=%s: removed %d keys (prefix=%r, kept %r)",
                redis_cfg.get("hostname", "localhost"),
                redis_cfg.get("port", 6379),
                redis_cfg.get("db", 0),
                n,
                key_prefix or "*",
                protected,
            )
        finally:
            ctx.close()

    for label, cfg_key in (("Queries", "queries_redis"), ("KG", "kg_redis")):
        cfg = config.get(cfg_key)
        if not isinstance(cfg, dict) or not cfg:
            continue
        cl = _client(cfg, default_port=6379)
        try:
            if key_prefix:
                n = _delete_keys_by_prefix(cl, key_prefix)
                logger.info(
                    "%s Redis %s:%s db=%s: removed %d keys (prefix=%r)",
                    label,
                    cfg.get("hostname", "localhost"),
                    cfg.get("port", 6379),
                    cfg.get("db", 0),
                    n,
                    key_prefix,
                )
            else:
                cl.flushdb()
                logger.info(
                    "%s Redis %s:%s db=%s: FLUSHDB",
                    label,
                    cfg.get("hostname", "localhost"),
                    cfg.get("port", 6379),
                    cfg.get("db", 0),
                )
        finally:
            cl.close()
