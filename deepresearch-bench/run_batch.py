#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import redis
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ming.core.config import create_ming_deep_research_config, load_config
from ming.orchestrator import MingDeepResearch

logger = logging.getLogger("deepresearch_bench")

_TAVILY_USAGE_URL = "https://api.tavily.com/usage"
_HOTSWAP_THRESHOLD = 20  # credits remaining before rotating to next key


@dataclass
class _TavilyKey:
    label: str
    key: str


@dataclass
class TavilyKeyManager:
    """Manages a pool of Tavily API keys and hot-swaps when a key is near its free limit.

    Only used by run_batch. Not part of the deployment or standalone paths.
    """

    _keys: list[_TavilyKey] = field(default_factory=list)
    _current_index: int = field(default=0, init=False)
    _last_usage_info: dict[str, Any] | None = field(default=None, init=False)

    @classmethod
    def from_sources(cls, keys_file: Path | None, env_key: str | None) -> TavilyKeyManager:
        """Build a manager from the env key and an optional keys file.

        The env key is placed first so it is tried before the file keys.
        Duplicate keys (by value) are silently dropped.
        """
        seen: set[str] = set()
        keys: list[_TavilyKey] = []

        def _add(label: str, key: str) -> None:
            stripped = key.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                keys.append(_TavilyKey(label=label, key=stripped))

        if env_key:
            _add("env", env_key)

        if keys_file and keys_file.exists():
            for line in keys_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    label, _, raw_key = line.partition(":")
                    _add(label.strip(), raw_key.strip())
                else:
                    _add(f"key-{len(keys) + 1}", line)

        instance = cls(_keys=keys)
        if keys:
            instance._apply_current_key()
        return instance

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ensure_headroom(self) -> bool:
        """Check the active key and rotate if it is within the hotswap threshold.

        Returns True when a usable key is available, False when all keys are
        exhausted and the batch should stop.
        """
        while self._current_index < len(self._keys):
            remaining = self._query_remaining_credits()
            if remaining is None:
                # Usage check failed — assume OK and let the run proceed
                # rather than blocking on a transient API error.
                logger.warning(
                    "Could not verify Tavily credits for key '%s'. Proceeding anyway.",
                    self._current_key.label,
                )
                return True

            label = self._current_key.label
            logger.info(
                "Tavily key '%s': %d credits remaining (hotswap threshold: %d).",
                label,
                remaining,
                _HOTSWAP_THRESHOLD,
            )

            if remaining > _HOTSWAP_THRESHOLD:
                return True

            self._last_usage_info = None  # will be refreshed on next key

            logger.warning(
                "Tavily key '%s' has only %d credits remaining — rotating to next key.",
                label,
                remaining,
            )
            self._current_index += 1
            if self._current_index < len(self._keys):
                self._apply_current_key()

        logger.error(
            "All %d Tavily key(s) are exhausted (≤ %d credits each). "
            "Cannot start next report without exceeding free limits.",
            len(self._keys),
            _HOTSWAP_THRESHOLD,
        )
        return False

    def get_last_usage_info(self) -> dict[str, Any] | None:
        """Return the last fetched usage for the current key, for dashboard display."""
        if self._last_usage_info is None:
            return None
        return {
            "active_key_label": self._current_key.label,
            "key_preview": f"{self._current_key.key[:16]}…",
            **self._last_usage_info,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _current_key(self) -> _TavilyKey:
        return self._keys[self._current_index]

    def _apply_current_key(self) -> None:
        key = self._current_key
        os.environ["TAVILY_API_KEY"] = key.key
        logger.info("Active Tavily key: '%s' (%s…)", key.label, key.key[:16])

    def _query_remaining_credits(self) -> int | None:
        """Return credits remaining for the active key, or None on failure."""
        key = self._current_key.key
        try:
            resp = requests.get(
                _TAVILY_USAGE_URL,
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
        except requests.RequestException as exc:
            logger.warning("Tavily usage request failed: %s", exc)
            return None

        if resp.status_code != 200:
            logger.warning(
                "Tavily usage endpoint returned HTTP %d for key '%s'.",
                resp.status_code,
                self._current_key.label,
            )
            return None

        try:
            data = resp.json()
            # For hotswapping we treat `plan_limit` as the total credit pool and
            # `plan_usage` as the credits consumed. This batch run only consumes
            # search credits, so we disregard other usage categories.
            account = data.get("account") or {}
            plan_limit = account.get("plan_limit")
            raw_plan_usage = account.get("plan_usage", None)
            if raw_plan_usage is None:
                raw_plan_usage = account.get("search_usage", 0)
            plan_usage = int(raw_plan_usage or 0)
            if plan_limit is None:
                self._last_usage_info = {"usage": plan_usage, "limit": None, "remaining": 999_999}
                return 999_999
            limit_int = int(plan_limit)
            remaining = limit_int - plan_usage
            self._last_usage_info = {"usage": plan_usage, "limit": limit_int, "remaining": remaining}
            return remaining
        except Exception as exc:
            logger.warning("Failed to parse Tavily usage response: %s", exc)
            return None


@dataclass(frozen=True)
class RedisTarget:
    name: str
    host: str
    port: int
    db: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ming deepresearch over prompts in a query.jsonl file with resume support."
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=REPO_ROOT / "deepresearch-bench" / "query_data" / "query.jsonl",
        help="Path to the JSONL file containing prompt records.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "deepresearch-bench" / "reports",
        help="Directory where id_<prompt_id>.md files will be written.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=REPO_ROOT / "deepresearch-bench" / "reports" / "run_state.json",
        help="JSON file used to track progress for resume support.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config.json",
        help="Path to the Ming config file.",
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        dest="prompt_ids",
        help="Specific prompt id(s) to run. Repeat the flag to select multiple ids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N selected prompts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run prompts even if they are already marked completed.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a failed prompt instead of stopping immediately.",
    )
    parser.add_argument(
        "--no-clear-between-runs",
        action="store_true",
        help="Do not flush Redis before the batch starts and after each prompt.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--api-keys-file",
        type=Path,
        default=REPO_ROOT / "reports" / "api_keys.txt",
        help=(
            "Path to a text file containing Tavily API keys (one per line, "
            "format: 'Label: tvly-…'). Keys are tried in order; a key is "
            "rotated out when it has <= 20 free credits remaining."
        ),
    )
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "id" not in record or "prompt" not in record:
            raise ValueError(f"Missing required keys at {path}:{line_number}")
        queries.append(record)
    return queries


def build_redis_targets(config: dict[str, Any]) -> list[RedisTarget]:
    candidates = [
        (
            "context",
            config.get("redis", {}) or {},
            {"hostname": "localhost", "port": 6379, "db": 0},
        ),
        (
            "queries",
            config.get("queries_redis", {}) or {},
            {"hostname": "localhost", "port": 6379, "db": 1},
        ),
        (
            "kg",
            config.get("kg_redis", {}) or config.get("redis", {}) or {},
            {"hostname": "localhost", "port": 6379, "db": 0},
        ),
    ]

    unique_targets: dict[tuple[str, int, int], RedisTarget] = {}
    for name, raw_cfg, defaults in candidates:
        host = str(raw_cfg.get("hostname", defaults["hostname"]))
        port = int(raw_cfg.get("port", defaults["port"]))
        db = int(raw_cfg.get("db", defaults["db"]))
        unique_targets.setdefault((host, port, db), RedisTarget(name=name, host=host, port=port, db=db))
    return list(unique_targets.values())


def flush_redis_targets(targets: Iterable[RedisTarget]) -> None:
    for target in targets:
        client = redis.Redis(
            host=target.host,
            port=target.port,
            db=target.db,
            decode_responses=True,
        )
        try:
            client.flushdb()
            logger.info(
                "Flushed %s redis at %s:%d/%d",
                target.name,
                target.host,
                target.port,
                target.db,
            )
        finally:
            client.close()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"updated_at": None, "runs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now()
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def report_path_for(reports_dir: Path, prompt_id: Any) -> Path:
    return reports_dir / f"id_{str(prompt_id)}.md"


def ensure_completed_from_existing_report(
    state: dict[str, Any],
    prompt_id: Any,
    report_path: Path,
) -> None:
    key = str(prompt_id)
    if not report_path.exists():
        return
    if report_path.stat().st_size == 0:
        return
    run_state = state.setdefault("runs", {}).setdefault(key, {})
    run_state.setdefault("status", "completed")
    run_state.setdefault("report_path", str(report_path))


def should_run_prompt(
    state: dict[str, Any],
    prompt_id: Any,
    report_path: Path,
    force: bool,
) -> bool:
    ensure_completed_from_existing_report(state, prompt_id, report_path)
    if force:
        return True
    run_state = state.get("runs", {}).get(str(prompt_id), {})
    if run_state.get("status") == "completed" and report_path.exists() and report_path.stat().st_size > 0:
        return False
    return True


def close_orchestrator(orchestrator: MingDeepResearch) -> None:
    for attr_name in ("context_redis", "queries_redis", "kg_redis"):
        handle = getattr(orchestrator, attr_name, None)
        close_fn = getattr(handle, "close", None)
        if callable(close_fn):
            close_fn()


def run_prompt(
    *,
    prompt_record: dict[str, Any],
    config_path: Path,
    report_path: Path,
) -> str:
    config_dict = load_config(config_path)
    config = create_ming_deep_research_config(config_dict)
    config.draft_output_path = str(report_path)
    orchestrator = MingDeepResearch(config)
    try:
        return orchestrator.run(str(prompt_record["prompt"]))
    finally:
        close_orchestrator(orchestrator)


def select_queries(
    queries: list[dict[str, Any]],
    prompt_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = queries
    if prompt_ids:
        wanted = {str(prompt_id) for prompt_id in prompt_ids}
        selected = [query for query in queries if str(query["id"]) in wanted]
    if limit is not None:
        selected = selected[:limit]
    return selected


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(args.state_file)
    queries = load_queries(args.query_file)
    selected_queries = select_queries(queries, args.prompt_ids, args.limit)
    redis_targets = build_redis_targets(load_config(args.config))

    if not args.no_clear_between_runs:
        flush_redis_targets(redis_targets)

    key_manager = TavilyKeyManager.from_sources(
        keys_file=args.api_keys_file,
        env_key=os.environ.get("TAVILY_API_KEY"),
    )

    logger.info("Loaded %d prompt(s) to evaluate", len(selected_queries))
    for prompt_record in selected_queries:
        prompt_id = prompt_record["id"]
        report_path = report_path_for(args.reports_dir, prompt_id)
        state.setdefault("runs", {})

        if not should_run_prompt(state, prompt_id, report_path, args.force):
            logger.info("Skipping prompt %s because a completed report already exists", prompt_id)
            save_state(args.state_file, state)
            continue

        prompt_state = state["runs"].setdefault(str(prompt_id), {})
        prompt_state.update(
            {
                "status": "running",
                "started_at": utc_now(),
                "finished_at": None,
                "report_path": str(report_path),
                "prompt": prompt_record.get("prompt"),
                "topic": prompt_record.get("topic"),
                "language": prompt_record.get("language"),
                "error": None,
            }
        )
        save_state(args.state_file, state)

        if not key_manager.ensure_headroom():
            prompt_state = state["runs"].setdefault(str(prompt_id), {})
            prompt_state.update(
                {
                    "status": "failed",
                    "started_at": utc_now(),
                    "finished_at": utc_now(),
                    "report_path": str(report_path),
                    "prompt": prompt_record.get("prompt"),
                    "topic": prompt_record.get("topic"),
                    "language": prompt_record.get("language"),
                    "error": "All Tavily API keys exhausted — free credit limit reached on every key.",
                }
            )
            save_state(args.state_file, state)
            logger.error("Stopping batch: no Tavily keys have sufficient credits.")
            return 1

        usage_info = key_manager.get_last_usage_info()
        if usage_info:
            state["tavily"] = usage_info
            save_state(args.state_file, state)

        logger.info("Running prompt %s -> %s", prompt_id, report_path)
        try:
            report_markdown = run_prompt(
                prompt_record=prompt_record,
                config_path=args.config,
                report_path=report_path,
            )
            if not report_markdown.strip():
                raise RuntimeError("Deepresearch returned an empty markdown report")
            prompt_state["status"] = "completed"
            prompt_state["finished_at"] = utc_now()
            prompt_state["report_bytes"] = report_path.stat().st_size if report_path.exists() else len(
                report_markdown.encode("utf-8")
            )
            save_state(args.state_file, state)
        except KeyboardInterrupt:
            prompt_state["status"] = "interrupted"
            prompt_state["finished_at"] = utc_now()
            prompt_state["error"] = "Interrupted by user"
            save_state(args.state_file, state)
            raise
        except Exception as exc:
            logger.exception("Prompt %s failed", prompt_id)
            prompt_state["status"] = "failed"
            prompt_state["finished_at"] = utc_now()
            prompt_state["error"] = f"{type(exc).__name__}: {exc}"
            save_state(args.state_file, state)
            if not args.continue_on_error:
                return 1
        finally:
            if not args.no_clear_between_runs:
                flush_redis_targets(redis_targets)

    completed = sum(1 for run in state.get("runs", {}).values() if run.get("status") == "completed")
    failed = sum(1 for run in state.get("runs", {}).values() if run.get("status") == "failed")
    logger.info(
        "Batch finished with %d completed and %d failed prompt(s). State saved to %s",
        completed,
        failed,
        args.state_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
