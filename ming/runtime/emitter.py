from __future__ import annotations

import json
from typing import Any

from ming.runtime.contracts import (
    DEFAULT_RUNTIME_NAMESPACE,
    AngleSnapshot,
    CommandResult,
    CommandSnapshot,
    JobSnapshot,
    QueueSnapshot,
    RunSnapshot,
    RuntimeCommand,
    RuntimeEvent,
    angle_snapshot_key,
    command_results_stream_key,
    command_snapshot_key,
    events_stream_key,
    jobs_snapshot_key,
    queue_snapshot_key,
    runs_snapshot_key,
    to_jsonable,
    runtime_commands_stream_key,
)


class RuntimeEmitter:
    """Writes runtime commands, events, and snapshots to Redis."""

    def __init__(
        self,
        client: Any,
        *,
        namespace: str = DEFAULT_RUNTIME_NAMESPACE,
        stream_maxlen: int | None = None,
        snapshot_ttl_seconds: int | None = None,
    ) -> None:
        self.client = client
        self.namespace = namespace
        self.stream_maxlen = stream_maxlen
        self.snapshot_ttl_seconds = snapshot_ttl_seconds

    def append_command(self, command: RuntimeCommand) -> str:
        return self._xadd(runtime_commands_stream_key(self.namespace), command)

    def emit_event(self, event: RuntimeEvent) -> str:
        return self._xadd(events_stream_key(self.namespace), event)

    def emit_command_result(self, result: CommandResult) -> str:
        return self._xadd(command_results_stream_key(self.namespace), result)

    def write_command_snapshot(self, snapshot: CommandSnapshot) -> bool:
        return self._set_json(command_snapshot_key(snapshot.command_id, self.namespace), snapshot)

    def write_job_snapshot(self, snapshot: JobSnapshot) -> bool:
        return self._set_json(jobs_snapshot_key(snapshot.job_id, self.namespace), snapshot)

    def write_run_snapshot(self, snapshot: RunSnapshot) -> bool:
        return self._set_json(runs_snapshot_key(snapshot.run_id, self.namespace), snapshot)

    def write_angle_snapshot(self, snapshot: AngleSnapshot) -> bool:
        return self._set_json(
            angle_snapshot_key(snapshot.run_id, snapshot.angle_id, self.namespace),
            snapshot,
        )

    def write_queue_snapshot(self, snapshot: QueueSnapshot | dict[str, Any]) -> bool:
        return self._set_json(queue_snapshot_key(self.namespace), snapshot)

    def read_json(self, key: str) -> dict[str, Any] | list[Any] | None:
        raw = self.client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def _xadd(self, key: str, payload: Any) -> str:
        message = json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True)
        kwargs: dict[str, Any] = {}
        if self.stream_maxlen is not None:
            kwargs["maxlen"] = self.stream_maxlen
            kwargs["approximate"] = True
        return self.client.xadd(key, {"payload": message}, **kwargs)

    def _set_json(self, key: str, payload: Any) -> bool:
        message = json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True)
        if self.snapshot_ttl_seconds is not None:
            return bool(self.client.set(key, message, ex=self.snapshot_ttl_seconds))
        return bool(self.client.set(key, message))
