from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


DEFAULT_RUNTIME_NAMESPACE = "runtime"


class RuntimeValidationError(ValueError):
    """Raised when a runtime contract payload is invalid."""


class RuntimeCommandType(str, Enum):
    RUN_QUERY = "run_query"
    RUN_BATCH = "run_batch"


class RuntimeEventKind(str, Enum):
    COMMAND_RECEIVED = "command_received"
    COMMAND_REJECTED = "command_rejected"
    COMMAND_ACCEPTED = "command_accepted"
    COMMAND_EXPANDED = "command_expanded"
    JOB_QUEUED = "job_queued"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    STAGE_TRANSITION = "stage_transition"
    METRIC_UPDATE = "metric_update"
    WARNING = "warning"
    ERROR = "error"


class RuntimeStatus(str, Enum):
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPANDED = "expanded"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    SINGLE_RUN = "single_run"
    BATCH_PARENT = "batch_parent"
    BATCH_CHILD_RUN = "batch_child_run"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_commands_stream_key(namespace: str = DEFAULT_RUNTIME_NAMESPACE) -> str:
    return f"{namespace}:commands"


def events_stream_key(namespace: str = DEFAULT_RUNTIME_NAMESPACE) -> str:
    return f"{namespace}:events"


def command_results_stream_key(namespace: str = DEFAULT_RUNTIME_NAMESPACE) -> str:
    return f"{namespace}:command_results"


def queue_snapshot_key(namespace: str = DEFAULT_RUNTIME_NAMESPACE) -> str:
    return f"{namespace}:queue"


def command_snapshot_key(command_id: str, namespace: str = DEFAULT_RUNTIME_NAMESPACE) -> str:
    return f"{namespace}:commands:{_require_non_empty(command_id, 'command_id')}"


def jobs_snapshot_key(job_id: str, namespace: str = DEFAULT_RUNTIME_NAMESPACE) -> str:
    return f"{namespace}:jobs:{_require_non_empty(job_id, 'job_id')}"


def runs_snapshot_key(run_id: str, namespace: str = DEFAULT_RUNTIME_NAMESPACE) -> str:
    return f"{namespace}:runs:{_require_non_empty(run_id, 'run_id')}"


def angle_snapshot_key(
    run_id: str,
    angle_id: str,
    namespace: str = DEFAULT_RUNTIME_NAMESPACE,
) -> str:
    return (
        f"{namespace}:runs:{_require_non_empty(run_id, 'run_id')}:"
        f"angles:{_require_non_empty(angle_id, 'angle_id')}"
    )


def _require_non_empty(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise RuntimeValidationError(f"{field_name} must be non-empty")
    return normalized


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeValidationError(f"{field_name} must be an object")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _coerce_str_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeValidationError(f"{field_name} must be an object")
    return {str(k): v for k, v in value.items()}


@dataclass(frozen=True)
class CommandSource:
    kind: str
    client_id: str
    user: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CommandSource:
        source = _require_mapping(raw, "source")
        return cls(
            kind=_require_non_empty(str(source.get("kind", "")), "source.kind"),
            client_id=_require_non_empty(
                str(source.get("client_id", "")),
                "source.client_id",
            ),
            user=_optional_str(source.get("user")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "client_id": self.client_id,
            "user": self.user,
        }


@dataclass(frozen=True)
class RunQueryPayload:
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunQueryPayload:
        payload = _require_mapping(raw, "payload")
        return cls(
            prompt=_require_non_empty(str(payload.get("prompt", "")), "payload.prompt"),
            metadata=_coerce_str_dict(payload.get("metadata"), "payload.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BatchItem:
    id: str
    prompt: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], index: int) -> BatchItem:
        item = _require_mapping(raw, f"payload.items[{index}]")
        return cls(
            id=_require_non_empty(str(item.get("id", "")), f"payload.items[{index}].id"),
            prompt=_require_non_empty(
                str(item.get("prompt", "")),
                f"payload.items[{index}].prompt",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class RunBatchPayload:
    mode: str
    items: list[BatchItem]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunBatchPayload:
        payload = _require_mapping(raw, "payload")
        mode = _require_non_empty(str(payload.get("mode", "")), "payload.mode")
        if mode != "sequential":
            raise RuntimeValidationError("payload.mode must currently be 'sequential'")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise RuntimeValidationError("payload.items must be a non-empty list")
        items = [BatchItem.from_dict(item, index=i) for i, item in enumerate(raw_items)]
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise RuntimeValidationError("payload.items contains duplicate ids")
        return cls(mode=mode, items=items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class RuntimeCommand:
    command_id: str
    type: RuntimeCommandType
    submitted_at: str
    source: CommandSource
    payload: RunQueryPayload | RunBatchPayload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RuntimeCommand:
        data = _require_mapping(raw, "command")
        command_type_raw = _require_non_empty(str(data.get("type", "")), "type")
        try:
            command_type = RuntimeCommandType(command_type_raw)
        except ValueError as exc:
            raise RuntimeValidationError(f"Unsupported command type: {command_type_raw}") from exc

        payload_raw = _require_mapping(data.get("payload"), "payload")
        if command_type is RuntimeCommandType.RUN_QUERY:
            payload = RunQueryPayload.from_dict(payload_raw)
        else:
            payload = RunBatchPayload.from_dict(payload_raw)

        submitted_at = _optional_str(data.get("submitted_at")) or utc_now_iso()
        return cls(
            command_id=_require_non_empty(str(data.get("command_id", "")), "command_id"),
            type=command_type,
            submitted_at=submitted_at,
            source=CommandSource.from_dict(_require_mapping(data.get("source"), "source")),
            payload=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "type": self.type.value,
            "submitted_at": self.submitted_at,
            "source": self.source.to_dict(),
            "payload": self.payload.to_dict(),
        }


def parse_runtime_command(raw: Mapping[str, Any]) -> RuntimeCommand:
    return RuntimeCommand.from_dict(raw)


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    status: RuntimeStatus
    ts: str
    detail: str | None = None
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "status": self.status.value,
            "ts": self.ts,
            "detail": self.detail,
            "job_id": self.job_id,
        }


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    ts: str
    seq: int
    kind: RuntimeEventKind
    component: str
    status: str
    message: str
    command_id: str | None = None
    job_id: str | None = None
    run_id: str | None = None
    angle_id: str | None = None
    stage: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ts": self.ts,
            "seq": self.seq,
            "kind": self.kind.value,
            "component": self.component,
            "status": self.status,
            "message": self.message,
            "command_id": self.command_id,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "angle_id": self.angle_id,
            "stage": self.stage,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class CommandSnapshot:
    command_id: str
    type: str
    status: str
    submitted_at: str
    updated_at: str
    source: dict[str, Any]
    detail: str | None = None
    job_ids: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    command_id: str
    type: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    position: int | None = None
    run_id: str | None = None
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    job_id: str
    command_id: str
    status: str
    started_at: str
    prompt: str
    stage: str | None = None
    stage_status: str | None = None
    prompt_id: str | None = None
    finished_at: str | None = None
    active_angle_count: int = 0
    completed_angle_count: int = 0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AngleSnapshot:
    angle_id: str
    run_id: str
    topic: str
    status: str
    iteration: int
    stage: str | None = None
    success_criteria: str | None = None
    queries_total: int = 0
    context_ids_total: int = 0
    statistics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueueSnapshot:
    updated_at: str
    queued_job_ids: list[str] = field(default_factory=list)
    active_job_id: str | None = None
    completed_job_ids: list[str] = field(default_factory=list)
    failed_job_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            key: to_jsonable(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value
