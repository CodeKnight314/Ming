from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Mapping, Protocol
from uuid import uuid4

import redis
from dotenv import load_dotenv

from ming.core.redis import RedisDatabaseConfig
from ming.runtime.contracts import (
    DEFAULT_RUNTIME_NAMESPACE,
    CommandResult,
    CommandSnapshot,
    JobSnapshot,
    JobType,
    QueueSnapshot,
    RunSnapshot,
    RuntimeCommand,
    RuntimeEvent,
    RuntimeEventKind,
    RuntimeStatus,
    RuntimeValidationError,
    parse_runtime_command,
    runtime_commands_stream_key,
    utc_now_iso,
)
from ming.runtime.emitter import RuntimeEmitter
from ming.runtime.observer import RuntimeObserver

logger = logging.getLogger(__name__)


class JobExecutor(Protocol):
    def __call__(self, job: "RuntimeJob") -> dict[str, Any]:
        ...


@dataclass
class RuntimeServiceConfig:
    config_path: str | Path = "config.json"
    redis: RedisDatabaseConfig = field(
        default_factory=lambda: RedisDatabaseConfig(hostname="localhost", port=6379, db=0)
    )
    namespace: str = "runtime"
    stream_maxlen: int | None = 10_000
    snapshot_ttl_seconds: int | None = None
    max_queue_depth: int = 100
    command_block_ms: int = 1000
    command_count: int = 10
    start_from_latest: bool = True


@dataclass
class RuntimeJob:
    job_id: str
    command_id: str
    type: JobType
    payload: dict[str, Any]
    created_at: str
    position: int | None = None
    parent_job_id: str | None = None
    status: RuntimeStatus = RuntimeStatus.QUEUED
    started_at: str | None = None
    finished_at: str | None = None
    run_id: str | None = None
    error: str | None = None
    runtime_observer: RuntimeObserver | None = None

    def to_snapshot(self) -> JobSnapshot:
        return JobSnapshot(
            job_id=self.job_id,
            command_id=self.command_id,
            type=self.type.value,
            status=self.status.value,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            position=self.position,
            run_id=self.run_id,
            error=self.error,
            payload=self.payload,
        )


def close_orchestrator(orchestrator: Any) -> None:
    for attr_name in ("context_redis", "queries_redis", "kg_redis"):
        handle = getattr(orchestrator, attr_name, None)
        close_fn = getattr(handle, "close", None)
        if callable(close_fn):
            close_fn()


def default_job_executor_factory(
    config_path: str | Path,
    *,
    runtime_namespace: str = DEFAULT_RUNTIME_NAMESPACE,
) -> JobExecutor:
    def _executor(job: RuntimeJob) -> dict[str, Any]:
        from ming.core.config import create_ming_deep_research_config, load_config
        from ming.core.redis_flush import flush_research_redis_for_new_run
        from ming.orchestrator import MingDeepResearch

        prompt = str(job.payload.get("prompt", "")).strip()
        if not prompt:
            raise RuntimeValidationError("Job payload.prompt must be non-empty")

        config_dict = load_config(config_path)
        flush_research_redis_for_new_run(
            config_dict, runtime_namespace=runtime_namespace
        )
        config = create_ming_deep_research_config(config_dict)

        prompt_id = job.payload.get("prompt_id")
        if prompt_id is not None:
            draft_name = f"runtime_{job.job_id}_{prompt_id}.md"
        else:
            draft_name = f"runtime_{job.job_id}.md"
        config.draft_output_path = str(Path("reports") / draft_name)

        orchestrator = MingDeepResearch(config)
        observer = job.runtime_observer
        if observer is not None:
            orchestrator.runtime_observer = observer
        try:
            report_markdown = orchestrator.run(prompt)
        finally:
            close_orchestrator(orchestrator)

        return {
            "report_markdown": report_markdown,
            "report_length": len(report_markdown or ""),
        }

    return _executor


class RuntimeService:
    def __init__(
        self,
        config: RuntimeServiceConfig,
        *,
        redis_client: redis.Redis | None = None,
        executor: JobExecutor | None = None,
        id_factory: Callable[[str], str] | None = None,
        now_fn: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.config = config
        self.client = redis_client or redis.Redis(
            host=config.redis.hostname,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=True,
        )
        self.emitter = RuntimeEmitter(
            self.client,
            namespace=config.namespace,
            stream_maxlen=config.stream_maxlen,
            snapshot_ttl_seconds=config.snapshot_ttl_seconds,
        )
        self.executor = executor or default_job_executor_factory(
            config.config_path,
            runtime_namespace=config.namespace,
        )
        self.id_factory = id_factory or (lambda prefix: f"{prefix}_{uuid4().hex}")
        self.now_fn = now_fn

        self._queue: Deque[str] = deque()
        self._jobs: dict[str, RuntimeJob] = {}
        self._command_to_jobs: dict[str, list[str]] = {}
        self._command_snapshots: dict[str, CommandSnapshot] = {}
        self._active_job_id: str | None = None
        self._last_command_stream_id = "$" if config.start_from_latest else "0-0"

        self._write_queue_snapshot()

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()

    def run_forever(self) -> None:
        try:
            while True:
                self.poll_once()
                self.run_next_job()
        finally:
            self.close()

    def poll_once(self) -> int:
        entries = self.client.xread(
            {runtime_commands_stream_key(self.config.namespace): self._last_command_stream_id},
            count=max(1, self.config.command_count),
            block=max(0, self.config.command_block_ms),
        )
        processed = 0
        for _, stream_entries in entries or []:
            for entry_id, fields in stream_entries:
                self._last_command_stream_id = entry_id
                self._handle_stream_entry(entry_id, fields)
                processed += 1
        return processed

    def run_next_job(self) -> bool:
        if self._active_job_id is not None or not self._queue:
            return False

        job_id = self._queue.popleft()
        job = self._jobs[job_id]
        self._active_job_id = job_id
        job.status = RuntimeStatus.RUNNING
        job.started_at = self.now_fn()
        run_id = self.id_factory("run")
        job.run_id = run_id
        observer = RuntimeObserver(
            self.emitter,
            command_id=job.command_id,
            job_id=job.job_id,
            run_id=run_id,
            prompt=str(job.payload.get("prompt", "")),
            prompt_id=str(job.payload.get("prompt_id")) if job.payload.get("prompt_id") is not None else None,
        )
        job.runtime_observer = observer
        self.emitter.write_job_snapshot(job.to_snapshot())
        self._write_queue_snapshot()
        self._emit_job_event(
            job,
            kind=RuntimeEventKind.JOB_STARTED,
            status=RuntimeStatus.RUNNING.value,
            message="Job started.",
        )

        run_snapshot = RunSnapshot(
            run_id=run_id,
            job_id=job.job_id,
            command_id=job.command_id,
            status=RuntimeStatus.RUNNING.value,
            started_at=job.started_at,
            prompt=str(job.payload.get("prompt", "")),
            prompt_id=str(job.payload.get("prompt_id")) if job.payload.get("prompt_id") is not None else None,
            stage="runtime_execute",
            stage_status=RuntimeStatus.RUNNING.value,
        )
        self.emitter.write_run_snapshot(run_snapshot)
        observer.update_run(
            status=RuntimeStatus.RUNNING.value,
            stage="runtime_execute",
            stage_status=RuntimeStatus.RUNNING.value,
        )
        self.emitter.emit_event(
            RuntimeEvent(
                event_id=self.id_factory("evt"),
                ts=self.now_fn(),
                seq=self._next_seq(),
                kind=RuntimeEventKind.RUN_STARTED,
                component="runtime_service",
                status=RuntimeStatus.RUNNING.value,
                message="Run started.",
                command_id=job.command_id,
                job_id=job.job_id,
                run_id=run_id,
                stage="runtime_execute",
            )
        )
        self._update_command_snapshot(
            job.command_id,
            status=RuntimeStatus.RUNNING.value,
            detail=f"Job {job.job_id} is running.",
        )
        self.emitter.emit_command_result(
            CommandResult(
                command_id=job.command_id,
                status=RuntimeStatus.RUNNING,
                ts=self.now_fn(),
                detail=f"Job {job.job_id} is running.",
                job_id=job.job_id,
            )
        )

        try:
            result = self.executor(job) or {}
            report_markdown = str(result.get("report_markdown", "") or "")
            if not report_markdown.strip():
                raise RuntimeError("Executor returned an empty markdown report")

            job.status = RuntimeStatus.COMPLETED
            job.finished_at = self.now_fn()
            self.emitter.write_job_snapshot(job.to_snapshot())
            self._emit_job_event(
                job,
                kind=RuntimeEventKind.JOB_COMPLETED,
                status=RuntimeStatus.COMPLETED.value,
                message="Job completed successfully.",
                metrics={"report_length": len(report_markdown)},
            )
            self.emitter.write_run_snapshot(
                observer.snapshot_terminated(
                    status=RuntimeStatus.COMPLETED.value,
                    finished_at=job.finished_at or self.now_fn(),
                    error=None,
                    extra_metrics={"report_length": len(report_markdown)},
                )
            )
            self.emitter.emit_event(
                RuntimeEvent(
                    event_id=self.id_factory("evt"),
                    ts=self.now_fn(),
                    seq=self._next_seq(),
                    kind=RuntimeEventKind.RUN_COMPLETED,
                    component="runtime_service",
                    status=RuntimeStatus.COMPLETED.value,
                    message="Run completed successfully.",
                    command_id=job.command_id,
                    job_id=job.job_id,
                    run_id=run_id,
                    stage="runtime_execute",
                    metrics={"report_length": len(report_markdown)},
                )
            )
        except Exception as exc:
            logger.exception("Runtime job %s failed", job.job_id)
            job.status = RuntimeStatus.FAILED
            job.finished_at = self.now_fn()
            job.error = f"{type(exc).__name__}: {exc}"
            self.emitter.write_job_snapshot(job.to_snapshot())
            self._emit_job_event(
                job,
                kind=RuntimeEventKind.JOB_FAILED,
                status=RuntimeStatus.FAILED.value,
                message=f"Job failed: {job.error}",
            )
            self.emitter.write_run_snapshot(
                observer.snapshot_terminated(
                    status=RuntimeStatus.FAILED.value,
                    finished_at=job.finished_at or self.now_fn(),
                    error=job.error,
                    extra_metrics=None,
                )
            )
            self.emitter.emit_event(
                RuntimeEvent(
                    event_id=self.id_factory("evt"),
                    ts=self.now_fn(),
                    seq=self._next_seq(),
                    kind=RuntimeEventKind.RUN_FAILED,
                    component="runtime_service",
                    status=RuntimeStatus.FAILED.value,
                    message=f"Run failed: {job.error}",
                    command_id=job.command_id,
                    job_id=job.job_id,
                    run_id=run_id,
                    stage="runtime_execute",
                )
            )
        finally:
            self._active_job_id = None
            self._refresh_command_terminal_state(job.command_id)
            self._write_queue_snapshot()
        return True

    def _handle_stream_entry(self, entry_id: str, fields: Mapping[str, Any]) -> None:
        raw_payload = fields.get("payload")
        if raw_payload is None:
            return
        try:
            data = json.loads(raw_payload)
            command = parse_runtime_command(data)
        except (json.JSONDecodeError, RuntimeValidationError) as exc:
            detail = f"Rejected command {entry_id}: {exc}"
            self.emitter.emit_event(
                RuntimeEvent(
                    event_id=self.id_factory("evt"),
                    ts=self.now_fn(),
                    seq=self._next_seq(),
                    kind=RuntimeEventKind.COMMAND_REJECTED,
                    component="runtime_service",
                    status=RuntimeStatus.REJECTED.value,
                    message=detail,
                )
            )
            return

        self._accept_command(command)

    def _accept_command(self, command: RuntimeCommand) -> None:
        queued_depth = len(self._queue) + (1 if self._active_job_id else 0)
        planned_job_count = 1
        if command.type.value == "run_batch":
            planned_job_count = len(command.payload.items) + 1
        if queued_depth + planned_job_count > self.config.max_queue_depth:
            detail = (
                f"Queue full: current depth {queued_depth}, "
                f"command would add {planned_job_count} job(s), max {self.config.max_queue_depth}."
            )
            self._write_rejected_command(command, detail)
            return

        snapshot = CommandSnapshot(
            command_id=command.command_id,
            type=command.type.value,
            status=RuntimeStatus.ACCEPTED.value,
            submitted_at=command.submitted_at,
            updated_at=self.now_fn(),
            source=command.source.to_dict(),
            detail="Command accepted and queued.",
            job_ids=[],
        )
        self._command_snapshots[command.command_id] = snapshot
        self.emitter.write_command_snapshot(snapshot)
        self.emitter.emit_event(
            RuntimeEvent(
                event_id=self.id_factory("evt"),
                ts=self.now_fn(),
                seq=self._next_seq(),
                kind=RuntimeEventKind.COMMAND_ACCEPTED,
                component="runtime_service",
                status=RuntimeStatus.ACCEPTED.value,
                message="Command accepted and queued.",
                command_id=command.command_id,
            )
        )
        self.emitter.emit_command_result(
            CommandResult(
                command_id=command.command_id,
                status=RuntimeStatus.ACCEPTED,
                ts=self.now_fn(),
                detail="Command accepted and queued.",
            )
        )

        if command.type.value == "run_query":
            self._enqueue_job(
                command_id=command.command_id,
                job_type=JobType.SINGLE_RUN,
                payload=command.payload.to_dict(),
                parent_job_id=None,
                position=None,
            )
            return

        parent_job_id = self._enqueue_job(
            command_id=command.command_id,
            job_type=JobType.BATCH_PARENT,
            payload={
                "mode": command.payload.mode,
                "item_count": len(command.payload.items),
            },
            parent_job_id=None,
            position=None,
            queue_job=False,
        )

        for index, item in enumerate(command.payload.items, start=1):
            self._enqueue_job(
                command_id=command.command_id,
                job_type=JobType.BATCH_CHILD_RUN,
                payload={"prompt_id": item.id, "prompt": item.prompt},
                parent_job_id=parent_job_id,
                position=index,
            )

        self._update_command_snapshot(
            command.command_id,
            status=RuntimeStatus.EXPANDED.value,
            detail=f"Batch expanded into {len(command.payload.items)} child job(s).",
        )
        self.emitter.emit_event(
            RuntimeEvent(
                event_id=self.id_factory("evt"),
                ts=self.now_fn(),
                seq=self._next_seq(),
                kind=RuntimeEventKind.COMMAND_EXPANDED,
                component="runtime_service",
                status=RuntimeStatus.EXPANDED.value,
                message=f"Batch expanded into {len(command.payload.items)} child job(s).",
                command_id=command.command_id,
                job_id=parent_job_id,
            )
        )
        self.emitter.emit_command_result(
            CommandResult(
                command_id=command.command_id,
                status=RuntimeStatus.EXPANDED,
                ts=self.now_fn(),
                detail=f"Batch expanded into {len(command.payload.items)} child job(s).",
                job_id=parent_job_id,
            )
        )

    def _write_rejected_command(self, command: RuntimeCommand, detail: str) -> None:
        snapshot = CommandSnapshot(
            command_id=command.command_id,
            type=command.type.value,
            status=RuntimeStatus.REJECTED.value,
            submitted_at=command.submitted_at,
            updated_at=self.now_fn(),
            source=command.source.to_dict(),
            detail=detail,
            error=detail,
        )
        self._command_snapshots[command.command_id] = snapshot
        self.emitter.write_command_snapshot(snapshot)
        self.emitter.emit_event(
            RuntimeEvent(
                event_id=self.id_factory("evt"),
                ts=self.now_fn(),
                seq=self._next_seq(),
                kind=RuntimeEventKind.COMMAND_REJECTED,
                component="runtime_service",
                status=RuntimeStatus.REJECTED.value,
                message=detail,
                command_id=command.command_id,
            )
        )
        self.emitter.emit_command_result(
            CommandResult(
                command_id=command.command_id,
                status=RuntimeStatus.REJECTED,
                ts=self.now_fn(),
                detail=detail,
            )
        )

    def _enqueue_job(
        self,
        *,
        command_id: str,
        job_type: JobType,
        payload: dict[str, Any],
        parent_job_id: str | None,
        position: int | None,
        queue_job: bool = True,
    ) -> str:
        job_id = self.id_factory("job")
        job = RuntimeJob(
            job_id=job_id,
            command_id=command_id,
            type=job_type,
            payload=payload,
            created_at=self.now_fn(),
            position=position,
            parent_job_id=parent_job_id,
        )
        self._jobs[job_id] = job
        self._command_to_jobs.setdefault(command_id, []).append(job_id)
        self.emitter.write_job_snapshot(job.to_snapshot())
        self._emit_job_event(
            job,
            kind=RuntimeEventKind.JOB_QUEUED,
            status=RuntimeStatus.QUEUED.value,
            message="Job queued.",
        )
        if queue_job:
            self._queue.append(job_id)
            self._write_queue_snapshot()
        self._update_command_snapshot(command_id, job_ids=self._command_to_jobs[command_id])
        return job_id

    def _refresh_command_terminal_state(self, command_id: str) -> None:
        job_ids = self._command_to_jobs.get(command_id, [])
        if not job_ids:
            return

        executable_jobs = [
            self._jobs[job_id]
            for job_id in job_ids
            if self._jobs[job_id].type is not JobType.BATCH_PARENT
        ]
        if not executable_jobs:
            return

        if any(job.status is RuntimeStatus.RUNNING for job in executable_jobs):
            return
        if any(job.status is RuntimeStatus.QUEUED for job in executable_jobs):
            return

        if any(job.status is RuntimeStatus.FAILED for job in executable_jobs):
            detail = "One or more jobs under this command failed."
            self._mark_parent_jobs(command_id, RuntimeStatus.FAILED, detail)
            self._update_command_snapshot(
                command_id,
                status=RuntimeStatus.FAILED.value,
                detail=detail,
                error=detail,
            )
            self.emitter.emit_command_result(
                CommandResult(
                    command_id=command_id,
                    status=RuntimeStatus.FAILED,
                    ts=self.now_fn(),
                    detail=detail,
                )
            )
            return

        self._mark_parent_jobs(
            command_id,
            RuntimeStatus.COMPLETED,
            "Batch parent completed after all child jobs completed.",
        )
        self._update_command_snapshot(
            command_id,
            status=RuntimeStatus.COMPLETED.value,
            detail="All jobs under this command completed successfully.",
        )
        self.emitter.emit_command_result(
            CommandResult(
                command_id=command_id,
                status=RuntimeStatus.COMPLETED,
                ts=self.now_fn(),
                detail="All jobs under this command completed successfully.",
            )
        )

    def _mark_parent_jobs(
        self,
        command_id: str,
        status: RuntimeStatus,
        detail: str,
    ) -> None:
        for job_id in self._command_to_jobs.get(command_id, []):
            job = self._jobs[job_id]
            if job.type is not JobType.BATCH_PARENT:
                continue
            if job.status in (RuntimeStatus.COMPLETED, RuntimeStatus.FAILED):
                continue
            job.status = status
            job.finished_at = self.now_fn()
            if status is RuntimeStatus.FAILED:
                job.error = detail
            self.emitter.write_job_snapshot(job.to_snapshot())

    def _update_command_snapshot(
        self,
        command_id: str,
        *,
        status: str | None = None,
        detail: str | None = None,
        error: str | None = None,
        job_ids: list[str] | None = None,
    ) -> None:
        current = self._command_snapshots[command_id]
        updated = CommandSnapshot(
            command_id=current.command_id,
            type=current.type,
            status=status or current.status,
            submitted_at=current.submitted_at,
            updated_at=self.now_fn(),
            source=current.source,
            detail=detail if detail is not None else current.detail,
            job_ids=job_ids if job_ids is not None else list(current.job_ids),
            error=error if error is not None else current.error,
        )
        self._command_snapshots[command_id] = updated
        self.emitter.write_command_snapshot(updated)

    def _emit_job_event(
        self,
        job: RuntimeJob,
        *,
        kind: RuntimeEventKind,
        status: str,
        message: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.emitter.emit_event(
            RuntimeEvent(
                event_id=self.id_factory("evt"),
                ts=self.now_fn(),
                seq=self._next_seq(),
                kind=kind,
                component="runtime_service",
                status=status,
                message=message,
                command_id=job.command_id,
                job_id=job.job_id,
                run_id=job.run_id,
                metrics=metrics or {},
            )
        )

    def _write_queue_snapshot(self) -> None:
        completed_job_ids = [
            job_id for job_id, job in self._jobs.items() if job.status is RuntimeStatus.COMPLETED
        ]
        failed_job_ids = [
            job_id for job_id, job in self._jobs.items() if job.status is RuntimeStatus.FAILED
        ]
        snapshot = QueueSnapshot(
            updated_at=self.now_fn(),
            queued_job_ids=list(self._queue),
            active_job_id=self._active_job_id,
            completed_job_ids=completed_job_ids,
            failed_job_ids=failed_job_ids,
        )
        self.emitter.write_queue_snapshot(snapshot)

    def _next_seq(self) -> int:
        raw = self.client.incr(f"{self.config.namespace}:seq")
        return int(raw)


def main() -> int:
    # Prefer repo-local .env over any pre-exported shell variables.
    load_dotenv(override=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = RuntimeService(RuntimeServiceConfig())
    service.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
