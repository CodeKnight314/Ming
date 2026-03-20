from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

from ming.runtime.contracts import (
    AngleSnapshot,
    RunSnapshot,
    RuntimeEvent,
    RuntimeEventKind,
    RuntimeStatus,
    utc_now_iso,
)
from ming.runtime.emitter import RuntimeEmitter


class RuntimeObserver:
    """Thread-safe helper for emitting run and angle telemetry."""

    def __init__(
        self,
        emitter: RuntimeEmitter,
        *,
        command_id: str,
        job_id: str,
        run_id: str,
        prompt: str,
        prompt_id: str | None = None,
        component: str = "runtime_service",
    ) -> None:
        self.emitter = emitter
        self.command_id = command_id
        self.job_id = job_id
        self.run_id = run_id
        self.prompt = prompt
        self.prompt_id = prompt_id
        self.component = component
        self._lock = threading.Lock()
        self._run_snapshot = RunSnapshot(
            run_id=run_id,
            job_id=job_id,
            command_id=command_id,
            status=RuntimeStatus.RUNNING.value,
            started_at=utc_now_iso(),
            prompt=prompt,
            prompt_id=prompt_id,
            stage=None,
            stage_status=None,
        )
        self._angle_snapshots: dict[str, AngleSnapshot] = {}
        self.emitter.write_run_snapshot(self._run_snapshot)

    def emit_event(
        self,
        *,
        kind: RuntimeEventKind | str,
        component: str,
        status: str,
        message: str,
        stage: str | None = None,
        angle_id: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        event_kind = kind if isinstance(kind, RuntimeEventKind) else RuntimeEventKind(str(kind))
        return self.emitter.emit_event(
            RuntimeEvent(
                event_id=self._new_id("evt"),
                ts=utc_now_iso(),
                seq=self._next_seq(),
                kind=event_kind,
                component=component,
                status=status,
                message=message,
                command_id=self.command_id,
                job_id=self.job_id,
                run_id=self.run_id,
                angle_id=angle_id,
                stage=stage,
                metrics=metrics or {},
            )
        )

    def update_run(
        self,
        *,
        status: str | None = None,
        stage: str | None = None,
        stage_status: str | None = None,
        finished_at: str | None = None,
        active_angle_count: int | None = None,
        completed_angle_count: int | None = None,
        error: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> RunSnapshot:
        with self._lock:
            merged_metrics = dict(self._run_snapshot.metrics)
            # Use `is not None` so we do not treat an empty dict as "skip";
            # also distinguishes "omit" (None) from explicit empty updates.
            if metrics is not None:
                merged_metrics.update(metrics)
            self._run_snapshot = replace(
                self._run_snapshot,
                status=status or self._run_snapshot.status,
                stage=stage if stage is not None else self._run_snapshot.stage,
                stage_status=stage_status if stage_status is not None else self._run_snapshot.stage_status,
                finished_at=finished_at if finished_at is not None else self._run_snapshot.finished_at,
                active_angle_count=(
                    active_angle_count
                    if active_angle_count is not None
                    else self._run_snapshot.active_angle_count
                ),
                completed_angle_count=(
                    completed_angle_count
                    if completed_angle_count is not None
                    else self._run_snapshot.completed_angle_count
                ),
                error=error if error is not None else self._run_snapshot.error,
                metrics=merged_metrics,
            )
            self.emitter.write_run_snapshot(self._run_snapshot)
            return self._run_snapshot

    def snapshot_terminated(
        self,
        *,
        status: str,
        finished_at: str,
        error: str | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> RunSnapshot:
        """Build a final run snapshot for Redis without dropping pipeline metrics."""
        with self._lock:
            merged_metrics = dict(self._run_snapshot.metrics)
            if extra_metrics is not None:
                merged_metrics.update(extra_metrics)
            return replace(
                self._run_snapshot,
                status=status,
                finished_at=finished_at,
                stage_status=status,
                error=error,
                metrics=merged_metrics,
            )

    def stage_transition(
        self,
        *,
        component: str,
        stage: str,
        status: str,
        message: str,
        metrics: dict[str, Any] | None = None,
        active_angle_count: int | None = None,
        completed_angle_count: int | None = None,
    ) -> None:
        self.update_run(
            stage=stage,
            stage_status=status,
            active_angle_count=active_angle_count,
            completed_angle_count=completed_angle_count,
            metrics=metrics,
        )
        self.emit_event(
            kind=RuntimeEventKind.STAGE_TRANSITION,
            component=component,
            status=status,
            message=message,
            stage=stage,
            metrics=metrics,
        )

    def register_angle(
        self,
        *,
        angle_id: str,
        topic: str,
        success_criteria: str,
    ) -> AngleSnapshot:
        with self._lock:
            existing = self._angle_snapshots.get(angle_id)
            if existing is not None:
                return existing
        snapshot = AngleSnapshot(
            angle_id=angle_id,
            run_id=self.run_id,
            topic=topic,
            success_criteria=success_criteria,
            status=RuntimeStatus.QUEUED.value,
            iteration=0,
        )
        with self._lock:
            self._angle_snapshots[angle_id] = snapshot
            self.emitter.write_angle_snapshot(snapshot)
            self._run_snapshot = replace(
                self._run_snapshot,
                active_angle_count=self._run_snapshot.active_angle_count + 1,
            )
            self.emitter.write_run_snapshot(self._run_snapshot)
        self.emit_event(
            kind=RuntimeEventKind.METRIC_UPDATE,
            component="orchestrator",
            status=RuntimeStatus.QUEUED.value,
            message=f"Registered angle {angle_id}.",
            angle_id=angle_id,
            metrics={"topic": topic},
        )
        return snapshot

    def update_angle(
        self,
        angle_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        iteration: int | None = None,
        queries_total: int | None = None,
        context_ids_total: int | None = None,
        statistics: dict[str, Any] | None = None,
        error: str | None = None,
        emit_event: bool = False,
        event_kind: RuntimeEventKind = RuntimeEventKind.METRIC_UPDATE,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> AngleSnapshot:
        with self._lock:
            current = self._angle_snapshots[angle_id]
            merged_statistics = dict(current.statistics)
            if statistics:
                merged_statistics.update(statistics)
            snapshot = replace(
                current,
                status=status or current.status,
                stage=stage if stage is not None else current.stage,
                iteration=iteration if iteration is not None else current.iteration,
                queries_total=queries_total if queries_total is not None else current.queries_total,
                context_ids_total=(
                    context_ids_total
                    if context_ids_total is not None
                    else current.context_ids_total
                ),
                statistics=merged_statistics,
                error=error if error is not None else current.error,
            )
            self._angle_snapshots[angle_id] = snapshot
            self.emitter.write_angle_snapshot(snapshot)
            if status == RuntimeStatus.COMPLETED.value and current.status != RuntimeStatus.COMPLETED.value:
                self._run_snapshot = replace(
                    self._run_snapshot,
                    active_angle_count=max(0, self._run_snapshot.active_angle_count - 1),
                    completed_angle_count=self._run_snapshot.completed_angle_count + 1,
                )
                self.emitter.write_run_snapshot(self._run_snapshot)
            elif status == RuntimeStatus.FAILED.value and current.status != RuntimeStatus.FAILED.value:
                self._run_snapshot = replace(
                    self._run_snapshot,
                    active_angle_count=max(0, self._run_snapshot.active_angle_count - 1),
                )
                self.emitter.write_run_snapshot(self._run_snapshot)
        if emit_event:
            event_metrics = {
                "iteration": snapshot.iteration,
                "queries_total": snapshot.queries_total,
                "context_ids_total": snapshot.context_ids_total,
            }
            if metrics:
                event_metrics.update(metrics)
            self.emit_event(
                kind=event_kind,
                component="research_subagent",
                status=snapshot.status,
                message=message or f"Angle {angle_id} updated.",
                stage=snapshot.stage,
                angle_id=angle_id,
                metrics=event_metrics,
            )
        return snapshot

    def _next_seq(self) -> int:
        raw = self.emitter.client.incr(f"{self.emitter.namespace}:seq")
        return int(raw)

    def _new_id(self, prefix: str) -> str:
        raw = self.emitter.client.incr(f"{self.emitter.namespace}:{prefix}:id")
        return f"{prefix}_{int(raw)}"
