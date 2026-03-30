use std::fs;
use std::path::{Path, PathBuf};
use std::collections::{BTreeMap, HashSet};
use anyhow::{Context, Result, anyhow};
use redis::{Commands, cmd, FromRedisValue, streams::StreamRangeReply};
use serde_json::{Value as JsonValue, json};

use crate::models::*;
use crate::utils::*;

#[derive(Clone)]
pub enum DataSource {
    Mock {
        path: PathBuf,
    },
    Live {
        client: redis::Client,
        namespace: String,
    },
}

impl DataSource {
    pub fn load(
        &self,
        preferred_run_id: Option<&str>,
    ) -> Result<(DashboardData, Option<String>, Option<String>)> {
        match self {
            Self::Mock { path } => {
                let raw = fs::read_to_string(path).with_context(|| {
                    format!("failed to read mock state file {}", path.display())
                })?;
                let parsed: MockFileState = serde_json::from_str(&raw).with_context(|| {
                    format!("failed to parse mock state file {}", path.display())
                })?;
                Ok((
                    DashboardData {
                        queue: parsed.queue,
                        active_run: parsed.active_run,
                        recent_events: parsed.recent_events,
                    },
                    Some(parsed.direct_query_draft),
                    Some(parsed.batch_json_draft),
                ))
            }
            Self::Live { client, namespace } => {
                let mut con = client
                    .get_connection()
                    .context("failed to connect to Redis for live TUI state")?;
                Ok((
                    load_live_dashboard(&mut con, namespace, preferred_run_id)?,
                    None,
                    None,
                ))
            }
        }
    }

    /// Submit a single research query. `batch_item_id`, when provided, is
    /// forwarded in metadata so the backend can name the output file
    /// `id_{batch_item_id}.md` instead of a generic job-id-based name.
    pub fn submit_run_query(&self, prompt: &str, batch_item_id: Option<&str>) -> Result<String> {
        let prompt = prompt.trim();
        if prompt.is_empty() {
            return Err(anyhow!("prompt is empty"));
        }
        match self {
            Self::Mock { .. } => Ok(format!(
                "Mock: would submit run_query ({} chars).",
                prompt.chars().count()
            )),
            Self::Live { client, namespace } => {
                let mut con = client
                    .get_connection()
                    .context("failed to connect to Redis for command submission")?;
                let command_id = format!("cmd_rs_{}", unix_millis()?);
                let metadata = match batch_item_id {
                    Some(id) => json!({ "prompt_id": id }),
                    None => json!({}),
                };
                let payload = json!({
                    "command_id": command_id,
                    "type": "run_query",
                    "submitted_at": iso_like_now()?,
                    "source": {
                        "kind": "tui",
                        "client_id": "runtime-tui",
                        "user": "local"
                    },
                    "payload": {
                        "prompt": prompt,
                        "metadata": metadata
                    }
                });
                xadd_command(&mut con, namespace, &payload, &command_id)
            }
        }
    }

    #[allow(dead_code)]
    pub fn submit_run_batch(&self, items: &[(String, String)]) -> Result<String> {
        if items.is_empty() {
            return Err(anyhow!("batch has no items"));
        }
        match self {
            Self::Mock { .. } => Ok(format!(
                "Mock: would submit run_batch ({} items).",
                items.len()
            )),
            Self::Live { client, namespace } => {
                let mut con = client
                    .get_connection()
                    .context("failed to connect to Redis for command submission")?;
                let command_id = format!("cmd_rs_{}", unix_millis()?);
                let json_items: Vec<JsonValue> = items
                    .iter()
                    .map(|(id, p)| json!({ "id": id, "prompt": p }))
                    .collect();
                let payload = json!({
                    "command_id": command_id,
                    "type": "run_batch",
                    "submitted_at": iso_like_now()?,
                    "source": {
                        "kind": "tui",
                        "client_id": "runtime-tui",
                        "user": "local"
                    },
                    "payload": {
                        "mode": "sequential",
                        "items": json_items
                    }
                });
                xadd_command(&mut con, namespace, &payload, &command_id)
            }
        }
    }

    /// Flush transient Redis state (runs, jobs, events) between batch items so
    /// each new job starts from a clean slate. The commands stream is preserved.
    pub fn flush_namespace_state(&self) -> Result<()> {
        match self {
            Self::Mock { .. } => Ok(()), // no-op in mock mode
            Self::Live { client, namespace } => {
                let mut con = client
                    .get_connection()
                    .context("failed to connect to Redis for flush")?;
                flush_namespace_keys(&mut con, namespace)
            }
        }
    }

    pub fn source_label(&self) -> String {
        match self {
            Self::Mock { path } => format!("mock:{}", path.display()),
            Self::Live { namespace, .. } => format!("live:{namespace}"),
        }
    }
}

pub fn xadd_command(
    con: &mut redis::Connection,
    namespace: &str,
    payload: &JsonValue,
    command_id: &str,
) -> Result<String> {
    let stream_key = format!("{namespace}:commands");
    let payload_string = serde_json::to_string(payload)?;
    let stream_id: String = cmd("XADD")
        .arg(&stream_key)
        .arg("*")
        .arg("payload")
        .arg(payload_string)
        .query(con)
        .context("failed to append command to Redis stream")?;
    Ok(format!(
        "Submitted {command_id} to {stream_key} ({stream_id})"
    ))
}

/// Delete all run, job, and event data under `namespace` so the next batch
/// item sees a clean TUI state. The queue and commands stream are left intact.
pub fn flush_namespace_keys(con: &mut redis::Connection, namespace: &str) -> Result<()> {
    for pattern in &[
        format!("{namespace}:runs:*"),
        format!("{namespace}:jobs:*"),
    ] {
        let keys: Vec<String> = con
            .scan_match::<_, String>(pattern.as_str())
            .with_context(|| format!("scan {pattern}"))?
            .collect();
        if !keys.is_empty() {
            let _: () = con.del(&keys).with_context(|| format!("del {pattern}"))?;
        }
    }
    // Trim the events stream to zero so stale events don't bleed through.
    let _: Result<(), _> = cmd("XTRIM")
        .arg(format!("{namespace}:events"))
        .arg("MAXLEN")
        .arg(0)
        .query(con);
    Ok(())
}

pub fn json_batch_id(value: &JsonValue) -> Result<String> {
    match value {
        JsonValue::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                return Err(anyhow!("batch id must be non-empty"));
            }
            Ok(t.to_string())
        }
        JsonValue::Number(n) => Ok(n.to_string()),
        _ => Err(anyhow!("batch id must be a string or number")),
    }
}

pub fn push_batch_object(obj: &serde_json::Map<String, JsonValue>, index: &str) -> Result<(String, String)> {
    let id = match obj.get("id") {
        Some(v) => json_batch_id(v)?,
        None => return Err(anyhow!("batch item {index}: missing id")),
    };
    let prompt = obj
        .get("prompt")
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("batch item {index}: prompt required"))?
        .to_string();
    Ok((id, prompt))
}

pub fn ensure_unique_batch_ids(items: &[(String, String)]) -> Result<()> {
    let mut seen = HashSet::new();
    for (id, _) in items {
        if !seen.insert(id.as_str()) {
            return Err(anyhow!("duplicate batch id: {id}"));
        }
    }
    Ok(())
}

/// JSON array of `{id, prompt}` or JSONL with one object per line (e.g. deepresearch-bench `query.jsonl`).
pub fn load_batch_items_from_path(path: &Path) -> Result<Vec<(String, String)>> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let trimmed_start = raw.trim_start();
    let items = if trimmed_start.starts_with('[') {
        let arr: Vec<JsonValue> =
            serde_json::from_str(&raw).context("parse batch file as JSON array")?;
        let mut out = Vec::new();
        for (i, v) in arr.iter().enumerate() {
            let obj = v
                .as_object()
                .ok_or_else(|| anyhow!("batch[{i}] must be an object"))?;
            out.push(push_batch_object(obj, &format!("[{i}]"))?);
        }
        out
    } else {
        let mut out = Vec::new();
        for (i, line) in raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: JsonValue =
                serde_json::from_str(line).with_context(|| format!("JSONL line {}", i + 1))?;
            let obj = v
                .as_object()
                .ok_or_else(|| anyhow!("JSONL line {}: object expected", i + 1))?;
            out.push(push_batch_object(obj, &format!("line {}", i + 1))?);
        }
        out
    };
    if items.is_empty() {
        return Err(anyhow!("no batch items found in {}", path.display()));
    }
    ensure_unique_batch_ids(&items)?;
    Ok(items)
}

pub fn load_live_dashboard(
    con: &mut redis::Connection,
    namespace: &str,
    preferred_run_id: Option<&str>,
) -> Result<DashboardData> {
    let queue_key = format!("{namespace}:queue");
    let queue_wire = get_json::<QueueSnapshotWire>(con, &queue_key)?.unwrap_or_default();

    let active_job = if let Some(job_id) = queue_wire.active_job_id.as_deref() {
        get_job_entry(con, namespace, job_id)?
    } else {
        None
    };

    let queued = queue_wire
        .queued_job_ids
        .iter()
        .filter_map(|job_id| get_job_entry(con, namespace, job_id).transpose())
        .collect::<Result<Vec<_>>>()?;
    let completed = queue_wire
        .completed_job_ids
        .iter()
        .rev()
        .take(6)
        .filter_map(|job_id| get_job_entry(con, namespace, job_id).transpose())
        .collect::<Result<Vec<_>>>()?;

    let failed = queue_wire
        .failed_job_ids
        .iter()
        .rev()
        .take(6)
        .filter_map(|job_id| get_job_entry(con, namespace, job_id).transpose())
        .collect::<Result<Vec<_>>>()?;

    let active_run = load_active_run_view(con, namespace, active_job.as_ref(), preferred_run_id)?;
    let recent_events = load_recent_events(con, namespace, 24)?;

    Ok(DashboardData {
        queue: QueueView {
            all_queued_count: queue_wire.queued_job_ids.len(),
            all_completed_count: queue_wire.completed_job_ids.len(),
            all_failed_count: queue_wire.failed_job_ids.len(),
            queued,
            active: active_job,
            completed,
            failed,
        },
        active_run,
        recent_events,
    })
}

pub fn get_job_entry(
    con: &mut redis::Connection,
    namespace: &str,
    job_id: &str,
) -> Result<Option<JobEntry>> {
    let key = format!("{namespace}:jobs:{job_id}");
    let wire = match get_json::<JobSnapshotWire>(con, &key)? {
        Some(value) => value,
        None => return Ok(None),
    };
    let prompt_label = job_prompt_label(&wire.payload);
    Ok(Some(JobEntry {
        job_id: wire.job_id,
        kind: wire.job_type,
        prompt_label,
        command_id: wire.command_id,
        status: wire.status,
    }))
}

pub fn job_prompt_label(payload: &JsonValue) -> String {
    if let Some(prompt) = payload.get("prompt").and_then(JsonValue::as_str) {
        return truncate(prompt, 46);
    }
    if let Some(item_id) = payload.get("prompt_id").and_then(JsonValue::as_str) {
        return format!("Batch item {item_id}");
    }
    if let Some(item_count) = payload.get("item_count").and_then(JsonValue::as_u64) {
        return format!("Batch parent ({item_count} items)");
    }
    "Unknown job".to_string()
}

pub fn load_active_run_view(
    con: &mut redis::Connection,
    namespace: &str,
    active_job: Option<&JobEntry>,
    preferred_run_id: Option<&str>,
) -> Result<RunView> {
    let run_wire = if let Some(active_job) = active_job {
        let job_key = format!("{namespace}:jobs:{}", active_job.job_id);
        let job_wire = get_json::<JobSnapshotWire>(con, &job_key)?;
        if let Some(run_id) = job_wire.and_then(|job| job.run_id) {
            get_json::<RunSnapshotWire>(con, &format!("{namespace}:runs:{run_id}"))?
        } else {
            latest_run_snapshot(con, namespace)?
        }
    } else if let Some(run_id) = preferred_run_id {
        // No active job, but we remember the last run — fetch it directly
        // to avoid latest_run_snapshot picking a different run from the scan.
        get_json::<RunSnapshotWire>(con, &format!("{namespace}:runs:{run_id}"))?
            .or(latest_run_snapshot(con, namespace)?)
    } else {
        latest_run_snapshot(con, namespace)?
    };

    if let Some(run_wire) = run_wire {
        let angles = load_angle_views(con, namespace, &run_wire.run_id)?;
        let stage_progress = derive_stage_progress(con, namespace, &run_wire, &angles)?;
        let elapsed_seconds = run_wire
            .metrics
            .get("elapsed_seconds")
            .and_then(JsonValue::as_f64)
            .unwrap_or_default() as u64;
        let progress_ratio = stage_completion_ratio(&stage_progress);
        let metrics = run_wire
            .metrics
            .iter()
            .map(|(label, value)| MetricEntry {
                label: label.clone(),
                value: json_value_label(value),
            })
            .collect::<Vec<_>>();
        return Ok(RunView {
            run_id: run_wire.run_id,
            prompt: run_wire.prompt,
            status: run_wire.status,
            stage: run_wire.stage.unwrap_or_else(|| "unknown".to_string()),
            elapsed_seconds,
            progress_ratio,
            stage_progress,
            angles,
            metrics,
        });
    }

    Ok(RunView {
        run_id: "none".to_string(),
        prompt: "No active run".to_string(),
        status: "idle".to_string(),
        stage: "idle".to_string(),
        elapsed_seconds: 0,
        progress_ratio: 0.0,
        stage_progress: STAGE_ORDER
            .iter()
            .map(|label| StageProgress {
                label: (*label).to_string(),
                status: "pending".to_string(),
                duration_seconds: 0,
            })
            .collect(),
        angles: Vec::new(),
        metrics: Vec::new(),
    })
}

pub fn run_snapshot_rank(run: &RunSnapshotWire) -> (u8, String) {
    let running = if run.status == "running" { 1u8 } else { 0u8 };
    let ts = run
        .finished_at
        .as_deref()
        .or(run.started_at.as_deref())
        .unwrap_or("")
        .to_string();
    (running, ts)
}

pub fn latest_run_snapshot(
    con: &mut redis::Connection,
    namespace: &str,
) -> Result<Option<RunSnapshotWire>> {
    let pattern = format!("{namespace}:runs:*");
    let keys = con
        .scan_match::<_, String>(pattern)
        .context("failed to scan run snapshots")?
        .collect::<Vec<_>>();
    let mut candidates: Vec<RunSnapshotWire> = Vec::new();
    for key in keys {
        if key.contains(":angles:") {
            continue;
        }
        if let Some(run) = get_json::<RunSnapshotWire>(con, &key)? {
            candidates.push(run);
        }
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    candidates.sort_by(|a, b| {
        let ra = run_snapshot_rank(a);
        let rb = run_snapshot_rank(b);
        // Running before completed; then lexicographically greatest ISO timestamp (newest).
        rb.cmp(&ra)
    });
    Ok(candidates.into_iter().next())
}

pub fn load_angle_views(
    con: &mut redis::Connection,
    namespace: &str,
    run_id: &str,
) -> Result<Vec<AngleView>> {
    let pattern = format!("{namespace}:runs:{run_id}:angles:*");
    let keys = con
        .scan_match::<_, String>(pattern)
        .context("failed to scan angle snapshots")?
        .collect::<Vec<_>>();
    let mut angles = Vec::new();
    for key in keys {
        if let Some(angle) = get_json::<AngleSnapshotWire>(con, &key)? {
            angles.push(AngleView {
                angle_id: angle.angle_id,
                topic: angle.topic,
                stage: angle.stage.unwrap_or_else(|| "unknown".to_string()),
                iteration: angle.iteration,
                queries_total: angle.queries_total,
                context_ids_total: angle.context_ids_total,
                status: angle.status,
            });
        }
    }
    angles.sort_by(|left, right| left.angle_id.cmp(&right.angle_id));
    Ok(angles)
}

pub fn derive_stage_progress(
    con: &mut redis::Connection,
    namespace: &str,
    run: &RunSnapshotWire,
    _angles: &[AngleView],
) -> Result<Vec<StageProgress>> {
    let events = load_recent_runtime_event_payloads(con, namespace, 40)?;
    let mut stage_status = BTreeMap::new();
    for event in events {
        if let Some(stage) = event.stage {
            stage_status.entry(stage).or_insert(event.status);
        }
    }

    let current_stage = run.stage.clone().unwrap_or_default();
    let current_stage_status = run
        .stage_status
        .clone()
        .unwrap_or_else(|| "running".to_string());

    let current_index = STAGE_ORDER
        .iter()
        .position(|&s| s == current_stage)
        .unwrap_or(0);

    let mut out = Vec::new();
    for (i, &label) in STAGE_ORDER.iter().enumerate() {
        let status = if i < current_index {
            stage_status
                .get(label)
                .cloned()
                .unwrap_or_else(|| "completed".to_string())
        } else if i == current_index {
            current_stage_status.clone()
        } else {
            "pending".to_string()
        };
        out.push(StageProgress {
            label: label.to_string(),
            status,
            duration_seconds: 0,
        });
    }
    Ok(out)
}

pub fn stage_completion_ratio(stage_progress: &[StageProgress]) -> f64 {
    let total = stage_progress.len();
    if total == 0 {
        return 0.0;
    }
    let mut completed = 0;
    for s in stage_progress {
        if s.status == "completed" || s.status == "success" {
            completed += 1;
        }
    }
    completed as f64 / total as f64
}

pub fn load_recent_events(
    con: &mut redis::Connection,
    namespace: &str,
    count: usize,
) -> Result<Vec<EventEntry>> {
    let payloads = load_recent_runtime_event_payloads(con, namespace, count)?;
    Ok(payloads
        .into_iter()
        .map(|p| EventEntry {
            ts: p.ts,
            component: p.component,
            message: p.message,
            stage: p.stage,
            level: normalize_event_level(&p.status),
        })
        .collect())
}

pub fn load_recent_runtime_event_payloads(
    con: &mut redis::Connection,
    namespace: &str,
    count: usize,
) -> Result<Vec<RuntimeEventPayload>> {
    let stream_key = format!("{namespace}:events");
    let range: StreamRangeReply = cmd("XREVRANGE")
        .arg(&stream_key)
        .arg("+")
        .arg("-")
        .arg("COUNT")
        .arg(count)
        .query(con)
        .context("failed to read events from Redis stream")?;

    let mut out = Vec::new();
    for entry in range.ids {
        if let Some(payload_val) = entry.map.get("payload") {
            let s = String::from_redis_value(payload_val)?;
            if let Ok(p) = serde_json::from_str::<RuntimeEventPayload>(&s) {
                out.push(p);
            }
        }
    }
    Ok(out)
}

pub fn get_json<T>(con: &mut redis::Connection, key: &str) -> Result<Option<T>>
where
    T: for<'de> serde::Deserialize<'de>,
{
    let val: Option<String> = con.get(key).context(format!("failed to GET {key}"))?;
    match val {
        Some(s) => Ok(Some(serde_json::from_str(&s)?)),
        None => Ok(None),
    }
}
