use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind,
};
use crossterm::execute;
use ratatui::{
    DefaultTerminal,
    layout::{Constraint, Direction, Layout, Rect},
    prelude::*,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Gauge, Paragraph, Row, Table, Wrap},
};
use redis::{Commands, FromRedisValue, cmd, streams::StreamRangeReply};
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};

const STAGE_ORDER: [&str; 9] = [
    "scout",
    "planning",
    "research_parallel",
    "kg_collect",
    "kg_filter",
    "kg_re",
    "entity_resolution",
    "outline",
    "write",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Focus {
    Input,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputMode {
    DirectQuery,
}

#[derive(Clone, Debug, Deserialize)]
struct DashboardData {
    #[allow(dead_code)]
    queue: QueueView,
    active_run: RunView,
    recent_events: Vec<EventEntry>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[allow(dead_code)]
struct QueueView {
    queued: Vec<JobEntry>,
    active: Option<JobEntry>,
    completed: Vec<JobEntry>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[allow(dead_code)]
struct JobEntry {
    job_id: String,
    kind: String,
    prompt_label: String,
    command_id: String,
    status: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RunView {
    run_id: String,
    #[allow(dead_code)]
    prompt: String,
    #[allow(dead_code)]
    status: String,
    stage: String,
    elapsed_seconds: u64,
    #[allow(dead_code)]
    progress_ratio: f64,
    stage_progress: Vec<StageProgress>,
    angles: Vec<AngleView>,
    metrics: Vec<MetricEntry>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct StageProgress {
    label: String,
    status: String,
    duration_seconds: u64,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct AngleView {
    angle_id: String,
    topic: String,
    stage: String,
    iteration: u32,
    queries_total: u32,
    context_ids_total: u32,
    status: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct MetricEntry {
    label: String,
    value: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct EventEntry {
    ts: String,
    component: String,
    message: String,
    stage: Option<String>,
    level: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct MockFileState {
    queue: QueueView,
    active_run: RunView,
    recent_events: Vec<EventEntry>,
    direct_query_draft: String,
    batch_json_draft: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct QueueSnapshotWire {
    queued_job_ids: Vec<String>,
    active_job_id: Option<String>,
    completed_job_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct JobSnapshotWire {
    job_id: String,
    command_id: String,
    #[serde(rename = "type")]
    job_type: String,
    status: String,
    payload: JsonValue,
    run_id: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RunSnapshotWire {
    run_id: String,
    prompt: String,
    status: String,
    stage: Option<String>,
    stage_status: Option<String>,
    metrics: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct AngleSnapshotWire {
    angle_id: String,
    topic: String,
    stage: Option<String>,
    iteration: u32,
    queries_total: u32,
    context_ids_total: u32,
    status: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RuntimeEventPayload {
    ts: String,
    component: String,
    message: String,
    stage: Option<String>,
    status: String,
    #[serde(rename = "metrics")]
    _metrics: BTreeMap<String, JsonValue>,
}

#[derive(Clone)]
enum DataSource {
    Mock {
        path: PathBuf,
    },
    Live {
        client: redis::Client,
        namespace: String,
    },
}

impl DataSource {
    fn load(
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

    fn submit(&self, mode: InputMode, input: &str) -> Result<String> {
        match self {
            Self::Mock { .. } => Ok("Mock mode: submission preview only.".to_string()),
            Self::Live { client, namespace } => {
                let mut con = client
                    .get_connection()
                    .context("failed to connect to Redis for command submission")?;
                let command_id = format!("cmd_rs_{}", unix_millis()?);
                let payload = match mode {
                    InputMode::DirectQuery => {
                        let prompt = input.trim();
                        if prompt.is_empty() {
                            return Err(anyhow!("prompt is empty"));
                        }
                        json!({
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
                                "metadata": {}
                            }
                        })
                    }
                };

                let stream_key = format!("{namespace}:commands");
                let payload_string = serde_json::to_string(&payload)?;
                let stream_id: String = cmd("XADD")
                    .arg(&stream_key)
                    .arg("*")
                    .arg("payload")
                    .arg(payload_string)
                    .query(&mut con)
                    .context("failed to append command to Redis stream")?;
                Ok(format!(
                    "Submitted {command_id} to {stream_key} ({stream_id})"
                ))
            }
        }
    }

    fn source_label(&self) -> String {
        match self {
            Self::Mock { path } => format!("mock:{}", path.display()),
            Self::Live { namespace, .. } => format!("live:{namespace}"),
        }
    }
}

struct App {
    data_source: DataSource,
    dashboard: DashboardData,
    focus: Focus,
    input_mode: InputMode,
    show_help: bool,
    direct_query_draft: String,
    last_status: String,
    last_refresh: Option<Instant>,
    // Remembered across refreshes so that once a run's job leaves the active
    // slot we still fetch that specific run snapshot rather than letting
    // latest_run_snapshot drift to a different (possibly empty) run.
    last_run_id: Option<String>,
}

impl App {
    fn new(
        data_source: DataSource,
        dashboard: DashboardData,
        direct_query_draft: String,
        _batch_json_draft: String,
    ) -> Self {
        let last_run_id = {
            let id = &dashboard.active_run.run_id;
            if id != "none" && !id.is_empty() {
                Some(id.clone())
            } else {
                None
            }
        };
        Self {
            data_source,
            dashboard,
            focus: Focus::Input,
            input_mode: InputMode::DirectQuery,
            show_help: false,
            direct_query_draft,
            last_status: "Ready.".to_string(),
            last_refresh: Some(Instant::now()),
            last_run_id,
        }
    }

    fn refresh(&mut self) {
        match self.data_source.load(self.last_run_id.as_deref()) {
            Ok((dashboard, direct_draft, batch_draft)) => {
                // Keep the run_id sticky as long as we've seen a real one.
                let new_id = &dashboard.active_run.run_id;
                if new_id != "none" && !new_id.is_empty() {
                    self.last_run_id = Some(new_id.clone());
                }
                self.dashboard = dashboard;
                if let Some(value) = direct_draft {
                    self.direct_query_draft = value;
                }
                let _ = batch_draft;
                self.last_refresh = Some(Instant::now());
                self.last_status = format!("Refreshed from {}.", self.data_source.source_label());
            }
            Err(err) => {
                self.last_status = format!("Refresh failed: {err}");
            }
        }
    }

    fn submit_current(&mut self) {
        let input = self.active_input_text().to_string();
        match self.data_source.submit(self.input_mode, &input) {
            Ok(message) => {
                self.last_status = message;
                self.direct_query_draft.clear();
                // Drop the sticky run ID so the next refresh picks up the new
                // run rather than continuing to show the just-completed one.
                self.last_run_id = None;
                // Reset the displayed run immediately — don't wait for the next
                // poll cycle to clear old stage/metric data.
                self.dashboard.active_run = RunView {
                    run_id: "none".to_string(),
                    prompt: String::new(),
                    status: "queued".to_string(),
                    stage: "queued".to_string(),
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
                };
            }
            Err(err) => {
                self.last_status = format!("Submit failed: {err}");
            }
        }
    }

    fn next_focus(&mut self) {
        self.focus = Focus::Input;
    }

    fn prev_focus(&mut self) {
        self.focus = Focus::Input;
    }

    fn active_input_text(&self) -> &str {
        &self.direct_query_draft
    }

    fn handle_char(&mut self, ch: char) {
        if self.focus != Focus::Input {
            return;
        }
        self.direct_query_draft.push(ch);
    }

    fn backspace(&mut self) {
        if self.focus != Focus::Input {
            return;
        }
        self.direct_query_draft.pop();
    }

}

fn main() -> Result<()> {
    let data_source = parse_data_source_from_args()?;
    let (dashboard, direct_query_draft, batch_json_draft) = data_source.load(None)?;
    let direct_query_draft = direct_query_draft.unwrap_or_default();
    let _ = batch_json_draft;

    let terminal = ratatui::init();
    // Capture mouse events so scroll wheel is absorbed by the app and does not
    // leak through to the host terminal's scrollback buffer.
    execute!(std::io::stdout(), EnableMouseCapture).ok();

    let result = run_app(
        terminal,
        App::new(data_source, dashboard, direct_query_draft, String::new()),
    );

    execute!(std::io::stdout(), DisableMouseCapture).ok();
    ratatui::restore();

    result
}

fn parse_data_source_from_args() -> Result<DataSource> {
    let mut args = std::env::args().skip(1);
    let mut mock_path: Option<PathBuf> = None;
    let mut redis_url = "redis://127.0.0.1:6379/0".to_string();
    let mut namespace = "runtime".to_string();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mock" => {
                let path = args
                    .next()
                    .ok_or_else(|| anyhow!("--mock requires a file path"))?;
                mock_path = Some(PathBuf::from(path));
            }
            "--redis-url" => {
                redis_url = args
                    .next()
                    .ok_or_else(|| anyhow!("--redis-url requires a value"))?;
            }
            "--namespace" => {
                namespace = args
                    .next()
                    .ok_or_else(|| anyhow!("--namespace requires a value"))?;
            }
            other => {
                return Err(anyhow!(
                    "unknown argument: {other}. Use --mock <path> or --redis-url <url> --namespace <name>"
                ));
            }
        }
    }

    if let Some(path) = mock_path {
        return Ok(DataSource::Mock { path });
    }

    let client = redis::Client::open(redis_url.clone())
        .with_context(|| format!("failed to parse redis url {redis_url}"))?;
    Ok(DataSource::Live { client, namespace })
}

fn run_app(mut terminal: DefaultTerminal, mut app: App) -> Result<()> {
    let mut last_poll = Instant::now();

    loop {
        if last_poll.elapsed() >= Duration::from_millis(1200) {
            app.refresh();
            last_poll = Instant::now();
        }

        terminal.draw(|frame| render(frame, &app))?;

        if !event::poll(Duration::from_millis(150))? {
            continue;
        }

        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }

            match key.code {
                KeyCode::Char('q') => return Ok(()),
                KeyCode::Char('r') => {
                    app.refresh();
                    last_poll = Instant::now();
                }
                KeyCode::Tab => app.next_focus(),
                KeyCode::BackTab => app.prev_focus(),
                KeyCode::Char('?') => app.show_help = !app.show_help,
                KeyCode::Backspace => app.backspace(),
                KeyCode::Enter => {
                    if app.focus == Focus::Input {
                        app.submit_current();
                    }
                }
                KeyCode::Char(ch) => app.handle_char(ch),
                _ => {}
            }
        }
    }
}

fn load_live_dashboard(
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

    let active_run = load_active_run_view(con, namespace, active_job.as_ref(), preferred_run_id)?;
    let recent_events = load_recent_events(con, namespace, 12)?;

    Ok(DashboardData {
        queue: QueueView {
            queued,
            active: active_job,
            completed,
        },
        active_run,
        recent_events,
    })
}

fn get_job_entry(
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

fn job_prompt_label(payload: &JsonValue) -> String {
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

fn load_active_run_view(
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

fn latest_run_snapshot(
    con: &mut redis::Connection,
    namespace: &str,
) -> Result<Option<RunSnapshotWire>> {
    let pattern = format!("{namespace}:runs:*");
    let mut best: Option<(String, RunSnapshotWire)> = None;
    let keys = con
        .scan_match::<_, String>(pattern)
        .context("failed to scan run snapshots")?
        .collect::<Vec<_>>();
    for key in keys {
        if key.contains(":angles:") {
            continue;
        }
        if let Some(run) = get_json::<RunSnapshotWire>(con, &key)? {
            let better = best
                .as_ref()
                .map(|(_, current)| run.status == "running" || current.status != "running")
                .unwrap_or(true);
            if better {
                best = Some((key, run));
            }
        }
    }
    Ok(best.map(|(_, run)| run))
}

fn load_angle_views(
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

fn derive_stage_progress(
    con: &mut redis::Connection,
    namespace: &str,
    run: &RunSnapshotWire,
    _angles: &[AngleView],
) -> Result<Vec<StageProgress>> {
    let events = load_recent_runtime_event_payloads(con, namespace, 40)?;
    let mut stage_status = BTreeMap::new();
    // XREVRANGE returns newest-first; use or_insert so the newest event wins.
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
        .position(|stage| *stage == current_stage)
        .unwrap_or(usize::MAX);

    // When the overall run is done the orchestrator sets run.stage to something
    // outside STAGE_ORDER (e.g. "completed", null). current_index becomes
    // usize::MAX which would otherwise prevent the index-based "completed"
    // fallback from firing, resetting every stage to pending.
    let run_is_terminal = matches!(
        run.status.as_str(),
        "completed" | "done" | "success" | "finished"
    );

    Ok(STAGE_ORDER
        .iter()
        .enumerate()
        .map(|(index, stage)| {
            let status = if run_is_terminal {
                // Run finished: mark everything completed, only trust explicit
                // "failed" events to flag individual stage failures.
                match stage_status.get(*stage).map(String::as_str) {
                    Some("failed") => "failed".to_string(),
                    _ => "completed".to_string(),
                }
            } else if current_index != usize::MAX && index < current_index {
                // Past stages are definitively completed once the orchestrator
                // has moved on, regardless of any stale "running" events.
                "completed".to_string()
            } else if *stage == current_stage {
                // Normalize: anything that isn't an explicit terminal state is
                // "running" so ▶ lights up regardless of the exact string written.
                match current_stage_status.as_str() {
                    "completed" | "failed" => current_stage_status.clone(),
                    _ => "running".to_string(),
                }
            } else {
                // Future stages are always pending during an active run.
                // The event map is NOT consulted here: the stream is shared
                // across runs, so it contains "completed" events from the
                // previous run which would falsely mark these stages as done.
                "pending".to_string()
            };
            StageProgress {
                label: (*stage).to_string(),
                status,
                duration_seconds: 0,
            }
        })
        .collect())
}

fn stage_completion_ratio(stage_progress: &[StageProgress]) -> f64 {
    if stage_progress.is_empty() {
        return 0.0;
    }
    let completed = stage_progress
        .iter()
        .filter(|stage| stage.status == "completed")
        .count() as f64;
    let running = stage_progress
        .iter()
        .filter(|stage| stage.status == "running")
        .count() as f64;
    ((completed + 0.5 * running) / stage_progress.len() as f64).clamp(0.0, 1.0)
}

fn load_recent_events(
    con: &mut redis::Connection,
    namespace: &str,
    count: usize,
) -> Result<Vec<EventEntry>> {
    let payloads = load_recent_runtime_event_payloads(con, namespace, count)?;
    let mut entries = payloads
        .into_iter()
        .map(|payload| EventEntry {
            ts: payload.ts,
            component: payload.component,
            message: payload.message,
            stage: payload.stage,
            level: normalize_event_level(&payload.status),
        })
        .collect::<Vec<_>>();
    entries.reverse();
    Ok(entries)
}

fn load_recent_runtime_event_payloads(
    con: &mut redis::Connection,
    namespace: &str,
    count: usize,
) -> Result<Vec<RuntimeEventPayload>> {
    let key = format!("{namespace}:events");
    let reply: StreamRangeReply = cmd("XREVRANGE")
        .arg(&key)
        .arg("+")
        .arg("-")
        .arg("COUNT")
        .arg(count)
        .query(con)
        .with_context(|| format!("failed to load events from {key}"))?;
    let mut payloads = Vec::new();
    for stream_id in reply.ids {
        if let Some(value) = stream_id.map.get("payload") {
            let raw = String::from_redis_value(value)
                .context("failed to decode runtime event payload")?;
            let parsed: RuntimeEventPayload =
                serde_json::from_str(&raw).context("failed to parse runtime event payload JSON")?;
            payloads.push(parsed);
        }
    }
    Ok(payloads)
}

fn get_json<T>(con: &mut redis::Connection, key: &str) -> Result<Option<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let raw: Option<String> = con
        .get(key)
        .with_context(|| format!("failed to GET {key}"))?;
    match raw {
        Some(value) => {
            Ok(Some(serde_json::from_str(&value).with_context(|| {
                format!("failed to parse JSON from {key}")
            })?))
        }
        None => Ok(None),
    }
}

fn normalize_event_level(status: &str) -> String {
    match status {
        "failed" | "error" => "error".to_string(),
        "warning" => "warning".to_string(),
        _ => "info".to_string(),
    }
}

fn json_value_label(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(v) => v.to_string(),
        JsonValue::Number(v) => v.to_string(),
        JsonValue::String(v) => v.clone(),
        JsonValue::Array(v) => format!("[{} items]", v.len()),
        JsonValue::Object(v) => format!("{{{} keys}}", v.len()),
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    trimmed.chars().take(max_chars).collect::<String>() + "..."
}

fn unix_millis() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time is before UNIX_EPOCH")?
        .as_millis())
}

fn iso_like_now() -> Result<String> {
    Ok(format!("{}Z", unix_millis()?))
}

fn render(frame: &mut Frame, app: &App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header bar
            Constraint::Min(14),   // main: pipeline + KG/Angles | Events
            Constraint::Length(5), // query input
            Constraint::Length(1), // footer
        ])
        .split(frame.area());

    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(69), Constraint::Percentage(31)])
        .split(root[1]);

    render_header(frame, root[0], app);
    render_run_center(frame, main[0], app);
    render_events(frame, main[1], app);
    render_input(frame, root[2], app);
    render_footer(frame, root[3], app);

    if app.show_help {
        render_help(frame);
    }
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let run = &app.dashboard.active_run;
    let stage_color = match run.stage.as_str() {
        "idle" | "none" => Color::DarkGray,
        _ => Color::Cyan,
    };
    let refresh_label = app
        .last_refresh
        .map(|t| {
            let s = t.elapsed().as_secs();
            if s < 2 {
                "just now".to_string()
            } else {
                format!("{s}s ago")
            }
        })
        .unwrap_or_else(|| "never".to_string());

    let spans = vec![
        Span::styled(
            " MING ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled("run:", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!(" {} ", run.run_id),
            Style::default().fg(Color::White),
        ),
        Span::styled("  stage:", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!(" {} ", run.stage),
            Style::default().fg(stage_color),
        ),
        Span::styled("  elapsed:", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!(" {}s", run.elapsed_seconds),
            Style::default().fg(Color::White),
        ),
        Span::styled("  refreshed:", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!(" {} ", refresh_label),
            Style::default().fg(Color::White),
        ),
        Span::styled(
            format!("  [{}]", app.data_source.source_label()),
            Style::default().fg(Color::DarkGray),
        ),
    ];

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let spans = vec![
        Span::styled(&app.last_status, Style::default().fg(Color::White)),
        Span::styled(
            "    q: quit   r: refresh   ?: help",
            Style::default().fg(Color::DarkGray),
        ),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_run_center(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // stage pipeline (2 text lines + 2 borders)
            Constraint::Length(3), // progress gauge
            Constraint::Min(8),    // KG pipeline + angles
        ])
        .split(area);

    render_stage_pipeline(frame, chunks[0], app);
    render_progress_gauge(frame, chunks[1], app);

    let lower = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(chunks[2]);

    render_kg_pipeline(frame, lower[0], app);
    render_angle_statuses(frame, lower[1], app);
}

fn render_stage_pipeline(frame: &mut Frame, area: Rect, app: &App) {
    let stage_defs: &[(&str, &str)] = &[
        ("scout", "scout"),
        ("planning", "plan"),
        ("research_parallel", "rsrch"),
        ("kg_collect", "coll"),
        ("kg_filter", "filt"),
        ("kg_re", "re"),
        ("entity_resolution", "resol"),
        ("outline", "outln"),
        ("write", "write"),
    ];

    let stages = &app.dashboard.active_run.stage_progress;
    let mut label_spans: Vec<Span> = vec![];
    let mut icon_spans: Vec<Span> = vec![];

    for (i, (key, short)) in stage_defs.iter().enumerate() {
        let status = stages
            .iter()
            .find(|s| &s.label == key)
            .map(|s| s.status.as_str())
            .unwrap_or("pending");

        let (icon, color, bold) = match status {
            "completed" => ("✓", Color::Green, false),
            "running" => ("▶", Color::Yellow, true),
            "failed" => ("✗", Color::Red, false),
            _ => ("·", Color::DarkGray, false),
        };

        let w = 7usize;
        let label_style = Style::default().fg(color);
        let icon_style = if bold {
            Style::default().fg(color).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(color)
        };

        label_spans.push(Span::styled(
            format!("{:^width$}", short, width = w),
            label_style,
        ));
        icon_spans.push(Span::styled(
            format!("{:^width$}", icon, width = w),
            icon_style,
        ));

        if i + 1 < stage_defs.len() {
            let arrow_color = if status == "completed" {
                Color::Green
            } else {
                Color::DarkGray
            };
            label_spans.push(Span::styled("─", Style::default().fg(Color::DarkGray)));
            icon_spans.push(Span::styled("─", Style::default().fg(arrow_color)));
        }
    }

    let text = vec![Line::from(label_spans), Line::from(icon_spans)];
    frame.render_widget(
        Paragraph::new(text).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Pipeline ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )),
        ),
        area,
    );
}

fn render_progress_gauge(frame: &mut Frame, area: Rect, app: &App) {
    let run = &app.dashboard.active_run;
    let ratio = run.progress_ratio.clamp(0.0, 1.0);
    let pct = (ratio * 100.0).round() as u16;
    let label = format!("{pct}%  ·  elapsed {}s", run.elapsed_seconds);

    let color = if ratio >= 1.0 {
        Color::Green
    } else if ratio > 0.0 {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    frame.render_widget(
        Gauge::default()
            .block(Block::default().borders(Borders::ALL))
            .gauge_style(Style::default().fg(color).bg(Color::DarkGray))
            .ratio(ratio)
            .label(label),
        area,
    );
}

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    let text_with_cursor = format!("{}▌", app.active_input_text());
    frame.render_widget(
        Paragraph::new(text_with_cursor)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(
                        " Query ",
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .title_bottom(Line::from(vec![
                        Span::styled(
                            " Enter ",
                            Style::default().fg(Color::Black).bg(Color::Yellow),
                        ),
                        Span::styled(" to submit  ", Style::default().fg(Color::DarkGray)),
                    ])),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn render_kg_pipeline(frame: &mut Frame, area: Rect, app: &App) {
    let stages = [
        ("kg_collect", "Collect"),
        ("kg_filter", "Filter"),
        ("kg_re", "NER / RE"),
        ("entity_resolution", "Resolve"),
    ];

    let lines = stages
        .iter()
        .flat_map(|(key, label)| {
            let processed =
                metric_value(app, &format!("{key}_processed")).unwrap_or_else(|| "-".to_string());
            let total =
                metric_value(app, &format!("{key}_total")).unwrap_or_else(|| "-".to_string());
            let elapsed = metric_value(app, &format!("{key}_elapsed_seconds"))
                .unwrap_or_else(|| "-".to_string());
            let status = app
                .dashboard
                .active_run
                .stage_progress
                .iter()
                .find(|s| &s.label == key)
                .map(|s| s.status.as_str())
                .unwrap_or("pending");

            let (icon, color) = match status {
                "completed" => ("✓", Color::Green),
                "running" => ("▶", Color::Yellow),
                "failed" => ("✗", Color::Red),
                _ => ("·", Color::DarkGray),
            };

            vec![
                Line::from(vec![
                    Span::styled(
                        icon,
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(
                        format!("{:<12}", label),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("{}/{}", processed, total),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled(
                        format!("  {}s", elapsed),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]),
                Line::from(""),
            ]
        })
        .collect::<Vec<_>>();

    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(
                        " KG Pipeline ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_angle_statuses(frame: &mut Frame, area: Rect, app: &App) {
    let rows = app
        .dashboard
        .active_run
        .angles
        .iter()
        .map(|angle| {
            let (icon, color) = match angle.status.as_str() {
                "completed" => ("✓", Color::Green),
                "running" => ("▶", Color::Yellow),
                "failed" => ("✗", Color::Red),
                _ => ("·", Color::DarkGray),
            };
            Row::new(vec![
                format!("{} {}", icon, angle.topic),
                angle.stage.clone(),
                format!("{}q {}s", angle.queries_total, angle.context_ids_total),
            ])
            .style(Style::default().fg(color))
        })
        .collect::<Vec<_>>();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(50),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ],
    )
    .header(
        Row::new(vec!["Angle", "Stage", "Q/Src"])
            .style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(
                " Angles ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
    );

    frame.render_widget(table, area);
}

fn render_events(frame: &mut Frame, area: Rect, app: &App) {
    let lines = app
        .dashboard
        .recent_events
        .iter()
        .map(|entry| {
            let (level_icon, color) = match entry.level.as_str() {
                "error" => ("✗", Color::Red),
                "warning" => ("!", Color::Yellow),
                _ => ("·", Color::DarkGray),
            };
            // Show only time portion (last 8 chars) if the timestamp is long
            let ts_short = if entry.ts.len() > 8 {
                &entry.ts[entry.ts.len() - 8..]
            } else {
                &entry.ts
            };
            Line::from(vec![
                Span::styled(
                    level_icon,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(
                    format!("{} ", ts_short),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("[{}]", entry.component),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" "),
                Span::styled(entry.message.clone(), Style::default().fg(color)),
                Span::styled(
                    entry
                        .stage
                        .as_ref()
                        .map(|s| format!(" ({s})"))
                        .unwrap_or_default(),
                    Style::default().fg(Color::DarkGray),
                ),
            ])
        })
        .collect::<Vec<_>>();

    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(
                        " Events ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_help(frame: &mut Frame) {
    let area = centered_rect(46, 52, frame.area());
    frame.render_widget(Clear, area);

    let help_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  q          ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("quit", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(
                "  r          ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("refresh now", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(
                "  ?          ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("toggle this help", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(
                "  Enter      ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("submit query", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(
                "  Backspace  ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("delete last character", Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Type directly to compose a query.",
            Style::default().fg(Color::DarkGray),
        )]),
        Line::from(vec![Span::styled(
            "  Auto-refreshes every 1.2s.",
            Style::default().fg(Color::DarkGray),
        )]),
    ];

    frame.render_widget(
        Paragraph::new(help_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow))
                    .title(Span::styled(
                        " Help ",
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    )),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn metric_value(app: &App, key: &str) -> Option<String> {
    app.dashboard
        .active_run
        .metrics
        .iter()
        .find(|metric| metric.label == key)
        .map(|metric| metric.value.clone())
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
