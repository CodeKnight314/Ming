use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
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
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
};
use redis::{Commands, FromRedisValue, cmd, streams::StreamRangeReply};
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};

/// Gemini-CLI–inspired dark theme (RGB works in most modern terminals).
mod theme {
    use ratatui::style::Color;

    pub const ACCENT_BLUE: Color = Color::Rgb(138, 180, 248);
    pub const ACCENT_PURPLE: Color = Color::Rgb(197, 151, 230);
    pub const TEXT: Color = Color::Rgb(232, 234, 237);
    pub const MUTED: Color = Color::Rgb(139, 148, 158);
    pub const SUBTLE: Color = Color::Rgb(72, 79, 88);
    pub const SUCCESS: Color = Color::Rgb(129, 201, 149);
    pub const WARN: Color = Color::Rgb(242, 204, 96);
    pub const ERROR: Color = Color::Rgb(255, 123, 114);
    pub const INPUT_BG: Color = Color::Rgb(30, 34, 42);
}

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

#[derive(Clone, Debug, Deserialize)]
struct DashboardData {
    #[allow(dead_code)]
    queue: QueueView,
    active_run: RunView,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    duration_seconds: u64,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct AngleView {
    angle_id: String,
    topic: String,
    stage: String,
    #[allow(dead_code)]
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
#[allow(dead_code)]
struct EventEntry {
    ts: String,
    component: String,
    message: String,
    stage: Option<String>,
    level: String,
}

#[derive(Debug, Deserialize)]
struct WriteSectionProgress {
    title: String,
    status: String,
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
    #[serde(default)]
    started_at: Option<String>,
    #[serde(default)]
    finished_at: Option<String>,
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

    fn submit_run_query(&self, prompt: &str) -> Result<String> {
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
                        "metadata": {}
                    }
                });
                xadd_command(&mut con, namespace, &payload, &command_id)
            }
        }
    }

    fn submit_run_batch(&self, items: &[(String, String)]) -> Result<String> {
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

    fn source_label(&self) -> String {
        match self {
            Self::Mock { path } => format!("mock:{}", path.display()),
            Self::Live { namespace, .. } => format!("live:{namespace}"),
        }
    }
}

fn xadd_command(
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

fn json_batch_id(value: &JsonValue) -> Result<String> {
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

fn push_batch_object(obj: &serde_json::Map<String, JsonValue>, index: &str) -> Result<(String, String)> {
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

fn ensure_unique_batch_ids(items: &[(String, String)]) -> Result<()> {
    let mut seen = HashSet::new();
    for (id, _) in items {
        if !seen.insert(id.as_str()) {
            return Err(anyhow!("duplicate batch id: {id}"));
        }
    }
    Ok(())
}

/// JSON array of `{id, prompt}` or JSONL with one object per line (e.g. deepresearch-bench `query.jsonl`).
fn load_batch_items_from_path(path: &Path) -> Result<Vec<(String, String)>> {
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

struct App {
    data_source: DataSource,
    dashboard: DashboardData,
    focus: Focus,
    show_help: bool,
    direct_query_draft: String,
    last_status: String,
    last_refresh: Option<Instant>,
    // Remembered across refreshes so that once a run's job leaves the active
    // slot we still fetch that specific run snapshot rather than letting
    // latest_run_snapshot drift to a different (possibly empty) run.
    last_run_id: Option<String>,
    /// Drives per-stage spinners (increments each frame).
    ui_tick: u32,
    /// Set when the user submits `/exit`; main loop stops on next frame.
    exit_requested: bool,
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
            show_help: false,
            direct_query_draft,
            last_status: "Ready.".to_string(),
            last_refresh: Some(Instant::now()),
            last_run_id,
            ui_tick: 0,
            exit_requested: false,
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
                // Do not overwrite `direct_query_draft` here: mock reload would reset the
                // composer every poll (~1.2s) and make typing impossible.
                let _ = (direct_draft, batch_draft);
                self.last_refresh = Some(Instant::now());
                // Do not overwrite `last_status` on success — periodic refresh would erase
                // submission messages; timing already shows in the header.
            }
            Err(err) => {
                self.last_status = format!("Refresh failed: {err}");
            }
        }
    }

    fn reset_dashboard_for_new_command(&mut self) {
        self.last_run_id = None;
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

    fn finish_run_query_submit(&mut self, prompt: &str) -> Result<String> {
        let msg = self.data_source.submit_run_query(prompt)?;
        self.reset_dashboard_for_new_command();
        self.direct_query_draft.clear();
        Ok(msg)
    }

    fn finish_batch_submit(&mut self, path: PathBuf) -> Result<String> {
        let items = load_batch_items_from_path(&path)?;
        let msg = self.data_source.submit_run_batch(&items)?;
        self.reset_dashboard_for_new_command();
        self.direct_query_draft.clear();
        Ok(msg)
    }

    fn handle_slash_command(&mut self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return;
        }
        let cmd = parts[0];
        match cmd {
            "/exit" => {
                self.exit_requested = true;
                self.last_status = "Exiting.".to_string();
            }
            "/batch" => {
                let path = parts[1..].join(" ");
                if path.is_empty() {
                    self.last_status =
                        "Usage: /batch <path>  (file: JSON array or JSONL; see F1 help)"
                            .to_string();
                    return;
                }
                match self.finish_batch_submit(PathBuf::from(path)) {
                    Ok(m) => self.last_status = m,
                    Err(e) => self.last_status = format!("{e:#}"),
                }
            }
            _ => {
                self.last_status =
                    "Unknown command — use /batch <path> or /exit.".to_string();
            }
        }
    }

    fn submit_current(&mut self) {
        let input = self.active_input_text().to_string();
        let trimmed = input.trim().to_string();
        if trimmed.is_empty() {
            self.last_status = "Nothing to submit.".to_string();
            return;
        }
        if trimmed.starts_with('/') {
            self.handle_slash_command(&trimmed);
            // Consume the command line so Enter does not re-run the same command.
            self.direct_query_draft.clear();
            return;
        }
        match self.finish_run_query_submit(&trimmed) {
            Ok(message) => self.last_status = message,
            Err(err) => self.last_status = format!("Submit failed: {err:#}"),
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

        app.ui_tick = app.ui_tick.wrapping_add(1);
        terminal.draw(|frame| render(frame, &app))?;

        if !event::poll(Duration::from_millis(150))? {
            continue;
        }

        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }

            match key.code {
                KeyCode::Tab => app.next_focus(),
                KeyCode::BackTab => app.prev_focus(),
                KeyCode::F(1) => app.show_help = !app.show_help,
                KeyCode::Esc => {
                    if app.show_help {
                        app.show_help = false;
                    }
                }
                KeyCode::Backspace => app.backspace(),
                KeyCode::Enter => {
                    if app.focus == Focus::Input {
                        app.submit_current();
                        if app.exit_requested {
                            return Ok(());
                        }
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
    let recent_events = load_recent_events(con, namespace, 24)?;

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

/// Prefer an in-flight run; otherwise pick the most recently finished (or started) snapshot.
/// Avoids scan-order noise where an older run (e.g. metrics wiped pre-fix) overwrote a good one.
fn run_snapshot_rank(run: &RunSnapshotWire) -> (u8, String) {
    let running = if run.status == "running" { 1u8 } else { 0u8 };
    let ts = run
        .finished_at
        .as_deref()
        .or(run.started_at.as_deref())
        .unwrap_or("")
        .to_string();
    (running, ts)
}

fn latest_run_snapshot(
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
        // Preserve structured metrics (e.g. write_sections_progress_json) so the TUI can parse them.
        JsonValue::Array(_) | JsonValue::Object(_) => serde_json::to_string(value).unwrap_or_default(),
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

fn run_is_terminal(run: &RunView) -> bool {
    matches!(
        run.status.as_str(),
        "completed" | "done" | "success" | "finished"
    )
}

fn run_is_real(run: &RunView) -> bool {
    !run.run_id.is_empty() && run.run_id != "none"
}

const KG_STAGE_KEYS: [&str; 4] = [
    "kg_collect",
    "kg_filter",
    "kg_re",
    "entity_resolution",
];

fn kg_has_failed_stage(run: &RunView) -> bool {
    KG_STAGE_KEYS.iter().any(|key| {
        run.stage_progress
            .iter()
            .find(|s| s.label == *key)
            .map(|s| s.status == "failed")
            .unwrap_or(false)
    })
}

const SPINNER_FRAMES: &[&str] =
    &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Pipeline order: earliest stage first in the list (top), latest last (bottom of the stage list).
/// `bottom_anchor_lines` pads above so the block sits low in the Activity panel when there is room.
const ACTIVITY_STACK: &[&str] = &[
    "scout",
    "planning",
    "research_parallel",
    "__kg__",
    "outline",
    "write",
];

fn stage_row_status<'a>(run: &'a RunView, key: &str) -> &'a str {
    run.stage_progress
        .iter()
        .find(|s| s.label == key)
        .map(|s| s.status.as_str())
        .unwrap_or("pending")
}

fn activity_block_visible(run: &RunView, block: &str) -> bool {
    if !run_is_real(run) {
        return false;
    }
    if run_is_terminal(run) {
        return true;
    }
    match block {
        "scout" => stage_row_status(run, "scout") != "pending" || run.stage == "scout",
        "planning" => stage_row_status(run, "planning") != "pending" || run.stage == "planning",
        "research_parallel" => {
            stage_row_status(run, "research_parallel") != "pending"
                || run.stage == "research_parallel"
        }
        "__kg__" => KG_STAGE_KEYS.iter().any(|k| {
            stage_row_status(run, k) != "pending" || run.stage == *k
        }),
        "outline" => stage_row_status(run, "outline") != "pending" || run.stage == "outline",
        "write" => stage_row_status(run, "write") != "pending" || run.stage == "write",
        _ => false,
    }
}

fn kg_row_aggregate_status(run: &RunView) -> &'static str {
    if kg_has_failed_stage(run) {
        return "failed";
    }
    if KG_STAGE_KEYS
        .iter()
        .all(|k| stage_row_status(run, k) == "completed")
    {
        return "completed";
    }
    if KG_STAGE_KEYS
        .iter()
        .any(|k| stage_row_status(run, k) == "running")
    {
        return "running";
    }
    if KG_STAGE_KEYS
        .iter()
        .any(|k| stage_row_status(run, k) != "pending")
    {
        return "running";
    }
    "pending"
}

fn stage_glyph(status: &str, tick: u32) -> (String, Style) {
    match status {
        // Writer agent (and some backends) use "done" for finished sections.
        "completed" | "done" => (
            "✓ ".to_string(),
            Style::default()
                .fg(theme::SUCCESS)
                .add_modifier(Modifier::BOLD),
        ),
        "failed" | "error" => (
            "✗ ".to_string(),
            Style::default()
                .fg(theme::ERROR)
                .add_modifier(Modifier::BOLD),
        ),
        "running" => {
            let f = SPINNER_FRAMES[tick as usize % SPINNER_FRAMES.len()];
            (
                format!("{f} "),
                Style::default()
                    .fg(theme::ACCENT_BLUE)
                    .add_modifier(Modifier::BOLD),
            )
        }
        _ => (
            "· ".to_string(),
            Style::default().fg(theme::SUBTLE),
        ),
    }
}

fn block_heading_status<'a>(run: &'a RunView, block: &str) -> &'a str {
    match block {
        "scout" => stage_row_status(run, "scout"),
        "planning" => stage_row_status(run, "planning"),
        "research_parallel" => stage_row_status(run, "research_parallel"),
        "__kg__" => kg_row_aggregate_status(run),
        "outline" => stage_row_status(run, "outline"),
        "write" => stage_row_status(run, "write"),
        _ => "pending",
    }
}

fn block_display_title<'a>(block: &'a str) -> &'a str {
    match block {
        "scout" => "Scouting",
        "planning" => "Planning",
        "research_parallel" => "Parallel research",
        "__kg__" => "Knowledge graph",
        "outline" => "Outline",
        "write" => "Writing report",
        _ => block,
    }
}

fn push_stage_header(out: &mut Vec<Line<'static>>, run: &RunView, block: &str, tick: u32) {
    let status = block_heading_status(run, block);
    let st = if run_is_terminal(run) && status == "pending" {
        "completed"
    } else {
        status
    };
    let (glyph, gstyle) = stage_glyph(st, tick);
    let title = block_display_title(block).to_string();
    out.push(Line::from(vec![
        Span::styled(glyph, gstyle),
        Span::styled(
            title,
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        ),
    ]));
}

fn push_kg_sublines(out: &mut Vec<Line<'static>>, app: &App) {
    let stages = [
        ("kg_collect", "Collect"),
        ("kg_filter", "Filter"),
        ("kg_re", "NER / RE"),
        ("entity_resolution", "Resolve"),
    ];
    for (key, label) in stages {
        let processed =
            metric_value(app, &format!("{key}_processed")).unwrap_or_else(|| "—".to_string());
        let total = metric_value(app, &format!("{key}_total")).unwrap_or_else(|| "—".to_string());
        let elapsed = metric_value(app, &format!("{key}_elapsed_seconds"))
            .unwrap_or_else(|| "—".to_string());
        let status = stage_row_status(&app.dashboard.active_run, key);
        let (icon, accent) = match status {
            "completed" => ("✓", theme::SUCCESS),
            "running" => ("▶", theme::WARN),
            "failed" => ("✗", theme::ERROR),
            _ => ("·", theme::SUBTLE),
        };
        let row_fg = if status == "running" {
            theme::TEXT
        } else {
            theme::MUTED
        };
        out.push(Line::from(vec![
            Span::styled("      ", theme::SUBTLE),
            Span::styled(
                icon,
                Style::default().fg(accent).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                format!("{label:<12}"),
                Style::default().fg(row_fg),
            ),
            Span::styled(
                format!("{processed}/{total}"),
                Style::default().fg(theme::TEXT),
            ),
            Span::styled(
                format!("  {elapsed}s"),
                Style::default().fg(theme::SUBTLE),
            ),
        ]));
    }
}

fn push_angle_sublines(out: &mut Vec<Line<'static>>, run: &RunView) {
    for angle in &run.angles {
        let (icon, accent) = match angle.status.as_str() {
            "completed" => ("✓", theme::SUCCESS),
            "running" => ("▶", theme::WARN),
            "failed" | "error" => ("✗", theme::ERROR),
            _ => ("·", theme::SUBTLE),
        };
        let row_fg = if angle.status == "running" {
            theme::TEXT
        } else {
            theme::MUTED
        };
        let topic = truncate(&angle.topic, 52);
        out.push(Line::from(vec![
            Span::styled("      ", theme::SUBTLE),
            Span::styled(
                icon,
                Style::default().fg(accent).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(topic, Style::default().fg(row_fg)),
            Span::styled(
                format!(
                    "  · {}  · {}q {} src",
                    angle.stage, angle.queries_total, angle.context_ids_total
                ),
                Style::default().fg(theme::SUBTLE),
            ),
        ]));
    }
}

fn append_scout_details(out: &mut Vec<Line<'static>>, app: &App) {
    let q = metric_value(app, "query_count").unwrap_or_else(|| "0".to_string());
    let s = metric_value(app, "search_result_count").unwrap_or_else(|| "0".to_string());
    out.push(Line::from(vec![Span::styled(
        format!("      Searches run: {q}"),
        Style::default().fg(theme::MUTED),
    )]));
    out.push(Line::from(vec![Span::styled(
        format!("      Sources collected: {s}"),
        Style::default().fg(theme::MUTED),
    )]));
}

fn append_planning_details(out: &mut Vec<Line<'static>>, app: &App) {
    let n = metric_value(app, "research_angle_count").unwrap_or_else(|| "—".to_string());
    out.push(Line::from(vec![Span::styled(
        format!("      Research angles: {n}"),
        Style::default().fg(theme::MUTED),
    )]));
}

fn append_outline_details(out: &mut Vec<Line<'static>>, app: &App, run: &RunView) {
    let st = stage_row_status(run, "outline");
    if let Some(raw) = metric_value(app, "outline_section_titles_json") {
        if let Ok(titles) = serde_json::from_str::<Vec<String>>(&raw) {
            let done = st == "completed";
            for t in titles {
                let icon = if done { "✓" } else { "○" };
                let line = truncate(&t, 68);
                out.push(Line::from(vec![
                    Span::styled("      ", theme::SUBTLE),
                    Span::styled(
                        icon,
                        Style::default()
                            .fg(if done {
                                theme::SUCCESS
                            } else {
                                theme::WARN
                            })
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(line, Style::default().fg(theme::MUTED)),
                ]));
            }
            return;
        }
    }
    if st == "running" || st == "pending" {
        out.push(Line::from(vec![Span::styled(
            "      Structuring sections…",
            Style::default().fg(theme::MUTED),
        )]));
    }
}

fn append_write_details(out: &mut Vec<Line<'static>>, app: &App, run: &RunView, tick: u32) {
    let st = stage_row_status(run, "write");
    if let Some(raw) = metric_value(app, "write_sections_progress_json") {
        if let Ok(rows) = serde_json::from_str::<Vec<WriteSectionProgress>>(&raw) {
            for row in rows {
                let row_st = row.status.as_str();
                let (g, gs) = stage_glyph(row_st, tick);
                let title = truncate(&row.title, 58);
                out.push(Line::from(vec![
                    Span::styled("      ", theme::SUBTLE),
                    Span::styled(g, gs),
                    Span::styled(title, Style::default().fg(theme::MUTED)),
                ]));
            }
            return;
        }
    }
    let done = metric_value(app, "write_sections_completed").unwrap_or_else(|| "0".to_string());
    let tot = metric_value(app, "write_sections_total").unwrap_or_else(|| "—".to_string());
    if st == "running"
        || st == "completed"
        || st == "failed"
        || (st == "pending" && done != "0")
    {
        out.push(Line::from(vec![Span::styled(
            format!("      Sections: {done}/{tot}"),
            Style::default().fg(theme::MUTED),
        )]));
    }
}

fn append_activity_block(out: &mut Vec<Line<'static>>, app: &App, block: &str, tick: u32) {
    let run = &app.dashboard.active_run;
    if !activity_block_visible(run, block) {
        return;
    }
    push_stage_header(out, run, block, tick);
    match block {
        "scout" => append_scout_details(out, app),
        "planning" => append_planning_details(out, app),
        "research_parallel" => {
            if run.angles.is_empty() {
                out.push(Line::from(vec![Span::styled(
                    "      (Waiting for angle snapshots…)",
                    Style::default().fg(theme::SUBTLE),
                )]));
            } else {
                push_angle_sublines(out, run);
            }
        }
        "__kg__" => push_kg_sublines(out, app),
        "outline" => append_outline_details(out, app, run),
        "write" => append_write_details(out, app, run, tick),
        _ => {}
    }
    out.push(Line::from(""));
}

fn bottom_anchor_lines(mut lines: Vec<Line<'static>>, max_height: u16) -> Vec<Line<'static>> {
    let max = max_height as usize;
    if lines.len() > max {
        lines = lines[lines.len() - max..].to_vec();
    } else {
        let pad = max - lines.len();
        for _ in 0..pad {
            lines.insert(0, Line::from(""));
        }
    }
    lines
}

fn build_activity_content(app: &App, inner_height: u16) -> Vec<Line<'static>> {
    let run = &app.dashboard.active_run;
    let mut stack: Vec<Line<'static>> = Vec::new();
    if run_is_real(run) {
        for block in ACTIVITY_STACK {
            append_activity_block(&mut stack, app, block, app.ui_tick);
        }
    } else {
        stack.push(Line::from(vec![Span::styled(
            "  Ready — submit a prompt below.",
            Style::default().fg(theme::MUTED),
        )]));
        stack.push(Line::from(""));
    }
    bottom_anchor_lines(stack, inner_height)
}

fn render_activity_panel(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(theme::SUBTLE))
        .title(Span::styled(
            " Activity ",
            Style::default()
                .fg(theme::ACCENT_BLUE)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    let body = build_activity_content(app, inner.height);
    frame.render_widget(
        Paragraph::new(body)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

const HEADER_ROWS: u16 = 2;
const FOOTER_ROWS: u16 = 1;
/// Max visible rows for the draft text inside the Query panel (grows upward as you type).
const MAX_COMPOSER_DISPLAY_LINES: usize = 4;

fn query_input_block() -> Block<'static> {
    Block::default()
        .borders(Borders::TOP)
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(theme::ACCENT_BLUE))
        .title(Span::styled(
            " Query ",
            Style::default()
                .fg(theme::ACCENT_BLUE)
                .add_modifier(Modifier::BOLD),
        ))
        .style(Style::default().bg(theme::INPUT_BG))
}

/// Wrap draft into at most `max_lines` rows. First row reserves two cells for the `› ` prefix.
fn wrap_draft_chunks(draft: &str, inner_width: u16, max_lines: usize) -> Vec<String> {
    let max_lines = max_lines.clamp(1, 64);
    let w = inner_width.max(1) as usize;
    let first_max = w.saturating_sub(2).max(1);
    let cont_max = w.max(1);

    if draft.is_empty() {
        return vec![String::new()];
    }

    let mut rows: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut row_len = 0usize;
    let mut limit = first_max;

    for ch in draft.chars() {
        if rows.len() >= max_lines {
            break;
        }
        let ch_w = 1usize;
        if row_len + ch_w > limit && !current.is_empty() {
            rows.push(std::mem::take(&mut current));
            row_len = 0;
            limit = cont_max;
            if rows.len() >= max_lines {
                break;
            }
        }
        current.push(ch);
        row_len += ch_w;
    }

    if rows.len() < max_lines {
        rows.push(current);
    }
    rows.truncate(max_lines);
    if rows.is_empty() {
        vec![String::new()]
    } else {
        rows
    }
}

fn build_query_panel_lines(app: &App, full_width: u16) -> (u16, Vec<Line<'static>>) {
    let block = query_input_block();
    let inner_w = block
        .inner(Rect::new(0, 0, full_width, 255))
        .width
        .max(1);

    let chunks = wrap_draft_chunks(
        app.active_input_text(),
        inner_w,
        MAX_COMPOSER_DISPLAY_LINES,
    );

    let mut lines: Vec<Line<'static>> = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let is_last = i + 1 == chunks.len();
        let body: String = if is_last {
            format!("{chunk}▌")
        } else {
            chunk.clone()
        };
        if i == 0 {
            lines.push(Line::from(vec![
                Span::styled("› ", Style::default().fg(theme::ACCENT_PURPLE)),
                Span::styled(body, Style::default().fg(theme::TEXT)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default().fg(theme::SUBTLE)),
                Span::styled(body, Style::default().fg(theme::TEXT)),
            ]));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("Research prompt or ", Style::default().fg(theme::SUBTLE)),
        Span::styled("/batch …  /exit", Style::default().fg(theme::ACCENT_PURPLE)),
        Span::styled("  ·  ", Style::default().fg(theme::SUBTLE)),
        Span::styled("F1", Style::default().fg(theme::ACCENT_BLUE)),
        Span::styled(" help  ·  ", Style::default().fg(theme::SUBTLE)),
        Span::styled("Enter", Style::default().fg(theme::ACCENT_BLUE)),
        Span::styled(" submit", Style::default().fg(theme::MUTED)),
    ]));

    let inner_h = lines.len() as u16;
    let total_h = inner_h.saturating_add(1);
    (total_h, lines)
}

fn render(frame: &mut Frame, app: &App) {
    let full = frame.area();
    let (query_h, query_lines) = build_query_panel_lines(app, full.width);

    let activity_h = full
        .height
        .saturating_sub(HEADER_ROWS + query_h + FOOTER_ROWS);

    let x = full.x;
    let y = full.y;
    let w = full.width;

    render_header(frame, Rect::new(x, y, w, HEADER_ROWS), app);
    render_activity_panel(
        frame,
        Rect::new(x, y + HEADER_ROWS, w, activity_h),
        app,
    );
    render_input(
        frame,
        Rect::new(x, y + HEADER_ROWS + activity_h, w, query_h),
        query_lines,
    );
    render_footer(
        frame,
        Rect::new(x, y + HEADER_ROWS + activity_h + query_h, w, FOOTER_ROWS),
        app,
    );

    if app.show_help {
        render_help(frame);
    }
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let run = &app.dashboard.active_run;
    let stage_color = match run.stage.as_str() {
        "idle" | "none" | "queued" => theme::MUTED,
        _ => theme::ACCENT_BLUE,
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

    let brand = Line::from(vec![
        Span::styled("›› ", theme::ACCENT_BLUE),
        Span::styled(
            "MING",
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  runtime TUI", Style::default().fg(theme::MUTED)),
        Span::styled(
            format!("     {}", app.data_source.source_label()),
            Style::default().fg(theme::SUBTLE),
        ),
    ]);

    let meta = Line::from(vec![
        Span::styled("run ", Style::default().fg(theme::MUTED)),
        Span::styled(
            truncate(&run.run_id, 18),
            Style::default().fg(theme::TEXT),
        ),
        Span::styled("   stage ", Style::default().fg(theme::MUTED)),
        Span::styled(run.stage.clone(), Style::default().fg(stage_color)),
        Span::styled(
            format!("   {}s", run.elapsed_seconds),
            Style::default().fg(theme::TEXT),
        ),
        Span::styled("   ↻ ", Style::default().fg(theme::MUTED)),
        Span::styled(refresh_label, Style::default().fg(theme::MUTED)),
    ]);

    frame.render_widget(Paragraph::new(vec![brand, meta]), area);
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let spans = vec![
        Span::styled(&app.last_status, Style::default().fg(theme::TEXT)),
        Span::styled(
            "     /exit quit   F1 help   Esc close help",
            Style::default().fg(theme::SUBTLE),
        ),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_input(frame: &mut Frame, area: Rect, lines: Vec<Line<'static>>) {
    let block = query_input_block();
    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn render_help(frame: &mut Frame) {
    let area = centered_rect(52, 58, frame.area());
    frame.render_widget(Clear, area);

    let key = |k: &str| {
        Span::styled(
            format!("  {:<11}", k),
            Style::default()
                .fg(theme::ACCENT_BLUE)
                .add_modifier(Modifier::BOLD),
        )
    };
    let help_text = vec![
        Line::from(""),
        Line::from(vec![
            key("/exit"),
            Span::styled("quit (submit with Enter)", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("F1"),
            Span::styled("toggle this help", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("Esc"),
            Span::styled("close help", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("Enter"),
            Span::styled("submit prompt, /batch <path>, or /exit", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("Backspace"),
            Span::styled("delete character", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Batch file (/batch <path>):",
            Style::default().fg(theme::ACCENT_BLUE),
        )]),
        Line::from(vec![Span::styled(
            "  JSON array, or JSONL with one object per line. Each object:",
            Style::default().fg(theme::MUTED),
        )]),
        Line::from(vec![Span::styled(
            r#"  {"id": <integer>, "prompt": "<research query>"}"#,
            Style::default().fg(theme::TEXT),
        )]),
        Line::from(vec![Span::styled(
            "  Example array: [ {\"id\": 1, \"prompt\": \"…\"}, {\"id\": 2, \"prompt\": \"…\"} ]",
            Style::default().fg(theme::MUTED),
        )]),
        Line::from(vec![Span::styled(
            "  Activity panel: current stage + subtasks; status line keeps your last action.",
            Style::default().fg(theme::MUTED),
        )]),
    ];

    frame.render_widget(
        Paragraph::new(help_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Plain)
                    .border_style(Style::default().fg(theme::ACCENT_PURPLE))
                    .title(Span::styled(
                        " Help  ·  F1 / Esc ",
                        Style::default()
                            .fg(theme::TEXT)
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
