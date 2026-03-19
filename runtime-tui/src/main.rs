use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    DefaultTerminal,
    layout::{Constraint, Direction, Layout, Rect},
    prelude::*,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Gauge, List, ListItem, Paragraph, Row, Table, Tabs, Wrap},
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
    Queue,
    Angles,
    Input,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputMode {
    DirectQuery,
    BatchJson,
}

impl InputMode {
    fn title(self) -> &'static str {
        match self {
            Self::DirectQuery => "Direct Query",
            Self::BatchJson => "Sequential Batch",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct DashboardData {
    queue: QueueView,
    active_run: RunView,
    recent_events: Vec<EventEntry>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct QueueView {
    queued: Vec<JobEntry>,
    active: Option<JobEntry>,
    completed: Vec<JobEntry>,
}

#[derive(Clone, Debug, Default, Deserialize)]
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
    prompt: String,
    status: String,
    stage: String,
    elapsed_seconds: u64,
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
    fn load(&self) -> Result<(DashboardData, Option<String>, Option<String>)> {
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
                Ok((load_live_dashboard(&mut con, namespace)?, None, None))
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
                    InputMode::BatchJson => {
                        let items: JsonValue = serde_json::from_str(input.trim())
                            .context("batch JSON must be a JSON array of {id, prompt}")?;
                        if !items.is_array() {
                            return Err(anyhow!("batch JSON must be an array"));
                        }
                        json!({
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
                                "items": items
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
    queue_index: usize,
    angle_index: usize,
    show_help: bool,
    direct_query_draft: String,
    batch_json_draft: String,
    last_status: String,
    last_refresh: Option<Instant>,
}

impl App {
    fn new(
        data_source: DataSource,
        dashboard: DashboardData,
        direct_query_draft: String,
        batch_json_draft: String,
    ) -> Self {
        Self {
            data_source,
            dashboard,
            focus: Focus::Queue,
            input_mode: InputMode::DirectQuery,
            queue_index: 0,
            angle_index: 0,
            show_help: false,
            direct_query_draft,
            batch_json_draft,
            last_status: "Ready.".to_string(),
            last_refresh: Some(Instant::now()),
        }
    }

    fn refresh(&mut self) {
        match self.data_source.load() {
            Ok((dashboard, direct_draft, batch_draft)) => {
                self.dashboard = dashboard;
                if let Some(value) = direct_draft {
                    self.direct_query_draft = value;
                }
                if let Some(value) = batch_draft {
                    self.batch_json_draft = value;
                }
                self.queue_index = self
                    .queue_index
                    .min(self.queue_items().len().saturating_sub(1));
                self.angle_index = self
                    .angle_index
                    .min(self.dashboard.active_run.angles.len().saturating_sub(1));
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
                match self.input_mode {
                    InputMode::DirectQuery => self.direct_query_draft.clear(),
                    InputMode::BatchJson => self.batch_json_draft.clear(),
                }
            }
            Err(err) => {
                self.last_status = format!("Submit failed: {err}");
            }
        }
    }

    fn next_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Queue => Focus::Angles,
            Focus::Angles => Focus::Input,
            Focus::Input => Focus::Queue,
        };
    }

    fn prev_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Queue => Focus::Input,
            Focus::Angles => Focus::Queue,
            Focus::Input => Focus::Angles,
        };
    }

    fn selected_angle(&self) -> Option<&AngleView> {
        self.dashboard.active_run.angles.get(self.angle_index)
    }

    fn queue_items(&self) -> Vec<&JobEntry> {
        let mut items = Vec::new();
        if let Some(active) = self.dashboard.queue.active.as_ref() {
            items.push(active);
        }
        items.extend(self.dashboard.queue.queued.iter());
        items.extend(self.dashboard.queue.completed.iter().take(4));
        items
    }

    fn move_down(&mut self) {
        match self.focus {
            Focus::Queue => {
                let len = self.queue_items().len();
                if len > 0 {
                    self.queue_index = (self.queue_index + 1).min(len - 1);
                }
            }
            Focus::Angles => {
                let len = self.dashboard.active_run.angles.len();
                if len > 0 {
                    self.angle_index = (self.angle_index + 1).min(len - 1);
                }
            }
            Focus::Input => {}
        }
    }

    fn move_up(&mut self) {
        match self.focus {
            Focus::Queue => self.queue_index = self.queue_index.saturating_sub(1),
            Focus::Angles => self.angle_index = self.angle_index.saturating_sub(1),
            Focus::Input => {}
        }
    }

    fn toggle_input_mode(&mut self) {
        self.input_mode = match self.input_mode {
            InputMode::DirectQuery => InputMode::BatchJson,
            InputMode::BatchJson => InputMode::DirectQuery,
        };
    }

    fn active_input_text(&self) -> &str {
        match self.input_mode {
            InputMode::DirectQuery => &self.direct_query_draft,
            InputMode::BatchJson => &self.batch_json_draft,
        }
    }

    fn handle_char(&mut self, ch: char) {
        if self.focus != Focus::Input {
            return;
        }
        match self.input_mode {
            InputMode::DirectQuery => self.direct_query_draft.push(ch),
            InputMode::BatchJson => self.batch_json_draft.push(ch),
        }
    }

    fn backspace(&mut self) {
        if self.focus != Focus::Input {
            return;
        }
        match self.input_mode {
            InputMode::DirectQuery => {
                self.direct_query_draft.pop();
            }
            InputMode::BatchJson => {
                self.batch_json_draft.pop();
            }
        }
    }

    fn preview_submission(&self) -> String {
        match self.input_mode {
            InputMode::DirectQuery => {
                let prompt = self.direct_query_draft.trim();
                if prompt.is_empty() {
                    "Direct query command preview will appear here.".to_string()
                } else {
                    serde_json::to_string_pretty(&json!({
                        "type": "run_query",
                        "payload": { "prompt": prompt }
                    }))
                    .unwrap_or_else(|_| "Failed to render preview.".to_string())
                }
            }
            InputMode::BatchJson => {
                let trimmed = self.batch_json_draft.trim();
                if trimmed.is_empty() {
                    "Sequential batch command preview will appear here.".to_string()
                } else {
                    match serde_json::from_str::<JsonValue>(trimmed) {
                        Ok(value) => serde_json::to_string_pretty(&json!({
                            "type": "run_batch",
                            "payload": {
                                "mode": "sequential",
                                "items": value
                            }
                        }))
                        .unwrap_or_else(|_| "Failed to format preview.".to_string()),
                        Err(err) => format!("Invalid JSON draft: {err}"),
                    }
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let data_source = parse_data_source_from_args()?;
    let (dashboard, direct_query_draft, batch_json_draft) = data_source.load()?;
    let direct_query_draft = direct_query_draft.unwrap_or_default();
    let batch_json_draft = batch_json_draft.unwrap_or_else(default_batch_template);

    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let terminal = ratatui::init();

    let result = run_app(
        terminal,
        App::new(data_source, dashboard, direct_query_draft, batch_json_draft),
    );

    ratatui::restore();
    disable_raw_mode().ok();
    execute!(std::io::stdout(), LeaveAlternateScreen).ok();

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

            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('s') {
                app.submit_current();
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
                KeyCode::Down | KeyCode::Char('j') => app.move_down(),
                KeyCode::Up | KeyCode::Char('k') => app.move_up(),
                KeyCode::Char('m') => app.toggle_input_mode(),
                KeyCode::Char('?') => app.show_help = !app.show_help,
                KeyCode::Backspace => app.backspace(),
                KeyCode::Enter => {
                    if app.focus == Focus::Input {
                        app.handle_char('\n');
                    }
                }
                KeyCode::Char(ch) => app.handle_char(ch),
                _ => {}
            }
        }
    }
}

fn load_live_dashboard(con: &mut redis::Connection, namespace: &str) -> Result<DashboardData> {
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

    let active_run = load_active_run_view(con, namespace, active_job.as_ref())?;
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
) -> Result<RunView> {
    let run_wire = if let Some(active_job) = active_job {
        let job_key = format!("{namespace}:jobs:{}", active_job.job_id);
        let job_wire = get_json::<JobSnapshotWire>(con, &job_key)?;
        if let Some(run_id) = job_wire.and_then(|job| job.run_id) {
            get_json::<RunSnapshotWire>(con, &format!("{namespace}:runs:{run_id}"))?
        } else {
            latest_run_snapshot(con, namespace)?
        }
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
    for event in events {
        if let Some(stage) = event.stage {
            stage_status.insert(stage, event.status);
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

    Ok(STAGE_ORDER
        .iter()
        .enumerate()
        .map(|(index, stage)| {
            let status = if let Some(value) = stage_status.get(*stage) {
                value.clone()
            } else if *stage == current_stage {
                current_stage_status.clone()
            } else if index < current_index {
                "completed".to_string()
            } else if index > current_index {
                "pending".to_string()
            } else {
                "running".to_string()
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

fn default_batch_template() -> String {
    "[\n  {\n    \"id\": \"b1\",\n    \"prompt\": \"First prompt\"\n  },\n  {\n    \"id\": \"b2\",\n    \"prompt\": \"Second prompt\"\n  }\n]\n".to_string()
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
            Constraint::Length(4),
            Constraint::Min(20),
            Constraint::Length(10),
        ])
        .split(frame.area());

    render_header(frame, root[0], app);

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(28),
            Constraint::Percentage(42),
            Constraint::Percentage(30),
        ])
        .split(root[1]);

    render_queue(frame, middle[0], app);
    render_run_center(frame, middle[1], app);
    render_input(frame, middle[2], app);
    render_events(frame, root[2], app);

    if app.show_help {
        render_help(frame);
    }
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Length(2)])
        .split(area);

    let top = vec![
        Span::styled(
            "Ming Runtime TUI",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            format!(
                "source={} run={} stage={} elapsed={}s",
                app.data_source.source_label(),
                app.dashboard.active_run.run_id,
                app.dashboard.active_run.stage,
                app.dashboard.active_run.elapsed_seconds
            ),
            Style::default().fg(Color::Gray),
        ),
    ];
    frame.render_widget(
        Paragraph::new(Line::from(top)).block(
            Block::default()
                .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
                .title("Status"),
        ),
        chunks[0],
    );
    frame.render_widget(
        Paragraph::new(app.last_status.clone())
            .block(Block::default().borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM))
            .wrap(Wrap { trim: true }),
        chunks[1],
    );
}

fn render_queue(frame: &mut Frame, area: Rect, app: &App) {
    let items = app
        .queue_items()
        .into_iter()
        .enumerate()
        .map(|(index, job)| {
            let prefix = if index == app.queue_index && app.focus == Focus::Queue {
                "▸ "
            } else {
                "  "
            };
            let style = match job.status.as_str() {
                "running" => Style::default().fg(Color::Yellow),
                "completed" => Style::default().fg(Color::Green),
                "failed" => Style::default().fg(Color::Red),
                _ => Style::default().fg(Color::White),
            };
            ListItem::new(Line::from(vec![
                Span::raw(prefix),
                Span::styled(format!("[{}] ", job.kind), Style::default().fg(Color::Cyan)),
                Span::styled(job.prompt_label.clone(), style),
                Span::styled(
                    format!("  {} / {}", job.job_id, job.command_id),
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
        })
        .collect::<Vec<_>>();

    let list = List::new(items).block(titled_block("Queue", app.focus == Focus::Queue));
    frame.render_widget(list, area);
}

fn render_run_center(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(11),
            Constraint::Length(7),
            Constraint::Min(8),
        ])
        .split(area);

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Run Progress"))
        .gauge_style(Style::default().fg(Color::Cyan))
        .ratio(app.dashboard.active_run.progress_ratio)
        .label(format!(
            "{} | {}",
            app.dashboard.active_run.status, app.dashboard.active_run.prompt
        ));
    frame.render_widget(gauge, chunks[0]);

    let stage_rows = app
        .dashboard
        .active_run
        .stage_progress
        .iter()
        .map(|stage| {
            let status_style = match stage.status.as_str() {
                "completed" => Style::default().fg(Color::Green),
                "running" => Style::default().fg(Color::Yellow),
                "failed" => Style::default().fg(Color::Red),
                _ => Style::default().fg(Color::Gray),
            };
            Row::new(vec![
                stage.label.clone(),
                stage.status.clone(),
                format!("{}s", stage.duration_seconds),
            ])
            .style(status_style)
        })
        .collect::<Vec<_>>();
    let stage_table = Table::new(
        stage_rows,
        [
            Constraint::Percentage(50),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ],
    )
    .header(
        Row::new(vec!["Stage", "Status", "Duration"])
            .style(Style::default().add_modifier(Modifier::BOLD)),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Stage Timeline"),
    );
    frame.render_widget(stage_table, chunks[1]);

    let metric_lines = app
        .dashboard
        .active_run
        .metrics
        .iter()
        .map(|metric| {
            Line::from(vec![
                Span::styled(
                    format!("{:<18}", metric.label),
                    Style::default().fg(Color::Gray),
                ),
                Span::styled(metric.value.clone(), Style::default().fg(Color::Cyan)),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(metric_lines)
            .block(Block::default().borders(Borders::ALL).title("Run Metrics"))
            .wrap(Wrap { trim: true }),
        chunks[2],
    );

    let lower = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(chunks[3]);

    let angle_items = app
        .dashboard
        .active_run
        .angles
        .iter()
        .enumerate()
        .map(|(index, angle)| {
            let prefix = if index == app.angle_index && app.focus == Focus::Angles {
                "▸ "
            } else {
                "  "
            };
            ListItem::new(Line::from(vec![
                Span::raw(prefix),
                Span::styled(angle.angle_id.clone(), Style::default().fg(Color::Cyan)),
                Span::raw(" "),
                Span::raw(angle.topic.clone()),
            ]))
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(angle_items).block(titled_block("Angles", app.focus == Focus::Angles)),
        lower[0],
    );

    let angle_detail = if let Some(angle) = app.selected_angle() {
        vec![
            Line::from(vec![Span::styled(
                angle.topic.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            )]),
            Line::from(format!("status: {}", angle.status)),
            Line::from(format!("stage: {}", angle.stage)),
            Line::from(format!("iteration: {}", angle.iteration)),
            Line::from(format!("queries: {}", angle.queries_total)),
            Line::from(format!("contexts: {}", angle.context_ids_total)),
        ]
    } else {
        vec![Line::from("No angle selected.")]
    };
    frame.render_widget(
        Paragraph::new(angle_detail)
            .block(Block::default().borders(Borders::ALL).title("Angle Detail"))
            .wrap(Wrap { trim: true }),
        lower[1],
    );
}

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(10),
        ])
        .split(area);

    let titles = ["Direct Query", "Sequential Batch"]
        .iter()
        .map(|title| Line::from(*title))
        .collect::<Vec<_>>();
    let selected = match app.input_mode {
        InputMode::DirectQuery => 0,
        InputMode::BatchJson => 1,
    };
    frame.render_widget(
        Tabs::new(titles)
            .select(selected)
            .block(titled_block("Command", app.focus == Focus::Input))
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        chunks[0],
    );

    frame.render_widget(
        Paragraph::new(app.active_input_text())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(app.input_mode.title()),
            )
            .wrap(Wrap { trim: false }),
        chunks[1],
    );

    frame.render_widget(
        Paragraph::new(app.preview_submission())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Submission Preview (Ctrl-S to send)"),
            )
            .wrap(Wrap { trim: false }),
        chunks[2],
    );
}

fn render_events(frame: &mut Frame, area: Rect, app: &App) {
    let lines = app
        .dashboard
        .recent_events
        .iter()
        .map(|entry| {
            let color = match entry.level.as_str() {
                "warning" => Color::Yellow,
                "error" => Color::Red,
                _ => Color::Gray,
            };
            Line::from(vec![
                Span::styled(
                    format!("{} ", entry.ts),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("[{}] ", entry.component),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(entry.message.clone(), Style::default().fg(color)),
                Span::raw(
                    entry
                        .stage
                        .as_ref()
                        .map(|stage| format!(" ({stage})"))
                        .unwrap_or_default(),
                ),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Recent Events"),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_help(frame: &mut Frame) {
    let area = centered_rect(60, 48, frame.area());
    frame.render_widget(Clear, area);
    let help = Paragraph::new(vec![
        Line::from("q: quit"),
        Line::from("r: refresh now"),
        Line::from("tab / shift-tab: move focus"),
        Line::from("j / k or arrows: move selection"),
        Line::from("m: switch command mode"),
        Line::from("Ctrl-S: submit current draft to Redis"),
        Line::from("?: toggle help"),
        Line::from("type when Input is focused"),
    ])
    .block(Block::default().borders(Borders::ALL).title("Help"))
    .wrap(Wrap { trim: true });
    frame.render_widget(help, area);
}

fn titled_block<'a>(title: &'a str, focused: bool) -> Block<'a> {
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(border_style)
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
