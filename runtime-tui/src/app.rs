use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;
use anyhow::Result;

use crate::models::*;
use crate::data::{DataSource, load_batch_items_from_path};

// ── Command registry (sorted for display) ────────────────────────────
pub const COMMANDS: &[&str] = &[
    "/batch",
    "/exit",
    "/set-openrouter-key",
    "/set-tavily-key",
    "/workers",
];

/// TUI-side state for a running batch. Items are submitted to the backend one
/// at a time; the next item is only sent once the current run reaches a
/// terminal state, with a Redis flush in between.
pub struct BatchState {
    /// Items not yet submitted.
    pub pending: VecDeque<(String, String)>,
    /// How many items have fully finished (terminal state reached).
    pub completed: usize,
    /// Total items in the original batch.
    pub total: usize,
    /// True immediately after submission until we observe a new run_id,
    /// preventing the old completed run from triggering the next advance.
    pub waiting_for_new_run: bool,
}

/// Passive progress tracker for concurrent batch (service-managed execution).
pub struct BatchProgress {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
}

pub struct App {
    pub data_source: DataSource,
    pub dashboard: DashboardData,
    pub focus: Focus,
    pub show_help: bool,
    pub direct_query_draft: String,
    pub last_status: String,
    pub last_refresh: Option<Instant>,
    pub last_run_id: Option<String>,
    pub last_active_job_id: Option<String>,
    /// Wall-clock instant when the TUI first observed the current run_id.
    /// Used to drive a live elapsed timer independent of backend metrics.
    pub run_started_at: Option<Instant>,
    /// Active batch, if one is in progress.
    pub batch_state: Option<BatchState>,
    pub ui_tick: u32,
    pub exit_requested: bool,

    // ── Multi-run worker selection ───────────────────────────────────
    /// Index into `dashboard.active_runs` for the detail view.
    pub selected_worker: usize,
    /// Passive batch progress tracker (for concurrent batch mode).
    pub batch_progress: Option<BatchProgress>,

    // ── Auto-completion state ────────────────────────────────────────
    /// Filtered commands matching the current `/` prefix.
    pub completion_matches: Vec<&'static str>,
    /// Index into `completion_matches` for Tab cycling (`None` = no selection).
    pub completion_selected: Option<usize>,

    // ── Concurrency ──────────────────────────────────────────────────
    /// Number of concurrent workers for batch processing (1–5).
    pub max_concurrent_workers: usize,

    // ── Async key validation ─────────────────────────────────────────
    /// Receiver for a pending API-key validation result.
    /// Tuple: (env_var_name, Ok(key_value) | Err(error_message)).
    pub pending_key_validation: Option<std::sync::mpsc::Receiver<(String, Result<String, String>)>>,
}

impl App {
    pub fn new(
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
            last_active_job_id: None,
            run_started_at: None,
            batch_state: None,
            ui_tick: 0,
            exit_requested: false,
            selected_worker: 0,
            batch_progress: None,
            completion_matches: Vec::new(),
            completion_selected: None,
            max_concurrent_workers: 1,
            pending_key_validation: None,
        }
    }

    pub fn refresh(&mut self) {
        match self.data_source.load(self.last_run_id.as_deref()) {
            Ok((dashboard, direct_draft, batch_draft)) => {
                let new_active_job_id = dashboard.queue.active.as_ref().map(|j| j.job_id.clone());

                if new_active_job_id.is_some() && new_active_job_id != self.last_active_job_id {
                    self.last_run_id = None;
                }
                self.last_active_job_id = new_active_job_id;

                let new_id = &dashboard.active_run.run_id;
                if new_id != "none" && !new_id.is_empty() {
                    if self.last_run_id.as_deref() != Some(new_id.as_str()) {
                        // First time we see this run_id — start the local clock.
                        self.run_started_at = Some(Instant::now());
                    }
                    self.last_run_id = Some(new_id.clone());
                }

                // If we're waiting for a batch to start (sequential or concurrent),
                // don't overwrite the "queued" placeholder with "idle" from Redis.
                let waiting_for_start = self.batch_state.as_ref().map_or(false, |b| b.waiting_for_new_run)
                    || self.batch_progress.is_some();
                let backend_idle = dashboard.active_run.run_id == "none"
                    || dashboard.active_run.status == "idle";
                if waiting_for_start && backend_idle && dashboard.active_runs.is_empty() {
                    // Keep the old dashboard (shows "queued"), only update queue/events.
                    self.dashboard.queue = dashboard.queue;
                    self.dashboard.recent_events = dashboard.recent_events;
                } else {
                    self.dashboard = dashboard;
                }
                let _ = (direct_draft, batch_draft);
                self.last_refresh = Some(Instant::now());
                // Clamp worker selection.
                if !self.dashboard.active_runs.is_empty() {
                    if self.selected_worker >= self.dashboard.active_runs.len() {
                        self.selected_worker = 0;
                    }
                }
                // Update concurrent batch progress from queue state.
                if let Some(ref mut bp) = self.batch_progress {
                    bp.completed = self.dashboard.queue.all_completed_count;
                    bp.failed = self.dashboard.queue.all_failed_count;
                    if bp.completed + bp.failed >= bp.total && self.dashboard.active_runs.is_empty() {
                        self.last_status = format!(
                            "Batch complete: {} done, {} failed of {} total.",
                            bp.completed, bp.failed, bp.total
                        );
                        self.batch_progress = None;
                    }
                }
            }
            Err(err) => {
                self.last_status = format!("Refresh failed: {err}");
            }
        }
        self.tick_batch();
    }

    /// Advance a TUI-side batch: when the current run reaches a terminal state,
    /// flush Redis and submit the next queued item.
    fn tick_batch(&mut self) {
        let mut batch = match self.batch_state.take() {
            Some(b) => b,
            None => return,
        };

        // Wait until a new run_id appears after the last submission.
        if batch.waiting_for_new_run {
            if self.last_run_id.is_some() {
                batch.waiting_for_new_run = false;
            }
            self.batch_state = Some(batch);
            return;
        }

        let run = &self.dashboard.active_run;
        let is_done = matches!(
            run.status.as_str(),
            "completed" | "done" | "success" | "finished" | "failed" | "error"
        );
        if !is_done {
            self.batch_state = Some(batch);
            return;
        }

        // Current job is terminal — advance.
        batch.completed += 1;
        let total = batch.total;
        let completed = batch.completed;

        if let Some((item_id, prompt)) = batch.pending.pop_front() {
            // Flush Redis state before submitting the next item.
            if let Err(e) = self.data_source.flush_namespace_state() {
                self.last_status = format!("Redis flush failed: {e:#}");
                // Abort the batch rather than risk dirty state.
                return;
            }
            match self.data_source.submit_run_query(&prompt, Some(&item_id)) {
                Ok(_) => {
                    self.reset_dashboard_for_new_command();
                    batch.waiting_for_new_run = true;
                    self.last_status =
                        format!("Batch: {}/{total} submitted", completed + 1);
                    self.batch_state = Some(batch);
                }
                Err(e) => {
                    self.last_status = format!("Batch submit failed: {e:#}");
                    // batch_state stays None — batch aborted.
                }
            }
        } else {
            self.last_status =
                format!("Batch complete: {completed}/{total} items processed.");
            // batch_state stays None.
        }
    }

    pub fn reset_dashboard_for_new_command(&mut self) {
        self.last_run_id = None;
        self.run_started_at = None;
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

    pub fn finish_run_query_submit(&mut self, prompt: &str) -> Result<String> {
        let msg = self.data_source.submit_run_query(prompt, None)?;
        self.batch_state = None;
        self.reset_dashboard_for_new_command();
        self.direct_query_draft.clear();
        Ok(msg)
    }

    /// Load a batch file, submit the first item immediately, and keep the
    /// remaining items in `batch_state` for sequential TUI-side dispatch.
    /// Items whose `reports/id_{id}.md` already exists are skipped.
    pub fn finish_batch_submit(&mut self, path: PathBuf) -> Result<String> {
        let all_items = load_batch_items_from_path(&path)?;
        let total_loaded = all_items.len();

        let pending_items: Vec<(String, String)> = all_items
            .into_iter()
            .filter(|(id, _)| {
                let report = PathBuf::from("reports").join(format!("id_{id}.md"));
                !report.exists()
            })
            .collect();

        let skipped = total_loaded - pending_items.len();
        if pending_items.is_empty() {
            return Ok(format!(
                "All {total_loaded} items already have reports; nothing to do."
            ));
        }

        let mut items: VecDeque<(String, String)> = pending_items.into_iter().collect();
        let total = items.len();

        let (first_id, first_prompt) = items.pop_front().unwrap(); // guaranteed ≥ 1
        let msg = self.data_source.submit_run_query(&first_prompt, Some(&first_id))?;

        self.batch_state = Some(BatchState {
            pending: items,
            completed: 0,
            total,
            waiting_for_new_run: true,
        });
        self.reset_dashboard_for_new_command();
        self.direct_query_draft.clear();
        let skip_note = if skipped > 0 {
            format!(", {skipped} skipped")
        } else {
            String::new()
        };
        Ok(format!("Batch started ({total} items{skip_note}): {msg}"))
    }

    /// Submit an entire batch as a single RUN_BATCH command for concurrent
    /// server-side execution. The TUI becomes a passive observer.
    pub fn finish_concurrent_batch_submit(&mut self, path: PathBuf) -> Result<String> {
        let all_items = load_batch_items_from_path(&path)?;
        let total_loaded = all_items.len();

        let pending_items: Vec<(String, String)> = all_items
            .into_iter()
            .filter(|(id, _)| {
                let report = PathBuf::from("reports").join(format!("id_{id}.md"));
                !report.exists()
            })
            .collect();

        let skipped = total_loaded - pending_items.len();
        if pending_items.is_empty() {
            return Ok(format!(
                "All {total_loaded} items already have reports; nothing to do."
            ));
        }

        let total = pending_items.len();
        let workers = self.max_concurrent_workers;
        let msg = self.data_source.submit_run_batch(&pending_items, workers)?;

        self.batch_state = None; // Not using client-side batch state.
        self.batch_progress = Some(BatchProgress {
            total,
            completed: 0,
            failed: 0,
        });
        self.reset_dashboard_for_new_command();
        self.direct_query_draft.clear();
        let skip_note = if skipped > 0 {
            format!(", {skipped} skipped")
        } else {
            String::new()
        };
        Ok(format!(
            "Concurrent batch started ({total} items, {workers} workers{skip_note}): {msg}"
        ))
    }

    pub fn handle_slash_command(&mut self, line: &str) {
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
                if self.max_concurrent_workers > 1 {
                    match self.finish_concurrent_batch_submit(PathBuf::from(path)) {
                        Ok(m) => self.last_status = m,
                        Err(e) => self.last_status = format!("{e:#}"),
                    }
                } else {
                    match self.finish_batch_submit(PathBuf::from(path)) {
                        Ok(m) => self.last_status = m,
                        Err(e) => self.last_status = format!("{e:#}"),
                    }
                }
            }
            "/set-tavily-key" => {
                let key = parts[1..].join(" ");
                if key.is_empty() {
                    self.last_status = "Usage: /set-tavily-key <key>".to_string();
                    return;
                }
                self.validate_and_set_key(
                    "TAVILY_API_KEY",
                    "https://api.tavily.com/usage",
                    key,
                );
            }
            "/set-openrouter-key" => {
                let key = parts[1..].join(" ");
                if key.is_empty() {
                    self.last_status = "Usage: /set-openrouter-key <key>".to_string();
                    return;
                }
                self.validate_and_set_key(
                    "OPENROUTER_API_KEY",
                    "https://openrouter.ai/api/v1/auth/key",
                    key,
                );
            }
            "/workers" => {
                match parts.get(1).and_then(|s| s.parse::<usize>().ok()) {
                    Some(n) if (1..=5).contains(&n) => {
                        self.max_concurrent_workers = n;
                        self.last_status = format!("Concurrent workers set to {n}.");
                    }
                    None if parts.len() == 1 => {
                        self.last_status =
                            format!("Workers: {} (use /workers N to change, 1-5)", self.max_concurrent_workers);
                    }
                    _ => {
                        self.last_status = "Usage: /workers N  (1-5)".to_string();
                    }
                }
            }
            _ => {
                self.last_status =
                    "Unknown command. Available: /batch /exit /set-tavily-key /set-openrouter-key /workers"
                        .to_string();
            }
        }
    }

    pub fn submit_current(&mut self) {
        let input = self.active_input_text().to_string();
        let trimmed = input.trim().to_string();
        if trimmed.is_empty() {
            self.last_status = "Nothing to submit.".to_string();
            return;
        }
        if trimmed.starts_with('/') {
            self.handle_slash_command(&trimmed);
            self.direct_query_draft.clear();
            return;
        }
        match self.finish_run_query_submit(&trimmed) {
            Ok(message) => self.last_status = message,
            Err(err) => self.last_status = format!("Submit failed: {err:#}"),
        }
    }

    pub fn next_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Input if self.dashboard.active_runs.len() > 1 => Focus::WorkerList,
            _ => Focus::Input,
        };
    }

    pub fn prev_focus(&mut self) {
        self.focus = match self.focus {
            Focus::WorkerList => Focus::Input,
            _ => Focus::Input,
        };
    }

    pub fn select_worker_next(&mut self) {
        let len = self.dashboard.active_runs.len();
        if len > 0 {
            self.selected_worker = (self.selected_worker + 1) % len;
        }
    }

    pub fn select_worker_prev(&mut self) {
        let len = self.dashboard.active_runs.len();
        if len > 0 {
            self.selected_worker = if self.selected_worker == 0 { len - 1 } else { self.selected_worker - 1 };
        }
    }

    pub fn active_input_text(&self) -> &str {
        &self.direct_query_draft
    }

    pub fn handle_char(&mut self, ch: char) {
        if self.focus != Focus::Input {
            return;
        }
        self.direct_query_draft.push(ch);
        self.update_completions();
    }

    pub fn backspace(&mut self) {
        if self.focus != Focus::Input {
            return;
        }
        self.direct_query_draft.pop();
        self.update_completions();
    }

    // ── Auto-completion ──────────────────────────────────────────────

    /// Recompute `completion_matches` from the current draft.
    pub fn update_completions(&mut self) {
        let draft = self.direct_query_draft.trim_start();
        if draft.starts_with('/') && !draft.contains(' ') {
            let lower = draft.to_ascii_lowercase();
            self.completion_matches = COMMANDS
                .iter()
                .copied()
                .filter(|cmd| cmd.starts_with(&lower))
                .collect();
            // Reset selection if the previous selection is out of bounds.
            if let Some(idx) = self.completion_selected {
                if idx >= self.completion_matches.len() {
                    self.completion_selected = None;
                }
            }
        } else {
            self.completion_matches.clear();
            self.completion_selected = None;
        }
    }

    /// Cycle through completion matches (called on Tab).
    pub fn cycle_completion(&mut self) {
        if self.completion_matches.is_empty() {
            return;
        }
        let next = match self.completion_selected {
            None => 0,
            Some(i) => (i + 1) % self.completion_matches.len(),
        };
        self.completion_selected = Some(next);
        self.direct_query_draft = self.completion_matches[next].to_string();
    }

    // ── API key validation ───────────────────────────────────────────

    fn validate_and_set_key(&mut self, env_key: &str, api_url: &str, key: String) {
        self.last_status = format!("Validating {env_key}...");
        let (tx, rx) = std::sync::mpsc::channel();
        let env_key_owned = env_key.to_string();
        let url = api_url.to_string();
        std::thread::spawn(move || {
            let result = (|| -> Result<String, String> {
                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(10))
                    .build()
                    .map_err(|e| format!("HTTP client error: {e}"))?;
                let resp = client
                    .get(&url)
                    .header("Authorization", format!("Bearer {key}"))
                    .send()
                    .map_err(|e| format!("Request failed: {e}"))?;
                if resp.status().is_success() {
                    Ok(key)
                } else {
                    Err(format!("Validation failed (HTTP {})", resp.status()))
                }
            })();
            let _ = tx.send((env_key_owned, result));
        });
        self.pending_key_validation = Some(rx);
    }

    /// Poll the pending validation receiver. Called from the main event loop.
    pub fn poll_key_validation(&mut self) {
        let rx = match self.pending_key_validation.as_ref() {
            Some(rx) => rx,
            None => return,
        };
        match rx.try_recv() {
            Ok((env_key, result)) => {
                match result {
                    Ok(value) => {
                        match crate::utils::write_env_key(&env_key, &value) {
                            Ok(_) => self.last_status = format!("{env_key} validated and saved."),
                            Err(e) => self.last_status = format!("Key valid but .env write failed: {e}"),
                        }
                    }
                    Err(msg) => self.last_status = msg,
                }
                self.pending_key_validation = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {} // still pending
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.last_status = "Validation thread disconnected.".to_string();
                self.pending_key_validation = None;
            }
        }
    }
}
