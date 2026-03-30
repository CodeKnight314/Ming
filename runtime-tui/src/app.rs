use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;
use anyhow::Result;

use crate::models::*;
use crate::data::{DataSource, load_batch_items_from_path};

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
                self.dashboard = dashboard;
                let _ = (direct_draft, batch_draft);
                self.last_refresh = Some(Instant::now());
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
        self.focus = Focus::Input;
    }

    pub fn prev_focus(&mut self) {
        self.focus = Focus::Input;
    }

    pub fn active_input_text(&self) -> &str {
        &self.direct_query_draft
    }

    pub fn handle_char(&mut self, ch: char) {
        if self.focus != Focus::Input {
            return;
        }
        self.direct_query_draft.push(ch);
    }

    pub fn backspace(&mut self) {
        if self.focus != Focus::Input {
            return;
        }
        self.direct_query_draft.pop();
    }
}
