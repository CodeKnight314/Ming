use std::collections::BTreeMap;
use serde::Deserialize;
use serde_json::Value as JsonValue;

pub const STAGE_ORDER: [&str; 9] = [
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
pub enum Focus {
    Input,
    WorkerList,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DashboardData {
    #[allow(dead_code)]
    pub queue: QueueView,
    pub active_run: RunView,
    /// Multiple active runs when concurrent batch is in progress.
    #[serde(default)]
    pub active_runs: Vec<RunView>,
    #[allow(dead_code)]
    pub recent_events: Vec<EventEntry>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[allow(dead_code)]
pub struct QueueView {
    pub queued: Vec<JobEntry>,
    pub active: Option<JobEntry>,
    pub completed: Vec<JobEntry>,
    pub failed: Vec<JobEntry>,
    // Raw counts from the queue snapshot (not capped at display limit)
    #[serde(default)]
    pub all_queued_count: usize,
    #[serde(default)]
    pub all_completed_count: usize,
    #[serde(default)]
    pub all_failed_count: usize,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[allow(dead_code)]
pub struct JobEntry {
    pub job_id: String,
    pub kind: String,
    pub prompt_label: String,
    pub command_id: String,
    pub status: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct RunView {
    pub run_id: String,
    #[allow(dead_code)]
    pub prompt: String,
    #[allow(dead_code)]
    pub status: String,
    pub stage: String,
    pub elapsed_seconds: u64,
    #[allow(dead_code)]
    pub progress_ratio: f64,
    pub stage_progress: Vec<StageProgress>,
    pub angles: Vec<AngleView>,
    pub metrics: Vec<MetricEntry>,
}

impl RunView {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status.as_str(),
            "completed" | "done" | "success" | "finished"
        )
    }

    pub fn is_real(&self) -> bool {
        !self.run_id.is_empty() && self.run_id != "none"
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct StageProgress {
    pub label: String,
    pub status: String,
    #[allow(dead_code)]
    pub duration_seconds: u64,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct AngleView {
    pub angle_id: String,
    pub topic: String,
    pub stage: String,
    #[allow(dead_code)]
    pub iteration: u32,
    pub queries_total: u32,
    pub context_ids_total: u32,
    pub status: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct MetricEntry {
    pub label: String,
    pub value: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[allow(dead_code)]
pub struct EventEntry {
    pub ts: String,
    pub component: String,
    pub message: String,
    pub stage: Option<String>,
    pub level: String,
}

#[derive(Debug, Deserialize)]
pub struct WriteSectionProgress {
    pub title: String,
    pub status: String,
    /// Reasoning loop substage: "gathering", "drafting", "critiquing", "revising"
    #[serde(default)]
    pub substage: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct MockFileState {
    pub queue: QueueView,
    pub active_run: RunView,
    #[serde(default)]
    pub active_runs: Vec<RunView>,
    pub recent_events: Vec<EventEntry>,
    pub direct_query_draft: String,
    pub batch_json_draft: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct QueueSnapshotWire {
    pub queued_job_ids: Vec<String>,
    pub active_job_id: Option<String>,
    #[serde(default)]
    pub active_job_ids: Vec<String>,
    pub completed_job_ids: Vec<String>,
    pub failed_job_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct JobSnapshotWire {
    pub job_id: String,
    pub command_id: String,
    #[serde(rename = "type")]
    pub job_type: String,
    pub status: String,
    pub payload: JsonValue,
    pub run_id: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct RunSnapshotWire {
    pub run_id: String,
    pub prompt: String,
    pub status: String,
    pub stage: Option<String>,
    pub stage_status: Option<String>,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub finished_at: Option<String>,
    pub metrics: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct AngleSnapshotWire {
    pub angle_id: String,
    pub topic: String,
    pub stage: Option<String>,
    pub iteration: u32,
    pub queries_total: u32,
    pub context_ids_total: u32,
    pub status: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct RuntimeEventPayload {
    pub ts: String,
    pub component: String,
    pub message: String,
    pub stage: Option<String>,
    pub status: String,
    #[serde(rename = "metrics")]
    pub _metrics: BTreeMap<String, JsonValue>,
}
