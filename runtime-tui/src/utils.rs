use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::Result;
use serde_json::Value as JsonValue;

pub fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars - 3).collect();
    format!("{}...", truncated)
}

pub fn unix_millis() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis())
}

pub fn iso_like_now() -> Result<String> {
    // Basic ISO-ish timestamp without adding a full chrono dependency if not needed,
    // though the project might already have it. (Checking main.rs imports... it doesn't).
    Ok(format!("{:?}", SystemTime::now()))
}

pub fn json_value_label(value: &JsonValue) -> String {
    match value {
        JsonValue::String(s) => s.clone(),
        JsonValue::Number(n) => n.to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Null => "null".to_string(),
        _ => value.to_string(),
    }
}

pub fn normalize_event_level(status: &str) -> String {
    match status.to_lowercase().as_str() {
        "error" | "failed" | "fatal" => "error".to_string(),
        "warn" | "warning" => "warn".to_string(),
        "info" | "success" | "completed" => "info".to_string(),
        _ => "debug".to_string(),
    }
}
