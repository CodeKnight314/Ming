use std::fs;
use std::path::PathBuf;
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

/// Write or update a `KEY="value"` entry in `.env`. Creates the file if absent.
/// Uses atomic rename for safety.
pub fn write_env_key(key: &str, value: &str) -> Result<()> {
    let env_path = PathBuf::from(".env");
    let contents = fs::read_to_string(&env_path).unwrap_or_default();
    let prefix = format!("{key}=");
    let new_line = format!("{key}=\"{value}\"");

    let mut found = false;
    let mut lines: Vec<String> = contents
        .lines()
        .map(|line| {
            if line.starts_with(&prefix) {
                found = true;
                new_line.clone()
            } else {
                line.to_string()
            }
        })
        .collect();

    if !found {
        lines.push(new_line);
    }

    let tmp_path = env_path.with_extension("tmp");
    fs::write(&tmp_path, lines.join("\n") + "\n")?;
    fs::rename(&tmp_path, &env_path)?;
    Ok(())
}
