use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    prelude::*,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
};
use crate::app::App;
use crate::theme;
use crate::models::{RunView, WriteSectionProgress};
use crate::utils::truncate;

pub const HEADER_ROWS: u16 = 2;
pub const FOOTER_ROWS: u16 = 1;
pub const MAX_COMPOSER_DISPLAY_LINES: usize = 4;

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const ACTIVITY_STACK: &[&str] = &[
    "scout",
    "planning",
    "research_parallel",
    "__kg__",
    "outline",
    "write",
];

const KG_STAGE_KEYS: [&str; 4] = [
    "kg_collect",
    "kg_filter",
    "kg_re",
    "entity_resolution",
];

pub fn render(frame: &mut Frame, app: &App) {
    let full = frame.area();
    let (query_h, query_lines, batch_label) = build_query_panel_lines(app, full.width);

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
        batch_label,
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

fn render_input(frame: &mut Frame, area: Rect, lines: Vec<Line<'static>>, batch_label: Option<String>) {
    let block = query_input_block(batch_label);
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

fn build_activity_content(app: &App, inner_height: u16) -> Vec<Line<'static>> {
    let run = &app.dashboard.active_run;
    let mut stack: Vec<Line<'static>> = Vec::new();

    if let Some((done, total)) = batch_progress(app) {
        let pending = total.saturating_sub(done).saturating_sub(1); // exclude current in-flight
        stack.push(Line::from(vec![
            Span::styled("  Batch  ", Style::default().fg(theme::ACCENT_PURPLE).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{done}/{total}"), Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("  ✓ {done}  queued {pending}"),
                Style::default().fg(theme::MUTED),
            ),
        ]));
        stack.push(Line::from(""));
    }

    if run.is_real() {
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

fn activity_block_visible(run: &RunView, block: &str) -> bool {
    if !run.is_real() {
        return false;
    }
    if run.is_terminal() {
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

fn push_stage_header(out: &mut Vec<Line<'static>>, run: &RunView, block: &str, tick: u32) {
    let status = block_heading_status(run, block);
    let st = if run.is_terminal() && status == "pending" {
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

fn kg_has_failed_stage(run: &RunView) -> bool {
    KG_STAGE_KEYS.iter().any(|key| {
        run.stage_progress
            .iter()
            .find(|s| s.label == *key)
            .map(|s| s.status == "failed")
            .unwrap_or(false)
    })
}

fn stage_row_status<'a>(run: &'a RunView, key: &str) -> &'a str {
    run.stage_progress
        .iter()
        .find(|s| s.label == key)
        .map(|s| s.status.as_str())
        .unwrap_or("pending")
}

fn stage_glyph(status: &str, tick: u32) -> (String, Style) {
    match status {
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

fn build_query_panel_lines(app: &App, full_width: u16) -> (u16, Vec<Line<'static>>, Option<String>) {
    let batch_label = batch_progress(app).map(|(done, total)| format!("[{done}/{total}]"));
    let block = query_input_block(batch_label.clone());
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
    (total_h, lines, batch_label)
}

fn query_input_block(batch_label: Option<String>) -> Block<'static> {
    let title = match batch_label {
        Some(label) => format!(" Query {label} "),
        None => " Query ".to_string(),
    };
    Block::default()
        .borders(Borders::TOP)
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(theme::ACCENT_BLUE))
        .title(Span::styled(
            title,
            Style::default()
                .fg(theme::ACCENT_BLUE)
                .add_modifier(Modifier::BOLD),
        ))
        .style(Style::default().bg(theme::INPUT_BG))
}

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

fn batch_progress(app: &App) -> Option<(usize, usize)> {
    let b = app.batch_state.as_ref()?;
    Some((b.completed, b.total))
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
