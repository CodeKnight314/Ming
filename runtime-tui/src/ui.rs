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

fn status_icon(status: &str, ui_tick: u32) -> &'static str {
    match status {
        "completed" | "success" | "done" => "✓",
        "failed" | "error" => "✗",
        "running" => SPINNER_FRAMES[ui_tick as usize % SPINNER_FRAMES.len()],
        _ => "·",
    }
}

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

    // Completion popup (rendered above the input panel).
    if !app.completion_matches.is_empty() {
        let input_area = Rect::new(x, y + HEADER_ROWS + activity_h, w, query_h);
        render_completion_popup(frame, input_area, app);
    }

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

    let active_count = app.dashboard.active_runs.len();
    let meta = if active_count > 1 {
        Line::from(vec![
            Span::styled(
                format!("workers {active_count} active"),
                Style::default().fg(theme::ACCENT_BLUE).add_modifier(Modifier::BOLD),
            ),
            Span::styled("   Tab to focus worker list   ", Style::default().fg(theme::MUTED)),
            Span::styled("↻ ", Style::default().fg(theme::MUTED)),
            Span::styled(refresh_label, Style::default().fg(theme::MUTED)),
        ])
    } else {
        Line::from(vec![
            Span::styled("run ", Style::default().fg(theme::MUTED)),
            Span::styled(
                truncate(&run.run_id, 18),
                Style::default().fg(theme::TEXT),
            ),
            Span::styled("   stage ", Style::default().fg(theme::MUTED)),
            Span::styled(run.stage.clone(), Style::default().fg(stage_color)),
            Span::styled(
                format!(
                    "   {}",
                    format_elapsed(
                        app.run_started_at
                            .map(|t| t.elapsed().as_secs())
                            .unwrap_or(run.elapsed_seconds)
                    )
                ),
                Style::default().fg(theme::TEXT),
            ),
            Span::styled("   ↻ ", Style::default().fg(theme::MUTED)),
            Span::styled(refresh_label, Style::default().fg(theme::MUTED)),
        ])
    };

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

fn render_completion_popup(frame: &mut Frame, input_area: Rect, app: &App) {
    let matches = &app.completion_matches;
    if matches.is_empty() {
        return;
    }
    let max_show: u16 = 6;
    let height = (matches.len() as u16).min(max_show) + 2; // +2 for borders
    let width = matches.iter().map(|s| s.len()).max().unwrap_or(10) as u16 + 4;
    let width = width.min(input_area.width);

    // Position above the input area.
    let y = input_area.y.saturating_sub(height);
    let popup = Rect::new(input_area.x + 2, y, width, height);

    frame.render_widget(Clear, popup);

    let lines: Vec<Line> = matches
        .iter()
        .enumerate()
        .map(|(i, cmd)| {
            let selected = app.completion_selected == Some(i);
            let style = if selected {
                Style::default()
                    .fg(theme::ACCENT_PURPLE)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::TEXT)
            };
            let prefix = if selected { "› " } else { "  " };
            Line::from(Span::styled(format!("{prefix}{cmd}"), style))
        })
        .collect();

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Plain)
                .border_style(Style::default().fg(theme::SUBTLE)),
        ),
        popup,
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
        Line::from(vec![Span::styled(
            "  Keyboard",
            Style::default().fg(theme::ACCENT_BLUE).add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![
            key("Tab"),
            Span::styled("cycle command completions", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("Enter"),
            Span::styled("submit prompt or command", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("F1"),
            Span::styled("toggle this help", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("Esc"),
            Span::styled("close help", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Commands",
            Style::default().fg(theme::ACCENT_BLUE).add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![
            key("/batch"),
            Span::styled("<path>  run batch JSONL/JSON file", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("/workers"),
            Span::styled("N       set concurrent batch workers (1-5)", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("/set-tavily-key"),
            Span::styled("<key>  validate & save Tavily API key", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("/set-openrouter-key"),
            Span::styled("<key>  validate & save OpenRouter key", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            key("/exit"),
            Span::styled("quit", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Batch file format:",
            Style::default().fg(theme::ACCENT_BLUE).add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![Span::styled(
            "  JSON array or JSONL with one object per line. Each object:",
            Style::default().fg(theme::MUTED),
        )]),
        Line::from(vec![Span::styled(
            r#"  {"id": <integer>, "prompt": "<research query>"}"#,
            Style::default().fg(theme::TEXT),
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
    // Multi-run concurrent mode: split into worker table + selected detail.
    if app.dashboard.active_runs.len() > 1 {
        render_multi_run_activity(frame, area, app);
        return;
    }

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

fn render_multi_run_activity(frame: &mut Frame, area: Rect, app: &App) {
    let runs = &app.dashboard.active_runs;
    // Worker table: 1 header + 1 per worker + 1 batch progress + 2 borders = min ~6 rows.
    let table_h = (runs.len() as u16 + 4).min(area.height.saturating_sub(6));
    let detail_h = area.height.saturating_sub(table_h);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(table_h), Constraint::Min(detail_h)])
        .split(area);

    // ── Worker Table ─────────────────────────────────────────────────
    let focus_worker = app.focus == crate::models::Focus::WorkerList;
    let border_color = if focus_worker { theme::ACCENT_BLUE } else { theme::SUBTLE };
    let table_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(
            " Workers ",
            Style::default().fg(theme::ACCENT_BLUE).add_modifier(Modifier::BOLD),
        ));
    let _table_inner = table_block.inner(chunks[0]);

    let mut lines: Vec<Line<'static>> = Vec::new();
    // Header line.
    lines.push(Line::from(vec![
        Span::styled(
            format!("{:<5} {:<30} {:<20} {:>7}", "Slot", "Query", "Stage", "Elapsed"),
            Style::default().fg(theme::MUTED).add_modifier(Modifier::BOLD),
        ),
    ]));

    for (i, run) in runs.iter().enumerate() {
        let selected = i == app.selected_worker;
        let prompt_label = truncate(&run.prompt, 28);
        let stage = truncate(&run.stage, 18);
        let elapsed = if run.elapsed_seconds >= 60 {
            format!("{}m{}s", run.elapsed_seconds / 60, run.elapsed_seconds % 60)
        } else {
            format!("{}s", run.elapsed_seconds)
        };
        let prefix = if selected { "› " } else { "  " };
        let style = if selected {
            Style::default().fg(theme::ACCENT_BLUE).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::TEXT)
        };
        lines.push(Line::from(Span::styled(
            format!("{prefix}w{i:<3} {prompt_label:<30} {stage:<20} {elapsed:>7}"),
            style,
        )));
    }

    // Batch progress line.
    if let Some(ref bp) = app.batch_progress {
        let pct = if bp.total > 0 { (bp.completed * 100) / bp.total } else { 0 };
        lines.push(Line::from(Span::styled(
            format!(
                "  Batch: {}/{} done  {} failed  ({}%)",
                bp.completed, bp.total, bp.failed, pct
            ),
            Style::default().fg(theme::MUTED),
        )));
    }

    frame.render_widget(
        Paragraph::new(lines).block(table_block).wrap(Wrap { trim: false }),
        chunks[0],
    );

    // ── Selected Worker Detail ───────────────────────────────────────
    let detail_run = runs.get(app.selected_worker).unwrap_or(&app.dashboard.active_run);
    let detail_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(theme::SUBTLE))
        .title(Span::styled(
            format!(" Worker w{} Detail ", app.selected_worker),
            Style::default().fg(theme::ACCENT_BLUE).add_modifier(Modifier::BOLD),
        ));
    let detail_inner = detail_block.inner(chunks[1]);
    let body = build_single_run_detail(detail_run, app.ui_tick, detail_inner.height);
    frame.render_widget(
        Paragraph::new(body).block(detail_block).wrap(Wrap { trim: false }),
        chunks[1],
    );
}

/// Build activity lines for a single RunView (reused by multi-run detail panel).
fn build_single_run_detail(run: &RunView, ui_tick: u32, inner_height: u16) -> Vec<Line<'static>> {
    let mut stack: Vec<Line<'static>> = Vec::new();

    // Stage progress.
    for sp in &run.stage_progress {
        let icon = status_icon(&sp.status, ui_tick);
        let style = match sp.status.as_str() {
            "completed" | "success" | "done" => Style::default().fg(theme::SUCCESS),
            "running" => Style::default().fg(theme::WARN),
            _ => Style::default().fg(theme::SUBTLE),
        };
        stack.push(Line::from(vec![
            Span::styled(format!("  {icon} "), style.add_modifier(Modifier::BOLD)),
            Span::styled(sp.label.clone(), Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD)),
        ]));
    }

    // Angles (if in research stage).
    if !run.angles.is_empty() {
        stack.push(Line::from(""));
        for angle in &run.angles {
            let icon = status_icon(&angle.status, ui_tick);
            let style = match angle.status.as_str() {
                "completed" | "success" | "done" => Style::default().fg(theme::SUCCESS),
                "running" => Style::default().fg(theme::WARN),
                _ => Style::default().fg(theme::SUBTLE),
            };
            stack.push(Line::from(vec![
                Span::styled(format!("  {icon} "), style.add_modifier(Modifier::BOLD)),
                Span::styled(
                    format!(
                        "{} [q:{} s:{}]",
                        truncate(&angle.topic, 40),
                        angle.queries_total,
                        angle.context_ids_total,
                    ),
                    Style::default().fg(theme::TEXT),
                ),
            ]));
        }
    }

    // Bottom-align: pad top with empty lines.
    let content_h = stack.len() as u16;
    if content_h < inner_height {
        let pad = (inner_height - content_h) as usize;
        let mut padded = vec![Line::from(""); pad];
        padded.append(&mut stack);
        stack = padded;
    }
    stack
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
        append_token_stats(&mut stack, app);
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
                let has_substage = row_st == "running"
                    && row.substage.as_ref().is_some_and(|s| !s.is_empty());
                let title_width = if has_substage { 44 } else { 58 };
                let title = truncate(&row.title, title_width);
                let mut spans = vec![
                    Span::styled("      ", theme::SUBTLE),
                    Span::styled(g, gs),
                    Span::styled(title, Style::default().fg(theme::MUTED)),
                ];
                if has_substage {
                    let tag = row.substage.as_deref().unwrap_or("");
                    spans.push(Span::styled(
                        format!("  {tag}"),
                        Style::default().fg(theme::SUBTLE),
                    ));
                }
                out.push(Line::from(spans));
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

fn format_token_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn append_token_stats(out: &mut Vec<Line<'static>>, app: &App) {
    let raw = match metric_value(app, "token_stats_json") {
        Some(r) => r,
        None => return,
    };

    #[derive(serde::Deserialize)]
    struct ModelUsage {
        input_tokens: u64,
        output_tokens: u64,
        call_count: u64,
    }

    #[derive(serde::Deserialize)]
    struct TokenStats {
        #[serde(default)]
        models: std::collections::BTreeMap<String, ModelUsage>,
        #[serde(default)]
        total_input_tokens: u64,
        #[serde(default)]
        total_output_tokens: u64,
        #[serde(default)]
        total_llm_calls: u64,
        #[serde(default)]
        total_web_queries: u64,
    }

    let stats: TokenStats = match serde_json::from_str(&raw) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Only show if there's data
    if stats.total_llm_calls == 0 && stats.total_web_queries == 0 {
        return;
    }

    out.push(Line::from(""));
    out.push(Line::from(vec![
        Span::styled("  ─── ", Style::default().fg(theme::SUBTLE)),
        Span::styled("Token Usage", Style::default().fg(theme::MUTED).add_modifier(Modifier::BOLD)),
        Span::styled(" ───", Style::default().fg(theme::SUBTLE)),
    ]));

    // Summary line: total in/out + web queries
    let summary = format!(
        "  in: {}  out: {}  calls: {}  web: {}",
        format_token_count(stats.total_input_tokens),
        format_token_count(stats.total_output_tokens),
        stats.total_llm_calls,
        stats.total_web_queries,
    );
    out.push(Line::from(vec![
        Span::styled(summary, Style::default().fg(theme::MUTED)),
    ]));

    // Per-model breakdown
    for (name, usage) in &stats.models {
        let short_name = name.rsplit('/').next().unwrap_or(name);
        let line = format!(
            "    {:<24} in: {:>7}  out: {:>7}  ×{}",
            truncate(short_name, 24),
            format_token_count(usage.input_tokens),
            format_token_count(usage.output_tokens),
            usage.call_count,
        );
        out.push(Line::from(vec![
            Span::styled(line, Style::default().fg(theme::SUBTLE)),
        ]));
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

fn format_elapsed(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else {
        let m = secs / 60;
        let s = secs % 60;
        if s == 0 {
            format!("{m}m")
        } else {
            format!("{m}m{s:02}s")
        }
    }
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
