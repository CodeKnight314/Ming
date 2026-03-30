mod theme;
mod models;
mod utils;
mod data;
mod app;
mod ui;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind,
};
use crossterm::execute;
use ratatui::DefaultTerminal;

use crate::app::App;
use crate::data::DataSource;
use crate::ui::render;

fn main() -> Result<()> {
    let data_source = parse_data_source_from_args()?;
    let (dashboard, direct_query_draft, batch_json_draft) = data_source.load(None)?;
    let direct_query_draft = direct_query_draft.unwrap_or_default();
    let _ = batch_json_draft;

    let terminal = ratatui::init();
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
                    if app.focus == crate::models::Focus::Input {
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
