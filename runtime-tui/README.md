# Runtime TUI Mock

This crate is a standalone Rust frontend for the Ming runtime control plane.

It supports both:

- live Redis mode for the real Python runtime service
- mock mode for layout-only iteration

## Run

Start Redis:

```bash
./startup.sh
```

Start the Python runtime service:

```bash
python -m ming.runtime.service
```

Run the live TUI:

```bash
cargo run --manifest-path runtime-tui/Cargo.toml -- --redis-url redis://127.0.0.1:6379/0 --namespace runtime
```

Run in mock mode:

```bash
cargo run --manifest-path runtime-tui/Cargo.toml -- --mock runtime-tui/mock_state.json
```

## Controls

- `q`: quit
- `r`: refresh now
- `tab` / `shift-tab`: change focus
- `j` / `k` or arrow keys: move selection
- `m`: switch between direct query and sequential batch mock forms
- `enter`: submit the current direct query or batch JSON to Redis
- `?`: toggle help

## Mock Goals

- left pane: queue visibility
- center pane: active run timeline, KG pipeline progress, and passive research angle statuses
- right pane: direct query and sequential batch command composition
- bottom pane: recent runtime events
