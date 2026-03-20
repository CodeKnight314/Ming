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

- `/exit` (typed in the query bar, then Enter): quit
- `enter`: submit the query bar (normal text = `run_query` to Redis)
- `F1`: toggle help overlay, `Esc`: close help (when open). Plain `?` types into the query (no shortcut conflict).

### Query bar

Everything looks good now. We want to add a /stats command to inform how much users spent on the session? We need to collect per model input/output token, how many search credits used then output it in the activity 


- Any line **not** starting with `/` is sent as a single `run_query`.
- **`/batch <path>`** submits `run_batch`. The file is either a JSON **array** or **JSONL**; each item must look like `{"id": <integer>, "prompt": "<research query>"}` (see **F1** help for the exact shape).

The status line is **not** overwritten on each auto-refresh (errors still update the line). The composer is **not** reloaded from disk on refresh, so mock mode does not fight your typing.

Mock mode (`--mock`) previews submissions without Redis.

## Layout

- **Top:** header (brand, run id, raw stage key, elapsed, last refresh).
- **Middle:** **Activity** fills remaining space; stages list in **pipeline order** (earliest at the top of the list, latest at the bottom). When the panel is taller than the content, the block is padded from above so it sits toward the **bottom** of the Activity area.
- **Bottom:** query panel sits directly above the status footer; its height fits the content (no empty rows inside). The draft text wraps to up to **4** lines, then a spacer and the hint row; the Activity area gets all space in between.

## Mock Goals

- Exercise layout: stage stack, KG / angles, wrapped query bar, and Redis-backed (or mock) command submission.
