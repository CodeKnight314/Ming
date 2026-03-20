<p align="center" style="margin: 0;">
  <span style="display: inline-flex; align-items: center; gap: 10px;">
    <picture>
      <img src="assets/ming.png" alt="Ming-DeepResearch logo" height="60" style="border-radius: 8px;" />
    </picture>
    <span style="font-size: 2.2rem; font-weight: 700;">
      <u>Ming DeepResearch</u>
    </span>
  </span>
</p>

## Overview
Ming DeepResearch (明 — "clarity") is a multi-agent deep research system that leverages knowledge graphs to structure and manage greater volume of web-sourced content at competitive precision. Designed for flexibility, Ming supports a range of report depths while delivering high-quality results at feasible cost. For reference, a standard ~13,000-word report, for example, costs only $0.68 (including web search credits) with Ming DeepResearch while out-performing proprietary services.

Ming offers the following research modes to support varying research needs:
- 🌒 Crescent — concise, focused brief
- 🌓 Quarter — standard depth report
- 🌕 Lunar — exhaustive deep dive

For evaluation purposes, we submit standard depth (Quarter) reports to DeepResearch Bench for review. We use the following models to support Ming DeepResearch: 

| Purpose               | Model                         |
|-----------------------|-------------------------------|
| Scout                 | qwen/qwen3.5-flash-02-23      |
| Research              | qwen/qwen3.5-flash-02-23      |
| Entity Extraction     | google/gemma-3n-e4b-it        |
| Outline               | qwen/qwen3.5-plus-02-15       |
| Writing               | qwen/qwen3.5-plus-02-15       | 


## Results

## Installation

- **Python** 3.11 or newer
- **Docker** — required to create `./startup.sh` to run Redis locally 
- **Rust** — needed for the runtime TUI

```bash 
cd /path/to/Ming
uv venv 
.venv/source/activate
pip install -e .
```

## Usage

Ming DeepResearch offers two modes of user interface to utilize the deepresearch agent: A Terminal User Interface with complete observability and a Command Line Interface for direct query.

### 1. Running Terminal User Interface

Run the following to start hosting the Redis Docker containers and python backend.
```bash 
bash startup.sh
```
In a separate terminal, activate the Rust-based Terminal User Interface:
```bash
cargo run --manifest-path runtime-tui/Cargo.toml -- --redis-url redis://127.0.0.1:6379/0 --namespace runtime
```

### 2. Running Command Line Interface

Start Redis first (same as the TUI flow), e.g. `bash startup.sh`, then run the orchestrator from the repo root.

**Single query** — runs one research job and prints elapsed time (minutes):

```bash
python -m ming.orchestrator --query "Write me a deep research report on the history of artificial intelligence"
```

Optional **`--config path/to/config.json`** (default: `config.json`) selects which Redis endpoints are cleared before the run.

**Batch (DeepResearch Bench)** — read `deepresearch-bench/query_data/query.jsonl` (one JSON object per line with `id` and `prompt`). Each row runs sequentially. Reports are written as `id_<id>.md`. If `id_<id>.md` already exists under the submission directory, that row is skipped. Research Redis (context, queries store, KG) is flushed before **each** new query so runs do not share state.

Default submission directory is `../submission` relative to the JSONL file (e.g. `query_data/query.jsonl` → `deepresearch-bench/submission/`).

```bash
python -m ming.orchestrator --jsonl deepresearch-bench/query_data/query.jsonl
```

Override the output folder:

```bash
python -m ming.orchestrator --jsonl deepresearch-bench/query_data/query.jsonl \
  --submission-dir deepresearch-bench/submission
```

## Limitations 

## Acknowledgements
Special thanks to the following repositories for sharing their inspiring designs and insights!
- [Open Deep Research](https://github.com/langchain-ai/open_deep_research)
- [Onyx Deep Research](https://github.com/onyx-dot-app/onyx)
- [Nvidia AI-Q](https://github.com/NVIDIA-AI-Blueprints/aiq/tree/drb1)