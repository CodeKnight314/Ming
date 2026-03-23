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

<div align="center">
  <img src="assets/TUI.png" alt="Ming DeepResearch runtime TUI" style="max-width: min(100%, 960px); height: auto; border-radius: 8px;" />
</div>

## Overview
Ming DeepResearch (明 — "clarity") is a multi-agent deep research system that leverages knowledge graphs to structure and manage greater volume of web-sourced content at competitive precision. Designed for cost-effectiveness, Ming delivers high-quality deep research reports at feasible cost. For reference, a standard 13,000~15,000 word report costs Ming DeepResearch only $0.72~ (including web search credits) with Ming DeepResearch while out-performing robust proprietary services. Furthermore, Ming can be configured for more flexible deployment through varied configurations.

We use the following models to support Ming DeepResearch: 

| Purpose               | Model                         |
|-----------------------|-------------------------------|
| Scout                 | qwen/qwen3.5-flash-02-23      |
| Research              | qwen/qwen3.5-flash-02-23      |
| Entity Extraction     | google/gemma-3n-e4b-it        |
| Outline               | qwen/qwen3.5-plus-02-15       |
| Writing               | qwen/qwen3.5-plus-02-15       | 


## Results
Below are our results on [DeepResearch Bench](https://muset-ai-deepresearch-bench-leaderboard.hf.space/#):

Total evaluation cost (including reruning specific tasks): $76.32

## Installation

- **Python** 3.11 or newer
- **Docker** — required to create `./startup.sh` to run Redis locally 
- **Rust** — needed for the runtime TUI

You can optionally set up a virtual environment before running the installation script
```bash 
cd /path/to/Ming
uv venv 
.venv/source/activate
```
Run the following in terminal to complete installtion
```bash
pip install -e .
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
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

```bash
python -m ming.orchestrator --query "Write me a deep research report on the history of artificial intelligence"
```

**Batch (DeepResearch Bench)** — read `deepresearch-bench/query_data/query.jsonl` (one JSON object per line with `id` and `prompt`). Each row runs sequentially. Reports are written as `id_<id>.md`. If `id_<id>.md` already exists under the submission directory, that row is skipped. Research Redis (context, queries store, KG) is flushed before **each** new query so runs do not share state.

Default submission directory is `../submission` relative to the JSONL file (e.g. `query_data/query.jsonl` → `deepresearch-bench/submission/`).

```bash
python -m ming.orchestrator --jsonl deepresearch-bench/query_data/query.jsonl
```

## Limitations 

## Acknowledgements
Special thanks to the following repositories for sharing their inspiring designs and insights!
- [Open Deep Research](https://github.com/langchain-ai/open_deep_research)
- [Onyx Deep Research](https://github.com/onyx-dot-app/onyx)
- [Nvidia AI-Q](https://github.com/NVIDIA-AI-Blueprints/aiq/tree/drb1)