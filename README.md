# DL Lumbar Degenerative Diseases

Research and deployment pipeline for lumbar degenerative disease image classification based on RSNA 2024.

## Environment

- Package manager: `uv`
- Local development: macOS
- Training server: Linux + CUDA (RTX 5090)
- Client runtime: Windows (CUDA-first with CPU fallback)

## Quick Start

```bash
uv python install 3.11
uv sync --extra dev

# Data exploration and plots
uv run lumbar-cli eda --dataset-root ./rsna-2024-lumbar-spine-degenerative-classification

# Preprocess (study-level split)
uv run lumbar-cli preprocess --dataset-root ./rsna-2024-lumbar-spine-degenerative-classification

# Train one model
uv run lumbar-cli train --config configs/train/default.yaml

# Evaluate and compare
uv run lumbar-cli evaluate --run-dir artifacts/runs/<run_id>

# Streamlit platform
uv run streamlit run apps/streamlit/Home.py
```

## Sync to Linux Server

```bash
bash scripts/sync_to_server.sh <user>@<host> /path/to/project
```

## Health Check

```bash
uv run lumbar-cli healthcheck --target mac
uv run lumbar-cli healthcheck --target linux
uv run lumbar-cli healthcheck --target windows
```
