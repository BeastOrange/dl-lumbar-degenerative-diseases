# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A deep learning research pipeline for lumbar degenerative disease image classification, based on the RSNA 2024 Kaggle competition. Three-axis (T1/T2/STIR) DICOM spine images are classified into Normal/Mild, Moderate, or Severe severity. The platform supports training, evaluation, and a Streamlit UI.

## Common Commands

```bash
# Install dependencies
uv sync --extra dev

# Data exploration and plots
uv run lumbar-cli eda --dataset-root ./rsna-2024-lumbar-spine-degenerative-classification

# Preprocess (study-level split, generates manifests in artifacts/processed)
uv run lumbar-cli preprocess --dataset-root ./rsna-2024-lumbar-spine-degenerative-classification

# Train one model (results in artifacts/runs/<run-id>/)
uv run lumbar-cli train --config configs/train/default.yaml

# Evaluate one run and generate figures
uv run lumbar-cli evaluate --run-dir artifacts/runs/<run-id>

# Compare all runs and build ranking
uv run lumbar-cli compare --runs-root artifacts/runs --primary-metric val_macro_f1

# Health check
uv run lumbar-cli healthcheck --target mac

# Run Streamlit platform
uv run streamlit run apps/streamlit/Home.py

# Run tests
uv run pytest tests/ -q
# Single test
uv run pytest tests/train/test_trainer.py -v
```

## Architecture

```
src/dl_lumbar_dd/
├── cli.py                  # Entry point; routes subcommands to handlers
├── config.py               # load_yaml() — single YAML loader used everywhere
├── constants.py            # TARGET_COLUMNS (25 conditions), SERIES_TYPES, label mappings
├── data/
│   ├── commands.py        # run_eda(), run_preprocess() — top-level data workflow
│   ├── dicom.py            # build_three_view_tensor() — reads middle DICOM slice per view
│   ├── ingest.py           # load_rsna_tables(), build_study_index() — reads 3 RSNA CSVs
│   ├── reporting.py        # generate_eda_reports(), generate_preprocess_reports()
│   └── splits.py           # build_split_manifests() — StratifiedKFold study-level splits
├── models/
│   ├── backbones.py        # BACKBONE_FACTORIES: 5 model architectures + helper classes
│   ├── blocks.py           # CBAM, DenseReuseProjection, FeatureVolume3D, HierarchicalFeatureFusion, MultiViewFusionAdapter
│   └── registry.py         # create_model(), available_models(), LumbarModel class
├── train/
│   ├── commands.py         # run_training() — loads YAML → builds dataloaders → trains → persists artifacts
│   ├── config.py           # TrainingConfig (dataclass), TrainingResult (dataclass)
│   ├── data.py             # LumbarStudyDataset, build_dataloaders(), prepare_bundle_and_manifests()
│   ├── metrics.py          # classification_metrics(), metric_row() — F1/precision/recall/kappa
│   └── trainer.py          # Trainer class — PyTorch loop with AMP, early stopping, focal loss
├── eval/
│   ├── commands.py         # run_evaluation(), run_comparison()
│   ├── comparison.py       # build_ranking_table(), save_ranking_table()
│   └── metrics.py          # eval_metrics() — sklearn wrappers
├── visualization/
│   ├── __init__.py
│   └── plots.py            # save_training_history, save_confusion_matrix, save_multiclass_roc
└── utils/
    └── io.py               # ensure_dir(), write_json()

apps/streamlit/             # Streamlit UI (6 pages)
```

## Key Data Flows

**Preprocessing:** RSNA CSVs → `load_rsna_tables()` → `build_study_index()` → `build_split_manifests()` → `save_split_manifests()` → CSV manifests in `artifacts/processed/`

**Training:** YAML config → `TrainingConfig` → `build_dataloaders()` → `LumbarStudyDataset` (loads middle DICOM slice per view, stacks 3 views as channels) → `Trainer.fit()` → saves `best.ckpt`, `metrics.csv`, `history.json`, `predictions.csv` under `artifacts/runs/<run-id>/`

**Model architecture:** LumbarModel wraps a backbone encoder + grayscale-to-RGB adapter + ImageNet normalizer + (optional) MultiViewFusionAdapter for multi-view weighting + linear head. Five backbones registered in BACKBONE_FACTORIES.

## Label Encoding

`SEVERITY_TO_INDEX` maps: Normal/Mild → 0, Moderate → 1, Severe → 2

## Config Structure

Train configs use a flat YAML structure (not nested). The CLI `train` command reads fields directly with `.get()` calls. See `configs/train/default.yaml` for the canonical schema.

## Important Paths

- Dataset: `rsna-2024-lumbar-spine-degenerative-classification/` (train.csv, train_series_descriptions.csv, train_label_coordinates.csv, train_images/)
- Artifacts: `artifacts/` (runs/, processed/, metadata/, figures/)
- Reports: `reports/figures/`

## Remote Training

Training runs on a Linux server via `scripts/sync_to_server.sh` and `scripts/server_train.sh`. The server uses the same codebase with local data paths. Run locally first to verify; sync and train remotely for GPU.