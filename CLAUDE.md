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
├── cli.py                  # Entry point (lumbar-cli); routes subcommands to handlers
├── config.py               # load_yaml() — single YAML loader used everywhere
├── constants.py            # TARGET_COLUMNS (25 conditions), SERIES_TYPES, label mappings
├── data/                   # DICOM loading, EDA, preprocessing, study-level splits
├── models/                 # Backbone encoders, custom blocks, LumbarModel registry
├── train/                  # Training loop, dataset, config dataclasses, metrics
├── eval/                   # Evaluation, cross-run comparison and ranking
├── visualization/          # Confusion matrix, ROC curves, training history plots
└── utils/                  # File I/O helpers

apps/streamlit/             # Streamlit UI (6 pages: Home, Data Explorer, Data Analysis,
                            #   Preprocessing QA, Training Dashboard, Model Comparison,
                            #   Inference & Explainability)
```

## Model Architecture

**LumbarModel** (in `models/registry.py`) pipeline:
Input (batch, 3 views, 1, H, W) → per-view: grayscale→RGB adapter → ImageNet normalize → backbone encoder → stack view features → MultiViewFusionAdapter (learned attention weights, if enabled) or mean pool → LayerNorm → Dropout → classification head(s).

- **Single-task:** 1 linear head → (batch, num_classes)
- **Multi-task:** N linear heads (one per condition) → (batch, num_tasks, num_classes)

### Five Backbone Architectures

| Registry Key | Base Model | Custom Block | feature_dim |
|---|---|---|---|
| `convnext_tiny_cbam` | ConvNeXt-Tiny | CBAM (channel + spatial attention) | 768 |
| `densenet121_dense_reuse` | DenseNet-121 | DenseReuseProjection | 1024 |
| `resnet101_3d` | ResNet-101 | FeatureVolume3D (reshape to 3D volume + Conv3d) | 2048 |
| `swin_transformer` | Swin-T | HierarchicalFeatureFusion (local + global branches) | 768 |
| `vit_base_posenc` | ViT-B/16 | None (pure ViT) | 768 |

## Key Data Flows

**Preprocessing:** RSNA CSVs → `load_rsna_tables()` → `build_study_index()` → `build_split_manifests()` → `save_split_manifests()` → CSV manifests in `artifacts/processed/`

**Training:** YAML config → `TrainingConfig` → `build_dataloaders()` → `LumbarStudyDataset` (loads middle DICOM slice per view, stacks 3 views as channels) → `Trainer.fit()` → saves `best.ckpt`, `metrics.csv`, `history.json`, `predictions.csv` under `artifacts/runs/<run-id>/`

## Training Features

**Loss functions:** `cross_entropy` (with optional `label_smoothing` and `class_weight_mode=balanced`) or `focal` (configurable `focal_gamma`, default 2.0). Focal loss is incompatible with label_smoothing and class_weight_mode.

**Multi-task learning:** Set `target_columns: all` for all 25 conditions, or provide an explicit list. Each task gets its own linear head and loss; losses are summed. Task-0 metrics are used for monitoring and early stopping.

**TTA (Test-Time Augmentation):** Set `tta_count` > 1. At validation, averages logits over N augmented views (first view unaugmented).

**Augmentation modes** (`train_augment_mode`):
- `off`/`null`: No augmentation
- `light`: Brightness ±5%, contrast ±5%, Gaussian noise (σ=0.003)
- `medium`: All of light + shared affine (rotation ±10°, scale 0.95–1.05, translate ±2.5%). No flips — preserves left/right foraminal anatomy.

**Other:** AdamW or SGD optimizer, CosineAnnealingLR or StepLR scheduler, AMP (CUDA only), WeightedRandomSampler (`sampler_mode=balanced`), early stopping on val_macro_f1.

## Label Encoding

`SEVERITY_TO_INDEX` maps: Normal/Mild → 0, Moderate → 1, Severe → 2

25 target conditions = 5 spinal canal stenosis + 10 neural foraminal narrowing (L/R) + 10 subarticular stenosis (L/R), across L1-L2 through L5-S1.

## Config Structure

Train configs use a flat YAML structure (not nested). The CLI `train` command reads fields directly with `.get()` calls. See `configs/train/default.yaml` for the canonical schema. Key non-obvious fields: `target_columns` (null=single-task, "all"=25 tasks, or explicit list), `tta_count`, `focal_gamma`, `overfit_subset_size` (shares same subset for train+val).

## Environment Variables

- `LUMBAR_CONVNEXT_TINY_WEIGHTS`: Path to offline ConvNeXt-Tiny weights file
- `LUMBAR_TQDM`: Controls tqdm progress display
- `HF_ENDPOINT`: HuggingFace mirror URL (set to hf-mirror.com on Chinese servers)

## Important Paths

- Dataset: `rsna-2024-lumbar-spine-degenerative-classification/` (train.csv, train_series_descriptions.csv, train_label_coordinates.csv, train_images/)
- Artifacts: `artifacts/` (runs/, processed/, metadata/, figures/)
- Reports: `reports/figures/`

## Remote Training

Training runs on a Linux server via `scripts/sync_to_server.sh` and `scripts/server_train.sh`. The server uses the same codebase with local data paths. `sync_to_server.sh` supports `--with-data` to include DICOM images. `server_train.sh` uses nohup background training with PID tracking, Chinese PyPI/HuggingFace mirrors, and `uv` for dependency management. Run locally first to verify; sync and train remotely for GPU.
