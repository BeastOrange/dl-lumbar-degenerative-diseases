# Lumbar DD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a full research-and-demo workflow for RSNA 2024 lumbar degenerative disease classification.

**Architecture:** The project is split into four parallel tracks: data pipeline, model training, evaluation/visualization, and Streamlit/deployment. Shared constants and config loaders provide stable interfaces, while each track writes isolated modules and tests.

**Tech Stack:** uv, Python 3.11, PyTorch, timm, pydicom, scikit-learn, matplotlib/seaborn, Streamlit.

---

## Parallel Workstreams

1. **Data**: metadata parsing, split generation, DICOM preprocessing, EDA figure generation.
2. **Model + Train**: registry for five required models, fusion adapter, trainer loop, artifact persistence.
3. **Eval + Viz**: metrics, confusion/ROC/history plots, multi-run ranking.
4. **App + Deploy**: Streamlit pages, rsync sync script, Linux training script, Windows bootstrap.

## Integration + Gate

- Integration command path: `lumbar-cli`.
- Validation gate: unit tests + CLI smoke + healthcheck.
- Every change must be committed and pushed to GitHub.
