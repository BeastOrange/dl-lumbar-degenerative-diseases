from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from _app import DATASET_DIR, render_dataframe, safe_read_csv, setup_page  # noqa: E402

setup_page(
    "Data Explorer",
    "📚",
    "Inspect the raw RSNA 2024 dataset footprint, tabular labels, and study-level files before preprocessing.",
)

st.subheader("Dataset footprint")
if DATASET_DIR.exists():
    train_csv = safe_read_csv(DATASET_DIR / "train.csv")
    series_csv = safe_read_csv(DATASET_DIR / "train_series_descriptions.csv")
    coord_csv = safe_read_csv(DATASET_DIR / "train_label_coordinates.csv")
    c1, c2, c3 = st.columns(3)
    c1.metric("Label Rows", str(len(train_csv)) if train_csv is not None else "N/A")
    c2.metric("Series Rows", str(len(series_csv)) if series_csv is not None else "N/A")
    c3.metric("Coordinate Rows", str(len(coord_csv)) if coord_csv is not None else "N/A")
else:
    st.warning("The dataset directory is not present in the current workspace.")

render_dataframe("Train Labels", safe_read_csv(DATASET_DIR / "train.csv"), height=360)
render_dataframe("Series Descriptions", safe_read_csv(DATASET_DIR / "train_series_descriptions.csv"), height=320)
render_dataframe("Coordinate Labels", safe_read_csv(DATASET_DIR / "train_label_coordinates.csv"), height=320)
