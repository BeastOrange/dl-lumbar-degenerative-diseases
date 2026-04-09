from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from _app import latest_run_dir, render_dataframe, render_images, run_metrics_table, safe_read_csv, setup_page  # noqa: E402

setup_page(
    "Training Dashboard",
    "🏋️",
    "Track the latest experiments, epoch-by-epoch metrics, and training history figures.",
)

latest_run = latest_run_dir()
if latest_run is None:
    st.warning("No training run is available yet. Start a training job to populate this page.")
else:
    st.caption(f"Latest run: {latest_run.name}")
    render_dataframe("Latest Run Metrics", safe_read_csv(latest_run / "metrics.csv"), height=320)

render_images("Training History", ["history_*.png", "train_history*.png"], caption_prefix="History")
render_dataframe("Recent Run Summary", run_metrics_table(), height=340)
