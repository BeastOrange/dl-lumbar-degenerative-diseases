from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from _app import render_dataframe, render_images, run_metrics_table, safe_read_csv, setup_page  # noqa: E402

setup_page(
    "Model Comparison",
    "⚖️",
    "Compare candidate architectures and fusion variants with ranking tables and evaluation plots.",
)

ranking = safe_read_csv(Path("artifacts/metadata/model_ranking.csv"))
if ranking is not None and not ranking.empty:
    render_dataframe("Model Ranking", ranking, height=320)
else:
    metrics = run_metrics_table()
    if metrics.empty:
        st.warning("Model ranking is not available yet. Run evaluation first.")
    else:
        render_dataframe("Derived Ranking from Recent Runs", metrics.sort_values(by=metrics.columns[-1]), height=320)

render_images(
    "Evaluation Figures",
    ["confusion_matrix*.png", "roc_curve*.png", "comparison_*.png"],
    caption_prefix="Evaluation",
)
