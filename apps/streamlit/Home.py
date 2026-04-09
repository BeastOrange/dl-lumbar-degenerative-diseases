from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from _app import metadata_snapshot, recent_run_dirs, render_dataframe, safe_read_csv, setup_page  # noqa: E402

setup_page(
    "Lumbar Degenerative Disease Research Hub",
    "🦴",
    "A unified dashboard for dataset profiling, preprocessing QA, model training history, "
    "comparison analytics, and deployment readiness.",
)

snapshot = metadata_snapshot()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset", "Ready" if snapshot["dataset_present"] else "Missing")
col2.metric("Tracked Runs", str(snapshot["run_count"]))
col3.metric("Metadata Files", str(snapshot["metadata_files"]))
col4.metric("Figure Files", str(snapshot["figure_files"]))

st.markdown("### Workspace status")
st.info(
    "The platform reads generated artifacts from `artifacts/` and `reports/figures/`. "
    "Pages remain available before training completes and will show warnings for missing outputs."
)

recent = recent_run_dirs(limit=6)
if recent:
    st.markdown("### Recent experiment folders")
    st.write("\n".join(f"- `{path.name}`" for path in recent))
else:
    st.warning("No experiment runs are available yet. Start with dataset analysis and preprocessing.")

summary_csv = Path("artifacts/metadata/dataset_summary.csv")
render_dataframe("Dataset Summary Preview", safe_read_csv(summary_csv), height=260)
