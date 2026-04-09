from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from _app import render_images, setup_page  # noqa: E402

setup_page(
    "Inference & Explainability",
    "🔬",
    "Preview incoming scans, inspect explainability artifacts, and monitor deployment readiness.",
)

uploaded = st.file_uploader("Upload a PNG/JPG preview image for presentation", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    st.image(uploaded, caption="Uploaded preview", use_container_width=True)
    st.info(
        "This page is wired for explainability outputs. Once inference artifacts are generated, "
        "heatmaps and case summaries will appear below."
    )
else:
    st.caption("Upload is optional. The page still visualizes existing explainability outputs.")

render_images(
    "Explainability Outputs",
    ["gradcam*.png", "attention_*.png", "explainability_*.png"],
    caption_prefix="Explainability",
)
