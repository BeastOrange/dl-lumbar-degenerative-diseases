"""Shared helpers for the Streamlit research dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
DATASET_DIR = PROJECT_ROOT / "rsna-2024-lumbar-spine-degenerative-classification"

CARD_STYLE = """
<style>
:root {
  --bg: #f3efe6;
  --card: rgba(255,255,255,0.82);
  --ink: #1f2937;
  --accent: #bf5b35;
  --accent-soft: #f2cbb7;
}
.stApp {
  background:
    radial-gradient(circle at top left, rgba(191,91,53,0.18), transparent 28%),
    radial-gradient(circle at top right, rgba(28,86,107,0.14), transparent 24%),
    linear-gradient(135deg, #f4eee1 0%, #efe9df 45%, #f8f5ef 100%);
  color: var(--ink);
}
.block-container {
  padding-top: 2.2rem;
  padding-bottom: 3rem;
}
.hero-card, .metric-card {
  background: var(--card);
  border: 1px solid rgba(31,41,55,0.08);
  border-radius: 22px;
  padding: 1.2rem 1.3rem;
  box-shadow: 0 12px 28px rgba(31,41,55,0.08);
}
.hero-kicker {
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--accent);
  font-size: 0.76rem;
  font-weight: 700;
}
.hero-title {
  font-size: 2.2rem;
  font-weight: 800;
  margin: 0.2rem 0 0.4rem 0;
}
.hero-copy {
  font-size: 1rem;
  line-height: 1.65;
}
.small-muted {
  color: rgba(31,41,55,0.72);
  font-size: 0.92rem;
}
</style>
"""


def setup_page(title: str, icon: str, description: str) -> None:
    """Apply a shared layout and page chrome."""
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(CARD_STYLE, unsafe_allow_html=True)
    st.markdown(
        (
            '<div class="hero-card">'
            '<div class="hero-kicker">Lumbar Imaging Research Platform</div>'
            f'<div class="hero-title">{title}</div>'
            f'<div class="hero-copy">{description}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    st.write("")


def card_metric(label: str, value: str, help_text: str) -> None:
    """Render one dashboard metric card."""
    st.markdown(
        (
            '<div class="metric-card">'
            f'<div class="hero-kicker">{label}</div>'
            f'<div class="hero-title" style="font-size:1.55rem;">{value}</div>'
            f'<div class="small-muted">{help_text}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    """Read JSON data when the file exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Read CSV data when the file exists."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def recent_run_dirs(limit: int = 8) -> list[Path]:
    """Return recent run directories sorted by modification time."""
    if not RUNS_DIR.exists():
        return []
    runs = [path for path in RUNS_DIR.iterdir() if path.is_dir()]
    return sorted(runs, key=lambda item: item.stat().st_mtime, reverse=True)[:limit]


def latest_run_dir() -> Path | None:
    """Return the latest run directory, if any."""
    runs = recent_run_dirs(limit=1)
    return runs[0] if runs else None


def show_missing_artifact(message: str) -> None:
    """Display a consistent warning for absent artifacts."""
    st.warning(message)


def render_dataframe(title: str, data: pd.DataFrame | None, *, height: int = 320) -> None:
    """Show a dataframe or an informative warning."""
    st.subheader(title)
    if data is None or data.empty:
        show_missing_artifact(f"{title} is not available yet.")
        return
    st.dataframe(data, use_container_width=True, height=height)


def render_images(title: str, patterns: list[str], *, caption_prefix: str = "Artifact") -> None:
    """Render all matching image files from the reports directory."""
    st.subheader(title)
    images = []
    for pattern in patterns:
        images.extend(sorted(REPORTS_DIR.glob(pattern)))
    if not images:
        show_missing_artifact(f"No images found for {title.lower()}.")
        return
    for image_path in images[:8]:
        st.image(str(image_path), caption=f"{caption_prefix}: {image_path.name}", use_container_width=True)


def run_metrics_table() -> pd.DataFrame:
    """Collect per-run metric summaries for dashboard tables."""
    rows: list[dict[str, Any]] = []
    for run_dir in recent_run_dirs(limit=40):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        frame = pd.read_csv(metrics_path)
        if frame.empty:
            continue
        last_row = frame.iloc[-1].to_dict()
        rows.append({"run_id": run_dir.name, **last_row})
    return pd.DataFrame(rows)


def metadata_snapshot() -> dict[str, Any]:
    """Summarize local directories for top-level visibility."""
    snapshot = {
        "dataset_present": DATASET_DIR.exists(),
        "run_count": len(recent_run_dirs(limit=1000)),
        "metadata_files": len(list(METADATA_DIR.glob("*"))) if METADATA_DIR.exists() else 0,
        "figure_files": len(list(REPORTS_DIR.glob("*"))) if REPORTS_DIR.exists() else 0,
    }
    return snapshot
