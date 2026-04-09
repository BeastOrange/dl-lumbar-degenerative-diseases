from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from _app import METADATA_DIR, render_dataframe, render_images, safe_read_csv, setup_page  # noqa: E402

setup_page(
    "Data Analysis",
    "📈",
    "Review English visual summaries for class balance, missing labels, and sequence composition.",
)

render_images(
    "Analysis Figures",
    ["data_*distribution*.png", "data_*missing*.png", "eda_*.png", "dataset_*.png"],
    caption_prefix="Analysis figure",
)
render_dataframe("Dataset Summary Table", safe_read_csv(METADATA_DIR / "dataset_summary.csv"))
render_dataframe("Series Distribution Table", safe_read_csv(METADATA_DIR / "series_distribution.csv"))
