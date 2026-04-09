from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from _app import METADATA_DIR, render_dataframe, render_images, safe_read_csv, safe_read_json, setup_page  # noqa: E402

setup_page(
    "Preprocessing QA",
    "🧪",
    "Validate study splits, processed sample coverage, and before-versus-after quality panels.",
)

split_json = safe_read_json(METADATA_DIR / "split_summary.json")
if split_json is None:
    from streamlit import warning

    warning("Split summary is not available yet. Run preprocessing to generate study-level splits.")
else:
    import pandas as pd
    from streamlit import subheader, dataframe

    subheader("Split Summary")
    dataframe(pd.DataFrame([split_json]), use_container_width=True)

render_dataframe("Processed Samples Index", safe_read_csv(METADATA_DIR / "processed_samples.csv"))
render_images(
    "Preprocessing Panels",
    ["preprocess_*.png", "qa_*.png", "before_after_*.png"],
    caption_prefix="QA panel",
)
