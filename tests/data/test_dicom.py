from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dl_lumbar_dd.data.dicom import build_three_view_tensor
from dl_lumbar_dd.data.ingest import load_rsna_tables

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.helpers import create_mock_rsna_dataset


def test_build_three_view_tensor_returns_normalized_array(tmp_path) -> None:
    dataset_root = create_mock_rsna_dataset(tmp_path, study_count=3)
    bundle = load_rsna_tables(dataset_root)

    tensor = build_three_view_tensor(
        dataset_root=dataset_root,
        study_id=int(bundle.train.iloc[0]["study_id"]),
        series_table=bundle.series,
        image_size=32,
    )

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
