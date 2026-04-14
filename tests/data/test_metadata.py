from __future__ import annotations

import sys
from pathlib import Path

from dl_lumbar_dd.data.ingest import load_rsna_tables, build_study_index

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.helpers import create_mock_rsna_dataset


def test_load_rsna_tables_limits_studies(tmp_path) -> None:
    dataset_root = create_mock_rsna_dataset(tmp_path, study_count=5)

    bundle = load_rsna_tables(dataset_root, max_studies=3)

    assert bundle.train.study_id.nunique() == 3
    assert bundle.series.study_id.nunique() == 3
    assert bundle.coordinates.study_id.nunique() == 3


def test_build_study_index_contains_series_counts(tmp_path) -> None:
    dataset_root = create_mock_rsna_dataset(tmp_path, study_count=4)
    bundle = load_rsna_tables(dataset_root)

    study_index = build_study_index(bundle)

    assert set(study_index.columns) >= {"study_id", "series_count", "stratify_key"}
    assert study_index["series_count"].min() == 3
    assert study_index["study_id"].is_unique


def test_build_study_index_adds_target_aware_stratify_columns(tmp_path) -> None:
    dataset_root = create_mock_rsna_dataset(tmp_path, study_count=4)
    bundle = load_rsna_tables(dataset_root)

    study_index = build_study_index(bundle, target_column="spinal_canal_stenosis_l4_l5")

    assert set(study_index.columns) >= {"target_label", "target_stratify_key"}
    assert study_index["target_label"].isin({"Normal/Mild", "Moderate", "Severe"}).all()
    assert study_index["target_stratify_key"].str.contains("|", regex=False).all()
