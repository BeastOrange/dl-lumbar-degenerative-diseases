from __future__ import annotations

import sys
from pathlib import Path

from dl_lumbar_dd.data.ingest import build_study_index, load_rsna_tables
from dl_lumbar_dd.data.splits import build_split_manifests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.helpers import create_mock_rsna_dataset


def test_build_split_manifests_has_no_leakage(tmp_path) -> None:
    dataset_root = create_mock_rsna_dataset(tmp_path, study_count=9)
    bundle = load_rsna_tables(dataset_root)
    study_index = build_study_index(bundle)

    manifests = build_split_manifests(study_index, seed=7, train_ratio=0.67)

    train_ids = set(manifests.train["study_id"].tolist())
    val_ids = set(manifests.validation["study_id"].tolist())
    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(study_index["study_id"].tolist())


def test_build_split_manifests_supports_kfolds(tmp_path) -> None:
    dataset_root = create_mock_rsna_dataset(tmp_path, study_count=9)
    study_index = build_study_index(load_rsna_tables(dataset_root))

    manifests = build_split_manifests(study_index, seed=11, train_ratio=0.8, folds=3)

    assert len(manifests.folds) == 3
    for fold in manifests.folds:
        train_ids = set(fold.train["study_id"].tolist())
        val_ids = set(fold.validation["study_id"].tolist())
        assert train_ids.isdisjoint(val_ids)


def test_build_split_manifests_falls_back_when_stratify_is_infeasible(tmp_path) -> None:
    dataset_root = create_mock_rsna_dataset(tmp_path, study_count=6)
    study_index = build_study_index(load_rsna_tables(dataset_root))

    manifests = build_split_manifests(study_index, seed=3, train_ratio=0.8, folds=3)

    assert len(manifests.train) + len(manifests.validation) == len(study_index)
