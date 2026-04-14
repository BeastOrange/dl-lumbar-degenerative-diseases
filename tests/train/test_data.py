from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Subset, WeightedRandomSampler

from dl_lumbar_dd.train import data as train_data


def _write_manifest(path: Path, study_ids: list[int]) -> None:
    pd.DataFrame({"study_id": study_ids}).to_csv(path, index=False)


def _severity_from_index(index: int) -> str:
    mapping = {
        0: "Normal/Mild",
        1: "Moderate",
        2: "Severe",
    }
    return mapping[index]


def _extract_study_ids(dataset: Any) -> list[int]:
    if isinstance(dataset, Subset):
        parent_ids = _extract_study_ids(dataset.dataset)
        return [int(parent_ids[index]) for index in dataset.indices]
    if hasattr(dataset, "study_ids"):
        return [int(study_id) for study_id in dataset.study_ids]
    raise AssertionError(f"Unsupported dataset type: {type(dataset)!r}")


def _extract_label_indices(dataset: Any) -> list[int]:
    if isinstance(dataset, Subset):
        parent_labels = _extract_label_indices(dataset.dataset)
        return [int(parent_labels[index]) for index in dataset.indices]
    if hasattr(dataset, "label_indices"):
        return [int(label) for label in dataset.label_indices]
    raise AssertionError(f"Unsupported dataset type: {type(dataset)!r}")


def test_build_dataloaders_balanced_sampler_uses_weighted_sampler(tmp_path: Path, monkeypatch) -> None:
    target_column = "spinal_canal_stenosis_l4_l5"
    train_ids = [101, 102, 103, 104, 105, 106]
    val_ids = [201, 202]
    train_labels = [0, 0, 0, 0, 1, 2]

    train_manifest = tmp_path / "train_manifest.csv"
    val_manifest = tmp_path / "validation_manifest.csv"
    _write_manifest(train_manifest, train_ids)
    _write_manifest(val_manifest, val_ids)

    label_by_study = dict(zip(train_ids, train_labels, strict=True))
    for study_id in val_ids:
        label_by_study[study_id] = 0
    bundle = SimpleNamespace(
        dataset_root=tmp_path,
        series=pd.DataFrame({"study_id": train_ids + val_ids, "series_id": [1] * (len(train_ids) + len(val_ids))}),
        train=pd.DataFrame(
            {
                "study_id": list(label_by_study.keys()),
                target_column: [_severity_from_index(label_by_study[study_id]) for study_id in label_by_study],
            }
        ),
    )

    monkeypatch.setattr(
        train_data,
        "prepare_bundle_and_manifests",
        lambda **_: (bundle, train_manifest, val_manifest),
    )

    train_loader, _ = train_data.build_dataloaders(
        dataset_root=tmp_path,
        processed_root=tmp_path,
        target_column=target_column,
        image_size=224,
        batch_size=4,
        num_workers=0,
        seed=7,
        folds=5,
        train_ratio=0.8,
        sampler_mode="balanced",
    )

    assert isinstance(train_loader.sampler, WeightedRandomSampler)

    dataset_labels = _extract_label_indices(train_loader.dataset)
    weights = train_loader.sampler.weights.tolist()
    per_class_avg = {
        class_index: sum(weight for label, weight in zip(dataset_labels, weights, strict=True) if label == class_index)
        / sum(1 for label in dataset_labels if label == class_index)
        for class_index in {0, 1, 2}
    }
    assert per_class_avg[1] > per_class_avg[0]
    assert per_class_avg[2] > per_class_avg[0]


def test_build_dataloaders_overfit_subset_binds_train_and_val_to_same_subset(tmp_path: Path, monkeypatch) -> None:
    target_column = "spinal_canal_stenosis_l4_l5"
    train_ids = [301, 302, 303, 304, 305]
    val_ids = [401, 402, 403]
    train_labels = [0, 1, 2, 0, 1]

    train_manifest = tmp_path / "train_manifest.csv"
    val_manifest = tmp_path / "validation_manifest.csv"
    _write_manifest(train_manifest, train_ids)
    _write_manifest(val_manifest, val_ids)

    label_by_study = dict(zip(train_ids, train_labels, strict=True))
    for study_id in val_ids:
        label_by_study[study_id] = 0
    bundle = SimpleNamespace(
        dataset_root=tmp_path,
        series=pd.DataFrame({"study_id": train_ids + val_ids, "series_id": [1] * (len(train_ids) + len(val_ids))}),
        train=pd.DataFrame(
            {
                "study_id": list(label_by_study.keys()),
                target_column: [_severity_from_index(label_by_study[study_id]) for study_id in label_by_study],
            }
        ),
    )

    monkeypatch.setattr(
        train_data,
        "prepare_bundle_and_manifests",
        lambda **_: (bundle, train_manifest, val_manifest),
    )

    train_loader, val_loader = train_data.build_dataloaders(
        dataset_root=tmp_path,
        processed_root=tmp_path,
        target_column=target_column,
        image_size=224,
        batch_size=2,
        num_workers=0,
        seed=7,
        folds=5,
        train_ratio=0.8,
        overfit_subset_size=2,
    )

    train_subset_ids = _extract_study_ids(train_loader.dataset)
    val_subset_ids = _extract_study_ids(val_loader.dataset)
    train_subset_labels = _extract_label_indices(train_loader.dataset)
    val_subset_labels = _extract_label_indices(val_loader.dataset)

    assert len(train_subset_ids) == 2
    assert train_subset_ids == val_subset_ids
    assert train_subset_labels == val_subset_labels
    assert set(train_subset_ids).issubset(set(train_ids))


def test_build_dataloaders_train_augment_mode_enables_train_only_augmentation(tmp_path: Path, monkeypatch) -> None:
    target_column = "spinal_canal_stenosis_l4_l5"
    train_ids = [501, 502, 503]
    val_ids = [601, 602]

    train_manifest = tmp_path / "train_manifest.csv"
    val_manifest = tmp_path / "validation_manifest.csv"
    _write_manifest(train_manifest, train_ids)
    _write_manifest(val_manifest, val_ids)

    labels = {
        501: "Normal/Mild",
        502: "Moderate",
        503: "Severe",
        601: "Normal/Mild",
        602: "Moderate",
    }
    bundle = SimpleNamespace(
        dataset_root=tmp_path,
        series=pd.DataFrame({"study_id": train_ids + val_ids, "series_id": [1] * (len(train_ids) + len(val_ids))}),
        train=pd.DataFrame(
            {
                "study_id": list(labels.keys()),
                target_column: [labels[study_id] for study_id in labels],
            }
        ),
    )
    monkeypatch.setattr(
        train_data,
        "prepare_bundle_and_manifests",
        lambda **_: (bundle, train_manifest, val_manifest),
    )

    train_loader, val_loader = train_data.build_dataloaders(
        dataset_root=tmp_path,
        processed_root=tmp_path,
        target_column=target_column,
        image_size=224,
        batch_size=2,
        num_workers=0,
        seed=7,
        folds=5,
        train_ratio=0.8,
        train_augment_mode="light",
    )

    # contract: train/val 数据集都暴露稳定字段 augment_mode，且仅训练集启用增强
    assert hasattr(train_loader.dataset, "augment_mode")
    assert hasattr(val_loader.dataset, "augment_mode")
    assert train_loader.dataset.augment_mode == "light"
    assert val_loader.dataset.augment_mode in (None, "off")


def test_build_dataloaders_augmentation_preserves_tensor_shape_and_label_type_and_off_mode_is_deterministic(
    tmp_path: Path, monkeypatch
) -> None:
    target_column = "spinal_canal_stenosis_l4_l5"
    train_ids = [701, 702]
    val_ids = [801]

    train_manifest = tmp_path / "train_manifest.csv"
    val_manifest = tmp_path / "validation_manifest.csv"
    _write_manifest(train_manifest, train_ids)
    _write_manifest(val_manifest, val_ids)

    labels = {
        701: "Normal/Mild",
        702: "Severe",
        801: "Moderate",
    }
    bundle = SimpleNamespace(
        dataset_root=tmp_path,
        series=pd.DataFrame({"study_id": train_ids + val_ids, "series_id": [1] * (len(train_ids) + len(val_ids))}),
        train=pd.DataFrame(
            {
                "study_id": list(labels.keys()),
                target_column: [labels[study_id] for study_id in labels],
            }
        ),
    )
    monkeypatch.setattr(
        train_data,
        "prepare_bundle_and_manifests",
        lambda **_: (bundle, train_manifest, val_manifest),
    )
    monkeypatch.setattr(
        train_data,
        "build_three_view_tensor",
        lambda **_: torch.ones((3, 224, 224), dtype=torch.float32).numpy(),
    )

    aug_train_loader, _ = train_data.build_dataloaders(
        dataset_root=tmp_path,
        processed_root=tmp_path,
        target_column=target_column,
        image_size=224,
        batch_size=1,
        num_workers=0,
        seed=7,
        folds=5,
        train_ratio=0.8,
        train_augment_mode="light",
    )
    off_train_loader, _ = train_data.build_dataloaders(
        dataset_root=tmp_path,
        processed_root=tmp_path,
        target_column=target_column,
        image_size=224,
        batch_size=1,
        num_workers=0,
        seed=7,
        folds=5,
        train_ratio=0.8,
        train_augment_mode="off",
    )

    aug_x, aug_y = aug_train_loader.dataset[0]
    off_x_1, off_y_1 = off_train_loader.dataset[0]
    off_x_2, off_y_2 = off_train_loader.dataset[0]

    assert tuple(aug_x.shape) == (3, 1, 224, 224)
    assert tuple(off_x_1.shape) == (3, 1, 224, 224)
    assert aug_y.dtype == torch.long
    assert off_y_1.dtype == torch.long
    assert torch.equal(off_x_1, off_x_2)
    assert int(off_y_1.item()) == int(off_y_2.item())


def test_prepare_bundle_and_manifests_reuses_matching_cache_and_rebuilds_on_target_change(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "dataset"
    processed_root = tmp_path / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    train_manifest = processed_root / "train_manifest.csv"
    val_manifest = processed_root / "validation_manifest.csv"
    meta_path = processed_root / "split_manifest_meta.json"
    _write_manifest(train_manifest, [1, 2, 3])
    _write_manifest(val_manifest, [4, 5])

    bundle = SimpleNamespace(
        dataset_root=dataset_root,
        series=pd.DataFrame({"study_id": [1, 2, 3, 4, 5], "series_id": [1, 1, 1, 1, 1]}),
        train=pd.DataFrame({"study_id": [1, 2, 3, 4, 5], "spinal_canal_stenosis_l4_l5": ["Normal/Mild"] * 5}),
    )
    meta_path.write_text(
        json.dumps(
            {
                "version": 2,
                "target_column": "spinal_canal_stenosis_l4_l5",
                "seed": 7,
                "folds": 3,
                "train_ratio": 0.8,
                "max_studies": None,
            }
        ),
        encoding="utf-8",
    )

    build_study_index_calls: list[str] = []
    build_split_calls: list[str] = []

    monkeypatch.setattr(train_data, "load_rsna_tables", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(
        train_data,
        "build_study_index",
        lambda current_bundle, target_column: build_study_index_calls.append(target_column) or pd.DataFrame({"study_id": [1]}),
    )
    monkeypatch.setattr(
        train_data,
        "build_split_manifests",
        lambda *args, **kwargs: build_split_calls.append("called") or SimpleNamespace(train=pd.DataFrame(), validation=pd.DataFrame(), folds=[]),
    )
    monkeypatch.setattr(
        train_data,
        "save_split_manifests",
        lambda manifests, output_root: [
            train_manifest.write_text("study_id\n11\n", encoding="utf-8"),
            val_manifest.write_text("study_id\n22\n", encoding="utf-8"),
        ],
    )

    train_path, val_path = train_manifest, val_manifest
    returned_bundle, returned_train, returned_val = train_data.prepare_bundle_and_manifests(
        dataset_root=dataset_root,
        processed_root=processed_root,
        target_column="spinal_canal_stenosis_l4_l5",
        seed=7,
        folds=3,
        train_ratio=0.8,
    )
    assert returned_bundle is bundle
    assert returned_train == train_path
    assert returned_val == val_path
    assert build_study_index_calls == []
    assert build_split_calls == []

    returned_bundle, returned_train, returned_val = train_data.prepare_bundle_and_manifests(
        dataset_root=dataset_root,
        processed_root=processed_root,
        target_column="left_subarticular_stenosis_l4_l5",
        seed=7,
        folds=3,
        train_ratio=0.8,
    )
    assert returned_bundle is bundle
    assert returned_train == train_path
    assert returned_val == val_path
    assert build_study_index_calls == ["left_subarticular_stenosis_l4_l5"]
    assert build_split_calls == ["called"]

    refreshed_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert refreshed_meta["version"] == 2
    assert refreshed_meta["target_column"] == "left_subarticular_stenosis_l4_l5"


def test_prepare_bundle_and_manifests_filters_missing_target_labels_before_split(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    processed_root = tmp_path / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    bundle = SimpleNamespace(
        dataset_root=dataset_root,
        series=pd.DataFrame({"study_id": [1, 2, 3], "series_id": [1, 1, 1]}),
        train=pd.DataFrame(
            {
                "study_id": [1, 2, 3],
                "spinal_canal_stenosis_l4_l5": ["Normal/Mild", "Moderate", None],
            }
        ),
    )
    study_index = pd.DataFrame(
        {
            "study_id": [1, 2, 3],
            "target_label": ["Normal/Mild", "Moderate", "Missing"],
            "target_stratify_key": ["Normal/Mild|mild_0", "Moderate|moderate_0", "Missing|mild_0"],
        }
    )
    captured_study_ids: list[int] = []

    monkeypatch.setattr(train_data, "load_rsna_tables", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(train_data, "build_study_index", lambda *_args, **_kwargs: study_index.copy())
    monkeypatch.setattr(
        train_data,
        "build_split_manifests",
        lambda filtered_index, **_kwargs: captured_study_ids.extend(filtered_index["study_id"].tolist())
        or SimpleNamespace(
            train=filtered_index.assign(split="train"),
            validation=filtered_index.head(0).assign(split="validation"),
            folds=[],
        ),
    )

    returned_bundle, train_path, validation_path = train_data.prepare_bundle_and_manifests(
        dataset_root=dataset_root,
        processed_root=processed_root,
        target_column="spinal_canal_stenosis_l4_l5",
        seed=7,
        folds=3,
        train_ratio=0.8,
    )

    assert returned_bundle is bundle
    assert captured_study_ids == [1, 2]
    assert train_path.exists()
    assert validation_path.exists()


def test_lumbar_study_dataset_raises_for_missing_or_unknown_target_label(tmp_path: Path) -> None:
    manifest = pd.DataFrame({"study_id": [101, 102]})
    bundle = SimpleNamespace(
        dataset_root=tmp_path,
        series=pd.DataFrame({"study_id": [101, 102], "series_id": [1, 1]}),
        train=pd.DataFrame(
            {
                "study_id": [101, 102],
                "spinal_canal_stenosis_l4_l5": ["Normal/Mild", "Missing"],
            }
        ),
    )

    try:
        train_data.LumbarStudyDataset(
            manifest,
            bundle=bundle,
            target_column="spinal_canal_stenosis_l4_l5",
            image_size=224,
        )
    except ValueError as error:
        assert "Unsupported target label" in str(error)
    else:
        raise AssertionError("LumbarStudyDataset should reject missing or unknown target labels")
