"""Data loading utilities for model training."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dl_lumbar_dd.constants import SEVERITY_TO_INDEX
from dl_lumbar_dd.data.dicom import build_three_view_tensor
from dl_lumbar_dd.data.ingest import RSNATables, build_study_index, load_rsna_tables
from dl_lumbar_dd.data.splits import build_split_manifests, save_split_manifests
from dl_lumbar_dd.utils.io import ensure_dir


class LumbarStudyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset that builds one three-view sample per study."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        bundle: RSNATables,
        target_column: str,
        image_size: int,
        max_samples: int | None = None,
        augment_mode: str | None = None,
    ) -> None:
        self.dataset_root = bundle.dataset_root
        self.series_table = bundle.series
        self.image_size = image_size
        self.augment_mode = _normalize_augment_mode(augment_mode)
        records = manifest.sort_values("study_id").reset_index(drop=True)
        if max_samples is not None:
            records = records.head(max_samples)
        self.study_ids = records["study_id"].astype(int).tolist()
        self.labels = bundle.train.set_index("study_id")[target_column].to_dict()
        self.label_indices = [self._resolve_label_index(study_id) for study_id in self.study_ids]

    def __len__(self) -> int:
        return len(self.study_ids)

    @staticmethod
    def _severity_to_index(severity: object) -> int:
        return SEVERITY_TO_INDEX.get(str(severity), 0)

    def _resolve_label_index(self, study_id: int) -> int:
        severity = self.labels.get(study_id)
        if severity is None:
            raise ValueError(f"Missing target label for study_id={study_id}")
        severity_key = str(severity)
        if severity_key not in SEVERITY_TO_INDEX:
            raise ValueError(f"Unsupported target label for study_id={study_id}: {severity_key}")
        return SEVERITY_TO_INDEX[severity_key]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        study_id = self.study_ids[index]
        image = build_three_view_tensor(
            dataset_root=self.dataset_root,
            study_id=study_id,
            series_table=self.series_table,
            image_size=self.image_size,
        )
        label = self.label_indices[index]
        # 输出 [view, channel, height, width]，匹配融合模块的 5D 输入约定
        tensor = torch.from_numpy(image).unsqueeze(1).to(dtype=torch.float32)
        tensor = self._apply_augmentation(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)

    def _apply_augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.augment_mode is None:
            return tensor
        if self.augment_mode != "light":
            raise ValueError(f"Unsupported augment_mode: {self.augment_mode}")
        augmented = tensor.clone()
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.10
        contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.10
        augmented = augmented * brightness
        mean = augmented.mean(dim=(-2, -1), keepdim=True)
        augmented = (augmented - mean) * contrast + mean
        augmented = augmented + torch.randn_like(augmented) * 0.003
        return augmented.clamp(0.0, 1.0)


def prepare_bundle_and_manifests(
    dataset_root: str | Path,
    processed_root: str | Path,
    target_column: str,
    *,
    seed: int,
    folds: int,
    train_ratio: float,
    max_studies: int | None = None,
) -> tuple[RSNATables, Path, Path]:
    """Load metadata and ensure split manifests are available."""
    bundle = load_rsna_tables(dataset_root, max_studies=max_studies)
    root = ensure_dir(processed_root)
    train_manifest = root / "train_manifest.csv"
    validation_manifest = root / "validation_manifest.csv"
    manifest_meta = root / "split_manifest_meta.json"
    cache_key = {
        "version": 2,
        "target_column": target_column,
        "seed": seed,
        "folds": folds,
        "train_ratio": train_ratio,
        "max_studies": max_studies,
    }
    if (
        train_manifest.exists()
        and validation_manifest.exists()
        and manifest_meta.exists()
        and _load_manifest_cache_key(manifest_meta) == cache_key
    ):
        return bundle, train_manifest, validation_manifest

    study_index = build_study_index(bundle, target_column=target_column)
    if "target_label" in study_index.columns:
        study_index = study_index[study_index["target_label"] != "Missing"].copy()
    manifests = build_split_manifests(study_index, seed=seed, folds=folds, train_ratio=train_ratio)
    save_split_manifests(manifests, root)
    manifest_meta.write_text(json.dumps(cache_key, indent=2, sort_keys=True), encoding="utf-8")
    return bundle, train_manifest, validation_manifest


def build_dataloaders(
    *,
    dataset_root: str | Path,
    processed_root: str | Path,
    target_column: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    folds: int,
    train_ratio: float,
    max_studies: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    sampler_mode: str | None = None,
    overfit_subset_size: int | None = None,
    train_augment_mode: str | None = None,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """Build train/validation dataloaders with study-level manifests."""
    bundle, train_path, validation_path = prepare_bundle_and_manifests(
        dataset_root=dataset_root,
        processed_root=processed_root,
        target_column=target_column,
        seed=seed,
        folds=folds,
        train_ratio=train_ratio,
        max_studies=max_studies,
    )
    train_manifest = pd.read_csv(train_path)
    validation_manifest = pd.read_csv(validation_path)
    if overfit_subset_size is not None:
        shared_manifest = train_manifest.head(overfit_subset_size).copy()
        train_manifest = shared_manifest
        validation_manifest = shared_manifest.copy()
    train_dataset = LumbarStudyDataset(
        train_manifest,
        bundle=bundle,
        target_column=target_column,
        image_size=image_size,
        max_samples=max_train_samples,
        augment_mode=train_augment_mode,
    )
    validation_dataset = LumbarStudyDataset(
        validation_manifest,
        bundle=bundle,
        target_column=target_column,
        image_size=image_size,
        max_samples=max_val_samples,
        augment_mode=None,
    )
    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_sampler = _build_train_sampler(train_dataset, sampler_mode=sampler_mode, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **common_loader_args,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common_loader_args)
    return train_loader, validation_loader


def _build_train_sampler(
    dataset: LumbarStudyDataset,
    *,
    sampler_mode: str | None,
    seed: int,
) -> WeightedRandomSampler | None:
    mode = (sampler_mode or "").strip().lower()
    if not mode:
        return None
    if mode != "balanced":
        raise ValueError(f"Unsupported sampler_mode: {sampler_mode}")

    class_counts = Counter(int(label) for label in dataset.label_indices)
    weights = torch.tensor(
        [1.0 / class_counts[int(label)] for label in dataset.label_indices],
        dtype=torch.double,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def _normalize_augment_mode(value: str | None) -> str | None:
    mode = (value or "").strip().lower()
    if mode in {"", "off", "none"}:
        return None
    return mode


def _load_manifest_cache_key(path: Path) -> dict[str, object] | None:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return content if isinstance(content, dict) else None
