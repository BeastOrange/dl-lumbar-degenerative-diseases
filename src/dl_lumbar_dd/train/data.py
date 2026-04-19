"""Data loading utilities for model training."""

from __future__ import annotations

import math
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
    """Dataset that builds one three-view sample per study.

    Supports both single-task (one label per study) and multi-task
    (25 labels per study, one per condition) modes.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        bundle: RSNATables,
        image_size: int,
        max_samples: int | None = None,
        augment_mode: str | None = None,
        target_columns: list[str] | None = None,
        target_column: str | None = None,
        num_slices: int = 1,
    ) -> None:
        self.dataset_root = bundle.dataset_root
        self.series_table = bundle.series
        self.image_size = image_size
        self.num_slices = num_slices
        self.augment_mode = _normalize_augment_mode(augment_mode)
        records = manifest.sort_values("study_id").reset_index(drop=True)
        if max_samples is not None:
            records = records.head(max_samples)
        self.study_ids = records["study_id"].astype(int).tolist()
        # Resolve: explicit list wins; single-column string promoted to list
        if target_columns is not None:
            self.target_columns = target_columns
        elif target_column is not None:
            self.target_columns = [target_column]
        else:
            self.target_columns = None
        self._load_labels(bundle)

    def _load_multi_task_labels(
        self, bundle: RSNATables, target_columns: list[str]
    ) -> list[list[int]]:
        """Load all target columns and resolve to label indices per study."""
        label_df = bundle.train.set_index("study_id")[target_columns]
        indices: list[list[int]] = []
        for study_id in self.study_ids:
            row_indices: list[int] = []
            for col in target_columns:
                severity = label_df.at[study_id, col]
                if pd.isna(severity):
                    severity = "Normal/Mild"  # treat missing as normal
                row_indices.append(SEVERITY_TO_INDEX.get(str(severity), 0))
            indices.append(row_indices)
        return indices

    def _load_labels(self, bundle: RSNATables) -> None:
        """Populate self.label_indices for single-task or multi-task mode."""
        if self.target_columns is None:
            self.label_indices = []
            return
        raw = self._load_multi_task_labels(bundle, self.target_columns)
        # Flatten to list[int] when single-task for backward compat
        if len(self.target_columns) == 1:
            self.label_indices = [item[0] for item in raw]
        else:
            self.label_indices = raw

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        study_id = self.study_ids[index]
        image = build_three_view_tensor(
            dataset_root=self.dataset_root,
            study_id=study_id,
            series_table=self.series_table,
            image_size=self.image_size,
            num_slices=self.num_slices,
        )
        label = self.label_indices[index]
        tensor = torch.from_numpy(image).to(dtype=torch.float32)
        if self.num_slices == 1:
            tensor = tensor.unsqueeze(1)  # (views, 1, H, W)
        # else: tensor is already (views, num_slices, H, W)
        tensor = self._apply_augmentation(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)

    def _apply_augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.augment_mode is None:
            return tensor

        # --- Intensity augmentations (all modes) ---
        augmented = tensor.clone()
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.10
        contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.10
        augmented = augmented * brightness
        mean = augmented.mean(dim=(-2, -1), keepdim=True)
        augmented = (augmented - mean) * contrast + mean
        augmented = augmented + torch.randn_like(augmented) * 0.003

        # --- Geometric augmentations (medium mode only) ---
        # NOTE: No flip is applied — axial T2 carries definitive left/right anatomical
        # orientation (left vs right foraminal stenosis), which would be corrupted.
        if self.augment_mode == "medium":
            # Geometric augmentation is applied with the SAME transform to ALL three views
            # to preserve spatial correspondence across T1-sag / T2-sag / T2-axial.
            angle_deg = (torch.rand(1).item() - 0.5) * 20.0  # ±10 degrees (reduced from ±15)
            angle_rad = angle_deg * (math.pi / 180.0)
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.10  # 0.95 ~ 1.05 (reduced)
            tx = (torch.rand(1).item() - 0.5) * 0.05  # ±2.5% of width
            ty = (torch.rand(1).item() - 0.5) * 0.05  # ±2.5% of height

            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            # Single 2D affine matrix shared across all views: (2, 3)
            theta = torch.tensor(
                [[scale * cos_a, -scale * sin_a, tx],
                 [scale * sin_a,  scale * cos_a, ty]],
                dtype=torch.float32,
            )
            # Same transform for all views: repeat theta for each view (3x)
            grid = torch.nn.functional.affine_grid(
                theta.unsqueeze(0).repeat(augmented.shape[0], 1, 1),
                augmented.shape,
                align_corners=False,
            )
            augmented = torch.nn.functional.grid_sample(
                augmented, grid,
                mode="bilinear", padding_mode="zeros", align_corners=False
            )

        elif self.augment_mode != "light":
            raise ValueError(f"Unsupported augment_mode: {self.augment_mode}")

        return augmented.clamp(0.0, 1.0)


def apply_tta_augmentation(tensor: torch.Tensor) -> torch.Tensor:
    """Apply a single random light augmentation (for TTA use)."""
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
        "version": 3,
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
    target_columns: list[str] | None = None,
    num_slices: int = 1,
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
        image_size=image_size,
        max_samples=max_train_samples,
        augment_mode=train_augment_mode,
        target_columns=target_columns,
        target_column=target_column,
        num_slices=num_slices,
    )
    validation_dataset = LumbarStudyDataset(
        validation_manifest,
        bundle=bundle,
        image_size=image_size,
        max_samples=max_val_samples,
        augment_mode=None,
        target_columns=target_columns,
        target_column=target_column,
        num_slices=num_slices,
    )
    common_loader_args: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common_loader_args["prefetch_factor"] = 2
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

    # Multi-task: use the first task's labels for balancing
    first = dataset.label_indices[0]
    if isinstance(first, list):
        labels_for_sampler = [item[0] for item in dataset.label_indices]
    else:
        labels_for_sampler = dataset.label_indices

    class_counts = Counter(int(label) for label in labels_for_sampler)
    weights = torch.tensor(
        [1.0 / class_counts[int(label)] for label in labels_for_sampler],
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


def build_fold_dataloaders(
    *,
    fold_idx: int,
    dataset_root: str | Path,
    processed_root: str | Path,
    target_column: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_studies: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    sampler_mode: str | None = None,
    train_augment_mode: str | None = None,
    target_columns: list[str] | None = None,
    num_slices: int = 1,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """Build dataloaders for a specific cross-validation fold."""
    root = Path(processed_root)
    fold_train = root / f"fold_{fold_idx}_train.csv"
    fold_val = root / f"fold_{fold_idx}_validation.csv"
    if not fold_train.exists() or not fold_val.exists():
        raise FileNotFoundError(f"Fold {fold_idx} manifests not found in {root}")

    bundle = load_rsna_tables(dataset_root, max_studies=max_studies)
    train_manifest = pd.read_csv(fold_train)
    val_manifest = pd.read_csv(fold_val)

    train_dataset = LumbarStudyDataset(
        train_manifest, bundle=bundle, image_size=image_size,
        max_samples=max_train_samples, augment_mode=train_augment_mode,
        target_columns=target_columns, target_column=target_column,
        num_slices=num_slices,
    )
    val_dataset = LumbarStudyDataset(
        val_manifest, bundle=bundle, image_size=image_size,
        max_samples=max_val_samples, augment_mode=None,
        target_columns=target_columns, target_column=target_column,
        num_slices=num_slices,
    )

    common_loader_args: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common_loader_args["prefetch_factor"] = 2

    train_sampler = _build_train_sampler(train_dataset, sampler_mode=sampler_mode, seed=seed)
    train_loader = DataLoader(train_dataset, shuffle=train_sampler is None, sampler=train_sampler, **common_loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)
    return train_loader, val_loader
