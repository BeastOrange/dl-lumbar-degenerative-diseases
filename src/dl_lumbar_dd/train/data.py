"""Data loading utilities for model training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

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
    ) -> None:
        self.dataset_root = bundle.dataset_root
        self.series_table = bundle.series
        self.image_size = image_size
        records = manifest.sort_values("study_id").reset_index(drop=True)
        if max_samples is not None:
            records = records.head(max_samples)
        self.study_ids = records["study_id"].astype(int).tolist()
        self.labels = bundle.train.set_index("study_id")[target_column].to_dict()

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        study_id = self.study_ids[index]
        image = build_three_view_tensor(
            dataset_root=self.dataset_root,
            study_id=study_id,
            series_table=self.series_table,
            image_size=self.image_size,
        )
        severity = str(self.labels.get(study_id, "Normal/Mild"))
        label = SEVERITY_TO_INDEX.get(severity, 0)
        # 输出 [view, channel, height, width]，匹配融合模块的 5D 输入约定
        tensor = torch.from_numpy(image).unsqueeze(1).to(dtype=torch.float32)
        return tensor, torch.tensor(label, dtype=torch.long)


def prepare_bundle_and_manifests(
    dataset_root: str | Path,
    processed_root: str | Path,
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
    if train_manifest.exists() and validation_manifest.exists():
        return bundle, train_manifest, validation_manifest

    study_index = build_study_index(bundle)
    manifests = build_split_manifests(study_index, seed=seed, folds=folds, train_ratio=train_ratio)
    save_split_manifests(manifests, root)
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
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """Build train/validation dataloaders with study-level manifests."""
    bundle, train_path, validation_path = prepare_bundle_and_manifests(
        dataset_root=dataset_root,
        processed_root=processed_root,
        seed=seed,
        folds=folds,
        train_ratio=train_ratio,
        max_studies=max_studies,
    )
    train_manifest = pd.read_csv(train_path)
    validation_manifest = pd.read_csv(validation_path)
    train_dataset = LumbarStudyDataset(
        train_manifest,
        bundle=bundle,
        target_column=target_column,
        image_size=image_size,
        max_samples=max_train_samples,
    )
    validation_dataset = LumbarStudyDataset(
        validation_manifest,
        bundle=bundle,
        target_column=target_column,
        image_size=image_size,
        max_samples=max_val_samples,
    )
    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common_loader_args)
    return train_loader, validation_loader
