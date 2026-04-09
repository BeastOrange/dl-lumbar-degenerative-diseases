from __future__ import annotations

from pathlib import Path
import sys

from dl_lumbar_dd.data.commands import run_eda, run_preprocess

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.helpers import create_mock_rsna_dataset


def test_run_eda_creates_english_outputs(tmp_path) -> None:
    dataset_root = create_mock_rsna_dataset(tmp_path, study_count=6)
    figures_root = tmp_path / "figures"

    result = run_eda(
        dataset_root=dataset_root,
        figures_root=figures_root,
        metadata_root=tmp_path / "metadata",
        max_studies=4,
    )

    expected = {
        "class_distribution.png",
        "missing_values.png",
        "series_distribution.png",
        "eda_summary.csv",
    }
    produced = {Path(path).name for path in result["outputs"]}
    assert expected.issubset(produced)


def test_run_preprocess_saves_split_metadata(tmp_path) -> None:
    dataset_root = create_mock_rsna_dataset(tmp_path, study_count=8)

    result = run_preprocess(
        dataset_root=dataset_root,
        output_root=tmp_path / "processed",
        metadata_root=tmp_path / "metadata",
        figures_root=tmp_path / "figures",
        max_studies=8,
        seed=19,
        folds=3,
        train_ratio=0.75,
    )

    assert (tmp_path / "processed" / "train_manifest.csv").exists()
    assert (tmp_path / "processed" / "validation_manifest.csv").exists()
    assert (tmp_path / "metadata" / "split_summary.json").exists()
    assert result["study_count"] == 8
