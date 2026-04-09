from __future__ import annotations

from pathlib import Path

from dl_lumbar_dd.visualization import save_confusion_matrix, save_multiclass_roc, save_training_history


CLASS_NAMES = ["Normal/Mild", "Moderate", "Severe"]


def test_plot_helpers_create_output_files(tmp_path: Path) -> None:
    confusion_path = save_confusion_matrix(
        y_true=[0, 1, 2, 1],
        y_pred=[0, 1, 1, 1],
        output_path=tmp_path / "confusion.png",
        class_names=CLASS_NAMES,
        normalize=True,
    )
    roc_path = save_multiclass_roc(
        y_true=[0, 1, 2, 1],
        y_score=[
            [0.9, 0.08, 0.02],
            [0.1, 0.8, 0.1],
            [0.1, 0.5, 0.4],
            [0.1, 0.7, 0.2],
        ],
        output_path=tmp_path / "roc.png",
        class_names=CLASS_NAMES,
    )
    history_path = save_training_history(
        {"train_loss": [1.0, 0.7], "val_loss": [1.1, 0.8], "val_macro_f1": [0.4, 0.6]},
        output_path=tmp_path / "history.png",
    )

    assert confusion_path.exists()
    assert roc_path.exists()
    assert history_path.exists()


def test_save_multiclass_roc_handles_single_class_without_failure(tmp_path: Path) -> None:
    output_path = save_multiclass_roc(
        y_true=[1, 1, 1],
        y_score=[
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.7, 0.1],
        ],
        output_path=tmp_path / "single_class_roc.png",
        class_names=CLASS_NAMES,
    )

    assert output_path.exists()
