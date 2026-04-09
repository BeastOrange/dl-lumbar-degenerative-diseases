from __future__ import annotations

import csv
from pathlib import Path

from dl_lumbar_dd.eval import build_ranking_table, save_ranking_table, summarize_run


def _write_run(run_dir: Path, model_name: str, fusion_enabled: bool, rows: list[dict[str, str]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        file.write(f"model_name: {model_name}\n")
        file.write(f"fusion_enabled: {'true' if fusion_enabled else 'false'}\n")

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "val_macro_f1", "val_macro_auc", "val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_run_uses_best_primary_metric(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    _write_run(
        run_dir,
        model_name="convnext_tiny_cbam",
        fusion_enabled=True,
        rows=[
            {"epoch": "1", "val_macro_f1": "0.61", "val_macro_auc": "0.72", "val_accuracy": "0.66"},
            {"epoch": "2", "val_macro_f1": "0.75", "val_macro_auc": "0.80", "val_accuracy": "0.70"},
        ],
    )

    summary = summarize_run(run_dir)

    assert summary.model_name == "convnext_tiny_cbam"
    assert summary.fusion_enabled is True
    assert summary.best_epoch == 2
    assert summary.primary_score == 0.75
    assert summary.final_val_macro_auc == 0.80


def test_build_ranking_table_sorts_descending_and_saves_outputs(tmp_path: Path) -> None:
    _write_run(
        tmp_path / "run_b",
        model_name="swin_transformer",
        fusion_enabled=True,
        rows=[{"epoch": "1", "val_macro_f1": "0.83", "val_macro_auc": "0.89", "val_accuracy": "0.81"}],
    )
    _write_run(
        tmp_path / "run_a",
        model_name="densenet121_dense_reuse",
        fusion_enabled=False,
        rows=[{"epoch": "1", "val_macro_f1": "0.77", "val_macro_auc": "0.84", "val_accuracy": "0.75"}],
    )

    ranking = build_ranking_table(tmp_path)

    assert [row["run_id"] for row in ranking] == ["run_b", "run_a"]
    assert ranking[0]["primary_metric"] == "val_macro_f1"

    csv_path = tmp_path / "reports" / "ranking.csv"
    json_path = tmp_path / "reports" / "ranking.json"
    save_ranking_table(ranking, output_csv=csv_path, output_json=json_path)

    assert csv_path.exists()
    assert json_path.exists()
