"""Run comparison helpers for experiment ranking."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from dl_lumbar_dd.utils.io import ensure_dir


@dataclass(slots=True)
class RunSummary:
    run_id: str
    model_name: str
    fusion_enabled: bool | None
    best_epoch: int | None
    primary_metric: str
    primary_score: float | None
    final_val_macro_f1: float | None
    final_val_macro_auc: float | None
    final_val_accuracy: float | None
    run_dir: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return asdict(self)


def _to_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _load_metrics_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file) or {}
    return content if isinstance(content, dict) else {}


def summarize_run(run_dir: str | Path, primary_metric: str = "val_macro_f1") -> RunSummary:
    """Read one run directory and build a sortable summary."""
    run_path = Path(run_dir)
    rows = _load_metrics_rows(run_path / "metrics.csv")
    if not rows:
        raise ValueError(f"No metrics rows found in {run_path}")

    config = _load_config(run_path)
    ranked_rows = [row for row in rows if _to_float(row.get(primary_metric)) is not None]
    best_row = max(ranked_rows, key=lambda row: float(row[primary_metric])) if ranked_rows else rows[-1]
    final_row = rows[-1]

    return RunSummary(
        run_id=run_path.name,
        model_name=str(config.get("model_name", run_path.name)),
        fusion_enabled=(
            None if "fusion_enabled" not in config else bool(config.get("fusion_enabled"))
        ),
        best_epoch=_to_int(best_row.get("epoch")),
        primary_metric=primary_metric,
        primary_score=_to_float(best_row.get(primary_metric)),
        final_val_macro_f1=_to_float(final_row.get("val_macro_f1")),
        final_val_macro_auc=_to_float(final_row.get("val_macro_auc")),
        final_val_accuracy=_to_float(final_row.get("val_accuracy")),
        run_dir=str(run_path),
    )


def scan_run_summaries(
    runs_root: str | Path,
    primary_metric: str = "val_macro_f1",
) -> list[RunSummary]:
    """Scan all run directories that contain a metrics file."""
    root = Path(runs_root)
    if not root.exists():
        return []

    summaries: list[RunSummary] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        summaries.append(summarize_run(run_dir, primary_metric=primary_metric))
    return summaries


def build_ranking_table(
    runs_root: str | Path,
    primary_metric: str = "val_macro_f1",
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Return sorted run summaries ready for CSV or JSON serialization."""
    summaries = scan_run_summaries(runs_root, primary_metric=primary_metric)

    def ranking_key(summary: RunSummary) -> tuple[bool, float]:
        score = summary.primary_score
        return (score is None, -score if descending and score is not None else score or 0.0)

    ranked = sorted(summaries, key=ranking_key)
    return [summary.as_dict() for summary in ranked]


def save_ranking_table(
    ranking_table: list[dict[str, Any]],
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> None:
    """Persist ranking outputs in CSV and/or JSON format."""
    if output_csv is not None:
        csv_path = Path(output_csv)
        ensure_dir(csv_path.parent)
        fieldnames = list(ranking_table[0].keys()) if ranking_table else []
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ranking_table)

    if output_json is not None:
        json_path = Path(output_json)
        ensure_dir(json_path.parent)
        json_path.write_text(json.dumps(ranking_table, indent=2), encoding="utf-8")
