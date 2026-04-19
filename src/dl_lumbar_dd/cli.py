"""Project CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dl_lumbar_dd.config import load_yaml
from dl_lumbar_dd.constants import ARTIFACTS_DIR, DEFAULT_DATASET_ROOT, REPORTS_FIGURES_DIR
from dl_lumbar_dd.data.commands import run_eda, run_preprocess
from dl_lumbar_dd.eval.commands import run_comparison, run_evaluation
from dl_lumbar_dd.healthcheck import run_healthcheck
from dl_lumbar_dd.train.commands import run_cv_training, run_training


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for data, training, and evaluation workflows."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    result = args.handler(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumbar-cli", description="Lumbar DD pipeline command line")
    subparsers = parser.add_subparsers(dest="command")

    health_parser = subparsers.add_parser("healthcheck", help="Check runtime readiness")
    health_parser.add_argument("--target", choices=["mac", "linux", "windows"], required=True)
    health_parser.set_defaults(handler=_handle_healthcheck)

    eda_parser = subparsers.add_parser("eda", help="Generate dataset EDA outputs")
    eda_parser.add_argument("--config", default="configs/data/default.yaml")
    eda_parser.add_argument("--dataset-root")
    eda_parser.add_argument("--figures-root")
    eda_parser.add_argument("--metadata-root")
    eda_parser.add_argument("--max-studies", type=int)
    eda_parser.set_defaults(handler=_handle_eda)

    preprocess_parser = subparsers.add_parser("preprocess", help="Generate study-level manifests and QA plots")
    preprocess_parser.add_argument("--config", default="configs/data/default.yaml")
    preprocess_parser.add_argument("--dataset-root")
    preprocess_parser.add_argument("--output-root")
    preprocess_parser.add_argument("--figures-root")
    preprocess_parser.add_argument("--metadata-root")
    preprocess_parser.add_argument("--max-studies", type=int)
    preprocess_parser.add_argument("--seed", type=int)
    preprocess_parser.add_argument("--folds", type=int)
    preprocess_parser.add_argument("--train-ratio", type=float)
    preprocess_parser.add_argument("--image-size", type=int)
    preprocess_parser.set_defaults(handler=_handle_preprocess)

    train_parser = subparsers.add_parser("train", help="Train one model configuration")
    train_parser.add_argument("--config", default="configs/train/default.yaml")
    train_parser.add_argument("--cv", action="store_true", help="Run cross-validation ensemble")
    train_parser.set_defaults(handler=_handle_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate one run and generate figures")
    evaluate_parser.add_argument("--run-dir")
    evaluate_parser.add_argument("--output-root", default=str(REPORTS_FIGURES_DIR / "eval"))
    evaluate_parser.set_defaults(handler=_handle_evaluate)

    compare_parser = subparsers.add_parser("compare", help="Build ranking across all runs")
    compare_parser.add_argument("--runs-root", default=str(ARTIFACTS_DIR / "runs"))
    compare_parser.add_argument("--output-root", default=str(REPORTS_FIGURES_DIR))
    compare_parser.add_argument("--primary-metric", default="val_macro_f1")
    compare_parser.set_defaults(handler=_handle_compare)
    return parser


def _handle_healthcheck(args: argparse.Namespace) -> dict[str, Any]:
    return run_healthcheck(args.target)


def _handle_eda(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    return run_eda(
        dataset_root=_choose_path(args.dataset_root, config.get("dataset_root"), str(DEFAULT_DATASET_ROOT)),
        figures_root=_choose_path(args.figures_root, config.get("figures_root"), str(REPORTS_FIGURES_DIR)),
        metadata_root=_choose_path(args.metadata_root, config.get("metadata_root"), str(ARTIFACTS_DIR / "metadata")),
        max_studies=args.max_studies or _optional_int(config.get("max_studies")),
    )


def _handle_preprocess(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    return run_preprocess(
        dataset_root=_choose_path(args.dataset_root, config.get("dataset_root"), str(DEFAULT_DATASET_ROOT)),
        output_root=_choose_path(args.output_root, config.get("output_root"), str(ARTIFACTS_DIR / "processed")),
        figures_root=_choose_path(args.figures_root, config.get("figures_root"), str(REPORTS_FIGURES_DIR)),
        metadata_root=_choose_path(args.metadata_root, config.get("metadata_root"), str(ARTIFACTS_DIR / "metadata")),
        max_studies=args.max_studies or _optional_int(config.get("max_studies")),
        seed=args.seed if args.seed is not None else int(config.get("seed", 42)),
        folds=args.folds if args.folds is not None else int(config.get("folds", 3)),
        train_ratio=(
            args.train_ratio
            if args.train_ratio is not None
            else float(config.get("split", {}).get("train_ratio", 0.8))
        ),
        image_size=args.image_size if args.image_size is not None else int(config.get("image_size", 224)),
    )


def _handle_train(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "cv", False):
        return run_cv_training(args.config)
    return run_training(args.config)


def _handle_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError("No run_dir specified and no runs found in artifacts/runs")
    return run_evaluation(run_dir, args.output_root)


def _handle_compare(args: argparse.Namespace) -> dict[str, Any]:
    return run_comparison(
        runs_root=args.runs_root,
        output_root=args.output_root,
        primary_metric=args.primary_metric,
    )


def _latest_run_dir() -> Path | None:
    runs_root = ARTIFACTS_DIR / "runs"
    if not runs_root.exists():
        return None
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _choose_path(cli_value: str | None, config_value: Any, fallback: str) -> str:
    if cli_value:
        return cli_value
    if isinstance(config_value, str) and config_value.strip():
        return config_value
    return fallback


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
