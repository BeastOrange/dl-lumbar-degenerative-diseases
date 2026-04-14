"""Evaluation helpers."""

from dl_lumbar_dd.eval.comparison import RunSummary, build_ranking_table, save_ranking_table, scan_run_summaries, summarize_run
from dl_lumbar_dd.eval.metrics import compute_classification_metrics, safe_log_loss, safe_macro_auc

__all__ = [
    "RunSummary",
    "build_ranking_table",
    "compute_classification_metrics",
    "run_comparison",
    "run_evaluation",
    "safe_log_loss",
    "safe_macro_auc",
    "save_ranking_table",
    "scan_run_summaries",
    "summarize_run",
]


def __getattr__(name: str) -> object:
    if name in {"run_comparison", "run_evaluation"}:
        from dl_lumbar_dd.eval.commands import run_comparison, run_evaluation

        exports = {
            "run_comparison": run_comparison,
            "run_evaluation": run_evaluation,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
