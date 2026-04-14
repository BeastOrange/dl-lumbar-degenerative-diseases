"""Training exports."""

from dl_lumbar_dd.train.config import TrainingConfig, TrainingResult
from dl_lumbar_dd.train.trainer import Trainer

__all__ = ["Trainer", "TrainingConfig", "TrainingResult", "run_training"]


def __getattr__(name: str) -> object:
    if name == "run_training":
        from dl_lumbar_dd.train.commands import run_training

        return run_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
