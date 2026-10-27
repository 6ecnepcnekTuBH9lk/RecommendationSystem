"""Immutable observations of the existing training loop."""

from dataclasses import dataclass
import numpy as np


def single_target_metrics(rank: int | None) -> tuple[float, float]:
    """Existing Recall/HitRate and NDCG contribution for one relevant item."""
    return (1.0, 1.0 / np.log2(rank + 1)) if rank is not None else (0.0, 0.0)


@dataclass(frozen=True)
class TrainingEpochMetrics:
    epoch: int
    loss: float
    recall: float
    ndcg: float


@dataclass(frozen=True)
class TrainingRunMetrics:
    epochs_completed: int
    best_epoch: int
    best_recall: float
    best_ndcg: float
    best_metric_name: str
    early_stopped: bool
    history: tuple[TrainingEpochMetrics, ...]

    def __post_init__(self):
        object.__setattr__(self, "history", tuple(self.history))
