"""Immutable observations of the existing training loop."""

from dataclasses import dataclass


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
