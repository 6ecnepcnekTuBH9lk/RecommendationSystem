"""Scorer-neutral ranking on the one shared warm-user/warm-item case set."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
import numpy as np

from Application.model.training_metrics import single_target_metrics
from .temporal import TemporalProtocolError, TemporalSnapshot


class ScoringBackend(Protocol):
    def score(self, customer_id: str, candidate_items: tuple[str, ...]) -> Sequence[float]: ...


@dataclass(frozen=True)
class TemporalMetrics:
    cutoff: datetime
    k: int
    evaluated_users: int
    recall: float
    ndcg: float


def evaluate_snapshot(snapshot: TemporalSnapshot, scorer: ScoringBackend, k: int = 10) -> TemporalMetrics:
    if type(k) is not int or k < 1:
        raise TemporalProtocolError("Ranking K must be a positive integer")
    contributions = []
    for case in snapshot.cases:
        candidates = snapshot.candidates_for(case)
        if (case.target_item in case.seen_items or case.target_item not in candidates
                or case.seen_items != snapshot.seen_at_cutoff.get(case.customer_id)):
            raise TemporalProtocolError("Invalid temporal evaluation case")
        scores = np.asarray(scorer.score(case.customer_id, candidates), dtype=np.float64)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise TemporalProtocolError("Scorer must return one finite score per candidate")
        # Canonical sorted candidate IDs provide deterministic score ties across backends.
        ranking = np.argsort(-scores, kind="stable")[:k]
        target = candidates.index(case.target_item)
        positions = np.flatnonzero(ranking == target)
        contributions.append(single_target_metrics(int(positions[0]) + 1 if len(positions) else None))
    recall, ndcg = np.mean(contributions, axis=0) if contributions else (0.0, 0.0)
    return TemporalMetrics(snapshot.cutoff, k, len(contributions), float(recall), float(ndcg))
