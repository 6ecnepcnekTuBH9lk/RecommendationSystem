"""Explicit BPR adapter; targets stay external and cannot drive trainer early stopping."""

from dataclasses import dataclass
import torch

from Application.model.bpr_preparation import (
    BprPreparationConfig, BprWeightConfig, DateMode, prepare_bpr_training_history, to_bpr_event,
)
from Application.model.training_data import PreparedBprData, validate_prepared_data
from .ranking import evaluate_snapshot
from .temporal import TemporalProtocolError, TemporalSnapshot


@dataclass(frozen=True, repr=False)
class BprTemporalData:
    snapshot: TemporalSnapshot
    training: PreparedBprData


def prepare_bpr_snapshot(snapshot: TemporalSnapshot, weights: BprWeightConfig = BprWeightConfig()) -> BprTemporalData:
    if not snapshot.history:
        raise TemporalProtocolError("BPR snapshot training history is empty")
    prepared = prepare_bpr_training_history(
        (to_bpr_event(event, weights) for event in snapshot.history),
        BprPreparationConfig(weights=weights, date_mode=DateMode.FULL_TIMESTAMP),
    )
    validate_prepared_data(prepared)
    return BprTemporalData(snapshot, prepared)


class _BprScorer:
    def __init__(self, model, mappings, device):
        self.model, self.mappings, self.device = model, mappings, device

    @torch.no_grad()
    def score(self, customer_id, candidate_items):
        users = torch.full((len(candidate_items),), self.mappings.user2idx[customer_id],
                           dtype=torch.long, device=self.device)
        items = torch.tensor([self.mappings.item2idx[item] for item in candidate_items],
                             dtype=torch.long, device=self.device)
        return self.model.score(users, items).detach().cpu().numpy()


def evaluate_bpr_snapshot(model, data: BprTemporalData, k: int = 10, device=torch.device("cpu")):
    validate_prepared_data(data.training)
    maps = data.training.mappings
    if (len(data.training.splits.eval_users) or set(maps.idx2item) != set(data.snapshot.item_universe)
            or set(maps.idx2user) != set(data.snapshot.seen_at_cutoff)
            or model.user_emb.num_embeddings != len(maps.idx2user)
            or model.item_emb.num_embeddings != len(maps.idx2item)):
        raise TemporalProtocolError("BPR model/data must match the historical snapshot")
    for user, seen in data.snapshot.seen_at_cutoff.items():
        actual = {maps.idx2item[int(i)] for i in data.training.splits.user_pos_train[maps.user2idx[user]]}
        if actual != seen:
            raise TemporalProtocolError("BPR training positives must match cutoff history")
    was_training = model.training
    model.eval()
    try:
        return evaluate_snapshot(data.snapshot, _BprScorer(model, maps, device), k)
    finally:
        model.train(was_training)
