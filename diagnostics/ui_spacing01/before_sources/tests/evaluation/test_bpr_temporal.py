from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest
import torch

from Application.evaluation.bpr import BprTemporalData, evaluate_bpr_snapshot, prepare_bpr_snapshot
from Application.evaluation.temporal import TemporalProtocolError, build_temporal_protocol
from Application.interactions import InteractionType as Kind
from Application.model import BPRMF as core
from Application.model.bpr_preparation import BprPreparationConfig, DateMode, prepare_bpr, to_bpr_event
from Application.model.training_metrics import single_target_metrics


def model_for(data):
    model = core.BPRMF(len(data.training.mappings.idx2user), len(data.training.mappings.idx2item), 1)
    with torch.no_grad():
        model.user_emb.weight.fill_(1)
        for item, index in data.training.mappings.item2idx.items():
            model.item_emb.weight[index].fill_({"A": 100, "B": 90, "C": 2, "D": 3, "E": 1}.get(item, 0))
    return model


def assert_preparation_equal(first, second):
    assert first.mappings.user2idx == second.mappings.user2idx
    assert first.mappings.item2idx == second.mappings.item2idx
    for name in ("train_pairs", "train_weights", "eval_users", "eval_items"):
        np.testing.assert_array_equal(getattr(first.splits, name), getattr(second.splits, name))
    assert first.splits.user_pos_train == second.splits.user_pos_train


def test_all_history_without_double_holdout(event, config):
    records = [event("u", f"item{index}", index / 2) for index in range(11)]
    records += [event("witness", "target", 1), event("u", "target", 11)]
    snapshot = build_temporal_protocol(records, config).validation
    legacy = prepare_bpr((to_bpr_event(e) for e in snapshot.history),
                         BprPreparationConfig(date_mode=DateMode.FULL_TIMESTAMP))
    assert len(legacy.splits.eval_users) == 1
    data = prepare_bpr_snapshot(snapshot)
    assert len(data.training.splits.eval_users) == len(data.training.splits.eval_items) == 0
    assert len(data.training.splits.train_pairs) == 12
    assert data.training.diagnostics.train_events_before_aggregation == len(snapshot.history)
    user = data.training.mappings.user2idx["u"]
    assert len(data.training.splits.user_pos_train[user]) == 11
    assert data.training.mappings.item2idx["item10"] in data.training.splits.user_pos_train[user]


def test_weights_and_all_validation_period_history(event, config):
    records = [event("u", "A", 1, Kind.PURCHASE, Decimal("2.5")), event("u", "A", 2),
               event("other", "B", 1), event("u", "A", 11, Kind.FAVORITE), event("u", "B", 12),
               event("u", "B", 13), event("u", "new", 22, Kind.PURCHASE, Decimal("100"))]
    protocol = build_temporal_protocol(records, config)
    validation, test = prepare_bpr_snapshot(protocol.validation), prepare_bpr_snapshot(protocol.test)

    def weights(data):
        maps = data.training.mappings
        return {(maps.idx2user[int(u)], maps.idx2item[int(i)]): float(w)
                for (u, i), w in zip(data.training.splits.train_pairs, data.training.splits.train_weights)}

    assert weights(validation)[("u", "A")] == pytest.approx(25.1)
    assert weights(test)[("u", "A")] == pytest.approx(27.1)
    assert weights(test)[("u", "B")] == pytest.approx(0.2)
    assert "new" not in test.training.mappings.item2idx
    assert validation.training.splits.user_pos_train[validation.training.mappings.user2idx["u"]] == {
        validation.training.mappings.item2idx["A"]}


def test_future_cannot_affect_bpr_mappings_pairs_weights_or_seen(event, config):
    past = [event("u", "A", 1), event("u", "A", 2), event("other", "B", 1)]
    first = build_temporal_protocol(past, config)
    second = build_temporal_protocol(past + [event("new_user", "new_item", 22, Kind.PURCHASE, Decimal("100"))], config)
    for name in ("validation", "test"):
        a, b = prepare_bpr_snapshot(getattr(first, name)), prepare_bpr_snapshot(getattr(second, name))
        assert_preparation_equal(a.training, b.training)
        assert "new_user" not in b.training.mappings.user2idx
        assert "new_item" not in b.training.mappings.item2idx


@pytest.mark.parametrize("k", [1, 2, 100])
def test_bpr_adapter_reuses_existing_metric_math_and_seen_mask(protocol, k):
    data = prepare_bpr_snapshot(protocol.validation)
    model = model_for(data)
    model.train()
    result = evaluate_bpr_snapshot(model, data, k)
    assert model.training
    maps = data.training.mappings
    external_cases_for_comparison = replace(data.training.splits,
        eval_users=np.array([maps.user2idx[c.customer_id] for c in data.snapshot.cases], dtype=np.int64),
        eval_items=np.array([maps.item2idx[c.target_item] for c in data.snapshot.cases], dtype=np.int64))
    model.eval()
    expected = core._eval_bprmf_recall_ndcg(model, external_cases_for_comparison,
                                          len(maps.idx2item), k, torch.device("cpu"))
    assert (result.recall, result.ndcg) == expected
    assert result.evaluated_users == len(data.snapshot.cases)
    assert result.recall == int(k >= 2)


def test_real_short_training_on_both_snapshots_without_target_input(protocol, tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(core, "_save_artifacts", lambda *args, **kwargs: pytest.fail("Research must not publish"))
    monkeypatch.setattr(core, "_read_csv_pipe", lambda *args: pytest.fail("No source reads in synthetic training"))
    cfg = core.TrainConfig(data_dir=str(tmp_path), use_item_features=False, early_stop=False,
                           epochs=1, embedding_dim=4, batch_size=3, n_neg=1, topk=2)
    for snapshot in (protocol.validation, protocol.test):
        data = prepare_bpr_snapshot(snapshot)
        core._set_seed(cfg.seed)
        model, splits = core.train_prepared_data(cfg, data.training, torch.device("cpu"))
        assert splits is data.training.splits
        assert len(splits.eval_users) == 0
        result = evaluate_bpr_snapshot(model, data, 2)
        assert result.evaluated_users == 1
        assert np.isfinite([result.recall, result.ndcg]).all()
    assert list(tmp_path.iterdir()) == []


def test_reject_internal_holdout_or_wrong_model_dimensions(protocol):
    data = prepare_bpr_snapshot(protocol.validation)
    maps = data.training.mappings
    splits = replace(data.training.splits, eval_users=np.array([maps.user2idx["u"]]),
                     eval_items=np.array([maps.item2idx["C"]]))
    invalid = BprTemporalData(data.snapshot, replace(data.training, splits=splits))
    with pytest.raises(TemporalProtocolError):
        evaluate_bpr_snapshot(model_for(data), invalid)
    with pytest.raises(TemporalProtocolError):
        evaluate_bpr_snapshot(core.BPRMF(len(maps.idx2user) + 1, len(maps.idx2item), 1), data)


def test_reject_inconsistent_snapshot_seen_or_item_universe(protocol):
    data = prepare_bpr_snapshot(protocol.validation)
    for snapshot in (replace(data.snapshot, item_universe=("A",)),
                     replace(data.snapshot, seen_at_cutoff={"u": frozenset({"B"}), "other": frozenset({"C", "D", "E"})})):
        with pytest.raises(TemporalProtocolError):
            evaluate_bpr_snapshot(model_for(data), BprTemporalData(snapshot, data.training))


@pytest.mark.parametrize("training", [True, False])
def test_model_mode_restored_on_scoring_error(protocol, monkeypatch, training):
    data = prepare_bpr_snapshot(protocol.validation)
    model = model_for(data)
    model.train(training)

    def fail(*args):
        assert not model.training
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(model, "score", fail)
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        evaluate_bpr_snapshot(model, data)
    assert model.training is training


def test_empty_bpr_history_is_explicit_error(config):
    with pytest.raises(TemporalProtocolError, match="empty"):
        prepare_bpr_snapshot(build_temporal_protocol([], config).validation)


@pytest.mark.parametrize("rank,expected", [(None, (0, 0)), (1, (1, 1)), (2, (1, 1 / np.log2(3)))])
def test_shared_single_target_metric_formula(rank, expected):
    assert single_target_metrics(rank) == expected
