"""Synthetic tests only: observation must preserve the continuous final-state baseline."""
from dataclasses import asdict, replace

import pytest
import torch

from Application.evaluation.bpr import prepare_bpr_snapshot
from Application.evaluation.temporal import TemporalProtocolError
from scripts import run_bpr_convergence as conv


@pytest.fixture
def cfg(tmp_path):
    return conv.core.TrainConfig(data_dir=str(tmp_path), epochs=5, embedding_dim=4,
                                 batch_size=2, n_neg=2, lr=.03, seed=42,
                                 early_stop=False, use_item_features=False)


def state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def forbid(*args, **kwargs):
    pytest.fail("Forbidden training/scoring/publication")


def test_baseline_changes_only_epoch_budget():
    cfg = conv.core.TrainConfig(epochs=200, early_stop=False, use_item_features=False)
    frozen = asdict(cfg)
    plan = {"hyperparameters": frozen, "temporal_config": conv.temporal_dict(conv.benchmark_config())}
    assert asdict(conv.baseline_config(plan)) == {**frozen, "epochs": 30}
    assert plan["hyperparameters"] == frozen


@pytest.mark.parametrize("field,value", [("batch_size", 512), ("w_view_item", .5),
                                         ("lr", .003), ("early_stop", True), ("seed", 43)])
def test_baseline_rejects_variants(field, value):
    cfg = conv.core.TrainConfig(epochs=200, early_stop=False, use_item_features=False)
    plan = {"hyperparameters": asdict(replace(cfg, **{field: value})),
            "temporal_config": conv.temporal_dict(conv.benchmark_config())}
    with pytest.raises(ValueError, match="frozen historical"):
        conv.baseline_config(plan)


@pytest.mark.parametrize("observer_kind", ["noop", "read"])
def test_observer_preserves_bit_exact_cpu_final_state(cfg, protocol, monkeypatch, observer_kind):
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    data = prepare_bpr_snapshot(protocol.validation)
    conv.core._set_seed(cfg.seed)
    reference, _, reference_metrics = conv.core.train_prepared_data_with_metrics(cfg, data.training, torch.device("cpu"))
    seen, captured = [], []
    def observe(model, epoch):
        assert not torch.is_grad_enabled()
        seen.append(epoch.epoch)
        if observer_kind == "read":
            assert all(torch.isfinite(p).all().item() for p in model.parameters())
            captured.append((state(model), epoch.loss))
    conv.core._set_seed(cfg.seed)
    observed, _, observed_metrics = conv.core.train_prepared_data_with_metrics(
        cfg, data.training, torch.device("cpu"), epoch_observer=observe)
    assert seen == [1, 2, 3, 4, 5]
    assert observed_metrics == reference_metrics
    assert_state(state(observed), state(reference))
    if captured:
        assert_state(state(observed), captured[-1][0])


def test_checkpoint_order_continuous_optimizer_and_final_state(cfg, protocol, config, monkeypatch):
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(conv.core, "_save_artifacts", forbid)
    data = prepare_bpr_snapshot(protocol.validation)
    conv.core._set_seed(cfg.seed)
    baseline, _, _ = conv.core.train_prepared_data_with_metrics(cfg, data.training, torch.device("cpu"))
    expected_final = state(baseline)
    calls, captures, optimizers, rows, returned = [], [], [], [], []
    original_score = conv.score_checkpoint
    original_step = torch.optim.Adam.step
    original_train = conv.core.train_prepared_data_with_metrics
    def score(model, *args):
        assert not torch.is_grad_enabled()
        calls.append(len(rows) + 1)
        captures.append(state(model))
        return original_score(model, *args)
    def step(optimizer, *args, **kwargs):
        optimizers.append(id(optimizer))
        return original_step(optimizer, *args, **kwargs)
    def train(*args, **kwargs):
        result = original_train(*args, **kwargs)
        returned.append(state(result[0]))
        return result
    monkeypatch.setattr(conv, "score_checkpoint", score)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    monkeypatch.setattr(conv.core, "train_prepared_data_with_metrics", train)
    result = conv.train_trajectory(protocol.validation, data.training, cfg, config,
                                    torch.device("cpu"), (1, 3, 5), rows.append)
    assert calls == [1, 3, 5]
    assert len(set(optimizers)) == 1 and len(returned) == 1
    assert result["epochs_completed"] == result["returned_state_epoch"] == 5
    assert result["internal_best_epoch"] == -1
    assert_state(returned[0], captures[-1])
    assert_state(returned[0], expected_final)
    assert not torch.equal(returned[0]["user_emb.weight"], captures[0]["user_emb.weight"])
    assert [r["epoch"] for r in rows if "validation" in r] == [1, 3, 5]
    assert all(r["training_seconds"] > 0 for r in rows)
    assert result["training_seconds"] == sum(r["training_seconds"] for r in rows)
    assert not result["test_performance_computed"] and not result["publication_executed"]


def test_test_snapshot_rejected_before_training_or_scoring(cfg, protocol, config, monkeypatch):
    data = prepare_bpr_snapshot(protocol.validation)
    monkeypatch.setattr(conv.core, "train_prepared_data_with_metrics", forbid)
    monkeypatch.setattr(conv, "evaluate_bpr_snapshot", forbid)
    with pytest.raises(TemporalProtocolError, match="only.*validation"):
        conv.train_trajectory(protocol.test, data.training, cfg, config, torch.device("cpu"), (1, 3, 5), forbid)
    with pytest.raises(TemporalProtocolError, match="only.*validation"):
        conv.score_checkpoint(None, protocol.test, data.training, config, torch.device("cpu"))


@pytest.mark.parametrize("checkpoints", [(0, 3, 5), (1, 3, 6), (1, 3), (3, 1, 5), (1, 1, 5)])
def test_invalid_checkpoints_rejected_before_trainer(cfg, protocol, config, monkeypatch, checkpoints):
    data = prepare_bpr_snapshot(protocol.validation)
    monkeypatch.setattr(conv.core, "train_prepared_data_with_metrics", forbid)
    with pytest.raises(ValueError, match="Checkpoints"):
        conv.train_trajectory(protocol.validation, data.training, cfg, config,
                               torch.device("cpu"), checkpoints, forbid)


def test_existing_evaluator_used_without_state_mode_or_gradient_changes(cfg, protocol, config, monkeypatch):
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    data = prepare_bpr_snapshot(protocol.validation)
    conv.core._set_seed(cfg.seed)
    model, _, _ = conv.core.train_prepared_data_with_metrics(cfg, data.training, torch.device("cpu"))
    before = state(model)
    mode = model.training
    result = conv.score_checkpoint(model, protocol.validation, data.training, config, torch.device("cpu"))
    assert_state(state(model), before)
    assert model.training == mode
    assert result["cases"] == len(protocol.validation.cases)
    assert set(result["metrics"]["overall"]) == {"5", "10", "20"}
    assert set(result["metrics"]) == {"overall", "VIEW", "FAVORITE", "PURCHASE"}
    for k in (5, 10, 20):
        expected = conv.evaluate_bpr_snapshot(model, data, k, torch.device("cpu"))
        assert result["metrics"]["overall"][str(k)] == {
            "cases": expected.evaluated_users, "ndcg": expected.ndcg, "recall": expected.recall}


def test_existing_output_refuses_automatic_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(conv, "OUTPUT_DIR", tmp_path)
    (tmp_path / "results.json").write_text("{}")
    monkeypatch.setattr(conv, "load_canonical_events", forbid)
    with pytest.raises(RuntimeError, match="automatic repeat refused"):
        conv.main()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_actual_cuda_observation_returns_final_checkpoint(cfg, protocol, config, monkeypatch):
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(conv.core, "_save_artifacts", forbid)
    data = prepare_bpr_snapshot(protocol.validation)
    returned, checkpoints = [], []
    original_train, original_score = conv.core.train_prepared_data_with_metrics, conv.score_checkpoint
    def train(*args, **kwargs):
        result = original_train(*args, **kwargs)
        returned.append(state(result[0]))
        return result
    def score(model, *args):
        checkpoints.append(state(model))
        return original_score(model, *args)
    monkeypatch.setattr(conv.core, "train_prepared_data_with_metrics", train)
    monkeypatch.setattr(conv, "score_checkpoint", score)
    result = conv.train_trajectory(protocol.validation, data.training, cfg, config,
                                    torch.device("cuda"), (1, 3, 5), lambda row: None)
    assert result["device"] == "cuda:0"
    assert result["cuda_memory"]["peak_reserved_bytes"] >= result["cuda_memory"]["peak_allocated_bytes"] > 0
    assert result["finite_loss_and_parameters"]
    assert len(checkpoints) == 3 and len(returned) == 1
    assert_state(returned[0], checkpoints[-1])
    assert not torch.equal(returned[0]["user_emb.weight"], checkpoints[0]["user_emb.weight"])
