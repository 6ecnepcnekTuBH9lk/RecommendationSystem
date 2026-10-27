"""Safety gates and lightweight counters for the one-epoch GPU cost pilot."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from Application.model.training_data import Mappings, PreparedBprData, Splits
from scripts import run_bpr_gpu_cost as cost


@pytest.fixture
def cfg(tmp_path):
    return cost.core.TrainConfig(data_dir=str(tmp_path), epochs=1,
                                 early_stop=False, use_item_features=False)


@pytest.fixture
def training():
    return PreparedBprData(
        Mappings({"u0": 0, "u1": 1}, ["u0", "u1"],
                 {"i0": 0, "i1": 1, "i2": 2}, ["i0", "i1", "i2"]),
        Splits(np.array([[0, 0], [1, 1]]), np.array([.1, 10.]),
               np.array([], dtype=np.int64), np.array([], dtype=np.int64), [{0}, {1}]))


def forbidden(*args, **kwargs):
    pytest.fail("Training/evaluation/publication must not occur")


def test_one_epoch_config_changes_only_epoch_budget(cfg):
    frozen = asdict(replace(cfg, epochs=200))
    plan = {"hyperparameters": frozen, "temporal_config": cost.temporal_dict(cost.benchmark_config())}
    observed = asdict(cost.one_epoch_config(plan))
    assert observed == {**frozen, "epochs": 1}
    assert plan["hyperparameters"] == frozen


@pytest.mark.parametrize("field,value", [
    ("epochs", 200), ("epochs", 2), ("seed", 43), ("early_stop", True),
    ("use_item_features", True), ("w_view_item", .5),
])
def test_reject_unsafe_budget_before_training(cfg, monkeypatch, field, value):
    monkeypatch.setattr(cost.core, "train_prepared_data_with_metrics", forbidden)
    with pytest.raises(ValueError, match="exactly one"):
        cost.measure_epoch(None, replace(cfg, **{field: value}))


def test_cuda_unavailable_never_falls_back_to_cpu(cfg, training, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cost.core, "train_prepared_data_with_metrics", forbidden)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        cost.measure_epoch(training, cfg)


def test_internal_eval_is_rejected_before_training(cfg, training, monkeypatch):
    training.splits.eval_users = np.array([0])
    training.splits.eval_items = np.array([2])
    monkeypatch.setattr(cost.core, "train_prepared_data_with_metrics", forbidden)
    with pytest.raises(ValueError, match="empty internal evaluation"):
        cost.measure_epoch(training, cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_one_cuda_epoch_counts_actual_steps_and_examples(cfg, training, monkeypatch):
    from Application.evaluation.experiments import bpr_weights as exp
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(cost.core, "_save_artifacts", forbidden)
    monkeypatch.setattr(exp, "evaluate_bpr_snapshot", forbidden)
    result = cost.measure_epoch(training, cfg)
    assert result["epochs_completed"] == 1
    assert result["optimizer_steps"] == 1 and result["examples_processed"] == 2
    assert result["finite_loss_and_parameters"] and result["device"] == "cuda:0"
    assert result["pairs_per_second"] == 2 / result["training_seconds"]
    assert result["cuda_memory"]["peak_reserved_bytes"] >= result["cuda_memory"]["peak_allocated_bytes"] > 0
    assert not result["quality_metrics_computed"] and not result["publication_executed"]

