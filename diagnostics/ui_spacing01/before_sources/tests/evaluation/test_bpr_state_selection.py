"""Real optimizer steps on tiny data; independent snapshots expose CPU aliases."""

import numpy as np
import pytest
import torch

from Application.model import BPRMF as core
from Application.model.training_data import Mappings, PreparedBprData, Splits


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def train_tiny(monkeypatch, tmp_path, *, early_stop, internal_eval, copied_cpu=False, metric="ndcg", device="cpu"):
    # CUDA -> CPU makes an independent copy. Emulate only this copy behavior on
    # CPU so the disabled-stopping regression is covered without requiring GPU.
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    if copied_cpu:
        monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *args, **kwargs: self.clone())
    prepared = PreparedBprData(
        Mappings({"u0": 0, "u1": 1}, ["u0", "u1"],
                 {"i0": 0, "i1": 1, "i2": 2}, ["i0", "i1", "i2"]),
        Splits(np.array([[0, 0], [1, 1]]), np.array([.1, 10.]),
               np.array([0] if internal_eval else [], dtype=np.int64),
               np.array([2] if internal_eval else [], dtype=np.int64), [{0}, {1}]))
    cfg = core.TrainConfig(data_dir=str(tmp_path), epochs=4, embedding_dim=4,
                           batch_size=2, n_neg=2, lr=.03, seed=42, topk=2,
                           use_item_features=False, early_stop=early_stop,
                           early_stop_metric=metric, early_stop_patience=2,
                           early_stop_min_epochs=1)
    epochs, loads = [], []
    original_eval = core._eval_bprmf_recall_ndcg
    original_load = core.BPRMF.load_state_dict

    def evaluate(model, *args):
        epochs.append(snapshot(model))
        # NDCG is best at epoch 1; Recall at epoch 2. Later values worsen.
        if internal_eval:
            return [(.1, .8), (.9, .7), (.8, .6), (.7, .5)][len(epochs) - 1]
        return original_eval(model, *args)

    def load(model, state, *args, **kwargs):
        loads.append({name: value.clone() for name, value in state.items()})
        return original_load(model, state, *args, **kwargs)

    monkeypatch.setattr(core, "_eval_bprmf_recall_ndcg", evaluate)
    monkeypatch.setattr(core.BPRMF, "load_state_dict", load)
    core._set_seed(cfg.seed)
    model, _, metrics = core.train_prepared_data_with_metrics(cfg, prepared, torch.device(device))
    assert not torch.equal(epochs[0]["user_emb.weight"], epochs[-1]["user_emb.weight"])
    return model, metrics, epochs, loads, cfg, prepared


@pytest.mark.parametrize("internal_eval", [False, True])
@pytest.mark.parametrize("copied_cpu", [False, True])
def test_disabled_stopping_returns_final_without_restore(monkeypatch, tmp_path, internal_eval, copied_cpu):
    model, metrics, epochs, loads, _, _ = train_tiny(
        monkeypatch, tmp_path, early_stop=False, internal_eval=internal_eval, copied_cpu=copied_cpu)
    assert metrics.epochs_completed == 4 and not metrics.early_stopped
    assert_state(snapshot(model), epochs[-1])
    assert loads == []


@pytest.mark.parametrize("metric,best_epoch,completed", [("ndcg", 1, 3), ("recall", 2, 4)])
def test_enabled_stopping_restores_independent_best_state(monkeypatch, tmp_path, metric, best_epoch, completed):
    model, metrics, epochs, loads, _, _ = train_tiny(
        monkeypatch, tmp_path, early_stop=True, internal_eval=True, metric=metric)
    assert metrics.early_stopped and metrics.epochs_completed == completed
    assert metrics.best_epoch == best_epoch
    assert len(loads) == 1
    assert_state(snapshot(model), epochs[best_epoch - 1])


def test_empty_eval_disables_monitoring_and_returns_final(monkeypatch, tmp_path):
    model, metrics, epochs, loads, _, _ = train_tiny(
        monkeypatch, tmp_path, early_stop=True, internal_eval=False, copied_cpu=True)
    assert metrics.epochs_completed == 4 and not metrics.early_stopped
    assert (metrics.best_epoch, metrics.best_recall, metrics.best_ndcg) == (-1, -1., -1.)
    assert loads == []
    assert_state(snapshot(model), epochs[-1])


@pytest.mark.parametrize("early_stop", [False, True])
def test_artifact_serializes_selected_state_only_in_temporary_directory(monkeypatch, tmp_path, early_stop):
    model, _, epochs, _, cfg, prepared = train_tiny(
        monkeypatch, tmp_path, early_stop=early_stop, internal_eval=early_stop, copied_cpu=True)
    expected = epochs[0] if early_stop else epochs[-1]
    artifact_dir = tmp_path / "synthetic_artifacts"
    generation = core._save_artifacts(cfg, prepared.mappings, model, model_dir=str(artifact_dir))
    checkpoint = torch.load(artifact_dir / "runs" / generation / "bprmf.pt", weights_only=False)
    assert_state(checkpoint["state_dict"], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("internal_eval", [False, True])
def test_cuda_disabled_stopping_returns_actual_final_state(monkeypatch, tmp_path, internal_eval):
    model, metrics, epochs, loads, _, _ = train_tiny(
        monkeypatch, tmp_path, early_stop=False, internal_eval=internal_eval, device="cuda")
    assert metrics.epochs_completed == 4 and not metrics.early_stopped
    assert model.user_emb.weight.device.type == "cuda"
    assert all(torch.isfinite(value).all().item() for value in model.state_dict().values())
    assert all(np.isfinite(epoch.loss) for epoch in metrics.history)
    assert loads == []
    # Compare tensors from this GPU run only, not CPU/GPU arithmetic.
    assert_state(snapshot(model), epochs[-1])
