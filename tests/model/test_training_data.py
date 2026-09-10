from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from Application.model import BPRMF as bpr
from Application.model.training_data import Mappings, Splits, PreparedBprData, PreparedDataError, validate_prepared_data
from Application.model.bpr_preparation import BprMappings, BprSplits, prepare_bpr


@pytest.fixture
def prepared():
    return PreparedBprData(
        Mappings({"u0": 0, "u1": 1}, ["u0", "u1"], {"i0": 0, "i1": 1, "i2": 2}, ["i0", "i1", "i2"]),
        Splits(np.array([[0, 0], [1, 1]]), np.array([0.1, 10.0]), np.array([0]), np.array([2]), [{0}, {1}]),
    )


def test_shared_reexports_and_source_neutral_result():
    assert bpr.Mappings is BprMappings is Mappings
    assert bpr.Splits is BprSplits is Splits
    assert isinstance(prepare_bpr([]), PreparedBprData)


def test_controlled_mutability_is_explicit_and_revalidated(prepared):
    validate_prepared_data(prepared)
    weights = prepared.splits.train_weights
    alias = PreparedBprData(prepared.mappings, prepared.splits)
    weights[0] = 0
    assert alias.splits.train_weights is weights
    validate_prepared_data(alias)  # zero weight is legal in legacy loss
    weights[0] = np.nan
    with pytest.raises(PreparedDataError):
        validate_prepared_data(alias)
    assert "u0" not in repr(prepared) + repr(prepared.mappings) + repr(prepared.splits)


@pytest.mark.parametrize("field,value", [
    ("train_pairs", np.array([[2, 0]])), ("train_pairs", np.array([[0, 3], [1, 1]])),
    ("train_pairs", np.array([[-1, 0], [1, 1]])), ("train_pairs", np.array([[0., 0.], [1., 1.]])),
    ("train_pairs", np.array([0, 1])), ("train_pairs", np.zeros((2, 3), dtype=int)),
    ("train_weights", np.array([1.])), ("train_weights", np.array([np.nan, 1.])),
    ("train_weights", np.array([np.inf, 1.])), ("train_weights", np.array([-1., 1.])),
    ("train_weights", np.array([[1., 1.]])), ("eval_users", np.array([2])),
    ("eval_items", np.array([3])), ("eval_items", np.array([-1])),
    ("eval_items", np.array([1, 2])), ("eval_users", np.array([0.])),
    ("user_pos_train", [{0}]), ("user_pos_train", [{2}, {1}]),
])
def test_invalid_splits(prepared, field, value):
    broken = replace(prepared, splits=replace(prepared.splits, **{field: value}))
    with pytest.raises(PreparedDataError):
        validate_prepared_data(broken)


@pytest.mark.parametrize("field,value", [
    ("user2idx", {"u0": 1, "u1": 0}), ("user2idx", {"u0": 0}),
    ("idx2user", ["u0", "u0"]), ("item2idx", {"i0": 0, "i1": 1, "i2": 9}),
    ("idx2item", ["i0", "i1"]), ("user2idx", {"u0": False, "u1": 1}),
])
def test_invalid_mappings(prepared, field, value):
    with pytest.raises(PreparedDataError):
        validate_prepared_data(replace(prepared, mappings=replace(prepared.mappings, **{field: value})))


def test_empty_and_wrong_input():
    for data in (None, object(), prepare_bpr([])):
        with pytest.raises(PreparedDataError):
            validate_prepared_data(data)


def write_csv(tmp_path):
    frames = [pd.DataFrame({"MindboxID": ["u0", "u1"], "КодНоменклатуры": ["i0", "i1"],
                            "Количество": ["1.5", "2"], "Дата": ["2026-01-01", "2026-01-02"]}),
              pd.DataFrame({"MindboxID": ["u0"], "КодНоменклатуры": ["i2"],
                            "ТипТовара": ["Номенклатура"], "Дата": ["2026-01-03"]}),
              pd.DataFrame(columns=["MindboxID", "КодНоменклатуры", "Дата"])]
    for name, frame in zip(("Заказы", "Просмотры", "Избранное"), frames):
        frame.to_csv(tmp_path / f"{name}.csv", sep="|", encoding="utf-8-sig", index=False)
    return frames


def assert_splits_equal(left, right):
    for name in ("train_pairs", "train_weights", "eval_users", "eval_items"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    assert left.user_pos_train == right.user_pos_train


def test_csv_preparation_and_public_entrypoint_parity(tmp_path, monkeypatch):
    frames = write_csv(tmp_path)
    cfg = bpr.TrainConfig(data_dir=str(tmp_path), min_user_interactions_for_eval=2)
    maps = bpr._build_mappings(*frames)
    events = bpr._collect_user_item_events(*frames, maps, cfg)
    expected = bpr._train_test_split_last_per_user(events, cfg, len(maps.idx2user))
    data = bpr.prepare_training_data_from_csv(cfg)
    assert data.mappings == maps
    assert_splits_equal(data.splits, expected)
    calls = []
    real_prepare = bpr.prepare_training_data_from_csv

    def prepare(config):
        calls.append("prepare")
        return real_prepare(config)

    def train(config, actual, device):
        calls.append("train")
        assert actual.mappings == maps
        assert_splits_equal(actual.splits, expected)
        return object(), actual.splits

    monkeypatch.setattr(bpr, "prepare_training_data_from_csv", prepare)
    monkeypatch.setattr(bpr, "train_prepared_data", train)
    monkeypatch.setattr(bpr, "_save_artifacts", lambda *args: calls.append("save"))
    assert bpr._train_in_this_process(cfg)
    assert calls == ["prepare", "train", "save"]


@pytest.mark.parametrize("features", [False, True])
def test_prepared_core_without_interaction_csv(tmp_path, monkeypatch, prepared, features):
    cfg = bpr.TrainConfig(data_dir=str(tmp_path), use_item_features=features, epochs=1,
                          embedding_dim=4, batch_size=2, n_neg=1, topk=2)
    if features:
        pd.DataFrame({"КодНоменклатуры": ["i0", "i1", "i2"], "Марка": ["a", "b", "a"]}).to_csv(
            tmp_path / "Номенклатура.csv", sep="|", index=False, encoding="utf-8-sig")
    reads = []
    original = bpr._read_csv_pipe

    def read(path):
        assert str(path).endswith("Номенклатура.csv")
        reads.append(path)
        return original(path)

    def forbidden(*args, **kwargs):
        pytest.fail("Prepared core must not prepare interaction sources")

    monkeypatch.setattr(bpr, "_read_csv_pipe", read)
    for name in ("_require_interaction_sources", "_validate_interaction_source_schemas",
                 "_collect_user_item_events", "_train_test_split_last_per_user", "_save_artifacts"):
        monkeypatch.setattr(bpr, name, forbidden)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    bpr._set_seed(cfg.seed)
    model, splits = bpr.train_prepared_data(cfg, prepared, torch.device("cpu"))
    assert splits is prepared.splits
    assert bool(reads) == features
    assert model.use_item_features == features
    assert len(list(tmp_path.iterdir())) == int(features)


def test_short_training_legacy_wrapper_vs_prepared(tmp_path, monkeypatch):
    frames = write_csv(tmp_path)
    cfg = bpr.TrainConfig(data_dir=str(tmp_path), use_item_features=False, epochs=2,
                          embedding_dim=4, batch_size=3, n_neg=1, topk=2)
    maps = bpr._build_mappings(*frames)
    events = bpr._collect_user_item_events(*frames, maps, cfg)
    data = bpr.prepare_training_data_from_csv(cfg)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    bpr._set_seed(cfg.seed)
    old_model, old_split = bpr.train_bprmf(maps, events, cfg, torch.device("cpu"))
    bpr._set_seed(cfg.seed)
    new_model, new_split = bpr.train_prepared_data(cfg, data, torch.device("cpu"))
    assert_splits_equal(old_split, new_split)
    for name, value in old_model.state_dict().items():
        torch.testing.assert_close(value, new_model.state_dict()[name])
    assert not (tmp_path / "Модель").exists()


def test_validation_precedes_training_side_effects(prepared, monkeypatch):
    prepared.splits.train_weights[0] = np.inf
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: pytest.fail("Must validate first"))
    with pytest.raises(PreparedDataError):
        bpr.train_prepared_data(bpr.TrainConfig(), prepared, torch.device("cpu"))
