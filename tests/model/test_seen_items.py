from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import torch

from Application.model import BPRMF as core
from Application.model.bpr_preparation import BprEvent, BprPreparationConfig, prepare_bpr
from Application.interactions import InteractionType
from Application.model.seen_items import SeenItemsIndex, SeenItemsError, build_seen_items_index, seen_items_from_checkpoint


@pytest.fixture
def events():
    # u1: eval-only c; u2: held-out a repeats a train positive; u3 ineligible.
    rows = [("u1", "a", "VIEW"), ("u1", "a", "VIEW"), ("u1", "a", "FAVORITE"),
            ("u1", "a", "PURCHASE"), ("u1", "c", "VIEW"),
            ("u2", "a", "VIEW"), ("u2", "b", "VIEW"), ("u2", "a", "PURCHASE"),
            ("u3", "b", "VIEW")]
    return [BprEvent(u, i, datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=n),
                     InteractionType[kind], {"VIEW": .1, "FAVORITE": 2., "PURCHASE": 10.}[kind])
            for n, (u, i, kind) in enumerate(rows)]


@pytest.fixture
def prepared(events):
    return prepare_bpr(events, BprPreparationConfig(min_user_interactions_for_eval=3))


def test_all_pre_holdout_pairs_and_immutable_deterministic_layout(events, prepared):
    index = build_seen_items_index(prepared)
    expected = {(e.customer_id, e.item_id) for e in events}
    maps = prepared.mappings
    actual = {(u, maps.idx2item[i]) for n, u in enumerate(maps.idx2user) for i in index.items_for_user(n)}
    assert actual == expected
    np.testing.assert_array_equal(index.indptr, [0, 2, 4, 5])
    np.testing.assert_array_equal(index.indices, [0, 1, 0, 2, 2])
    assert maps.item2idx["c"] not in prepared.splits.user_pos_train[maps.user2idx["u1"]]
    assert maps.item2idx["a"] in prepared.splits.user_pos_train[maps.user2idx["u2"]]
    assert "u1" not in repr(index)
    with pytest.raises(FrozenInstanceError):
        index.num_users = 0
    for array in (index.indptr, index.indices, index.items_for_user(0)):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    original = np.array([0, 1], dtype=np.int64)
    copied = SeenItemsIndex(1, 2, original, np.array([1], dtype=np.int64))
    original[1] = 0
    assert copied.indptr[1] == 1


@pytest.mark.parametrize("p,i", [([1, 1], [0]), ([0, 2], [0]), ([0, -1], []),
                                ([0, 1], [-1]), ([0, 1], [2]), ([0, 2], [0, 0]), ([0, 2], [1, 0])])
def test_invalid_layout(p, i):
    with pytest.raises(SeenItemsError):
        SeenItemsIndex(1, 2, np.array(p, dtype=np.int64), np.array(i, dtype=np.int64))


def test_dtype_dimensions_and_lookup_validation():
    with pytest.raises(SeenItemsError):
        SeenItemsIndex(3, 2, np.array([0, 1, 0, 1], dtype=np.int64), np.array([0], dtype=np.int64))
    with pytest.raises(SeenItemsError):
        SeenItemsIndex(1, 2, np.array([0., 1.]), np.array([0], dtype=np.int64))
    index = SeenItemsIndex(1, 2, np.array([0, 0], dtype=np.int64), np.array([], dtype=np.int64))
    for user in (-1, 1, True, .5):
        with pytest.raises(SeenItemsError):
            index.items_for_user(user)


def test_save_rejects_dimension_mismatch_before_publication(tmp_path, monkeypatch, prepared):
    monkeypatch.chdir(tmp_path)
    index = SeenItemsIndex(1, 3, np.array([0, 0], dtype=np.int64), np.array([], dtype=np.int64))
    with pytest.raises(SeenItemsError):
        core._save_artifacts(core.TrainConfig(), prepared.mappings, object(), seen_items=index)
    assert not (tmp_path / "Модель").exists()


@pytest.mark.parametrize("embedded", [False, True])
def test_artifact_roundtrip_and_print_without_csv(tmp_path, monkeypatch, prepared, capsys, embedded):
    monkeypatch.chdir(tmp_path)
    cfg = core.TrainConfig(data_dir=str(tmp_path), embedding_dim=1, use_item_features=False)
    model = core.BPRMF(3, 3, 1)
    with torch.no_grad():
        model.user_emb.weight.fill_(1)
        model.item_emb.weight.copy_(torch.tensor([[3.], [2.], [1.]]))
    index = build_seen_items_index(prepared)
    core._save_artifacts(cfg, prepared.mappings, model, seen_items=index if embedded else None)
    _, checkpoint = core._load_artifacts()
    loaded = seen_items_from_checkpoint(checkpoint)
    assert (loaded is not None) == embedded
    if embedded:
        np.testing.assert_array_equal(loaded.indices, index.indices)
    calls = []
    def legacy(*args):
        calls.append("legacy")
        return np.array([0, 1], dtype=np.int64)
    def forbidden(*args):
        pytest.fail("Embedded seen must not read interaction CSV")
    monkeypatch.setattr(core, "_user_seen_items_from_processed", forbidden if embedded else legacy)
    monkeypatch.setattr(core, "_build_user_seen_sets", forbidden)
    monkeypatch.setattr(core, "_require_interaction_sources", forbidden if embedded else lambda *a: None)
    monkeypatch.setattr(core, "_validate_interaction_source_schemas", forbidden if embedded else lambda *a: None)
    monkeypatch.setattr(core, "_read_csv_pipe", forbidden)
    monkeypatch.setattr(core, "_load_item_names", lambda *a: {})
    core.print_recommendations("u1", k=1)
    assert "01. b | score=" in capsys.readouterr().out
    assert calls == ([] if embedded else ["legacy"])


@pytest.mark.parametrize("corrupt", ["partial", "range", "dtype", "duplicates"])
def test_checkpoint_corruption_rejected(prepared, corrupt):
    checkpoint = {"model_type": "bprmf", "config": {}, "state_dict": {}, "num_users": 3, "num_items": 3,
                  "seen_items_indptr": np.array([0, 2, 2, 2], dtype=np.int64)}
    if corrupt != "partial":
        checkpoint["seen_items_indices"] = np.array([0, 3] if corrupt == "range" else [0, 0],
                                                    dtype=float if corrupt == "dtype" else np.int64)
    with pytest.raises(SeenItemsError):
        core._validate_model_artifacts({"idx2user": prepared.mappings.idx2user,
                                        "idx2item": prepared.mappings.idx2item}, checkpoint)


def test_csv_seen_parity_with_nomenclature_view_rule(tmp_path, events):
    for name, kind in (("Заказы", "PURCHASE"), ("Избранное", "FAVORITE"), ("Просмотры", "VIEW")):
        rows = [{"MindboxID": e.customer_id, "КодНоменклатуры": e.item_id, "Количество": 1,
                 "Дата": e.timestamp.strftime("%Y-%m-%d"), "ТипТовара": "Номенклатура"}
                for e in events if e.interaction_type.value == kind]
        if kind == "VIEW":
            rows.append({"MindboxID": "u3", "КодНоменклатуры": "c", "Количество": 1,
                         "Дата": "2026-02-01", "ТипТовара": "Группа"})
        pd.DataFrame(rows).to_csv(tmp_path / f"{name}.csv", sep="|", encoding="utf-8-sig", index=False)
    cfg = core.TrainConfig(data_dir=str(tmp_path), min_user_interactions_for_eval=3)
    prepared = core.prepare_training_data_from_csv(cfg)
    seen = build_seen_items_index(prepared)
    maps = prepared.mappings
    old = core._build_user_seen_sets(str(tmp_path), maps.user2idx, maps.item2idx)
    for u, u_idx in maps.user2idx.items():
        assert set(seen.items_for_user(u_idx)) == old[u_idx]
        assert old[u_idx] == set(core._user_seen_items_from_processed(str(tmp_path), u, maps.item2idx, cfg))
