"""Synthetic ranking/loader contracts; no production model or canonical reads."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch

from Application.model import BPRMF as core
from Application.model import raw_bpr_inference as raw


def _publish(root, generation):
    (root / "current.json").write_text(json.dumps({"generation": generation}), encoding="utf-8")


def _write(root, generation, mappings, checkpoint):
    directory = root if generation is None else root / "runs" / generation
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mappings.json").write_text(json.dumps(mappings), encoding="utf-8")
    torch.save(checkpoint, directory / "bprmf.pt")
    return directory


@pytest.fixture
def artifact(tmp_path):
    root = tmp_path / "model"
    users = ["user-exact", "00123"]
    items = ["GIFT-AW-ZERO-STOCK", *[f"ORIGINAL-{i:02d}" for i in range(11, 0, -1)]]
    matrix = np.full((12, 2), -1, dtype=np.int64)
    matrix[5] = [0, 1]
    model = core.BPRMF(2, 12, 2, num_item_feats=2, item_feat_mat=torch.tensor(matrix),
                       feature_scale=0.2, feature_norm="mean", feature_dropout=0.9)
    with torch.no_grad():
        model.user_emb.weight.copy_(torch.eye(2))
        model.item_emb.weight.copy_(torch.stack((torch.arange(11, -1, -1), torch.zeros(12)), dim=1))
        model.item_emb.weight[11, 0] = 0.123456789
        model.feat_emb.weight.copy_(torch.tensor([[10., 0.], [30., 0.]]))
    model.eval()
    checkpoint = {
        "model_type": "bprmf", "num_users": 2, "num_items": 12,
        "config": {"embedding_dim": 2, "use_item_features": True, "feature_scale": 0.2,
                   "feature_norm": "mean", "feature_dropout": 0.9},
        "state_dict": model.state_dict(), "feat2idx": {"token-a": 0, "token-b": 1},
        "item_feat_mat": matrix,
        "seen_items_indptr": np.array([0, 1, 1], dtype=np.int64),
        "seen_items_indices": np.array([0], dtype=np.int64),
        "train_item_meta": {items[0]: {"ВидНоменклатуры": "Подарочные карты",
                                      "Коллекция": "Осень-Зима 2025", "Остаток": "0"}},
    }
    mappings = {"idx2user": users, "idx2item": items}
    directory = _write(root, "generation-a", mappings, checkpoint)
    _publish(root, "generation-a")
    return root, directory, mappings, checkpoint, model


def _request(artifact, *, user="user-exact", k=10, service=None):
    service = service or raw.RawBprInferenceService()
    return service.get_recommendations(user, k=k, model_dir=artifact[0])


def test_saved_features_original_scores_ranks_and_precision(artifact):
    result = _request(artifact, k=100)
    _, _, mappings, _, model = artifact
    with torch.inference_mode():
        expected = model.score(torch.zeros(12, dtype=torch.long), torch.arange(12)).numpy()
        original_dot = (model.user_emb.weight[0] @ model.item_vec_all().T).numpy()
    order = np.argsort(-expected, kind="stable")
    assert result.status is raw.RawBprStatus.OK
    assert result.model_generation == "generation-a"
    assert result.model_user_id == "user-exact"
    assert len(result.recommendations) == 12
    assert [r.rank for r in result.recommendations] == list(range(1, 13))
    assert [r.model_item_code for r in result.recommendations] == [mappings["idx2item"][i] for i in order]
    assert [r.raw_score for r in result.recommendations] == original_dot[order].tolist()
    np.testing.assert_allclose([r.raw_score for r in result.recommendations], expected[order], rtol=1e-6)
    scores = [r.raw_score for r in result.recommendations]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 1  # Not sigmoid/softmax/percent output.
    assert scores[-1] != round(scores[-1], 2)
    # Mean (not sum): item 5 scores 6 + .2 * mean(10, 30) = 10.
    assert next(r.raw_score for r in result.recommendations if r.model_item_code == mappings["idx2item"][5]) == 10


def test_candidate_sources_same_scores_snapshot_and_detached_historical_metadata(artifact):
    service = raw.RawBprInferenceService()
    paths = (artifact[1] / "bprmf.pt", artifact[1] / "mappings.json")
    before = [p.read_bytes() for p in paths]
    original = service.get_recommendations("user-exact", k=100, model_dir=artifact[0])
    candidates, metadata = service.get_candidate_sources("user-exact", model_dir=artifact[0])
    assert candidates == original
    assert metadata == artifact[3]["train_item_meta"]
    metadata.clear()
    assert service.get_source_metadata(model_dir=artifact[0])[1] == artifact[3]["train_item_meta"]
    assert [p.read_bytes() for p in paths] == before


def test_prefixes_exact_ties_use_mapping_order_and_repeat(artifact):
    service = raw.RawBprInferenceService()
    results = [_request(artifact, k=k, service=service) for k in (1, 3, 5, 10)]
    for k, result in zip((1, 3, 5, 10), results):
        assert len(result.recommendations) == k
        assert result.recommendations == results[-1].recommendations[:k]
    assert _request(artifact, service=service) == results[-1]
    # Item 1 and item 5 tie at 10; the lower mapping index wins.
    assert results[-1].recommendations[1].model_item_code == artifact[2]["idx2item"][1]
    assert results[-1].recommendations[2].model_item_code == artifact[2]["idx2item"][5]
    # All scores for the second user tie, despite codes being out of lexical order.
    tied = _request(artifact, user="00123", k=12, service=service)
    assert [r.model_item_code for r in tied.recommendations] == artifact[2]["idx2item"]


def test_seen_categories_collections_stocks_and_external_sources_never_filter(artifact, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # A present but malformed catalog must also be irrelevant.
    (tmp_path / "input_data").mkdir()
    (tmp_path / "input_data/nomenclature.csv").write_text("invalid catalog", encoding="utf-8")
    def forbidden(*args, **kwargs):
        pytest.fail("Raw inference must not read legacy interactions, settings, catalog or exporter")
    for name in ("_require_interaction_sources", "_load_item_names", "_load_item_stocks",
                 "_load_selected_collections_from_settings", "_build_user_seen_sets",
                 "_load_historical_item_conversion", "export_recommendations_excel"):
        monkeypatch.setattr(core, name, forbidden)
    result = _request(artifact)
    assert result.recommendations[0].model_item_code == "GIFT-AW-ZERO-STOCK"
    # Original recommendation is kept even though it is seen, gift, AW and out of stock.
    assert not (tmp_path / "input_data/orders.csv").exists()


def test_missing_catalog_and_disabled_item_features(artifact):
    root, directory, mappings, checkpoint, model = artifact
    checkpoint["feat2idx"] = {}
    checkpoint["item_feat_mat"] = np.full((12, 2), -1, dtype=np.int64)
    checkpoint["config"]["use_item_features"] = False
    checkpoint["state_dict"] = {k: v for k, v in model.state_dict().items()
                                if k in ("user_emb.weight", "item_emb.weight")}
    torch.save(checkpoint, directory / "bprmf.pt")
    result = _request(artifact)
    assert [r.model_item_code for r in result.recommendations] == mappings["idx2item"][:10]
    assert not (root.parent / "input_data").exists()


@pytest.mark.parametrize("user", ["unknown", "123", "user-exact.0", "USER-EXACT"])
def test_unknown_user_is_normal_without_id_guessing(artifact, user):
    result = _request(artifact, user=user)
    assert result.status is raw.RawBprStatus.USER_NOT_IN_MODEL
    assert result.model_user_id == user
    assert result.model_generation == "generation-a"
    assert result.recommendations == ()


@pytest.mark.parametrize("user", [None, 123, "", " user-exact", "user-exact "])
def test_invalid_user_is_rejected_without_normalization(artifact, user):
    with pytest.raises((TypeError, ValueError)):
        _request(artifact, user=user)


@pytest.mark.parametrize("k", [0, -1, True, False, 1.5, "10", None])
def test_invalid_k_is_rejected(artifact, k):
    with pytest.raises(ValueError, match="k must"):
        _request(artifact, k=k)


def test_public_api_result_is_deeply_immutable_and_typed(artifact, monkeypatch):
    monkeypatch.setattr(raw, "_DEFAULT_SERVICE", raw.RawBprInferenceService())
    result = raw.get_raw_bpr_recommendations("user-exact", model_dir=str(artifact[0]))
    with pytest.raises(FrozenInstanceError):
        result.status = raw.RawBprStatus.USER_NOT_IN_MODEL
    with pytest.raises(FrozenInstanceError):
        result.recommendations[0].raw_score = 0.
    with pytest.raises(TypeError):
        result.recommendations[0] = result.recommendations[1]
    with pytest.raises(TypeError):
        raw.RawBprResult("g", "user", "OK", ())
    with pytest.raises(TypeError):
        raw.RawBprResult("g", "user", raw.RawBprStatus.OK, [])
    with pytest.raises(ValueError):
        raw.RawBprRecommendation(1, "item", float("nan"))


def test_repeat_uses_one_cpu_model_and_restores_allowlist(artifact, monkeypatch):
    service = raw.RawBprInferenceService()
    real_load = torch.load
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA must not be required"))
    # Preserve safe globals that the caller had already registered.
    with torch.serialization.safe_globals([np.ndarray]):
        before = list(torch.serialization.get_safe_globals())
        _request(artifact, service=service)
        _request(artifact, user="00123", service=service)
        assert torch.serialization.get_safe_globals() == before
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert service._loaded.model.training is False
    assert service._loaded.model.feature_dropout == 0
    assert all(not p.requires_grad and p.device.type == "cpu" for p in service._loaded.model.parameters())
    assert service._loaded.item_vectors.device.type == "cpu"


def test_generation_switch_reloads_and_releases_previous_model(artifact, monkeypatch):
    root, _, mappings, checkpoint, _ = artifact
    service = raw.RawBprInferenceService()
    _request(artifact, service=service)
    old_model = weakref.ref(service._loaded.model)
    checkpoint["state_dict"]["item_emb.weight"][11, 0] = 100
    _write(root, "generation-b", mappings, checkpoint)
    _publish(root, "generation-b")
    real_load = raw._load_checkpoint
    def load(path):
        assert old_model() is None  # Do not retain two complete generations.
        return real_load(path)
    monkeypatch.setattr(raw, "_load_checkpoint", load)
    result = _request(artifact, service=service)
    assert result.model_generation == "generation-b"
    assert result.recommendations[0].model_item_code == mappings["idx2item"][11]
    assert result.recommendations[0].raw_score == 100


def test_other_directory_does_not_reuse_cached_generation(artifact, tmp_path):
    _, _, mappings, checkpoint, _ = artifact
    service = raw.RawBprInferenceService()
    _request(artifact, service=service)
    other = tmp_path / "other-model"
    checkpoint["state_dict"]["item_emb.weight"][11, 0] = 50
    _write(other, "generation-a", mappings, checkpoint)
    _publish(other, "generation-a")
    result = service.get_recommendations("user-exact", model_dir=other)
    assert result.recommendations[0].raw_score == 50


def test_artifact_replacement_in_same_generation_invalidates_cache(artifact):
    _, directory, _, checkpoint, _ = artifact
    service = raw.RawBprInferenceService()
    _request(artifact, service=service)
    checkpoint["state_dict"]["item_emb.weight"][11, 0] = 99
    replacement = directory / "replacement.pt"
    torch.save(checkpoint, replacement)
    replacement.replace(directory / "bprmf.pt")
    assert _request(artifact, service=service).recommendations[0].raw_score == 99


@pytest.mark.parametrize("payload", ["broken", "[]", '{}', '{"generation":"../escape"}',
                                     '{"generation":42}', '{"generation":"missing"}'])
def test_invalid_active_generation_never_uses_cached_model(artifact, payload):
    service = raw.RawBprInferenceService()
    _request(artifact, service=service)
    (artifact[0] / "current.json").write_text(payload, encoding="utf-8")
    with pytest.raises(raw.RawBprArtifactError):
        _request(artifact, service=service)
    assert service._loaded is None


@pytest.mark.parametrize("mutation", ["json", "not_object", "missing", "length", "duplicate",
                                      "wrong_type", "empty_id", "duplicate_field"])
def test_invalid_mappings_rejected(artifact, mutation):
    _, directory, mappings, _, _ = artifact
    if mutation == "json":
        text = "broken"
    elif mutation == "not_object":
        text = "[]"
    elif mutation == "duplicate_field":
        text = '{"idx2user":[],"idx2user":[],"idx2item":[]}'
    else:
        if mutation == "missing":
            del mappings["idx2user"]
        elif mutation == "length":
            mappings["idx2user"].pop()
        elif mutation == "duplicate":
            mappings["idx2item"][1] = mappings["idx2item"][0]
        elif mutation == "wrong_type":
            mappings["idx2user"][0] = 1
        elif mutation == "empty_id":
            mappings["idx2user"][0] = ""
        text = json.dumps(mappings)
    (directory / "mappings.json").write_text(text, encoding="utf-8")
    with pytest.raises(raw.RawBprArtifactError):
        _request(artifact)


@pytest.mark.parametrize("mutation", ["bytes", "missing", "dimension", "model_type", "state_shape",
                                      "nan", "config", "feature_matrix", "feature_mapping", "feature_weights"])
def test_invalid_checkpoint_rejected(artifact, mutation):
    _, directory, _, checkpoint, _ = artifact
    if mutation == "bytes":
        (directory / "bprmf.pt").write_bytes(b"broken checkpoint")
    else:
        if mutation == "missing":
            del checkpoint["state_dict"]
        elif mutation == "dimension":
            checkpoint["num_users"] = True
        elif mutation == "model_type":
            checkpoint["model_type"] = "unsupported"
        elif mutation == "state_shape":
            checkpoint["state_dict"]["item_emb.weight"] = torch.ones(1, 2)
        elif mutation == "nan":
            checkpoint["state_dict"]["user_emb.weight"][0, 0] = float("nan")
        elif mutation == "config":
            checkpoint["config"]["embedding_dim"] = "2"
        elif mutation == "feature_matrix":
            checkpoint["item_feat_mat"][0, 0] = 500
        elif mutation == "feature_mapping":
            checkpoint["feat2idx"]["token-b"] = 0
        elif mutation == "feature_weights":
            checkpoint["state_dict"]["feat_emb.weight"][0, 0] = float("inf")
        torch.save(checkpoint, directory / "bprmf.pt")
    with pytest.raises(raw.RawBprArtifactError):
        _request(artifact)


class _UnapprovedCheckpointType:
    pass


def test_unapproved_pickle_type_is_rejected_without_unsafe_retry(artifact, monkeypatch):
    _, directory, _, checkpoint, _ = artifact
    checkpoint["unapproved"] = _UnapprovedCheckpointType()
    torch.save(checkpoint, directory / "bprmf.pt")
    before = list(torch.serialization.get_safe_globals())
    real_load = torch.load
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(raw.RawBprArtifactError):
        _request(artifact)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert torch.serialization.get_safe_globals() == before


def test_missing_artifacts_and_legacy_pair(artifact, tmp_path):
    service = raw.RawBprInferenceService()
    with pytest.raises(raw.RawBprArtifactError):
        service.get_recommendations("user", model_dir=tmp_path / "missing")
    root = tmp_path / "legacy"
    _write(root, None, artifact[2], artifact[3])
    result = service.get_recommendations("user-exact", model_dir=root)
    assert result.model_generation is None
    assert result.status is raw.RawBprStatus.OK


def test_concurrent_requests_load_once_and_return_identical_results(artifact, monkeypatch):
    service = raw.RawBprInferenceService()
    real_load = raw._load_checkpoint
    calls = []
    def load(path):
        calls.append(path)
        return real_load(path)
    monkeypatch.setattr(raw, "_load_checkpoint", load)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: _request(artifact, service=service), range(8)))
    assert len(calls) == 1
    assert all(result == results[0] for result in results)


def test_artifact_change_during_load_is_rejected(artifact, monkeypatch):
    real_load = raw._load_checkpoint
    def load(path):
        checkpoint = real_load(path)
        with Path(path).open("ab") as stream:
            stream.write(b"changed")
        return checkpoint
    monkeypatch.setattr(raw, "_load_checkpoint", load)
    service = raw.RawBprInferenceService()
    with pytest.raises(raw.RawBprArtifactError):
        _request(artifact, service=service)
    assert service._loaded is None
