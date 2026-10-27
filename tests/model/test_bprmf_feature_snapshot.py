"""Small real trainer/serialization tests; never access the published model."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from Application.model import BPRMF as core
from Application.model import raw_bpr_inference as raw
from Application.recommendation_catalog import item_from_metadata
from Application.tabs.training_workflow import production_config


@pytest.fixture
def training(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    codes = ["001001", "001002", "001003"]
    maps = core.Mappings({"synthetic-a": 0, "synthetic-b": 1}, ["synthetic-a", "synthetic-b"],
                         dict(zip(codes, range(3))), codes)
    splits = core.Splits(np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int64),
                         np.array([1., 3., 6.]), np.array([], dtype=np.int64),
                         np.array([], dtype=np.int64), [{0}, {0, 1}])
    data = core.PreparedBprData(maps, splits)
    cfg = core.TrainConfig(data_dir=str(tmp_path), embedding_dim=4, epochs=1, batch_size=3,
                           n_neg=1, early_stop=False, max_item_features=4,
                           item_feature_cols=["Марка", "КатегорияНаСайте", "СезонНоски", "ВидНоменклатуры"])
    frame = pd.DataFrame([
        {"КодНоменклатуры": codes[0], "Марка": "BRAND", "КатегорияНаСайте": "CAT-OLD", "СезонНоски": "Зима"},
        {"КодНоменклатуры": codes[0], "Марка": "BRAND", "КатегорияНаСайте": "CAT-LAST", "СезонНоски": "Зима"},
        {"КодНоменклатуры": codes[1]}, {"КодНоменклатуры": codes[2]},
    ])
    path = tmp_path / "nomenclature.csv"
    frame.to_csv(path, sep="|", index=False, encoding="utf-8-sig")
    return cfg, data, path


def train(training):
    cfg, data, _ = training
    core._set_seed(42)
    model, _ = core.train_prepared_data(cfg, data, torch.device("cpu"))
    model.eval()
    return model


@pytest.mark.parametrize("norm", ["mean", "sum", "sqrt"])
@pytest.mark.parametrize("change", ["unchanged", "values", "remove"])
def test_frozen_round_trip_preserves_vectors_and_raw_scores(training, tmp_path, monkeypatch, norm, change):
    cfg, data, path = training
    cfg.feature_norm = norm
    reads = []
    read = core._read_csv_pipe
    monkeypatch.setattr(core, "_read_csv_pipe", lambda path: reads.append(path) or read(path))
    before_pairs = data.splits.train_pairs.copy()
    before_weights = data.splits.train_weights.copy()
    model = train(training)
    assert len(reads) == 1
    snapshot = model._item_feature_snapshot
    before_vectors = model.item_vec_all().clone()
    if change == "values":
        path.write_text("КодНоменклатуры|Марка\n001001|OTHER\n", encoding="utf-8")
    elif change == "remove":
        path.unlink()
    monkeypatch.setattr(core, "_read_csv_pipe", lambda _: pytest.fail("save must not read the catalog"))
    directory = tmp_path / "model"
    core._save_artifacts(cfg, data.mappings, model, model_dir=str(directory))
    maps, checkpoint = core._load_artifacts(str(directory))
    raw._validate_checkpoint(maps, checkpoint)
    assert checkpoint["feat2idx"] == snapshot.vocabulary()
    np.testing.assert_array_equal(checkpoint["item_feat_mat"], snapshot.matrix)
    torch.testing.assert_close(checkpoint["state_dict"]["feat_emb.weight"], model.feat_emb.weight, rtol=0, atol=0)
    loaded, _, _, _ = core._build_model_from_ckpt(checkpoint, torch.device("cpu"))
    torch.testing.assert_close(loaded.item_vec_all(), before_vectors, rtol=0, atol=0)
    service = raw.RawBprInferenceService()
    result = service.get_recommendations("synthetic-a", k=3, model_dir=str(directory))
    assert result.status is raw.RawBprStatus.OK
    for rec in result.recommendations:
        index = data.mappings.item2idx[rec.model_item_code]
        expected = model.score(torch.tensor([0]), torch.tensor([index])).item()
        assert rec.raw_score == pytest.approx(expected, abs=1e-8)
    assert checkpoint["train_item_meta"]["001001"]["СезонНоски"] == "Зима"
    assert item_from_metadata("001001", checkpoint["train_item_meta"]["001001"]).get("СезонНоски") == "Зима"
    assert item_from_metadata("001002", checkpoint["train_item_meta"]["001002"]).get("СезонНоски") == ""
    np.testing.assert_array_equal(data.splits.train_pairs, before_pairs)
    np.testing.assert_array_equal(data.splits.train_weights, before_weights)
    # Old artifacts without this optional metadata remain valid.
    for values in checkpoint["train_item_meta"].values():
        values.pop("СезонНоски", None)
    raw._validate_checkpoint(maps, checkpoint)


def test_diagnostics_exact_coverage_conflicts_and_privacy(training, capsys):
    model = train(training)
    snapshot = model._item_feature_snapshot
    report = snapshot.diagnostics()
    assert report["available_fields"] == ["Марка", "СезонНоски", "КатегорияНаСайте"]
    assert report["missing_fields"] == ["ВидНоменклатуры"]
    assert report["unique_tokens"] == 3 and report["items_without_features"] == 2
    assert report["duplicate_codes"] == report["category_conflict_codes"] == 1
    assert report["conflicting_codes"] == report["conflicting_value_excess"] == 1
    assert report["training_coverage_available"]
    assert report["conflicting_fields"]["КатегорияНаСайте"] == 1
    assert report["model_conflicting_fields"]["КатегорияНаСайте"] == 1
    brand = report["per_field"]["Марка"]
    assert brand["item_fraction"] == pytest.approx(1 / 3)
    assert brand["train_pairs"] == 2
    assert brand["train_pair_fraction"] == pytest.approx(2 / 3)
    assert brand["confidence_fraction"] == pytest.approx(.4)
    assert "КатегорияНаСайте=CAT-LAST" in snapshot.vocabulary()
    assert "КатегорияНаСайте=CAT-OLD" not in snapshot.vocabulary()
    text = capsys.readouterr().out
    assert all(secret not in text for secret in ("synthetic-a", "001001", "BRAND", "CAT-LAST", "Зима"))
    with pytest.raises(ValueError):
        snapshot.matrix.setflags(write=True)


@pytest.mark.parametrize("mutation", ["matrix", "weights", "config", "order", "missing_snapshot"])
def test_inconsistency_never_replaces_current(training, tmp_path, mutation):
    cfg, data, _ = training
    model = train(training)
    directory = tmp_path / "model"
    core._save_artifacts(cfg, data.mappings, model, model_dir=str(directory))
    current = (directory / "current.json").read_bytes()
    generations = sorted(p.name for p in (directory / "runs").iterdir())
    if mutation == "matrix":
        model.item_feat_mat[0, 0] = -1
    elif mutation == "weights":
        with torch.no_grad():
            model.feat_emb.weight[0, 0] = float("nan")
    elif mutation == "config":
        cfg.feature_scale += .1
    elif mutation == "order":
        data.mappings.idx2item.reverse()
    else:
        del model._item_feature_snapshot
    with pytest.raises(ValueError):
        core._save_artifacts(cfg, data.mappings, model, model_dir=str(directory))
    assert (directory / "current.json").read_bytes() == current
    assert sorted(p.name for p in (directory / "runs").iterdir()) == generations
    assert list((directory / ".staging").iterdir()) == []


def test_serialized_vocabulary_swap_rejected_before_publication(training, tmp_path, monkeypatch):
    cfg, data, _ = training
    model = train(training)
    save = core.torch.save

    def corrupt(checkpoint, stream):
        checkpoint = deepcopy(checkpoint)
        tokens = list(checkpoint["feat2idx"])
        a, b = tokens[:2]
        checkpoint["feat2idx"][a], checkpoint["feat2idx"][b] = checkpoint["feat2idx"][b], checkpoint["feat2idx"][a]
        save(checkpoint, stream)

    monkeypatch.setattr(core.torch, "save", corrupt)
    directory = tmp_path / "model"
    with pytest.raises(ValueError, match="Serialized feature payload"):
        core._save_artifacts(cfg, data.mappings, model, model_dir=str(directory))
    assert not (directory / "current.json").exists()
    assert list((directory / ".staging").iterdir()) == []


@pytest.mark.parametrize("norm", ["mean", "sum", "sqrt"])
def test_norm_config_gui_cli_training_log_agree(training, tmp_path, norm, capsys):
    cfg, data, _ = training
    settings = tmp_path / "settings"
    settings.mkdir()
    config_path = settings / "train_config.json"
    config_path.write_text(json.dumps({"feature_norm": norm}), encoding="utf-8")
    assert core._load_train_config_from_json(str(config_path)).feature_norm == norm
    assert production_config({}, settings, tmp_path)["feature_norm"] == norm
    cfg.feature_norm = norm
    model = train(training)
    assert model.feature_norm == norm
    assert f"Способ объединения признаков номенклатуры: {norm.upper()}" in capsys.readouterr().out
    assert data.splits.train_pairs.shape == (3, 2)


@pytest.mark.parametrize("norm", ["typo", "", "avg", "sqrt_mean"])
def test_invalid_norm_rejected_before_training_and_gui_process(training, tmp_path, monkeypatch, norm):
    cfg, data, _ = training
    config_path = tmp_path / "train_config.json"
    config_path.write_text(json.dumps({"feature_norm": norm}), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_norm"):
        core._load_train_config_from_json(str(config_path))
    with pytest.raises(ValueError, match="feature_norm"):
        production_config({}, tmp_path, tmp_path)
    cfg.feature_norm = norm
    monkeypatch.setattr(core, "_prepare_item_feature_snapshot", lambda *a: pytest.fail("must validate before catalog read"))
    with pytest.raises(ValueError, match="feature_norm"):
        core.train_prepared_data(cfg, data, torch.device("cpu"))


@pytest.mark.parametrize("legacy,canonical", [("avg", "mean"), ("sqrt_mean", "sqrt"), ("sqrt-norm", "sqrt")])
def test_legacy_checkpoint_norm_aliases_keep_scores(training, tmp_path, legacy, canonical):
    cfg, data, _ = training
    cfg.feature_norm = canonical
    model = train(training)
    directory = tmp_path / "model"
    core._save_artifacts(cfg, data.mappings, model, model_dir=str(directory))
    mappings, checkpoint = core._load_artifacts(str(directory))
    checkpoint["config"]["feature_norm"] = legacy
    raw._validate_checkpoint(mappings, checkpoint)
    loaded, _, _, _ = core._build_model_from_ckpt(checkpoint, torch.device("cpu"))
    torch.testing.assert_close(loaded.item_vec_all(), model.item_vec_all(), rtol=0, atol=0)


def test_missing_feature_config_defaults_and_disabled_snapshot(training, tmp_path):
    cfg, data, path = training
    cfg.use_item_features = False
    model = train(training)
    assert not model.use_item_features
    path.unlink()
    directory = tmp_path / "model"
    core._save_artifacts(cfg, data.mappings, model, model_dir=str(directory))
    maps, checkpoint = core._load_artifacts(str(directory))
    checkpoint["config"].pop("feature_norm")
    raw._validate_checkpoint(maps, checkpoint)
    loaded, loaded_cfg, _, _ = core._build_model_from_ckpt(checkpoint, torch.device("cpu"))
    assert loaded_cfg.feature_norm == "mean"
    torch.testing.assert_close(loaded.item_vec_all(), model.item_vec_all())
    assert checkpoint["train_item_meta"]["001001"]["СезонНоски"] == "Зима"


def test_invalid_saved_norm_rejected_by_common_and_raw_validator(training, tmp_path):
    cfg, data, _ = training
    model = train(training)
    directory = tmp_path / "model"
    core._save_artifacts(cfg, data.mappings, model, model_dir=str(directory))
    mappings, checkpoint = core._load_artifacts(str(directory))
    checkpoint["config"]["feature_norm"] = "typo"
    for validate in (core._validate_model_artifacts, raw._validate_checkpoint):
        with pytest.raises(ValueError, match="feature_norm"):
            validate(mappings, checkpoint)


@pytest.mark.parametrize("missing_catalog", [True, False])
def test_empty_features_keep_base_vectors_and_safe_diagnostics(training, tmp_path, missing_catalog):
    cfg, data, path = training
    if missing_catalog:
        path.unlink()
    else:
        pd.DataFrame({"КодНоменклатуры": data.mappings.idx2item}).to_csv(path, sep="|", index=False)
    model = train(training)
    report = model._item_feature_snapshot.diagnostics()
    assert report["items_without_features"] == 3 and report["unique_tokens"] == 0
    assert all(values["train_pair_fraction"] == 0 for values in report["per_field"].values())
    torch.testing.assert_close(model.item_vec_all(), model.item_emb.weight)
    directory = tmp_path / "model"
    core._save_artifacts(cfg, data.mappings, model, model_dir=str(directory))
    mappings, checkpoint = core._load_artifacts(str(directory))
    raw._validate_checkpoint(mappings, checkpoint)
