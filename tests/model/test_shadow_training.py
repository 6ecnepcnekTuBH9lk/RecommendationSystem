from dataclasses import FrozenInstanceError
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from Application.model import BPRMF as core
from Application.model import mindbox_shadow_training as orchestration
from Application.model.shadow_training import run_shadow_training
from Application.model.training_data import Mappings, Splits, PreparedBprData
from Application.model.training_quality import TrainingQualityDiagnostics
from Application.mindbox import daily_training_batch as daily
from Application.mindbox.training_batch import TrainingBatchWindow, TrainingBatchExport
from scripts.mindbox_shadow_train import main


def forbidden(*args, **kwargs):
    pytest.fail("Forbidden network/publication/training call")


@pytest.fixture(autouse=True)
def isolation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core, "_save_artifacts", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    # PyTorch interop initialization is process-global and can only run once.
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(orchestration, "REPORT_ROOT", tmp_path / "reports")
    production = tmp_path / "Модель"
    for name in ("current.json", "runs/old/model.pt", ".staging/old/sentinel"):
        path = production / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"production sentinel\x00\xff")
    def snapshot():
        return {str(p.relative_to(production)): p.read_bytes() if p.is_file() else None
                for p in production.rglob("*")}
    before = snapshot()
    yield
    assert snapshot() == before


@pytest.fixture
def prepared():
    users = ["SECRET_CUSTOMER_A", "SECRET_CUSTOMER_B"]
    items = ["SECRET_ITEM_A", "SECRET_ITEM_B", "SECRET_ITEM_C"]
    return PreparedBprData(Mappings(dict(zip(users, range(2))), users, dict(zip(items, range(3))), items),
        Splits(np.array([[0, 0], [1, 1]]), np.array([.1, 10.]), np.array([0]), np.array([2]), [{0}, {1}]))


def config(tmp_path, **kwargs):
    return core.TrainConfig(data_dir=str(tmp_path), epochs=2, embedding_dim=4, batch_size=2, n_neg=1, topk=2, **kwargs)


@pytest.mark.parametrize("features", [False, True])
def test_metrics_legacy_parity_and_determinism(prepared, tmp_path, features):
    cfg = config(tmp_path, use_item_features=features)
    pd.DataFrame({"КодНоменклатуры": prepared.mappings.idx2item, "Марка": ["a", "b", "a"]}).to_csv(
        tmp_path / "Номенклатура.csv", sep="|", encoding="utf-8-sig", index=False)
    core._set_seed(cfg.seed)
    old_model, old_splits = core.train_prepared_data(cfg, prepared, torch.device("cpu"))
    core._set_seed(cfg.seed)
    new_model, new_splits, metrics = core.train_prepared_data_with_metrics(cfg, prepared, torch.device("cpu"))
    assert old_splits is new_splits is prepared.splits
    assert new_model.use_item_features == features
    for name, value in old_model.state_dict().items():
        torch.testing.assert_close(value, new_model.state_dict()[name], rtol=0, atol=0)
    result = run_shadow_training(cfg, prepared, TrainingQualityDiagnostics(), torch.device("cpu"))
    assert result.training_metrics == metrics
    assert metrics.epochs_completed == 2
    assert all(np.isfinite([m.loss, m.recall, m.ndcg]).all() for m in metrics.history)
    assert "SECRET" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        metrics.history[0].epoch = 99
    with pytest.raises(FrozenInstanceError):
        result.training_completed = False
    with pytest.raises(TypeError):
        result.preparation_summary["users"] = 0


@pytest.mark.parametrize("metric,best_epoch", [("ndcg", 1), ("recall", 2)])
def test_existing_early_stop_observations(prepared, tmp_path, monkeypatch, metric, best_epoch):
    cfg = config(tmp_path, use_item_features=False, early_stop_metric=metric,
                 early_stop_patience=1, early_stop_min_epochs=1)
    cfg.epochs = 5
    values = iter([(.1, .8), (.9, .7), (.8, .6)])
    monkeypatch.setattr(core, "_eval_bprmf_recall_ndcg", lambda *a: next(values))
    _, _, metrics = core.train_prepared_data_with_metrics(cfg, prepared, torch.device("cpu"))
    assert metrics.best_epoch == best_epoch
    assert metrics.early_stopped
    assert metrics.epochs_completed == best_epoch + 1
    best = metrics.history[best_epoch - 1]
    assert (metrics.best_recall, metrics.best_ndcg) == (best.recall, best.ndcg)


@pytest.mark.parametrize("problem", ["unresolved", "unsupported", "invalid"])
def test_block_never_trains(prepared, tmp_path, monkeypatch, problem):
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", forbidden)
    diagnostics = TrainingQualityDiagnostics(**({f"{problem}_products": 1} if problem != "invalid" else {}))
    if problem == "invalid":
        prepared.splits.train_weights[0] = np.nan
    result = run_shadow_training(config(tmp_path), prepared, diagnostics, torch.device("cpu"))
    assert not result.training_started and not result.training_completed
    assert not result.quality_report.training_allowed


def synthetic_manifest(tmp_path, problem):
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    until = since + timedelta(days=1)
    root = tmp_path / "raw"
    actions = [{"ids": {"mindboxId": f"SECRET_ACTION_{i}"},
                "customer": {"ids": {"mindboxId": user}},
                "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}},
                "dateTimeUtc": since.isoformat(), "creationDateTimeUtc": since.isoformat(),
                "products": [{"ids": {"offline1C": item + "_SECRET"}}]}
               for i, (user, item) in enumerate([("SECRET_A", "001234"), ("SECRET_B", "001235"),
                                                ("SECRET_A", "001236")])]
    if problem in ("warn", "block"):
        actions.append({**actions[0], "products": [] if problem == "warn" else [{"ids": {"offline1C": "999999_SECRET"}}]})
    components = []
    for name, records, key in (("customer_merges", [], "customerMerges"), ("actions", actions, "customerActions"),
                               ("orders", [], "orders")):
        directory = root / name / "20260101_000000"
        directory.mkdir(parents=True)
        (directory / f"{name}_part_001.json").write_text(json.dumps({key: records}), encoding="utf-8")
        entry = TrainingBatchExport(name, "SECRET_EXPORT", name, directory.relative_to(root).as_posix(), 1)
        components.append(daily.BatchComponent(name, since, until, name, "READY", entry))
    batch = daily.ChunkedTrainingBatch("a" * 32, since, TrainingBatchWindow(since, until, since),
                                       "0" * 64, tuple(components), True)
    manifest = root / "training_batches" / batch.batch_id / "manifest.json"
    manifest.parent.mkdir(parents=True)
    daily._atomic_write(manifest, batch)
    catalog = tmp_path / "Номенклатура.csv"
    catalog.write_text("КодНоменклатуры|Марка\n001234|a\n001235|b\n001236|a\n", encoding="utf-8-sig")
    return manifest, root, catalog


@pytest.mark.parametrize("problem,exit_code", [("pass", 0), ("warn", 0), ("block", 1), ("prepare", 1), ("train", 1)])
def test_full_synthetic_manifest_cli_and_safe_report(tmp_path, monkeypatch, capsys, problem, exit_code):
    manifest, root, catalog = synthetic_manifest(tmp_path, problem)
    if problem == "prepare":
        manifest.write_text("broken", encoding="utf-8")
    if problem in ("block", "prepare"):
        monkeypatch.setattr(core, "train_prepared_data_with_metrics", forbidden)
    if problem == "train":
        def fail(*args):
            raise RuntimeError("SECRET_CUSTOMER secret@example.test https://signed.invalid")
        monkeypatch.setattr(core, "train_prepared_data_with_metrics", fail)
    assert main(["--manifest", str(manifest), "--raw-root", str(root), "--catalog", str(catalog),
                 "--epochs", "2", "--device", "cpu"]) == exit_code
    output = capsys.readouterr()
    reports = list((tmp_path / "reports").glob("*/report.json"))
    assert len(reports) == (0 if problem == "prepare" else 1)
    text = output.out + output.err
    if reports:
        text += reports[0].read_text(encoding="utf-8")
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        assert report["production_model_published"] is False
        assert report["training"]["started"] == (problem != "block")
        if exit_code == 0:
            assert report["training"]["metrics"]["epochs_completed"] == 2
            assert "включены, применяются" in output.out
        if problem == "warn":
            assert report["quality"]["level"] == "WARN"
            assert report["dataset"]["complete"] is False
            assert "Quality: WARN" in output.out
    for secret in ("SECRET", "001234", "001235", "001236", "idx2user", "idx2item", "mappings",
                   "secret@example.test", "SecretKey", "Authorization", "https://"):
        assert secret not in text


@pytest.mark.parametrize("failure", [False, True])
def test_atomic_report(tmp_path, monkeypatch, failure):
    path = tmp_path / "report.json"
    path.write_text('{"old": true}', encoding="utf-8")
    calls = []
    fsync, replace = orchestration.os.fsync, orchestration.os.replace
    def sync(fd):
        calls.append("fsync")
        return fsync(fd)
    def publish(source, target):
        assert calls == ["fsync"]
        assert json.loads(Path(source).read_text(encoding="utf-8")) == {"new": True}
        assert json.loads(path.read_text()) == {"old": True}
        if failure:
            raise OSError("synthetic publication failure")
        return replace(source, target)
    monkeypatch.setattr(orchestration.os, "fsync", sync)
    monkeypatch.setattr(orchestration.os, "replace", publish)
    if failure:
        with pytest.raises(OSError):
            orchestration._atomic_report(path, {"new": True})
    else:
        orchestration._atomic_report(path, {"new": True})
    assert json.loads(path.read_text()) == ({"old": True} if failure else {"new": True})
    assert list(tmp_path.glob("tmp*")) == []


def test_same_config_used_and_quality_shown_before_training(tmp_path, monkeypatch):
    manifest, root, catalog = synthetic_manifest(tmp_path, "warn")
    cfg = config(tmp_path, w_view_item=.7)
    prepare = orchestration.prepare_training_data_from_chunked_batch
    train = core.train_prepared_data_with_metrics
    calls = []
    def preparing(*args, **kwargs):
        assert kwargs["train_config"] is cfg
        calls.append("prepare")
        return prepare(*args, **kwargs)
    def quality(report):
        assert report.training_allowed
        calls.append("quality")
    def training(passed_cfg, data, device):
        assert passed_cfg is cfg
        assert calls == ["prepare", "quality"]
        assert data.splits.train_weights.sum() == pytest.approx(2.1)
        return train(passed_cfg, data, device)
    monkeypatch.setattr(orchestration, "prepare_training_data_from_chunked_batch", preparing)
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", training)
    result = orchestration.shadow_train_daily_manifest(manifest, raw_root=root, catalog_path=catalog, cfg=cfg,
                                                       device=torch.device("cpu"), on_quality=quality)
    assert result.training_completed


def test_config_allowlist_excludes_arbitrary_secrets(tmp_path):
    cfg = config(tmp_path)
    cfg.SecretKey = "SECRET_KEY"
    cfg.item_feature_cols = ["Марка", "secret@example.test"]
    encoded = json.dumps(orchestration._safe_config(cfg), ensure_ascii=False)
    assert "SECRET" not in encoded and "secret@example.test" not in encoded
    assert "Марка" in encoded
