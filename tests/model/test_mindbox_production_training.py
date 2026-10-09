from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import json
import io
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import requests
import torch

from Application.mindbox import daily_training_batch as daily
from Application.mindbox import canonical_storage as store
from Application.mindbox import customer_profile_snapshot as profiles
from Application.mindbox.client import MindboxClient
from Application.mindbox.training_batch import TrainingBatchWindow, TrainingBatchExport
from Application.model import BPRMF as core
from Application.model import mindbox_production_training as production
from scripts import mindbox_production_train as cli


OLD = "1" * 32
EXTERNAL = "2" * 32


def forbidden(*args, **kwargs):
    pytest.fail("Forbidden dependency in production training")


def test_gui_cancellation_at_epoch_preserves_publication(tmp_path):
    args = source(tmp_path)
    args['cfg'].epochs = 2
    old = old_model(production.MODEL_ROOT)
    epochs = []
    def observer(model, epoch):
        epochs.append(epoch.epoch)
        raise KeyboardInterrupt
    result = production.train_and_publish_production_model(**args, epoch_observer=observer)
    assert epochs == [1] and result.cancelled and not result.publication_started
    assert_old(production.MODEL_ROOT, old)


def test_gui_cancellation_during_artifact_precommit_keeps_old_generation(tmp_path, monkeypatch):
    args = source(tmp_path)
    old = old_model(production.MODEL_ROOT)
    cancelled = []
    save = core._save_artifacts
    def artifacts(*args, **kwargs):
        cancelled.append(True)
        return save(*args, **kwargs)
    def check():
        if cancelled:
            raise KeyboardInterrupt
    monkeypatch.setattr(core, '_save_artifacts', artifacts)
    result = production.train_and_publish_production_model(**args, check_cancel=check)
    assert result.cancelled and result.publication_started and not result.published
    assert_old(production.MODEL_ROOT, old)


@pytest.mark.parametrize('answer,expected', [('NO\n', 130), ('', 130), ('PUBLISH\n', 0)])
def test_managed_gui_publication_handshake(tmp_path, monkeypatch, capsys, answer, expected):
    args = canonical_source(tmp_path)
    old = old_model(production.MODEL_ROOT)
    monkeypatch.setattr(sys, 'stdin', io.StringIO(answer))
    assert cli.main(gui_argv(args) + ['--cancel-file', str(tmp_path / 'cancel')]) == expected
    recorded = events(capsys.readouterr().out)
    assert any(e['stage'] == 'epoch' and e['epoch'] == 1 for e in recorded)
    assert any(e['stage'] == 'publication_ready' for e in recorded)
    result = next(e for e in recorded if e['stage'] == 'finished')
    if expected:
        assert result['cancelled'] and not result['published']
        assert_old(production.MODEL_ROOT, old)
    else:
        assert result['published'] and result['postpublish_validation']


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(production, "MODEL_ROOT", tmp_path / "model")
    monkeypatch.setattr(production, "REPORT_ROOT", tmp_path / "reports")
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(MindboxClient, "__init__", forbidden)
    monkeypatch.setattr(profiles, "load_customer_contact_index", forbidden)
    monkeypatch.setattr(profiles, "load_customer_profile_snapshot", forbidden)
    monkeypatch.setattr(core, "export_recommendations_excel", forbidden)
    monkeypatch.setattr(core, "prepare_training_data_from_csv", forbidden)


def source(tmp_path, problem="pass"):
    raw = tmp_path / "raw"
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    until = since + timedelta(days=1)
    actions = [{"ids": {"mindboxId": f"SECRET_ACTION_{i}"},
        "customer": {"ids": {"mindboxId": user}},
        "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}},
        "dateTimeUtc": since.isoformat(), "creationDateTimeUtc": since.isoformat(),
        "products": [{"ids": {"offline1C": item + "_SECRET_ITEM"}}]}
        for i, (user, item) in enumerate([("SECRET_A", "123456"), ("SECRET_B", "123457"), ("SECRET_A", "123458")])]
    if problem in ("warn", "block"):
        actions.append({**actions[0], "products": [] if problem == "warn" else [{"ids": {"offline1C": "999999_SECRET_ITEM"}}]})
    components = []
    for name, key, records in (("customer_merges", "customerMerges", []), ("actions", "customerActions", actions), ("orders", "orders", [])):
        directory = raw / name / "20260101_000000"
        directory.mkdir(parents=True)
        (directory / f"{name}_part_001.json").write_text(json.dumps({key: records}), encoding="utf-8")
        entry = TrainingBatchExport(name, "SECRET_EXPORT", name, directory.relative_to(raw).as_posix(), 1)
        components.append(daily.BatchComponent(name, since, until, name, "READY", entry))
    batch = daily.ChunkedTrainingBatch("a" * 32, since, TrainingBatchWindow(since, until, since), "0" * 64, tuple(components), True)
    manifest = raw / "training_batches" / batch.batch_id / "manifest.json"
    manifest.parent.mkdir(parents=True)
    daily._atomic_write(manifest, batch)
    catalog = tmp_path / "nomenclature.csv"
    catalog.write_text("КодНоменклатуры|Марка\n123456|a\n123457|b\n123458|a\n", encoding="utf-8-sig")
    cfg = core.TrainConfig(data_dir=str(tmp_path), epochs=1, embedding_dim=4, batch_size=2, n_neg=1, topk=2)
    return dict(training_manifest=manifest, raw_root=raw, catalog_path=catalog, cfg=cfg, device=torch.device("cpu"))


def old_model(root):
    run = root / "runs" / OLD
    run.mkdir(parents=True)
    (run / "mappings.json").write_bytes(b"old mappings sentinel")
    (run / "bprmf.pt").write_bytes(b"old checkpoint sentinel")
    current = root / "current.json"
    current.write_bytes(json.dumps({"generation": OLD}, indent=2).encode())
    return current.read_bytes()


def assert_old(root, content):
    assert (root / "current.json").read_bytes() == content
    assert (root / "runs" / OLD / "bprmf.pt").read_bytes() == b"old checkpoint sentinel"
    assert [p.name for p in (root / "runs").iterdir()] == [OLD]
    assert not list((root / ".staging").glob("*"))
    assert not list(root.glob(".current.*.tmp"))


@pytest.mark.parametrize("problem", ["pass", "warn", "block", "invalid"])
def test_preflight_readonly(tmp_path, monkeypatch, problem):
    args = source(tmp_path, problem)
    root = production.MODEL_ROOT
    old = old_model(root)
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", forbidden)
    monkeypatch.setattr(core, "_save_artifacts", forbidden)
    if problem == "invalid":
        args["training_manifest"].write_text("SecretKey", encoding="utf-8")
    result = production.preflight_production_training(**args)
    assert result.error_code == {"pass": None, "warn": None, "block": "QUALITY_BLOCK", "invalid": "PREPARATION_FAILED"}[problem]
    assert_old(root, old)
    assert not (tmp_path / ".mindbox-production.lock").exists()
    assert not production.REPORT_ROOT.exists()


@pytest.mark.parametrize('unknown,invalid,unsupported,attempts,level', [
    (20, 0, 0, 1206884, 'WARN'), (1, 0, 0, 10, 'BLOCK'),
    (0, 1, 0, 1206884, 'BLOCK'), (0, 0, 1, 1206884, 'BLOCK'),
])
def test_preflight_resolution_policy_and_warn_ack_before_any_trainer(tmp_path, monkeypatch, unknown, invalid, unsupported, attempts, level):
    from dataclasses import replace
    args = source(tmp_path)
    batch = daily.load_chunked_training_batch(args['training_manifest'], raw_root=args['raw_root'], require_complete=True)
    prepared = daily.prepare_training_data_from_chunked_batch(batch, raw_root=args['raw_root'],
        catalog_path=args['catalog_path'], train_config=args['cfg'], diagnose=True)
    d = prepared.diagnostics
    counts = replace(d.resolution.total, interactions_total=attempts, resolved=attempts-unknown-invalid-unsupported,
                     unresolved=unknown+invalid+unsupported, unknown_candidate=unknown,
                     invalid_id=invalid, unsupported_namespace=unsupported)
    prepared = replace(prepared, diagnostics=replace(d, resolution=replace(d.resolution, total=counts)))
    monkeypatch.setattr(production, 'prepare_training_data_from_chunked_batch', lambda *a, **k: prepared)
    monkeypatch.setattr(core, 'train_prepared_data_with_metrics', forbidden)
    monkeypatch.setattr(core, '_save_artifacts', forbidden)
    result = production.preflight_production_training(**args)
    assert result.quality_report.level.value == level
    assert result.error_code == (None if level == 'WARN' else 'QUALITY_BLOCK')
    assert not result.training_started and not result.publication_started
    if level == 'WARN':
        refused = production.train_and_publish_production_model(**args, allow_warn=False)
        assert refused.error_code == 'WARN_NOT_ACKNOWLEDGED'
        assert not refused.training_started and not refused.publication_started
    assert not (production.MODEL_ROOT / 'current.json').exists()


@pytest.mark.parametrize("problem,allow_warn", [("pass", False), ("warn", True)])
@pytest.mark.parametrize("existing", [False, True])
def test_real_train_publish_reload_and_safe_result(tmp_path, problem, allow_warn, existing, capsys):
    args = source(tmp_path, problem)
    if existing:
        old_model(production.MODEL_ROOT)
    result = production.train_and_publish_production_model(**args, allow_warn=allow_warn)
    assert result.error_code is None
    assert result.published and result.prepublish_disk_validation and result.postpublish_validation
    assert result.training_metrics.epochs_completed == 1
    assert result.previous_generation == (OLD if existing else None)
    assert production._current(production.MODEL_ROOT)[0] == result.published_generation
    assert len(list((production.MODEL_ROOT / "runs").iterdir())) == (2 if existing else 1)
    maps, checkpoint = core._load_artifacts(str(production.MODEL_ROOT))
    assert production.seen_items_from_checkpoint(checkpoint) is not None
    assert production.analytics_from_checkpoint(checkpoint) is not None
    model, _, users, items = core._build_model_from_ckpt(checkpoint, torch.device("cpu"))
    assert (users, items) == (len(maps["idx2user"]), len(maps["idx2item"])) == (2, 3)
    assert model is not None
    assert "customer_contacts" not in checkpoint
    assert not list(production.MODEL_ROOT.rglob("*.csv"))
    assert not list(production.MODEL_ROOT.rglob("*.xlsx"))
    assert all(not (tmp_path / name).exists() for name in ("orders.csv", "views.csv", "favorites.csv"))
    output = capsys.readouterr()
    text = Path(result.report_path).read_text(encoding="utf-8") + repr(result) + output.out + output.err
    assert all(value not in text for value in ("SECRET", "123456", "SecretKey", "@", str(tmp_path)))
    with pytest.raises(FrozenInstanceError):
        result.published = False
    with pytest.raises(TypeError):
        result.dataset["users"] = 99


@pytest.mark.parametrize("problem,code", [("warn", "WARN_NOT_ACKNOWLEDGED"), ("block", "QUALITY_BLOCK")])
def test_quality_stops_before_training(tmp_path, monkeypatch, problem, code):
    args = source(tmp_path, problem)
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", forbidden)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == code
    assert not result.training_started
    assert not production.MODEL_ROOT.exists()


@pytest.mark.parametrize("cancel", [False, True])
def test_training_failure_unchanged(tmp_path, monkeypatch, cancel):
    args = source(tmp_path)
    old = old_model(production.MODEL_ROOT)
    def fail(*args):
        raise KeyboardInterrupt() if cancel else RuntimeError("SecretKey secret@example.test")
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", fail)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == ("CANCELLED" if cancel else "TRAINING_FAILED")
    assert not result.published
    assert_old(production.MODEL_ROOT, old)


@pytest.mark.parametrize("when", ["training", "before_commit"])
def test_external_current_guard(tmp_path, monkeypatch, when):
    args = source(tmp_path)
    old_model(production.MODEL_ROOT)
    def change():
        (production.MODEL_ROOT / "current.json").write_text(json.dumps({"generation": EXTERNAL}), encoding="utf-8")
    if when == "training":
        train = core.train_prepared_data_with_metrics
        def changed(*a):
            value = train(*a)
            change()
            return value
        monkeypatch.setattr(core, "train_prepared_data_with_metrics", changed)
    else:
        rename = core.os.rename
        def changed(*a):
            rename(*a)
            change()
        monkeypatch.setattr(core.os, "rename", changed)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == "CURRENT_MODEL_CHANGED"
    assert not result.published
    assert production._current(production.MODEL_ROOT)[0] == EXTERNAL
    assert [p.name for p in (production.MODEL_ROOT / "runs").iterdir()] == [OLD]


@pytest.mark.parametrize("missing", ["seen", "analytics"])
def test_required_embedded_data(tmp_path, monkeypatch, missing):
    args = source(tmp_path)
    if missing == "seen":
        monkeypatch.setattr(production, "build_seen_items_index", lambda _: None)
    else:
        prepare = production._prepare
        def incomplete(*a):
            data, result = prepare(*a)
            data.analytics = None
            return data, result
        monkeypatch.setattr(production, "_prepare", incomplete)
    monkeypatch.setattr(core, "_save_artifacts", forbidden)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == "ARTIFACT_FAILED"
    assert not result.published


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_post_validation_rollback_conditional(tmp_path, monkeypatch, existing, external):
    args = source(tmp_path)
    old = old_model(production.MODEL_ROOT) if existing else None
    def fail(*a):
        if external:
            (production.MODEL_ROOT / "current.json").write_text(json.dumps({"generation": EXTERNAL}), encoding="utf-8")
        raise ValueError("SECRET invalid state")
    monkeypatch.setattr(core, "_build_model_from_ckpt", fail)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == "POST_PUBLISH_VALIDATION_FAILED"
    assert not result.published
    assert result.rolled_back == (not external)
    assert production._current(production.MODEL_ROOT)[0] == (EXTERNAL if external else OLD if existing else None)
    if old and not external:
        assert_old(production.MODEL_ROOT, old)
    assert not (production.MODEL_ROOT / "runs" / result.published_generation).exists()


@pytest.mark.parametrize("after_commit", [False, True])
def test_interrupt_at_exact_commit_reports_truth(tmp_path, monkeypatch, after_commit):
    args = source(tmp_path)
    old = old_model(production.MODEL_ROOT)
    replace = core.os.replace
    def interrupt(source, target):
        if Path(target) == production.MODEL_ROOT / "current.json":
            if after_commit:
                replace(source, target)
            raise KeyboardInterrupt()
        return replace(source, target)
    monkeypatch.setattr(core.os, "replace", interrupt)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == "CANCELLED" and result.cancelled
    assert result.published == after_commit
    if after_commit:
        assert production._current(production.MODEL_ROOT)[0] == result.published_generation
        core._load_artifacts(str(production.MODEL_ROOT))
    else:
        assert_old(production.MODEL_ROOT, old)


def test_report_failure_keeps_published_model(tmp_path, monkeypatch):
    args = source(tmp_path)
    def fail(*a):
        raise OSError("SecretKey")
    monkeypatch.setattr(production, "_atomic_report", fail)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == "REPORT_FAILED"
    assert result.published and result.postpublish_validation
    assert result.report_path is None
    assert production._current(production.MODEL_ROOT)[0] == result.published_generation


def test_os_lock_rejects_other_process_and_releases_after_death(tmp_path):
    args = source(tmp_path)
    script = ("from pathlib import Path; from Application.model.mindbox_production_training import _publication_lock; "
              "import sys; lock = _publication_lock(Path(sys.argv[1])); lock.__enter__(); print('locked', flush=True); sys.stdin.readline()")
    child = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], cwd=production.PROJECT_ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        assert child.stdout.readline().strip() == "locked"
        result = production.train_and_publish_production_model(**args)
        assert result.error_code == "LOCKED"
        assert not result.training_started
    finally:
        child.kill()
        child.communicate(timeout=15)
    with production._publication_lock(tmp_path):
        pass  # Lock file remains but OS ownership is released.


@pytest.mark.parametrize("bad", ["--epochs", "--model-dir"])
def test_cli_has_no_unsafe_overrides(bad):
    with pytest.raises(SystemExit) as caught:
        cli.main(["publish", "--manifest", "synthetic", bad, "2"])
    assert caught.value.code == 2


def test_cli_preflight_and_cancel(tmp_path, monkeypatch, capsys):
    args = source(tmp_path, "warn")
    argv = ["--manifest", str(args["training_manifest"]), "--raw-root", str(args["raw_root"]),
            "--catalog", str(args["catalog_path"])]
    assert cli.main(["preflight", *argv]) == 0
    assert not production.MODEL_ROOT.exists()
    assert cli.main(["publish", *argv]) == 1
    def cancel(*a):
        raise KeyboardInterrupt()
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", cancel)
    assert cli.main(["publish", *argv, "--allow-warn"]) == 130
    output = capsys.readouterr()
    assert "WARN_NOT_ACKNOWLEDGED" in output.err
    assert str(tmp_path) not in output.out + output.err


@pytest.mark.parametrize("missing", ["seen", "analytics", "dimensions"])
def test_post_reload_contract_failure_rolls_back(tmp_path, monkeypatch, missing):
    args = source(tmp_path)
    old = old_model(production.MODEL_ROOT)
    load = core._load_artifacts
    def damaged(*a):
        maps, checkpoint = load(*a)
        if missing == "seen":
            del checkpoint["seen_items_indptr"], checkpoint["seen_items_indices"]
        elif missing == "analytics":
            del checkpoint["interaction_analytics"]
        else:
            checkpoint["num_users"] += 1
        return maps, checkpoint
    monkeypatch.setattr(core, "_load_artifacts", damaged)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == "POST_PUBLISH_VALIDATION_FAILED"
    assert result.rolled_back and not result.published
    assert_old(production.MODEL_ROOT, old)


def test_rollback_failure_is_visible_and_keeps_publication_truth(tmp_path, monkeypatch):
    args = source(tmp_path)
    old_model(production.MODEL_ROOT)
    def fail(*a):
        raise OSError("SecretKey")
    monkeypatch.setattr(production, "_verify", fail)
    monkeypatch.setattr(production, "_rollback", fail)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == "ROLLBACK_FAILED"
    assert result.published and not result.rolled_back
    assert production._current(production.MODEL_ROOT)[0] == result.published_generation


def test_lock_covers_prepare_training_and_commit(tmp_path, monkeypatch):
    args = source(tmp_path)
    visited = []
    def wrap(owner, name):
        original = getattr(owner, name)
        def checked(*a, **kw):
            with pytest.raises(production._Failure, match="LOCKED"):
                with production._publication_lock(tmp_path):
                    pass
            visited.append(name)
            return original(*a, **kw)
        monkeypatch.setattr(owner, name, checked)
    wrap(production, "_prepare")
    wrap(core, "train_prepared_data_with_metrics")
    wrap(core, "_save_artifacts")
    result = production.train_and_publish_production_model(**args)
    assert result.published
    assert visited == ["_prepare", "train_prepared_data_with_metrics", "_save_artifacts"]


def test_cli_reports_published_when_audit_fails_and_uses_production_config(tmp_path, monkeypatch, capsys):
    args = source(tmp_path)
    def published(*a, **kw):
        assert kw["cfg"].epochs == core.TrainConfig().epochs == 200
        assert "model_dir" not in kw
        return production.ProductionTrainingResult(published=True, error_code="REPORT_FAILED")
    monkeypatch.setattr(production, "train_and_publish_production_model", published)
    assert cli.main(["publish", "--manifest", str(args["training_manifest"]), "--catalog", str(args["catalog_path"])]) == 1
    output = capsys.readouterr()
    assert "Production model published: YES" in output.out
    assert "Audit report: FAILED" in output.out


def test_invalid_current_manifest_does_not_train(tmp_path, monkeypatch):
    args = source(tmp_path)
    old_model(production.MODEL_ROOT)
    current = production.MODEL_ROOT / "current.json"
    current.write_text('{"generation":"../SecretKey"}', encoding="utf-8")
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", forbidden)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == "ARTIFACT_FAILED"
    assert not result.training_started
    assert "SecretKey" not in Path(result.report_path).read_text(encoding="utf-8")


def test_orphan_cleanup_failure_preserves_current_and_reports_flag(tmp_path, monkeypatch):
    args = source(tmp_path)
    old = old_model(production.MODEL_ROOT)
    replace = core.os.replace
    def fail_commit(source, target):
        if Path(target) == production.MODEL_ROOT / "current.json":
            raise OSError("SecretKey")
        return replace(source, target)
    def fail_cleanup(*a):
        raise OSError("SecretKey")
    monkeypatch.setattr(core.os, "replace", fail_commit)
    monkeypatch.setattr(core.shutil, "rmtree", fail_cleanup)
    result = production.train_and_publish_production_model(**args)
    assert result.error_code == "PUBLICATION_FAILED"
    assert result.cleanup_failed and not result.published
    assert (production.MODEL_ROOT / "current.json").read_bytes() == old
    assert len(list((production.MODEL_ROOT / "runs").iterdir())) == 2


def canonical_source(tmp_path, problem="pass"):
    args = source(tmp_path, problem)
    batch = daily.load_chunked_training_batch(args["training_manifest"], raw_root=args["raw_root"])
    with store.storage_lock(args["raw_root"]):
        for component in batch.components:
            store.publish(args["raw_root"], component.name, component.since, component.until,
                          args["raw_root"] / component.export.relative_directory)
    args["training_manifest"] = args["raw_root"] / "canonical/training.json"
    return args


@pytest.mark.parametrize("entrypoint", ["production", "shadow"])
def test_canonical_training_entrypoints_use_exact_event_time(tmp_path, monkeypatch, entrypoint):
    args = source(tmp_path)
    batch = daily.load_chunked_training_batch(args["training_manifest"], raw_root=args["raw_root"])
    actions_component = next(c for c in batch.components if c.name == "actions")
    path = args["raw_root"] / actions_component.export.relative_directory / "actions_part_001.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Reverse source order within one type: latest event is the first physical row.
    raw["customerActions"][0]["dateTimeUtc"] = "2026-01-01T20:00:00Z"
    raw["customerActions"][2]["dateTimeUtc"] = "2026-01-01T08:00:00Z"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with store.storage_lock(args["raw_root"]):
        for component in batch.components:
            store.publish(args["raw_root"], component.name, component.since, component.until,
                          args["raw_root"] / component.export.relative_directory)
    args["training_manifest"] = args["raw_root"] / "canonical/training.json"
    args["cfg"].min_user_interactions_for_eval = 2
    original, calls = core.train_prepared_data_with_metrics, []
    def checked(cfg, prepared, device):
        user, item = prepared.splits.eval_users[0], prepared.splits.eval_items[0]
        assert prepared.mappings.idx2user[user] == "SECRET_A"
        assert prepared.mappings.idx2item[item] == "123456"
        assert prepared.splits.user_pos_train[user] == {prepared.mappings.item2idx["123458"]}
        calls.append(True)
        return original(cfg, prepared, device)
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", checked)
    if entrypoint == "production":
        result = production.train_and_publish_production_model(**args)
        assert result.error_code is None and result.published
    else:
        from Application.model import mindbox_shadow_training as shadow
        monkeypatch.setattr(shadow, "REPORT_ROOT", tmp_path / "shadow_reports")
        result = shadow.shadow_train_daily_manifest(
            args["training_manifest"], raw_root=args["raw_root"], catalog_path=args["catalog_path"],
            cfg=args["cfg"], device=args["device"])
        assert result.error_code is None and result.training_completed
        assert not production.MODEL_ROOT.exists()
    assert calls == [True]


def gui_argv(args):
    config = args["catalog_path"].parent / "gui-config.json"
    config.write_text(json.dumps(vars(args["cfg"]), ensure_ascii=False), encoding="utf-8")
    return ["gui", "--manifest", str(args["training_manifest"]), "--raw-root", str(args["raw_root"]),
            "--catalog", str(args["catalog_path"]), "--config", str(config), "--device", "auto"]


def events(output):
    return [json.loads(line[len(cli.EVENT_PREFIX):]) for line in output.splitlines()
            if line.startswith(cli.EVENT_PREFIX)]


@pytest.mark.parametrize("problem,answer,exit_code", [
    ("pass", "", 0), ("warn", "YES\n", 0), ("warn", "NO\n", 130),
    ("warn", "", 130), ("block", "YES\n", 1),
])
def test_gui_cli_canonical_quality_and_publication(tmp_path, monkeypatch, capsys, problem, answer, exit_code):
    args = canonical_source(tmp_path, problem)
    old = old_model(production.MODEL_ROOT)
    monkeypatch.setattr(sys, "stdin", io.StringIO(answer))
    if exit_code:
        monkeypatch.setattr(core, "train_prepared_data_with_metrics", forbidden)
    else:
        train = core.train_prepared_data_with_metrics
        def checked(cfg, data, device):
            assert vars(cfg) == vars(args["cfg"])
            assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")
            assert cfg.item_feature_cols == core.TrainConfig().item_feature_cols
            return train(cfg, data, device)
        monkeypatch.setattr(core, "train_prepared_data_with_metrics", checked)
    assert cli.main(gui_argv(args)) == exit_code
    output = capsys.readouterr()
    recorded = events(output.out)
    assert recorded[0]["batch_id"] == store.catalog(args["raw_root"])["revision"]
    assert recorded[0]["quality"]["level"] == problem.upper()
    if problem == "warn":
        issue = recorded[0]["quality"]["issues"][0]
        assert issue["code"] == "MAPPED_ACTION_WITHOUT_PRODUCT"
        assert issue["count"] == 1 and issue["rate"] == 0.25
        assert issue["breakdown"] == {"ProsmotrProdukta": 1}
    if exit_code:
        assert len(recorded) == 1
        assert_old(production.MODEL_ROOT, old)
    else:
        result = recorded[-1]
        assert result["published"] and result["postpublish_validation"] and result["prepublish_disk_validation"]
        assert production._current(production.MODEL_ROOT)[0] == result["published_generation"]
        maps, checkpoint = core._load_artifacts(str(production.MODEL_ROOT))
        model, _, users, items = core._build_model_from_ckpt(checkpoint, torch.device("cpu"))
        assert (users, items) == (len(maps["idx2user"]), len(maps["idx2item"])) == (2, 3)
        assert model is not None and production.seen_items_from_checkpoint(checkpoint) is not None
        assert production.analytics_from_checkpoint(checkpoint) is not None
        assert (production.MODEL_ROOT / "runs" / OLD / "bprmf.pt").read_bytes() == b"old checkpoint sentinel"
    assert not any(p.name in ("orders.csv", "views.csv", "favorites.csv") for p in tmp_path.rglob("*.csv"))
    assert "SECRET" not in output.out + output.err


@pytest.mark.parametrize("cancel", [False, True])
def test_gui_cli_training_failure_preserves_previous_model(tmp_path, monkeypatch, capsys, cancel):
    args = canonical_source(tmp_path)
    old = old_model(production.MODEL_ROOT)
    def fail(*a):
        raise KeyboardInterrupt() if cancel else RuntimeError("SECRET")
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", fail)
    assert cli.main(gui_argv(args)) == (130 if cancel else 1)
    result = events(capsys.readouterr().out)[-1]
    assert result["error_code"] == ("CANCELLED" if cancel else "TRAINING_FAILED")
    assert not result["published"] and result["training_started"]
    assert_old(production.MODEL_ROOT, old)


def test_gui_cli_changed_batch_requires_new_preflight(tmp_path, monkeypatch, capsys):
    args = canonical_source(tmp_path, "warn")
    old = old_model(production.MODEL_ROOT)
    preflight = production.preflight_production_training
    def changed(*a, **kw):
        result = preflight(*a, **kw)
        catalog = store.catalog(args["raw_root"])
        catalog["revision"] = "b" * 32
        store.atomic_json(args["raw_root"] / "canonical/catalog.json", catalog)
        return result
    monkeypatch.setattr(production, "preflight_production_training", changed)
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", forbidden)
    monkeypatch.setattr(sys, "stdin", io.StringIO("YES\n"))
    assert cli.main(gui_argv(args)) == 1
    assert events(capsys.readouterr().out)[-1]["error_code"] == "PREFLIGHT_CHANGED"
    assert_old(production.MODEL_ROOT, old)


@pytest.mark.parametrize("bad_config", ["missing", "malformed", "other_catalog"])
def test_gui_cli_invalid_config_never_uses_defaults(tmp_path, monkeypatch, bad_config):
    args = canonical_source(tmp_path)
    argv = gui_argv(args)
    path = Path(argv[argv.index("--config") + 1])
    if bad_config == "missing":
        path.unlink()
    elif bad_config == "malformed":
        path.write_text("{invalid", encoding="utf-8")
    else:
        path.write_text(json.dumps({"data_dir": str(tmp_path / "other")}), encoding="utf-8")
    monkeypatch.setattr(production, "preflight_production_training", forbidden)
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", forbidden)
    assert cli.main(argv) == 1
    assert not production.MODEL_ROOT.exists()
