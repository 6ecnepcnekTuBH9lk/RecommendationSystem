from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
import sys

import pytest
import torch

from Application.customer_profiles import CustomerContactIndex
from Application.mindbox import daily_training_batch as daily
from Application.mindbox import customer_profile_snapshot as profiles
from Application.mindbox.training_batch import TrainingBatchWindow, TrainingBatchExport
from Application.model import BPRMF as core
from Application.model import mindbox_recommendation_smoke as smoke
from Application.model.interaction_analytics import InteractionAnalytics
from scripts.mindbox_recommendation_smoke import main


def forbidden(*args, **kwargs):
    pytest.fail("Forbidden legacy/network call")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(smoke, "REPORT_ROOT", tmp_path / "reports")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    production = tmp_path / "Модель"
    for name in ("current.json", "runs/old-generation/bprmf.pt", ".staging/pending/sentinel", "Рекомендации.xlsx"):
        path = production / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sentinel\x00\xff")
    before = smoke._tree_state(production)
    roots = []
    factory = smoke.tempfile.TemporaryDirectory
    def temporary(**kwargs):
        result = factory(**kwargs)
        roots.append(Path(result.name).resolve())
        return result
    monkeypatch.setattr(smoke.tempfile, "TemporaryDirectory", temporary)
    yield roots
    assert Path.cwd() == tmp_path
    assert smoke._tree_state(production) == before
    assert all(not p.exists() for p in roots)


def inputs(tmp_path, problem="pass"):
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    until = since + timedelta(days=1)
    raw = tmp_path / "raw"
    users = ["SECRET_CUSTOMER_A", "SECRET_CUSTOMER_B", "SECRET_CUSTOMER_C"]
    actions = []
    for i in range(25):
        actions.append({"ids": {"mindboxId": f"SECRET_ACTION_{i}"},
            "customer": {"ids": {"mindboxId": users[min(i // 12, 2)]}},
            "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}},
            "dateTimeUtc": since.isoformat(), "creationDateTimeUtc": since.isoformat(),
            "products": [{"ids": {"offline1C": f"{100000 + i % 24}_SECRET_ITEM"}}]})
    if problem in ("warn", "block"):
        actions.append({**actions[0], "products": [] if problem == "warn" else [{"ids": {"offline1C": "999999_SECRET_ITEM"}}]})
    components = []
    for name, key, records in (("customer_merges", "customerMerges", []),
                               ("actions", "customerActions", actions), ("orders", "orders", [])):
        directory = raw / name / "20260101_000000"
        directory.mkdir(parents=True)
        (directory / f"{name}_part_001.json").write_text(json.dumps({key: records}), encoding="utf-8")
        export = TrainingBatchExport(name, "SECRET_EXPORT", name, directory.relative_to(raw).as_posix(), 1)
        components.append(daily.BatchComponent(name, since, until, name, "READY", export))
    batch = daily.ChunkedTrainingBatch("a" * 32, since, TrainingBatchWindow(since, until, since),
                                      "0" * 64, tuple(components), True)
    manifest = raw / "training_batches" / batch.batch_id / "manifest.json"
    manifest.parent.mkdir(parents=True)
    daily._atomic_write(manifest, batch)
    directory = raw / "customers" / "20260101_000000"
    directory.mkdir(parents=True)
    records = [{"ids": {"mindboxId": user}, "email": "secret@example.test",
                "lastActivatedCard": {"ids": {"number": "SECRET_CARD"}},
                **({"mobilePhone": "+79161234567"} if index < 2 else {})} for index, user in enumerate(users)]
    (directory / "customers_part_001.json").write_text(json.dumps({"customers": records}), encoding="utf-8")
    snapshot = profiles.CustomerProfileSnapshot("b" * 32, since.isoformat(), "customers/20260101_000000", 1,
                                               "customer_merges/20260101_000000", 1, batch.batch_id)
    profile = profiles._publish_snapshot(snapshot, raw)
    catalog = tmp_path / "Номенклатура.csv"
    catalog.write_text("КодНоменклатуры|Марка|НазваниеНаСайте|ВидНоменклатуры|Коллекция|Остаток\n" +
        "".join(f"{100000 + i}|brand|Synthetic {i}|kind|NOS|100\n" for i in range(24)), encoding="utf-8-sig")
    settings = tmp_path / "Настройки"
    settings.mkdir()
    (settings / "filter_settings.json").write_text('{"active_collections": ["NOS"]}', encoding="utf-8")
    (settings / "sentinel.txt").write_text("SecretKey", encoding="utf-8")
    return manifest, profile, raw, catalog


def run(source, **kwargs):
    manifest, profile, raw, catalog = source
    cfg = core.TrainConfig(data_dir=str(catalog.parent), epochs=1, embedding_dim=4,
                           batch_size=16, n_neg=1, topk=10)
    return smoke.run_recommendation_smoke(manifest, profile, raw_root=raw, catalog_path=catalog,
        cfg=cfg, device=torch.device("cpu"), max_export_users=1, **kwargs)


@pytest.mark.parametrize("problem", ["pass", "warn"])
def test_real_e2e_offline_serialization_export(environment, tmp_path, monkeypatch, capsys, problem):
    source = inputs(tmp_path, problem)
    for name in ("_require_interaction_sources", "_validate_interaction_source_schemas",
                 "_load_historical_item_conversion", "_build_user_seen_sets"):
        monkeypatch.setattr(core, name, forbidden)
    export = core.export_recommendations_excel
    load = core._load_artifacts
    evidence = {"loads": 0, "ranks": 0, "export": False}
    def checked_load(model_dir):
        maps, ckpt = load(model_dir)
        assert smoke.seen_items_from_checkpoint(ckpt) is not None
        assert smoke.analytics_from_checkpoint(ckpt) is not None
        assert Path(ckpt["config"]["data_dir"]).is_absolute()
        assert "contacts" not in ckpt and "customer_contacts" not in ckpt
        for path in Path(model_dir).rglob("*"):
            if path.is_file():
                assert all(secret not in path.read_bytes() for secret in
                           (b"secret@example.test", b"79161234567", b"SECRET_CARD"))
        evidence["loads"] += 1
        return maps, ckpt
    monkeypatch.setattr(core, "_load_artifacts", checked_load)
    def checked_export(**kwargs):
        root = Path.cwd()
        assert root in environment
        assert list((root / "ВходныеДанные").iterdir()) == [root / "ВходныеДанные" / "Номенклатура.csv"]
        assert (root / "Настройки" / "sentinel.txt").read_text() == "SecretKey"
        assert isinstance(kwargs["customer_contacts"], CustomerContactIndex)
        assert kwargs["customer_contacts"].diagnostics.with_phone == 2
        assert kwargs["filter_seen"] and kwargs["device_str"] == "cpu"
        evidence["export"] = True
        return export(**kwargs)
    monkeypatch.setattr(core, "export_recommendations_excel", checked_export)
    # Nested legacy functions cannot be monkeypatched as module attributes.
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name in ("_load_user_fields", "_accumulate_activity"):
            forbidden()
        if event == "return" and frame.f_code.co_name == "_rank_users_by_loyalty":
            assert isinstance(frame.f_locals["analytics"], InteractionAnalytics)
            assert frame.f_locals["require_phone"] is True
            assert arg == [0, 1]
            evidence["ranks"] += 1
    try:
        sys.setprofile(profile)
        report, path = run(source)
    finally:
        sys.setprofile(previous)
    assert report["status"] == "COMPLETED", (report["error_code"], evidence)
    assert report["quality"]["level"] == problem.upper()
    assert report["dataset"]["complete"] == (problem == "pass")
    assert report["training"]["epochs_completed"] == 1
    assert report["export"]["xlsx_rows"] == report["export"]["format1_rows"] == 1
    assert report["export"]["kanzler_ml_rows"] == 10
    assert report["cleanup"]["temporary_artifacts_removed"]
    assert evidence == {"loads": 2, "ranks": 1, "export": True}
    assert json.loads(path.read_text(encoding="utf-8")) == report
    output = capsys.readouterr()
    text = path.read_text(encoding="utf-8") + output.out + output.err
    assert all(secret not in text for secret in ("SECRET", "SecretKey", "100001", "secret@example.test",
                                                "79161234567", str(tmp_path)))
    assert list(path.parent.iterdir()) == [path]


def test_block_does_not_train_or_export(environment, tmp_path, monkeypatch):
    source = inputs(tmp_path, "block")
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", forbidden)
    monkeypatch.setattr(core, "export_recommendations_excel", forbidden)
    report, _ = run(source)
    assert report["error_code"] == "QUALITY_BLOCK"
    assert report["training"]["epochs_completed"] == 0
    assert not report["artifact"]["saved"]


@pytest.mark.parametrize("target,code", [("train_prepared_data_with_metrics", "TRAINING_FAILED"),
    ("_save_artifacts", "ARTIFACT_FAILED"), ("_load_artifacts", "ARTIFACT_FAILED"),
    ("load_customer_contact_index", "CONTACTS_FAILED"), ("export_recommendations_excel", "EXPORT_FAILED"),
    ("_validate_outputs", "VALIDATION_FAILED")])
@pytest.mark.parametrize("cancel", [False, True])
def test_failure_and_cancellation_cleanup(environment, tmp_path, monkeypatch, target, code, cancel):
    source = inputs(tmp_path)
    def fail(*args, **kwargs):
        # Include a partial PII output to prove outer cleanup on all stages.
        (Path.cwd() / "partial.csv").write_text("secret@example.test", encoding="utf-8")
        print("SECRET_CUSTOMER SecretKey")
        raise KeyboardInterrupt() if cancel else RuntimeError("secret@example.test SecretKey")
    owner = smoke if target in ("load_customer_contact_index", "_validate_outputs") else core
    monkeypatch.setattr(owner, target, fail)
    report, path = run(source)
    assert report["error_code"] == ("CANCELLED" if cancel else code)
    assert report["cleanup"]["temporary_artifacts_removed"]
    assert "SECRET" not in path.read_text(encoding="utf-8")
    assert "secret@example.test" not in path.read_text(encoding="utf-8")


def test_cli_cancellation_130(environment, tmp_path, monkeypatch, capsys):
    manifest, profile, raw, catalog = inputs(tmp_path)
    def cancel(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", cancel)
    assert main(["--training-manifest", str(manifest), "--profile-manifest", str(profile),
                 "--raw-root", str(raw), "--catalog", str(catalog), "--epochs", "1"]) == 130
    output = capsys.readouterr()
    assert "CANCELLED" in output.out
    assert str(tmp_path) not in output.out + output.err


def test_invalid_preparation_safe_report(environment, tmp_path):
    source = inputs(tmp_path)
    source[0].write_text("SecretKey broken", encoding="utf-8")
    report, _ = run(source)
    assert report["error_code"] == "PREPARATION_FAILED"


@pytest.mark.parametrize("corruption", ["zip", "xlsx_header", "csv_header", "missing", "counts"])
def test_actual_output_corruption_rejected(environment, tmp_path, monkeypatch, corruption):
    source = inputs(tmp_path)
    validate = smoke._validate_outputs
    def corrupt(root, k):
        if corruption == "zip":
            (root / "recommendations.xlsx").write_bytes(b"invalid zip")
        elif corruption == "xlsx_header":
            from openpyxl import Workbook
            book = Workbook(write_only=True)
            book.create_sheet("Рекомендации").append(["WrongHeader"])
            book.save(root / "recommendations.xlsx")
            book.close()
        elif corruption == "csv_header":
            (root / "format1.csv").write_text("Wrong;Header\n", encoding="utf-8-sig")
        elif corruption == "missing":
            (root / "format1.csv").unlink()
        else:
            (root / "format1.csv").write_text("CustomerID;ProductID\n", encoding="utf-8-sig")
        return validate(root, k)
    monkeypatch.setattr(smoke, "_validate_outputs", corrupt)
    report, _ = run(source)
    assert report["error_code"] == "VALIDATION_FAILED"
    assert not report["export"]["completed"]


def test_zero_rows_allowed_when_business_filters_exclude_items(environment, tmp_path):
    source = inputs(tmp_path)
    catalog = source[-1]
    catalog.write_text(catalog.read_text(encoding="utf-8-sig").replace("|100\n", "|0\n"), encoding="utf-8-sig")
    report, _ = run(source)
    assert report["status"] == "COMPLETED"
    assert all(report["export"][key] == 0 for key in ("xlsx_rows", "format1_rows", "kanzler_ml_rows"))


def test_cleanup_failure_is_reported(environment, tmp_path, monkeypatch):
    source = inputs(tmp_path, "block")
    factory = smoke.tempfile.TemporaryDirectory
    def temporary(**kwargs):
        result = factory(**kwargs)
        cleanup = result.cleanup
        def fail_after_removal():
            cleanup()
            raise OSError("SecretKey secret@example.test")
        result.cleanup = fail_after_removal
        return result
    monkeypatch.setattr(smoke.tempfile, "TemporaryDirectory", temporary)
    report, path = run(source)
    assert report["error_code"] == "CLEANUP_FAILED"
    assert not report["cleanup"]["temporary_artifacts_removed"]
    assert "SecretKey" not in path.read_text(encoding="utf-8")


def test_atomic_report_failure_removes_temporary_file(environment, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("SecretKey")
    monkeypatch.setattr(smoke.os, "replace", fail)
    path = tmp_path / "safe-report" / "report.json"
    with pytest.raises(OSError):
        smoke._atomic_report(path, {"status": "FAILED"})
    assert list(path.parent.iterdir()) == []
