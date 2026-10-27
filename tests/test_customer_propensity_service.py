"""Tiny canonical history, synthetic key and immutable prediction SQLite only."""
from contextlib import closing
import hashlib
import hmac
import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from Application import customer_propensity_service as module
from Application.customer_propensity_service import CustomerPropensity, CustomerPropensityService, PropensityStatus as S
from Application.customer_propensity_presentation import propensity_content
from Application.customer_card_data import ViewState
from Application.model.propensity_ml.contracts import COLUMNS, FEATURES, canonical, checksum
from Application.model.propensity_ml.predictions import PredictionStore
from tests.test_customer_history_index import source as source_fixture, build


@pytest.fixture
def source(tmp_path):
    return source_fixture.__wrapped__(tmp_path)


@pytest.fixture
def backend(source, tmp_path, monkeypatch):
    ready = build(source)
    folder = tmp_path / "p"
    folder.mkdir()
    secret = bytes(range(32))
    key = tmp_path / "key"
    key.write_bytes(secret)
    key_id = hmac.new(secret, b"kanzler-propensity-key-id-v1", hashlib.sha256).hexdigest()
    pseudo = module.pseudonym("C", secret, key_id)
    c = {"pipeline": "calibrated", "feature_schema_version": 2, "input_features": list(FEATURES),
         "model_columns": list(COLUMNS), "horizon_days": 30, "target": "order_placed_within_30_days",
         "model_version": "a"*64, "dataset_version": "b"*64, "prediction_date": "2026-09-01T00:00:00Z",
         "target_until": "2026-10-01T00:00:00Z", "source_version": ready.source_version,
         "generation": ready.generation, "key_id": key_id}
    with closing(sqlite3.connect(folder / "predictions.sqlite")) as db:
        db.executescript("PRAGMA user_version=1; CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT); "
                         "CREATE TABLE predictions(customer_pseudonym TEXT PRIMARY KEY,probability REAL,warning_mask INTEGER) WITHOUT ROWID;")
        db.execute("INSERT INTO metadata VALUES('contract',?)", (canonical(c).decode(),))
        db.execute("INSERT INTO predictions VALUES(?,?,?)", (pseudo, .123456, 0))
        db.commit()
    payload = folder / "predictions.sqlite"
    m = {"artifact_schema": 1, "status": "RESEARCH_ONLY", "contract": c, "rows": 1,
         "files": {"predictions.sqlite": {"sha256": checksum(payload), "bytes": payload.stat().st_size}}}
    m["version"] = hashlib.sha256(canonical(m)).hexdigest()
    (folder/"manifest.json").write_bytes(canonical(m))
    store = PredictionStore(folder, verify_integrity=True)
    service = CustomerPropensityService(prediction_dir=folder, key_path=key, cache_dir=source[1].cache_dir)
    # Model/dataset compatibility is covered by the artifact suite and real
    # bounded probe. This fixture isolates canonical gate + key + real RO lookup.
    service._store = store
    monkeypatch.setattr(service, "_files", lambda: tuple(module._stamp(folder/n) for n in ("manifest.json", "predictions.sqlite")))
    service._artifact_version = service._files()
    monkeypatch.setattr(module, "_restricted_key", lambda p: p.read_bytes())
    return service, source[0], key


def test_hmac_fixed_vector():
    secret = bytes(range(32))
    key_id = hmac.new(secret, b"kanzler-propensity-key-id-v1", hashlib.sha256).hexdigest()
    assert module.pseudonym("SYNTHETIC-001", secret, key_id) == hmac.new(secret, b"kanzler-propensity-customer-v1\0SYNTHETIC-001", hashlib.sha256).hexdigest()
    with pytest.raises(module._Stale):
        module.pseudonym("SYNTHETIC-001", secret, "0"*64)
    with pytest.raises(module._Identity):
        module.pseudonym(" SYNTHETIC-001", secret, key_id)


def test_real_canonical_gate_and_lookup(backend, monkeypatch):
    service, root, _ = backend
    monkeypatch.setattr(module.history, "_validate_storage", lambda *a: pytest.fail("History scan forbidden"))
    result = service.get("C", raw_root=root)
    assert result.status is S.READY and result.probability == .123456
    assert result.prediction_date == "2026-09-01T00:00:00Z"
    assert service.validate(result, raw_root=root) == result


def test_alias_not_used_as_hmac_input(backend):
    service, root, _ = backend
    assert service.get("A", raw_root=root).status is S.IDENTITY_NOT_CONFIRMED


def test_missing_prediction(backend):
    service, root, _ = backend
    result = service.get("SYNTHETIC-MISSING", raw_root=root)
    assert result.status is S.NOT_FOUND and result.probability is None
    assert service.validate(result, raw_root=root) == result


def test_key_unavailable_and_no_creation(backend):
    service, root, key = backend
    key.unlink()
    assert service.get("C", raw_root=root).status is S.KEY_UNAVAILABLE
    assert not key.exists()


def test_wrong_key_id(backend):
    service, root, key = backend
    key.write_bytes(b"wrong-key"*8)
    result = service.get("C", raw_root=root)
    assert result.status is S.STALE and result.key_error == "KEY_ID_MISMATCH"


@pytest.mark.parametrize("field", ["source_version", "generation"])
def test_source_mismatch(backend, field):
    service, root, _ = backend
    service._store.manifest["contract"][field] = "different"
    assert service.get("C", raw_root=root).status is S.STALE


def test_changed_generation_at_delivery(backend):
    service, root, _ = backend
    result = service.get("C", raw_root=root)
    path = service._history.index_path(raw_root=root)
    with closing(sqlite3.connect(path)) as db:
        db.execute("UPDATE metadata SET value=? WHERE key='generation'", ("f"*32,))
        db.commit()
    assert service.validate(result, raw_root=root).status is S.STALE


def test_key_change_at_delivery(backend):
    service, root, key = backend
    result = service.get("C", raw_root=root)
    key.write_bytes(b"another-key"*4)
    assert service.validate(result, raw_root=root).status is S.STALE


def test_cancelled(backend):
    service, root, _ = backend
    assert service.get("C", raw_root=root, cancelled=lambda: True).status is S.CANCELLED


def test_artifact_missing(backend):
    service, root, _ = backend
    (service.prediction_dir/"predictions.sqlite").unlink()
    assert service.get("C", raw_root=root).status is S.NOT_READY


def test_bad_manifest_without_secret_read(tmp_path, monkeypatch):
    folder = tmp_path/"p"
    folder.mkdir()
    (folder/"manifest.json").write_text("not json")
    (folder/"predictions.sqlite").write_bytes(b"not sqlite")
    monkeypatch.setattr(module, "_restricted_key", lambda *a: pytest.fail("No secret read before artifact validation"))
    service = CustomerPropensityService(prediction_dir=folder)
    assert service.get("SYNTHETIC").status is S.ERROR


def test_no_sensitive_logging(backend, monkeypatch, caplog):
    service, root, _ = backend
    monkeypatch.setattr(service._store, "get_prediction", lambda *a: (_ for _ in ()).throw(RuntimeError("PRIVATE_CANONICAL PRIVATE_SECRET PRIVATE_HMAC")))
    assert service.get("C", raw_root=root).status is S.ERROR
    assert "PRIVATE" not in caplog.text


def prediction(*, warnings=(), probability=.123456):
    return CustomerPropensity(S.WARNING if warnings else S.READY, "C", probability,
                              "2026-09-01T00:00:00Z", "2026-10-01T00:00:00Z", "a"*64, warnings)


def test_ru_format_and_horizon():
    content = propensity_content(prediction())
    assert [field.title for field in content.fields] == [
        "Вероятность оформления заказа в следующие 30 дней", "Прогнозируемый период", "Модель"]
    assert content.fields[0].value == "12,35%"
    assert content.fields[1].value == "с 01.09.2026 до 01.10.2026"
    assert content.fields[2].value == "Customer Propensity"
    assert all("POSITIVE" not in (f.value or "") for f in content.fields)


@pytest.mark.parametrize("probability,value", [(0,"0,00%"),(.49981,"49,98%"),
                                               (1,"100,00%"),(1e-30,"< 0,01%")])
def test_probability_uses_two_decimals_without_false_zero(probability, value):
    result = prediction(probability=probability)
    assert propensity_content(result).fields[0].value == value
    assert result.probability == probability


def test_loading_keeps_three_placeholders_without_section_message():
    content = propensity_content(state=ViewState.LOADING)
    assert content.state is ViewState.LOADING and content.message == ""
    assert len(content.fields) == 3
    assert all(field.state is ViewState.LOADING and field.value is None for field in content.fields)


def test_near_zero_not_displayed_as_zero():
    content = propensity_content(prediction(warnings=("NEAR_ZERO_WITH_RECENT_FAVORITE",), probability=1e-30))
    assert content.fields[0].value == "Оценка ненадёжна"
    assert "занижать" in content.message


def test_low_individualization():
    content = propensity_content(prediction(warnings=("LOW_INDIVIDUALIZATION",)))
    assert content.fields[0].value == "12,35%"
    assert "Недостаточно индивидуальных сигналов" in content.message


@pytest.mark.parametrize("state", [S.NOT_FOUND,S.NOT_READY,S.STALE,S.IDENTITY_NOT_CONFIRMED,S.KEY_UNAVAILABLE,S.ERROR,S.CANCELLED])
def test_failure_never_zero(state):
    content = propensity_content(CustomerPropensity(state))
    assert content.state is not ViewState.READY
    assert content.message
    assert len(content.fields) == 3
    assert all(f.value is None for f in content.fields)


@pytest.mark.parametrize("value", [float("nan"),float("inf"),-.1,1.1])
def test_bad_probabilities_fail_closed(value):
    with pytest.raises(ValueError):
        prediction(probability=value)


@pytest.fixture
def key_settings(tmp_path, monkeypatch):
    settings = tmp_path / "customer_propensity.json"
    monkeypatch.setattr(module, "KEY_SETTINGS_PATH", settings, raising=False)
    return settings


def test_configured_physical_key_is_independent_of_home_appdata_and_cwd(key_settings, tmp_path, monkeypatch):
    physical = tmp_path / "Packages/Codex/LocalCache/Local/key"
    physical.parent.mkdir(parents=True)
    physical.write_bytes(bytes(range(32)))
    key_settings.write_text(json.dumps({"key_file": str(physical)}), encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "other-home"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "redirected-appdata"))
    monkeypatch.chdir(tmp_path)
    assert module.default_key_path() == physical
    service = CustomerPropensityService()
    monkeypatch.setattr(module, "_restricted_key", lambda p: p.read_bytes())
    assert service._key() == bytes(range(32))
    assert not (tmp_path / "other-home").exists()


def test_no_configuration_retains_single_legacy_location(key_settings, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "wrong"))
    assert module.default_key_path() == tmp_path / "AppData/Local/KANZLER-AI/propensity-secrets/propensity-v2.key"


@pytest.mark.parametrize("payload", ["not json", "{}", '[]', 'null', '{"key_file":"relative/key"}',
                                      '{"key_file":null}', '{"key_file":12}',
                                      '{"key_file":"/absolute","extra":true}'])
def test_bad_configuration_fails_closed_without_fallback(key_settings, payload, caplog):
    key_settings.write_text(payload, encoding="utf-8")
    service = CustomerPropensityService()
    with pytest.raises(module._KeyUnavailable) as caught:
        service._key()
    assert caught.value.reason == "CONFIG_INVALID"


def test_explicit_constructor_path_takes_precedence_over_configuration(key_settings, tmp_path):
    key_settings.write_text("invalid json", encoding="utf-8")
    service = CustomerPropensityService(key_path=tmp_path / "explicit")
    assert service.key_path == tmp_path / "explicit"


def test_missing_configured_key_does_not_fallback_or_create(key_settings, tmp_path):
    missing = tmp_path / "missing-original"
    key_settings.write_text(json.dumps({"key_file": str(missing)}), encoding="utf-8")
    with pytest.raises(module._KeyUnavailable) as caught:
        CustomerPropensityService()._key()
    assert caught.value.reason == "FILE_NOT_FOUND"
    assert not missing.exists()


@pytest.mark.parametrize("size", [0, 31, 4097])
def test_key_format_diagnosed_before_acl(tmp_path, size, monkeypatch):
    path = tmp_path / "key"
    path.write_bytes(b"x" * size)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("No ACL for bad format"))
    with pytest.raises(module._KeyUnavailable) as caught:
        module._restricted_key(path)
    assert caught.value.reason == "INVALID_FORMAT"


def test_key_access_denied_is_distinct(tmp_path, monkeypatch):
    path = tmp_path / "key"
    monkeypatch.setattr(Path, "stat", lambda *a, **k: (_ for _ in ()).throw(PermissionError("PRIVATE_PATH")))
    with pytest.raises(module._KeyUnavailable) as caught:
        module._restricted_key(path)
    assert caught.value.reason == "ACCESS_DENIED"
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.skipif(module.os.name != "nt", reason="Windows ACL checker")
@pytest.mark.parametrize("outcome,reason", [
    (SimpleNamespace(returncode=2, stdout=b"", stderr=b"PRIVATE_PATH"), "UNSAFE_ACL"),
    (SimpleNamespace(returncode=1, stdout=b"", stderr=b"PRIVATE_PATH"), "ACL_CHECK_FAILED"),
    (subprocess.TimeoutExpired("PRIVATE_COMMAND", 5), "ACL_CHECK_TIMEOUT"),
    (FileNotFoundError("PRIVATE_PATH"), "ACL_CHECKER_UNAVAILABLE")])
def test_acl_failures_remain_closed(tmp_path, monkeypatch, outcome, reason):
    path = tmp_path / "key"
    path.write_bytes(bytes(range(32)))
    def run(*a, **k):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(module._KeyUnavailable) as caught:
        module._restricted_key(path)
    assert caught.value.reason == reason
    assert "PRIVATE" not in str(caught.value)


def test_key_error_has_safe_reason_without_exception_details(backend, monkeypatch, caplog):
    service, root, _ = backend
    monkeypatch.setattr(module, "_restricted_key", lambda p: (_ for _ in ()).throw(module._KeyUnavailable("ACCESS_DENIED")))
    result = service.get("C", raw_root=root)
    assert result.status is S.KEY_UNAVAILABLE and result.key_error == "ACCESS_DENIED"
    assert "reason=ACCESS_DENIED" in caplog.text
    assert "PRIVATE" not in caplog.text
