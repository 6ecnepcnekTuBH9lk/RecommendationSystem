from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Application.mindbox import training_batch as batch_api
from Application.mindbox.raw_reader import RawExportError
from Application.mindbox.exceptions import MindboxExportTimeoutError
from scripts.mindbox_training_batch import main


SECRETS = ("SECRET_KEY", "Bearer SECRET_AUTH", "https://synthetic.invalid/part?signature=SECRET_URL",
           "SECRET_CUSTOMER", "SECRET_PRODUCT", "secret@example.test", "+79999999999")


class Client:
    def __init__(self, fail=None):
        self.config = SimpleNamespace(operations={name: f"Export.{name}" for name in batch_api.EXPORTS})
        self.calls = []
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def start_export(self, operation, payload):
        self.calls.append(("start", operation, payload.copy()))
        return f"id-{operation}"

    def wait_for_export(self, operation, export_id, **kwargs):
        self.calls.append(("wait", operation))
        if operation.endswith(str(self.fail)):
            raise MindboxExportTimeoutError(" ".join(SECRETS))
        return [SECRETS[2], SECRETS[2]]

    def download_export(self, name, urls, storage):
        self.calls.append(("download", name))

        def download(url, path):
            path.write_text(json.dumps({"synthetic": list(SECRETS)}), encoding="utf-8")

        return storage.save_export(name, urls, download)


@pytest.fixture
def window():
    return batch_api.TrainingBatchWindow(datetime(2026, 8, 1, tzinfo=timezone.utc),
                                         datetime(2026, 9, 1, tzinfo=timezone.utc),
                                         datetime(2025, 1, 1, tzinfo=timezone.utc))


@pytest.fixture
def published(tmp_path, window):
    client = Client()
    batch = batch_api.create_training_batch(client, raw_root=tmp_path, window=window)
    manifest = tmp_path / "training_batches" / batch.batch_id / "manifest.json"
    return batch, manifest, client


def test_three_exports_sequential_same_window_relative_paths_and_load(published, tmp_path):
    batch, manifest, client = published
    assert [call[0] for call in client.calls] == ["start", "wait", "download"] * 3
    starts = client.calls[::3]
    assert starts[1][2] == starts[2][2] == {"sinceDateTimeUtc": "2026-08-01 00:00", "tillDateTimeUtc": "2026-09-01 00:00"}
    assert starts[0][2] == {"sinceDateTimeUtc": "2025-01-01 00:00", "tillDateTimeUtc": "2026-09-01 00:00"}
    assert batch_api.load_training_batch(manifest, raw_root=tmp_path) == batch
    for entry in batch.exports:
        assert not Path(entry.relative_directory).is_absolute()
        assert entry.parts_count == 2
    text = manifest.read_text(encoding="utf-8") + repr(batch)
    for secret in SECRETS:
        assert secret not in text
    assert "urls" not in text and "Authorization" not in text


@pytest.mark.parametrize("field,delta", [("merge_since", timedelta(days=1000)),
                                         ("interaction_since", timedelta(days=40)),
                                         ("interaction_until", timedelta(days=-31))])
def test_invalid_period(window, field, delta):
    with pytest.raises(batch_api.TrainingBatchError):
        replace(window, **{field: getattr(window, field) + delta})


def test_naive_and_subminute_rejected(window):
    for value in (window.merge_since.replace(tzinfo=None), window.merge_since.replace(second=1)):
        with pytest.raises(batch_api.TrainingBatchError):
            replace(window, merge_since=value)
    assert replace(window, merge_since=window.interaction_since)


@pytest.mark.parametrize("fail", batch_api.EXPORTS)
def test_export_failure_no_manifest_and_raw_not_rolled_back(tmp_path, window, fail):
    with pytest.raises(MindboxExportTimeoutError):
        batch_api.create_training_batch(Client(fail), raw_root=tmp_path, window=window)
    assert not list(tmp_path.rglob("manifest.json"))
    assert len(list(tmp_path.rglob("*_part_*.json"))) == batch_api.EXPORTS.index(fail) * 2


def test_atomic_flush_fsync_before_replace(tmp_path, window, monkeypatch):
    replace_fn, fsync = batch_api.os.replace, batch_api.os.fsync
    calls = []

    def sync(fd):
        calls.append("sync")
        fsync(fd)

    def publish(source, target):
        if Path(target).name == "manifest.json":
            assert calls[-1] == "sync"
            assert not Path(target).exists()
            assert len(json.loads(Path(source).read_text(encoding="utf-8"))["exports"]) == 3
            calls.append("publish")
        return replace_fn(source, target)

    monkeypatch.setattr(batch_api.os, "fsync", sync)
    monkeypatch.setattr(batch_api.os, "replace", publish)
    batch_api.create_training_batch(Client(), raw_root=tmp_path, window=window)
    assert calls[-2:] == ["sync", "publish"]
    assert calls.count("publish") == 1


def test_atomic_publication_failure_no_final_manifest(tmp_path, window, monkeypatch):
    original = batch_api.os.replace

    def fail(source, target):
        if Path(target).name == "manifest.json":
            raise OSError("synthetic failure")
        return original(source, target)

    monkeypatch.setattr(batch_api.os, "replace", fail)
    with pytest.raises(OSError):
        batch_api.create_training_batch(Client(), raw_root=tmp_path, window=window)
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob("manifest.tmp"))
    assert len(list(tmp_path.rglob("*_part_*.json"))) == 6


@pytest.mark.parametrize("problem", ["version", "missing", "duplicate", "nonexistent", "traversal", "absolute", "windows", "parts", "wrong_type", "extra"])
def test_invalid_manifest(published, tmp_path, problem):
    _, manifest, _ = published
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if problem == "version":
        raw["schema_version"] = 2
    elif problem == "missing":
        raw["exports"].pop()
    elif problem == "duplicate":
        raw["exports"][1] = raw["exports"][0]
    elif problem == "extra":
        raw["urls"] = [SECRETS[2]]
    elif problem == "parts":
        raw["exports"][0]["parts_count"] = 3
    else:
        raw["exports"][0]["relative_directory"] = {
            "nonexistent": "customer_merges/19990101_000000", "traversal": "../outside",
            "absolute": "/outside", "windows": "C:\\outside", "wrong_type": "orders/20260101_000000",
        }[problem]
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((batch_api.TrainingBatchError, RawExportError)):
        batch_api.load_training_batch(manifest, raw_root=tmp_path)


def test_bridge_exact_directories_no_latest(published, tmp_path, monkeypatch):
    from Application.model import mindbox_training_preparation as pipeline
    from Application.mindbox import raw_reader
    batch, _, _ = published
    marker = object()
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(pipeline, "prepare_training_data_from_mindbox", prepare)
    monkeypatch.setattr(raw_reader, "select_export_directory", lambda *a, **k: pytest.fail("No latest selection"))
    cfg = object()
    assert batch_api.prepare_training_data_from_batch(batch, raw_root=tmp_path, catalog_path="catalog.csv",
                                                       train_config=cfg, diagnose=True) is marker
    for entry in batch.exports:
        assert captured[f"{entry.name}_export_dir"] == (tmp_path / entry.relative_directory).resolve()
    assert captured["train_config"] is cfg and captured["diagnose"] is True


def test_cli_mocked_export_and_offline_validate(tmp_path, window, monkeypatch, capsys):
    import Application.mindbox as mindbox
    client = Client()
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *args: client.config)
    monkeypatch.setattr(mindbox, "MindboxClient", lambda config: client)
    assert main(["export", "--raw-root", str(tmp_path), "--since", "2026-08-01 00:00", "--until", "2026-09-01 00:00",
                 "--merge-since", "2025-01-01 00:00"]) == 0
    manifest, = tmp_path.rglob("manifest.json")
    monkeypatch.setattr(mindbox, "MindboxClient", lambda *args: pytest.fail("Offline must not create client"))
    assert main(["validate", "--raw-root", str(tmp_path), "--manifest", str(manifest)]) == 0
    output = capsys.readouterr()
    for secret in SECRETS:
        assert secret not in output.out + output.err


def test_cli_failure_does_not_leak_transport_error(tmp_path, monkeypatch, capsys):
    import Application.mindbox as mindbox
    client = Client("orders")
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *args: client.config)
    monkeypatch.setattr(mindbox, "MindboxClient", lambda config: client)
    assert main(["export", "--raw-root", str(tmp_path), "--since", "2026-08-01 00:00", "--until", "2026-09-01 00:00",
                 "--merge-since", "2025-01-01 00:00"]) == 1
    output = capsys.readouterr()
    for secret in SECRETS:
        assert secret not in output.out + output.err


@pytest.mark.parametrize("complete", [True, False])
def test_cli_offline_prepare_completeness(published, tmp_path, monkeypatch, capsys, complete):
    import Application.mindbox as mindbox
    _, manifest, _ = published
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *a: pytest.fail("No env in offline path"))
    stats = SimpleNamespace(unique_users=2, unique_items=3, events_total=10, train_pairs_after_aggregation=4,
                            eval_events=1, total_train_weight=20)
    result = SimpleNamespace(complete=complete, diagnostics=SimpleNamespace(bpr=stats,
        malformed_actions=0 if complete else 1,
        malformed_action_system_names={} if complete else {"ProsmotrProdukta": 1},
        resolution=SimpleNamespace(total=SimpleNamespace(unresolved=0))))
    monkeypatch.setattr(batch_api, "prepare_training_data_from_batch", lambda *a, **kw: result)
    assert main(["prepare", "--raw-root", str(tmp_path), "--manifest", str(manifest), "--diagnose"]) == int(not complete)
    assert f"Training data complete: {complete}" in capsys.readouterr().out
