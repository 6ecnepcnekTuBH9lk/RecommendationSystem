from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Application.mindbox import daily_training_batch as api
from Application.mindbox.training_batch import TrainingBatchWindow, TrainingBatchError
from scripts.mindbox_daily_batch import main


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


SECRETS = ["SECRET_KEY", "Authorization SECRET_AUTH", "https://synthetic.invalid/?signed=SECRET_URL",
           "SECRET_CUSTOMER", "SECRET_PRODUCT", "secret@example.test", "+79999999999"]


class Client:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []
        self.config = SimpleNamespace(api_url="https://synthetic.invalid", endpoint_id="test",
            secret_key=SECRETS[0], operations={name: f"Export.{name}" for name in ("customer_merges", "actions", "orders")})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def start_export(self, operation, payload):
        self.calls.append((operation.split(".")[-1], payload.copy()))
        if len(self.calls) == self.fail:
            raise RuntimeError(" ".join(SECRETS))
        return str(len(self.calls))

    def wait_for_export(self, *args, **kwargs):
        return [SECRETS[2], SECRETS[2]]

    def download_export(self, name, urls, storage):
        def write(url, path):
            path.write_text(json.dumps({"raw": SECRETS}), encoding="utf-8")
        return storage.save_export(name, urls, write)


@pytest.fixture
def window():
    return TrainingBatchWindow(utc("2026-08-01"), utc("2026-08-04"), utc("2025-01-01"))


def state_path(root):
    path, = root.glob("training_batches/*/state.json")
    return path


@pytest.mark.parametrize("since,until,count", [("2026-08-01", "2026-08-02", 1),
    ("2026-08-01", "2026-08-08", 7), ("2026-01-31", "2026-02-02", 2),
    ("2025-12-31", "2026-01-02", 2), ("2024-02-28", "2024-03-01", 2)])
def test_split_days(since, until, count):
    days = api.split_daily_windows(utc(since), utc(until))
    assert len(days) == count
    assert days[0].since == utc(since) and days[-1].until == utc(until)
    assert all(day.until - day.since == timedelta(days=1) for day in days)
    assert all(left.until == right.since for left, right in zip(days, days[1:]))


@pytest.mark.parametrize("since,until", [("2026-08-02", "2026-08-01"), ("2026-08-01", "2026-08-01"),
                                         ("2026-08-01T01:00", "2026-08-02")])
def test_invalid_split(since, until):
    with pytest.raises(TrainingBatchError):
        api.split_daily_windows(utc(since), utc(until))


def test_timezone_contract():
    with pytest.raises(TrainingBatchError):
        api.split_daily_windows(datetime(2026, 1, 1), utc("2026-01-02"))
    offset = timezone(timedelta(hours=3))
    assert len(api.split_daily_windows(utc("2026-01-01").astimezone(offset), utc("2026-01-02"))) == 1


@pytest.mark.parametrize("failure", [1, 2, 3, 5])
def test_resume_skips_ready_after_restart(tmp_path, window, failure):
    first = Client(failure)
    with pytest.raises(api.ChunkedBatchError) as exc:
        api.create_chunked_training_batch(first, raw_root=tmp_path, window=window)
    state = state_path(tmp_path)
    assert exc.value.state_path == state
    assert not state.with_name("manifest.json").exists()
    saved = api.load_chunked_training_batch(state, raw_root=tmp_path)
    assert saved.diagnostics["components_ready"] == failure - 1
    assert saved.diagnostics["components_failed"] == 1
    second = Client()
    final = api.resume_chunked_training_batch(second, state_path=state, raw_root=tmp_path)
    assert len(second.calls) == 7 - (failure - 1)
    expected = [("customer_merges", "2025-01-01 00:00"), ("actions", "2026-08-01 00:00"),
                ("orders", "2026-08-01 00:00"), ("actions", "2026-08-02 00:00"),
                ("orders", "2026-08-02 00:00"), ("actions", "2026-08-03 00:00"), ("orders", "2026-08-03 00:00")]
    assert [(name, payload["sinceDateTimeUtc"]) for name, payload in second.calls] == expected[failure - 1:]
    assert final.transport_complete and final.diagnostics["days_ready"] == 3
    assert all(c.export.parts_count == 2 for c in final.components)
    assert api.load_chunked_training_batch(state.with_name("manifest.json"), raw_root=tmp_path, require_complete=True) == final
    again = Client()
    assert api.resume_chunked_training_batch(again, state_path=state, raw_root=tmp_path) == final
    assert again.calls == []


def test_checkpoint_and_final_publication_order(tmp_path, window, monkeypatch):
    original = api._atomic_write
    writes = []

    def write(path, batch):
        writes.append((path.name, batch.diagnostics["components_ready"]))
        if path.name == "manifest.json":
            assert all(c.status == "READY" for c in batch.components)
        original(path, batch)

    monkeypatch.setattr(api, "_atomic_write", write)
    api.create_chunked_training_batch(Client(), raw_root=tmp_path, window=window)
    assert writes == [("state.json", n) for n in range(8)] + [("manifest.json", 7)]


def test_failed_final_replace_can_resume_without_exports(tmp_path, window, monkeypatch):
    original = api.os.replace

    def replace_file(source, target):
        if Path(target).name == "manifest.json":
            raise OSError("synthetic")
        original(source, target)

    monkeypatch.setattr(api.os, "replace", replace_file)
    with pytest.raises(api.ChunkedBatchError):
        api.create_chunked_training_batch(Client(), raw_root=tmp_path, window=window)
    state = state_path(tmp_path)
    assert api.load_chunked_training_batch(state, raw_root=tmp_path).diagnostics["components_ready"] == 7
    assert not state.with_name("manifest.json").exists()
    monkeypatch.setattr(api.os, "replace", original)
    client = Client()
    api.resume_chunked_training_batch(client, state_path=state, raw_root=tmp_path)
    assert client.calls == []


@pytest.mark.parametrize("damage", ["json", "gap", "duplicate", "period", "traversal", "missing_part", "version"])
def test_state_validation(tmp_path, window, damage):
    api.create_chunked_training_batch(Client(), raw_root=tmp_path, window=window)
    state = state_path(tmp_path)
    data = json.loads(state.read_text(encoding="utf-8"))
    if damage == "gap":
        data["components"].pop()
    elif damage == "duplicate":
        data["components"][3] = data["components"][1]
    elif damage == "period":
        data["window"]["interaction_since"] = "2026-08-02T00:00:00+00:00"
    elif damage == "traversal":
        data["components"][0]["export"]["relative_directory"] = "../outside"
    elif damage == "missing_part":
        data["components"][0]["export"]["parts_count"] = 3
    elif damage == "version":
        data["schema_version"] = 1
    state.write_text("{" if damage == "json" else json.dumps(data), encoding="utf-8")
    with pytest.raises(TrainingBatchError):
        api.load_chunked_training_batch(state, raw_root=tmp_path)


def test_config_mismatch_and_writer_lock(tmp_path, window):
    api.create_chunked_training_batch(Client(), raw_root=tmp_path, window=window)
    state = state_path(tmp_path)
    client = Client()
    client.config.endpoint_id = "another"
    with pytest.raises(TrainingBatchError, match="config"):
        api.resume_chunked_training_batch(client, state_path=state, raw_root=tmp_path)
    assert client.calls == []
    with api._writer_lock(state.parent):
        with pytest.raises(TrainingBatchError, match="writer"):
            api.resume_chunked_training_batch(Client(), state_path=state, raw_root=tmp_path)


def test_atomic_failed_checkpoint_preserves_previous_state(tmp_path, window, monkeypatch):
    api.create_chunked_training_batch(Client(), raw_root=tmp_path, window=window)
    state = state_path(tmp_path)
    before = state.read_bytes()
    batch = api.load_chunked_training_batch(state, raw_root=tmp_path)
    monkeypatch.setattr(api.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("synthetic")))
    with pytest.raises(OSError):
        api._atomic_write(state, replace(batch, transport_complete=True))
    assert state.read_bytes() == before


def test_cli_failure_resume_status_security(tmp_path, monkeypatch, capsys):
    import Application.mindbox as mindbox
    client = Client(5)
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *a: client.config)
    monkeypatch.setattr(mindbox, "MindboxClient", lambda *a: client)
    assert main(["export-daily", "--since", "2026-08-01", "--until", "2026-08-04",
                 "--merge-since", "2025-01-01 00:00", "--raw-root", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert "Last completed: 2026-08-02 actions" in output.out
    state = state_path(tmp_path)
    client = Client()
    assert main(["resume", "--state", str(state), "--raw-root", str(tmp_path)]) == 0
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *a: pytest.fail("Offline must not load env"))
    assert main(["status", "--state", str(state), "--raw-root", str(tmp_path)]) == 0
    assert main(["validate", "--manifest", str(state.with_name("manifest.json")), "--raw-root", str(tmp_path)]) == 0
    output_text = output.out + output.err + capsys.readouterr().out
    metadata = state.read_text(encoding="utf-8") + state.with_name("manifest.json").read_text(encoding="utf-8")
    for secret in SECRETS:
        assert secret not in output_text + metadata


def test_process_exit_leaves_pending_checkpoint(tmp_path, window):
    client = Client()
    original = client.wait_for_export

    def exit_process(*args, **kwargs):
        if len(client.calls) == 3:
            raise SystemExit(99)
        return original(*args, **kwargs)

    client.wait_for_export = exit_process
    with pytest.raises(SystemExit):
        api.create_chunked_training_batch(client, raw_root=tmp_path, window=window)
    state = state_path(tmp_path)
    batch = api.load_chunked_training_batch(state, raw_root=tmp_path)
    assert [c.status for c in batch.components[:3]] == ["READY", "READY", "PENDING"]
    resumed = Client()
    api.resume_chunked_training_batch(resumed, state_path=state, raw_root=tmp_path)
    assert resumed.calls[0][0] == "orders"
    assert len(resumed.calls) == 5


def test_download_failure_is_resumable(tmp_path, window):
    client = Client()
    client.download_export = lambda *a, **kw: (_ for _ in ()).throw(OSError("download error"))
    with pytest.raises(api.ChunkedBatchError):
        api.create_chunked_training_batch(client, raw_root=tmp_path, window=window)
    state = state_path(tmp_path)
    batch = api.load_chunked_training_batch(state, raw_root=tmp_path)
    assert batch.components[0].status == "FAILED"
    assert not state.with_name("manifest.json").exists()
    api.resume_chunked_training_batch(Client(), state_path=state, raw_root=tmp_path)


def test_state_flush_fsync_before_replace(tmp_path, window, monkeypatch):
    api.create_chunked_training_batch(Client(), raw_root=tmp_path, window=window)
    state = state_path(tmp_path)
    batch = api.load_chunked_training_batch(state, raw_root=tmp_path)
    calls = []
    original_sync, original_replace = api.os.fsync, api.os.replace

    def sync(fd):
        calls.append("fsync")
        original_sync(fd)

    def publish(source, target):
        assert calls == ["fsync"]
        assert json.loads(Path(source).read_text(encoding="utf-8"))["batch_id"] == batch.batch_id
        calls.append("replace")
        original_replace(source, target)

    monkeypatch.setattr(api.os, "fsync", sync)
    monkeypatch.setattr(api.os, "replace", publish)
    api._atomic_write(state, batch)
    assert calls == ["fsync", "replace"]
