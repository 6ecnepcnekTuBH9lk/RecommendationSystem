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


def test_state_created_callback_precedes_first_network_call(tmp_path, window):
    client = Client()
    notifications = []

    def created(path):
        assert not client.calls
        batch = api.load_chunked_training_batch(path, raw_root=tmp_path)
        assert all(c.status == "PENDING" for c in batch.components)
        notifications.append(path)

    api.create_chunked_training_batch(client, raw_root=tmp_path, window=window, on_state_created=created)
    assert notifications == [state_path(tmp_path)]


def test_cli_prints_durable_state_before_export(tmp_path, monkeypatch, capsys):
    import Application.mindbox as mindbox
    client = PrefixClient()
    original = client.start_export

    def start_export(operation, payload):
        if not client.calls:
            output = capsys.readouterr().out
            assert f"State: {next(tmp_path.glob('canonical/jobs/*/state.json'))}" in output
        return original(operation, payload)

    client.start_export = start_export
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *a: client.config)
    monkeypatch.setattr(mindbox, "MindboxClient", lambda *a: client)
    assert main(["export-daily", "--since", "2026-08-01", "--until", "2026-08-04",
                 "--merge-since", "2025-01-01 00:00", "--raw-root", str(tmp_path)]) == 0


class PrefixClient(Client):
    def download_export(self, name, urls, storage):
        assert name in ("customer_merges", "actions", "orders")  # Never CustomersAPI.
        key = {"customer_merges": "customerMerges", "actions": "customerActions", "orders": "orders"}[name]
        records = []
        if name == "actions":
            stamp = self.calls[-1][1]["sinceDateTimeUtc"].replace(" ", "T") + ":00Z"
            records = [{"ids": {"mindboxId": "SECRET_ACTION"}, "customer": {"ids": {"mindboxId": "SECRET_CUSTOMER"}},
                "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}},
                "dateTimeUtc": stamp, "creationDateTimeUtc": stamp,
                "products": [{"ids": {"offline1C": "123456_SECRET"}}]}]
        def write(url, path):
            path.write_text(json.dumps({key: records}), encoding="utf-8")
        return storage.save_export(name, urls, write)


def prefix_source(root, ready_days=6, partial_actions=False):
    client = PrefixClient(fail=2 + ready_days * 2 + int(partial_actions))
    window = TrainingBatchWindow(utc("2026-08-01"), utc("2026-08-14"), utc("2025-01-01"))
    with pytest.raises(api.ChunkedBatchError):
        api.create_chunked_training_batch(client, raw_root=root, window=window)
    state = state_path(root)
    if partial_actions:
        batch = api.load_chunked_training_batch(state, raw_root=root)
        api._atomic_write(state, replace(batch, components=tuple(
            replace(c, status="PENDING") if c.status == "FAILED" else c for c in batch.components)))
    return state


@pytest.mark.parametrize("partial_actions", [False, True])
def test_finalize_prefix_reuses_six_days_and_prepares(tmp_path, partial_actions, monkeypatch):
    from Application.model import BPRMF as core
    import shutil
    import socket
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("No network"))
    for name in ("train_prepared_data_with_metrics", "export_recommendations_excel", "prepare_training_data_from_csv"):
        monkeypatch.setattr(core, name, lambda *a, **kw: pytest.fail("No training/export/CSV"))
    monkeypatch.setattr(shutil, "copyfile", lambda *a, **kw: pytest.fail("No raw copies"))
    state = prefix_source(tmp_path, partial_actions=partial_actions)
    old = state.read_bytes()
    source = api.load_chunked_training_batch(state, raw_root=tmp_path)
    raw_before = {p: p.read_bytes() for p in tmp_path.glob("*/*/*_part_*.json")}
    client = PrefixClient()
    final = api.finalize_chunked_training_batch_prefix(client, state_path=state, raw_root=tmp_path)
    assert client.calls == [("customer_merges", {"sinceDateTimeUtc": "2025-01-01 00:00", "tillDateTimeUtc": "2026-08-07 00:00"})]
    assert final.batch_id != source.batch_id
    assert final.diagnostics["days_ready"] == 6
    assert final.window.interaction_until == utc("2026-08-07")
    assert final.components[1:] == source.components[1:13]
    assert final.components[0].export.relative_directory != source.components[0].export.relative_directory
    assert state.read_bytes() == old
    assert not state.with_name("manifest.json").exists()
    assert all(p.read_bytes() == data for p, data in raw_before.items())
    manifest = tmp_path / "training_batches" / final.batch_id / "manifest.json"
    assert api.load_chunked_training_batch(manifest, raw_root=tmp_path, require_complete=True) == final
    assert api.load_chunked_training_batch(manifest.with_name("state.json"), raw_root=tmp_path) == final
    catalog = tmp_path / "nomenclature.csv"
    catalog.write_text("КодНоменклатуры\n123456\n", encoding="utf-8-sig")
    prepared = api.prepare_training_data_from_chunked_batch(final, raw_root=tmp_path, catalog_path=catalog,
        train_config=core.TrainConfig(data_dir=str(tmp_path)), diagnose=True)
    assert prepared.diagnostics.view_interactions == 12  # Two published parts per day.
    assert all(not (tmp_path / f"{name}.csv").exists() for name in ("Заказы", "Просмотры", "Избранное"))


def test_finalize_prefix_minimum_day_and_source_lock(tmp_path):
    state = prefix_source(tmp_path, ready_days=0, partial_actions=True)
    client = PrefixClient()
    with pytest.raises(TrainingBatchError, match="one complete"):
        api.finalize_chunked_training_batch_prefix(client, state_path=state, raw_root=tmp_path)
    with api._writer_lock(state.parent):
        with pytest.raises(TrainingBatchError, match="writer"):
            api.finalize_chunked_training_batch_prefix(client, state_path=state, raw_root=tmp_path)
    assert client.calls == []


@pytest.mark.parametrize("failure", ["start", "wait", "download", "cancel"])
def test_finalize_prefix_failure_does_not_publish_or_change_source(tmp_path, monkeypatch, failure):
    state = prefix_source(tmp_path, ready_days=1)
    old = state.read_bytes()
    client = PrefixClient()
    def fail(*a, **kw):
        raise KeyboardInterrupt() if failure == "cancel" else RuntimeError(" ".join(SECRETS))
    monkeypatch.setattr(client, {"start": "start_export", "wait": "wait_for_export", "download": "download_export",
                                "cancel": "wait_for_export"}[failure], fail)
    with pytest.raises(KeyboardInterrupt if failure == "cancel" else TrainingBatchError):
        api.finalize_chunked_training_batch_prefix(client, state_path=state, raw_root=tmp_path)
    assert not list(tmp_path.glob("training_batches/*/manifest.json"))
    assert state.read_bytes() == old


@pytest.mark.parametrize("damage", ["traversal", "config", "old_merges"])
def test_finalize_prefix_validation_protections(tmp_path, monkeypatch, damage):
    state = prefix_source(tmp_path, ready_days=1)
    client = PrefixClient()
    if damage == "traversal":
        data = json.loads(state.read_text(encoding="utf-8"))
        data["components"][1]["export"]["relative_directory"] = "../outside"
        state.write_text(json.dumps(data), encoding="utf-8")
    elif damage == "config":
        client.config.endpoint_id = "other"
    else:
        batch = api.load_chunked_training_batch(state, raw_root=tmp_path)
        parts = api.part_files(tmp_path / batch.components[0].export.relative_directory, "customer_merges")
        monkeypatch.setattr(client, "download_export", lambda *a, **kw: parts)
    old = state.read_bytes()
    with pytest.raises(TrainingBatchError):
        api.finalize_chunked_training_batch_prefix(client, state_path=state, raw_root=tmp_path)
    assert state.read_bytes() == old
    assert not list(tmp_path.glob("training_batches/*/manifest.json"))
    if damage != "old_merges":
        assert client.calls == []


def test_finalize_prefix_atomic_commit_and_lock_against_resume(tmp_path, monkeypatch):
    state = prefix_source(tmp_path, ready_days=1)
    old = state.read_bytes()
    client = PrefixClient()
    start = client.start_export
    def locked_start(*a):
        with pytest.raises(TrainingBatchError, match="writer"):
            api.resume_chunked_training_batch(PrefixClient(), state_path=state, raw_root=tmp_path)
        return start(*a)
    monkeypatch.setattr(client, "start_export", locked_start)
    original = api.os.replace
    def fail_manifest(source, target):
        if Path(target).name == "manifest.json":
            assert Path(target).with_name("state.json").exists()
            raise OSError("SecretKey")
        return original(source, target)
    monkeypatch.setattr(api.os, "replace", fail_manifest)
    with pytest.raises(TrainingBatchError):
        api.finalize_chunked_training_batch_prefix(client, state_path=state, raw_root=tmp_path)
    assert state.read_bytes() == old
    assert not list(tmp_path.glob("training_batches/*/manifest.json"))
    assert not list(tmp_path.glob("training_batches/*/.state-*.tmp"))


@pytest.mark.parametrize("cancel", [False, True])
def test_finalize_prefix_cli_safe_output(tmp_path, monkeypatch, capsys, cancel):
    import Application.mindbox as mindbox
    state = prefix_source(tmp_path)
    old = state.read_bytes()
    client = PrefixClient()
    if cancel:
        def interrupt(*a, **kw):
            raise KeyboardInterrupt()
        monkeypatch.setattr(client, "start_export", interrupt)
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *a: client.config)
    monkeypatch.setattr(mindbox, "MindboxClient", lambda *a: client)
    assert main(["finalize-prefix", "--state", str(state), "--raw-root", str(tmp_path), "--timeout", "3600"]) == (130 if cancel else 0)
    output = capsys.readouterr()
    assert all(secret not in output.out + output.err for secret in SECRETS)
    if not cancel:
        assert "Ready days: 6" in output.out
        assert "Final window: 2026-08-01 -> 2026-08-07" in output.out
    assert state.read_bytes() == old
    assert not state.with_name("manifest.json").exists()


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
    client = PrefixClient(5)
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *a: client.config)
    monkeypatch.setattr(mindbox, "MindboxClient", lambda *a: client)
    assert main(["export-daily", "--since", "2026-08-01", "--until", "2026-08-04",
                 "--merge-since", "2025-01-01 00:00", "--raw-root", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert '"source": "orders"' in output.out and '"category": "failed"' in output.out
    state = next(tmp_path.glob("canonical/jobs/*/state.json"))
    client = PrefixClient()
    assert main(["resume", "--state", str(state), "--raw-root", str(tmp_path)]) == 0
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *a: pytest.fail("Offline must not load env"))
    assert main(["status", "--state", str(state), "--raw-root", str(tmp_path)]) == 0
    assert main(["validate", "--manifest", str((tmp_path / "canonical/training.json")), "--raw-root", str(tmp_path)]) == 0
    output_text = output.out + output.err + capsys.readouterr().out
    metadata = state.read_text(encoding="utf-8") + (tmp_path / "canonical/training.json").read_text(encoding="utf-8")
    # Structured message is now preserved for local diagnostics. Technical output
    # and stored metadata still must not acquire raw records or credentials.
    error = json.loads(next(line[7:] for line in output.out.splitlines() if line.startswith("Error: ")))
    assert error["message"] == " ".join(SECRETS).replace(SECRETS[0], "[секрет скрыт]")
    assert SECRETS[0] not in output_text
    technical_output = "\n".join(line for line in output_text.splitlines() if not line.startswith("Error: "))
    for secret in SECRETS:
        assert secret not in technical_output + metadata


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


@pytest.mark.parametrize("custom", [False, True])
def test_selection_persistence_resume_and_v2_defaults(tmp_path, window, custom):
    from Application.mindbox.selection import DEFAULT_SELECTION
    selection = replace(DEFAULT_SELECTION, view_action_system_names=("CustomView",)) if custom else DEFAULT_SELECTION
    with pytest.raises(api.ChunkedBatchError):
        api.create_chunked_training_batch(Client(4), raw_root=tmp_path, window=window, selection=selection)
    state = state_path(tmp_path)
    document = json.loads(state.read_text(encoding="utf-8"))
    assert document["schema_version"] == 4
    assert document["selection"]["view_action_system_names"] == list(selection.view_action_system_names)
    assert api.load_chunked_training_batch(state, raw_root=tmp_path).selection == selection
    final = api.resume_chunked_training_batch(Client(), state_path=state, raw_root=tmp_path)
    manifest = state.with_name("manifest.json")
    assert final.selection == selection
    assert api.load_chunked_training_batch(manifest, raw_root=tmp_path, require_complete=True).selection == selection
    for path in (state, manifest):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["schema_version"] = 2
        del data["selection"], data["source_kind"], data["merge_source_training_batch_id"]
        path.write_text(json.dumps(data), encoding="utf-8")
    assert api.load_chunked_training_batch(manifest, raw_root=tmp_path, require_complete=True).selection == DEFAULT_SELECTION
    assert api.resume_chunked_training_batch(Client(), state_path=state, raw_root=tmp_path).selection == DEFAULT_SELECTION


def test_prefix_carries_selection(tmp_path):
    from Application.mindbox.selection import DEFAULT_SELECTION
    state = prefix_source(tmp_path, ready_days=1)
    selection = replace(DEFAULT_SELECTION, purchase_line_statuses=("CustomStatus",))
    source = replace(api.load_chunked_training_batch(state, raw_root=tmp_path), selection=selection)
    api._atomic_write(state, source)
    before = state.read_bytes()
    final = api.finalize_chunked_training_batch_prefix(PrefixClient(), state_path=state, raw_root=tmp_path)
    manifest = tmp_path / "training_batches" / final.batch_id / "manifest.json"
    assert api.load_chunked_training_batch(manifest, raw_root=tmp_path, require_complete=True).selection == selection
    assert state.read_bytes() == before


def test_cli_custom_selection_and_unchanged_payload(tmp_path, monkeypatch):
    import Application.mindbox as mindbox
    client = PrefixClient()
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *a: client.config)
    monkeypatch.setattr(mindbox, "MindboxClient", lambda *a: client)
    args = ["export-daily", "--since", "2026-08-01", "--until", "2026-08-02",
            "--merge-since", "2025-01-01 00:00", "--raw-root", str(tmp_path)]
    assert main(args + ["--view-action", "CustomView", "--view-action", "OtherView",
                        "--favorite-action", "CustomFavorite", "--purchase-status", "CustomStatus",
                        "--action-product-namespace", "kanzlerKz", "--order-product-namespace", "offline1C"]) == 0
    selection = api.load_chunked_training_batch(tmp_path / "canonical/training.json", raw_root=tmp_path).selection
    assert selection.view_action_system_names == ("CustomView", "OtherView")
    assert selection.favorite_action_system_names == ("CustomFavorite",)
    assert selection.purchase_line_statuses == ("CustomStatus",)
    assert selection.action_product_namespaces == ("kanzlerKz",)
    assert selection.order_product_namespaces == ("offline1C",)
    assert all(set(payload) == {"sinceDateTimeUtc", "tillDateTimeUtc"} for _, payload in client.calls)
    client.calls.clear()
    assert main(args + ["--view-action", "same", "--favorite-action", "same"]) == 1
    assert client.calls == []


@pytest.mark.parametrize("damage", [None, {}, {"view_action_system_names": []}])
def test_v3_requires_complete_valid_selection(tmp_path, window, damage):
    api.create_chunked_training_batch(Client(), raw_root=tmp_path, window=window)
    state = state_path(tmp_path)
    data = json.loads(state.read_text(encoding="utf-8"))
    data["selection"] = damage
    state.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TrainingBatchError):
        api.load_chunked_training_batch(state, raw_root=tmp_path)
