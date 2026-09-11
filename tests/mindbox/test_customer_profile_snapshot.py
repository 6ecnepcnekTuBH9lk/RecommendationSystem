from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket

import pytest

from Application.mindbox import customer_profile_snapshot as api
from Application.mindbox import daily_training_batch as daily
from Application.mindbox.training_batch import TrainingBatchExport, TrainingBatchWindow
from scripts.mindbox_customer_profiles import main


SECRETS = ("SECRET_CUSTOMER", "secret@example.test", "+79161234567", "SECRET_CARD", "SECRET_NAME", "SecretKey", "https://signed.invalid")


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("No live network"))
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    until = since + timedelta(days=1)
    components = []
    for name in ("customer_merges", "actions", "orders"):
        directory = tmp_path / name / "20260101_000000"
        directory.mkdir(parents=True)
        payload = {"customerMerges": []} if name == "customer_merges" else {"must_not_read": SECRETS}
        (directory / f"{name}_part_001.json").write_text(json.dumps(payload), encoding="utf-8")
        entry = TrainingBatchExport(name, "SECRET_EXPORT", name, directory.relative_to(tmp_path).as_posix(), 1)
        components.append(daily.BatchComponent(name, since, until, name, "READY", entry))
    batch = daily.ChunkedTrainingBatch("b" * 32, since, TrainingBatchWindow(since, until, since), "0" * 64, tuple(components), True)
    manifest = tmp_path / "training_batches" / batch.batch_id / "manifest.json"
    manifest.parent.mkdir(parents=True)
    daily._atomic_write(manifest, batch)

    class Client:
        calls = []
        def export(self, name, **kwargs):
            self.calls.append(name)
            def write(url, path):
                path.write_text(json.dumps({"customers": [{"ids": {"mindboxId": SECRETS[0]}, "email": SECRETS[1],
                    "mobilePhone": SECRETS[2], "firstName": SECRETS[4], "lastActivatedCard": {"ids": {"number": SECRETS[3]}}}]}), encoding="utf-8")
            return kwargs["storage"].save_export(name, [SECRETS[-1]], write)
    return tmp_path, manifest, Client()


def create(source):
    root, manifest, client = source
    snapshot = api.create_customer_profile_snapshot(client, training_manifest=manifest, raw_root=root)
    return snapshot, api.snapshot_manifest_path(root, snapshot.snapshot_id)


def test_snapshot_export_validate_inspect_metadata_only(source, monkeypatch, capsys):
    root, _, client = source
    snapshot, manifest = create(source)
    assert client.calls == ["customers"]
    text = manifest.read_text(encoding="utf-8")
    assert all(secret not in text for secret in SECRETS)
    original = Path.open
    def safe_open(path, *args, **kwargs):
        assert "_part_" not in path.name  # Validation may read manifests only.
        return original(path, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path, "open", safe_open)
        assert main(["validate", "--manifest", str(manifest), "--raw-root", str(root)]) == 0
    reads = []
    iterator = api.iter_export
    def read(name, **kwargs):
        reads.append(name)
        return iterator(name, **kwargs)
    monkeypatch.setattr(api, "iter_export", read)
    index = api.load_customer_contact_index(manifest, {"idx2user": [SECRETS[0], "missing"]}, raw_root=root)
    assert reads == ["customer_merges", "customers"]
    assert index.diagnostics.matched_profiles == 1 and index.diagnostics.missing_profiles == 1
    assert index.contacts[0].discount_card == SECRETS[3]
    assert main(["inspect", "--manifest", str(manifest), "--raw-root", str(root)]) == 0
    output = capsys.readouterr()
    assert all(secret not in output.out + output.err + repr(index) for secret in SECRETS)
    assert "with_phone: 1" in output.out
    assert list(manifest.parent.iterdir()) == [manifest]
    assert api.load_customer_profile_snapshot(manifest, raw_root=root) == snapshot


@pytest.mark.parametrize("field,value", [("customers_directory", "../customers/20260101_000000"),
    ("customers_directory", "customers/../../outside"), ("customers_directory", "C:/secret"),
    ("customers_parts", 2), ("customer_merges_parts", 0), ("originating_training_batch_id", "c" * 32),
    ("transport_complete", False), ("schema_version", 2), ("created_at", "invalid-secret")])
def test_corrupt_snapshot_rejected_safely(source, field, value, capsys):
    root, _, _ = source
    snapshot, manifest = create(source)
    content = asdict(snapshot)
    content[field] = value
    manifest.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(api.ProfileSnapshotError):
        api.load_customer_profile_snapshot(manifest, raw_root=root)
    assert main(["validate", "--manifest", str(manifest), "--raw-root", str(root)]) == 1
    output = capsys.readouterr()
    assert value not in output.err if isinstance(value, str) else True


@pytest.mark.parametrize("fail", [False, True])
def test_atomic_snapshot_publication(source, monkeypatch, fail):
    root, _, _ = source
    snapshot, _ = create(source)
    next_snapshot = replace(snapshot, snapshot_id="d" * 32)
    target = api.snapshot_manifest_path(root, next_snapshot.snapshot_id)
    calls = []
    sync, publish = api.os.fsync, api.os.replace
    def fsync(fd):
        calls.append("fsync")
        sync(fd)
    def replace_file(source, destination):
        assert calls == ["fsync"]
        assert not target.exists()
        assert json.loads(Path(source).read_text(encoding="utf-8"))["snapshot_id"] == next_snapshot.snapshot_id
        if fail:
            raise OSError("synthetic error")
        publish(source, destination)
    monkeypatch.setattr(api.os, "fsync", fsync)
    monkeypatch.setattr(api.os, "replace", replace_file)
    if fail:
        with pytest.raises(OSError):
            api._publish_snapshot(next_snapshot, root)
        assert not target.exists()
    else:
        assert api._publish_snapshot(next_snapshot, root) == target
    assert not target.with_name("manifest.tmp").exists()


def test_origin_merge_mismatch_and_invalid_raw_no_pii(source, capsys):
    root, _, _ = source
    snapshot, manifest = create(source)
    other = root / "customer_merges" / "20260102_000000"
    other.mkdir()
    (other / "customer_merges_part_001.json").write_text('{"customerMerges": []}')
    altered = replace(snapshot, customer_merges_directory="customer_merges/20260102_000000")
    with pytest.raises(api.ProfileSnapshotError):
        api.validate_customer_profile_snapshot(altered, raw_root=root)
    raw = next((root / snapshot.customers_directory).glob("*.json"))
    raw.write_text(json.dumps({"customers": [{"ids": {"mindboxId": SECRETS[0]}, "mobilePhone": [SECRETS[2]]}]}))
    assert main(["inspect", "--manifest", str(manifest), "--raw-root", str(root)]) == 1
    output = capsys.readouterr()
    assert all(secret not in output.out + output.err for secret in SECRETS)


def test_export_failure_no_manifest(source):
    root, training, client = source
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic transport failure")
    client.export = fail
    with pytest.raises(RuntimeError):
        api.create_customer_profile_snapshot(client, training_manifest=training, raw_root=root)
    assert not list(root.glob("customer_profile_snapshots/*/manifest.json"))
