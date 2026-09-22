from datetime import datetime, timezone
from decimal import Decimal
import json
import socket
from types import SimpleNamespace

import pytest

from Application.mindbox import canonical_storage as store, canonical_jobs as jobs, canonical_customers as customers
from Application.mindbox.raw_reader import EXPORT_ROOTS, iter_export
from Application.mindbox.training_batch import TrainingBatchWindow


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("No real network"))


class Client:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []
        self.config = SimpleNamespace(api_url="synthetic", endpoint_id="test", operations={name: name for name in EXPORT_ROOTS})
        self.records = []

    def start_export(self, operation, payload):
        self.calls.append((operation, payload))
        if len(self.calls) == self.fail:
            raise RuntimeError("synthetic failure")
        return str(len(self.calls))

    def wait_for_export(self, *args, **kwargs):
        return ["synthetic"]

    def download_export(self, name, urls, storage):
        def write(url, path):
            path.write_text(json.dumps({EXPORT_ROOTS[name]: self.records if name == "customers" else []}), encoding="utf-8")
        return storage.save_export(name, urls, write)


def run(root, since="2026-09-01", until="2026-09-04", client=None):
    return jobs.create_job(client or Client(), raw_root=root,
        window=TrainingBatchWindow(utc(since), utc(until), utc("2025-01-01")))


def test_daily_add_refresh_and_no_historical_objects(tmp_path):
    run(tmp_path)
    before = store.catalog(tmp_path)
    run(tmp_path, "2026-09-03", "2026-09-06")
    after = store.catalog(tmp_path)
    assert len(after["actions"]) == len(after["orders"]) == 5
    assert after["actions"]["2026-09-01"] == before["actions"]["2026-09-01"]
    assert after["actions"]["2026-09-03"]["directory"] != before["actions"]["2026-09-03"]["directory"]
    assert len(list((tmp_path / "canonical/objects").iterdir())) == 11
    assert len(store.continuous_range(after)) == 5
    assert store.current_batch(tmp_path).window.interaction_until == utc("2026-09-06")


def test_failure_and_resume_preserve_published_days(tmp_path):
    run(tmp_path, until="2026-09-02")
    old = store.catalog(tmp_path)
    paths = []
    with pytest.raises(RuntimeError):
        jobs.create_job(Client(fail=5), raw_root=tmp_path,
            window=TrainingBatchWindow(utc("2026-09-01"), utc("2026-09-04"), utc("2025-01-01")),
            on_state_created=paths.append)
    data = store.catalog(tmp_path)
    assert data["orders"]["2026-09-01"] != old["orders"]["2026-09-01"]
    assert "2026-09-02" in data["actions"] and "2026-09-02" not in data["orders"]
    client = Client()
    jobs.resume_job(client, raw_root=tmp_path, state_path=paths[0])
    assert [name for name, _ in client.calls] == ["orders", "actions", "orders"]


def test_gap_chooses_longest_then_latest(tmp_path):
    run(tmp_path, until="2026-09-03")
    run(tmp_path, "2026-09-05", "2026-09-07")
    assert [pair["actions"]["since"][:10] for pair in store.continuous_range(store.catalog(tmp_path))] == ["2026-09-05", "2026-09-06"]
    run(tmp_path, "2026-09-03", "2026-09-05")
    assert len(store.continuous_range(store.catalog(tmp_path))) == 6


def test_merges_never_shrink_and_failed_replacement_keeps_old(tmp_path):
    run(tmp_path)
    previous = store.catalog(tmp_path)["customer_merges"]
    with pytest.raises(RuntimeError):
        run(tmp_path, "2026-09-01", "2026-09-02", Client(fail=1))
    assert store.catalog(tmp_path)["customer_merges"] == previous
    client = Client()
    run(tmp_path, "2026-09-01", "2026-09-02", client)
    assert client.calls[0][1]["tillDateTimeUtc"] == "2026-09-04 00:00"


def test_invalid_raw_keeps_old_day(tmp_path):
    run(tmp_path)
    before = store.catalog(tmp_path)
    path = tmp_path / "staged"
    path.mkdir()
    (path / "actions_part_001.json").write_text('{"customerActions": [}', encoding="utf-8")
    with store.storage_lock(tmp_path), pytest.raises(Exception):
        store.publish(tmp_path, "actions", utc("2026-09-01"), utc("2026-09-02"), path)
    assert store.catalog(tmp_path) == before


def test_month_windows():
    result = jobs.monthly_windows(utc("2025-01-15"), utc("2025-05-20"))
    assert len(result) == 5
    assert result[0] == (utc("2025-01-15"), utc("2025-02-01"))
    assert result[-1] == (utc("2025-05-01"), utc("2025-05-20"))
    assert len(jobs.monthly_windows(utc("2025-01-01"), utc("2025-06-01"))) == 5


def customer_record(key, changed="2026-01-02", email="old@example.test"):
    return {"ids": {"mindboxId": key}, "changeDateTimeUtc": changed + "T00:00:00Z", "email": email}


def test_month_upsert_timestamp_precedence_count_and_partial_success(tmp_path):
    client = Client(fail=3)
    client.records = [customer_record(1), customer_record("1"), customer_record(2)]
    with pytest.raises(RuntimeError):
        jobs.create_job(client, raw_root=tmp_path, customers=True, since=utc("2026-01-01"), until=utc("2026-04-01"))
    assert customers.customer_summary(tmp_path)["count"] == 2
    assert customers.customer_summary(tmp_path)["intervals"] == [[utc("2026-01-01").isoformat(), utc("2026-03-01").isoformat()]]
    client = Client()
    client.records = [customer_record(1, "2026-03-03", "new@example.test"), customer_record(3)]
    jobs.create_job(client, raw_root=tmp_path, customers=True, since=utc("2026-03-01"), until=utc("2026-04-01"))
    assert customers.customer_summary(tmp_path)["count"] == 3
    client.records = [customer_record(1)]
    jobs.create_job(client, raw_root=tmp_path, customers=True, since=utc("2026-01-01"), until=utc("2026-02-01"))
    with customers.connect(customers.database(tmp_path)) as connection:
        assert "new@example.test" in connection.execute("SELECT raw FROM profiles WHERE id='1'").fetchone()[0]


def test_failed_month_rolls_back_data_metadata_and_receipt(tmp_path):
    client = Client()
    client.records = [customer_record(1)]
    jobs.create_job(client, raw_root=tmp_path, customers=True, since=utc("2026-01-01"), until=utc("2026-02-01"))
    previous = customers.customer_summary(tmp_path)
    directory = tmp_path / "invalid"
    directory.mkdir()
    (directory / "customers_part_001.json").write_text(json.dumps({"customers": [customer_record(2), {"ids": {}}]}))
    with pytest.raises(Exception):
        customers.apply_month(tmp_path, directory, utc("2026-02-01"), utc("2026-03-01"), job="failed", index=0)
    assert customers.customer_summary(tmp_path) == previous
    assert not customers.month_committed(tmp_path, "failed", 0)


def test_manual_full_replaces_same_customer_store(tmp_path):
    run(tmp_path)
    path = tmp_path / "manual.json"
    for keys in ([1, 1, 2], [3]):
        path.write_text(json.dumps({"customers": [customer_record(key) for key in keys]}))
        customers.import_full(tmp_path, path)
        assert customers.customer_summary(tmp_path)["count"] == len(set(keys))
    assert len(list((tmp_path / "canonical").glob("*.sqlite"))) == 1
    assert list((tmp_path / "canonical").glob(".customers-*")) == []


def test_decimal_roundtrip():
    raw = {"value": Decimal("12345678901234567890.123456789")}
    assert json.loads(customers.encode(raw), parse_float=Decimal) == raw


def test_commit_before_checkpoint_resumes_without_redownload(tmp_path, monkeypatch):
    original = jobs.atomic_json
    states = []
    def fail_checkpoint(path, data):
        if Path(path).name == "state.json" and data["components"][0]["done"]:
            raise OSError("checkpoint interruption")
        original(path, data)
    from pathlib import Path
    monkeypatch.setattr(jobs, "atomic_json", fail_checkpoint)
    with pytest.raises(OSError):
        jobs.create_job(Client(), raw_root=tmp_path, window=TrainingBatchWindow(
            utc("2026-09-01"), utc("2026-09-02"), utc("2025-01-01")), on_state_created=states.append)
    monkeypatch.setattr(jobs, "atomic_json", original)
    client = Client()
    jobs.resume_job(client, raw_root=tmp_path, state_path=states[0])
    assert [name for name, _ in client.calls] == ["actions", "orders"]


def test_manual_failure_and_cancel_leave_current_database(tmp_path):
    run(tmp_path)
    path = tmp_path / "manual.json"
    path.write_text(json.dumps({"customers": [customer_record(1)]}))
    customers.import_full(tmp_path, path)
    before = customers.database(tmp_path).read_bytes()
    path.write_text(json.dumps({"customers": [customer_record(2), {"ids": {}}]}))
    with pytest.raises(Exception):
        customers.import_full(tmp_path, path)
    assert customers.database(tmp_path).read_bytes() == before
    with pytest.raises(InterruptedError):
        customers.import_full(tmp_path, path, cancelled=lambda: True)
    assert customers.database(tmp_path).read_bytes() == before


def test_abandoned_manual_database_cleanup_is_confined(tmp_path):
    run(tmp_path)
    abandoned = tmp_path / "canonical/.customers-interrupted.sqlite"
    abandoned.write_bytes(b"partial database")
    unrelated = tmp_path / "keep.sqlite"
    unrelated.write_bytes(b"unrelated")
    source = tmp_path / "manual.json"
    source.write_text('{"customers": []}')
    customers.import_full(tmp_path, source)
    assert not abandoned.exists()
    assert unrelated.read_bytes() == b"unrelated"


def test_summary_does_not_read_raw(tmp_path, monkeypatch):
    run(tmp_path)
    client = Client()
    client.records = [customer_record(1)]
    jobs.create_job(client, raw_root=tmp_path, customers=True, since=utc("2026-01-01"), until=utc("2026-02-01"))
    monkeypatch.setattr(store, "iter_export", lambda *a, **k: pytest.fail("No raw scan for summary"))
    monkeypatch.setattr(customers, "iter_export", lambda *a, **k: pytest.fail("No raw scan for summary"))
    assert store.summary(tmp_path)["orders"] is not None
    assert customers.customer_summary(tmp_path)["count"] == 1


def test_customers_resume_only_uncommitted_months(tmp_path):
    states = []
    with pytest.raises(RuntimeError):
        jobs.create_job(Client(fail=2), raw_root=tmp_path, customers=True,
            since=utc("2026-01-15"), until=utc("2026-04-15"), on_state_created=states.append)
    client = Client()
    jobs.resume_job(client, raw_root=tmp_path, state_path=states[0])
    assert len(client.calls) == 3
    assert client.calls[0][1]["sinceDateTimeUtc"] == "2026-02-01 00:00"
    assert client.calls[-1][1]["tillDateTimeUtc"] == "2026-04-15 00:00"


def test_new_endpoint_cannot_mix_with_existing_data(tmp_path):
    run(tmp_path)
    client = Client()
    client.config.endpoint_id = "different"
    with pytest.raises(ValueError):
        run(tmp_path, client=client)
    assert not client.calls


def test_canonical_contacts_use_existing_projection(tmp_path):
    from Application.mindbox.customer_profile_snapshot import load_customer_contact_index
    client = Client()
    client.records = [customer_record(1)]
    path = jobs.create_job(client, raw_root=tmp_path, customers=True, since=utc("2026-01-01"), until=utc("2026-02-01"))
    index = load_customer_contact_index(path, {"idx2user": ["1", "missing"]}, raw_root=tmp_path)
    assert index.diagnostics.matched_profiles == 1
    assert index.diagnostics.missing_profiles == 1


def test_customer_update_consumes_records_incrementally(tmp_path, monkeypatch):
    processed = 0
    original = customers._upsert
    def upsert(*args):
        nonlocal processed
        original(*args)
        processed += 1
    def records(*args, **kwargs):
        for index in range(1000):
            assert processed == index  # A list/full read before upsert would fail.
            yield customer_record(index % 10)
    monkeypatch.setattr(customers, "_upsert", upsert)
    monkeypatch.setattr(customers, "iter_export", records)
    customers.apply_month(tmp_path, tmp_path, utc("2026-01-01"), utc("2026-02-01"), job="stream", index=0)
    assert processed == 1000
    assert customers.customer_summary(tmp_path)["count"] == 10


def test_process_death_recovers_uncommitted_customer_transaction(tmp_path):
    import subprocess
    import sys
    client = Client()
    client.records = [customer_record(1)]
    jobs.create_job(client, raw_root=tmp_path, customers=True, since=utc("2026-01-01"), until=utc("2026-02-01"))
    previous = customers.customer_summary(tmp_path)
    script = """import os, sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute('PRAGMA cache_size=1')
c.execute('BEGIN IMMEDIATE')
c.execute('INSERT INTO profiles VALUES (?, ?, ?)', ('uncommitted', '', 'x' * 1000000))
os._exit(77)
"""
    result = subprocess.run([sys.executable, "-c", script, str(customers.database(tmp_path))], check=False)
    assert result.returncode == 77
    assert customers.customer_summary(tmp_path) == previous
    with customers.connect(customers.database(tmp_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 1


def test_current_manifest_uses_shared_preparation(tmp_path, monkeypatch):
    from Application.mindbox.daily_training_batch import load_chunked_training_batch, prepare_training_data_from_chunked_batch
    from Application.model import mindbox_training_preparation as preparation
    path = run(tmp_path)
    batch = load_chunked_training_batch(path, raw_root=tmp_path, require_complete=True)
    received = {}
    monkeypatch.setattr(preparation, "_prepare_training_data_from_mindbox_sources", lambda **kwargs: received.update(kwargs))
    prepare_training_data_from_chunked_batch(batch, raw_root=tmp_path, catalog_path="unused", train_config=None)
    assert len(received["actions_export_dirs"]) == 3
    assert received["selection"] == batch.selection
    assert list(iter_export("orders", input_dir=received["orders_export_dirs"][0])) == []


def manual_pair(root, since="2026-01-01", until="2026-07-01"):
    from Application.mindbox.manual_import import import_interactions
    paths = {}
    for name in ("actions", "orders"):
        paths[name] = root / f"{name}.json"
        paths[name].write_text(json.dumps({EXPORT_ROOTS[name]: []}))
    return import_interactions(**paths, raw_root=root,
                               window=TrainingBatchWindow(utc(since), utc(until), utc(since)))


def test_manual_overlap_extension_summary_and_single_preparation_source(tmp_path, monkeypatch):
    from Application.mindbox.daily_training_batch import load_chunked_training_batch, prepare_training_data_from_chunked_batch
    from Application.model import mindbox_training_preparation as preparation
    run(tmp_path, "2026-06-20", "2026-07-04")
    old = store.catalog(tmp_path)
    manifest = manual_pair(tmp_path)
    data = store.catalog(tmp_path)
    assert data["actions"] == old["actions"] and data["orders"] == old["orders"]
    assert all((tmp_path / entry["directory"]).is_dir() for name in ("actions", "orders") for entry in old[name].values())
    batch = load_chunked_training_batch(manifest, raw_root=tmp_path, require_complete=True)
    assert batch.window.interaction_since == utc("2026-01-01")
    assert batch.window.interaction_until == utc("2026-07-04")
    assert len(batch.components) == 9  # Merges + one long pair + three daily pairs.
    assert batch.diagnostics["days_total"] == 184
    received = {}
    monkeypatch.setattr(preparation, "_prepare_training_data_from_mindbox_sources", lambda **kwargs: received.update(kwargs))
    prepare_training_data_from_chunked_batch(batch, raw_root=tmp_path, catalog_path="unused", train_config=None)
    for name in ("actions", "orders"):
        expected = (tmp_path / data["manual_interactions"][name]["directory"],
                    *(tmp_path / data[name][f"2026-07-0{day}"]["directory"] for day in (1, 2, 3)))
        assert received[f"{name}_export_dirs"] == expected
        assert len(set(received[f"{name}_export_dirs"])) == 4
        assert store.summary(tmp_path)[name] == {"since": utc("2026-01-01").isoformat(),
            "until": utc("2026-07-04").isoformat(), "updated": data["manual_interactions"]["updated"]}
    # A later API refresh retains manual priority, raw and global selection policy.
    from dataclasses import replace
    from Application.mindbox.selection import DEFAULT_SELECTION
    selection = replace(DEFAULT_SELECTION, view_action_system_names=("OtherView",))
    jobs.create_job(Client(), raw_root=tmp_path,
        window=TrainingBatchWindow(utc("2026-06-30"), utc("2026-07-02"), utc("2025-01-01")), selection=selection)
    assert store.catalog(tmp_path)["manual_interactions"] == data["manual_interactions"]
    assert store.current_batch(tmp_path).selection == selection
    assert len(store.current_batch(tmp_path).components) == 9


@pytest.mark.parametrize("manual_since,manual_until,api_since,api_until,expected_since,expected_until", [
    ("2026-01-01", "2026-07-01", "2026-07-03", "2026-07-05", "2026-01-01", "2026-07-01"),
    ("2026-06-01", "2026-06-03", "2026-07-01", "2026-07-04", "2026-07-01", "2026-07-04"),
    ("2026-06-01", "2026-06-04", "2026-07-01", "2026-07-04", "2026-07-01", "2026-07-04"),
    ("2026-07-01", "2026-07-04", "2026-06-01", "2026-06-04", "2026-07-01", "2026-07-04"),
    ("2026-07-01", "2026-07-04", "2026-06-29", "2026-07-01", "2026-06-29", "2026-07-04"),
])
def test_manual_and_api_contiguous_range_rules(tmp_path, manual_since, manual_until, api_since, api_until, expected_since, expected_until):
    # Get sufficiently broad merges without creating unrelated interaction days.
    directory = tmp_path / "merges"
    directory.mkdir()
    (directory / "customer_merges_part_001.json").write_text('{"customerMerges": []}')
    with store.storage_lock(tmp_path):
        store.publish(tmp_path, "customer_merges", utc("2025-01-01"), utc("2027-01-01"), directory)
    run(tmp_path, api_since, api_until)
    manual_pair(tmp_path, manual_since, manual_until)
    batch = store.current_batch(tmp_path)
    assert (batch.window.interaction_since, batch.window.interaction_until) == (utc(expected_since), utc(expected_until))


def test_manual_component_requires_full_merge_coverage_even_after_metadata_change(tmp_path):
    run(tmp_path, "2026-07-01", "2026-07-04")
    manual_pair(tmp_path)
    data = store.catalog(tmp_path)
    data["customer_merges"]["since"] = utc("2026-06-01").isoformat()
    pairs = store.continuous_range(data)
    assert len(pairs) == 3
    assert all(pair["actions"]["source_kind"] == "API" for pair in pairs)


def test_v1_api_mutation_upgrades_without_losing_partitions(tmp_path):
    run(tmp_path)
    data = store.catalog(tmp_path)
    data.pop("manual_interactions")
    data["schema_version"] = 1
    store.atomic_json(tmp_path / "canonical/catalog.json", data)
    run(tmp_path, "2026-09-04", "2026-09-05")
    current = store.catalog(tmp_path)
    assert current["schema_version"] == 2 and current["manual_interactions"] is None
    assert current["orders"]["2026-09-01"] == data["orders"]["2026-09-01"]
