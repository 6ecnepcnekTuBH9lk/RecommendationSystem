from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from Application.mindbox import canonical_storage as storage, store_catalog as stores, manual_import
from Application.mindbox.training_batch import TrainingBatchWindow


def utc(day):
    return datetime(2026, 1, day, tzinfo=timezone.utc)


def raw(key="A", name="Магазин A"):
    return {"ids": {"mindboxId": "PRIVATE_ORDER"}, "customer": {"ids": {"mindboxId": "PRIVATE_CUSTOMER"}},
            "firstAction": {"dateTimeUtc": utc(1).isoformat(), "channel": {"ids": {"externalId": key}, "name": name}},
            "lines": []}


def publish(root, name, records, start=1, end=2):
    directory = root / "staging" / name
    directory.mkdir(parents=True)
    envelope = {"orders": "orders", "actions": "customerActions", "customer_merges": "customerMerges"}[name]
    (directory / f"{name}_part_001.json").write_text(json.dumps({envelope: records}), encoding="utf-8")
    with storage.storage_lock(root):
        return storage.publish(root, name, utc(start), utc(end), directory)


@pytest.fixture
def root(tmp_path):
    publish(tmp_path, "customer_merges", [], 1, 10)
    return tmp_path


def payload(root):
    return json.loads((root / "canonical/store_catalog.json").read_text(encoding="utf-8"))


def no_scan(*args, **kwargs):
    pytest.fail("Unexpected raw Orders scan")


def test_collector_frequency_ties_blank_identity_and_bounded_state():
    records = [raw(None), raw("A", " Z "), raw("A", "A"), raw("B", "Most"), raw("B", "Most"),
               raw("B", "Rare"), raw("C", " "), raw("D", None)]
    expected = {"A": {"A": 1, "Z": 1}, "B": {"Most": 2, "Rare": 1},
                "C": {"Не указано": 1}, "D": {"Не указано": 1}}
    for rows in (records, reversed(records)):
        collector = stores.StoreSourceCollector()
        for record in rows:
            collector.add(record)
        assert collector.names == expected
        assert set(vars(collector)) == {"names"}
        encoded = json.dumps(collector.summary({"since": utc(1).isoformat(), "until": utc(2).isoformat(), "parts": 1}))
        assert "PRIVATE_ORDER" not in encoded and "PRIVATE_CUSTOMER" not in encoded
        assert min(collector.names["A"], key=lambda n: (-collector.names["A"][n], n)) == "A"


def test_api_one_validation_pass_and_day_replacement(root, monkeypatch):
    calls = []
    original = storage.iter_export
    def observed(name, **kwargs):
        calls.append(name)
        yield from original(name, **kwargs)
    monkeypatch.setattr(storage, "iter_export", observed)
    monkeypatch.setattr(stores, "iter_export", no_scan)
    first = publish(root, "orders", [raw("X", "X")])
    assert calls == ["orders"]
    assert stores.ensure_store_catalog(root) == (("X", "X"),)
    second = publish(root, "orders", [raw("Y", "Y")])
    assert calls == ["orders", "orders"]
    assert stores.ensure_store_catalog(root) == (("Y", "Y"),)
    assert set(payload(root)["sources"]) == {second["directory"]}
    assert first["directory"] not in payload(root)["sources"]


def test_actions_and_customer_revision_do_not_invalidate(root, monkeypatch):
    publish(root, "orders", [raw()])
    before = (root / "canonical/store_catalog.json").read_bytes()
    signature = stores.orders_signature(storage.catalog(root))
    publish(root, "actions", [])
    data = storage.catalog(root)
    data["revision"] = "customer-only-change"
    data["customer_merges"]["updated"] = utc(9).isoformat()
    assert stores.orders_signature(data) == signature
    monkeypatch.setattr(stores, "iter_export", no_scan)
    assert stores.ensure_store_catalog(root) == (("A", "Магазин A"),)
    assert (root / "canonical/store_catalog.json").read_bytes() == before


def test_publish_on_old_installation_keeps_incoming_summary_without_rescan(root, monkeypatch):
    old = publish(root, "orders", [raw("OLD", "Old")])
    (root / "canonical/store_catalog.json").unlink()
    original = stores.iter_export
    monkeypatch.setattr(stores, "iter_export", no_scan)
    new = publish(root, "orders", [raw("NEW", "New")], 2, 3)
    assert not payload(root)["complete"]
    assert set(payload(root)["sources"]) == {new["directory"]}
    calls = []
    def observed(name, **kwargs):
        calls.append(Path(kwargs["input_dir"]))
        yield from original(name, **kwargs)
    monkeypatch.setattr(stores, "iter_export", observed)
    assert stores.ensure_store_catalog(root) == (("NEW", "New"), ("OLD", "Old"))
    assert calls == [root / old["directory"]]
    assert payload(root)["complete"]


def test_migration_once_and_corrupt_catalog_rebuild(root, monkeypatch):
    publish(root, "orders", [raw()])
    path = root / "canonical/store_catalog.json"
    path.unlink()
    calls = []
    original = stores.iter_export
    def observed(name, **kwargs):
        calls.append(kwargs["input_dir"])
        yield from original(name, **kwargs)
    monkeypatch.setattr(stores, "iter_export", observed)
    assert stores.ensure_store_catalog(root) == (("A", "Магазин A"),)
    assert len(calls) == 1
    monkeypatch.setattr(stores, "iter_export", no_scan)
    assert stores.ensure_store_catalog(root) == (("A", "Магазин A"),)
    monkeypatch.setattr(stores, "iter_export", observed)
    path.write_text("broken", encoding="utf-8")
    assert stores.ensure_store_catalog(root) == (("A", "Магазин A"),)
    assert len(calls) == 2


def test_failed_derived_publication_recovers_only_new_source(root, monkeypatch, caplog):
    old = publish(root, "orders", [raw("A", "A")])
    atomic = storage.atomic_json
    def failure(path, value):
        if Path(path).name == "store_catalog.json":
            raise OSError("synthetic")
        atomic(path, value)
    monkeypatch.setattr(storage, "atomic_json", failure)
    new = publish(root, "orders", [raw("B", "B")], 2, 3)
    assert new["directory"] in {entry["directory"] for entry in storage.catalog(root)["orders"].values()}
    assert caplog.records
    monkeypatch.setattr(storage, "atomic_json", atomic)
    scanned = []
    original = stores.iter_export
    def observed(name, **kwargs):
        key = kwargs["input_dir"].relative_to(root).as_posix()
        assert key != old["directory"]
        scanned.append(key)
        yield from original(name, **kwargs)
    monkeypatch.setattr(stores, "iter_export", observed)
    assert stores.ensure_store_catalog(root) == (("A", "A"), ("B", "B"))
    assert scanned == [new["directory"]]


def manual_pair(root, start, end, key):
    actions, orders = root / "manual-actions.json", root / "manual-orders.json"
    actions.write_text('{"customerActions":[]}', encoding="utf-8")
    orders.write_text(json.dumps({"orders": [raw(key, key)]}), encoding="utf-8")
    return manual_import.import_interactions(actions, orders, raw_root=root,
        window=TrainingBatchWindow(utc(start), utc(end), utc(1)))


def test_manual_shared_collector_one_pass_precedence_and_migration(root, monkeypatch):
    api = publish(root, "orders", [raw("API", "API")])
    calls = []
    original = storage.iter_export
    def observed(name, **kwargs):
        calls.append(name)
        yield from original(name, **kwargs)
    monkeypatch.setattr(storage, "iter_export", observed)
    monkeypatch.setattr(stores, "iter_export", no_scan)
    manual_pair(root, 1, 3, "MANUAL")
    assert calls == ["actions", "orders"]
    assert stores.ensure_store_catalog(root) == (("MANUAL", "MANUAL"),)
    assert api["directory"] in payload(root)["sources"]
    manual_pair(root, 3, 4, "LATER")
    assert stores.ensure_store_catalog(root) == (("API", "API"), ("LATER", "LATER"))
    assert len(payload(root)["sources"]) == 2
    # Migration indexes both currently effective and shadowed daily sources.
    manual_pair(root, 1, 3, "MANUAL")
    (root / "canonical/store_catalog.json").unlink()
    monkeypatch.setattr(stores, "iter_export", original)
    assert stores.ensure_store_catalog(root) == (("MANUAL", "MANUAL"),)
    assert api["directory"] in payload(root)["sources"]


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "1"])
def test_invalid_counts_rejected(root, bad):
    entry = publish(root, "orders", [raw()])
    value = payload(root)
    value["sources"][entry["directory"]]["names"]["A"]["Магазин A"] = bad
    with pytest.raises(ValueError):
        stores.validate_catalog(value)


def test_strict_schema_sorted_unique_aggregate_and_atomic_determinism(root, monkeypatch):
    events = []
    fsync, replace = storage.os.fsync, storage.os.replace
    def sync(fd):
        events.append("sync")
        fsync(fd)
    def replaced(src, dst):
        if Path(dst).name == "store_catalog.json":
            assert events[-1] == "sync"
            assert stores.validate_catalog(json.loads(Path(src).read_text(encoding="utf-8")))
        replace(src, dst)
    monkeypatch.setattr(storage.os, "fsync", sync)
    monkeypatch.setattr(storage.os, "replace", replaced)
    publish(root, "orders", [raw("Z", "Z"), raw("A", "A")])
    original = payload(root)
    assert original["schema_version"] == 1 and original["complete"]
    for modify in (lambda v: v.update(schema_version=True), lambda v: v.update(extra=1),
                   lambda v: v["stores"].append(v["stores"][0]), lambda v: v["stores"].reverse()):
        broken = deepcopy(original)
        modify(broken)
        with pytest.raises(ValueError):
            stores.validate_catalog(broken)
    before = (root / "canonical/store_catalog.json").read_bytes()
    stores._write(root, storage.catalog(root), dict(reversed(list(original["sources"].items()))))
    assert (root / "canonical/store_catalog.json").read_bytes() == before
