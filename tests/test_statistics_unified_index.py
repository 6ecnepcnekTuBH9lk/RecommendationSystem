"""Synthetic canonical Orders and SQLite only."""

from collections import defaultdict
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from datetime import date
import json
import os
import shutil
import sqlite3

import pytest

from Application import statistics_index as legacy
from Application import statistics_unified_index as index
from Application.analysis_filter import AnalysisFilter
from Application.mindbox.canonical_customers import encode
from Application.statistics_slices import capture_sources
from Application.statistics_storage_safety import StatisticsIndexError, path_identity
from Application.statistics_unified_facts import reconcile_order
from Application.statistics_unified_policy import EvidenceQuality as Quality
from tests.test_dataset_statistics import STAMP, dataset as dataset_fixture, order, put_entry, save


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root, catalog, data = dataset_fixture.__wrapped__(tmp_path)
    target = tmp_path / "index"
    target.mkdir(mode=0o700)
    for module in (legacy, index):
        monkeypatch.setattr(module, "protect_directory", lambda path: path.chmod(0o700))
        monkeypatch.setattr(module, "validate_storage_root", lambda path, **kw: path_identity(path))
    return root, catalog, data, target


def _paths(dataset):
    root, catalog, _, target = dataset
    return dict(raw_root=root, catalog_path=catalog, index_dir=target)


def _build(dataset, **kwargs):
    return index.build_unified_index(**_paths(dataset), expected_bytes=1024**2, **kwargs)


def _source(dataset, *parts):
    root, _, data, _ = dataset
    entry = next(iter(data["orders"].values()))
    directory = root / entry["directory"]
    for ordinal, rows in enumerate(parts, 1):
        (directory / f"orders_part_{ordinal:03d}.json").write_text(encode({"orders": rows}), encoding="utf-8")
    entry["parts"] = len(parts)
    save(root, data)
    return directory


def _matrix():
    cases = {
        "f": ["F"], "sold": ["sold"], "cp": ["CP"], "delivering": ["delivering"],
        "f-cp": ["F", "CP"], "f-cd": ["F", "CD"], "f-return": ["F", "Return"],
        "cp-return": ["CP", "Return"], "only-return": ["Return"], "unknown": ["unknown-code"],
        "zero-qty": ["F"], "negative-qty": ["F"], "fractional-qty": ["F"], "missing-qty": ["F"],
        "missing-price": ["F"], "zero-price": ["F"], "kzt": ["F"], "duplicate": ["F", "CP"],
        "transition": ["F"], "owner": ["F"], "date": ["F"], "missing-line": ["F"],
        "bad-status": ["F"], "bad-price": ["F"], "no-id": ["F"], "partial-shape": ["F"],
        "empty": [], "missing-shape": [], "bad-qty": ["F"], "unknown-currency": ["F"],
    }
    raw = {name: order(statuses, "private-order-" + name) for name, statuses in cases.items()}
    for row in raw.values():
        for line in row["lines"]:
            line["id"] = "private-line-" + line["id"]
        row["deliveryCost"] = Decimal("0.000000000000000001")
    raw["f"]["lines"][0]["priceOfLine"] = Decimal("1234567890.123456789012345678901234")
    raw["f"]["lines"][0]["number"] = 913
    raw["cp"]["lines"][0]["priceOfLine"] = Decimal("4.321")
    for name, quantity in (("zero-qty", 0), ("negative-qty", -2), ("fractional-qty", Decimal("0.1234567890123456789")),
                           ("bad-qty", True)):
        raw[name]["lines"][0]["quantity"] = quantity
    del raw["missing-qty"]["lines"][0]["quantity"]
    del raw["missing-price"]["lines"][0]["priceOfLine"]
    raw["zero-price"]["lines"][0]["priceOfLine"] = 0
    raw["bad-price"]["lines"][0]["priceOfLine"] = "not-a-price"
    raw["kzt"]["lines"][0]["product"]["ids"] = {"kanzlerKz": "000001-size"}
    raw["unknown-currency"]["lines"][0]["product"]["ids"] = {"external": "unmapped"}
    raw["bad-status"]["lines"][0]["status"] = True
    raw["no-id"]["ids"] = {}
    raw["partial-shape"]["lines"].append(False)
    del raw["missing-shape"]["lines"]
    del raw["missing-line"]["lines"][0]["id"]
    changed = []
    for name in ("duplicate", "transition", "owner", "date", "missing-line"):
        item = deepcopy(raw[name])
        if name == "duplicate":
            changed.append(deepcopy(item))
            item["unneededField"] = "other"
            item["lines"].reverse()
        if name == "transition":
            item["lines"][0]["status"]["ids"]["externalId"] = "Return"
        if name == "owner":
            item["customer"]["ids"]["mindboxId"] = "private-conflicting-owner"
        if name == "date":
            item["firstAction"]["dateTimeUtc"] = "2026-01-02T13:15:00+03:00"
        if name == "missing-line":
            item["lines"][0]["priceOfLine"] = 500
        changed.append(item)
    return list(raw.values()), changed


def _map(mapping, sid, raw):
    assert (sid is None) == (raw is None)
    if sid is not None:
        assert mapping.setdefault(sid, raw) == raw


def _measure(value, mapping):
    return value if value.value is None else replace(value, value=mapping[value.value])


def _align_line(line, mapping):
    return replace(line, stable_line_key=_measure(line.stable_line_key, mapping))


def test_full_canonical_sqlite_core_parity(dataset):
    first, second = _matrix()
    _source(dataset, first, second)
    with index.reference_observations(raw_root=dataset[0], catalog_path=dataset[1]) as observations:
        reference = {row.source_ref: row for row in observations}
    manifest = _build(dataset)
    assert manifest["binding"]["schema"] == 4
    assert manifest["binding"]["counts"]["orders"] == len(first) + len(second)
    refs_by_order = defaultdict(list)
    for ref in reference.values():
        if ref.order_key.value is not None:
            refs_by_order[ref.order_key.value].append(ref)
    ref_facts = {
        key: reconcile_order(refs_by_order[key], order_sort_rank=rank)
        for rank, key in enumerate(sorted(refs_by_order), 1)
    }
    order_ids, customers, line_ids = {}, {}, {}
    line_identity = {}
    with index.open_unified_index(**_paths(dataset)) as reader:
        indexed = {row.observation.source_ref: row for row in reader.orders()}
        assert indexed.keys() == reference.keys()
        for ref, expected in reference.items():
            actual = indexed[ref].observation
            _map(order_ids, actual.order_key.value, expected.order_key.value)
            _map(customers, actual.customer_key.value, expected.customer_key.value)
            for line, original in zip(actual.lines, expected.lines, strict=True):
                _map(line_ids, line.stable_line_key.value, original.stable_line_key.value)
                if line.stable_line_key.value is not None:
                    scoped = (expected.order_key.value or ref, original.stable_line_key.value)
                    assert line_identity.setdefault(line.stable_line_key.value, scoped) == scoped
            assert replace(actual, order_key=_measure(actual.order_key, order_ids),
                           customer_key=_measure(actual.customer_key, customers),
                           lines=tuple(_align_line(line, line_ids) for line in actual.lines)) == expected
        assert len(set(order_ids.values())) == len(order_ids)
        assert len(set(customers.values())) == len(customers)
        assert len(set(line_identity.values())) == len(line_identity)
        facts = {}
        for fact in reader.facts():
            key = order_ids[fact.order_key]
            lines = tuple(sorted((_align_line(line, line_ids) for line in fact.resolved_lines),
                                 key=lambda line: (line.stable_line_key.value is None,
                                                   line.stable_line_key.value or "", line.ordinal or 0)))
            aligned = replace(fact, order_key=key, customer_key=_measure(fact.customer_key, customers),
                              resolved_lines=lines)
            assert aligned == ref_facts[key]
            assert key not in facts
            facts[key] = aligned
        assert facts.keys() == ref_facts.keys()
        assert sum(fact.purchase.observed.value is True for fact in facts.values()) == 18
        assert facts["private-order-f"].money[0].recognized_purchase.value == Decimal("1234567890.123456789012345678901234")
        assert facts["private-order-f-cp"].money[0].known_purchase_subtotal.value == 20
        assert facts["private-order-f-cp"].money[0].recognized_purchase.reasons == ("CP_ALLOCATION_UNKNOWN",)
        assert facts["private-order-transition"].purchase.positions.value == 1
        assert facts["private-order-transition"].state_quality is Quality.CONFLICT
        assert facts["private-order-only-return"].purchase.observed.value is False
        assert facts["private-order-duplicate"].snapshots.exact_duplicates == 1
        assert facts["private-order-duplicate"].snapshots.business_equivalent == 1
        assert facts["private-order-missing-line"].purchase.positions.value is None
        assert facts["private-order-zero-qty"].resolved_lines[0].status_code.value == "F"
        assert facts["private-order-zero-price"].money[0].recognized_purchase.value == 0
        assert facts["private-order-owner"].customer_key.quality is Quality.CONFLICT
        assert facts["private-order-date"].order_created_at.quality is Quality.CONFLICT
        first_order = next(row for row in indexed.values() if row.order_sort_rank == ref_facts["private-order-f"].order_sort_rank)
        assert first_order.line_numbers == ((0, 913),)
        assert len(list(reader.profile_links())) == 3
        assert reader.db.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 7
        assert reader.db.execute("SELECT COUNT(*) FROM item_metadata").fetchone()[0] == 2
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.db.execute("DELETE FROM order_observations")


def test_legacy_and_unified_pointers_are_independent(dataset):
    paths = _paths(dataset)
    old = legacy.build_statistics_index(**paths, expected_bytes=1024**2)
    target = dataset[3]
    pointer = target / "current.json"
    before = {path.name: path.read_bytes() for path in (target / old["generation"]).iterdir()}
    old_pointer = pointer.read_bytes()
    with pytest.raises(StatisticsIndexError, match="INDEX_MISSING"):
        with index.open_unified_index(**paths):
            pytest.fail("Schema 3 was opened by the v4 reader")
    _build(dataset)
    assert pointer.read_bytes() == old_pointer
    assert {path.name: path.read_bytes() for path in (target / old["generation"]).iterdir()} == before
    source, _ = capture_sources(dataset[0], dataset[1], AnalysisFilter())
    connection, opened = legacy.open_generation(target, source)
    try:
        assert opened["schema"] == 3
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    finally:
        connection.close()
    with index.open_unified_index(**paths) as reader:
        assert len(list(reader.facts())) == 1


def test_no_original_ids_profiles_or_contacts_in_published_components(dataset):
    raw = order(["F"], "synthetic-private-order-secret")
    raw["customer"]["ids"]["mindboxId"] = "synthetic-private-owner-secret"
    raw["lines"][0]["id"] = "synthetic-private-line-secret"
    raw["contacts"] = {"mobilePhone": "synthetic-private-phone-secret"}
    _source(dataset, [raw])
    with closing(sqlite3.connect(dataset[0] / "canonical/customers.sqlite")) as profiles, profiles:
        profiles.execute("INSERT INTO profiles VALUES(?, '', ?)",
                         ("synthetic-private-profile-secret", json.dumps({"birthDate": "1921-03-05", "sex": "male"})))
    manifest = _build(dataset)
    generation = dataset[3] / manifest["binding"]["generation"]
    for name in ("statistics.sqlite", "demographics.sqlite", "manifest.json"):
        contents = (generation / name).read_bytes()
        assert b"synthetic-private-" not in contents
        assert b"1921-03-05" not in contents
    with index.open_unified_index(**_paths(dataset)) as reader:
        assert len(list(reader.profile_links())) == 4
        with reader.profile_demographics() as profiles:
            assert any(born == date(1921, 3, 5) and sex == "male" for _, born, sex in profiles)
        assert [row[1] for row in reader.db.execute("PRAGMA demographics.table_info(profiles)")] == ["ordinal", "customer"]


@pytest.mark.parametrize("failure", ["cancel", "disk", "source", "publication"])
def test_failure_preserves_previous_pointer_and_removes_owned_staging(dataset, monkeypatch, failure):
    _build(dataset)
    target = dataset[3]
    pointer = target / index.POINTER_NAME
    before = pointer.read_bytes()
    children = set(target.iterdir())
    options = {}
    error = StatisticsIndexError
    reason = failure
    if failure == "cancel":
        samples = []
        options.update(observer=samples.append, cancelled=lambda: bool(samples))
        error, reason = InterruptedError, "cancelled"
    elif failure == "disk":
        monkeypatch.setattr(legacy.shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(10, 9, 1))
        reason = "INSUFFICIENT_DISK"
    elif failure == "source":
        root, _, data, _ = dataset
        part = root / next(iter(data["orders"].values()))["directory"] / "orders_part_001.json"
        original = index._populate

        def change_source(*args):
            counts = original(*args)
            part.write_bytes(part.read_bytes() + b" ")
            return counts

        monkeypatch.setattr(index, "_populate", change_source)
        reason = "STALE_SOURCE"
    else:
        original = index.atomic_json

        def fail_publication(path, value):
            if path.name == index.POINTER_NAME:
                raise OSError("simulated publication failure")
            original(path, value)

        monkeypatch.setattr(index, "atomic_json", fail_publication)
        reason = "SOURCE_OR_BUILD_UNAVAILABLE"
    with pytest.raises(error, match=reason):
        _build(dataset, **options)
    assert pointer.read_bytes() == before
    assert set(target.iterdir()) == children


def test_incompatible_projection_and_source_guard(dataset):
    manifest = _build(dataset)
    root, _, data, target = dataset
    pointer = target / index.POINTER_NAME
    previous = pointer.read_bytes()
    bad = json.loads(previous)
    bad["schema"] = 3
    pointer.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(StatisticsIndexError, match="INCOMPATIBLE_SCHEMA"):
        with index.open_unified_index(**_paths(dataset)):
            pytest.fail("Invalid pointer accepted")
    pointer.write_bytes(previous)
    part = root / next(iter(data["orders"].values()))["directory"] / "orders_part_001.json"
    original = part.read_bytes()
    part.write_bytes(original + b" ")
    with pytest.raises(StatisticsIndexError, match="STALE_SOURCE"):
        with index.open_unified_index(**_paths(dataset)):
            pytest.fail("Source change accepted")
    assert (target / manifest["binding"]["generation"]).exists()


def test_missing_projection_column_is_rejected_even_with_updated_file_signature(dataset):
    manifest = _build(dataset)
    target = dataset[3]
    generation = target / manifest["binding"]["generation"]
    database = generation / "statistics.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("ALTER TABLE order_line_observations DROP COLUMN input_quantity")
    manifest["files"][database.name]["file"] = list(index._stat(database))
    (generation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (target / index.POINTER_NAME).write_text(json.dumps({"schema": 4, "generation": generation.name,
                                                        "manifest_sha256": legacy.manifest_digest(manifest)}),
                                            encoding="utf-8")
    with pytest.raises(StatisticsIndexError, match="INCOMPATIBLE_SCHEMA"):
        with index.open_unified_index(**_paths(dataset)):
            pytest.fail("Missing projected field accepted")


def test_reference_disk_spool_and_batch_projection_of_many_repeats(dataset):
    raw = order(["F", "CP"], "synthetic-order")
    repeat = deepcopy(raw)
    repeat["lines"][0]["status"]["ids"]["externalId"] = "Return"
    _source(dataset, [raw] * 1002, [repeat])
    scratch = dataset[0] / "scratch"
    scratch.mkdir(mode=0o700)
    with index.reference_observations(raw_root=dataset[0], catalog_path=dataset[1]) as observations:
        expected = reconcile_order(observations, order_sort_rank=1)
    with index.reference_facts(raw_root=dataset[0], catalog_path=dataset[1],
                               scratch_dir=scratch, expected_bytes=1024**2) as facts:
        assert list(facts) == [expected]
    assert not list(scratch.iterdir())
    _build(dataset)
    with index.open_unified_index(**_paths(dataset)) as reader:
        fact, = reader.facts()
        assert fact.snapshots == expected.snapshots
        assert fact.snapshots.exact_duplicates == 1001
        assert fact.purchase == expected.purchase
        assert fact.money == expected.money


def test_manual_and_api_sources_share_scoped_order_identity(dataset):
    root, _, data, _ = dataset
    _source(dataset, [order(["Return"], "shared-order")])
    since, until = "2025-12-31T00:00:00+00:00", "2026-01-01T00:00:00+00:00"
    manual = order(["F"], "shared-order")
    data["manual_interactions"] = {
        "since": since, "until": until, "updated": STAMP,
        "orders": put_entry(root, "orders", [manual], 901, since, until, "MANUAL"),
        "actions": put_entry(root, "actions", [], 901, since, until, "MANUAL"),
    }
    save(root, data)
    manifest = _build(dataset)
    sources = manifest["binding"]["provenance"]["parts"]
    with index.open_unified_index(**_paths(dataset)) as reader:
        observations = list(reader.orders())
        assert [sources[row.observation.source_ref[0]][1]["source_kind"] for row in observations] == ["MANUAL", "API"]
        assert len({row.observation.order_key.value for row in observations}) == 1
        assert len({row.observation.lines[0].stable_line_key.value for row in observations}) == 1
        fact, = reader.facts()
        assert fact.purchase.observed.value is True
        assert fact.purchase.positions.value == 1
        assert fact.state_quality is Quality.CONFLICT


def test_foreign_staging_file_is_never_removed(dataset, monkeypatch):
    previous = _build(dataset)
    target = dataset[3]
    pointer = (target / index.POINTER_NAME).read_bytes()
    protected = index.protect_directory
    inserted = []

    def insert_unexpected(folder):
        protected(folder)
        path = folder / "foreign.txt"
        path.write_text("owned by someone else")
        inserted.append(path)
        raise StatisticsIndexError("SIMULATED_ACL_FAILURE")

    monkeypatch.setattr(index, "protect_directory", insert_unexpected)
    with pytest.raises(StatisticsIndexError, match="UNEXPECTED_STAGING_CONTENT"):
        _build(dataset)
    assert inserted[0].read_text() == "owned by someone else"
    assert (target / index.POINTER_NAME).read_bytes() == pointer
    assert (target / previous["binding"]["generation"]).is_dir()


def test_unexpected_journal_is_rejected_before_reader(dataset):
    manifest = _build(dataset)
    database = dataset[3] / manifest["binding"]["generation"] / "statistics.sqlite"
    database.with_suffix(".sqlite-wal").write_bytes(b"synthetic sidecar")
    with pytest.raises(StatisticsIndexError, match="UNEXPECTED_SQLITE_SIDECAR"):
        with index.open_unified_index(**_paths(dataset)):
            pytest.fail("Sidecar was accepted")


def test_bad_source_json_cannot_replace_published_generation(dataset):
    _build(dataset)
    root, _, data, target = dataset
    previous = (target / index.POINTER_NAME).read_bytes()
    folders = set(target.iterdir())
    part = root / next(iter(data["orders"].values()))["directory"] / "orders_part_001.json"
    part.write_text('{"orders":[{}', encoding="utf-8")
    with pytest.raises(StatisticsIndexError, match="SOURCE_OR_BUILD_UNAVAILABLE"):
        _build(dataset)
    assert (target / index.POINTER_NAME).read_bytes() == previous
    assert set(target.iterdir()) == folders


def test_schema4_build_and_read_with_real_private_acl(tmp_path):
    root, catalog, _ = dataset_fixture.__wrapped__(tmp_path)
    target = tmp_path / "protected-index"
    target.mkdir(mode=0o700)
    index.protect_directory(target)
    paths = dict(raw_root=root, catalog_path=catalog, index_dir=target)
    index.build_unified_index(**paths, expected_bytes=1024**2)
    with index.open_unified_index(**paths) as reader:
        assert len(list(reader.facts())) == 1


def test_open_uses_projected_orders_and_rechecks_source_at_exit(dataset, monkeypatch):
    _build(dataset)

    def reject_raw(*args):
        pytest.fail("INDEXED tried to parse a raw Orders/Actions part")

    monkeypatch.setattr(index, "iter_json_records", reject_raw)
    root, _, data, _ = dataset
    part = root / next(iter(data["orders"].values()))["directory"] / "orders_part_001.json"
    with pytest.raises(StatisticsIndexError, match="STALE_SOURCE"):
        with index.open_unified_index(**_paths(dataset)) as reader:
            assert len(list(reader.facts())) == 1
            part.write_bytes(part.read_bytes() + b" ")


@pytest.mark.skipif(os.name != "nt", reason="Windows memory monitor")
def test_memory_limit_rejects_build_without_publishing(dataset, monkeypatch):
    monkeypatch.setattr(legacy, "sample_memory", lambda: {
        "private_usage": 3 * legacy.GIB + 1,
        "available_physical": 4 * legacy.GIB,
        "available_commit": 4 * legacy.GIB,
    })
    with pytest.raises(StatisticsIndexError, match="INSUFFICIENT_MEMORY"):
        _build(dataset)
    assert {path.name for path in dataset[3].iterdir()} == {".writer.lock"}
