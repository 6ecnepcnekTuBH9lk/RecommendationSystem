"""Small synthetic canonical components only; never consume the real dataset."""

from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import Event

import pytest

from Application import customer_history_index as module
from Application.customer_history_index import CustomerHistoryIndexService, HistoryIndexStatus as S
from Application.mindbox.canonical_customers import encode
from Application.mindbox.daily_training_batch import _writer_lock
from Application.mindbox.selection import DEFAULT_SELECTION


STAMP = "2026-01-01T12:00:00.123456Z"


def action(key="action-1", customer="A", **fields):
    return {"ids": {"mindboxId": key}, "customer": {"ids": {"mindboxId": customer}},
            "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}, "name": "Синтетический просмотр"},
            "dateTimeUtc": STAMP, "creationDateTimeUtc": "2026-01-02T12:00:00Z",
            "products": [{"ids": {"offline1C": "000123456789"}}, {"ids": {"kanzlerKz": "000123456789"}},
                         {"ids": {"offline1C": "000123456789"}}],
            "productCategories": [{"ids": {"offline1C": "00001"}}],
            "channel": {"ids": {"externalId": "001", "systemName": "web", "mindboxId": "002"}, "name": "Канал"},
            "productView": {"price": Decimal("10.12345678901234567890"), "isAvailable": False}, **fields}


def order(key="order-1", customer="A", **fields):
    return {"ids": {"mindboxId": key, "retailOrderId": "000001"},
            "customer": {"ids": {"mindboxId": customer}},
            "firstAction": {"dateTimeUtc": STAMP, "channel": {"ids": {"externalId": "001"}, "name": "Канал"}},
            "totalPrice": Decimal("200.9999999999999999999"), "deliveryCost": Decimal("0.000001"),
            "lines": [{"number": 1, "product": {"ids": {"offline1C": "000123456789"}, "name": "Товар"},
                       "quantity": Decimal("-2.500"), "priceOfLine": Decimal("-99.12345678901234567890"),
                       "basePricePerItem": Decimal("50.0000"), "status": {"ids": {"externalId": "CP"}}},
                      {"id": "001", "number": 2, "product": {"ids": {"kanzlerKz": "000999"}},
                       "quantity": 1, "priceOfLine": 10, "status": {"ids": {"externalId": "unknown-cancel"}}}], **fields}


def merges():
    return [{"id": "m1", "dateTimeUtc": "2026-01-01T00:00:00Z",
             "resultingCustomer": {"ids": {"mindboxId": "B"}}, "mergedCustomers": [{"ids": {"mindboxId": "A"}}]},
            {"id": "m2", "dateTimeUtc": "2026-01-01T00:00:00Z",
             "resultingCustomer": {"ids": {"mindboxId": "C"}}, "mergedCustomers": [{"ids": {"mindboxId": "B"}}]}]


def write_part(directory, name, rows, number=1):
    root = {"actions": "customerActions", "orders": "orders", "customer_merges": "customerMerges"}[name]
    (directory / f"{name}_part_{number:03d}.json").write_text(encode({root: rows}), encoding="utf-8")


def component(root, name, token, rows, *, since="2026-01-01T00:00:00+00:00", until="2026-01-02T00:00:00+00:00", kind="API"):
    directory = root / "canonical/objects" / (token * 32) / name
    directory.mkdir(parents=True, exist_ok=True)
    write_part(directory, name, rows)
    return {"directory": directory.relative_to(root).as_posix(), "since": since, "until": until,
            "parts": 1, "source_kind": kind, "updated": "2026-02-01T00:00:00+00:00",
            "export_id": "synthetic-export" if kind == "API" else None, "operation": "synthetic-operation" if kind == "API" else "MANUAL"}


def save_catalog(root, metadata):
    (root / "canonical/catalog.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "raw"
    metadata = {"schema_version": 2, "revision": "synthetic-revision", "manual_interactions": None,
                "selection": asdict(DEFAULT_SELECTION),
                "customer_merges": component(root, "customer_merges", "1", merges(), since="2022-01-01T00:00:00+00:00", until="2026-03-01T00:00:00+00:00"),
                "actions": {"2026-01-01": component(root, "actions", "2", [action()])},
                "orders": {"2026-01-01": component(root, "orders", "3", [order()])}}
    (root / "canonical/.writer.lock").write_bytes(b"0")
    save_catalog(root, metadata)
    return root, CustomerHistoryIndexService(cache_dir=tmp_path / "cache"), metadata


def build(source, **kwargs):
    return source[1].build_index(raw_root=source[0], **kwargs)


def status(source, **kwargs):
    return source[1].get_status(raw_root=source[0], **kwargs)


def rows(source, sql, args=()):
    path = source[1].index_path(raw_root=source[0])
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(sql, args).fetchall()


def update_rows(source, name, items):
    root, _, metadata = source
    entry = metadata["customer_merges"] if name == "customer_merges" else next(iter(metadata[name].values()))
    write_part(root / entry["directory"], name, items)


def test_build_normalized_schema_precise_values_and_origins(source):
    result = build(source, batch_size=1)
    assert result.status is S.READY and not result.reused
    assert result.counts.raw_actions == result.counts.logical_actions == result.counts.action_variants == 1
    assert result.counts.raw_orders == result.counts.logical_orders == result.counts.order_snapshots == 1
    assert status(source, verify_integrity=True).status is S.READY
    expected_us = module._micros(datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc))
    assert rows(source, "SELECT DISTINCT event_us FROM events") == [(expected_us,)]
    assert rows(source, "SELECT DISTINCT canonical_customer_id FROM events") == [("C",)]
    assert rows(source, "SELECT COUNT(*) FROM record_origins") == [(2,)]
    assert rows(source, "SELECT COUNT(*) FROM product_occurrences WHERE role='product'") == [(3,)]
    assert rows(source, "SELECT COUNT(*) FROM product_occurrences WHERE role='category'") == [(1,)]
    assert rows(source, "SELECT COUNT(*) FROM product_occurrences WHERE role='line'") == [(2,)]
    assert rows(source, "SELECT quantity,price_of_line,base_price_per_item,raw_status FROM order_lines WHERE ordinal=0") == [
        ("-2.500", "-99.12345678901234567890", "50.0000", "CP")]
    assert rows(source, "SELECT total_price,delivery_cost FROM order_snapshots") == [("200.9999999999999999999", "0.000001")]
    assert rows(source, "SELECT view_price,view_available,channel_external_id,channel_mindbox_id FROM actions") == [
        ("10.12345678901234567890", 0, "001", "002")]
    assert rows(source, "SELECT DISTINCT namespace,value FROM product_keys WHERE role='product' ORDER BY namespace") == [
        ("kanzlerKz", "000123456789"), ("offline1C", "000123456789")]
    assert rows(source, "SELECT COUNT(*) FROM source_parts") == [(3,)]
    assert rows(source, "PRAGMA foreign_key_check") == []
    assert rows(source, "SELECT key FROM metadata WHERE key='normalizer_version'")


def test_action_repeats_conflicts_and_distinct_ids_do_not_duplicate_logical_events(source):
    first = action()
    conflict = action(products=[{"ids": {"offline1C": "different"}}])
    update_rows(source, "actions", [first, deepcopy(first), conflict, action("action-2")])
    result = build(source, batch_size=2)
    assert result.status is S.READY
    assert result.counts.raw_actions == 4 and result.counts.logical_actions == 2 and result.counts.action_variants == 3
    assert result.counts.identical_repeats == 1 and result.counts.conflicting_events == 1
    assert rows(source, "SELECT variant_count,occurrence_count FROM events WHERE source_event_id='action-1'") == [(2, 3)]
    assert rows(source, "SELECT COUNT(*) FROM actions") == [(3,)]


def test_order_snapshots_deduplicate_without_losing_origins_or_synthetic_lines(source):
    original = order()
    changed = deepcopy(original)
    changed["lines"].reverse()
    update_rows(source, "orders", [original, deepcopy(original), changed])
    result = build(source)
    assert result.status is S.READY and result.counts.logical_orders == 1 and result.counts.order_snapshots == 2
    assert result.counts.identical_repeats == 1 and result.counts.conflicting_events == 1
    assert rows(source, "SELECT COUNT(*) FROM order_lines") == [(4,)]
    assert rows(source, "SELECT COUNT(DISTINCT line_id),COUNT(source_line_key) FROM order_lines WHERE is_synthetic=1") == [(2, 0)]
    assert rows(source, "SELECT COUNT(*) FROM record_origins r JOIN variants v USING(variant_key) JOIN events e USING(event_key) WHERE e.source='ORDER'") == [(3,)]
    assert rows(source, "SELECT COUNT(*) FROM orders") == [(1,)]


@pytest.mark.parametrize("kind", ["actions", "orders"])
def test_disagreeing_owner_and_time_have_no_arbitrary_logical_choice(source, kind):
    make = action if kind == "actions" else order
    first, changed = make(), make(customer="OTHER")
    if kind == "actions":
        changed["dateTimeUtc"] = "2026-01-01T14:00:00Z"
        changed["actionTemplate"]["ids"]["systemName"] = "UnknownAction"
    else:
        changed["firstAction"]["dateTimeUtc"] = "2026-01-01T14:00:00Z"
    update_rows(source, kind, [first, changed, deepcopy(first)])
    result = build(source)
    assert result.status is S.READY and result.counts.owner_conflicts == result.counts.time_conflicts == 1
    key = "action-1" if kind == "actions" else "order-1"
    common = rows(source, "SELECT canonical_customer_id,event_us,owner_conflict,time_conflict,kind FROM events WHERE source_event_id=?", (key,))[0]
    assert common[:4] == (None, None, 1, 1)
    assert common[4] == (None if kind == "actions" else "ORDER")
    assert rows(source, "SELECT DISTINCT v.canonical_customer_id FROM variants v JOIN events e USING(event_key) WHERE source_event_id=? ORDER BY 1", (key,)) == [("C",), ("OTHER",)]
    assert rows(source, "SELECT COUNT(*) FROM event_customers ec JOIN events e USING(event_key) WHERE source_event_id=?", (key,)) == [(2,)]


def test_merge_aliases_same_canonical_are_not_false_owner_conflicts(source):
    update_rows(source, "actions", [action(customer="A"), action(customer="B")])
    result = build(source)
    assert result.status is S.READY and result.counts.conflicting_events == 1 and result.counts.owner_conflicts == 0
    assert rows(source, "SELECT canonical_customer_id FROM events WHERE source='ACTION'") == [("C",)]


def test_unresolved_product_identifiers_and_quality_are_not_lost(source):
    raw = action(products=[{"ids": {"website": "000001"}}, {"ids": {"offline1C": "001", "kanzlerKz": "002"}}, {}])
    update_rows(source, "actions", [raw])
    assert build(source).status is S.READY
    assert rows(source, "SELECT ordinal,selected_value FROM product_occurrences WHERE role='product' ORDER BY ordinal") == [(0, None), (1, None), (2, None)]
    assert rows(source, "SELECT COUNT(*) FROM product_keys WHERE role='product'") == [(3,)]
    qualities = rows(source, "SELECT quality FROM product_occurrences WHERE role='product' ORDER BY ordinal")
    assert "UNSUPPORTED_PRODUCT_NAMESPACE" in qualities[0][0]
    assert "AMBIGUOUS_PRODUCT_NAMESPACE" in qualities[1][0]
    assert "MISSING_PRODUCT" in qualities[2][0]


def test_all_periods_gaps_and_unpaired_components_are_retained(source):
    root, _, metadata = source
    metadata["actions"]["2026-01-05"] = component(root, "actions", "4", [action("late-action")], since="2026-01-05T00:00:00+00:00", until="2026-01-06T00:00:00+00:00")
    save_catalog(root, metadata)
    result = build(source)
    assert result.status is S.READY and result.counts.logical_actions == 2
    action_intervals = [c for c in result.coverage if c.component == "actions"]
    assert len(action_intervals) == 2 and action_intervals[0].until_utc < action_intervals[1].since_utc
    assert result.counts.logical_orders == 1


def test_manual_precedence_ignores_shadowed_api_but_records_duplicate_origins(source):
    root, _, metadata = source
    since, until = "2026-01-01T00:00:00+00:00", "2026-01-03T00:00:00+00:00"
    pair = {"since": since, "until": until, "updated": "2026-02-01T00:00:00+00:00"}
    for kind, values in (("actions", [action("manual-action")]), ("orders", [order("manual-order")])):
        pair[kind] = component(root, kind, "4", values, since=since, until=until, kind="MANUAL")
    metadata["manual_interactions"] = pair
    metadata["actions"]["2026-01-03"] = component(root, "actions", "5", [action("manual-action")], since=until, until="2026-01-04T00:00:00+00:00")
    save_catalog(root, metadata)
    result = build(source)
    assert result.status is S.READY and result.counts.logical_actions == result.counts.logical_orders == 1
    assert result.counts.identical_repeats == 1
    assert rows(source, "SELECT source_event_id FROM events ORDER BY source") == [("manual-action",), ("manual-order",)]
    assert rows(source, "SELECT COUNT(*) FROM record_origins") == [(3,)]
    old = root / metadata["actions"]["2026-01-01"]["directory"]
    write_part(old, "actions", [action("changed-ignored")])
    assert status(source).status is S.READY


def test_parts_and_component_global_ordinals_preserve_provenance(source):
    root, _, metadata = source
    entry = metadata["actions"]["2026-01-01"]
    write_part(root / entry["directory"], "actions", [action()], 2)
    entry["parts"] = 2
    save_catalog(root, metadata)
    assert build(source).status is S.READY
    assert rows(source, "SELECT r.record_ordinal FROM record_origins r JOIN sources s USING(source_key) WHERE s.component='actions' ORDER BY 1") == [(0,), (1,)]
    assert rows(source, "SELECT COUNT(*) FROM source_parts p JOIN sources s USING(source_key) WHERE s.component='actions'") == [(2,)]


@pytest.mark.parametrize("change", ["actions", "orders", "merges", "selection", "coverage", "schema", "normalizer"])
def test_relevant_inputs_invalidate_and_explicit_build_restores(source, change, monkeypatch):
    assert build(source).status is S.READY
    root, _, metadata = source
    if change in ("actions", "orders"):
        update_rows(source, change, [action("different") if change == "actions" else order("different")])
    elif change == "merges":
        update_rows(source, "customer_merges", [])
    elif change == "selection":
        metadata["selection"]["view_action_system_names"] = ["DifferentView"]
        save_catalog(root, metadata)
    elif change == "coverage":
        metadata["customer_merges"]["since"] = "2021-01-01T00:00:00+00:00"
        save_catalog(root, metadata)
    elif change == "schema":
        monkeypatch.setattr(module, "INDEX_SCHEMA_VERSION", module.INDEX_SCHEMA_VERSION + 1)
    else:
        monkeypatch.setattr(module.records, "HISTORY_SCHEMA_VERSION", 2)
    assert status(source).status is S.STALE
    result = build(source)
    assert result.status is S.READY and not result.reused
    assert status(source).status is S.READY


def test_code_fingerprint_lf_crlf_preserves_legacy_lf_hash(tmp_path):
    content = b'# synthetic source\nVALUE = "text"\n'
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(content)
    crlf.write_bytes(content.replace(b"\n", b"\r\n"))
    expected = hashlib.sha256(content).hexdigest()
    assert module._code_fingerprint(lf) == module._code_fingerprint(crlf) == expected


@pytest.mark.parametrize("content", [b"VALUE = 2\n", b"VALUE = 1 # comment\n", b"VALUE  = 1\n",
                                     b"VALUE = 1", b"VALUE = 1\r", b"\xef\xbb\xbfVALUE = 1\n"])
def test_code_fingerprint_keeps_all_other_byte_changes(tmp_path, content):
    path = tmp_path / "code.py"
    path.write_bytes(b"VALUE = 1\n")
    before = module._code_fingerprint(path)
    path.write_bytes(content)
    assert module._code_fingerprint(path) != before


@pytest.mark.parametrize("endings", [(b"\n", b"\n"), (b"\r\n", b"\n"),
                                     (b"\n", b"\r\n"), (b"\r\n", b"\r\n")])
def test_source_signature_and_published_index_survive_code_eol_only(source, tmp_path, monkeypatch, endings):
    modules = (module.normalization, module.records)
    files = [tmp_path / f"source-{n}.py" for n in range(2)]
    contents = [b"# normalization fixture\nVALUE = 1\n", b"# records fixture\nVALUE = 2\n"]
    for mod, path, content in zip(modules, files, contents):
        path.write_bytes(content)
        monkeypatch.setattr(mod, "__file__", str(path))
    root, service, _ = source
    captured = module._capture_sources(root)
    # Independent reconstruction of the pre-fix LF contract, not a new hash format.
    expected = module._hash([module.INDEX_SCHEMA_VERSION, module.records.HISTORY_SCHEMA_VERSION,
                             [hashlib.sha256(content).hexdigest() for content in contents], str(root),
                             asdict(captured.selection), [c.key for c in captured.components]])
    assert captured.version == expected
    built = build(source)
    assert built.status is S.READY and built.source_version == expected
    index_path = service.index_path(raw_root=root)
    before, before_stat = index_path.read_bytes(), module._stat(index_path)
    for path, content, ending in zip(files, contents, endings):
        path.write_bytes(content.replace(b"\n", ending))
    assert module._capture_sources(root).version == expected
    ready = status(source)
    assert ready.status is S.READY and ready.generation == built.generation and ready.source_version == expected
    assert index_path.read_bytes() == before and module._stat(index_path) == before_stat
    # No automatic trust of a legacy signature computed directly from CRLF bytes.
    if any(ending == b"\r\n" for ending in endings):
        legacy_crlf = module._hash([module.INDEX_SCHEMA_VERSION, module.records.HISTORY_SCHEMA_VERSION,
                                   [hashlib.sha256(path.read_bytes()).hexdigest() for path in files], str(root),
                                   asdict(captured.selection), [c.key for c in captured.components]])
        assert legacy_crlf != expected
    files[0].write_bytes(contents[0] + b"# substantive fingerprint change\n")
    assert module._capture_sources(root).version != expected
    assert status(source).status is S.STALE


@pytest.mark.parametrize("change", ["selection", "part_stat", "part_count", "schema", "normalizer_schema"])
def test_eol_canonicalization_does_not_bypass_other_source_gates(source, monkeypatch, change):
    assert build(source).status is S.READY
    root, _, metadata = source
    before = module._capture_sources(root).version
    if change == "selection":
        metadata["selection"]["view_action_system_names"] = ["DifferentView"]
        save_catalog(root, metadata)
    elif change == "part_stat":
        original = module._stat
        def changed_stat(path):
            values = original(path)
            return (*values[:2], values[2] + 1, values[3]) if path.suffix == ".json" else values
        monkeypatch.setattr(module, "_stat", changed_stat)
    elif change == "part_count":
        entry = metadata["actions"]["2026-01-01"]
        write_part(root / entry["directory"], "actions", [action("second")], 2)
        entry["parts"] = 2
        save_catalog(root, metadata)
    elif change == "schema":
        monkeypatch.setattr(module, "INDEX_SCHEMA_VERSION", module.INDEX_SCHEMA_VERSION + 1)
    else:
        monkeypatch.setattr(module.records, "HISTORY_SCHEMA_VERSION", module.records.HISTORY_SCHEMA_VERSION + 1)
    assert module._capture_sources(root).version != before
    assert status(source).status is S.STALE


def test_customers_bpr_and_global_revision_are_not_dependencies(source, monkeypatch):
    first = build(source)
    source[2]["revision"] = "unrelated-revision"
    save_catalog(source[0], source[2])
    (source[0] / "canonical/customers.sqlite").write_bytes(b"synthetic-customer-change")
    (source[0] / "model").mkdir()
    (source[0] / "model/current.json").write_text('{"changed": true}')
    monkeypatch.setattr(module, "iter_export", lambda *args, **kwargs: pytest.fail("No source scan on reuse/status"))
    assert status(source).status is S.READY
    reused = build(source)
    assert reused.status is S.READY and reused.reused and reused.generation == first.generation


def test_batching_progress_force_repeatability_and_no_global_id_sets(source, monkeypatch):
    update_rows(source, "actions", [action(f"a-{i}") for i in range(11)])
    original = module._write_batch
    sizes = []
    def write_batch(connection, batch):
        sizes.append(len(batch))
        original(connection, batch)
    monkeypatch.setattr(module, "_write_batch", write_batch)
    progress = []
    first = build(source, batch_size=3, progress_every=4, progress=progress.append)
    assert first.status is S.READY and max(sizes) <= 3 and sum(sizes) == 12
    assert [p.processed for p in progress if p.phase != "customer_merges"] == [4, 8, 12, 12]
    second = build(source, force=True, batch_size=1)
    assert second.status is S.READY and not second.reused and first.generation != second.generation
    assert second.counts == first.counts and rows(source, "SELECT COUNT(*) FROM events") == [(12,)]


def test_cancelled_or_failed_build_preserves_previous_index_and_removes_temporary(source):
    assert build(source).status is S.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    cancel = Event()
    result = build(source, force=True, cancelled=cancel.is_set, progress=lambda p: cancel.set(), progress_every=1, batch_size=1)
    assert result.status is S.CANCELLED
    assert path.read_bytes() == before and not list(path.parent.glob(".building-*"))
    assert status(source).status is S.READY
    update_rows(source, "actions", [action(customer=" invalid")])
    assert build(source).status is S.SOURCE_INCONSISTENT
    assert path.read_bytes() == before and not list(path.parent.glob(".building-*"))


def test_cancellation_after_data_commit_before_publication_never_overwrites(source):
    assert build(source).status is S.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    cancel = Event()
    result = build(source, force=True, cancelled=cancel.is_set, progress=lambda p: cancel.set() if p.phase == "complete" else None)
    assert result.status is S.CANCELLED and path.read_bytes() == before
    assert not list(path.parent.glob(".building-*"))


def test_changed_source_during_build_or_status_cannot_confirm_ready(source, monkeypatch):
    assert build(source).status is S.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    original = module._capture_sources
    calls = 0
    def changed(root):
        nonlocal calls
        calls += 1
        captured = original(root)
        return captured if calls == 1 else replace(captured, version="changed")
    monkeypatch.setattr(module, "_capture_sources", changed)
    assert build(source, force=True).status is S.STALE
    assert path.read_bytes() == before
    calls = 0
    assert status(source).status is S.STALE


def test_atomic_replace_failure_never_loses_published_index(source, monkeypatch):
    assert build(source).status is S.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    monkeypatch.setattr(module.os, "replace", MockReplace())
    assert build(source, force=True).status in (S.UNAVAILABLE, S.SOURCE_UNAVAILABLE)
    assert path.read_bytes() == before and status(source).status is S.READY
    assert not list(path.parent.glob(".building-*"))


class MockReplace:
    def __call__(self, *args):
        raise PermissionError("Synthetic publication blocked")


def test_build_lock_serializes_instances_and_canonical_writer_is_respected(source):
    root, service, _ = source
    service.cache_dir.mkdir()
    with _writer_lock(service.cache_dir):
        another = CustomerHistoryIndexService(cache_dir=service.cache_dir)
        assert another.build_index(raw_root=root).status is S.BUILD_IN_PROGRESS
    service._build_lock.acquire()
    try:
        assert build(source).status is S.BUILD_IN_PROGRESS
    finally:
        service._build_lock.release()
    with _writer_lock(root / "canonical"):
        assert build(source).status is S.SOURCE_UNAVAILABLE
    assert build(source).status is S.READY


@pytest.mark.parametrize("damage", ["file", "table", "counts", "generation"])
def test_corrupt_cache_is_distinct_and_explicitly_repairable(source, damage):
    assert build(source).status is S.READY
    path = source[1].index_path(raw_root=source[0])
    if damage == "file":
        path.write_bytes(b"invalid sqlite")
    else:
        with closing(sqlite3.connect(path)) as connection, connection:
            if damage == "table":
                connection.execute("DROP TABLE product_keys")
            elif damage == "counts":
                connection.execute("UPDATE metadata SET value='invalid' WHERE key='counts'")
            else:
                connection.execute("UPDATE metadata SET value='invalid' WHERE key='generation'")
    assert status(source).status is S.CORRUPT
    assert build(source).status is S.READY


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_cache_sidecars_block_recovery_or_replacement(source, suffix):
    assert build(source).status is S.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    Path(str(path) + suffix).write_bytes(b"pending")
    assert status(source).status is S.UNAVAILABLE
    assert build(source, force=True).status is S.UNAVAILABLE
    assert path.read_bytes() == before


def test_source_errors_do_not_masquerade_as_build_lock_contention(source):
    source[2]["selection"]["view_action_system_names"] = []
    save_catalog(source[0], source[2])
    assert build(source).status is S.SOURCE_INCONSISTENT


def test_missing_sources_and_insufficient_merge_coverage_are_safe(source):
    assert build(source).status is S.READY
    source[2]["customer_merges"]["until"] = "2025-12-31T00:00:00+00:00"
    save_catalog(source[0], source[2])
    assert status(source).status is S.SOURCE_INCONSISTENT
    assert build(source).status is S.SOURCE_INCONSISTENT
    (source[0] / "canonical/catalog.json").unlink()
    assert status(source).status is S.SOURCE_UNAVAILABLE


def test_canonical_content_unchanged_and_raw_pii_not_stored_or_logged(source, caplog, capsys):
    original = action()
    original["customer"]["email"] = "PRIVATE-person@example.test"
    update_rows(source, "actions", [original])
    before = {p.relative_to(source[0]): hashlib.sha256(p.read_bytes()).digest() for p in source[0].rglob("*") if p.is_file()}
    built = build(source)
    ready = status(source)
    after = {p.relative_to(source[0]): hashlib.sha256(p.read_bytes()).digest() for p in source[0].rglob("*") if p.is_file()}
    assert before == after and built.status is S.READY
    contents = source[1].index_path(raw_root=source[0]).read_bytes()
    assert b"PRIVATE-person" not in contents
    assert "PRIVATE-person" not in repr(built) + repr(ready) + caplog.text + str(capsys.readouterr())
    assert "action-1" not in repr(built)


def test_public_wrappers_missing_status_and_read_only_reuse(source, monkeypatch):
    monkeypatch.setattr(module, "_DEFAULT_SERVICE", source[1])
    assert module.get_customer_history_index_status(raw_root=source[0]).status is S.MISSING
    assert not source[1].cache_dir.exists()
    first = module.build_customer_history_index(raw_root=source[0])
    assert first.status is S.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    monkeypatch.setattr(module, "iter_export", lambda *args, **kwargs: pytest.fail("No rebuild or source scan"))
    assert module.get_customer_history_index_status(raw_root=source[0]).status is S.READY
    assert module.build_customer_history_index(raw_root=source[0]).reused
    assert path.read_bytes() == before


def test_empty_dataset_and_cache_outside_source(source):
    source[2]["actions"] = source[2]["orders"] = {}
    save_catalog(source[0], source[2])
    result = build(source)
    assert result.status is S.READY and result.counts.logical_actions == result.counts.logical_orders == 0
    assert result.coverage == ()
    unsafe = CustomerHistoryIndexService(cache_dir=source[0] / "canonical/cache")
    assert unsafe.build_index(raw_root=source[0]).status is S.SOURCE_INCONSISTENT
    assert not unsafe.cache_dir.exists()


def test_indices_exist_for_future_customer_type_and_time_queries(source):
    assert build(source).status is S.READY
    plan = rows(source, "EXPLAIN QUERY PLAN SELECT event_key FROM events WHERE canonical_customer_id=? "
                       "AND kind=? ORDER BY event_us DESC,event_key DESC LIMIT 50", ("C", "VIEW"))
    assert any("events_customer_kind_time" in row[3] for row in plan)
    assert not hasattr(source[1], "get_history")


@pytest.mark.parametrize("damage", ["owner", "date", "payload", "origin", "counts"])
def test_explicit_integrity_check_detects_consensus_and_payload_corruption(source, damage):
    assert build(source).status is S.READY
    with closing(sqlite3.connect(source[1].index_path(raw_root=source[0]))) as connection, connection:
        if damage == "owner":
            connection.execute("UPDATE events SET canonical_customer_id='wrong' WHERE source='ACTION'")
        elif damage == "date":
            connection.execute("UPDATE events SET event_us=0 WHERE source='ACTION'")
        elif damage == "payload":
            connection.execute("DELETE FROM actions")
        elif damage == "origin":
            connection.execute("DELETE FROM record_origins")
        else:
            metadata = json.loads(connection.execute("SELECT value FROM metadata WHERE key='counts'").fetchone()[0])
            metadata['raw_actions'] += 1
            connection.execute("UPDATE metadata SET value=? WHERE key='counts'", (json.dumps(metadata),))
    assert status(source, verify_integrity=True).status is S.CORRUPT


def test_cancellation_during_sql_work_is_classified_correctly(source, monkeypatch):
    assert build(source).status is S.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    cancel = Event()
    original = module._inspect
    def interrupted(path, captured, verify_integrity=False, cancelled=lambda: False):
        if verify_integrity:
            cancel.set()
            raise sqlite3.OperationalError('interrupted')
        return original(path, captured, verify_integrity)
    monkeypatch.setattr(module, '_inspect', interrupted)
    assert build(source, force=True, cancelled=cancel.is_set).status is S.CANCELLED
    assert path.read_bytes() == before
    assert not list(path.parent.glob('.building-*'))


def test_callback_exception_is_safe_and_does_not_publish(source, caplog, capsys):
    def callback(progress):
        raise ValueError('PRIVATE-person@example.test')
    result = build(source, progress=callback, progress_every=1)
    assert result.status is S.SOURCE_INCONSISTENT
    assert not source[1].index_path(raw_root=source[0]).exists()
    assert 'PRIVATE-person' not in repr(result) + caplog.text + str(capsys.readouterr())
    assert not list(source[1].cache_dir.glob('.building-*'))


def test_read_only_status_closes_connections_and_does_not_recover_sqlite(source, monkeypatch):
    assert build(source).status is S.READY
    original, connections = sqlite3.connect, []
    def connect(path, **kwargs):
        assert str(path).endswith('?mode=ro') and kwargs['uri']
        connection = original(path, **kwargs)
        connections.append(connection)
        return connection
    monkeypatch.setattr(module.sqlite3, 'connect', connect)
    assert status(source, verify_integrity=True).status is S.READY
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')


def test_previous_crash_leftovers_are_never_used_as_an_index(source):
    source[1].cache_dir.mkdir()
    (source[1].cache_dir / '.building-interrupted.sqlite').write_bytes(b'partial')
    assert status(source).status is S.MISSING
    assert build(source).status is S.READY
    assert status(source).status is S.READY
