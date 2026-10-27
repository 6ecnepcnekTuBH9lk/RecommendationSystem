"""Public history API on synthetic canonical files built by the real indexer."""

from contextlib import closing
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import sqlite3
import tracemalloc

import pytest

from Application import customer_history_service as module
from Application.customer_history_index import HistoryIndexStatus
from Application.customer_history_normalization import normalize_action, normalize_order
from Application.customer_history_records import HistoryKind as K, HistoryQuality as Q
from Application.customer_history_service import CustomerHistoryService, HistoryStatus as S
from Application.mindbox.adapters import adapt_customer_merge
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.daily_training_batch import _writer_lock
from tests.test_customer_history_index import (
    STAMP, action, build, component, merges, save_catalog, source as source_fixture, update_rows,
)


UTC = timezone.utc
TIME = datetime.fromisoformat(STAMP.replace("Z", "+00:00"))


@pytest.fixture
def source(tmp_path):
    return source_fixture.__wrapped__(tmp_path)


def query(source, customer="C", **kwargs):
    return CustomerHistoryService(cache_dir=source[1].cache_dir).get_customer_history(customer, raw_root=source[0], **kwargs)


def payload(value):
    """Compare every frozen field except origins, populated by the index builder."""
    if is_dataclass(value):
        return {field.name: payload(getattr(value, field.name)) for field in fields(value) if field.name != "provenance"}
    if isinstance(value, tuple):
        return tuple(payload(item) for item in value)
    return value


def test_typed_roundtrip_all_fields_and_logical_orders(source):
    from tests.test_customer_history_index import order
    favorite = action("favorite")
    favorite["actionTemplate"]["ids"]["systemName"] = "DobavlenieProduktaVSpisokVOperaciiDobavlenie"
    unknown = action("unknown", products=[{"ids": {"website": "00001"}}, {},
                                           {"ids": {"offline1C": "001", "kanzlerKz": "002"}}])
    unknown["actionTemplate"]["ids"]["systemName"] = "Other"
    unknown.pop("creationDateTimeUtc")
    first, changed = order(), order()
    changed["lines"][1]["quantity"] = Decimal("2.50")
    actions = [action(), favorite, unknown, action("foreign", customer="OTHER")]
    update_rows(source, "actions", actions)
    update_rows(source, "orders", [first, changed, deepcopy(first)])
    assert build(source).status is HistoryIndexStatus.READY
    result = query(source)
    assert result.status is S.PARTIAL and result.coverage_complete and result.has_conflicts
    assert {event.source_event_id for event in result.events} == {"action-1", "favorite", "unknown", "order-1"}
    assert {event.kind for event in result.events} == set(K)
    assert isinstance(result.events, tuple) and result.generation
    resolver = CustomerIdResolver(adapt_customer_merge(raw) for raw in merges())
    expected = [normalize_action(raw, resolver) for raw in actions[:-1]] + [normalize_order(raw, resolver) for raw in (first, changed)]
    restored = [variant.event for event in result.events for variant in event.variants]
    assert {event.fingerprint: payload(event) for event in restored} == {event.fingerprint: payload(event) for event in expected}
    ordered = next(event for event in result.events if event.kind is K.ORDER)
    assert len(ordered.variants) == 2 and ordered.occurrence_count == 3
    assert all(len(v.event.details.lines) == 2 for v in ordered.variants)
    assert sum(len(v.origins) for v in ordered.variants) == 3
    assert Q.CURRENCY_UNCONFIRMED in result.quality
    for variant in restored:
        assert variant.event_datetime_utc == TIME and variant.canonical_customer_id == "C"
        assert variant.source_customer_id == "A"
    with pytest.raises(FrozenInstanceError):
        result.events = ()
    with pytest.raises(FrozenInstanceError):
        restored[0].kind = K.ORDER


def test_ready_empty_alias_and_no_lifetime_claim(source):
    assert build(source).status is HistoryIndexStatus.READY
    assert query(source).status is S.READY
    assert query(source, "A").status is S.EMPTY  # API takes canonical, not historical aliases.
    empty = query(source, "UNSEEN")
    assert empty.status is S.EMPTY and not empty.events and len(empty.coverage) == 2
    assert empty.coverage_complete and empty.canonical_customer_id == "UNSEEN"


@pytest.mark.parametrize("customer", [True, None, 1.5, "", " C", [], {}])
def test_invalid_customer_before_any_io(source, customer, monkeypatch):
    monkeypatch.setattr(module.index, "_capture_sources", lambda *a: pytest.fail("invalid request must not read sources"))
    assert query(source, customer).status is S.INVALID_INPUT


@pytest.mark.parametrize("kwargs", [
    {"limit": True}, {"limit": 0}, {"limit": 201}, {"limit": 2.5},
    {"kinds": []}, {"kinds": "VIEW"}, {"kinds": ["OTHER"]}, {"kinds": [True]},
    {"since": datetime(2026, 1, 1)}, {"until": "2026-01-01"}, {"since": TIME, "until": TIME},
    {"since": TIME + timedelta(seconds=1), "until": TIME},
    {"cursor": "bad"}, {"cursor": True}, {"cursor": "a" * 4097},
])
def test_invalid_parameters(source, kwargs):
    assert query(source, **kwargs).status is S.INVALID_INPUT


def test_utc_filter_boundaries_and_all_variants_preserved(source):
    earlier = action("earlier", dateTimeUtc=(TIME - timedelta(microseconds=1)).isoformat())
    later = action("later", dateTimeUtc=(TIME + timedelta(microseconds=1)).isoformat())
    conflict = action("same", dateTimeUtc=TIME.isoformat())
    other = action("same", dateTimeUtc=later["dateTimeUtc"])
    other["actionTemplate"]["ids"]["systemName"] = "Unknown"
    update_rows(source, "actions", [earlier, action(), later, conflict, other])
    assert build(source).status is HistoryIndexStatus.READY
    result = query(source, kinds=[K.VIEW], since=TIME.astimezone(timezone(timedelta(hours=3))), until=TIME + timedelta(microseconds=1))
    assert {e.source_event_id for e in result.events} == {"action-1", "same"}
    uncertain = next(e for e in result.events if e.source_event_id == "same")
    assert uncertain.kind is None and uncertain.event_datetime_utc is None
    assert uncertain.time_conflict and uncertain.kind_conflict and len(uncertain.variants) == 2
    # Two separate variants matching different filters must not satisfy both.
    assert not query(source, kinds=[K.UNCLASSIFIED], since=TIME, until=TIME + timedelta(microseconds=1)).events


def test_owner_conflicts_are_candidate_links_not_assigned_or_lost(source):
    first = action("conflict")
    other = action("conflict", customer="OTHER", dateTimeUtc="2026-01-01T15:00:00Z")
    update_rows(source, "actions", [action("clear"), first, other, action("foreign", customer="OTHER")])
    assert build(source).status is HistoryIndexStatus.READY
    result = query(source, kinds=[K.VIEW])
    assert [e.source_event_id for e in result.events] == ["clear", "conflict"]
    event = result.events[-1]
    assert not event.membership_certain and event.owner_conflict and event.time_conflict
    assert event.canonical_customer_id is None and event.event_datetime_utc is None
    assert event.candidate_customer_ids == ("C", "OTHER")
    assert {v.event.canonical_customer_id for v in event.variants} == {"C", "OTHER"}
    assert result.status is S.PARTIAL
    assert not query(source, kinds=[K.VIEW], since=TIME + timedelta(seconds=1)).events
    assert "conflict" in {e.source_event_id for e in query(source, "OTHER").events}


def collect(source, **kwargs):
    output, pages, cursor = [], [], None
    while True:
        page = query(source, cursor=cursor, **kwargs)
        assert page.status in (S.READY, S.PARTIAL, S.EMPTY)
        output.extend(page.events)
        pages.append(page)
        cursor = page.next_cursor
        if cursor is None:
            return output, pages
        assert len(pages) < 100


def test_keyset_identical_timestamps_and_unknown_dates_no_duplicates(source):
    actions = [action(f"same-{i}") for i in range(123)]
    for i in range(9):
        actions.extend([action(f"unknown-{i}"), action(f"unknown-{i}", customer="OTHER", dateTimeUtc="2026-01-01T13:00:00Z")])
    update_rows(source, "actions", actions)
    assert build(source).status is HistoryIndexStatus.READY
    expected = query(source, limit=200).events
    actual, pages = collect(source, limit=7)
    assert [e.event_key for e in actual] == [e.event_key for e in expected]
    assert len(actual) == len({e.event_key for e in actual}) == 133
    assert len(pages) == 19 and all(len(p.events) == 7 for p in pages)
    assert all(e.event_datetime_utc is None for e in actual[-9:])


@pytest.mark.parametrize("change", ["customer", "kinds", "since", "until", "limit", "generation"])
def test_cursor_binding(source, change):
    update_rows(source, "actions", [action(str(i)) for i in range(4)])
    assert build(source).status is HistoryIndexStatus.READY
    cursor = query(source, limit=1).next_cursor
    kwargs = {"cursor": cursor, "limit": 1}
    customer = "C"
    if change == "customer":
        customer = "OTHER"
    elif change == "generation":
        assert build(source, force=True).status is HistoryIndexStatus.READY
    else:
        kwargs[change] = {"kinds": [K.VIEW], "since": TIME, "until": TIME + timedelta(seconds=1), "limit": 2}[change]
    assert query(source, customer, **kwargs).status is S.STALE_CURSOR


def test_cursor_filter_canonicalization_and_checksum(source):
    update_rows(source, "actions", [action(str(i)) for i in range(4)])
    assert build(source).status is HistoryIndexStatus.READY
    first = query(source, limit=1, kinds=[K.VIEW, K.UNCLASSIFIED], since=TIME)
    second = query(source, limit=1, kinds=["UNCLASSIFIED", "VIEW", "VIEW"], since=TIME.astimezone(timezone(timedelta(hours=3))), cursor=first.next_cursor)
    assert second.status is S.READY
    assert second.events[0].event_key != first.events[0].event_key
    corrupted = first.next_cursor[:-4] + "AAAA"
    assert query(source, cursor=corrupted).status is S.INVALID_INPUT


@pytest.mark.parametrize("damage", ["missing", "stale", "file", "schema", "payload", "origin", "owner", "quality", "sidecar",
                                   "availability", "synthetic", "flag"])
def test_unusable_index_never_returns_events(source, damage):
    assert build(source).status is HistoryIndexStatus.READY
    path = source[1].index_path(raw_root=source[0])
    if damage == "missing":
        path.unlink()
    elif damage == "stale":
        update_rows(source, "actions", [action("new")])
    elif damage == "file":
        path.write_bytes(b"invalid")
    elif damage == "sidecar":
        path.with_name(path.name + "-wal").write_bytes(b"pending")
    else:
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute({"schema": "DROP TABLE product_keys", "payload": "DELETE FROM actions",
                                "origin": "DELETE FROM record_origins", "owner": "UPDATE events SET owner_conflict=1",
                                "availability": "UPDATE actions SET view_available=2",
                                "synthetic": "UPDATE order_lines SET is_synthetic=2",
                                "flag": "UPDATE events SET time_conflict=2",
                                "quality": "UPDATE variants SET quality='[\"UNKNOWN\"]'"}[damage])
    result = query(source)
    assert result.status is S.INDEX_NOT_READY and not result.events and result.generation is None


def test_unavailable_or_inconsistent_source(source):
    assert build(source).status is HistoryIndexStatus.READY
    with _writer_lock(source[0] / "canonical"):
        assert query(source).status is S.SOURCE_UNAVAILABLE
    source[2]["customer_merges"]["until"] = "2025-01-01T00:00:00+00:00"
    save_catalog(source[0], source[2])
    assert query(source).status is S.SOURCE_INCONSISTENT
    (source[0] / "canonical/catalog.json").unlink()
    assert query(source).status is S.SOURCE_UNAVAILABLE


def test_coverage_gaps_missing_component_and_requested_outside_interval(source):
    source[2]["actions"]["2026-01-05"] = component(source[0], "actions", "4", [action("later")],
                                                  since="2026-01-05T00:00:00+00:00", until="2026-01-06T00:00:00+00:00")
    save_catalog(source[0], source[2])
    assert build(source).status is HistoryIndexStatus.READY
    assert query(source, kinds=[K.VIEW]).status is S.PARTIAL
    assert query(source, kinds=[K.ORDER]).status is S.READY
    assert query(source, kinds=[K.VIEW], since=datetime(2026, 1, 1, tzinfo=UTC), until=datetime(2026, 1, 2, tzinfo=UTC)).status is S.READY
    assert query(source, kinds=[K.VIEW], since=datetime(2021, 1, 1, tzinfo=UTC)).status is S.PARTIAL
    source[2]["orders"] = {}
    save_catalog(source[0], source[2])
    assert build(source).status is HistoryIndexStatus.READY
    empty = query(source, kinds=[K.ORDER])
    assert empty.status is S.PARTIAL and not empty.events and not empty.coverage_complete


@pytest.mark.parametrize("kwargs", [
    {"until": datetime(2026, 1, 1, tzinfo=UTC)},
    {"since": datetime(2026, 1, 2, tzinfo=UTC)},
    {"since": datetime(2027, 1, 1, tzinfo=UTC)},
    {"until": datetime(2025, 1, 1, tzinfo=UTC)},
])
def test_half_open_range_wholly_outside_coverage_is_partial(source, kwargs):
    assert build(source).status is HistoryIndexStatus.READY
    result = query(source, kinds=[K.VIEW], **kwargs)
    assert result.status is S.PARTIAL and not result.coverage_complete and not result.events


def test_order_owner_and_date_conflicts_keep_complete_snapshots(source):
    from tests.test_customer_history_index import order
    first, other = order(), order(customer="OTHER")
    other["firstAction"]["dateTimeUtc"] = "2026-01-01T13:00:00Z"
    update_rows(source, "orders", [first, other])
    assert build(source).status is HistoryIndexStatus.READY
    result = query(source, kinds=[K.ORDER])
    assert len(result.events) == 1 and result.status is S.PARTIAL
    event = result.events[0]
    assert event.owner_conflict and event.time_conflict and event.event_datetime_utc is None
    assert not event.membership_certain and len(event.variants) == 2
    assert all(len(v.event.details.lines) == 2 for v in event.variants)
    assert not query(source, kinds=[K.ORDER], since=TIME + timedelta(seconds=1)).events


def test_all_candidates_mirror_common_date_and_schema_v1_is_not_queried(source):
    update_rows(source, "actions", [action("conflict"), action("conflict", customer="OTHER"),
                                    action("dated"), action("dated", dateTimeUtc="2026-01-01T13:00:00Z")])
    assert build(source).status is HistoryIndexStatus.READY
    path = source[1].index_path(raw_root=source[0])
    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute("SELECT COUNT(*) FROM event_customers ec JOIN events e USING(event_key) "
                                  "WHERE ec.event_us IS NOT e.event_us").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM event_customers ec JOIN events e USING(event_key) "
                                  "WHERE e.source_event_id='conflict' AND ec.event_us IS NOT NULL").fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM event_customers ec JOIN events e USING(event_key) "
                                  "WHERE e.source_event_id='dated' AND ec.event_us IS NULL").fetchone() == (1,)
        connection.execute("UPDATE metadata SET value=1 WHERE key='schema_version'")
    assert query(source).status is S.INDEX_NOT_READY
    assert build(source).status is HistoryIndexStatus.READY
    assert query(source).events


def test_mirrored_link_corruption_fails_selected_page_and_explicit_integrity(source):
    assert build(source).status is HistoryIndexStatus.READY
    with closing(sqlite3.connect(source[1].index_path(raw_root=source[0]))) as connection, connection:
        connection.execute("UPDATE event_customers SET event_us=0")
    assert query(source).status is S.INDEX_NOT_READY
    assert source[1].get_status(raw_root=source[0], verify_integrity=True).status is HistoryIndexStatus.CORRUPT


def test_no_source_scan_no_build_readonly_closed_no_pii(source, monkeypatch, caplog, capsys):
    assert build(source).status is HistoryIndexStatus.READY
    path = source[1].index_path(raw_root=source[0])
    before = {p: hashlib.sha256(p.read_bytes()).digest() for p in source[0].rglob("*") if p.is_file()}
    index_before = path.read_bytes()
    monkeypatch.setattr(module.index, "iter_export", lambda *a, **kw: pytest.fail("source scan forbidden"))
    monkeypatch.setattr(module.index.CustomerHistoryIndexService, "build_index", lambda *a, **kw: pytest.fail("auto build forbidden"))
    original = sqlite3.connect
    connections, statements = [], []
    def connect(database, **kwargs):
        assert database.endswith("?mode=ro") and kwargs["uri"]
        connection = original(database, **kwargs)
        connection.set_trace_callback(statements.append)
        connections.append(connection)
        return connection
    monkeypatch.setattr(module.sqlite3, "connect", connect)
    monkeypatch.setattr(module, "_DEFAULT_SERVICE", CustomerHistoryService(cache_dir=source[1].cache_dir))
    result = module.get_customer_history("C", raw_root=source[0])
    assert result.status is S.READY and len(connections) == 1
    assert all("OFFSET" not in sql.upper() for sql in statements)
    assert path.read_bytes() == index_before
    assert before == {p: hashlib.sha256(p.read_bytes()).digest() for p in source[0].rglob("*") if p.is_file()}
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    assert "action-1" not in repr(result) + caplog.text + str(capsys.readouterr())


def test_source_or_index_replaced_midquery_discard_page(source, monkeypatch):
    assert build(source).status is HistoryIndexStatus.READY
    original = module.index._capture_sources
    calls = 0
    def changed(root):
        nonlocal calls
        calls += 1
        result = original(root)
        return replace(result, version="changed") if calls > 1 else result
    monkeypatch.setattr(module.index, "_capture_sources", changed)
    assert query(source).status is S.INDEX_NOT_READY
    monkeypatch.setattr(module.index, "_capture_sources", original)
    original_event = module._logical_event
    def replace_index(connection, row):
        event = original_event(connection, row)
        path = source[1].index_path(raw_root=source[0])
        path.touch()
        return event
    monkeypatch.setattr(module, "_logical_event", replace_index)
    assert query(source).status is S.INDEX_NOT_READY


def test_bounded_page_memory_and_existing_index_plans(source, monkeypatch):
    update_rows(source, "actions", [action(f"large-{i}") for i in range(4000)])
    assert build(source).status is HistoryIndexStatus.READY
    calls = []
    original = module._logical_event
    def counted(connection, row):
        calls.append(row["event_key"])
        return original(connection, row)
    monkeypatch.setattr(module, "_logical_event", counted)
    tracemalloc.start()
    try:
        result = query(source, limit=50)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.status is S.READY and result.next_cursor and len(calls) == 50
    assert peak < 4 * 1024 * 1024
    sql, args = module._page_query(module._request("C", None, None, None, 50), None)
    with closing(sqlite3.connect(source[1].index_path(raw_root=source[0]))) as connection:
        plan = [row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + sql, args)]
    assert any("event_customers_customer" in step for step in plan)
    assert not any("SCAN e" in step or "SCAN ec" in step for step in plan)
    with closing(sqlite3.connect(source[1].index_path(raw_root=source[0]))) as connection:
        instructions = [0]
        connection.set_progress_handler(lambda: instructions.__setitem__(0, instructions[0] + 100) or 0, 100)
        assert len(connection.execute(sql, args).fetchall()) == 51
    assert instructions[0] < 5000  # Previously 38,700 for this 4,000-event fixture.
    decoded = module._decode_cursor(result.next_cursor)
    sql, args = module._page_query(module._request("C", None, None, None, 50), decoded[3:])
    with closing(sqlite3.connect(source[1].index_path(raw_root=source[0]))) as connection:
        instructions = [0]
        connection.set_progress_handler(lambda: instructions.__setitem__(0, instructions[0] + 100) or 0, 100)
        rows = connection.execute(sql, args).fetchall()
        plan = [row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + sql, args)]
    assert len(rows) == 51 and instructions[0] < 15000
    assert any("(event_us,event_key)<(?,?)" in step for step in plan)
