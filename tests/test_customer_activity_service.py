"""Only tiny synthetic canonical exports and derived SQLite databases."""

from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3
from threading import Event

import pytest

from Application import customer_history_index as index
from Application.customer_activity_service import (
    ActivityStatus as S, CustomerActivityService, _ACTIONS_SQL, _ORDERS_SQL,
)
from Application.customer_activity_presentation import activity_content
from tests.test_customer_history_index import (
    action, order, build, source as source_fixture, update_rows, save_catalog,
)


NOW = datetime(2026, 1, 3, tzinfo=timezone.utc)


def purchase(key="o1", *, namespace="offline1C", qty="2.5", price="123.4500", status="F", **kwargs):
    return order(key, lines=[{"id": "l1", "number": 1, "product": {"ids": {namespace: "123456-size 40"}},
                             "quantity": Decimal(qty), "priceOfLine": Decimal(price),
                             "status": {"ids": {"externalId": status}}}], **kwargs)


@pytest.fixture
def history(tmp_path):
    source = source_fixture.__wrapped__(tmp_path)
    service = CustomerActivityService(cache_dir=source[1].cache_dir, clock=lambda: NOW)
    return source, service


def prepared(history, orders=(), actions=()):
    source, service = history
    update_rows(source, "orders", list(orders))
    update_rows(source, "actions", list(actions))
    assert build(source).status is index.HistoryIndexStatus.READY
    return service.get("C", raw_root=source[0])


def test_all_metrics_decimal_logical_events_duplicates_and_full_customer(history):
    p = purchase()
    p["lines"].append({**deepcopy(p["lines"][0]), "id": "l2", "number": 2, "quantity": 3})
    favorite = action("f")
    favorite["actionTemplate"]["ids"]["systemName"] = "DobavlenieProduktaVSpisokVOperaciiDobavlenie"
    acts = [action(f"v{i}") for i in range(65)] + [favorite, action("foreign", customer="FOREIGN")]
    result = prepared(history, [p, deepcopy(p), purchase("o2", qty="1", price="0"), purchase("other", customer="FOREIGN")], acts + [deepcopy(acts[0])])
    assert result.status is S.READY and result.complete
    assert result.orders == 2 and result.units == Decimal("6.5")
    assert result.money[0].amount == Decimal("246.9000") and result.money[0].average_check == Decimal("123.4500")
    assert result.money[0].orders == 2
    assert result.views == 65 and result.favorites == 1 and result.unique_viewed_products == 1
    assert result.days_since_purchase == 0 and result.purchase_as_of_utc.date().isoformat() == "2026-01-01"
    assert result.calculated_at_utc == NOW and result.canonical_customer_id == "C"
    assert dict(result.diagnostics)["logical_orders"] == 2
    with pytest.raises(FrozenInstanceError):
        result.orders = 99
    fields = activity_content(result).fields
    assert fields[2].value == "246,90 ₽" and fields[3].value == "123,45 ₽"
    assert "01.01.2026" in fields[6].note


@pytest.mark.parametrize("status,bought", [("CP", True), ("delivering", True), ("F", True),
                                         ("otmenen", False), ("Mindbox-Tech-Cancellation", False),
                                         ("out-of-stock", False), ("Return", False), ("klientskii-vozvrat", False)])
def test_existing_purchase_semantics(history, status, bought):
    result = prepared(history, [purchase(status=status)])
    assert result.orders == int(bought)
    assert result.units == (Decimal("2.5") if bought else 0)
    assert result.days_since_purchase == (0 if bought else None)
    assert result.status is S.READY
    if not bought:
        assert activity_content(result).fields[6].message == "Нет покупок"


@pytest.mark.parametrize("change,metric", [("quantity", "units"), ("priceOfLine", "money")])
def test_conflicting_snapshots_do_not_choose_a_winner(history, change, metric):
    first, second = purchase(), purchase()
    second["lines"][0][change] = 10
    result = prepared(history, [first, second, deepcopy(first)], [action()])
    assert result.orders == 1 and result.views == 1
    assert getattr(result, metric) is None
    assert result.last_purchase_utc is not None
    assert result.units is not None if metric != "units" else result.money is not None


@pytest.mark.parametrize("status,expected", [("Return", 1), ("CP", 1), ("otmenen", None), ("CD", None), ("N", None)])
def test_conflicting_statuses_and_returns_preserve_gross_purchase_semantics(history, status, expected):
    result = prepared(history, [purchase(), purchase(status=status)], [action()])
    assert result.orders == expected and result.views == 1
    if expected:
        assert result.units == Decimal("2.5") and result.money[0].amount == Decimal("123.4500")
    else:
        assert result.units is None and result.money is None and result.days_since_purchase is None


def test_partial_cancellation_counts_order_once(history):
    p = purchase()
    cancelled = deepcopy(p["lines"][0])
    cancelled.update(id="l2", number=2, quantity=1000, priceOfLine=1000000)
    cancelled["status"]["ids"]["externalId"] = "otmenen"
    p["lines"].append(cancelled)
    result = prepared(history, [p])
    assert result.orders == 1 and result.units == Decimal("2.5")
    assert result.money[0].amount == Decimal("123.45")


@pytest.mark.parametrize("namespace,currency", [("offline1C", "RUB"), ("kanzlerKz", "KZT")])
def test_currency_sums_use_line_amount_not_quantity_or_delivery(history, namespace, currency):
    result = prepared(history, [purchase(namespace=namespace, qty="5", price="10.12345678901234567890")])
    assert result.money[0].currency == currency
    assert result.money[0].amount == Decimal("10.12345678901234567890")
    assert result.money[0].average_check == result.money[0].amount


def test_two_currencies_and_correct_order_denominators(history):
    result = prepared(history, [purchase("r1", price="100"), purchase("r2", price="300"), purchase("k1", namespace="kanzlerKz", price="900")])
    rub, kzt = result.money
    assert (rub.orders, rub.amount, rub.average_check) == (2, 400, 200)
    assert (kzt.orders, kzt.amount, kzt.average_check) == (1, 900, 900)
    fields = activity_content(result).fields
    assert fields[2].value == "400 ₽\n900 ₸"


@pytest.mark.parametrize("mixed", [False, True])
def test_unknown_or_mixed_currency_only_invalidates_money(history, mixed):
    p = purchase(namespace="website" if not mixed else "offline1C")
    if mixed:
        other = deepcopy(p["lines"][0])
        other.update(id="l2", number=2)
        other["product"]["ids"] = {"kanzlerKz": "654321-size"}
        p["lines"].append(other)
    result = prepared(history, [p], [action()])
    assert result.orders == 1 and result.units is not None and result.views == 1
    assert result.money is None and result.days_since_purchase == 0
    assert any(i.reason == "currency" for i in result.issues)


@pytest.mark.parametrize("qty,price,orders,units,money", [
    ("0", "10", None, None, False), ("-1", "10", None, None, False),
    ("1.25", "10", 1, Decimal("1.25"), True), ("1", "-10", 1, Decimal(1), False),
])
def test_invalid_and_fractional_values_do_not_create_fake_totals(history, qty, price, orders, units, money):
    result = prepared(history, [purchase(qty=qty, price=price)], [action()])
    assert result.orders == orders and result.units == units
    assert (result.money is not None) is money and result.views == 1


def test_empty_customer_is_observed_zero_not_missing_source(history):
    result = prepared(history)
    assert result.status is S.EMPTY and result.orders == result.views == result.favorites == 0
    assert result.units == 0 and result.money == () and result.days_since_purchase is None
    assert result.complete


@pytest.mark.parametrize("products,unique", [([], 0), ([{}], None),
    ([{"ids": {"website": "123456"}}], None),
    ([{"ids": {"offline1C": "123456-size"}}, {"ids": {"kanzlerKz": "123456-other"}}], 1),
    ([{"ids": {"offline1C": "123456-size"}}, {"ids": {"offline1C": "654321-size"}}], 2)])
def test_logical_views_and_product_occurrences_are_different_metrics(history, products, unique):
    result = prepared(history, actions=[action(products=products)])
    assert result.views == 1 and result.unique_viewed_products == unique and result.favorites == 0


def test_action_variants_count_once_but_product_conflict_is_not_hidden(history):
    a, b = action(), action()
    b["products"] = [{"ids": {"offline1C": "654321-size"}}]
    result = prepared(history, actions=[a, b, deepcopy(a)])
    assert result.views == 1 and result.unique_viewed_products is None and result.orders == 0


def test_owner_and_kind_conflicts_affect_only_relevant_metrics(history):
    a, b = action(), action(customer="FOREIGN")
    result = prepared(history, [purchase()], [a, b])
    assert result.orders == 1 and result.views is None and result.favorites == 0
    b = action()
    b["actionTemplate"]["ids"]["systemName"] = "DobavlenieProduktaVSpisokVOperaciiDobavlenie"
    result = prepared(history, [purchase()], [a, b])
    assert result.orders == 1 and result.views is None and result.favorites is None


def test_unclassified_is_not_a_view_or_favorite(history):
    a = action()
    a["actionTemplate"]["ids"]["systemName"] = "Unclassified"
    result = prepared(history, actions=[a])
    assert result.views == result.favorites == result.unique_viewed_products == 0


@pytest.mark.parametrize("missing", ["actions", "orders"])
def test_independent_coverage_without_a_component(history, missing):
    source, service = history
    update_rows(source, "orders", [purchase()])
    update_rows(source, "actions", [action()])
    source[2][missing] = {}
    save_catalog(source[0], source[2])
    assert build(source).status is index.HistoryIndexStatus.READY
    result = service.get("C", raw_root=source[0])
    assert result.views is None if missing == "actions" else result.orders is None
    assert result.orders is not None if missing == "actions" else result.views is not None


def test_gap_in_actions_does_not_hide_order_metrics(history, monkeypatch):
    source, service = history
    prepared(history, [purchase()], [action()])
    capture = index._capture_sources
    def sources(root):
        original = capture(root)
        from dataclasses import replace
        pieces = []
        for c in original.components:
            if c.name == "actions":
                pieces.extend((replace(c, until=c.since+timedelta(hours=1)), replace(c, since=c.until-timedelta(hours=1))))
            else:
                pieces.append(c)
        return replace(original, components=tuple(pieces))
    monkeypatch.setattr(index, "_capture_sources", sources)
    result = service.get("C", raw_root=source[0])
    assert result.orders == 1 and result.views is None and not result.actions_coverage_complete


def test_snapshot_freshness_cancellation_missing_and_corrupt_index(history, monkeypatch):
    source, service = history
    result = prepared(history, [purchase()])
    stop = Event(); stop.set()
    assert service.get("C", raw_root=source[0], cancelled=stop.is_set).status is S.CANCELLED
    path = source[1].index_path(raw_root=source[0])
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE metadata SET value=? WHERE key='generation'", ("e"*32,))
    assert service.validate(result, raw_root=source[0]).status is S.STALE
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DROP TABLE order_lines")
    assert service.get("C", raw_root=source[0]).status is S.ERROR
    path.unlink()
    assert service.get("C", raw_root=source[0]).status is S.UNAVAILABLE


def test_constant_query_count_indexed_selection_and_closed_connections(history, monkeypatch):
    source, service = history
    prepared(history, [purchase(f"o{i}") for i in range(70)], [action(f"a{i}") for i in range(80)])
    reader, queries, connections = index._read_index, [], []
    @contextmanager
    def tracked(path):
        with reader(path) as db:
            connections.append(db)
            db.set_trace_callback(queries.append)
            yield db
    monkeypatch.setattr(index, "_read_index", tracked)
    result = service.get("C", raw_root=source[0])
    assert result.orders == 70 and result.views == 80
    assert dict(result.diagnostics)["logical_orders"] == 70
    assert sum("FROM event_customers" in q for q in queries) == 2
    with reader(source[1].index_path(raw_root=source[0])) as db:
        for sql in (_ORDERS_SQL, _ACTIONS_SQL):
            plan = [r[3] for r in db.execute("EXPLAIN QUERY PLAN " + sql, ("C",))]
            assert any("event_customers_customer" in p for p in plan)
            assert not any("SCAN ec" in p or "SCAN e " in p for p in plan)
    for db in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")


@pytest.mark.parametrize("stamp,days", [("2026-01-03T00:00:00Z", 0), ("2026-01-02T23:59:59Z", 1),
                                        ("2025-12-31T23:59:59Z", 3), ("2025-12-03T00:00:00Z", 31)])
def test_calendar_recency_utc_not_elapsed_hours(history, stamp, days, monkeypatch):
    source, service = history
    p = purchase()
    p["firstAction"]["dateTimeUtc"] = stamp
    prepared(history, [p])
    inspect = index._inspect_connection
    def coverage(*args, **kwargs):
        # Isolate recency from the daily-catalog fixture's one-day coverage.
        from dataclasses import replace
        ready = inspect(*args, **kwargs)
        spans = tuple(replace(c, since_utc=datetime(2025, 12, 1, tzinfo=timezone.utc),
                              until_utc=datetime(2026, 2, 1, tzinfo=timezone.utc)) if c.component == "orders" else c
                      for c in ready.coverage)
        return replace(ready, coverage=spans)
    monkeypatch.setattr(index, "_inspect_connection", coverage)
    result = service.get("C", raw_root=source[0])
    assert result.days_since_purchase == days and result.purchase_as_of_utc == NOW


def test_future_order_date_never_becomes_a_negative_recency(history):
    p = purchase()
    p["firstAction"]["dateTimeUtc"] = "2026-01-04T00:00:00Z"
    result = prepared(history, [p], [action()])
    assert result.days_since_purchase is None and result.orders is None
    assert result.views == 1 and any(i.reason == "purchase_time" for i in result.issues)


def test_missing_order_time_and_quantity_do_not_mask_actions(history):
    source, service = history
    prepared(history, [purchase()], [action()])
    with closing(sqlite3.connect(source[1].index_path(raw_root=source[0]))) as db, db:
        db.execute("UPDATE order_lines SET quantity='not-a-decimal'")
    result = service.get("C", raw_root=source[0])
    assert result.orders is None and result.units is None and result.views == 1


def test_source_change_during_snapshot_discards_all_numbers(history, monkeypatch):
    source, service = history
    prepared(history, [purchase()], [action()])
    capture = index._capture_sources
    calls = []
    def changed(root):
        from dataclasses import replace
        original = capture(root)
        calls.append(1)
        return original if len(calls) == 1 else replace(original, version="changed")
    monkeypatch.setattr(index, "_capture_sources", changed)
    result = service.get("C", raw_root=source[0])
    assert result.status is S.STALE and result.orders is None and result.views is None


def test_cancel_during_stream_closes_connection_and_exposes_no_partial_counts(history, monkeypatch):
    source, service = history
    prepared(history, [purchase(f"o{i}") for i in range(70)])
    reader, connections, stop = index._read_index, [], Event()
    @contextmanager
    def interrupted(path):
        with reader(path) as db:
            connections.append(db)
            db.set_trace_callback(lambda sql: stop.set() if "FROM event_customers" in sql else None)
            yield db
    monkeypatch.setattr(index, "_read_index", interrupted)
    result = service.get("C", raw_root=source[0], cancelled=stop.is_set)
    assert result.status is S.CANCELLED and result.orders is None
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")


def test_very_long_decimal_totals_remain_exact(history):
    amount = "1234567890.123456789012345678901234567890123456789012345678901234567890"
    result = prepared(history, [purchase("a", price=amount), purchase("b", price=amount)])
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec = 100
        expected = Decimal(amount) * 2
    assert result.money[0].amount == expected


def test_streamed_python_memory_and_no_raw_export_reads(history, monkeypatch):
    import tracemalloc
    source, service = history
    prepared(history, [purchase()], [action(f"view-{n}") for n in range(2000)])
    monkeypatch.setattr(index, "iter_export", lambda *a, **kw: pytest.fail("KPI must not read raw exports"))
    tracemalloc.start()
    try:
        result = service.get("C", raw_root=source[0])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.views == 2000 and result.unique_viewed_products == 1
    assert peak < 4 * 1024 * 1024
    print(f"activity_synthetic views=2000 python_peak_bytes={peak}")
