"""One-order operations and synthetic existing-driver/index parity only."""

from contextlib import closing
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from Application import dataset_statistics as statistics
from Application import statistics_purchase_facts as facts
from Application.analysis_filter import AnalysisFilter
from Application.interactions import InteractionBuilder
from Application.mindbox.adapters._common import AdapterError, timestamp
from Application.mindbox.records import ProductKey
from Application.product_resolution import ProductResolver
from Application.statistics_purchase_policy import StatusGroup
from Application.statistics_query import IndexedInput
from tests.test_statistics_purchase_policy import line as line_fixture
from tests.test_statistics_index import dataset as indexed_fixture, build, query
from tests.test_dataset_statistics import order, action, DEFAULT_SELECTION, put_entry, save
from tests.test_analysis_filter import reference, sources


line = line_fixture
AT = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)


def fact(lines, **kwargs):
    options = dict(identity_space="REFERENCE:synthetic", customer_id="canonical", order_id="order",
                   order_datetime_utc=AT)
    options.update(kwargs)
    return facts.build_purchase_order_fact(lines, **options)


@pytest.mark.parametrize("statuses,strict_count,extended_count,returns", [
    (("F",), 1, 1, 0), (("sold",), 1, 1, 0), (("CP",), 0, 1, 0),
    (("delivering",), 0, 0, 0), (("F", "CP"), 1, 2, 0),
    (("F", "Return"), 1, 1, 1), (("Return",), 0, 0, 1),
    (("F", "F", "F"), 3, 3, 0), (("unknown", "CD"), 0, 0, 0), ((), 0, 0, 0),
])
def test_policy_summaries_count_one_order_not_lines(line, statuses, strict_count, extended_count, returns):
    result = fact(replace(line, line_status=status) for status in statuses)
    assert result.strict.line_count == strict_count
    assert result.extended.line_count == extended_count
    assert int(result.strict.has_purchase) == int(strict_count > 0)
    assert int(result.extended.has_purchase) == int(extended_count > 0)
    assert result.strict.units == Decimal(3) * strict_count
    assert result.extended.units == Decimal(3) * extended_count
    assert result.strict.amount == (Decimal("10.25") * strict_count if strict_count else None)
    assert result.extended.amount == (Decimal("10.25") * extended_count if extended_count else None)
    assert result.return_line_count == returns
    assert strict_count <= extended_count
    if not strict_count:
        assert result.strict.currency is None and not result.strict.monetary_available
        assert not result.strict.monetary_issues


@pytest.mark.parametrize("quantity,accepted", [
    (Decimal(0), False), (Decimal(1), True), (Decimal("3.000"), True),
    (Decimal("1.5"), False), (Decimal(-1), False), ("3", False), (None, False),
    (True, False), (Decimal("NaN"), False), (Decimal("Infinity"), False), (1.0, True),
])
def test_quantity_uses_existing_analytical_contract(line, quantity, accepted):
    result = fact([replace(line, quantity=quantity)])
    assert result.strict.has_purchase == result.extended.has_purchase == accepted
    assert result.strict.units == (Decimal(str(quantity)) if accepted else Decimal(0))


@pytest.mark.parametrize("namespace,currency", [("offline1C", "RUB"), ("kanzlerKz", "KZT")])
@pytest.mark.parametrize("price", [Decimal("0"), Decimal("10.25")])
def test_whole_line_price_and_zero_price_without_delivery_or_order_total(line, namespace, currency, price):
    result = fact([replace(line, product=ProductKey(namespace, "unresolved"), price_of_line=price,
                           delivery_cost=Decimal(900), order_total_price=Decimal(9999))])
    assert result.strict.currency == currency
    assert result.strict.amount == price  # quantity is three; no multiplication.
    assert result.strict.monetary_available and result.strict.has_purchase


@pytest.mark.parametrize("namespaces,currency,issue", [
    (("offline1C", "kanzlerKz"), "mixed", facts.MonetaryIssue.MIXED_CURRENCY),
    (("unsupported",), "unknown", facts.MonetaryIssue.UNKNOWN_CURRENCY),
    (("offline1C", "unsupported"), "unknown", facts.MonetaryIssue.UNKNOWN_CURRENCY),
    ((None,), "unknown", facts.MonetaryIssue.UNKNOWN_CURRENCY),
])
def test_unavailable_currency_keeps_purchase(line, namespaces, currency, issue):
    lines = [replace(line, product=ProductKey(ns, "unknown") if ns else None) for ns in namespaces]
    result = fact(lines)
    assert result.strict.has_purchase and result.strict.line_count == len(lines)
    assert result.strict.currency == currency and result.strict.amount is None
    assert result.strict.monetary_issues == frozenset({issue})
    assert not result.strict.monetary_available


@pytest.mark.parametrize("price", [None, "10", True, Decimal(-1), Decimal("NaN"), Decimal("Infinity"), object()])
def test_bad_price_is_unavailable_not_zero_and_does_not_remove_purchase(line, price):
    result = fact([line, replace(line, price_of_line=price)])
    assert result.strict.line_count == 2 and result.strict.units == 6
    assert result.strict.has_purchase and not result.strict.monetary_available
    assert result.strict.currency == "RUB" and result.strict.amount is None
    assert result.strict.monetary_issues == frozenset({facts.MonetaryIssue.INVALID_PRICE})


def test_policy_specific_currency_and_price_validity(line):
    partial = replace(line, line_status="CP", product=ProductKey("kanzlerKz", "unknown"), price_of_line=None)
    result = fact([line, partial])
    assert result.strict.currency == "RUB" and result.strict.amount == Decimal("10.25")
    assert result.extended.has_purchase and result.extended.currency == "mixed"
    assert result.extended.amount is None
    assert result.extended.monetary_issues == frozenset({facts.MonetaryIssue.MIXED_CURRENCY,
                                                       facts.MonetaryIssue.INVALID_PRICE})


def test_one_pass_no_builder_resolution_io_or_input_mutation(line, monkeypatch):
    lines = [line, replace(line, line_status="CP"), replace(line, line_status="Return"),
             replace(line, line_status="klientskii-vozvrat"), replace(line, line_status="kurerskii-vozvrat")]
    original = [asdict(item) for item in lines]
    visited = []

    class Once:
        def __init__(self):
            self.started = False

        def __iter__(self):
            assert not self.started
            self.started = True
            yield from lines

    eligibility = facts.eligible_purchase_policies

    def counted(status, quantity):
        visited.append(status)
        return eligibility(status, quantity)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected I/O, resolution or production builder call")

    with monkeypatch.context() as patch:
        patch.setattr(facts, "eligible_purchase_policies", counted)
        patch.setattr(InteractionBuilder, "from_order_line", forbidden)
        patch.setattr(ProductResolver, "resolve", forbidden)
        patch.setattr("builtins.open", forbidden)
        patch.setattr(os, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        result = fact(Once())
    assert visited == [item.line_status for item in lines]
    assert result.return_line_count == 3
    assert dict(result.status_groups)[StatusGroup.RETURNED] == 3
    assert [asdict(item) for item in lines] == original
    with pytest.raises(FrozenInstanceError):
        result.strict.amount = Decimal(999)
    with pytest.raises(FrozenInstanceError):
        result.customer_id = "changed"


def test_no_qt_import_or_file_creation_on_import():
    script = """
import os, sys
def guard(event, args):
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+') or
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            raise AssertionError('Unexpected write on import')
    if event in ('os.mkdir', 'os.remove', 'os.rename', 'sqlite3.connect'):
        raise AssertionError('Unexpected side effect on import')
sys.addaudithook(guard)
from Application import statistics_purchase_facts
assert not any(k.startswith(('PyQt', 'qt_material')) for k in sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("count", [0, 5])
def test_cancellation_propagates_without_partial_fact(line, count):
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == (1 if not count else 3):
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        fact((line for _ in range(count)), check=check)


@pytest.mark.parametrize("overrides", [
    {"customer_id": " PRIVATE "}, {"order_id": True}, {"identity_space": ""},
    {"order_datetime_utc": AT.replace(tzinfo=None)},
    {"order_datetime_utc": AT.astimezone(timezone(timedelta(hours=3)))},
])
def test_invalid_context_fails_without_echoing_ids(line, overrides):
    with pytest.raises(ValueError) as caught:
        fact([line], **overrides)
    assert "PRIVATE" not in str(caught.value)


def test_private_id_spaces_are_explicit_and_not_in_repr(line):
    reference_fact = fact([line], customer_id="PRIVATE_CUSTOMER", order_id="PRIVATE_ORDER")
    indexed_fact = fact([line], identity_space="INDEXED:synthetic-generation", customer_id=1, order_id=1)
    assert reference_fact.identity_space != indexed_fact.identity_space
    assert reference_fact.strict == indexed_fact.strict
    assert "PRIVATE_CUSTOMER" not in repr(reference_fact) and "PRIVATE_ORDER" not in repr(reference_fact)


@pytest.mark.parametrize("first_status,quantity", [("Return", 1), ("F", 0)])
def test_caller_selects_first_snapshot_before_policy_not_first_purchasable_snapshot(first_status, quantity):
    from Application.mindbox.adapters import adapt_order
    from Application.mindbox.identity import CustomerIdResolver
    from Application.mindbox.order_dedup import OrderSnapshots
    first = order([first_status], "same")
    first["lines"][0]["quantity"] = quantity
    conflict = order(["F"], "same")
    snapshots, selected = OrderSnapshots(), []
    identities = CustomerIdResolver()
    for raw in (first, deepcopy(first), conflict):
        if snapshots.accept(raw, diagnose=True):
            lines = adapt_order(raw, identities)
            selected.append(fact(lines, customer_id=lines[0].customer_id,
                                 order_id=raw["ids"]["mindboxId"],
                                 order_datetime_utc=timestamp(raw, "firstAction.dateTimeUtc")))
    assert len(selected) == 1
    assert not selected[0].strict.has_purchase and not selected[0].extended.has_purchase
    assert snapshots.identical == snapshots.conflicting == 1


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    data = indexed_fixture.__wrapped__(tmp_path, monkeypatch)
    reference(data)
    first = order(["F", "CP", "Return", "F"], "first")
    first["customer"]["ids"]["mindboxId"] = "old"
    first["lines"][0]["product"]["ids"] = {"offline1C": "999999-size"}
    first["lines"][3]["quantity"] = 1.25  # Existing adapted input excludes fractions in BOTH modes.
    conflict = deepcopy(first)
    conflict["customer"]["ids"]["mindboxId"] = "phantom"
    conflict["lines"][0]["status"]["ids"]["externalId"] = "Return"
    second = order(["sold", "F"], "second")
    second["lines"][0]["priceOfLine"] = 0
    second["lines"][1]["product"]["ids"] = {"kanzlerKz": "000002-size"}
    returned = order(["Return", "delivering"], "returned")
    returned["customer"]["ids"]["mindboxId"] = "other"
    empty = order([], "empty")
    sources(data, [action(DEFAULT_SELECTION.view_action_system_names[0], ["000001-size"])],
            [first, deepcopy(first), conflict, second, returned, empty])
    return data


def test_existing_driver_adapts_once_before_products_and_same_facts_in_both_modes(dataset, monkeypatch):
    build(dataset)
    root, catalog, _ = dataset
    for selection in (AnalysisFilter(), AnalysisFilter("2026-01-01", "2026-01-01"),
                      AnalysisFilter(collections=("missing",))):
        before = statistics.calculate_statistics(raw_root=root, catalog_path=catalog, analysis_filter=selection,
                                                 calculated_at=AT)
        reference_facts, indexed_facts, reads = [], [], []
        adapt = statistics.adapt_order
        indexed_order = IndexedInput.order
        read = statistics.iter_export

        def record_reference(raw, identities, **kwargs):
            lines = adapt(raw, identities, **kwargs)
            reference_facts.append(facts.build_purchase_order_fact(
                lines, identity_space="REFERENCE:synthetic", order_id=raw["ids"]["mindboxId"],
                customer_id=identities.resolve(raw["customer"]["ids"]["mindboxId"]),
                order_datetime_utc=timestamp(raw, "firstAction.dateTimeUtc")))
            return lines

        def record_indexed(source, raw):
            lines = indexed_order(source, raw)
            assert all(not hasattr(item, "order_id") for item in lines)
            indexed_facts.append(facts.build_purchase_order_fact(
                lines, identity_space="INDEXED:synthetic", order_id=raw["ids"]["mindboxId"],
                customer_id=raw["customer"]["ids"]["mindboxId"], order_datetime_utc=raw["_date"]))
            return lines

        def counted_read(name, **kwargs):
            reads.append(name)
            yield from read(name, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(statistics, "adapt_order", record_reference)
            patch.setattr(statistics, "iter_export", counted_read)
            actual_reference = statistics.calculate_statistics(raw_root=root, catalog_path=catalog,
                                                                analysis_filter=selection, calculated_at=AT)
        assert reads.count("orders") == 1
        assert asdict(actual_reference) == asdict(before)
        with monkeypatch.context() as patch:
            patch.setattr(IndexedInput, "order", record_indexed)
            patch.setattr(statistics, "iter_export", lambda *a, **kw: pytest.fail("raw read in INDEXED"))
            actual_indexed = query(dataset, selection, calculated_at=AT)
        assert asdict(actual_indexed) == asdict(before)
        assert len(reference_facts) == len(indexed_facts) == 4  # One per selected snapshot, not per policy.
        assert reference_facts[0].customer_id == reference_facts[1].customer_id == "new"
        assert reference_facts[0].strict.line_count == 1 and reference_facts[0].extended.line_count == 2
        assert reference_facts[2].strict.has_purchase is False  # Return + delivering.
        customers, reverse_customers = {}, {}
        for expected, actual in zip(reference_facts, indexed_facts):
            # Published SID maps are intentionally absent. Verify the identity
            # partition in BOTH directions, never compare raw IDs to SIDs.
            assert customers.setdefault(expected.customer_id, actual.customer_id) == actual.customer_id
            assert reverse_customers.setdefault(actual.customer_id, expected.customer_id) == expected.customer_id
            assert (actual.strict, actual.extended, actual.status_groups, actual.order_datetime_utc) == (
                expected.strict, expected.extended, expected.status_groups, expected.order_datetime_utc)
        assert len({item.order_id for item in reference_facts}) == len({item.order_id for item in indexed_facts}) == 4


def test_orders_outside_actions_coverage_remain_available_without_period_normalization(dataset, monkeypatch):
    root, catalog, data = dataset
    first, later = order(["F"], "first"), order(["CP"], "later")
    later["firstAction"]["dateTimeUtc"] = "2026-01-02T12:00:00Z"
    data["manual_interactions"] = None
    data["actions"] = {"2026-01-01": put_entry(root, "actions", [], 111)}
    data["orders"] = {"2026-01-01": put_entry(root, "orders", [first], 112),
                      "2026-01-02": put_entry(root, "orders", [later], 113,
                                            "2026-01-02T00:00:00+00:00", "2026-01-03T00:00:00+00:00")}
    save(root, data)
    built = build(dataset)
    from Application.statistics_index import open_generation
    from Application.statistics_slices import capture_sources
    provenance, selection = capture_sources(root, catalog, AnalysisFilter())
    db, _ = open_generation(root / "analytics", provenance)
    with closing(db):
        source = IndexedInput(db, provenance, selection, lambda: None, {"sqlite_seconds": 0., "source_headers": 0})
        result = [facts.build_purchase_order_fact(source.order(raw), identity_space="INDEXED:" + built["generation"],
                  customer_id=raw["customer"]["ids"]["mindboxId"], order_id=raw["ids"]["mindboxId"],
                  order_datetime_utc=raw["_date"]) for raw in source.records("orders", [True])]
    assert len(result) == 2
    assert result[1].order_datetime_utc.day == 2 and result[1].extended.has_purchase
    assert not AnalysisFilter("2026-01-01", "2026-01-01").contains(result[1].order_datetime_utc)


def test_invalid_source_price_still_fails_original_adapters_and_indexed_path(dataset):
    from Application.statistics_index import open_generation
    from Application.statistics_slices import capture_sources
    root, catalog, data = dataset
    broken = order(["F"], "bad-price")
    del broken["lines"][0]["priceOfLine"]
    path = root / data["manual_interactions"]["orders"]["directory"] / "orders_part_001.json"
    path.write_text(json.dumps({"orders": [broken]}), encoding="utf-8")
    with pytest.raises(AdapterError):
        statistics.calculate_statistics(raw_root=root, catalog_path=catalog)
    build(dataset)
    provenance, selection = capture_sources(root, catalog, AnalysisFilter())
    db, _ = open_generation(root / "analytics", provenance)
    with closing(db):
        source = IndexedInput(db, provenance, selection, lambda: None, {"sqlite_seconds": 0., "source_headers": 0})
        raw = next(source.records("orders", [True]))
        with pytest.raises(AdapterError):
            source.order(raw)
