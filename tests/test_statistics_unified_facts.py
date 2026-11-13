from datetime import datetime, timezone
from decimal import Decimal, localcontext
from itertools import permutations

import pytest

from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.records import CustomerMergeRecord
from Application.statistics_unified_facts import (
    OrderKind,
    SnapshotCounts,
    normalize_order_observation,
    reconcile_order,
)
from Application.statistics_unified_policy import EvidenceQuality as Quality, Measured

_DATE = "2025-01-05T03:00:00+03:00"
_MISSING = object()


def _line(status, *, key="line-one", qty=1, price=100, namespace="offline1C", item="A"):
    line = {"product": {"ids": {namespace: item}}}
    if key is not None:
        line["id"] = key
    if status is not _MISSING:
        line["status"] = {"ids": {"externalId": status}}
    if qty is not _MISSING:
        line["quantity"] = qty
    if price is not _MISSING:
        line["priceOfLine"] = price
    return line


def _raw(*lines, owner="private-buyer-ABC", key="private-order-XYZ", date=_DATE, **attrs):
    return {
        "ids": {"mindboxId": key, "retailOrderId": "alternate-order"},
        "customer": {"ids": {"mindboxId": owner}},
        "firstAction": {"dateTimeUtc": date, "channel": {"ids": {"externalId": "web"}, "name": "Web"}},
        "lines": list(lines),
        **attrs,
    }


def _observations(*snapshots, identities=None):
    resolver = identities if identities is not None else CustomerIdResolver()
    return tuple(normalize_order_observation(raw, (0, 1, i), resolver, lambda key: key.value)
                 for i, raw in enumerate(snapshots))


def _fact(*snapshots, predicate=None, identities=None):
    return reconcile_order(_observations(*snapshots, identities=identities),
                           order_sort_rank=4, product_predicate=predicate)


@pytest.mark.parametrize("status,kind,purchase", [
    ("F", OrderKind.COMPLETED_ONLY, True),
    ("sold", OrderKind.COMPLETED_ONLY, True),
    ("CP", OrderKind.PARTIAL_ONLY, True),
    ("delivering", OrderKind.IN_PROGRESS_ONLY, False),
    ("CD", OrderKind.CANCELLED_ONLY, False),
    ("Mindbox-Tech-Cancellation", OrderKind.CANCELLED_ONLY, False),
    ("Return", OrderKind.RETURNED_ONLY, False),
    ("klientskii-vozvrat", OrderKind.RETURNED_ONLY, False),
    ("kurerskii-vozvrat", OrderKind.RETURNED_ONLY, False),
])
def test_single_state_and_candidate(status, kind, purchase):
    fact = _fact(_raw(_line(status)))
    assert fact.current_kind is kind
    assert fact.purchase.observed == Measured(purchase, Quality.KNOWN)
    assert fact.purchase.positions.value == int(purchase)
    assert fact.purchase.fulfilled_units.quality is Quality.UNKNOWN
    rub, kzt = fact.money
    assert (rub.currency, kzt.currency) == ("RUB", "KZT")
    assert rub.source_nominal.value == 100
    assert rub.confirmed_refund.reasons == rub.net_after_refunds.reasons == ("LEDGER_UNAVAILABLE",)
    if fact.has_cancelled:
        assert rub.cancelled_nominal.value == 100
    if fact.has_returned:
        assert rub.returned_nominal.value == 100


@pytest.mark.parametrize("second,kind,positions,flag", [
    ("sold", OrderKind.COMPLETED_ONLY, 2, "has_completed"),
    ("CP", OrderKind.MIXED, 2, "has_partial"),
    ("CD", OrderKind.MIXED, 1, "has_cancelled"),
    ("Return", OrderKind.MIXED, 1, "has_returned"),
    ("delivering", OrderKind.MIXED, 1, "has_in_progress"),
])
def test_mixed_order_is_one_purchase(second, kind, positions, flag):
    fact = _fact(_raw(_line("F"), _line(second, key="line-two")))
    assert fact.current_kind is kind
    assert fact.purchase.observed.value is True
    assert fact.purchase.positions.value == positions
    assert fact.has_completed and getattr(fact, flag)


def test_cp_preserves_subtotal_and_nominal_without_allocating():
    fact = _fact(_raw(_line("F"), _line("CP", key="line-two", qty=5, price=1000)))
    rub = fact.money[0]
    assert fact.purchase.nominal_units.value == 6
    assert fact.purchase.fulfilled_units.value is None
    assert rub.partial_nominal.value == 1000
    assert rub.known_purchase_subtotal.value == 100
    assert rub.recognized_purchase == Measured(None, Quality.UNKNOWN, ("CP_ALLOCATION_UNKNOWN",))
    assert (rub.valued_purchase_positions, rub.unvalued_purchase_positions) == (1, 1)
    cp_return = _fact(_raw(_line("CP"), _line("Return", key="line-two")))
    assert cp_return.current_kind is OrderKind.MIXED
    assert cp_return.has_partial and cp_return.has_returned
    assert cp_return.purchase.positions.value == 1
    assert cp_return.money[0].known_purchase_subtotal.value is None
    assert cp_return.money[0].returned_nominal.value == 100


@pytest.mark.parametrize("value,reason", [
    (0, "ZERO_QUANTITY"),
    (-1, "NEGATIVE_QUANTITY"),
    (Decimal("1.5"), "FRACTIONAL_QUANTITY"),
    (_MISSING, "MISSING_QUANTITY"),
    ("1", "INVALID_QUANTITY"),
    (True, "INVALID_QUANTITY"),
    (Decimal("NaN"), "INVALID_QUANTITY"),
])
def test_bad_quantity_does_not_drop_the_status(value, reason):
    fact = _fact(_raw(_line("F", qty=value)))
    assert fact.current_kind is OrderKind.COMPLETED_ONLY
    assert fact.resolved_lines[0].status_code.value == "F"
    assert fact.purchase.observed.value is False
    assert fact.purchase.positions.value == 0
    assert fact.money[0].source_nominal.value == 100
    assert reason in fact.diagnostics


@pytest.mark.parametrize("price,quality,amount", [
    (Decimal("100.50"), Quality.KNOWN, Decimal("100.50")),
    (0, Quality.KNOWN, Decimal(0)),
    (_MISSING, Quality.UNKNOWN, None),
    (-1, Quality.INVALID, None),
    ("100", Quality.INVALID, None),
    (Decimal("Infinity"), Quality.INVALID, None),
])
def test_price_once_and_purchase_independent_of_money(price, quality, amount):
    fact = _fact(_raw(_line("F", qty=2, price=price)))
    assert fact.purchase.positions.value == 1
    assert fact.purchase.nominal_units.value == 2
    assert fact.money[0].recognized_purchase.quality is quality
    assert fact.money[0].recognized_purchase.value == amount
    assert fact.money[0].known_purchase_subtotal.value == amount


def test_decimal_sums_keep_exact_digits_under_low_precision():
    with localcontext() as context:
        context.prec = 6
        fact = _fact(_raw(_line("F", price=Decimal("1000000000000000000.000001")),
                          _line("sold", key="line-two", price=Decimal("0.000003"))))
    assert fact.money[0].recognized_purchase.value == Decimal("1000000000000000000.000004")


def test_two_currencies_are_separate_and_unknown_currency_preserves_subtotal():
    first = _line("F", price=150)
    mixed = _fact(_raw(first, _line("F", key="line-two", namespace="kanzlerKz", price=200)))
    assert [value.recognized_purchase.value for value in mixed.money] == [150, 200]
    assert mixed.order_currency.reasons == ("MIXED_ORDER_CURRENCY",)
    assert mixed.purchase.positions.value == 2
    unknown = _fact(_raw(first, _line("F", key="line-two", namespace="unrecognized", price=200)))
    assert unknown.order_currency.quality is Quality.UNKNOWN
    assert unknown.money[0].known_purchase_subtotal.value == 150
    assert unknown.money[0].recognized_purchase.value is None
    assert unknown.money[1].recognized_purchase.value is None


def test_conflicting_explicit_currency_cannot_be_counted():
    fact = _fact(_raw(_line("F"), currency="KZT"))
    assert fact.purchase.observed.value is True
    assert fact.order_currency.quality is Quality.CONFLICT
    assert all(entry.recognized_purchase.value is None for entry in fact.money)


@pytest.mark.parametrize("code,reason", [("future", "UNKNOWN_CODE"), (_MISSING, "MISSING_CODE"), (False, "INVALID_CODE")])
def test_bad_status_keeps_raw_price_without_inventing_sale(code, reason):
    fact = _fact(_raw(_line(code)))
    assert fact.current_kind is OrderKind.UNAVAILABLE
    assert fact.purchase.observed.value is False
    assert fact.money[0].source_nominal.value == 100
    assert reason in fact.diagnostics


def test_empty_differs_from_missing_or_malformed_composition():
    empty = _fact(_raw())
    assert empty.current_kind is OrderKind.EMPTY
    assert empty.purchase.positions.value == empty.money[0].recognized_purchase.value == 0
    missing = _raw()
    del missing["lines"]
    fact = _fact(missing)
    assert fact.current_kind is OrderKind.UNAVAILABLE
    assert fact.purchase.observed.quality is Quality.INSUFFICIENT_COVERAGE
    assert fact.money[0].recognized_purchase.value is None
    partial = _fact(_raw(_line("F"), False))
    assert partial.purchase.observed.value is True
    assert partial.purchase.known_positions == 1 and partial.purchase.positions.value is None
    assert partial.money[0].known_purchase_subtotal.value == 100
    assert partial.snapshots.raw_lines == 2


def test_exact_duplicate_business_equivalent_and_permutation():
    first = _raw(_line("F"), _line("sold", key="line-two"))
    same = _raw(_line("sold", key="line-two"), _line("F"), unused_meta="changed")
    observations = _observations(first, first, same)
    expected = reconcile_order(observations, order_sort_rank=4)
    assert expected.snapshots == SnapshotCounts(3, 6, 1, 1, 0)
    assert expected.purchase.evidence_sources == ((0, 1, 0), (0, 1, 1), (0, 1, 2))
    assert expected.purchase.positions.value == 2
    assert expected.money[0].recognized_purchase.value == 200
    for ordered in permutations(observations):
        assert reconcile_order(ordered, order_sort_rank=4) == expected


def test_f_return_conflict_retains_historical_purchase_and_known_price():
    observations = _observations(_raw(_line("F")), _raw(_line("Return", price=120)))
    fact = reconcile_order(observations, order_sort_rank=1)
    assert reconcile_order(reversed(observations), order_sort_rank=1) == fact
    assert fact.current_kind is OrderKind.CONFLICT
    assert fact.purchase.observed.value is True
    assert fact.purchase.positions.value == 1
    assert fact.money[0].known_purchase_subtotal.value == 100
    assert fact.money[0].recognized_purchase.quality is Quality.CONFLICT
    assert fact.money[0].returned_nominal.value is None
    assert fact.money[0].confirmed_refund.value is None
    assert fact.snapshots.conflicting_snapshots == 1


def test_cp_f_for_the_same_line_counts_once_and_never_values_cp():
    fact = _fact(_raw(_line("CP")), _raw(_line("F")))
    assert fact.purchase.observed.value is True
    assert fact.purchase.positions.value == 1
    assert fact.current_kind is OrderKind.CONFLICT
    assert fact.money[0].known_purchase_subtotal.value is None
    assert "CP_ALLOCATION_UNKNOWN" in fact.money[0].recognized_purchase.reasons


def test_price_and_quantity_conflicts_are_independent_of_status():
    fact = _fact(_raw(_line("F", qty=1)), _raw(_line("F", qty=2)))
    assert fact.current_kind is OrderKind.COMPLETED_ONLY
    assert fact.purchase.positions.value == 1
    assert fact.purchase.nominal_units.reasons == ("QUANTITY_CONFLICT",)
    assert fact.money[0].known_purchase_subtotal.value == 100
    assert fact.money[0].recognized_purchase.quality is Quality.CONFLICT
    price = _fact(_raw(_line("F")), _raw(_line("F", price=120)))
    assert price.current_kind is OrderKind.COMPLETED_ONLY
    assert price.money[0].recognized_purchase.reasons == ("PRICE_CONFLICT",)
    assert price.money[0].known_purchase_subtotal.value is None


def test_changed_basket_keeps_only_unambiguous_common_line():
    shared = _line("F", price=100)
    fact = _fact(_raw(shared, _line("F", key="left", price=300)),
                 _raw(shared, _line("F", key="right", price=500)))
    assert fact.current_kind is OrderKind.CONFLICT
    assert fact.purchase.known_positions == 1 and fact.purchase.positions.value is None
    assert fact.purchase.nominal_units.value is None
    assert fact.money[0].known_purchase_subtotal.value == 100
    assert fact.money[0].recognized_purchase.value is None
    assert "LINE_SET_CONFLICT" in fact.diagnostics


def test_local_and_duplicate_line_ids_do_not_invent_cross_snapshot_links():
    first = _raw(_line("F", key=None))
    equivalent = _raw(_line("F", key=None), ignored=True)
    fact = _fact(first, equivalent)
    assert fact.purchase.positions.value == 1 and fact.resolved_lines[0].ordinal == 0
    conflict = _fact(first, _raw(_line("F", key=None, price=200)))
    assert conflict.purchase.known_positions == 0
    assert conflict.purchase.positions.quality is Quality.INSUFFICIENT_COVERAGE
    assert conflict.money[0].known_purchase_subtotal.value is None
    dup = _fact(_raw(_line("F"), _line("F")))
    assert dup.purchase.observed.value is True
    assert dup.purchase.positions.quality is Quality.CONFLICT
    assert dup.purchase.known_positions == 0
    assert "LINE_ID_DUPLICATE" in dup.diagnostics


def test_proxy_dates_are_utc_and_execution_is_unknown():
    fact = _fact(_raw(_line("F")), _raw(_line("F"), date="2025-01-05T00:00:00"))
    assert fact.order_created_at.value == datetime(2025, 1, 5, tzinfo=timezone.utc)
    assert fact.policy.time_basis_id == "ORDER_CREATED_PURCHASE_PROXY_V1"
    assert fact.purchase.executed_at.quality is Quality.UNKNOWN
    conflict = _fact(_raw(_line("F")), _raw(_line("F"), date="2025-01-06T00:00:00Z"))
    assert conflict.order_created_at == Measured(None, Quality.CONFLICT, ("CREATED_TIME_CONFLICT",))
    assert len(conflict.observed_creation_dates) == 2
    assert conflict.purchase.observed.value is True
    assert conflict.money[0].recognized_purchase.value == 100
    for date, quality in ((None, Quality.UNKNOWN), ("not-a-date", Quality.INVALID)):
        incomplete = _fact(_raw(_line("F")), _raw(_line("F"), date=date))
        assert incomplete.order_created_at.quality is quality
        assert incomplete.observed_creation_dates == (fact.order_created_at.value,)
        assert incomplete.purchase.observed.value is True


def test_owner_conflict_blocks_attribution_without_erasing_order():
    a = _raw(_line("F"), owner="private-owner-A")
    b = _raw(_line("F"), owner="private-owner-B")
    fact = _fact(a, b)
    assert fact.customer_key == Measured(None, Quality.CONFLICT, ("OWNER_CONFLICT",))
    assert fact.purchase.observed.value is True
    assert fact.money[0].recognized_purchase.value == 100
    merged = CustomerIdResolver([CustomerMergeRecord(
        merge_id="synthetic-merge", event_datetime_utc=datetime(2025, 1, 5, tzinfo=timezone.utc),
        resulting_customer_id="private-owner-A", merged_customer_ids=("private-owner-B",),
    )])
    assert _fact(a, b, identities=merged).customer_key.value == "private-owner-A"
    assert _fact(a, _raw(_line("F"), owner=None)).customer_key.quality is Quality.UNKNOWN
    for item in (*_observations(a, b), fact, fact.purchase, *fact.resolved_lines):
        assert "private-" not in repr(item)


def test_product_filter_keeps_whole_order_state_and_uses_matching_positive_evidence():
    raw = _raw(_line("F", item="A"), _line("CD", key="line-two", item="B"))
    sliced = _fact(raw, predicate=lambda item: item == "B")
    assert sliced.current_kind is OrderKind.CANCELLED_ONLY
    assert sliced.whole_order_kind is OrderKind.MIXED
    assert sliced.purchase.observed.value is False
    assert sliced.money[0].cancelled_nominal.value == 100
    versions = (_raw(_line("F", item="A")), _raw(_line("Return", item="B")))
    assert _fact(*versions, predicate=lambda item: item == "B").purchase.observed.value is False
    known = _fact(*versions, predicate=lambda item: item == "A")
    assert known.purchase.observed.value is True
    assert known.purchase.positions.value is None
    assert "PRODUCT_CONFLICT" in known.diagnostics
    assert known.money[0].recognized_purchase.value is None
    all_products = _fact(*versions)
    assert all_products.purchase.positions.value == 1


def test_unresolved_product_does_not_create_scoped_zero_or_remove_order_evidence():
    raw = _raw(_line("F", namespace="unsupported"))
    assert _fact(raw).purchase.observed.value is True
    fact = _fact(raw, predicate=lambda item: item == "A")
    assert fact.purchase.observed.quality is Quality.INSUFFICIENT_COVERAGE
    assert fact.purchase.positions.value is None


def test_invalid_groups_and_cancel_fail_safely():
    bad = _observations(_raw(_line("F"), key=None))[0]
    assert bad.order_key.quality is Quality.UNKNOWN
    assert bad.lines[0].status_code.value == "F"
    with pytest.raises(ValueError, match="one valid order key"):
        reconcile_order([bad], order_sort_rank=1)
    raw = _raw(_line("F"))
    with pytest.raises(ValueError, match="one valid order key"):
        reconcile_order(_observations(raw, _raw(_line("F"), key="other")), order_sort_rank=1)

    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        reconcile_order(_observations(raw), order_sort_rank=1, check=cancel)


def test_purchase_code_changes_can_leave_full_recognized_amount_known():
    fact = _fact(_raw(_line("F")), _raw(_line("sold")))
    assert fact.current_kind is OrderKind.CONFLICT
    assert fact.resolved_lines[0].status_code.reasons == ("STATUS_CONFLICT",)
    assert fact.purchase.positions.value == 1
    assert fact.money[0].recognized_purchase == Measured(Decimal(100), Quality.KNOWN)


def test_nonpositive_snapshot_provenance_survives_dedup():
    sold = _raw(_line("F"))
    returned = _raw(_line("Return"))
    observations = _observations(sold, returned, returned)
    fact = reconcile_order(reversed(observations), order_sort_rank=1)
    assert fact.observation_groups == (((0, 1, 0),), ((0, 1, 1), (0, 1, 2)))
    assert fact.purchase.evidence_sources == ((0, 1, 0),)
    assert fact.snapshots == SnapshotCounts(3, 3, 1, 0, 1)


def test_damaged_numbers_cannot_claim_exact_duplicates():
    raw = _raw(_line("F", price=Decimal("NaN")))
    observations = _observations(raw, raw)
    assert all(observation.raw_fingerprint is None for observation in observations)
    fact = reconcile_order(observations, order_sort_rank=1)
    assert fact.snapshots == SnapshotCounts(2, 2, 0, 1, 0)
    assert fact.purchase.positions.value == 1
    assert fact.money[0].recognized_purchase.quality is Quality.INVALID


def test_unqualified_quantity_keeps_the_other_sale_as_a_subtotal():
    fact = _fact(_raw(_line("F"), _line("CP", key="line-two", qty="1", price=900)))
    assert fact.purchase.positions.value == 1
    assert fact.money[0].partial_nominal.value == 900
    assert fact.money[0].known_purchase_subtotal.value == 100
    assert fact.money[0].recognized_purchase.reasons == ("INVALID_QUANTITY",)
