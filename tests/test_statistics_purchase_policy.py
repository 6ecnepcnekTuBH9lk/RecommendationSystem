"""Synthetic business-contract checks; production rules stay independent."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, localcontext

import pytest

from Application import statistics_purchase_policy as policy
from Application.interactions import InteractionBuilder, PURCHASE_LINE_STATUSES
from Application.mindbox.records import OrderLineRecord, ProductKey
from Application.mindbox.selection import DEFAULT_SELECTION


STRICT = policy.STRICT_COMPLETED
EXTENDED = policy.EXTENDED_COMPLETED


@pytest.fixture
def line():
    return OrderLineRecord(
        order_id="synthetic-order", order_datetime_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        channel_external_id="synthetic", channel_name="Synthetic", source_customer_id="synthetic-source",
        customer_id="synthetic-canonical", line_id="synthetic-line", line_number=1,
        product=ProductKey("offline1C", "unresolved-product"), product_name=None,
        quantity=Decimal("3.000"), base_price_per_item=None, price_of_line=Decimal("10.25"), line_status="F",
    )


@pytest.mark.parametrize("status,expected", [
    ("F", {STRICT, EXTENDED}), ("sold", {STRICT, EXTENDED}), ("CP", {EXTENDED}),
    ("delivering", set()), ("Return", set()), ("unknown", set()),
])
def test_exact_policy_membership(status, expected):
    actual = policy.eligible_purchase_policies(status, Decimal("1"))
    assert actual == frozenset(expected)
    for selected in (STRICT, EXTENDED):
        assert policy.accepts_purchase_line(status, 1, policy=selected) == (selected in expected)


@pytest.mark.parametrize("codes,group", [
    ("F sold", policy.StatusGroup.COMPLETED),
    ("CP", policy.StatusGroup.PARTIALLY_COMPLETED),
    ("delivering Оплачен prepayed send-to-assembling assembling-complete redirect gotov-otpravka",
     policy.StatusGroup.PAID_OR_IN_PROGRESS),
    ("Return klientskii-vozvrat kurerskii-vozvrat", policy.StatusGroup.RETURNED),
    ("CD RJ otmenen no-product out-of-stock already-buyed delyvery-did-not-suit Mindbox-Tech-Cancellation",
     policy.StatusGroup.CANCELLED),
    ("N DA DD DB DG DS new no-call availability-confirmed offer-analog assembling ready-to-wait "
     "client-confirmed sh opr-perem soglasovanie tovar-naiden kv", policy.StatusGroup.ORDER_IN_PROGRESS),
    ("ready-for-self-pickup send-to-delivery arrived-in-pickup-point de", policy.StatusGroup.DELIVERY_IN_PROGRESS),
])
def test_known_groups_do_not_promote_intermediate_or_returned_statuses(codes, group):
    for status in codes.split():
        assert policy.classify_status(status) is group
        if group not in (policy.StatusGroup.COMPLETED, policy.StatusGroup.PARTIALLY_COMPLETED):
            assert policy.eligible_purchase_policies(status, 1) == frozenset()


@pytest.mark.parametrize("quantity", [
    0, -1, 0.0, -1.0, 1.5, Decimal("0"), Decimal("-0"), Decimal("-2"), Decimal("0.001"),
    Decimal("1.0000000000000000000000000001"), Decimal("NaN"), Decimal("sNaN"),
    Decimal("Infinity"), Decimal("-Infinity"), float("inf"), float("-inf"), float("nan"),
    True, False, None, "1", [], {},
])
@pytest.mark.parametrize("status", ["F", "CP"])
def test_invalid_or_nonpositive_integer_quantity_is_not_a_purchase(quantity, status):
    assert policy.eligible_purchase_policies(status, quantity) == frozenset()


@pytest.mark.parametrize("quantity", [1, 3, 10**100, 1.0, Decimal("1.000"), Decimal("2"), Decimal("1E+30")])
def test_positive_integral_quantity_without_rounding(quantity):
    assert policy.eligible_purchase_policies("F", quantity) == frozenset({STRICT, EXTENDED})
    assert policy.eligible_purchase_policies("CP", quantity) == frozenset({EXTENDED})


def test_decimal_context_does_not_round_fraction_into_an_eligible_quantity():
    quantity = Decimal("1.0000000000000000000000000001")
    with localcontext() as context:
        context.prec = 2
        assert not policy.accepts_purchase_line("F", quantity, policy=STRICT)


@pytest.mark.parametrize("status", ["f", "SOLD", "cp", " F", "F ", "Sold", "return", "RETURN", "F-more",
                                   "", None, 1, [], {}])
def test_unknown_spelling_is_not_normalized_or_promoted(status):
    assert policy.classify_status(status) is policy.StatusGroup.UNKNOWN
    assert policy.eligible_purchase_policies(status, 1) == frozenset()


@pytest.mark.parametrize("selected", ["STRICT", "strict_completed", "", None, [], {}])
def test_unknown_policy_fails_explicitly_without_echoing_input(selected):
    with pytest.raises(ValueError, match="^Unknown analytical purchase policy$"):
        policy.accepts_purchase_line("F", 1, policy=selected)


@pytest.mark.parametrize("price", [Decimal("0"), Decimal("10.25")])
@pytest.mark.parametrize("product", [None, ProductKey("unsupported", "unknown"), ProductKey("offline1C", "unknown")])
def test_zero_price_and_missing_product_resolution_do_not_change_eligibility(line, price, product):
    raw_line = replace(line, price_of_line=price, product=product)
    assert policy.classify_status(raw_line.line_status) is policy.StatusGroup.COMPLETED
    assert policy.eligible_purchase_policies(raw_line.line_status, raw_line.quantity) == frozenset({STRICT, EXTENDED})


def test_mixed_order_has_distinct_groups_and_one_line_contribution_per_policy(line):
    lines = (line, replace(line, line_id="partial", line_status="CP", price_of_line=Decimal("20.00")),
             replace(line, line_id="returned", line_status="Return", price_of_line=Decimal("8.00")),
             replace(line, line_id="paid", line_status="delivering", price_of_line=Decimal("15.00")))
    assert [policy.classify_status(item.line_status) for item in lines] == [
        policy.StatusGroup.COMPLETED, policy.StatusGroup.PARTIALLY_COMPLETED,
        policy.StatusGroup.RETURNED, policy.StatusGroup.PAID_OR_IN_PROGRESS,
    ]
    totals = {selected: dict(orders=set(), buyers=set(), lines=0, units=Decimal(0), amount=Decimal(0))
              for selected in (STRICT, EXTENDED)}
    for item in lines:
        for selected in policy.eligible_purchase_policies(item.line_status, item.quantity):
            result = totals[selected]
            result["orders"].add(item.order_id)
            result["buyers"].add(item.customer_id)
            result["lines"] += 1
            result["units"] += item.quantity
            result["amount"] += item.price_of_line
    assert totals[STRICT] == dict(orders={line.order_id}, buyers={line.customer_id}, lines=1,
                                  units=Decimal(3), amount=Decimal("10.25"))
    assert totals[EXTENDED] == dict(orders={line.order_id}, buyers={line.customer_id}, lines=2,
                                    units=Decimal(6), amount=Decimal("30.25"))
    # Price is a whole-line amount; returns are classified, not subtracted.
    assert line.quantity == Decimal(3) and line.price_of_line == Decimal("10.25")


def test_status_sets_and_policy_mapping_are_immutable():
    assert policy.STRICT_COMPLETED_STATUSES == frozenset({"F", "sold"})
    assert policy.EXTENDED_COMPLETED_STATUSES == frozenset({"F", "sold", "CP"})
    assert all(isinstance(value, frozenset) for value in policy.PURCHASE_POLICY_STATUSES.values())
    with pytest.raises(TypeError):
        policy.PURCHASE_POLICY_STATUSES[STRICT] = frozenset({"CP"})
    with pytest.raises(AttributeError):
        policy.STRICT_COMPLETED_STATUSES.add("CP")


def test_production_selection_builder_and_quantity_semantics_remain_independent(line):
    assert DEFAULT_SELECTION.purchase_line_statuses == ("CP", "delivering", "F")
    assert PURCHASE_LINE_STATUSES == frozenset({"CP", "delivering", "F"})
    for status in ("F", "sold", "CP", "delivering", "Return"):
        builder = InteractionBuilder()
        result = builder.from_order_line(replace(line, line_status=status))
        assert (result is not None) == (status in {"CP", "delivering", "F"})
    for quantity in (Decimal(0), Decimal("-1.5"), Decimal("1.5")):
        result = InteractionBuilder().from_order_line(replace(line, quantity=quantity))
        assert result is not None and result.quantity == quantity
        assert not policy.eligible_purchase_policies(line.line_status, quantity)
