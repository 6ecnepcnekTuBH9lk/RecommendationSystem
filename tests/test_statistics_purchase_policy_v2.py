from datetime import datetime, timezone
from decimal import Decimal

import pytest

from Application import statistics_unified_analytics as analytics
from Application import statistics_unified_index as index
from Application.statistics_unified_breakdowns import OrderBreakdowns
from Application.statistics_unified_dto import StatisticsResultError, read_envelope
from Application.statistics_unified_facts import normalize_order_observation, reconcile_order
from Application.mindbox.identity import CustomerIdResolver
from tests import test_statistics_unified_facts as facts
from tests import test_statistics_unified_integration as integration
from tests.test_statistics_unified_integration import source

__all__ = ["source"]


@pytest.mark.parametrize(("status", "quantity", "price", "count", "units", "amount"), [
    ("F", 1, 15000, 1, 1, 15000),
    ("sold", 1, 8000, 1, 1, 8000),
    ("CP", 3, 12000, 1, 3, 12000),
    ("CP", 1, 0, 1, 1, 0),
    ("CP", 2, -5, 1, 2, None),
    ("CP", 2, facts._MISSING, 1, 2, None),
    ("CP", 0, 12000, 0, 0, 0),
    ("CP", facts._MISSING, 12000, 0, 0, 0),
    ("RJ", 1, 5000, 0, 0, 0),
    ("Return", 1, 5000, 0, 0, 0),
    ("CD", 1, 5000, 0, 0, 0),
    ("unknown", 1, 5000, 0, 0, 0),
])
def test_line_cost_once(status, quantity, price, count, units, amount):
    order = facts._fact(facts._raw(facts._line(status, qty=quantity, price=price)))
    assert order.purchase.known_positions == count
    assert order.purchase.known_nominal_units == units
    assert order.money[0].recognized_purchase.value == amount
    assert order.money[0].valued_purchase_positions == (amount is not None and count == 1)


@pytest.mark.parametrize(("before", "after"), [("CP", "F"), ("F", "sold"), ("CP", "CP")])
def test_purchase_transitions_and_repeated_snapshots_count_once(before, after):
    first = facts._raw(facts._line(before, qty=3, price=12000))
    last = facts._raw(facts._line(after, qty=3, price=12000))
    order = facts._fact(first, first, last)
    assert order.purchase.observed.value is True
    assert order.purchase.known_positions == 1
    assert order.purchase.known_nominal_units == 3
    assert order.money[0].recognized_purchase.value == 12000


@pytest.mark.parametrize("change", [
    {"status": "Return"}, {"price": 500}, {"qty": 2}, {"namespace": "kanzlerKz"}, {"item": "B"},
])
def test_conflict_excludes_only_affected_position(change):
    conflict = facts._line(**({"status": "F", "price": 100} | change))
    first = facts._line("F", price=100)
    valid = facts._line("CP", key="valid", price=Decimal("1000.125"), item="C")
    order = reconcile_order(facts._observations(facts._raw(first, valid), facts._raw(conflict, valid)),
                            order_sort_rank=1, include_products=True)
    assert order.purchase.observed.value is True
    assert order.purchase.known_positions == 1
    assert order.purchase.excluded_positions == 1
    assert order.money[0].recognized_purchase.value == Decimal("1000.125")
    assert next(item for item in order.products if item.item_id == "C").purchase.known_positions == 1
    all_conflict = facts._fact(facts._raw(first), facts._raw(conflict))
    assert all_conflict.purchase.observed.value is False
    assert all_conflict.purchase.known_positions == 0


def test_duplicate_ids_and_zero_price_without_product_id():
    first = facts._line("CP", price=10)
    assert facts._fact(facts._raw(first, first)).purchase.known_positions == 1
    conflict = facts._fact(facts._raw(first, facts._line("CP", price=20)))
    assert conflict.purchase.known_positions == 0
    assert conflict.purchase.excluded_positions == 1
    observation = normalize_order_observation(facts._raw(facts._line("CP", price=0)), (0, 1, 0),
                                              CustomerIdResolver(), lambda _: None)
    order = reconcile_order((observation,), order_sort_rank=1, include_products=True)
    assert order.purchase.known_positions == 1
    assert not order.products
    assert order.money[0].recognized_purchase.value == 0


def test_basket_uses_other_positions_when_one_line_id_is_ambiguous():
    order = facts._fact(facts._raw(facts._line("CP", price=10), facts._line("CP", price=20),
                                  facts._line("F", key="valid", qty=3, price=100)))
    assert order.purchase.excluded_positions == 1
    basket = OrderBreakdowns()
    basket.add(order, lambda *args: None, whole_currency="RUB")
    assert basket.lines == {1: 1}
    assert basket.units == {3: 1}


def test_v2_reference_projection_and_prepared_generation_agree(source):
    integration._write(source)
    root, catalog, _, _, target = source
    reference = integration._report(source)
    index.build_unified_index(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    indexed = integration._report(source, reader="INDEXED", index_dir=target)
    analytics.prepare_analytics(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    prepared = integration._report(source, reader="INDEXED", index_dir=target, prepared_index=True)
    assert integration._business(reference) == integration._business(indexed) == integration._business(prepared)
    assert prepared.policy["policy_id"] == "UNIFIED_PURCHASE_V2"
    assert prepared.envelope()["schema_version"] == 13
    assert read_envelope(prepared.envelope()) == prepared
    with pytest.raises(StatisticsResultError, match="INCOMPATIBLE_RESULT_SCHEMA"):
        read_envelope({**prepared.envelope(), "schema_version": 12})
    assert prepared.orders["purchase_orders"]["known"] == 3
    assert prepared.orders["purchase_positions"]["known"] == 4
    assert Decimal(prepared.money["currencies"]["RUB"]["amounts"]["recognized_purchase"]["value"]) == Decimal(
        "1500.00000000000000000000000001"
    )
    assert prepared.money["currencies"]["KZT"]["amounts"]["recognized_purchase"]["value"] == "0"
    finance = prepared.customers["data"]["buyer_breakdowns"]
    assert sum(row[1] for row in finance["monthly_money"]["RUB"]) == 1
    assert len(finance["money"]["RUB"]["quintiles"]) == 1


def test_monthly_and_quintile_bases_preserve_ties_and_separate_currencies(source):
    october, november = integration._FIRST, integration._SECOND
    order = integration.cohort_tests._order
    rows = [order(f"rub-{i}", october, owner=f"customer-{i}", amount=amount)
            for i, amount in enumerate((5, 5, 10, 20, 30, 40))]
    rows[0]["lines"].extend(order("mixed", october, amount=200, namespace="kanzlerKz")["lines"])
    rows[0]["lines"][1].update(id="mixed-line", number=2)
    rows.extend((order("free-repeat", november, "CP", owner="customer-0", amount=0),
                 order("kzt-only", november, owner="only-kzt", amount=300, namespace="kanzlerKz"),
                 order("missing", november, owner="unpriced", amount="invalid")))
    integration.rfm_tests._write_orders(source, rows)
    period = (datetime(2025, 10, 1, tzinfo=timezone.utc), datetime(2025, 12, 1, tzinfo=timezone.utc))
    result = integration._report(source, order_period=period, buyer_period=period, customer_period=period)
    breakdown = result.customers["data"]["buyer_breakdowns"]
    assert breakdown["monthly_money"]["RUB"] == [["2025-10", 6, 6, "110"], ["2025-11", 1, 1, "0"]]
    assert breakdown["monthly_money"]["KZT"] == [["2025-10", 1, 1, "200"], ["2025-11", 1, 1, "300"]]
    rub, kzt = (breakdown["money"][currency] for currency in ("RUB", "KZT"))
    assert (rub["known_orders"], rub["known_buyers"], Decimal(rub["mean_order"])) == (7, 6, Decimal(110) / 7)
    assert (kzt["known_orders"], kzt["known_buyers"], Decimal(kzt["mean_known_per_buyer"])) == (2, 2, 250)
    groups = rub["quintiles"]
    assert len(groups) == 5 and groups[0][:3] == ["5", "5", 2] and groups[0][5] == 1
    assert sum(Decimal(group[4]) for group in groups) == 110
    assert read_envelope(result.envelope()) == result
