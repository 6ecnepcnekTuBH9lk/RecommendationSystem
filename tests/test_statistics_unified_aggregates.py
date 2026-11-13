"""Synthetic end-to-end tests of isolated UNIFIED aggregates."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from Application import statistics_unified_aggregate_input as inputs
from Application import statistics_unified_index as index
from Application.mindbox.canonical_customers import encode
from Application.mindbox.selection import DEFAULT_SELECTION
from Application.statistics_storage_safety import StatisticsIndexError, path_identity
from Application.statistics_unified_policy import EvidenceQuality as Quality, STATUS_CATALOG
from tests.test_dataset_statistics import action, dataset as dataset_fixture, order, put_entry, save

_NOW = datetime(2026, 1, 3, tzinfo=timezone.utc)
_FIRST = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SECOND = datetime(2026, 1, 2, tzinfo=timezone.utc)
_END = datetime(2026, 1, 3, tzinfo=timezone.utc)


@pytest.fixture
def source(tmp_path, monkeypatch):
    root, catalog, data = dataset_fixture.__wrapped__(tmp_path)
    catalog.write_text("КодНоменклатуры|Номенклатура|КатегорияНаСайте\n000001|A|cat-A\n000002|B|cat-B\n",
                       encoding="utf-8-sig")
    catalog.with_name("site_categories.csv").write_text(
        "КодКатегории|НазваниеКатегории|КодРодительскойКатегории\ncat-A|A|parent\ncat-B|B|parent\nparent|Root|\n",
        encoding="utf-8-sig")
    scratch = root / "scratch"
    target = root / "index"
    for path in (scratch, target):
        path.mkdir(mode=0o700)
    monkeypatch.setattr(index, "protect_directory", lambda path: path.chmod(0o700))
    for module in (index, inputs):
        monkeypatch.setattr(module, "validate_storage_root", lambda path, **kw: path_identity(path))
    return root, catalog, data, scratch, target


def _write(source, kind, records):
    root, _, data, _, _ = source
    entry = next(iter(data[kind].values()))
    key = "orders" if kind == "orders" else "customerActions"
    path = root / entry["directory"] / f"{kind}_part_001.json"
    path.write_text(encode({key: records}), encoding="utf-8")
    return path


def _options(source, **kwargs):
    root, catalog, _, scratch, _ = source
    return dict(raw_root=root, catalog_path=catalog, scratch_dir=scratch, expected_bytes=1024**2,
                calculated_at=_NOW, **kwargs)


def _reference(source, rows, **kwargs):
    _write(source, "orders", rows)
    return inputs.aggregate_reference(**_options(source, **kwargs))


def _by_code(result, code, bucket="CATALOG"):
    return next(row for row in result.statuses if row.code == code and row.bucket == bucket)


def _money(rows, unit="RUB"):
    return next(row for row in rows if row.currency == unit)


def _item(result, code):
    return next(row for row in result.products if row.item_id == code)


@pytest.mark.parametrize("statuses,purchases,positions,nominal,recognized", [
    (["F"], 1, 1, 0, 20),
    (["sold"], 1, 1, 0, 20),
    (["CP"], 1, 1, 20, None),
    (["delivering"], 0, 0, 0, 0),
    (["Return"], 0, 0, 0, 0),
    (["F", "CP"], 1, 2, 20, None),
    (["F", "CD"], 1, 1, 0, 20),
    (["F", "Return"], 1, 1, 0, 20),
    (["CP", "Return"], 1, 1, 20, None),
])
def test_order_status_purchase_and_amounts(source, statuses, purchases, positions, nominal, recognized):
    result = _reference(source, [order(statuses)])
    assert result.orders["orders"].total.value == 1
    assert result.orders["purchase_orders"].total.value == purchases
    assert result.orders["purchase_positions"].total.value == positions
    assert result.orders["orders_without_evidence"].total.value == 1 - purchases
    assert result.orders["cancelled_orders"].known == int("CD" in statuses)
    assert result.orders["returned_orders"].known == int("Return" in statuses)
    assert result.orders["all_returned_orders"].known == int(statuses == ["Return"])
    assert result.orders["mixed_orders"].known == int(len(statuses) > 1)
    rub = _money(result.order_money)
    assert rub.amounts["partial_nominal"].total.value == nominal
    assert rub.amounts["recognized_purchase"].total.value == recognized
    assert rub.amounts["confirmed_refund"].total.reasons == ("LEDGER_UNAVAILABLE",)
    assert rub.amounts["net_after_refunds"].total.value is None
    if "CP" in statuses:
        assert "CP_ALLOCATION_UNKNOWN" in rub.amounts["recognized_purchase"].total.reasons
    assert rub.valued_orders == int(recognized == 20)
    assert rub.unvalued_orders == int(recognized is None)


def test_exact_duplicates_and_historical_sale_conflict_are_distinct(source):
    paid = order(["F"], "one-order")
    returned = deepcopy(paid)
    returned["lines"][0]["status"]["ids"]["externalId"] = "Return"
    returned["lines"][0]["priceOfLine"] = 900
    result = _reference(source, [paid, paid, returned])
    assert result.orders["purchase_orders"].total.value == 1
    assert result.orders["purchase_positions"].total.value == 1
    assert result.orders["conflicting_orders"].total.value == 1
    assert result.orders["returned_units"].total.reasons == ("RETURNED_QUANTITY_UNVERIFIED",)
    assert result.customers["population"].total.value == 3
    assert result.customers["population_buyers"].total.value == 1
    assert result.customers["returned_clients"].total.value is None
    assert result.diagnostics["exact_duplicates"] == 1
    assert result.diagnostics["raw_orders"] == 3
    assert _by_code(result, "F").raw_occurrences == 2
    assert _by_code(result, "F").positions.total.value is None
    assert _by_code(result, "F").historical_positions.total.value == 1
    assert _money(result.order_money).amounts["known_purchase_subtotal"].total.value == 20
    assert _money(result.order_money).amounts["recognized_purchase"].total.quality is Quality.CONFLICT
    assert _money(result.order_money).amounts["returned_nominal"].total.value is None
    assert _item(result, "000001").purchase_positions.total.value == 1
    assert result.actions["PURCHASE"].total.value == 1
    again = _reference(source, [returned, paid, paid])
    assert replace(again, contexts=()) == replace(result, contexts=())


def test_catalog_and_bad_quantities_keep_status_counts(source):
    zero, invalid, unknown = order(["F"], "zero"), order(["F"], "bad"), order(["future"], "unknown")
    zero["lines"][0]["quantity"] = 0
    invalid["lines"][0]["quantity"] = Decimal("0.12345678901234567890123")
    missing = order(["F"], "missing")
    del missing["lines"][0]["status"]
    result = _reference(source, [zero, invalid, unknown, missing])
    assert {row.code for row in result.statuses if row.bucket == "CATALOG"} == set(STATUS_CATALOG)
    assert _by_code(result, "F").positions.known == 2
    assert _by_code(result, "F").orders.known == 2
    assert _by_code(result, "future", "UNKNOWN_CODE").raw_occurrences == 1
    assert _by_code(result, None, "MISSING_CODE").positions.known == 1
    assert _by_code(result, "sold").positions.total.value is None
    assert result.orders["purchase_orders"].total.value == 0
    assert result.diagnostics["FRACTIONAL_QUANTITY"] == 1
    assert result.diagnostics["ZERO_QUANTITY"] == 1
    complete = _reference(source, [zero])
    assert _by_code(complete, "sold").positions.total.value == 0
    assert _by_code(complete, "Mindbox-Tech-Cancellation").name is None


def test_fiscal_amounts_decimals_and_valued_denominators(source):
    a, b, c, d = [order(["F"], key) for key in ("a", "b", "c", "d")]
    a["lines"][0]["quantity"] = 3
    a["lines"][0]["priceOfLine"] = Decimal("1200.00000000000000000000000001")
    b["lines"][0]["priceOfLine"] = 0
    del c["lines"][0]["priceOfLine"]
    c["customer"]["ids"]["mindboxId"] = "other"
    d["customer"]["ids"]["mindboxId"] = "inactive"
    d["lines"][0]["product"]["ids"] = {"kanzlerKz": "000002-size"}
    d["lines"][0]["priceOfLine"] = Decimal("77.10")
    result = _reference(source, [a, b, c, d])
    rub, kzt = result.order_money
    assert rub.amounts["recognized_purchase"].total.quality is Quality.UNKNOWN
    assert rub.amounts["known_purchase_subtotal"].total.value == Decimal("1200.00000000000000000000000001")
    assert (rub.valued_orders, rub.unvalued_orders) == (2, 1)
    assert rub.mean_order.value == Decimal("600.000000000000000000000000005")
    assert kzt.amounts["recognized_purchase"].total.value == Decimal("77.10")
    assert kzt.valued_orders == 1
    product = _item(result, "000001")
    assert product.nominal_units.total.value == 7
    assert product.fulfilled_units.total.value is None
    assert result.buyer_money[0].fully_valued_buyers == 1
    assert result.buyer_money[0].unvalued_buyers == 1


def test_cp_only_unknown_is_not_promoted_to_zero_by_nonpurchase_order(source):
    result = _reference(source, [order(["CP"], "cp"), order(["CD"], "cancelled")])
    rub = _money(result.order_money)
    assert rub.amounts["known_purchase_subtotal"].known is None
    assert rub.amounts["known_purchase_subtotal"].total.value is None
    assert rub.amounts["recognized_purchase"].total.value is None
    assert rub.amounts["partial_nominal"].total.value == 20
    assert rub.amounts["cancelled_nominal"].total.value == 20
    assert rub.mean_order.value is None


def test_conflicting_owner_does_not_duplicate_population_or_buyer(source):
    a = order(["F"], "conflicted")
    b = deepcopy(a)
    b["customer"]["ids"]["mindboxId"] = "other"
    result = _reference(source, [a, b])
    assert result.orders["purchase_orders"].total.value == 1
    assert result.customers["population"].known == 3
    assert result.customers["buyers"].known == 0
    assert result.customers["buyers"].total.quality is Quality.INSUFFICIENT_COVERAGE
    assert result.customers["owner_conflict_orders"].known == 1
    assert result.actions["PURCHASE_users"].known == 0
    assert result.actions["PURCHASE"].total.value == 1


def test_product_filter_is_applied_after_reconciliation(source):
    basket = order(["F", "CD"], "basket")
    basket["lines"][1]["product"]["ids"] = {"offline1C": "000002-size"}
    result = _reference(source, [basket], selected_items=["000002"])
    assert result.orders["orders"].total.value == 1
    assert result.orders["mixed_orders"].known == 1
    assert result.orders["fully_cancelled_orders"].total.value == 0
    assert result.orders["cancelled_positions"].total.value == 1
    assert result.orders["purchase_orders"].total.value == 0
    assert result.customers["buyers"].total.value == 1
    assert result.observed_history["purchase_orders"].total.value == 1
    assert _item(result, "000002").cancelled_positions.total.value == 1
    assert result.contexts[3].product_filter_key is None


def test_proxy_periods_frequency_and_direct_category_pairs(source):
    first = order(["F", "F", "CP"], "first")
    first["firstAction"]["dateTimeUtc"] = "2026-01-01T13:30:00+03:00"
    first["lines"][2]["product"]["ids"] = {"offline1C": "000002-size"}
    later = order(["F"], "later")
    later["firstAction"]["dateTimeUtc"] = "2026-01-02T04:00:00Z"
    later["lines"][0]["product"]["ids"] = {"offline1C": "000002-size"}
    returned = deepcopy(later)
    returned["lines"][0]["status"]["ids"]["externalId"] = "Return"
    root, _, data, _, _ = source
    _write(source, "orders", [first, first])
    data["orders"][_SECOND.date().isoformat()] = put_entry(
        root, "orders", [later, returned], 900, _SECOND.isoformat(), _END.isoformat())
    save(root, data)
    result = inputs.aggregate_reference(**_options(source, order_period=(_FIRST, _SECOND),
                                                   buyer_period=(_FIRST, _END)))
    assert result.orders["purchase_orders"].total.value == 1
    assert result.customers["buyer_purchase_orders"].total.value == 2
    assert result.customers["repeat_buyers"].total.value == 1
    assert result.frequency == ((2, 1),)
    assert result.first_months == (("2026-01", 1),)
    assert result.second_intervals == ((1, 1),)
    assert result.first_categories == (("cat-A", 1), ("cat-B", 1))
    assert result.second_categories == (("cat-B", 1),)
    assert [(entry.categories, entry.orders.total.value) for entry in result.categories] == [
        (("cat-A",), 1), (("cat-B",), 2)]
    assert [(entry.categories, entry.orders.total.value) for entry in result.pairs] == [(("cat-A", "cat-B"), 1)]
    assert result.observed_history["purchase_orders"].total.value == 2


def test_partial_structure_keeps_historical_purchase_order_and_known_position(source):
    row = order(["F"], "partial-structure")
    row["lines"].append(False)
    result = _reference(source, [row])
    assert result.observed_history["purchase_orders"].total.value == 1
    assert result.observed_history["purchase_positions"].known == 1
    assert result.observed_history["purchase_positions"].total.value is None
    assert result.orders["incomplete_structure_orders"].total.value == 1


def test_undated_purchase_is_preserved_in_history_with_partial_period(source):
    a = order(["F"], "undated")
    del a["firstAction"]["dateTimeUtc"]
    result = _reference(source, [a])
    assert result.orders["purchase_orders"].known == 0
    assert result.orders["purchase_orders"].total.quality is Quality.INSUFFICIENT_COVERAGE
    assert result.observed_history["purchase_orders"].total.value == 1
    assert result.first_months == ()
    assert result.diagnostics["undated_orders"] == 1
    assert _money(result.buyer_money).mean_per_valued_buyer.value is None


def test_purchase_intersections_use_unique_unified_positions(source):
    view = DEFAULT_SELECTION.view_action_system_names[0]
    favorite = DEFAULT_SELECTION.favorite_action_system_names[0]
    events = [action(view, ["000001-a"], "old"), action(favorite, ["000001-a"], "new"),
              action(view, ["000002-a"], "other"), action("technical-order-event", ["000001-a"], "other"),
              action(favorite, ["000002-a"], "inactive")]
    _write(source, "actions", events)
    sold = order(["sold"], "sold")
    other = order(["delivering"], "shipping")
    other["customer"]["ids"]["mindboxId"] = "other"
    result = _reference(source, [sold, sold, other])
    values = {name: row.total.value for name, row in result.actions.items()}
    assert values["VIEW"] == values["FAVORITE"] == 2
    assert values["PURCHASE"] == 1
    assert values["VIEW_PURCHASE"] == values["FAVORITE_PURCHASE"] == 1
    assert values["VIEW_FAVORITE"] == values["VIEW_FAVORITE_PURCHASE"] == 1
    assert result.diagnostics["ACTION_UNMAPPED"] == 1
    assert result.customers["population"].total.value == 3


def test_explicit_population_period_uses_dated_evidence(source):
    result = _reference(source, [order(["F"], "purchase")], customer_period=(_FIRST, _SECOND))
    assert result.customers["population"].total.value == 2
    assert result.customers["population_buyers"].total.value == 1
    assert result.observed_history["purchase_orders"].total.value == 1
    all_observed = inputs.aggregate_reference(**_options(source))
    assert all_observed.customers["population"].total.value == 3


def test_end_to_end_reference_and_schema4_parity(source, monkeypatch):
    root, catalog, _, scratch, target = source
    mixed = order(["F", "CP", "CD", "Return"], "mixed")
    mixed["lines"][0]["priceOfLine"] = Decimal("1000.00000000000000000001")
    mixed["lines"][1]["product"]["ids"] = {"kanzlerKz": "000002-a"}
    mixed["lines"][2]["product"]["ids"] = {"offline1C": "000002-a"}
    cp = order(["CP"], "cp")
    cp["customer"]["ids"]["mindboxId"] = "other"
    conflict = order(["sold"], "owner")
    conflicting_owner = deepcopy(conflict)
    conflicting_owner["customer"]["ids"]["mindboxId"] = "other"
    changes = deepcopy(mixed)
    changes["lines"][0]["status"]["ids"]["externalId"] = "Return"
    invalid = order(["F", "custom-status"], "invalid")
    invalid["lines"][0]["quantity"] = -2
    del invalid["lines"][1]["status"]
    _write(source, "orders", [mixed, mixed, cp, changes, conflict, conflicting_owner, invalid])
    index.build_unified_index(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    with index.reference_observations(raw_root=root, catalog_path=catalog) as observations:
        by_ref = {row.source_ref: row for row in observations}
    customers, orders, lines = {}, {}, {}
    with index.open_unified_index(raw_root=root, catalog_path=catalog, index_dir=target) as reader:
        for row in reader.orders():
            indexed = row.observation
            reference = by_ref[indexed.source_ref]
            for table, sid, key in ((customers, indexed.customer_key.value, reference.customer_key.value),
                                    (orders, indexed.order_key.value, reference.order_key.value)):
                assert table.setdefault(sid, key) == key
            for a, b in zip(indexed.lines, reference.lines, strict=True):
                key = reference.order_key.value, b.stable_line_key.value
                assert lines.setdefault(a.stable_line_key.value, key) == key
    for mapped in (customers, orders, lines):
        assert len(set(mapped.values())) == len(mapped)
    for selection in (None, ["000002"]):
        options = _options(source, selected_items=selection)
        a = inputs.aggregate_reference(**options)
        def no_raw_records(*args, **kwargs):
            pytest.fail("INDEXED must read projected observations and Actions")

        with monkeypatch.context() as patch:
            patch.setattr(index, "_records", no_raw_records)
            patch.setattr(inputs, "_records", no_raw_records)
            b = inputs.aggregate_indexed(**options, index_dir=target)
        assert all(x.policy == y.policy and x.period == y.period and x.provenance_digest == y.provenance_digest
                   and x.product_filter_key == y.product_filter_key for x, y in zip(a.contexts, b.contexts, strict=True))
        assert a == replace(b, contexts=tuple(replace(y, identity_space=x.identity_space)
                                              for x, y in zip(a.contexts, b.contexts, strict=True)))
    assert not list(scratch.iterdir())


def test_cancellation_and_incompatible_projection_fail_without_result(source, monkeypatch):
    root, catalog, _, scratch, target = source
    index.build_unified_index(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    pointer = (target / index.POINTER_NAME).read_bytes()
    original = inputs.accumulate_aggregates
    state = {"cancelled": False}

    def interrupt(*args, **kwargs):
        state["cancelled"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(inputs, "accumulate_aggregates", interrupt)
    with pytest.raises(InterruptedError):
        inputs.aggregate_indexed(**_options(source, cancelled=lambda: state["cancelled"]), index_dir=target)
    assert not list(scratch.iterdir())
    assert (target / index.POINTER_NAME).read_bytes() == pointer
    (target / index.POINTER_NAME).write_bytes(pointer.replace(b'"schema": 4', b'"schema": 3'))
    with pytest.raises(StatisticsIndexError, match="INCOMPATIBLE_SCHEMA"):
        inputs.aggregate_indexed(**_options(source), index_dir=target)
