"""Synthetic RFM tests through the reconciled Orders and schema 4 entry points."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from Application import statistics_unified_aggregate_input as inputs
from Application import statistics_unified_index as index
from Application import statistics_unified_rfm as rfm
from Application.mindbox.canonical_customers import encode
from Application.mindbox.selection import DEFAULT_SELECTION
from Application.statistics_rfm_core import empirical_thresholds
from Application.statistics_storage_safety import StatisticsIndexError, path_identity
from Application.statistics_unified_policy import EvidenceQuality as Quality
from tests.test_dataset_statistics import action, order, put_entry, save
from tests import test_statistics_unified_aggregates as aggregate_tests


_AS_OF = "2026-08-31"
_START = "2022-09-01T00:00:00+00:00"
_END = "2026-09-01T00:00:00+00:00"
_NOW = datetime.fromisoformat(_END)
_RECENT = "2026-08-30T19:00:00-04:00"


@pytest.fixture
def source(tmp_path, monkeypatch):
    aggregate_source = aggregate_tests.source.__wrapped__(tmp_path, monkeypatch)
    root, _, data, _, _ = aggregate_source
    orders = put_entry(root, "orders", [], 30, _START, _END, kind="MANUAL")
    actions = put_entry(root, "actions", [], 30, _START, _END, kind="MANUAL")
    data["manual_interactions"] = dict(since=_START, until=_END, updated=orders["updated"],
                                       orders=orders, actions=actions)
    save(root, data)
    monkeypatch.setattr(rfm, "validate_storage_root", lambda path, **kw: path_identity(path))
    return aggregate_source


def _order(key, status="F", *, stamp=_RECENT, owner="new", amount=20, namespace="offline1C"):
    row = order([status], key)
    row["firstAction"]["dateTimeUtc"] = stamp
    row["customer"]["ids"]["mindboxId"] = owner
    row["lines"][0]["product"]["ids"] = {namespace: "000001-size"}
    row["lines"][0]["priceOfLine"] = amount
    return row


def _options(source, **overrides):
    root, catalog, _, scratch, _ = source
    return dict(dict(raw_root=root, catalog_path=catalog, scratch_dir=scratch, expected_bytes=1024**2,
                     calculated_at=_NOW, selected_end=_AS_OF, customer_key="new"), **overrides)


def _write_orders(source, orders):
    root, _, data, _, _ = source
    entry = data["manual_interactions"]["orders"]
    (root / entry["directory"] / "orders_part_001.json").write_text(encode({"orders": orders}), encoding="utf-8")


def _reference(source, orders, **options):
    _write_orders(source, orders)
    return rfm.rfm_reference(**_options(source, **options))


def _feature(result, currency="RUB", window="LAST_12M"):
    return result.customer[window][currency]


@pytest.mark.parametrize(("scores", "frequency", "prior", "segment"), [
    ((4, 4, 5), 2, 0, "CHAMPIONS"),
    ((4, 4, 4), 2, 0, "LOYAL"),
    ((3, 3, 1), 2, 0, "LOYAL"),
    ((4, 4, 5), 1, 3, "NEW_OR_RECENT"),
    ((2, 1, 1), 0, 2, "AT_RISK"),
    ((2, 2, 2), 1, 2, "AT_RISK"),
    ((2, 3, 5), 2, 0, "AT_RISK"),
    ((2, 1, 1), 0, 1, "HIBERNATING"),
    ((2, 1, 1), 1, 0, "HIBERNATING"),
    ((3, 1, 1), 0, 5, "OTHER"),
    ((4, 5, None), 3, 1, "UNSCORABLE"),
])
def test_segmentation_order_and_boundaries(scores, frequency, prior, segment):
    assert rfm._segment(*scores, frequency, prior) == segment


def test_earlier_repeat_history_is_currency_specific_in_all_windows(source):
    orders = [_order("old-rub-1", stamp="2022-10-01T00:00:00Z"),
              _order("old-rub-2", stamp="2022-11-01T00:00:00Z", status="CP"),
              _order("old-kzt", stamp="2022-11-01T00:00:00Z", namespace="kanzlerKz")]
    for i in range(6):
        for currency in ("offline1C", "kanzlerKz"):
            orders.append(_order(f"recent-{currency}-{i}", owner=f"recent-{i}", namespace=currency))
    result = _reference(source, orders)
    reordered = _reference(source, list(reversed(orders)))
    assert reordered.windows == result.windows and reordered.customer == result.customer
    assert result.dual_currency_buyers == 7
    for window, payload in result.windows.items():
        rub, kzt = (_feature(result, currency, window) for currency in ("RUB", "KZT"))
        assert (rub.segment, kzt.segment) == ("AT_RISK", "HIBERNATING")
        assert (rub.observed_frequency, kzt.observed_frequency) == ((2, 1) if window == "LAST_48M" else (0, 0))
        assert (rub.monetary.value, kzt.monetary.value) == ((40, 20) if window == "LAST_48M" else (0, 0))
        for facts in payload["currencies"].values():
            assert sum(row[1] for row in facts["segments"]) == facts["buyers"] == 7
            assert sum(row[2] for row in facts["segments"]) == pytest.approx(100)


@pytest.mark.parametrize(("status", "is_buyer", "amount", "reason"), [
    ("F", True, Decimal(20), None),
    ("sold", True, Decimal(20), None),
    ("CP", True, Decimal(20), None),
    ("delivering", False, None, "NOT_IN_CURRENCY_BASE"),
    ("Return", False, None, "NOT_IN_CURRENCY_BASE"),
])
def test_unified_evidence_changes_population_and_monetary(source, status, is_buyer, amount, reason):
    result = _reference(source, [_order("order", status)])
    feature = _feature(result)
    assert result.population == 3 and result.base_buyers == int(is_buyer)
    assert feature.in_base == is_buyer
    assert feature.frequency.value == int(is_buyer)
    assert feature.monetary.value == amount
    if is_buyer:
        assert feature.recency.value == 1
        assert feature.last_order.value == datetime(2026, 8, 30, 23, tzinfo=timezone.utc)
    if reason is not None:
        assert reason in feature.monetary.reasons
    if status == "CP":
        assert feature.partial_nominal == Decimal(20)
        assert feature.known_subtotal == 20
        assert feature.scores[2] == 1


@pytest.mark.parametrize("transitions", [("CP", "F"), ("F", "Return")])
def test_changed_and_duplicate_snapshots_keep_one_order(source, transitions):
    first, second = [_order("one-order", status, amount=100) for status in transitions]
    a = _reference(source, [first, first, second])
    b = _reference(source, [second, first, first])
    assert a.customer == b.customer and a.windows == b.windows and a.diagnostics == b.diagnostics
    feature = _feature(a)
    if transitions[-1] == "Return":
        assert feature.frequency.value == 0 and feature.recency.value is None
        assert feature.monetary.value is None and not feature.in_base
    else:
        assert feature.frequency.value == 1 and feature.recency.value == 1
        assert feature.monetary.value == feature.known_subtotal == 100


def test_real_zero_missing_price_and_exact_decimal_subtotal(source):
    zero = _order("free", amount=0)
    free = _feature(_reference(source, [zero]))
    assert free.monetary.value == 0 and free.known_subtotal == 0 and free.scores[2] == 1
    missing = _order("missing")
    del missing["lines"][0]["priceOfLine"]
    feature = _feature(_reference(source, [zero, missing]))
    assert feature.frequency.value == 2 and feature.known_subtotal == 0
    assert feature.monetary.value == 0
    amount = Decimal("10000.00000000000000000000000001")
    paid = _order("paid", amount=amount)
    cp = _order("cp", "CP", amount=5000)
    feature = _feature(_reference(source, [paid, cp, cp]))
    assert feature.frequency.value == 2
    assert feature.known_subtotal == Decimal("15000.00000000000000000000000001") and feature.partial_nominal == 5000
    assert feature.monetary.value == feature.known_subtotal and feature.scores[2] is not None
    assert feature.valued_orders == 2


def test_windows_comparison_and_cutoff_use_creation_date(source):
    orders = [_order(str(i), stamp=stamp) for i, stamp in enumerate((
        _RECENT, "2025-09-01T00:00:00Z", "2025-08-31T23:59:59Z", "2024-09-01T00:00:00Z",
        "2024-08-31T23:59:59Z", "2022-09-01T00:00:00Z", "2022-08-31T23:59:59Z", _END))]
    # This unrelated action time must have no effect on R.
    orders[0]["dateTimeUtc"] = "2027-02-01T00:00:00Z"
    result = _reference(source, orders, selected_end=None)
    assert result.context.as_of_date == date.fromisoformat(_AS_OF)
    assert result.context.policy.policy_id == "UNIFIED_PURCHASE_V2"
    assert result.context.policy.time_basis_id == "ORDER_CREATED_PURCHASE_PROXY_V1"
    assert not result.context.created_history_verified and not result.context.status_history_verified
    assert result.context.product_filter_key is None
    counts = {name: _feature(result, window=name).frequency.value for name in result.windows}
    assert counts == dict(LAST_12M=2, LAST_24M=4, LAST_48M=6, PREVIOUS_12M=2)
    assert result.comparison["observed_groups"] == dict(BOTH=1, LAST_ONLY=0, PREVIOUS_ONLY=0, NEITHER=0)
    assert result.comparison["order_change"] == 0
    assert result.comparison["money"]["RUB"]["known_amount_change"] == 0
    assert _feature(result).recency.value == 1


def test_two_currencies_and_mixed_order_keep_subtotals(source):
    rub = _order("rub", amount=100)
    kzt = _order("kzt", amount=200, namespace="kanzlerKz")
    result = _reference(source, [rub, kzt])
    assert result.dual_currency_buyers == 1
    assert _feature(result, None).frequency.value == 2
    assert _feature(result).monetary.value == 100
    assert _feature(result, "KZT").monetary.value == 200
    mixed = _order("mixed", amount=10)
    another = deepcopy(kzt["lines"][0])
    another.update(id="new-line", number=1, priceOfLine=30)
    mixed["lines"].append(another)
    result = _reference(source, [rub, kzt, mixed])
    assert _feature(result, None).frequency.value == 3
    for currency, subtotal in [("RUB", 110), ("KZT", 230)]:
        feature = _feature(result, currency)
        assert feature.frequency.value == 1
        assert feature.known_subtotal == subtotal
        assert feature.monetary.value == subtotal
    result = _reference(source, [mixed])
    assert result.unassigned_currency_buyers == 1
    assert result.windows["LAST_12M"]["currencies"]["KZT"]["known_subtotal"] == 30


def test_conflicting_snapshot_currency_and_owner(source):
    a = _order("currency")
    b = _order("currency", namespace="kanzlerKz")
    good = _order("attributed")
    c = _order("owner")
    d = _order("owner", owner="other")
    missing = _order("missing")
    del missing["customer"]
    result = _reference(source, [a, b, c, d, good, missing])
    assert result.base_buyers == 1 and result.population == 3
    assert result.diagnostics["unattributed_purchase_orders"] == 2
    assert result.diagnostics["OWNER_CONFLICT"] == 1
    assert _feature(result, None).frequency.value == 1
    assert _feature(result).frequency.value == 1
    assert _feature(result).monetary.value == 20


def test_conflicting_dates_preserve_observed_rf_and_independent_windows(source):
    paid = _order("paid", stamp="2026-08-01T00:00:00Z", amount=100)
    a = _order("date-conflict", stamp="2026-08-02T00:00:00Z")
    b = _order("date-conflict", stamp="2026-08-03T00:00:00Z")
    result = _reference(source, [paid, a, b])
    feature = _feature(result)
    assert feature.observed_frequency == 1 and feature.known_subtotal == 100
    assert feature.frequency.quality is feature.recency.quality is Quality.CONFLICT
    assert feature.monetary.value == 100
    assert feature.observed_last_order == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert feature.last_order.reasons == ("CREATED_TIME_CONFLICT",)
    assert result.diagnostics["purchase_date_unresolved"] == 1
    previous = _feature(result, window="PREVIOUS_12M")
    assert previous.frequency.value == 0 and previous.monetary.value == 0
    assert previous.recency.value is None and previous.segment == "UNSCORABLE"
    undated = _order("undated", stamp=None)
    result = _reference(source, [undated])
    assert result.base_buyers == 0
    assert "CREATED_TIME_MISSING" in _feature(result, None).frequency.reasons
    assert _feature(result).observed_last_order is None


def test_coverage_gap_can_close_recency_without_closing_complete_short_window(source):
    root, _, data, _, _ = source
    manual = data["manual_interactions"]
    for entry in (manual, manual["orders"], manual["actions"]):
        entry["since"] = "2025-09-01T00:00:00+00:00"
    data["orders"]["2024-06-01"] = put_entry(root, "orders", [], 9,
                                            "2024-06-01T00:00:00+00:00", "2024-06-02T00:00:00+00:00")
    save(root, data)
    result = _reference(source, [_order("old", stamp="2024-06-01T00:00:00Z")])
    short = _feature(result)
    assert short.observed_last_order is not None and short.recency.quality is Quality.INSUFFICIENT_COVERAGE
    assert short.frequency.value == 0 and short.monetary.value == 0 and short.scores[0] is None
    assert result.windows["LAST_24M"]["coverage"]["status"] == "PARTIAL"
    assert _feature(result, window="LAST_24M").scores == (None, None, None)
    assert result.comparison["coverage_reliable"] is False
    inactive = rfm.rfm_reference(**_options(source, customer_key="inactive"))
    assert not _feature(inactive, None).in_base and _feature(inactive, None).frequency.value == 0
    assert _feature(inactive, None, "LAST_24M").frequency.value is None


def test_ties_use_legacy_thresholds_and_two_orders_at_same_time(source):
    orders = [_order(f"order-{i}", owner=owner, amount=amount)
              for i, (owner, amount) in enumerate((("new", 10), ("new", 10), ("other", 20), ("third", 40)))]
    result = _reference(source, orders)
    assert _feature(result).frequency.value == 2
    rub = result.windows["LAST_12M"]["currencies"]["RUB"]
    assert rub["thresholds"]["M"]["values"] == empirical_thresholds({Decimal(20): 2, Decimal(40): 1})
    assert rub["thresholds"]["R"]["values"] == (1, 1, 1, 1)
    assert rub["monetary"]["buyers"] == 3 and rub["monetary"]["median"] == 20
    assert rub["recency"]["distribution"] == [[1, 3]]
    assert _feature(result).scores == (5, 4, 1)
    second = rfm.rfm_reference(**_options(source, customer_key="other"))
    assert _feature(second).scores[2] == _feature(result).scores[2]
    assert sum(count for _, count, _ in rub["segments"]) == 3


def test_end_to_end_schema4_parity_including_personal_quality(source, monkeypatch):
    root, catalog, _, scratch, target = source
    paid = _order("paid", amount=Decimal("100.000000000000000000000001"))
    returned = _order("paid", "Return")
    returned["lines"][0]["priceOfLine"] = paid["lines"][0]["priceOfLine"]
    conflicted = _order("date-conflict", owner="other")
    alternate = deepcopy(conflicted)
    alternate["firstAction"]["dateTimeUtc"] = "2026-08-29T01:00:00Z"
    _write_orders(source, [
        paid, paid, returned, conflicted, alternate,
        _order("cp", "CP", amount=5000), _order("old", stamp="2025-01-31T10:00:00Z", amount=7),
        _order("kzt", "sold", owner="other", namespace="kanzlerKz", amount=0),
        _order("buy-from-alias", owner="old", amount=3),
        _order("return-only", "Return", owner="inactive"),
    ])
    index.build_unified_index(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    with index.reference_observations(raw_root=root, catalog_path=catalog) as observations:
        reference = {row.source_ref: row for row in observations}
    sid_to_owner = {}
    with index.open_unified_index(raw_root=root, catalog_path=catalog, index_dir=target) as reader:
        for row in reader.orders():
            source_row = reference[row.observation.source_ref]
            sid = str(row.observation.customer_key.value)
            owner = source_row.customer_key.value
            assert sid_to_owner.setdefault(sid, owner) == owner
    assert set(sid_to_owner.values()) == {"new", "other", "inactive"}
    assert len(sid_to_owner) == len(set(sid_to_owner.values()))

    def no_raw(*args, **kwargs):
        pytest.fail("INDEXED RFM must use the schema 4 observation stream")

    for sid, owner in sid_to_owner.items():
        expected = rfm.rfm_reference(**_options(source, customer_key=owner))
        with monkeypatch.context() as patch:
            patch.setattr(index, "_records", no_raw)
            patch.setattr(inputs, "_records", no_raw)
            actual = rfm.rfm_indexed(**_options(source, customer_key=sid), index_dir=target)
        assert expected == replace(actual, context=replace(actual.context, identity_space=expected.context.identity_space))
    assert not list(scratch.iterdir())


def test_cancellation_during_ranking_discards_results_and_scratch(source, monkeypatch):
    _, _, _, scratch, _ = source
    _write_orders(source, [_order("paid")])
    rank = rfm._thresholds
    running = True

    def cancel_during_ranking(*args):
        nonlocal running
        running = False
        return rank(*args)

    monkeypatch.setattr(rfm, "_thresholds", cancel_during_ranking)
    with pytest.raises(InterruptedError):
        rfm.rfm_reference(**_options(source, cancelled=lambda: not running))
    assert not list(scratch.iterdir())
    monkeypatch.setattr(rfm, "_thresholds", rank)
    assert _feature(rfm.rfm_reference(**_options(source))).frequency.value == 1


def test_no_orders_keep_general_population_without_inventing_cutoff(source):
    root, _, data, _, target = source
    data["orders"].clear()
    data["manual_interactions"] = None
    save(root, data)
    result = rfm.rfm_reference(**_options(source, selected_end=None))
    assert result.population == 3 and result.base_buyers == 0
    assert result.context.as_of_date is None and result.windows == {}
    with pytest.raises(StatisticsIndexError):
        rfm.rfm_indexed(**_options(source, selected_end=None), index_dir=target)


def test_unknown_currency_keeps_behavior_and_blocks_full_money(source):
    ambiguous = _order("ambiguous", namespace="unrecognized")
    result = _reference(source, [ambiguous, _order("paid", amount=35)])
    assert _feature(result, None).frequency.value == 2
    rub = _feature(result)
    assert rub.frequency.value == 1 and rub.known_subtotal == 35
    assert rub.monetary.value == 35
    assert result.windows["LAST_12M"]["quality"]["unvalued_orders"] == 1


def test_comparison_retains_all_groups_and_zero_for_absent_valued_currency(source):
    prior = "2025-05-01T00:00:00Z"
    result = _reference(source, [
        _order("now", amount=30), _order("before", stamp=prior, amount=10),
        _order("recent-only", owner="other", amount=10),
        _order("previous-only", owner="third", stamp=prior, amount=10),
        _order("older", owner="older", stamp="2023-08-10T00:00:00Z", amount=40),
    ])
    comparison = result.comparison
    assert comparison["observed_groups"] == dict(BOTH=1, LAST_ONLY=1, PREVIOUS_ONLY=1, NEITHER=1)
    rub = comparison["money"]["RUB"]
    assert (rub["last_known_amount"], rub["previous_known_amount"], rub["known_amount_change"]) == (40, 20, 20)
    assert rub["known_amount_change_percent"] == 100
    kzt = comparison["money"]["KZT"]
    assert kzt["last_known_amount"] == kzt["previous_known_amount"] == 0
    assert kzt["known_amount_change_percent"] is None


def test_action_only_customer_belongs_to_population_not_rfm_base(source):
    root, _, data, _, _ = source
    action_dir = root / data["manual_interactions"]["actions"]["directory"]
    view = action(DEFAULT_SELECTION.view_action_system_names[0], ["000001-size"], "action-only", stamp=_RECENT)
    (action_dir / "actions_part_001.json").write_text(encode({"customerActions": [view]}), encoding="utf-8")
    result = _reference(source, [_order("paid")], customer_key="action-only")
    assert result.population == 4 and result.base_buyers == 1
    assert _feature(result, None).frequency.value == 0
    assert _feature(result, None).recency.value is None
    assert not _feature(result, None).in_base
