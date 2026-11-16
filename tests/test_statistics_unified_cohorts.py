"""Synthetic unified cohort acceptance, including a real schema 4 projection."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from Application import statistics_unified_aggregate_input as inputs
from Application import statistics_unified_cohorts as cohorts
from Application import statistics_unified_index as index
from Application.statistics_storage_safety import path_identity
from Application.statistics_unified_policy import EvidenceQuality as Quality
from tests import test_statistics_unified_rfm as rfm_tests
from tests.test_dataset_statistics import put_entry, save


@pytest.fixture
def source(tmp_path, monkeypatch):
    result = rfm_tests.source.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(cohorts, "validate_storage_root", lambda path, **kwargs: path_identity(path))
    return result


def _order(key, stamp, status="F", *, owner="new", amount=20, product="000001", namespace="offline1C", channel="shop"):
    value = rfm_tests._order(key, status, stamp=stamp, owner=owner, amount=amount, namespace=namespace)
    value["lines"][0]["product"]["ids"][namespace] = product + "-size"
    if channel is None:
        del value["firstAction"]["channel"]
    else:
        value["firstAction"]["channel"] = {"ids": {"externalId": channel}, "name": channel.title()}
    return value


def _options(source, **overrides):
    return rfm_tests._options(source, **overrides)


def _reference(source, orders, **options):
    rfm_tests._write_orders(source, orders)
    return cohorts.cohorts_reference(**_options(source, **options))


@pytest.mark.parametrize(("status", "buyers"), [("F", 1), ("sold", 1), ("CP", 1), ("delivering", 0), ("Return", 0)])
def test_purchase_evidence_controls_entry_into_cohort(source, status, buyers):
    order = _order("order", "2024-01-31T22:10:00-03:00", status)
    order["dateTimeUtc"] = "2026-01-01T00:00:00Z"
    result = _reference(source, [order])
    assert result.identified_order_customers == 1
    assert result.behavior["base_buyers"] == buyers
    assert result.context.policy.policy_id == "UNIFIED_PURCHASE_V2"
    assert result.context.policy.time_basis_id == "ORDER_CREATED_PURCHASE_PROXY_V1"
    if buyers:
        assert result.customer.first.value.created_at == datetime(2024, 2, 1, 1, 10, tzinfo=timezone.utc)
        assert result.customer.second.value is None
        assert result.behavior["cohorts"]["2024-02"]["retention"][0]["buyers"] == 1
    else:
        assert result.customer.first.reasons == ("NO_OBSERVED_PURCHASE",)
        assert result.behavior["summary"]["horizons"]["30"]["repeat_rate"] is None


@pytest.mark.parametrize("transition", [("CP", "F"), ("F", "Return")])
def test_changed_and_identical_snapshots_preserve_one_purchase(source, transition):
    first, changed = [_order("single", "2024-01-04T00:00:00Z", status) for status in transition]
    extra = deepcopy(first["lines"][0])
    extra.update(id="line2", number=2)
    first["lines"].append(extra)
    changed["lines"].append(dict(extra, status=changed["lines"][0]["status"]))
    a = _reference(source, [first, first, changed])
    b = _reference(source, [changed, first, first])
    assert a == replace(b, context=a.context)
    assert a.behavior["base_buyers"] == int(transition[-1] == "F")
    assert a.behavior["summary"]["observed_repeat_buyers"] == 0
    assert a.customer.second.value is None
    if transition[-1] == "F":
        amount = a.monetary["cohorts"]["2024-01"]["total"]["currencies"]["RUB"]["recognized"]
        assert amount.total.value == amount.known == 40
    else:
        assert not a.monetary["cohorts"]


@pytest.mark.parametrize("distance", [30, 60, 90, 180, 365, 366])
def test_inclusive_repeat_horizons_and_calendar_retention(source, distance):
    first = datetime(2024, 1, 1, tzinfo=timezone.utc)
    last = first + timedelta(days=distance, hours=23)
    second = _order("second", last.isoformat(), "sold")
    result = _reference(source, [_order("first", first.isoformat(), "CP"), second, deepcopy(second)])
    row = result.behavior["cohorts"]["2024-01"]
    assert row["second_purchase_days"]["distribution"] == [[distance, 1]]
    for horizon in (30, 60, 90, 180, 365):
        cell = row["horizons"][str(horizon)]
        assert cell["eligible"] == 1 and cell["repeated"] == int(distance <= horizon)
        assert cell["repeat_rate"] == 100 * int(distance <= horizon)
    assert row["retention"][0]["buyers"] == 1
    assert row["retention"][1]["active"] == int(last.month == 2 and last.year == 2024)
    assert row["retention"][12]["active"] == int(last.month == 1 and last.year == 2025)
    if distance == 30:
        assert row["horizons"]["30"]["repeated"] == 1 and row["retention"][1]["active"] == 0


def test_calendar_activity_counts_buyer_once_and_m12_is_inclusive(source):
    first = _order("first", "2024-01-25T00:00:00Z")
    rows = [first, _order("repeat", "2024-02-05T02:00:00Z"), _order("same-month", "2024-02-06T02:00:00Z"),
            _order("m12", "2025-01-07T02:00:00Z"), _order("m13", "2025-02-07T02:00:00Z")]
    result = _reference(source, rows)
    cohort = result.behavior["cohorts"]["2024-01"]
    assert cohort["retention"][1]["active"] == 1
    assert cohort["retention"][12]["active"] == 1
    assert cohort["retention"][2]["active"] == 0
    assert cohort["horizons"]["30"]["repeated"] == 1
    assert len(cohort["retention"]) == 13


def test_display_bounds_are_applied_after_searching_all_history(source):
    older = _order("older", "2024-11-10T00:00:00Z", product="000002")
    repeat = _order("repeat", "2026-03-01T00:00:00Z")
    recent = _order("recent", "2026-04-11T00:00:00Z", owner="other")
    result = _reference(source, [older, repeat, recent], selected_start="2026-01-01")
    assert result.behavior["base_buyers"] == 2 and result.behavior["hidden_buyers"] == 1
    assert set(result.behavior["cohorts"]) == {"2026-04"}
    assert result.customer.first.value.category_codes == ("cat-B",)
    assert result.customer.second.value.category_codes == ("cat-A",)
    options = _options(source, selected_items=["000001"])
    del options["selected_end"], options["customer_key"]
    orders = inputs.aggregate_reference(**options)
    assert orders.contexts[1].product_filter_key is not None
    assert result == replace(cohorts.cohorts_reference(**_options(source, selected_start="2026-01-01")),
                             context=result.context)


def test_censoring_and_incomplete_coverage_preserve_observed_repeat(source):
    root, _, data, _, _ = source
    manual = data["manual_interactions"]
    for entry in (manual, manual["orders"], manual["actions"]):
        entry["since"] = "2025-09-01T00:00:00+00:00"
    data["orders"]["2025-08-20"] = put_entry(root, "orders", [], 42,
                                            "2025-08-20T00:00:00+00:00", "2025-08-21T00:00:00+00:00")
    save(root, data)
    result = _reference(source, [
        _order("early", "2025-08-20T00:00:00Z"), _order("early-repeat", "2025-09-03T00:00:00Z"),
        _order("late", "2026-08-20T00:00:00Z", owner="other"),
        _order("late-repeat", "2026-08-25T00:00:00Z", owner="other"),
    ])
    early = result.behavior["cohorts"]["2025-08"]
    assert early["horizons"]["30"]["eligible"] == 0
    assert early["horizons"]["30"]["coverage_incomplete"] == 1
    assert early["horizons"]["30"]["observed_repeated_ineligible"] == 1
    assert early["horizons"]["30"]["rate_quality"].quality is Quality.INSUFFICIENT_COVERAGE
    late = result.behavior["cohorts"]["2026-08"]
    assert late["horizons"]["30"]["censored"] == 1
    assert late["horizons"]["30"]["repeat_rate"] is None
    assert late["retention"][1]["censored"] == 1 and late["retention"][1]["retention_rate"] is None


@pytest.mark.parametrize("missing", [False, True])
def test_unresolved_first_order_excludes_misleading_cohort(source, missing):
    dated = _order("dated", "2024-03-03T00:00:00Z")
    a = _order("uncertain", None if missing else "2024-01-02T00:00:00Z")
    rows = [dated, a]
    if not missing:
        b = deepcopy(a)
        b["firstAction"]["dateTimeUtc"] = "2024-04-02T00:00:00Z"
        rows.append(b)
    result = _reference(source, rows)
    assert result.behavior["base_buyers"] == 0 and result.identified_order_customers == 1
    assert result.customer.observed_first.created_at.month == 3
    assert result.customer.first.value is None and result.customer.second.value is None
    assert result.diagnostics["first_purchase_unresolved"] == 1
    if missing:
        assert result.customer.first.reasons == ("CREATED_TIME_MISSING",)
        assert result.customer.first.quality is Quality.UNKNOWN
    else:
        assert result.customer.first.reasons == ("CREATED_TIME_CONFLICT",)
        assert result.customer.first.quality is Quality.CONFLICT


def test_uncertain_second_preserves_dated_repeat_but_not_false_inactive_cells(source):
    a = _order("first", "2024-01-01T00:00:00Z")
    b = _order("possible-second", "2024-01-20T00:00:00Z")
    other = deepcopy(b)
    other["firstAction"]["dateTimeUtc"] = "2024-02-03T00:00:00Z"
    known = _order("dated-repeat", "2024-03-10T00:00:00Z")
    result = _reference(source, [a, b, other, known])
    assert result.customer.first.value is not None
    assert result.customer.second.value is None and result.customer.observed_second.created_at.month == 3
    assert result.customer.second.reasons == ("CREATED_TIME_CONFLICT",)
    row = result.behavior["cohorts"]["2024-01"]
    assert row["observed_repeat_buyers"] == 1 and row["second_purchase_days"]["buyers"] == 0
    h30, h90 = (row["horizons"][name] for name in ("30", "90"))
    assert h30["eligible"] == 1 and h30["confirmed_without_event"] == 0
    assert h30["repeat_rate"] is None and h30["uncertain_eligible_buyers"] == 1
    assert h90["eligible"] == h90["repeated"] == 1 and h90["repeat_rate"] == 100
    m1 = row["retention"][1]
    assert m1["eligible"] == 1 and m1["retention_rate"] is None and m1["uncertain_eligible_buyers"] == 1
    assert result.behavior["summary"]["retention"][1]["rate_quality"].quality is Quality.CONFLICT
    money = result.monetary["horizons"]["90"]["currencies"]["RUB"]["recognized"]
    assert money.total.value is None and money.known is None and "CREATED_TIME_CONFLICT" in money.total.reasons
    assert result.purchase_categories["unknown_second_buyers"] == 1


def test_decimal_money_separates_cp_nominal_unknown_and_real_zero(source):
    stamp = "2024-01-01T00:00:00Z"
    amount = Decimal("8000.00000000000000000000000001")
    paid = _order("paid", stamp, amount=amount)
    cp = _order("second", "2024-01-02T00:00:00Z", "CP", amount=4000)
    kzt = _order("kzt", "2024-01-05T00:00:00Z", owner="other", amount=3000, namespace="kanzlerKz")
    free = _order("free", stamp, owner="free", amount=0)
    result = _reference(source, [paid, cp, cp, kzt, free])
    rows = result.monetary["cohorts"]["2024-01"]
    total = rows["total"]
    assert (total["orders"], total["unvalued_orders"], total["fully_valued_buyers"], total["unvalued_buyers"]) == (4, 0, 3, 0)
    rub, kzt = (total["currencies"][currency] for currency in ("RUB", "KZT"))
    assert rub["recognized"].known == rub["recognized"].total.value == Decimal("12000.00000000000000000000000001")
    assert rub["partial_nominal"].total.value == 4000
    assert rub["valued_orders"] == 3 and rub["fully_valued_buyers"] == 2
    assert rub["mean_per_valued_buyer"] == Decimal("6000.000000000000000000000000005")
    assert kzt["recognized"].total.value == 3000
    assert kzt["fully_valued_buyers"] == 1 and kzt["mean_per_valued_buyer"] == 3000
    assert rows["first"]["currencies"]["RUB"]["recognized"].total.value == amount
    h30 = result.monetary["horizons"]["30"]
    assert h30["unvalued_orders"] == 0 and h30["buyers"] == 1
    assert h30["currencies"]["RUB"]["partial_nominal"].total.value == 4000
    result = _reference(source, [free])
    assert result.monetary["cohorts"]["2024-01"]["total"]["currencies"]["RUB"]["recognized"].total.value == 0


def test_same_timestamp_uses_stable_rank_and_direct_category_sets(source):
    stamp = "2024-01-01T00:00:00Z"
    first = _order("a-first", stamp, "CP", channel="web")
    second = _order("b-second", stamp, channel=None, product="000002")
    for n, product in enumerate(("000002", "000001", "999999"), 2):
        row = deepcopy(first["lines"][0])
        row.update(id=str(n), number=n, product={"ids": {"offline1C": product + "-size"}})
        first["lines"].append(row)
    result = _reference(source, [second, first, first])
    assert result.customer.first.value.order_sort_rank < result.customer.second.value.order_sort_rank
    assert result.behavior["summary"]["second_purchase_days"]["distribution"] == [[0, 1]]
    observed = result.customer.first.value
    assert observed.channel.value == ("web", "Web")
    assert result.customer.second.value.channel.reasons == ("CHANNEL_MISSING",)
    assert observed.category_codes == ("cat-A", "cat-B")
    assert observed.category_coverage.reasons == ("PRODUCT_SCOPE_UNRESOLVED",)
    pairs = {(x["first_code"], x["second_code"]): x["buyers"] for x in result.purchase_categories["transitions"]}
    assert pairs == {("cat-A", "cat-B"): 1, ("cat-B", "cat-B"): 1, (None, "cat-B"): 1}
    assert result.purchase_channels["unknown_transition_buyers"] == 1


def test_conflicting_owner_does_not_enter_cohort_twice(source):
    stamp = "2024-01-01T00:00:00Z"
    result = _reference(source, [
        _order("shared", stamp), _order("shared", stamp, owner="other"),
        _order("only", stamp, owner="old"),
    ])
    assert result.identified_order_customers == result.behavior["base_buyers"] == 1
    assert result.diagnostics["unattributed_purchase_orders"] == result.diagnostics["OWNER_CONFLICT"] == 1


def test_schema4_parity_compares_matrices_and_sequences(source, monkeypatch):
    root, catalog, _, scratch, target = source
    start = _order("start", "2024-01-31T12:00:00Z", "CP", amount=500, channel="site")
    second = _order("repeat", "2024-02-04T12:00:00Z", product="000002", channel="store")
    later_return = deepcopy(second)
    later_return["lines"][0]["status"]["ids"]["externalId"] = "Return"
    late = _order("late", "2026-08-30T12:00:00Z", "sold", owner="other", namespace="kanzlerKz")
    undated = _order("unresolved", None, owner="uncertain")
    dated = _order("known", "2024-02-04T12:00:00Z", owner="uncertain")
    rfm_tests._write_orders(source, [start, start, second, later_return, late, undated, dated])
    index.build_unified_index(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    with index.reference_observations(raw_root=root, catalog_path=catalog) as observations:
        source_owners = {row.source_ref: row.customer_key.value for row in observations}
    mapping = {}
    with index.open_unified_index(raw_root=root, catalog_path=catalog, index_dir=target) as reader:
        for row in reader.orders():
            sid = str(row.observation.customer_key.value)
            owner = source_owners[row.observation.source_ref]
            assert mapping.setdefault(sid, owner) == owner
    assert set(mapping.values()) == {"new", "other", "uncertain"} and len(mapping) == 3

    def no_raw(*args, **kwargs):
        pytest.fail("INDEXED must read the pinned observation and category projection")

    for sid, owner in mapping.items():
        expected = cohorts.cohorts_reference(**_options(source, customer_key=owner))
        with monkeypatch.context() as patch:
            patch.setattr(index, "_records", no_raw)
            patch.setattr(cohorts.reference, "_catalog_snapshot", no_raw)
            actual = cohorts.cohorts_indexed(**_options(source, customer_key=sid), index_dir=target)
        assert expected == replace(actual, context=replace(actual.context, identity_space=expected.context.identity_space))
    assert not list(scratch.iterdir())


def test_cancellation_during_matrix_calculation_discards_workspace(source, monkeypatch):
    _, _, _, scratch, _ = source
    rfm_tests._write_orders(source, [_order("first", "2024-01-01T00:00:00Z")])
    collect = cohorts._Buyers._collect
    cancelled = False

    def cancel(*args):
        nonlocal cancelled
        cancelled = True
        return collect(*args)

    monkeypatch.setattr(cohorts._Buyers, "_collect", cancel)
    with pytest.raises(InterruptedError):
        cohorts.cohorts_reference(**_options(source, cancelled=lambda: cancelled))
    assert not list(scratch.iterdir())
    monkeypatch.setattr(cohorts._Buyers, "_collect", collect)
    assert cohorts.cohorts_reference(**_options(source)).behavior["base_buyers"] == 1
