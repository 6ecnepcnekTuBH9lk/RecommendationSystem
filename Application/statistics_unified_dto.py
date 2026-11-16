"""Versioned statistics report and explicit dispatch for saved report versions."""

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
import json
import math
from pathlib import Path
import re

from Application import statistics_cache as legacy
from Application.order_statistics import CURRENCIES
from Application.statistics_cohorts_core import HORIZONS, MONTH_OFFSETS
from Application.statistics_index import manifest_digest
from Application.statistics_rfm_core import SEGMENTS, WINDOWS, rfm_windows
from Application.statistics_rfm_validation import rfm_cutoff
from Application.statistics_unified_aggregates import AggregateMetric
from Application.statistics_unified_facts import OrderKind, _sum
from Application.statistics_unified_policy import EvidenceQuality, Measured, PurchasePolicy, STATUS_CATALOG, StatusGroup
from Application.statistics_unified_projection import OBSERVATION_VERSION, PROJECTION_ID, SCHEMA_VERSION as INDEX_SCHEMA
from Application.statistics_unified_rfm import SEGMENTATION_POLICY


RESULT_SCHEMA = CACHE_SCHEMA = 13
SLICE_SCHEMA = DETAIL_FORMAT = 2
POLICY = asdict(PurchasePolicy())
CAPABILITIES = dict(order_created_proxy=True, execution_time=False, historical_as_of=False,
                    cp_allocation=False, split_lineage=False, refund_ledger=False)
CONTEXTS = ("raw_orders", "order_product_slice", "item_interactions", "customer_population",
            "buyer_period", "rfm_48m", "cohorts_history")
DETAIL_CONTEXTS = ("order_product_slice", "buyer_period", "rfm_48m")
PARAMETERS = ("order_period", "buyer_period", "customer_period", "product_filter_key", "selected_start", "selected_end")
VERSIONS = dict(quantity_policy_version=1, projection_version=PROJECTION_ID, observation_version=OBSERVATION_VERSION,
                cache_schema=CACHE_SCHEMA, slice_schema=SLICE_SCHEMA, rfm_segmentation=SEGMENTATION_POLICY)
_AMOUNT_FIELDS = {"source_nominal", "known_purchase_subtotal", "recognized_purchase", "partial_nominal",
                  "cancelled_nominal", "returned_nominal", "confirmed_refund", "net_after_refunds"}
_COUNTERS = ("raw_orders", "raw_lines", "exact_duplicates", "business_equivalent", "snapshot_conflicts",
             "undated_orders", "orders_outside_coverage", "unassigned_buyers")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class StatisticsResultError(ValueError):
    """Safe format/integrity error, without source contents."""


@dataclass(frozen=True, slots=True, repr=False)
class UnifiedResult:
    policy: dict
    calculated_at: str
    metadata: dict
    source: dict
    contexts: dict
    coverage: dict
    capabilities: dict
    actions: dict
    orders: dict
    products: dict
    customers: dict
    money: dict
    status_catalog: dict
    rfm: dict
    cohorts: dict
    diagnostics: dict

    def envelope(self):
        return {"schema_version": RESULT_SCHEMA, "result": {field.name: getattr(self, field.name) for field in fields(self)}}


def _wire(value):
    if isinstance(value, AggregateMetric):
        return dict(known=_wire(value.known), **_wire(value.total))
    if isinstance(value, Measured):
        return dict(value=_wire(value.value), quality=value.quality.value, reasons=list(value.reasons))
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _wire(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _wire(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire(part) for part in value]
    return value


def fingerprint(source, source_binding, calculated_at, parameters, cutoff, algorithm):
    return manifest_digest(dict(source=source, source_binding=source_binding, calculated_at=calculated_at,
                                parameters=parameters, cutoff=cutoff, algorithm=algorithm, policy=POLICY,
                                versions=VERSIONS, result_schema=RESULT_SCHEMA))


def _ratio(numerator, denominator):
    if numerator.total.value is None or denominator.total.value is None:
        issues = [value.total for value in (numerator, denominator) if value.total.value is None]
        return Measured(None, issues[0].quality, tuple({reason for value in issues for reason in value.reasons}))
    if not denominator.total.value:
        return Measured(None, EvidenceQuality.UNKNOWN, ("NO_OBSERVED_ORDERS",))
    return Measured(100 * numerator.total.value / denominator.total.value, EvidenceQuality.KNOWN)


def _section(context, data):
    return dict(context_id=context, data=data, quality={"basis": "OBSERVED_MEASUREMENTS"})


def _currency_rows(rows):
    return {row.currency: _wire(row) for row in rows}


def assemble_result(aggregates, rfm, cohorts, *, item_metadata, source, source_binding, parameters, algorithm, check):
    context_list = (*aggregates.contexts, rfm.context, cohorts.context)
    first = context_list[0]
    if ({context.context_id for context in context_list} != set(CONTEXTS)
            or any((context.policy, context.calculated_at, context.provenance_digest, context.identity_space)
                   != (first.policy, first.calculated_at, first.provenance_digest, first.identity_space)
                   for context in context_list) or first.provenance_digest != source["provenance_digest"]):
        raise StatisticsResultError("Unified sections do not share a pinned source")
    # No private keys leave the reader. The report identity is bound to the
    # canonical dataset, independent of how each reader represented identities.
    contexts = {context.context_id: dict(_wire(context), identity_space=first.provenance_digest) for context in context_list}
    calculated_at = first.calculated_at.isoformat()
    cutoff = _wire(rfm.context.as_of_date)
    metadata = dict(**VERSIONS, calculation_algorithm=algorithm, source_binding=source_binding,
                    parameters=parameters, fingerprint=fingerprint(source, source_binding, calculated_at,
                                                                   parameters, cutoff, algorithm),
                    detail=dict(format_version=DETAIL_FORMAT, available=source["reader"] == "INDEXED",
                                context_ids=list(DETAIL_CONTEXTS if source["reader"] == "INDEXED" else CONTEXTS),
                                provenance_digest=first.provenance_digest, policy_id=POLICY["policy_id"]))
    counts = aggregates.orders
    orders = dict(context_id="order_product_slice", unique_orders=counts["orders"], positions=counts["positions"],
                  purchase_orders=counts["purchase_orders"], purchase_positions=counts["purchase_positions"],
                  buyers=counts["buyers"], state_distribution=aggregates.order_states, state_flags=aggregates.order_flags,
                  cancellations=dict(positions=counts["cancelled_positions"], orders=counts["cancelled_orders"],
                                     only_orders=counts["fully_cancelled_orders"], mixed_orders=counts["mixed_cancelled_orders"],
                                     customers=counts["cancelled_customers"]),
                  return_states=dict(positions=counts["returned_positions"], orders=counts["returned_orders"],
                                     only_orders=counts["all_returned_orders"], mixed_orders=counts["mixed_returned_orders"],
                                     customers=counts["returned_customers"], returned_units=counts["returned_units"]),
                  mixed_states=dict(orders=counts["mixed_orders"],
                                    unresolved_state_share=_ratio(counts["unknown_state_orders"], counts["orders"])),
                  indicators={key: counts[key] for key in ("partial_orders", "partial_positions", "in_progress_positions",
                      "orders_without_evidence", "unresolved_purchase_orders", "mixed_slice_orders", "conflicting_orders",
                      "unknown_state_orders", "incomplete_structure_orders", "unresolved_scope_orders")})
    orders["breakdowns"] = aggregates.breakdowns["orders"]
    money = dict(context_id="order_product_slice", currencies=_currency_rows(aggregates.order_money),
                 unassigned_currency_orders=counts["unassigned_currency_orders"])
    status_rows = []
    for row in aggregates.statuses:
        check()
        info = STATUS_CATALOG.get(row.code)
        status_rows.append(dict(external_id=row.code, bucket=row.bucket, name=row.name, mindbox_category=row.mindbox_category,
                                business_group=row.business_group, metadata_confirmation=info.metadata_confirmation if info else None,
                                positions=row.positions, orders=row.orders, raw_occurrences=row.raw,
                                historical_positions=row.historical_positions))
    catalog = dict(context_id="order_product_slice", raw_context_id="raw_orders",
                   catalog_id=POLICY["status_catalog_id"], rows=status_rows)
    products = []
    for row in aggregates.products:
        check()
        item = item_metadata[row.item_id]
        products.append(dict(_wire(row), name=item.name, category=item.category, money=_currency_rows(row.money)))
    purchases = dict(context_id="buyer_period", buyer_money=_currency_rows(aggregates.buyer_money),
                     frequency=aggregates.frequency, first_months=aggregates.first_months,
                     second_intervals=aggregates.second_intervals, first_categories=aggregates.first_categories,
                     second_categories=aggregates.second_categories, categories=aggregates.categories, pairs=aggregates.pairs)
    customer_counts = {key: aggregates.customers[key] for key in
                       ("population", "population_buyers", "nonpurchase_clients", "cancelled_clients", "returned_clients")}
    purchases["counts"] = {key: value for key, value in aggregates.customers.items() if key not in customer_counts}
    customers = dict(counts=customer_counts, purchases=purchases,
                     customer_demography=aggregates.breakdowns["customer_demography"],
                     customer_activity=aggregates.breakdowns["customer_activity"],
                     buyer_breakdowns=aggregates.breakdowns["buyer_breakdowns"],
                     observed_history=dict(context_id="raw_orders", data=aggregates.observed_history))
    rfm_data = dict(context_id="rfm_48m", as_of_date=cutoff, base_buyers=rfm.base_buyers, windows=rfm.windows,
                    comparison=rfm.comparison, quality=dict(availability="AVAILABLE" if cutoff else "NO_ORDERS_COVERAGE",
                        population=rfm.population, dual_currency_buyers=rfm.dual_currency_buyers,
                        unassigned_currency_buyers=rfm.unassigned_currency_buyers, counters=rfm.diagnostics))
    cohorts_data = dict(context_id="cohorts_history", as_of_date=cohorts.context.as_of_date,
                        cohort_bounds=cohorts.diagnostics.get("cohort_bounds", {"start": None, "end": None}),
                        coverage=cohorts.diagnostics.get("effective_coverage", []),
                        months=cohorts.behavior.get("cohorts", {}), summary=cohorts.behavior.get("summary", {}),
                        money=cohorts.monetary, channels=cohorts.purchase_channels, categories=cohorts.purchase_categories,
                        quality=dict(cohorts.diagnostics, identified_order_customers=cohorts.identified_order_customers,
                                     base_buyers=cohorts.behavior.get("base_buyers", 0),
                                     displayed_buyers=cohorts.behavior.get("displayed_buyers", 0),
                                     hidden_buyers=cohorts.behavior.get("hidden_buyers", 0)))
    diagnostics = dict(context_id="raw_orders", counters={key: value for key, value in aggregates.diagnostics.items()
                       if not key.isupper()}, reason_counts={key: value for key, value in aggregates.diagnostics.items()
                                                           if key.isupper()})
    diagnostics["counters"].update(snapshot_conflicts=aggregates.diagnostics.get("conflicting_snapshots", 0),
                                   unassigned_buyers=aggregates.diagnostics.get("unattributed_purchase_orders", 0))
    for key in _COUNTERS:
        diagnostics["counters"].setdefault(key, 0)
    diagnostics["scope_by_counter"] = {key: ("item_interactions" if key.startswith(("unresolved_view", "unresolved_favorite"))
        else "order_product_slice" if key in ("products_with_purchases", "unresolved_purchase_items")
        else "buyer_period" if key in ("unattributed_orders", "owner_conflict_orders", "unassigned_buyers",
            "unattributed_purchase_orders", "orders_with_unresolved_categories", "first_category_unresolved",
            "second_category_unresolved") else "raw_orders") for key in diagnostics["counters"]}
    diagnostics["reason_stage"] = {
        key: "ACTION_INPUT_BEFORE_FILTER" if key.startswith("ACTION_") else "ORDER_RECONCILIATION"
        for key in diagnostics["reason_counts"]
    }
    result = UnifiedResult(
        dict(POLICY), calculated_at, metadata, source, contexts,
        {key: dict(orders=value["orders_intervals"], actions=value["actions_intervals"],
                   basis=value["coverage_basis"]) for key, value in contexts.items()}, dict(CAPABILITIES),
        _wire(dict(_section("item_interactions", aggregates.actions), breakdowns=aggregates.breakdowns["actions"])), _wire(orders),
        _wire(dict(_section("order_product_slice", products), breakdowns=aggregates.breakdowns["products"])), _wire(_section("customer_population", customers)),
        _wire(money), _wire(catalog), _wire(rfm_data), _wire(cohorts_data), diagnostics,
    )
    metadata["content_digest"] = content_digest(result.envelope()["result"])
    return validate_unified(result, check=check)


def content_digest(data):
    meta = {key: value for key, value in data["metadata"].items() if key != "content_digest"}
    return manifest_digest(dict(data, metadata=meta))


def _require(condition, code):
    if not condition:
        raise StatisticsResultError(code)


def _keys(value, names):
    _require(isinstance(value, dict) and set(value) == set(names.split()), "INVALID_RESULT_FIELDS")


def _count(value):
    return type(value) is int and value >= 0


def _decimal(value):
    return isinstance(value, str) and _DECIMAL.fullmatch(value) is not None


def _digest(value):
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _measure(value, test, *, known=False):
    _keys(value, "known value quality reasons" if known else "value quality reasons")
    status, amount, reasons = value["quality"], value["value"], value["reasons"]
    _require(status in {quality.value for quality in EvidenceQuality} and isinstance(reasons, list)
             and all(isinstance(x, str) and x for x in reasons)
             and reasons == sorted(set(reasons)), "INVALID_QUALITY")
    _require(test(amount) if status == "KNOWN" else amount is None and bool(reasons), "INVALID_MEASUREMENT")
    if known:
        part = value["known"]
        _require((part is None or test(part)) and (status != "KNOWN" or amount == part), "INVALID_KNOWN_SUBTOTAL")


def _count_cell(value):
    _measure(value, _count, known=True)
    _require(value["known"] is not None, "MISSING_COUNT_SUBTOTAL")
    return value["known"]


def _amount_cell(value):
    _measure(value, lambda amount: _decimal(amount) and Decimal(amount) >= 0, known=True)


def _finite(value):
    return type(value) in (int, float) and (type(value) is int or math.isfinite(value))


def _walk(value, check):
    check()
    if isinstance(value, dict):
        if {"value", "quality", "reasons"} <= value.keys():
            _measure(value, lambda x: _finite(x) or _decimal(x), known="known" in value)
        _require(all(isinstance(key, str) and key not in {"customer_id", "order_id", "source_ref", "phone", "email"}
                     for key in value), "INVALID_RESULT_KEY")
        for part in value.values():
            _walk(part, check)
    elif isinstance(value, list):
        for part in value:
            _walk(part, check)
    else:
        _require(value is None or type(value) in (str, bool) or _finite(value), "INVALID_RESULT_VALUE")


def _timestamp(value):
    stamp = datetime.fromisoformat(value)
    _require(stamp.utcoffset() == timedelta(0) and stamp.isoformat() == value, "INVALID_UTC_TIMESTAMP")
    return stamp


def _day(value):
    if value is None:
        return None
    day = date.fromisoformat(value)
    _require(day.isoformat() == value, "INVALID_CALENDAR_DATE")
    return day


def _intervals(values, *, merged=True):
    _require(isinstance(values, list), "INVALID_COVERAGE")
    result = []
    for interval in values:
        _require(isinstance(interval, list) and len(interval) == 2, "INVALID_INTERVAL")
        start, end = map(_timestamp, interval)
        _require(start < end and (not merged or not result or result[-1][1] < start), "INVALID_COVERAGE")
        result.append((start, end))
    return tuple(result)


def _bounds(intervals):
    return [intervals[0][0].isoformat(), intervals[-1][1].isoformat()] if intervals else None


def _contexts(result):
    from Application.statistics_period import shared_intervals

    source, meta, contexts = result.source, result.metadata, result.contexts
    _require(set(contexts) == set(CONTEXTS) == set(result.coverage), "INVALID_CONTEXT_IDS")
    digest, params = source["provenance_digest"], meta["parameters"]
    for name, row in contexts.items():
        _keys(row, "context_id policy identity_space calculated_at as_of_date period product_filter_key orders_intervals "
                   "actions_intervals coverage_basis created_history_verified status_history_verified provenance_digest")
        _require((row["context_id"], row["policy"], row["identity_space"], row["calculated_at"], row["provenance_digest"])
                 == (name, POLICY, digest, result.calculated_at, digest), "CONTEXT_BINDING_MISMATCH")
        _require(row["coverage_basis"] == "DECLARED_ORDERS_PROXY_V1" and row["created_history_verified"] is False
                 and row["status_history_verified"] is False, "INVALID_TIME_BASIS")
        _intervals(row["orders_intervals"])
        _intervals(row["actions_intervals"])
        if row["period"] is not None:
            _intervals([row["period"]])
        _require(result.coverage[name] == dict(orders=row["orders_intervals"], actions=row["actions_intervals"],
                                               basis=row["coverage_basis"]), "CONTEXT_COVERAGE_MISMATCH")
        _require(row["product_filter_key"] == (params["product_filter_key"] if name in
                 ("order_product_slice", "item_interactions") else None), "CONTEXT_FILTER_MISMATCH")
    raw = contexts["raw_orders"]
    orders, actions = raw["orders_intervals"], raw["actions_intervals"]
    common = shared_intervals(actions, orders)
    cutoff = rfm_cutoff(_intervals(orders), params["selected_end"], _timestamp(result.calculated_at))
    last = rfm_windows(cutoff)["LAST_48M"] if cutoff else None
    end = datetime.combine(cutoff + timedelta(days=1), time.min, timezone.utc) if cutoff else None
    cohort_period = [_timestamp(orders[0][0]).isoformat(), end.isoformat()] if end and orders and _timestamp(orders[0][0]) < end else None
    expected = {
        "raw_orders": (None, orders, actions),
        "order_product_slice": (params["order_period"] or _bounds(common), _wire(common) if orders else [],
                                _wire(common) if actions else []),
        "item_interactions": (params["order_period"] or _bounds(common), _wire(common) if orders else [],
                              _wire(common) if actions else []),
        "customer_population": (params["customer_period"], orders, actions),
        "buyer_period": (params["buyer_period"] or _bounds(_intervals(orders)), orders, []),
        "rfm_48m": (_wire(last), orders, []),
        "cohorts_history": (cohort_period, orders, []),
    }
    for name, (period, order_intervals, action_intervals) in expected.items():
        row = contexts[name]
        as_of = _wire(cutoff) if name in ("rfm_48m", "cohorts_history") else None
        _require((row["period"], row["orders_intervals"], row["actions_intervals"], row["as_of_date"])
                 == (period, order_intervals, action_intervals, as_of), "CONTEXT_SCOPE_MISMATCH")
    return cutoff


def _currencies(rows):
    _require(isinstance(rows, dict) and set(rows) == set(CURRENCIES), "INVALID_CURRENCIES")


def _money(rows, *, orders=None):
    _currencies(rows)
    for currency, row in rows.items():
        _keys(row, "currency amounts valued_orders unvalued_orders valued_positions unvalued_positions mean_order "
                   "fully_valued_buyers unvalued_buyers mean_per_valued_buyer")
        _require(currency == row["currency"] and set(row["amounts"]) == _AMOUNT_FIELDS, "INVALID_MONEY_FIELDS")
        for amount in row["amounts"].values():
            _amount_cell(amount)
        for field in ("confirmed_refund", "net_after_refunds"):
            _require(row["amounts"][field] == dict(known=None, value=None, quality="UNKNOWN",
                     reasons=["LEDGER_UNAVAILABLE"]), "INVALID_REFUND_VALUE")
        _require(all(_count(row[name]) for name in ("valued_orders", "unvalued_orders", "valued_positions",
                 "unvalued_positions", "fully_valued_buyers", "unvalued_buyers")), "INVALID_MONEY_DENOMINATOR")
        if orders is not None:
            _require(row["valued_orders"] + row["unvalued_orders"] <= orders, "INVALID_MONEY_DENOMINATOR")
        _measure(row["mean_order"], _decimal)
        _require((row["mean_order"]["value"] is not None) == bool(row["valued_orders"]), "INVALID_MEAN_DENOMINATOR")
        if row["mean_per_valued_buyer"] is not None:
            _measure(row["mean_per_valued_buyer"], _decimal)
            _require(row["mean_per_valued_buyer"]["value"] is None or row["fully_valued_buyers"] > 0,
                     "INVALID_MEAN_DENOMINATOR")


def _orders_and_catalog(result, check):
    row = result.orders
    _keys({key: value for key, value in row.items() if key != "breakdowns"},
          "context_id unique_orders positions purchase_orders purchase_positions buyers state_distribution state_flags "
               "cancellations return_states mixed_states indicators")
    total, positions, purchases, lines, buyers = (_count_cell(row[key]) for key in
        ("unique_orders", "positions", "purchase_orders", "purchase_positions", "buyers"))
    # Historical purchase positions may survive a conflict in current status.
    _require(buyers <= purchases <= total and lines <= positions, "INVALID_ORDER_TOTALS")
    states = row["state_distribution"]
    _require(set(states) == {kind.value for kind in OrderKind} and sum(map(_count_cell, states.values())) == total,
             "INVALID_STATE_DISTRIBUTION")
    _keys(row["state_flags"], "completed partial cancelled returned in_progress unknown")
    for value in row["state_flags"].values():
        _require(_count_cell(value) <= total, "INVALID_STATE_FLAGS")
    for group in ("cancellations", "return_states"):
        block = row[group]
        expected = "positions orders only_orders mixed_orders customers"
        _keys(block, expected + (" returned_units" if group == "return_states" else ""))
        amount = _count_cell(block["orders"])
        _require(_count_cell(block["customers"]) <= amount <= total and
                 _count_cell(block["only_orders"]) + _count_cell(block["mixed_orders"]) <= amount
                 and _count_cell(block["positions"]) <= positions, "INVALID_STATE_TOTALS")
        if group == "return_states":
            _amount_cell(block["returned_units"])
            _require(block["returned_units"]["value"] is None, "UNVERIFIED_RETURNED_UNITS")
    _keys(row["mixed_states"], "orders unresolved_state_share")
    _require(_count_cell(row["mixed_states"]["orders"]) <= total, "INVALID_MIXED_ORDERS")
    _measure(row["mixed_states"]["unresolved_state_share"], lambda x: _finite(x) and 0 <= x <= 100)
    _keys(row["indicators"], "partial_orders partial_positions in_progress_positions orders_without_evidence "
          "unresolved_purchase_orders mixed_slice_orders conflicting_orders unknown_state_orders "
          "incomplete_structure_orders unresolved_scope_orders")
    indicators = {key: _count_cell(value) for key, value in row["indicators"].items()}
    _require(purchases + indicators["orders_without_evidence"] + indicators["unresolved_purchase_orders"] == total,
             "INVALID_PURCHASE_DISTRIBUTION")
    catalog = result.status_catalog
    _keys(catalog, "context_id raw_context_id catalog_id rows")
    _require(catalog["catalog_id"] == POLICY["status_catalog_id"] and catalog["raw_context_id"] == "raw_orders"
             and isinstance(catalog["rows"], list), "INVALID_STATUS_CATALOG")
    seen = set()
    for status in catalog["rows"]:
        check()
        _keys(status, "external_id bucket name mindbox_category business_group metadata_confirmation positions orders "
                      "raw_occurrences historical_positions")
        code, bucket = status["external_id"], status["bucket"]
        key = code, bucket
        _require(key not in seen, "DUPLICATE_STATUS")
        seen.add(key)
        if bucket == "CATALOG":
            definition = STATUS_CATALOG.get(code)
            _require(definition is not None and
                     (status["name"], status["mindbox_category"], status["business_group"], status["metadata_confirmation"])
                     == (definition.name, definition.mindbox_category, definition.group.value, definition.metadata_confirmation),
                     "INVALID_STATUS_CATALOG")
        else:
            _require((bucket == "UNKNOWN_CODE" and isinstance(code, str) and code not in STATUS_CATALOG
                      or code is None and bucket in {"MISSING_CODE", "INVALID_CODE", "UNRESOLVED"})
                     and status["business_group"] == StatusGroup.UNKNOWN.value and
                     all(status[x] is None for x in ("name", "mindbox_category", "metadata_confirmation")),
                     "INVALID_STATUS_CATALOG")
        _require(_count_cell(status["positions"]) <= positions and _count_cell(status["orders"]) <= total
                 and _count_cell(status["historical_positions"]) <= lines, "INVALID_STATUS_COUNTS")
        _count_cell(status["raw_occurrences"])
    _require({(code, "CATALOG") for code in STATUS_CATALOG} <= seen and
             {(None, bucket) for bucket in ("MISSING_CODE", "INVALID_CODE", "UNRESOLVED")} <= seen, "MISSING_STATUS_ROWS")
    _keys(result.money, "context_id currencies unassigned_currency_orders")
    _require(_count_cell(result.money["unassigned_currency_orders"]) <= purchases, "INVALID_UNASSIGNED_CURRENCIES")
    _money(result.money["currencies"], orders=purchases)


def _actions_products_customers(result, check):
    for section in (result.actions, result.products, result.customers):
        _keys({key: value for key, value in section.items()
               if key != "breakdowns" or section is result.customers}, "context_id data quality")
        _require(section["quality"] == {"basis": "OBSERVED_MEASUREMENTS"}, "INVALID_SECTION_QUALITY")
    actions = result.actions["data"]
    _keys(actions, "VIEW FAVORITE PURCHASE VIEW_users FAVORITE_users PURCHASE_users "
                   "VIEW_PURCHASE FAVORITE_PURCHASE VIEW_FAVORITE VIEW_FAVORITE_PURCHASE")
    counts = {name: _count_cell(cell) for name, cell in actions.items()}
    for kind in ("VIEW", "FAVORITE", "PURCHASE"):
        _require(counts[kind + "_users"] <= counts[kind], "INVALID_ACTION_USERS")
    for parts in (("VIEW", "PURCHASE"), ("FAVORITE", "PURCHASE"), ("VIEW", "FAVORITE"),
                  ("VIEW", "FAVORITE", "PURCHASE")):
        _require(counts["_".join(parts)] <= min(counts[kind + "_users"] for kind in parts), "INVALID_INTERSECTION")
    _require(counts["PURCHASE"] == result.orders["purchase_positions"]["known"], "PURCHASE_POSITIONS_MISMATCH")
    products = result.products["data"]
    _require(isinstance(products, list), "INVALID_PRODUCT_ROWS")
    previous = None
    for row in products:
        check()
        _keys(row, "item_id name category purchase_orders purchase_positions buyers nominal_units fulfilled_units "
                   "partial_positions cancelled_positions returned_positions views favorites money")
        item = row["item_id"]
        _require(isinstance(item, str) and item and (previous is None or previous < item), "INVALID_PRODUCT_ID")
        _require(isinstance(row["name"], str) and (row["category"] is None or isinstance(row["category"], str)),
                 "INVALID_PRODUCT_METADATA")
        previous = item
        purchases = _count_cell(row["purchase_orders"])
        _require(_count_cell(row["buyers"]) <= purchases <= result.orders["purchase_orders"]["known"],
                 "INVALID_PRODUCT_PURCHASES")
        for key in ("purchase_positions", "partial_positions", "cancelled_positions", "returned_positions"):
            _count_cell(row[key])
        for key in ("nominal_units", "fulfilled_units"):
            _amount_cell(row[key])
        _require(_count(row["views"]) and _count(row["favorites"]), "INVALID_PRODUCT_ACTIONS")
        _money(row["money"], orders=purchases)
    customer = result.customers["data"]
    _keys({k: v for k, v in customer.items() if k not in ("customer_demography", "customer_activity", "buyer_breakdowns")},
          "counts purchases observed_history")
    _keys(customer["counts"], "population population_buyers nonpurchase_clients cancelled_clients returned_clients")
    population = _count_cell(customer["counts"]["population"])
    _require(all(_count_cell(cell) <= population for cell in customer["counts"].values()), "INVALID_POPULATION")
    history = customer["observed_history"]
    _keys(history, "context_id data")
    _require(history["context_id"] == "raw_orders", "INVALID_SECTION_CONTEXT")
    _keys(history["data"], "purchase_orders purchase_positions")
    for value in history["data"].values():
        _count_cell(value)
    purchases = customer["purchases"]
    _keys(purchases, "context_id counts buyer_money frequency first_months second_intervals first_categories "
                     "second_categories categories pairs")
    _require(purchases["context_id"] == "buyer_period", "INVALID_SECTION_CONTEXT")
    _keys(purchases["counts"], "buyers single_buyers repeat_buyers buyer_purchase_orders unattributed_orders owner_conflict_orders")
    purchase_counts = {name: _count_cell(cell) for name, cell in purchases["counts"].items()}
    frequency = purchases["frequency"]
    _require(isinstance(frequency, list) and all(isinstance(row, list) and len(row) == 2 and all(
        _count(n) and n > 0 for n in row) for row in frequency), "INVALID_BUYER_FREQUENCY")
    _require(sum(n for _, n in frequency) == purchase_counts["buyers"] and
             sum(f*n for f, n in frequency) == purchase_counts["buyer_purchase_orders"] and
             sum(n for f, n in frequency if f == 1) == purchase_counts["single_buyers"] and
             sum(n for f, n in frequency if f > 1) == purchase_counts["repeat_buyers"], "INVALID_BUYER_DENOMINATOR")
    _money(purchases["buyer_money"])
    breakdown = customer["buyer_breakdowns"]
    _currencies(breakdown["monthly_money"])
    _currencies(breakdown["money"])
    for currency in CURRENCIES:
        months = breakdown["monthly_money"][currency]
        finance = breakdown["money"][currency]
        _require(isinstance(months, list) and all(
            isinstance(row, list) and len(row) == 4 and isinstance(row[0], str)
            and re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", row[0])
            and _count(row[1]) and _count(row[2]) and row[1] <= row[2]
            and (_decimal(row[3]) and row[1] > 0 if row[2] else row[3] is None)
            for row in months), "INVALID_MONTHLY_MONEY")
        keys = [row[0] for row in months]
        _require(keys == sorted(set(keys)) and sum(row[2] for row in months) == finance["known_orders"]
                 and _decimal(finance["amount"])
                 and _sum([Decimal(row[3]) for row in months if row[3] is not None]) == Decimal(finance["amount"]),
                 "INVALID_MONTHLY_MONEY")
        groups = finance["quintiles"]
        _require(isinstance(groups, list) and len(groups) <= 5 and all(
            len(row) == 9 and all(_decimal(row[i]) or isinstance(row[i], str)
                                  and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?E[+-][0-9]{1,3}", row[i])
                                  for i in (0, 1, 4, 7))
            and Decimal(row[0]) <= Decimal(row[1]) and _count(row[2]) and row[2] > 0
            and _count(row[5]) and row[5] <= row[2] for row in groups), "INVALID_PURCHASE_QUINTILES")
        _require(all(Decimal(a[1]) < Decimal(b[0]) for a, b in zip(groups, groups[1:]))
                 and sum(row[2] for row in groups) == finance["known_buyers"]
                 and _sum([Decimal(row[4]) for row in groups]) == Decimal(finance["amount"]),
                 "INVALID_PURCHASE_QUINTILES")


def _rfm(result, cutoff, check):
    from Application.statistics_rfm_core import _coverage

    block = result.rfm
    _keys(block, "context_id as_of_date base_buyers windows comparison quality")
    _require(_day(block["as_of_date"]) == cutoff and _count(block["base_buyers"]), "INVALID_RFM_SCOPE")
    _keys(block["quality"], "availability population dual_currency_buyers unassigned_currency_buyers counters")
    base = block["base_buyers"]
    _require(all(_count(block["quality"][key]) for key in ("population", "dual_currency_buyers", "unassigned_currency_buyers"))
             and block["quality"]["population"] >= base >= block["quality"]["dual_currency_buyers"] +
             block["quality"]["unassigned_currency_buyers"], "INVALID_RFM_POPULATION")
    windows = block["windows"]
    if cutoff is None:
        _require(not windows and not block["comparison"] and not base
                 and block["quality"]["availability"] == "NO_ORDERS_COVERAGE", "INVALID_RFM_AVAILABILITY")
        return
    _require(set(windows) == set(WINDOWS) and block["quality"]["availability"] == "AVAILABLE", "INVALID_RFM_WINDOWS")
    intervals = _intervals(result.contexts["rfm_48m"]["orders_intervals"])
    for name, bounds in rfm_windows(cutoff).items():
        row = windows[name]
        _keys(row, "coverage behavior currencies quality")
        _require(row["coverage"] == _coverage(intervals, *bounds)
                 and row["behavior"]["buyers"] == base, "INVALID_RFM_COVERAGE")
        _currencies(row["currencies"])
        for currency in row["currencies"].values():
            check()
            count = currency["buyers"]
            _require(_count(count) and count <= base, "INVALID_RFM_BUYERS")
            for name_money in ("known_subtotal", "partial_nominal"):
                _require(currency[name_money] is None or _decimal(currency[name_money]), "INVALID_RFM_DECIMAL")
            monetary = currency["monetary"]
            _keys(monetary, "buyers mean median")
            _require(_count(monetary["buyers"]) and monetary["buyers"] <= count and
                     all(_decimal(monetary[key]) if monetary["buyers"] else monetary[key] is None
                         for key in ("mean", "median")), "INVALID_RFM_MONETARY")
            _keys(currency["thresholds"], "R F M")
            for metric, thresholds in currency["thresholds"].items():
                _keys(thresholds, "buyers values method")
                values, n = thresholds["values"], thresholds["buyers"]
                test = _decimal if metric == "M" else _count
                _require(_count(n) and n <= count and thresholds["method"] == "NEAREST_RANK_20_40_60_80_STRICT_GREATER"
                         and isinstance(values, list) and len(values) == (4 if n else 0) and all(test(v) for v in values)
                         and values == sorted(values, key=Decimal if metric == "M" else int)
                         and (row["coverage"]["status"] == "FULL" or not n), "INVALID_RFM_THRESHOLDS")
                if metric == "M":
                    _require(n == monetary["buyers"], "INVALID_RFM_MONETARY_DENOMINATOR")
            segments = currency["segments"]
            _require([segment[0] for segment in segments] == list(SEGMENTS)
                     and all(len(segment) == 3 and _count(segment[1]) and
                             segment[2] == (100*segment[1]/count if count else None) for segment in segments)
                     and sum(segment[1] for segment in segments) == count, "INVALID_RFM_SEGMENTS")
            scores = currency["score_counts"]
            _require(all(len(row) == 4 and _count(row[-1]) and all(x is None or type(x) is int and 1 <= x <= 5
                     for x in row[:3]) for row in scores) and sum(row[-1] for row in scores) == count
                     and sum(row[-1] for row in scores if None in row[:3]) == segments[-1][1],
                     "INVALID_RFM_SCORES")
    comparison = block["comparison"]
    groups = comparison["observed_groups"]
    _keys(groups, "BOTH LAST_ONLY PREVIOUS_ONLY NEITHER")
    _require(all(_count(value) for value in groups.values()) and sum(groups.values()) == base and
             comparison["last_orders"] == windows["LAST_12M"]["behavior"]["observed_orders"] and
             comparison["previous_orders"] == windows["PREVIOUS_12M"]["behavior"]["observed_orders"], "INVALID_RFM_COMPARISON")
    _currencies(comparison["money"])
    for row in comparison["money"].values():
        for key in ("last_known_amount", "previous_known_amount", "known_amount_change", "known_amount_change_percent"):
            _require(row[key] is None or _decimal(row[key]), "INVALID_RFM_COMPARISON_AMOUNT")


def _cohort_cell(row, size, *, monthly=False):
    numerator, observed, rate = (("active", "observed_active", "retention_rate") if monthly else
                                 ("repeated", "observed_repeated", "repeat_rate"))
    names = ("eligible", "censored", "coverage_incomplete", "censored_and_incomplete", "confirmed_without_event",
             numerator, observed, observed + "_ineligible", "uncertain_eligible_buyers")
    _require(row["cohort_size"] == size and all(_count(row[key]) and row[key] <= size for key in names),
             "INVALID_COHORT_COUNTS")
    eligible, confirmed, uncertain = (row[key] for key in ("eligible", "confirmed_without_event", "uncertain_eligible_buyers"))
    _require(eligible == size - row["censored"] - row["coverage_incomplete"] + row["censored_and_incomplete"]
             and row[numerator] + confirmed + uncertain == eligible and
             row[numerator] + row[observed + "_ineligible"] == row[observed], "INVALID_COHORT_DENOMINATOR")
    _measure(row["rate_quality"], _finite)
    _require(row[rate] == row["rate_quality"]["value"] == (100*row[numerator]/eligible if eligible and not uncertain else None),
             "INVALID_COHORT_RATE")


def _cohort_money(row, eligible):
    _require(row["eligible"] == eligible and all(_count(row[key]) for key in (
        "buyers", "fully_valued_buyers", "unvalued_buyers", "orders", "unvalued_orders")) and
        row["fully_valued_buyers"] + row["unvalued_buyers"] == row["buyers"] <= eligible and
        row["unvalued_orders"] <= row["orders"], "INVALID_COHORT_MONEY_BASE")
    _currencies(row["currencies"])
    for currency in row["currencies"].values():
        for name in ("recognized", "partial_nominal"):
            _amount_cell(currency[name])
        for metric in ("mean_valued_order", "mean_per_valued_buyer", "valued_order_subtotal", "valued_repeat_amount_rate"):
            if metric == "valued_repeat_amount_rate" and metric not in currency:
                continue
            _require(currency[metric] is None or _decimal(currency[metric]), "INVALID_COHORT_DECIMAL")
        _require(_count(currency["valued_orders"]) and _count(currency["fully_valued_buyers"]) and
                 currency["fully_valued_buyers"] <= row["fully_valued_buyers"] and
                 currency["valued_orders"] <= row["orders"] - row["unvalued_orders"], "INVALID_COHORT_MONEY_BASE")


def _cohorts(result, cutoff, check):
    block = result.cohorts
    _keys(block, "context_id as_of_date cohort_bounds coverage months summary money channels categories quality")
    _require(_day(block["as_of_date"]) == cutoff, "INVALID_COHORT_SCOPE")
    _keys(block["cohort_bounds"], "start end")
    if cutoff is None:
        _require(all(not block[key] for key in ("months", "summary", "money", "channels", "categories", "coverage"))
                 and block["quality"]["availability"] == "NO_ORDERS_COVERAGE", "INVALID_COHORT_AVAILABILITY")
        return
    params = result.metadata["parameters"]
    _require(block["cohort_bounds"] == dict(start=params["selected_start"][:7] if params["selected_start"] else None,
                                          end=params["selected_end"][:7] if params["selected_end"] else None),
             "INVALID_COHORT_SCOPE")
    end = datetime.combine(cutoff + timedelta(days=1), time.min, timezone.utc)
    coverage = _intervals(result.contexts["cohorts_history"]["orders_intervals"])
    _require(block["coverage"] == _wire([(a, min(b, end)) for a, b in coverage if a < end]), "INVALID_COHORT_COVERAGE")
    months, summary = block["months"], block["summary"]
    display, hidden, base = (block["quality"][key] for key in ("displayed_buyers", "hidden_buyers", "base_buyers"))
    _require(all(map(_count, (display, hidden, base))) and display + hidden == base and
             base <= block["quality"]["identified_order_customers"] and summary["cohort_count"] == len(months)
             and sum(row["cohort_size"] for row in months.values()) == display == summary["cohort_size"],
             "INVALID_COHORT_POPULATION")
    horizons = {str(horizon) for horizon in HORIZONS}
    for name, row in [*months.items(), (None, summary)]:
        check()
        size = row["cohort_size"]
        if name is not None:
            _require(isinstance(name, str) and _day(name + "-01").isoformat()[:7] == name
                     and (block["cohort_bounds"]["start"] is None or name >= block["cohort_bounds"]["start"])
                     and name <= cutoff.isoformat()[:7], "INVALID_COHORT_MONTH")
        _require(set(row["horizons"]) == horizons and [x["month_offset"] for x in row["retention"]]
                 == list(MONTH_OFFSETS) and row["retention"][0]["buyers"] == size, "INVALID_COHORT_MATRIX")
        for cell in row["horizons"].values():
            _cohort_cell(cell, size)
        for cell in row["retention"][1:]:
            _cohort_cell(cell, size, monthly=True)
    _keys(block["money"], "cohorts horizons")
    _require(set(block["money"]["cohorts"]) == set(months) and set(block["money"]["horizons"]) == horizons,
             "INVALID_COHORT_MONEY_FIELDS")
    for month, amounts in block["money"]["cohorts"].items():
        _keys(amounts, "total first repeat")
        for amount in amounts.values():
            _cohort_money(amount, months[month]["cohort_size"])
    for horizon, amounts in block["money"]["horizons"].items():
        _cohort_money(amounts, summary["horizons"][horizon]["eligible"])
    for name in ("channels", "categories"):
        _require(block[name]["buyers"] == display and block[name]["repeat_buyers"] == summary["observed_repeat_buyers"],
                 "INVALID_COHORT_TRANSITION_BASE")


def validate_unified(result, *, check=lambda: None):
    try:
        _require(isinstance(result, UnifiedResult) and result.policy == POLICY, "INCOMPATIBLE_POLICY")
        _require(result.capabilities == CAPABILITIES and all(type(x) is bool for x in result.capabilities.values()),
                 "INVALID_CAPABILITIES")
        _walk(result.envelope()["result"], check)
        _keys(result.source, "reader projection_schema provenance_digest")
        source, meta = result.source, result.metadata
        kind, digest = source["reader"], source["provenance_digest"]
        _require(kind in ("REFERENCE", "INDEXED") and _digest(digest) and
                 (type(source["projection_schema"]) is int and source["projection_schema"] == INDEX_SCHEMA
                  if kind == "INDEXED" else source["projection_schema"] is None), "INCOMPATIBLE_PROJECTION")
        _keys(meta, "quantity_policy_version projection_version observation_version cache_schema slice_schema rfm_segmentation "
                    "calculation_algorithm source_binding parameters fingerprint detail content_digest")
        _require(all(type(meta[key]) is type(value) and meta[key] == value for key, value in VERSIONS.items())
                 and _digest(meta["calculation_algorithm"]) and _digest(meta["fingerprint"])
                 and (_digest(meta["source_binding"]) if kind == "INDEXED" else meta["source_binding"] is None),
                 "INCOMPATIBLE_RESULT_METADATA")
        _require(type(meta["detail"]["format_version"]) is int and type(meta["detail"]["available"]) is bool
                 and meta["detail"] == dict(format_version=DETAIL_FORMAT, available=meta["detail"]["available"],
                                       context_ids=list(DETAIL_CONTEXTS if meta["detail"]["available"] else CONTEXTS),
                                       provenance_digest=digest, policy_id=POLICY["policy_id"])
                 and (not meta["detail"]["available"] or kind == "INDEXED"), "INVALID_DETAIL_BINDING")
        params = meta["parameters"]
        _keys(params, " ".join(PARAMETERS))
        _day(params["selected_start"])
        _day(params["selected_end"])
        _require(params["product_filter_key"] is None or _digest(params["product_filter_key"]), "INVALID_FILTER")
        for name in ("order_period", "buyer_period", "customer_period"):
            if params[name] is not None:
                _intervals([params[name]])
        cutoff = _contexts(result)
        _require(meta["fingerprint"] == fingerprint(source, meta["source_binding"], result.calculated_at, params,
                 _wire(cutoff), meta["calculation_algorithm"]), "RESULT_FINGERPRINT_MISMATCH")
        for block, context in ((result.orders, "order_product_slice"), (result.money, "order_product_slice"),
                (result.products, "order_product_slice"), (result.status_catalog, "order_product_slice"),
                (result.actions, "item_interactions"), (result.customers, "customer_population"),
                (result.rfm, "rfm_48m"), (result.cohorts, "cohorts_history"), (result.diagnostics, "raw_orders")):
            _require(block["context_id"] == context, "INVALID_SECTION_CONTEXT")
        _orders_and_catalog(result, check)
        _actions_products_customers(result, check)
        _rfm(result, cutoff, check)
        _cohorts(result, cutoff, check)
        diagnostics = result.diagnostics
        _keys(diagnostics, "context_id counters reason_counts scope_by_counter reason_stage")
        _require(set(_COUNTERS) <= diagnostics["counters"].keys() and
                 all(_count(x) for name in ("counters", "reason_counts") for x in diagnostics[name].values())
                 and set(diagnostics["scope_by_counter"]) == set(diagnostics["counters"])
                 and set(diagnostics["scope_by_counter"].values()) <= set(CONTEXTS)
                 and diagnostics["reason_stage"] == {
                     key: "ACTION_INPUT_BEFORE_FILTER" if key.startswith("ACTION_") else "ORDER_RECONCILIATION"
                     for key in diagnostics["reason_counts"]
                 }, "INVALID_DIAGNOSTICS")
        _require(meta["content_digest"] == content_digest(result.envelope()["result"]), "RESULT_CONTENT_MISMATCH")
        check()
        return result
    except StatisticsResultError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, ArithmeticError, RecursionError):
        raise StatisticsResultError("INVALID_UNIFIED_RESULT") from None


def read_envelope(value, *, check=lambda: None):
    """Use the schema's own validator. This function has no write or calculation path."""
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise StatisticsResultError("INCOMPATIBLE_RESULT_SCHEMA")
    version = value["schema_version"]
    if version == legacy.SCHEMA_VERSION:
        return legacy.validate_result(value.get("result"))
    if version != RESULT_SCHEMA:
        raise StatisticsResultError("INCOMPATIBLE_RESULT_SCHEMA")
    _keys(value, "schema_version result")
    _keys(value["result"], " ".join(field.name for field in fields(UnifiedResult)))
    return validate_unified(UnifiedResult(**value["result"]), check=check)


def read_statistics_result(path, *, check=lambda: None):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=legacy._unique_object)
    return read_envelope(value, check=check)
