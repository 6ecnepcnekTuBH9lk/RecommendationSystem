"""Isolated UNIFIED_PURCHASE_V1 aggregates over reconciled order evidence."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from itertools import combinations
import json
from types import SimpleNamespace

from Application.action_statistics import ActionAggregates
from Application.statistics_unified_breakdowns import BuyerBreakdowns, OrderBreakdowns, product_breakdowns

from Application.dataset_statistics import _bucket as activity_bucket, _distribution, _histogram_median
from Application.interactions import InteractionType
from Application.product_resolution import SUPPORTED_NAMESPACES
from Application.order_statistics import CURRENCIES
from Application.statistics_copurchases import MAX_DISTINCT_PAIRS, MAX_PAIRS_PER_ORDER
from Application.statistics_unified_facts import AnalyticsContext, OrderBusinessFact, OrderKind, OrderObservation, _issues, _result, _sum
from Application.statistics_unified_policy import EvidenceQuality as Quality, Measured, STATUS_CATALOG, StatusGroup

_ZERO = Decimal(0)
_GAP = (Quality.INSUFFICIENT_COVERAGE, "COVERAGE_GAP")
_DATE = (Quality.INSUFFICIENT_COVERAGE, "ORDER_DATE_UNRESOLVED")
_OWNER = (Quality.INSUFFICIENT_COVERAGE, "OWNER_UNRESOLVED")
_STATE = (Quality.INSUFFICIENT_COVERAGE, "ORDER_STATE_UNRESOLVED")
_PRODUCT = (Quality.INSUFFICIENT_COVERAGE, "PRODUCT_SCOPE_UNRESOLVED")
_NO_DENOMINATOR = (Quality.UNKNOWN, "NO_FULLY_VALUED_BASE")
_AMOUNT_FIELDS = ("source_nominal", "known_purchase_subtotal", "recognized_purchase",
                  "partial_nominal", "cancelled_nominal", "returned_nominal")
_ORDER_COUNTS = (
    "orders", "positions", "purchase_orders", "orders_without_evidence", "unresolved_purchase_orders",
    "purchase_positions", "in_progress_positions", "partial_positions", "partial_orders",
    "mixed_orders", "mixed_slice_orders", "cancelled_positions", "cancelled_orders",
    "fully_cancelled_orders", "mixed_cancelled_orders", "returned_positions", "returned_orders",
    "all_returned_orders", "mixed_returned_orders", "conflicting_orders", "incomplete_structure_orders",
    "unresolved_scope_orders", "unknown_state_orders",
)


@dataclass(frozen=True, slots=True)
class AggregateMetric:
    known: int | Decimal | None
    total: Measured[int | Decimal]


@dataclass(frozen=True, slots=True)
class CurrencyAggregate:
    currency: str
    amounts: dict[str, AggregateMetric]
    valued_orders: int
    unvalued_orders: int
    valued_positions: int
    unvalued_positions: int
    mean_order: Measured[Decimal]
    fully_valued_buyers: int = 0
    unvalued_buyers: int = 0
    mean_per_valued_buyer: Measured[Decimal] | None = None


@dataclass(frozen=True, slots=True)
class StatusAggregate:
    code: str | None
    bucket: str
    raw_occurrences: int
    positions: AggregateMetric
    orders: AggregateMetric
    historical_positions: AggregateMetric
    raw: AggregateMetric
    name: str | None
    mindbox_category: str | None
    business_group: StatusGroup


@dataclass(frozen=True, slots=True)
class ProductAggregate:
    item_id: str
    purchase_orders: AggregateMetric
    purchase_positions: AggregateMetric
    buyers: AggregateMetric
    nominal_units: AggregateMetric
    fulfilled_units: AggregateMetric
    partial_positions: AggregateMetric
    cancelled_positions: AggregateMetric
    returned_positions: AggregateMetric
    views: int
    favorites: int
    money: tuple[CurrencyAggregate, ...]


@dataclass(frozen=True, slots=True)
class CategoryAggregate:
    categories: tuple[str, ...]
    orders: AggregateMetric
    buyers: AggregateMetric


@dataclass(frozen=True, slots=True)
class AggregateResult:
    contexts: tuple[AnalyticsContext, ...]
    orders: dict[str, AggregateMetric]
    order_flags: dict[str, AggregateMetric]
    statuses: tuple[StatusAggregate, ...]
    actions: dict[str, AggregateMetric]
    customers: dict[str, AggregateMetric]
    frequency: tuple[tuple[int, int], ...]
    first_months: tuple[tuple[str, int], ...]
    second_intervals: tuple[tuple[int, int], ...]
    first_categories: tuple[tuple[str, int], ...]
    second_categories: tuple[tuple[str, int], ...]
    products: tuple[ProductAggregate, ...]
    categories: tuple[CategoryAggregate, ...]
    pairs: tuple[CategoryAggregate, ...]
    order_money: tuple[CurrencyAggregate, ...]
    buyer_money: tuple[CurrencyAggregate, ...]
    diagnostics: dict[str, int]
    observed_history: dict[str, AggregateMetric]
    order_states: dict[str, AggregateMetric]
    breakdowns: dict


@dataclass(frozen=True, slots=True, repr=False)
class ActionEvidence:
    customer_key: str | None
    occurred: datetime | None
    kind: InteractionType | None
    items: tuple[str | None, ...]
    reason: str | None = None
    dimensions: object = None
    has_product: bool | None = None
    system_name: str | None = None


def _add_decimal(left, right):
    return _sum([left, right])


def _count(known, issues=()):
    return AggregateMetric(known, _result(known, issues))


def _mean(amount, count):
    if not count:
        return _result(_ZERO, (_NO_DENOMINATOR,))
    with localcontext() as context:
        context.prec = max(28, len(amount.as_tuple().digits))
        context.rounding = ROUND_HALF_EVEN
        return _result(amount / Decimal(count), ())


def _bucket(status):
    if status.value is not None:
        return status.value, "UNKNOWN_CODE" if "UNKNOWN_CODE" in status.reasons else "CATALOG"
    if "MISSING_CODE" in status.reasons:
        return None, "MISSING_CODE"
    if "INVALID_CODE" in status.reasons:
        return None, "INVALID_CODE"
    return None, "UNRESOLVED"


def _covered(intervals, period):
    return period is not None and any(start <= period[0] < period[1] <= end for start, end in intervals)


def _in_period(occurred, intervals, period):
    return (occurred is not None and period is not None and period[0] <= occurred < period[1]
            and any(start <= occurred < end for start, end in intervals))


def order_in_detail_scope(whole, selected, scope):
    if whole is None or whole.order_created_at.value is None:
        return False
    if not _in_period(whole.order_created_at.value, scope.orders_intervals, scope.period):
        return False
    return (scope.product_filter_key is None or bool(selected.resolved_lines)
            or selected.purchase.observed.value is True)


def _inside_period(occurred, period):
    return occurred is not None and period is not None and period[0] <= occurred < period[1]


class _Money:
    def __init__(self):
        self.amounts = {currency: {} for currency in CURRENCIES}
        self.issues = defaultdict(set)
        self.valued, self.unvalued = Counter(), Counter()
        self.valued_positions, self.unvalued_positions = Counter(), Counter()
        self.cheque = defaultdict(lambda: _ZERO)

    def add(self, money, *, purchase, currency, fully_valued_currency):
        for value in money:
            unit = value.currency
            known = dict(value.known_nominals)
            for name in _AMOUNT_FIELDS:
                reading = getattr(value, name)
                self.issues[unit, name].update(_issues(reading))
                observed = reading.value if reading.value is not None else known.get(name)
                if observed is None and name == "recognized_purchase":
                    observed = value.known_purchase_subtotal.value
                if name in ("known_purchase_subtotal", "recognized_purchase") and observed == 0:
                    if (not purchase or not value.valued_purchase_positions
                            and value.recognized_purchase.value is not None):
                        continue
                if observed is not None:
                    self.amounts[unit][name] = _add_decimal(self.amounts[unit].get(name, _ZERO), observed)
            self.valued_positions[unit] += value.valued_purchase_positions
            self.unvalued_positions[unit] += value.unvalued_purchase_positions
            if purchase and unit == fully_valued_currency:
                self.valued[unit] += 1
                self.cheque[unit] = _add_decimal(self.cheque[unit], value.recognized_purchase.value)
            elif purchase and currency in (None, unit) and fully_valued_currency is None:
                self.unvalued[unit] += 1

    def result(self, issues, buyer_bases=None):
        rows = []
        for unit in CURRENCIES:
            amounts = {}
            for name in _AMOUNT_FIELDS:
                failures = set(issues) | self.issues[unit, name]
                known = self.amounts[unit].get(name)
                # A subtotal remains a subtotal even when only some inputs were valued.
                if name == "known_purchase_subtotal" and known is not None:
                    total = _result(known, ())
                else:
                    total = _result(known if known is not None else _ZERO, failures)
                amounts[name] = AggregateMetric(known if failures else total.value, total)
            for name in ("confirmed_refund", "net_after_refunds"):
                amounts[name] = AggregateMetric(None, _result(_ZERO, ((Quality.UNKNOWN, "LEDGER_UNAVAILABLE"),)))
            n, incomplete, amount = (buyer_bases or {}).get(unit, (0, 0, _ZERO))
            rows.append(CurrencyAggregate(
                unit, amounts, self.valued[unit], self.unvalued[unit], self.valued_positions[unit],
                self.unvalued_positions[unit], _mean(self.cheque[unit], self.valued[unit]),
                n, incomplete, None if buyer_bases is None else (
                    _result(_ZERO, issues) if issues else _mean(amount, n)),
            ))
        return tuple(rows)


def _valuation(purchase, money, currency):
    if purchase and currency in CURRENCIES:
        row = next(row for row in money if row.currency == currency)
        if row.recognized_purchase.value is not None:
            return currency
    return None


@dataclass(slots=True)
class _Item:
    counts: Counter = field(default_factory=Counter)
    units: Decimal = _ZERO
    issues: dict = field(default_factory=lambda: defaultdict(set))
    money: _Money = field(default_factory=_Money)

    def add(self, evidence):
        purchase = evidence.purchase
        self.counts["purchase_orders"] += purchase.observed.value is True
        self.counts["purchase_positions"] += purchase.known_positions
        self.units = _add_decimal(self.units, purchase.known_nominal_units)
        self.issues["purchase_orders"].update(_issues(purchase.observed))
        self.issues["purchase_positions"].update(_issues(purchase.positions))
        self.issues["nominal_units"].update(_issues(purchase.nominal_units))
        for key, group in (("partial_positions", StatusGroup.PARTIALLY_COMPLETED),
                           ("cancelled_positions", StatusGroup.CANCELLED), ("returned_positions", StatusGroup.RETURNED)):
            self.counts[key] += dict(evidence.current_groups).get(group, 0)
            self.issues[key].update(_issues(evidence.positions))
            if dict(evidence.current_groups).get(StatusGroup.UNKNOWN):
                self.issues[key].add(_STATE)
        currency = evidence.order_currency.value
        valued = _valuation(purchase.observed.value is True, evidence.money, currency)
        self.money.add(evidence.money, purchase=purchase.observed.value is True,
                       currency=currency, fully_valued_currency=valued)

    def result(self, code, buyers, issues, buyer_issues):
        def measure(name):
            return _count(self.counts[name], set(issues) | self.issues[name])
        units = AggregateMetric(self.units, _result(self.units, set(issues) | self.issues["nominal_units"]))
        fulfilled = AggregateMetric(None, _result(_ZERO, ((Quality.UNKNOWN, "FULFILLMENT_UNVERIFIED"),)))
        return ProductAggregate(
            code, measure("purchase_orders"), measure("purchase_positions"),
            _count(buyers, set(issues) | buyer_issues), units, fulfilled,
            measure("partial_positions"), measure("cancelled_positions"), measure("returned_positions"),
            self.counts["VIEW"], self.counts["FAVORITE"], self.money.result(issues),
        )


_MEMBERSHIP = """
CREATE TABLE member(kind TEXT, bucket TEXT, customer TEXT, PRIMARY KEY(kind,bucket,customer)) WITHOUT ROWID;
CREATE TABLE actions(customer TEXT PRIMARY KEY, mask INTEGER NOT NULL,
 interactions INTEGER NOT NULL, purchases INTEGER NOT NULL) WITHOUT ROWID;
CREATE TABLE buyers(customer TEXT PRIMARY KEY, frequency INTEGER NOT NULL) WITHOUT ROWID;
CREATE TABLE chronology(customer TEXT, stamp TEXT, rank INTEGER, categories TEXT,
 PRIMARY KEY(customer,stamp,rank)) WITHOUT ROWID;
CREATE TABLE cash(scope TEXT,currency TEXT,customer TEXT,amount TEXT,n INTEGER,bad INTEGER,
 PRIMARY KEY(scope,currency,customer)) WITHOUT ROWID;
CREATE TABLE seen(order_key TEXT PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE product_source(namespace TEXT, value TEXT, PRIMARY KEY(namespace,value)) WITHOUT ROWID;
"""


class _Members:
    """High-cardinality memberships and bounded two-order histories on disk."""

    def __init__(self, db, check):
        self.db, self.check = db, check
        db.executescript(_MEMBERSHIP)
        db.create_function("decimal_add", 2, lambda a, b: str(_add_decimal(Decimal(a), Decimal(b))))

    def add(self, kind, customer, bucket=""):
        if customer is not None:
            self.db.execute("INSERT OR IGNORE INTO member VALUES(?,?,?)", (kind, bucket, customer))

    def counts(self, kind):
        return dict(self.db.execute("SELECT bucket,COUNT(*) FROM member WHERE kind=? GROUP BY bucket", (kind,)))

    def count(self, kind):
        return self.counts(kind).get("", 0)

    def activity(self, customer, mask, interactions, purchases=0):
        if customer is not None:
            self.db.execute(
                "INSERT INTO actions VALUES(?,?,?,?) ON CONFLICT(customer) DO UPDATE SET "
                "mask=mask|excluded.mask,interactions=interactions+excluded.interactions,purchases=purchases+excluded.purchases",
                (customer, mask, interactions, purchases))

    def activity_summary(self):
        active, buyers, repeat, total = self.db.execute(
            "SELECT COUNT(*),COUNT(CASE WHEN purchases>0 THEN 1 END),"
            "COUNT(CASE WHEN purchases>1 THEN 1 END),COALESCE(SUM(interactions),0) FROM actions").fetchone()
        activity = dict(self.db.execute("SELECT interactions,COUNT(*) FROM actions GROUP BY interactions"))
        orders = dict(self.db.execute("SELECT purchases,COUNT(*) FROM actions WHERE purchases>0 GROUP BY purchases"))

        def distribution(histogram, limits, labels):
            buckets = Counter()
            for value, count in histogram.items():
                self.check()
                buckets[activity_bucket(value, limits, labels)] += count
            return _distribution(buckets, labels)

        return dict(active=active, buyers=buyers, repeat_buyers=repeat,
                    mean=total / active if active else 0, median=_histogram_median(activity),
                    buyer_rate=100 * buyers / active if active else 0,
                    order_mean=sum(n * count for n, count in orders.items()) / buyers if buyers else 0,
                    order_median=_histogram_median(orders),
                    repeat_rate=100 * repeat / buyers if buyers else 0,
                    interactions=distribution(activity, (1, 5, 10, 25, 50, 100),
                                              ("1", "2–5", "6–10", "11–25", "26–50", "51–100", "101+")),
                    orders=distribution(orders, (1, 2, 5, 10),
                                        ("1 заказ", "2 заказа", "3–5 заказов", "6–10 заказов", "11+ заказов")))

    def buy(self, fact, codes, unknown, valued):
        owner = fact.customer_key.value
        self.db.execute("INSERT INTO buyers VALUES(?,1) ON CONFLICT(customer) DO UPDATE SET frequency=frequency+1",
                        (owner,))
        stamp = fact.order_created_at.value.isoformat(timespec="microseconds")
        self.db.execute("INSERT INTO chronology VALUES(?,?,?,?)",
                        (owner, stamp, fact.order_sort_rank, json.dumps((sorted(codes), unknown))))
        self.db.execute("DELETE FROM chronology WHERE customer=? AND (stamp,rank) IN ("
                        "SELECT stamp,rank FROM chronology WHERE customer=? ORDER BY stamp,rank LIMIT -1 OFFSET 2)",
                        (owner, owner))
        self.cash(fact, valued, "buyer")

    def cash(self, fact, valued, scope):
        owner = fact.customer_key.value
        for row in fact.money:
            n = int(valued == row.currency)
            amount = row.recognized_purchase.value if n else _ZERO
            bad = int(valued is None)
            self.db.execute("INSERT INTO cash VALUES(?,?,?,?,?,?) ON CONFLICT(scope,currency,customer) DO UPDATE SET "
                            "amount=decimal_add(amount,excluded.amount),n=n+excluded.n,bad=bad|excluded.bad",
                            (scope, row.currency, owner, str(amount), n, bad))

    def buyer_money(self, scope):
        bases = {}
        for currency in CURRENCIES:
            count = unvalued = 0
            amount = _ZERO
            for value, n, bad in self.db.execute("SELECT amount,n,bad FROM cash WHERE scope=? AND currency=?", (scope, currency)):
                self.check()
                if bad:
                    unvalued += 1
                elif n:
                    count += 1
                    amount = _add_decimal(amount, Decimal(value))
            bases[currency] = (count, unvalued, amount)
        return bases


@dataclass(frozen=True, slots=True, repr=False)
class AggregateOrder:
    observations: tuple[OrderObservation, ...]
    whole: OrderBusinessFact | None
    selected: OrderBusinessFact | None


class _Aggregate:
    def __init__(self, contexts, db, metadata, check, category_reference):
        self.contexts, self.check = contexts, check
        self.order_scope, self.buyer_scope, self.customer_scope = contexts[1], contexts[3], contexts[4]
        self.members = _Members(db, check)
        self.buyer_breakdowns = BuyerBreakdowns(db, check)
        self.metadata = metadata
        self.category_reference = category_reference or {}
        self.counts, self.flags, self.action_counts = Counter(), Counter(), Counter()
        self.history, self.diagnostics = Counter(), Counter()
        self.order_states = Counter()
        self.action_breakdowns = ActionAggregates()
        self.event_names, self.source_actions = Counter(), Counter()
        self.namespace_counts = defaultdict(Counter)
        self.order_breakdowns = OrderBreakdowns()
        self.selected_owner_issues = defaultdict(set)
        self.raw_status, self.current_status, self.status_orders, self.historical_status = (
            Counter(), Counter(), Counter(), Counter())
        self.categories, self.pairs = Counter(), Counter()
        self.items = {}
        self.order_money, self.buyer_money = _Money(), _Money()
        self.order_issues, self.buyer_issues, self.owner_issues = set(), set(), set()
        self.buyer_purchase_issues = set()
        self.position_issues, self.purchase_issues, self.status_issues, self.history_issues = (
            set(), set(), set(), set())
        self.product_issues, self.category_issues, self.raw_issues = set(), set(), set()
        self.purchase_position_issues, self.state_issues = set(), set()
        self.action_issues = defaultdict(set)
        self.history_position_issues = set()
        self.population_issues, self.population_purchase_issues, self.population_state_issues = set(), set(), set()
        if not contexts[0].orders_intervals:
            self.raw_issues.add(_GAP)
            self.history_issues.add(_GAP)
            self.history_position_issues.add(_GAP)
        if self.customer_scope.period is not None:
            if not _covered(self.customer_scope.orders_intervals, self.customer_scope.period):
                self.population_purchase_issues.add(_GAP)
                self.population_state_issues.add(_GAP)
                self.population_issues.add(_GAP)
            if not _covered(self.customer_scope.actions_intervals, self.customer_scope.period):
                self.population_issues.add(_GAP)
        if not _covered(self.order_scope.orders_intervals, self.order_scope.period):
            self.order_issues.add(_GAP)
        if not _covered(self.buyer_scope.orders_intervals, self.buyer_scope.period):
            self.buyer_issues.add(_GAP)
        if not _covered(self.order_scope.actions_intervals, self.order_scope.period):
            for kind in ("VIEW", "FAVORITE"):
                self.action_issues[kind].add(_GAP)

    def _item(self, key):
        if key not in self.metadata:
            raise ValueError("Unified product is missing pinned metadata")
        return self.items.setdefault(key, _Item()) if key not in self.items else self.items[key]

    def _raw(self, observation):
        self.diagnostics["raw_orders"] += 1
        self.diagnostics["raw_lines"] += observation.raw_line_count
        if observation.lines_quality is not Quality.KNOWN:
            self.raw_issues.add((observation.lines_quality, "LINES_INCOMPLETE"))
            self.raw_status[(None, "UNRESOLVED")] += observation.raw_line_count - len(observation.lines)
        for line in observation.lines:
            self.check()
            self.raw_status[_bucket(line.status_code)] += 1

    def order(self, entry):
        for observation in entry.observations:
            self.check()
            self._raw(observation)
        whole, selected = entry.whole, entry.selected
        if whole is None:
            self.diagnostics["orders_without_valid_key"] += 1
            for issues in (self.history_issues, self.history_position_issues, self.order_issues, self.buyer_issues,
                           self.population_issues):
                issues.add((Quality.INSUFFICIENT_COVERAGE, "ORDER_ID_INVALID"))
            return
        db = self.members.db
        if not db.execute("INSERT OR IGNORE INTO seen VALUES(?)", (whole.order_key,)).rowcount:
            raise ValueError("Unified order groups must be unique")
        self.diagnostics.update(whole.diagnostics)
        for field_name in ("exact_duplicates", "business_equivalent", "conflicting_snapshots"):
            self.diagnostics[field_name] += getattr(whole.snapshots, field_name)
        self.history["purchase_orders"] += whole.purchase.observed.value is True
        self.history["purchase_positions"] += whole.purchase.known_positions
        self.history_issues.update(_issues(whole.purchase.observed))
        self.history_position_issues.update(_issues(whole.purchase.positions))
        date = whole.order_created_at.value
        if date is None:
            if self.customer_scope.period is None:
                self._population_order(whole)
            self.diagnostics["undated_orders"] += 1
            self.order_issues.add(_DATE)
            self.buyer_issues.add(_DATE)
            if self.customer_scope.period is not None:
                self.population_issues.add(_DATE)
            return
        if not any(start <= date < end for start, end in self.contexts[0].orders_intervals):
            self.diagnostics["orders_outside_coverage"] += 1
        if self.customer_scope.period is None or _inside_period(date, self.customer_scope.period):
            self._population_order(whole)
        if _in_period(date, self.buyer_scope.orders_intervals, self.buyer_scope.period):
            self._buyer(whole)
        if order_in_detail_scope(whole, selected, self.order_scope):
            self._selected(whole, selected)
        elif (_in_period(date, self.order_scope.orders_intervals, self.order_scope.period)
              and selected.positions.value != 0):
            self.counts["unresolved_scope_orders"] += 1
            self.order_issues.add(_PRODUCT)

    def _population_order(self, fact):
        owner = fact.customer_key.value
        if owner is None:
            self.population_issues.add(_OWNER)
        self.members.add("population", owner)
        if fact.purchase.observed.value is True:
            self.members.add("population_buyers", owner)
        elif fact.purchase.observed.value is False:
            self.members.add("nonpurchase_clients", owner)
        else:
            self.population_purchase_issues.update(_issues(fact.purchase.observed))
        if fact.has_cancelled:
            self.members.add("cancelled_clients", owner)
        if fact.has_returned:
            self.members.add("returned_clients", owner)
        if fact.state_quality is not Quality.KNOWN:
            self.population_state_issues.add(_STATE)

    def _buyer(self, fact):
        owner = fact.customer_key.value
        evidence = fact.purchase.observed
        self.buyer_purchase_issues.update(_issues(evidence))
        if owner is None:
            self.diagnostics["unattributed_orders"] += 1
            self.diagnostics["owner_conflict_orders"] += fact.customer_key.quality is Quality.CONFLICT
        valued = _valuation(evidence.value is True, fact.money, fact.order_currency.value)
        self.buyer_money.add(fact.money, purchase=evidence.value is True, currency=fact.order_currency.value,
                             fully_valued_currency=valued)
        if evidence.value is not True:
            return
        self.history["buyer_period_purchase_orders"] += 1
        if owner is None:
            self.history["unattributed_purchase_orders"] += 1
            self.owner_issues.add(_OWNER)
        codes = set()
        for item in fact.products:
            self.check()
            if item.purchase.known_positions:
                category = self.metadata[item.item_id].category
                if category:
                    codes.add(category)
        attributed = sum(item.purchase.known_positions for item in fact.products
                         if self.metadata[item.item_id].category)
        incomplete = fact.purchase.positions.value is None or attributed != fact.purchase.known_positions
        if incomplete:
            self.category_issues.add(_PRODUCT)
            self.diagnostics["orders_with_unresolved_categories"] += 1
        if len(codes) * (len(codes) - 1) // 2 > MAX_PAIRS_PER_ORDER:
            raise ValueError("Unified order category pair limit exceeded")
        for code in sorted(codes):
            self.check()
            self.categories[code] += 1
            self.members.add("category", owner, code)
        if len(codes) > 1:
            self.history["multicategory_orders"] += 1
            self.members.add("multicategory_buyers", owner)
        for pair in combinations(sorted(codes), 2):
            self.check()
            if pair not in self.pairs and len(self.pairs) >= MAX_DISTINCT_PAIRS:
                raise ValueError("Unified distinct category pair limit exceeded")
            self.pairs[pair] += 1
            self.members.add("pair", owner, json.dumps(pair, ensure_ascii=False))
        if owner is not None:
            self.members.buy(fact, codes, incomplete, valued)
            self.buyer_breakdowns.add(fact, valued)

    def _namespace(self, token, resolved):
        namespace = token[0][0] if token and len(token) == 1 else None
        bucket = namespace if namespace in SUPPORTED_NAMESPACES else "unsupported"
        self.namespace_counts[bucket][bool(resolved)] += 1
        if token:
            self.members.db.executemany("INSERT OR IGNORE INTO product_source VALUES(?,?)", token)

    def _resolution_summary(self, namespaces):
        source = self.members.db.execute("SELECT COUNT(*) FROM product_source").fetchone()[0]
        resolved = sum(row[2] for row in namespaces)
        unresolved = sum(row[3] for row in namespaces)
        n = resolved + unresolved
        return dict(source_products=source, resolved_products=sum(any(item.counts.values()) for item in self.items.values()),
                    resolved=resolved, unresolved=unresolved, rate=100 * resolved / n if n else 0)

    def _namespaces(self):
        rows = []
        for namespace in (*sorted(SUPPORTED_NAMESPACES), "unsupported"):
            counts = self.namespace_counts[namespace]
            resolved, unresolved = counts[True], counts[False]
            total = resolved + unresolved
            rows.append((namespace, total, resolved, unresolved, 100 * resolved / total if total else 0))
        return rows

    def _selected(self, whole, fact):
        count = self.counts
        owner = fact.customer_key.value
        evidence = fact.purchase.observed
        count["orders"] += 1
        self.order_breakdowns.add(fact, self.members.add, whole_currency=whole.order_currency.value)
        self.order_states[fact.current_kind] += 1
        count["positions"] += len(fact.resolved_lines)
        count["purchase_orders"] += evidence.value is True
        count["orders_without_evidence"] += evidence.value is False
        count["unresolved_purchase_orders"] += evidence.value is None
        count["purchase_positions"] += fact.purchase.known_positions
        count["mixed_orders"] += whole.current_kind is OrderKind.MIXED
        count["mixed_slice_orders"] += fact.current_kind is OrderKind.MIXED
        count["conflicting_orders"] += fact.state_quality is Quality.CONFLICT
        count["unknown_state_orders"] += fact.has_unknown
        count["incomplete_structure_orders"] += fact.positions.value is None
        count["fully_cancelled_orders"] += whole.current_kind is OrderKind.CANCELLED_ONLY
        count["all_returned_orders"] += whole.current_kind is OrderKind.RETURNED_ONLY
        count["mixed_cancelled_orders"] += whole.current_kind is OrderKind.MIXED and fact.has_cancelled
        count["mixed_returned_orders"] += whole.current_kind is OrderKind.MIXED and fact.has_returned
        for flag in ("completed", "partial", "cancelled", "returned", "in_progress", "unknown"):
            self.flags[flag] += getattr(fact, "has_" + flag)
        for name in ("partial", "cancelled", "returned"):
            count[name + "_orders"] += getattr(fact, "has_" + name)
        self.position_issues.update(_issues(fact.positions))
        self.purchase_position_issues.update(_issues(fact.purchase.positions))
        self.purchase_issues.update(_issues(evidence))
        self.status_issues.update(_issues(fact.positions))
        if fact.has_unknown or whole.has_unknown:
            self.state_issues.add(_STATE)
        statuses = set()
        for line in fact.resolved_lines:
            self.check()
            group = line.status_group
            count["in_progress_positions"] += group.in_progress
            count["partial_positions"] += group is StatusGroup.PARTIALLY_COMPLETED
            count["cancelled_positions"] += group is StatusGroup.CANCELLED
            count["returned_positions"] += group is StatusGroup.RETURNED
            code = _bucket(line.status_code)
            statuses.add(code)
            if group in (StatusGroup.COMPLETED, StatusGroup.PARTIALLY_COMPLETED):
                self._namespace(line.product_token, line.product_key.value is not None)
            self.current_status[code] += 1
            self.status_issues.update(_issues(line.status_code))
        self.status_orders.update(statuses)
        for status, positions in fact.historical_statuses:
            self.historical_status[_bucket(status)] += positions
        valued = _valuation(evidence.value is True, fact.money, fact.order_currency.value)
        for name, observed in (("buyers", evidence.value is True),
                               ("cancelled_customers", fact.has_cancelled), ("returned_customers", fact.has_returned)):
            if observed:
                self.members.add(name, owner)
                if owner is None:
                    self.selected_owner_issues[name].add(_OWNER)
        if evidence.value is True:
            count["unassigned_currency_orders"] += fact.order_currency.value is None
            if owner is not None:
                self.members.cash(fact, valued, "order")
        self.order_money.add(fact.money, purchase=evidence.value is True, currency=fact.order_currency.value,
                             fully_valued_currency=valued)
        self.action_counts["PURCHASE"] += fact.purchase.known_positions
        self.action_issues["PURCHASE"].update(_issues(fact.purchase.positions))
        if fact.purchase.known_positions:
            self.members.activity(owner, 4, fact.purchase.known_positions, int(evidence.value is True))
            if owner is None:
                self.action_issues["PURCHASE_OWNER"].add(_OWNER)
        for item in fact.products:
            self.check()
            self._item(item.item_id).add(item)
            if item.purchase.observed.value is True:
                self.members.add("item", owner, item.item_id)
        attributed = sum(item.purchase.known_positions for item in fact.products)
        if attributed != fact.purchase.known_positions or "PRODUCT_SCOPE_UNRESOLVED" in fact.diagnostics:
            self.product_issues.add(_PRODUCT)
            self.diagnostics["unresolved_purchase_items"] += fact.purchase.known_positions - attributed
        self.product_issues.update(_issues(fact.positions))

    def action(self, event, predicate):
        occurred, owner = event.occurred, event.customer_key
        if self.customer_scope.period is None or _inside_period(occurred, self.customer_scope.period):
            self.members.add("population", owner)
        elif occurred is None and owner is not None:
            self.population_issues.add((Quality.INSUFFICIENT_COVERAGE, "ACTION_DATE_UNRESOLVED"))
        if self.action_period is None or _inside_period(occurred, self.action_period):
            self.source_actions["all"] += 1
            if event.system_name:
                category = event.kind.value if event.kind is not None else "UNMAPPED"
                self.event_names[category, event.system_name] += 1
            if event.kind is not None:
                self.source_actions[event.kind.value] += 1
                if event.has_product is None:
                    self.source_actions["mapped_product_unknown"] += 1
                elif not event.has_product:
                    self.source_actions["mapped_without_product"] += 1
            elif event.reason == "ACTION_UNMAPPED":
                self.source_actions["unmapped"] += 1
            else:
                self.source_actions["classification_unknown"] += 1
        if event.reason:
            self.diagnostics[event.reason] += 1
        if event.kind is None:
            if event.reason == "ACTION_INVALID":
                for kind in ("VIEW", "FAVORITE"):
                    self.action_issues[kind].add((Quality.INSUFFICIENT_COVERAGE, event.reason))
            return
        kind = event.kind.value
        if occurred is None:
            self.action_issues[kind].add((Quality.INSUFFICIENT_COVERAGE, "ACTION_DATE_UNRESOLVED"))
            return
        dimensions = event.dimensions
        if dimensions is not None and owner is not None and (self.action_period is None or _inside_period(occurred, self.action_period)):
            accepted = sum(predicate is None or code is not None and predicate(code) for code in event.items)
            if accepted:
                interaction = SimpleNamespace(customer_id=owner, interaction_type=event.kind)
                self.action_breakdowns.add(dimensions, (interaction,) * accepted)
        if not _in_period(occurred, self.order_scope.actions_intervals, self.order_scope.period):
            return
        if event.reason in ("ACTION_INVALID", "ACTION_OWNER_MISSING"):
            self.action_issues[kind].add((Quality.INSUFFICIENT_COVERAGE, event.reason))
        if owner is None:
            return
        bit = 1 if event.kind is InteractionType.VIEW else 2
        for offset, code in enumerate(event.items):
            self.check()
            if predicate is not None and (code is None or not predicate(code)):
                if code is None:
                    self.action_issues[kind].add(_PRODUCT)
                continue
            self.action_counts[kind] += 1
            self.members.activity(owner, bit, 1)
            if dimensions is not None:
                product = dimensions.products[offset]
                self._namespace(((product.namespace, product.value),), code is not None)
            if code is None:
                self.diagnostics["unresolved_" + kind.lower() + "_items"] += 1
            else:
                self._item(code).counts[kind] += 1
                self.members.add("item_" + kind, owner, code)

    def _action_result(self):
        rows = {}
        masks = Counter(dict(self.members.db.execute("SELECT mask,COUNT(*) FROM actions GROUP BY mask")))
        for name, bit in (("VIEW", 1), ("FAVORITE", 2), ("PURCHASE", 4)):
            issues = self.action_issues[name] | (self.order_issues if bit == 4 else set())
            rows[name] = _count(self.action_counts[name], issues)
            if bit == 4:
                issues |= self.action_issues["PURCHASE_OWNER"]
            rows[name + "_users"] = _count(sum(n for mask, n in masks.items() if mask & bit), issues)
        for name, bits in (("VIEW_PURCHASE", 5), ("FAVORITE_PURCHASE", 6),
                           ("VIEW_FAVORITE", 3), ("VIEW_FAVORITE_PURCHASE", 7)):
            issues = set()
            for kind, bit in (("VIEW", 1), ("FAVORITE", 2), ("PURCHASE", 4)):
                if bits & bit:
                    issues.update(_issues(rows[kind + "_users"].total))
            rows[name] = _count(sum(n for mask, n in masks.items() if mask & bits == bits), issues)
        return rows

    def _statuses(self):
        keys = ({(code, "CATALOG") for code in STATUS_CATALOG}
                | {(None, bucket) for bucket in ("MISSING_CODE", "INVALID_CODE", "UNRESOLVED")}
                | self.raw_status.keys() | self.current_status.keys() | self.historical_status.keys())
        current = self.order_issues | self.status_issues
        historic = self.order_issues | self.purchase_issues | self.purchase_position_issues
        if self.historical_status[None, "UNRESOLVED"]:
            historic = historic | {(Quality.CONFLICT, "STATUS_CONFLICT")}
        return tuple(StatusAggregate(
            code, bucket, self.raw_status[code, bucket], _count(self.current_status[code, bucket], current),
            _count(self.status_orders[code, bucket], current),
            _count(self.historical_status[code, bucket], historic),
            _count(self.raw_status[code, bucket], self.raw_issues),
            STATUS_CATALOG[code].name if code in STATUS_CATALOG else None,
            STATUS_CATALOG[code].mindbox_category if code in STATUS_CATALOG else None,
            STATUS_CATALOG[code].group if code in STATUS_CATALOG else StatusGroup.UNKNOWN,
        ) for code, bucket in sorted(keys, key=lambda k: (k[1], k[0] or "")))

    def _chronology(self):
        firsts, intervals, first_codes, second_codes = Counter(), Counter(), Counter(), Counter()
        previous, first = None, None
        for customer, stamp, _, encoded in self.members.db.execute(
                "SELECT customer,stamp,rank,categories FROM chronology ORDER BY customer,stamp,rank"):
            self.check()
            occurred = datetime.fromisoformat(stamp)
            categories, unknown = json.loads(encoded)
            if customer != previous:
                first = occurred
                firsts[stamp[:7]] += 1
                first_codes.update(categories)
                self.diagnostics["first_category_unresolved"] += unknown
            else:
                intervals[(occurred.date() - first.date()).days] += 1
                second_codes.update(categories)
                self.diagnostics["second_category_unresolved"] += unknown
            previous = customer
        return tuple(tuple(sorted(counter.items())) for counter in (firsts, intervals, first_codes, second_codes))

    def result(self):
        db = self.members.db
        self.check()
        order_rows = {}
        for name in _ORDER_COUNTS:
            issues = self.order_issues
            if name == "positions":
                issues = issues | self.position_issues
            elif name in {"purchase_orders", "orders_without_evidence"}:
                issues = issues | self.purchase_issues
            elif name == "purchase_positions":
                issues = issues | self.purchase_position_issues
            elif name.startswith(("cancelled", "returned", "partial", "in_progress", "mixed", "fully_cancelled",
                                  "all_returned")):
                issues = issues | self.status_issues | self.state_issues
            order_rows[name] = _count(self.counts[name], issues)
        for name in ("buyers", "cancelled_customers", "returned_customers"):
            quality = self.purchase_issues if name == "buyers" else self.status_issues | self.state_issues
            order_rows[name] = _count(self.members.count(name), self.order_issues | quality | self.selected_owner_issues[name])
        order_rows["unassigned_currency_orders"] = _count(self.counts["unassigned_currency_orders"],
                                                          self.order_issues | self.purchase_issues)
        order_states = {kind.value: _count(self.order_states[kind], self.order_issues) for kind in OrderKind}
        order_rows["returned_units"] = AggregateMetric(
            None, _result(_ZERO, ((Quality.UNKNOWN, "RETURNED_QUANTITY_UNVERIFIED"),)))
        flags = {name: _count(self.flags[name], self.order_issues | self.status_issues | (self.state_issues if name != "unknown" else set()))
                 for name in ("completed", "partial", "cancelled", "returned", "in_progress", "unknown")}
        item_buyers = self.members.counts("item")
        products = tuple(item.result(code, item_buyers.get(code, 0),
                                     self.order_issues | self.product_issues,
                                     self.action_issues["PURCHASE_OWNER"])
                         for code, item in sorted(self.items.items()))
        frequency = tuple(db.execute("SELECT frequency,COUNT(*) FROM buyers GROUP BY frequency ORDER BY frequency"))
        buyers = sum(n for _, n in frequency)
        customers = {"population": _count(self.members.count("population"), self.population_issues)}
        for name in ("population_buyers", "nonpurchase_clients", "cancelled_clients", "returned_clients"):
            issues = (self.population_purchase_issues if name in ("population_buyers", "nonpurchase_clients")
                      else self.population_state_issues)
            customers[name] = _count(self.members.count(name), self.population_issues | issues)
        buyer_scope_issues = self.buyer_issues | self.buyer_purchase_issues
        buyer_issues = buyer_scope_issues | self.owner_issues
        customers.update(
            buyers=_count(buyers, buyer_issues),
            single_buyers=_count(dict(frequency).get(1, 0), buyer_issues),
            repeat_buyers=_count(sum(n for f, n in frequency if f > 1), buyer_issues),
            buyer_purchase_orders=_count(sum(f * n for f, n in frequency), buyer_issues),
            unattributed_orders=_count(self.diagnostics["unattributed_orders"], self.buyer_issues),
            owner_conflict_orders=_count(self.diagnostics["owner_conflict_orders"], self.buyer_issues),
        )
        category_issues = buyer_scope_issues | self.category_issues
        category_members, pair_members = self.members.counts("category"), self.members.counts("pair")
        categories = tuple(CategoryAggregate((key,), _count(n, category_issues),
                                             _count(category_members.get(key, 0), category_issues | self.owner_issues))
                           for key, n in sorted(self.categories.items()))
        pairs = tuple(CategoryAggregate(key, _count(n, category_issues),
                                        _count(pair_members.get(json.dumps(key, ensure_ascii=False), 0),
                                               category_issues | self.owner_issues))
                      for key, n in sorted(self.pairs.items()))
        chronology = self._chronology()
        namespaces = self._namespaces()
        customer_breakdowns = self.buyer_breakdowns.result(self._buyer_coverage, self.profiles_available)
        customer_breakdowns["copurchases"] = dict(
            orders=_count(self.history["buyer_period_purchase_orders"], buyer_scope_issues),
            multicategory_orders=_count(self.history["multicategory_orders"], category_issues),
            multicategory_buyers=_count(self.members.count("multicategory_buyers"), category_issues | self.owner_issues),
            unresolved_category_orders=self.diagnostics["orders_with_unresolved_categories"],
            labels={code: self.category_reference.get(code, (code, None)) for code in self.categories},
        )
        history = {
            "purchase_orders": _count(self.history["purchase_orders"], self.history_issues),
            "purchase_positions": _count(self.history["purchase_positions"], self.history_position_issues),
        }
        self.diagnostics["unattributed_purchase_orders"] = self.history["unattributed_purchase_orders"]
        self.diagnostics["products_with_purchases"] = sum(row.purchase_positions.known > 0 for row in products)
        self.check()
        return AggregateResult(self.contexts, order_rows, flags, self._statuses(), self._action_result(), customers,
                               frequency, *chronology, products, categories, pairs,
                               self.order_money.result(self.order_issues, self.members.buyer_money("order")),
                               self.buyer_money.result(buyer_scope_issues, self.members.buyer_money("buyer")),
                               dict(sorted(self.diagnostics.items())), history, order_states, dict(
                                   actions=dict(**self.action_breakdowns.result(), period=self.action_period,
                                                coverage=self.contexts[0].actions_intervals,
                                                source_events=dict(self.source_actions),
                                                event_names=[(*key, count) for key, count in sorted(self.event_names.items())]),
                                   orders=self.order_breakdowns.result(self.order_issues | self.purchase_issues,
                                                                      self.members.counts("order_store")),
                                   customer_demography=self.profile_summary,
                                   customer_activity=self.members.activity_summary(),
                                   buyer_breakdowns=customer_breakdowns,
                                   products=dict(product_breakdowns(
                                       products, self.metadata, self.members.counts("item_VIEW"),
                                       self.members.counts("item_FAVORITE")), namespaces=namespaces,
                                       resolution=self._resolution_summary(namespaces),
                                       category_labels={code: self.category_reference.get(code, (code, None))
                                                        for code in {item.category for item in self.metadata.values()}
                                                        if code}),
                               ))


def accumulate_aggregates(
    db, *, contexts: tuple[AnalyticsContext, ...], orders: Iterable[AggregateOrder],
    actions: Iterable[ActionEvidence], profile_keys: Iterable[str], metadata,
    product_predicate=None, action_period=None, profile_details=(), profiles_available=False,
    category_reference=None, check=lambda: None,
) -> AggregateResult:
    """Consume a pinned source once. The caller owns the disposable SQLite workspace."""
    expected = ("raw_orders", "order_product_slice", "item_interactions", "buyer_period", "customer_population")
    if tuple(context.context_id for context in contexts) != expected:
        raise ValueError("Invalid unified aggregate scopes")
    if len({(context.identity_space, context.policy, context.provenance_digest) for context in contexts}) != 1:
        raise ValueError("Unified aggregate scopes must share a pinned identity space")
    if (contexts[1].product_filter_key is None) != (product_predicate is None):
        raise ValueError("Unified product predicate does not match its declared scope")
    accumulator = _Aggregate(contexts, db, metadata, check, category_reference)
    accumulator.action_period = action_period
    if contexts[4].period is None:
        for key in profile_keys:
            check()
            accumulator.members.add("population", key)
    for i, entry in enumerate(orders, 1):
        check()
        accumulator.order(entry)
        if i % 1000 == 0:
            db.commit()
    for i, event in enumerate(actions, 1):
        check()
        accumulator.action(event, product_predicate)
        if i % 1000 == 0:
            db.commit()
    db.commit()
    accumulator.profiles_available = profiles_available
    buyer = contexts[3]
    accumulator._buyer_coverage = tuple((max(a, buyer.period[0]), min(b, buyer.period[1]))
                                        for a, b in buyer.orders_intervals
                                        if buyer.period is not None and max(a, buyer.period[0]) < min(b, buyer.period[1]))
    accumulator.profile_summary = accumulator.buyer_breakdowns.demographics(
        profile_details, available=profiles_available,
        restricted=action_period is not None or product_predicate is not None,
        as_of=buyer.calculated_at.astimezone().date(),
    )
    return accumulator.result()
