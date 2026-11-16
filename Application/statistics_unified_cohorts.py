"""Isolated all-product cohorts over reconciled UNIFIED_PURCHASE_V1 orders."""

from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from itertools import groupby
import json
from pathlib import Path
import sqlite3

from Application import dataset_statistics as reference
from Application.order_statistics import CURRENCIES
from Application.statistics_cohorts_categories import CategoryCounts, category_codes, category_reference
from Application.statistics_cohorts_channels import ChannelCounts, channel_value
from Application.statistics_cohorts_core import (
    HORIZONS, MONTH_OFFSETS, CohortAggregates, _Buyer, _interval_summary, _month, _month_date, _month_key, _utc,
)
from Application.statistics_cohorts_money import _add
from Application.statistics_cohorts_validation import cohort_cutoff
from Application.statistics_index import BuildGuard, GIB, manifest_digest
from Application.statistics_storage_safety import StatisticsIndexError, reject_reparse, validate_storage_root
from Application.statistics_unified_aggregate_input import _effective, _orders
from Application.statistics_unified_aggregates import AggregateMetric
from Application.statistics_unified_facts import AnalyticsContext, _issues, _result
from Application.statistics_unified_index import (
    _configure, _inputs, _separate, _staging, open_unified_index, reference_order_reader,
)
from Application.statistics_unified_policy import EvidenceQuality as Quality, Measured, PurchasePolicy


_ZERO = Decimal(0)
_NO_PURCHASE = (Quality.UNKNOWN, "NO_OBSERVED_PURCHASE")
_NO_ELIGIBLE = (Quality.INSUFFICIENT_COVERAGE, "NO_ELIGIBLE_BUYERS")
_PRODUCT = (Quality.INSUFFICIENT_COVERAGE, "PRODUCT_SCOPE_UNRESOLVED")
_SCHEMA = """
CREATE TABLE cohort_population(customer TEXT PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE cohort_orders(customer TEXT, uncertain INTEGER, stamp TEXT, rank INTEGER, payload TEXT NOT NULL,
                           PRIMARY KEY(customer, uncertain, stamp, rank)) WITHOUT ROWID;
CREATE TABLE cohort_category_pairs(first TEXT, second TEXT, buyers INTEGER NOT NULL,
                                   PRIMARY KEY(first, second)) WITHOUT ROWID;
"""


@dataclass(frozen=True, slots=True)
class ObservedPurchase:
    order_sort_rank: int
    created_at: datetime
    channel: Measured[tuple[str, str | None]]
    category_codes: tuple[str, ...]
    category_coverage: Measured[bool]


@dataclass(frozen=True, slots=True)
class CohortSequence:
    first: Measured[ObservedPurchase]
    second: Measured[ObservedPurchase]
    observed_first: ObservedPurchase | None
    observed_second: ObservedPurchase | None


@dataclass(frozen=True, slots=True, repr=False)
class UnifiedCohortsResult:
    context: AnalyticsContext
    identified_order_customers: int
    behavior: dict
    monetary: dict
    purchase_channels: dict
    purchase_categories: dict
    diagnostics: dict
    customer: CohortSequence | None


def _packed(issues):
    return sorted({(quality.value, reason) for quality, reason in issues})


def _unpacked(issues):
    return {(Quality(quality), reason) for quality, reason in issues}


def _money_payload(fact):
    currency_issues = _issues(fact.order_currency)
    rows, priced = {}, []
    for row in fact.money:
        subtotal = row.recognized_purchase.value if row.valued_purchase_positions else None
        if subtotal is not None:
            priced.append(row.currency)
        nominal = dict(row.known_nominals).get("partial_nominal")
        rows[row.currency] = (
            None if subtotal is None else str(subtotal), None if nominal is None else str(nominal),
            _packed([*currency_issues, *_issues(row.recognized_purchase)]), _packed(_issues(row.partial_nominal)),
        )
    return priced, rows


def _categories(fact, item_categories, categories, check):
    codes = set()
    attributed = 0
    issues = set(_issues(fact.purchase.positions))
    for item in fact.products:
        check()
        if item.purchase.known_positions:
            code = item_categories[item.item_id]
            if code in categories:
                codes.add(code)
                attributed += item.purchase.known_positions
    if issues or attributed != fact.purchase.known_positions or not codes:
        issues.add(_PRODUCT)
        codes.add(None)
    checked = category_codes(codes, categories)
    return sorted(code for code in checked if code is not None), _packed(issues)


def _spool_order(db, entry, core, item_categories, categories, diagnostics, check):
    fact = entry.whole
    if fact is None:
        diagnostics["ORDER_ID_INVALID"] += 1
        return
    purchase, stamp = fact.purchase.observed, fact.order_created_at.value
    owner = fact.customer_key.value
    if core is None:
        if owner is not None:
            db.execute("INSERT OR IGNORE INTO cohort_population VALUES(?)", (owner,))
        return
    if stamp is not None:
        if stamp >= core.end:
            diagnostics["facts_after_cutoff"] += purchase.value is True
            return
        index = bisect_right(core._starts, stamp) - 1
        if index < 0 or stamp >= core.coverage[index][1]:
            diagnostics["observed_purchase_facts_outside_coverage"] += purchase.value is True
            return
    if owner is None:
        if purchase.value is True:
            diagnostics["unattributed_purchase_orders"] += 1
            diagnostics.update(fact.customer_key.reasons)
        return
    db.execute("INSERT OR IGNORE INTO cohort_population VALUES(?)", (owner,))
    if purchase.value is False:
        return
    issues = _issues(purchase) + _issues(fact.order_created_at)
    if issues:
        diagnostics["orders_with_uncertain_sequence"] += 1
        diagnostics.update({reason for _, reason in issues})
        if stamp is not None:
            candidates = (stamp,)
        elif fact.observed_creation_dates and fact.order_created_at.reasons == ("CREATED_TIME_CONFLICT",):
            candidates = tuple(stamp for stamp in fact.observed_creation_dates if any(
                start <= stamp < end for start, end in core.coverage))
            if not candidates:
                return
        else:
            candidates = None
        payload = dict(issues=_packed(issues), dates=None if candidates is None else [d.isoformat() for d in candidates])
    else:
        codes, category_issues = _categories(fact, item_categories, categories, check)
        payload = dict(money=_money_payload(fact), channel=fact.channel.value, channel_issues=_packed(_issues(fact.channel)),
                       categories=codes, category_issues=category_issues)
    db.execute("INSERT INTO cohort_orders VALUES(?,?,?,?,?)", (owner, bool(issues),
               "" if stamp is None else stamp.isoformat(timespec="microseconds"), fact.order_sort_rank,
               json.dumps(payload, ensure_ascii=False)))


def _spool(db, orders, core, item_categories, categories, check):
    diagnostics = Counter()
    for i, entry in enumerate(orders, 1):
        check()
        if i % 1000 == 0:
            db.commit()
        _spool_order(db, entry, core, item_categories, categories, diagnostics, check)
    db.commit()
    return diagnostics


def _sum_optional(left, right):
    return left if right is None else right if left is None else _add(left, right)


def _mean(amount, count, *, percent=False):
    if not count or amount is None:
        return None
    with localcontext() as context:
        context.prec = max(28, len(amount.as_tuple().digits))
        context.rounding = ROUND_HALF_EVEN
        return (amount.scaleb(2) if percent else amount) / count


@dataclass(slots=True)
class _Cash:
    orders: int = 0
    priced_orders: int = 0
    valued: Counter = field(default_factory=Counter)
    valued_amount: dict = field(default_factory=dict)
    subtotal: dict = field(default_factory=dict)
    nominal: dict = field(default_factory=dict)
    issues: dict = field(default_factory=lambda: defaultdict(set))
    nominal_issues: dict = field(default_factory=lambda: defaultdict(set))

    def add(self, payload):
        self.orders += 1
        valued, rows = payload
        self.priced_orders += bool(valued)
        for currency, (subtotal, nominal, issues, nominal_issues) in rows.items():
            self.issues[currency].update(_unpacked(issues))
            self.nominal_issues[currency].update(_unpacked(nominal_issues))
            if subtotal is not None:
                amount = Decimal(subtotal)
                self.subtotal[currency] = _sum_optional(self.subtotal.get(currency), amount)
                if currency in valued:
                    self.valued[currency] += 1
                    self.valued_amount[currency] = _sum_optional(self.valued_amount.get(currency), amount)
            if nominal is not None:
                self.nominal[currency] = _sum_optional(self.nominal.get(currency), Decimal(nominal))

    def uncertain(self, issues):
        for currency in CURRENCIES:
            self.issues[currency].update(issues)
            self.nominal_issues[currency].update(issues)

    def merge(self, other):
        self.orders += other.orders
        self.priced_orders += other.priced_orders
        self.valued.update(other.valued)
        for currency in CURRENCIES:
            for name in ("valued_amount", "subtotal", "nominal"):
                target = getattr(self, name)
                target[currency] = _sum_optional(target.get(currency), getattr(other, name).get(currency))
            self.issues[currency].update(other.issues[currency])
            self.nominal_issues[currency].update(other.nominal_issues[currency])

    def currency(self, currency, extra):
        subtotal, nominal = self.subtotal.get(currency), self.nominal.get(currency)
        recognized = _result(subtotal or _ZERO, extra if subtotal is not None else
                             self.issues[currency] | extra | ({(Quality.UNKNOWN, "NO_ACCEPTED_PURCHASE_COST")}
                                                            if self.orders else set()))
        partial = _result(nominal or _ZERO, self.nominal_issues[currency] | extra)
        return dict(recognized=AggregateMetric(recognized.value if subtotal is None else subtotal, recognized),
                    partial_nominal=AggregateMetric(partial.value if nominal is None else nominal, partial),
                    valued_order_subtotal=self.valued_amount.get(currency),
                    valued_orders=self.valued[currency],
                    mean_valued_order=_mean(self.valued_amount.get(currency), self.valued[currency]))


@dataclass(slots=True)
class _MoneyGroup:
    cash: _Cash = field(default_factory=_Cash)
    buyers: int = 0
    fully_valued: int = 0
    currency_buyers: Counter = field(default_factory=Counter)
    currency_amount: dict = field(default_factory=dict)

    def add(self, cash):
        self.buyers += 1
        self.cash.merge(cash)
        if cash.priced_orders:
            self.fully_valued += 1
            for currency, amount in cash.valued_amount.items():
                if amount is not None:
                    self.currency_buyers[currency] += 1
                    self.currency_amount[currency] = _sum_optional(self.currency_amount.get(currency), amount)

    def result(self, eligible):
        extra = set() if eligible else {_NO_ELIGIBLE}
        rows = {}
        for currency in CURRENCIES:
            row = self.cash.currency(currency, extra)
            n = self.currency_buyers[currency]
            rows[currency] = dict(**row, fully_valued_buyers=n,
                                  mean_per_valued_buyer=_mean(self.currency_amount.get(currency), n))
        return dict(eligible=eligible, buyers=self.buyers, fully_valued_buyers=self.fully_valued,
                    unvalued_buyers=self.buyers-self.fully_valued, orders=self.cash.orders,
                    unvalued_orders=self.cash.orders-self.cash.priced_orders, currencies=rows)


@dataclass(slots=True)
class _MonthMoney:
    total: _MoneyGroup = field(default_factory=_MoneyGroup)
    first: _MoneyGroup = field(default_factory=_MoneyGroup)
    repeat: _MoneyGroup = field(default_factory=_MoneyGroup)


class _CategoryPairs:
    """Counter operations used by the existing category transitions calculator."""

    def __init__(self, db, check):
        self.db, self.check = db, check
        self.size = 0

    def __len__(self):
        return self.size

    def __contains__(self, pair):
        return bool(self[pair])

    def __getitem__(self, pair):
        row = self.db.execute("SELECT buyers FROM cohort_category_pairs WHERE first=? AND second=?",
                              tuple(code or "" for code in pair)).fetchone()
        return row[0] if row else 0

    def __setitem__(self, pair, count):
        key = tuple(code or "" for code in pair)
        if self.db.execute("INSERT OR IGNORE INTO cohort_category_pairs VALUES(?,?,?)", (*key, count)).rowcount:
            self.size += 1
        else:
            self.db.execute("UPDATE cohort_category_pairs SET buyers=? WHERE first=? AND second=?", (count, *key))

    def items(self):
        with closing(self.db.execute("SELECT first,second,buyers FROM cohort_category_pairs")) as rows:
            for a, b, n in rows:
                self.check()
                yield (a or None, b or None), n


def _observed(stamp, rank, payload):
    channel = tuple(payload["channel"]) if payload["channel"] is not None else None
    return ObservedPurchase(rank, stamp, _result(channel, _unpacked(payload["channel_issues"])),
                            tuple(payload["categories"]), _result(True, _unpacked(payload["category_issues"])))


@dataclass(slots=True)
class _Sequence:
    dated: _Buyer = field(default_factory=_Buyer)
    first: ObservedPurchase | None = None
    second: ObservedPurchase | None = None
    first_money: _Cash = field(default_factory=_Cash)
    second_money: _Cash = field(default_factory=_Cash)
    later_money: _Cash = field(default_factory=_Cash)
    first_issues: set = field(default_factory=set)
    second_issues: set = field(default_factory=set)
    money_issues: set = field(default_factory=set)
    repeat_risks: dict = field(default_factory=lambda: defaultdict(set))
    retention_risks: dict = field(default_factory=lambda: defaultdict(set))

    def add(self, stamp, rank, payload):
        stamp = datetime.fromisoformat(stamp)
        key = stamp, 0, rank
        self.dated.add(key)
        if self.first is None:
            self.first = _observed(stamp, rank, payload)
            self.first_money.add(payload["money"])
        else:
            if self.second is None:
                self.second = _observed(stamp, rank, payload)
                self.second_money.add(payload["money"])
            self.later_money.add(payload["money"])

    def uncertain(self, rank, payload):
        issues = _unpacked(payload["issues"])
        if self.first is None or payload["dates"] is None:
            self.first_issues.update(issues)
            self.second_issues.update(issues)
            return
        for stamp in map(datetime.fromisoformat, payload["dates"]):
            key = stamp, 0, rank
            if key < self.dated.first:
                self.first_issues.update(issues)
            if self.dated.second is None or key < self.dated.second:
                self.second_issues.update(issues)
            self.money_issues.update(issues)
            days = (stamp.date() - self.first.created_at.date()).days
            for horizon in HORIZONS:
                if 0 <= days <= horizon:
                    self.repeat_risks[horizon].update(issues)
            offset = _month(stamp) - _month(self.first.created_at)
            if 0 < offset <= MONTH_OFFSETS[-1]:
                self.retention_risks[offset].update(issues)

    def result(self):
        first = self.first_issues or ({_NO_PURCHASE} if self.first is None else set())
        second = self.first_issues | self.second_issues
        if not second and self.second is None:
            second.add(_NO_PURCHASE)
        return CohortSequence(_result(self.first, first), _result(self.second, second), self.first, self.second)


def _category_bucket(purchase):
    if purchase is None:
        return frozenset({None})
    codes = set(purchase.category_codes)
    if purchase.category_coverage.value is None:
        codes.add(None)
    return frozenset(codes)


@dataclass(slots=True)
class _Risks:
    count: int = 0
    issues: set = field(default_factory=set)

    def add(self, issues):
        self.count += 1
        self.issues.update(issues)

    def merge(self, other):
        self.count += other.count
        self.issues.update(other.issues)

    def apply(self, row, rate):
        observed_rate = row[rate]
        issues = self.issues
        if not row["eligible"]:
            issues = {(Quality.INSUFFICIENT_COVERAGE, reason) for reason in row["reasons"]} or {_NO_ELIGIBLE}
        row["uncertain_eligible_buyers"] = self.count
        row["confirmed_without_event"] -= self.count
        row["rate_quality"] = _result(observed_rate, issues)
        row[rate] = row["rate_quality"].value


class _Buyers:
    """Single-pass buyer source for the legacy cohort formulas and unified sidecars."""

    def __init__(self, db, core, reference, diagnostics, customer_key, check, source_db=None):
        self.db, self.core, self.reference, self.check = db, core, reference, check
        self.diagnostics, self.customer_key = diagnostics, customer_key
        self.source_db = db if source_db is None else source_db
        self.base_buyers = 0
        self.customer = None
        self.channels = ChannelCounts()
        self.categories = CategoryCounts()
        self.categories.pairs = _CategoryPairs(db, check)
        self.money = defaultdict(_MonthMoney)
        self.horizon_money = {horizon: _MoneyGroup() for horizon in HORIZONS}
        self.repeat_risks, self.retention_risks = defaultdict(_Risks), defaultdict(_Risks)
        self.known_intervals = defaultdict(Counter)
        self.incomplete_intervals = Counter()

    def __len__(self):
        return self.base_buyers

    def values(self):
        with closing(self.source_db.execute("""SELECT customer,uncertain,stamp,rank,payload FROM cohort_orders
                                       ORDER BY customer,uncertain,stamp,rank""")) as rows:
            for customer, group in groupby(rows, key=lambda row: row[0]):
                self.check()
                sequence = _Sequence()
                for _, uncertain, stamp, rank, payload in group:
                    self.check()
                    data = json.loads(payload)
                    if uncertain:
                        sequence.uncertain(rank, data)
                    else:
                        sequence.add(stamp, rank, data)
                result = sequence.result()
                if customer == self.customer_key:
                    self.customer = result
                if sequence.first_issues:
                    self.diagnostics["first_purchase_unresolved"] += 1
                    continue
                if sequence.first is None:
                    continue
                self.base_buyers += 1
                month = _month(sequence.first.created_at)
                if ((self.core.cohort_start is None or self.core.cohort_start <= month)
                        and (self.core.cohort_end is None or month <= self.core.cohort_end)):
                    self._collect(sequence, result, month)
                yield sequence.dated
        if self.customer is None and self.customer_key is not None:
            member = self.source_db.execute("SELECT 1 FROM cohort_population WHERE customer=?", (self.customer_key,)).fetchone()
            if member is not None:
                self.customer = _Sequence().result()

    def _collect(self, sequence, result, month):
        first = result.first.value
        second = result.second.value
        has_repeat = sequence.second is not None
        if sequence.second_issues:
            self.diagnostics["second_purchase_unresolved"] += 1
        if first.category_coverage.value is None:
            self.diagnostics["first_category_unresolved"] += 1
        if has_repeat and (second is None or second.category_coverage.value is None):
            self.diagnostics["second_category_unresolved"] += 1
        unknown_channel = (None, None)
        self.channels.add(channel_value(*(first.channel.value or unknown_channel)),
                          channel_value(*((second.channel.value if second else None) or unknown_channel)) if has_repeat else None)
        self.categories.add(_category_bucket(first), _category_bucket(second) if has_repeat else None, self.check)

        amounts = self.money[month]
        amounts.first.add(sequence.first_money)
        sequence.later_money.uncertain(sequence.money_issues)
        if sequence.later_money.orders:
            amounts.repeat.add(sequence.later_money)
        else:
            amounts.repeat.cash.uncertain(sequence.money_issues)
        total = _Cash()
        total.merge(sequence.first_money)
        total.merge(sequence.later_money)
        amounts.total.add(total)
        observed_days = None if not has_repeat else (sequence.second.created_at.date() - first.created_at.date()).days
        if second is not None:
            self.known_intervals[month][observed_days] += 1
            self.incomplete_intervals[month] += not self.core._covered(
                _utc(first.created_at.date()), _utc(second.created_at.date() + timedelta(days=1)))
        for horizon in HORIZONS:
            self.check()
            censored, incomplete = self.core._eligibility(
                _utc(first.created_at.date()), _utc(first.created_at.date() + timedelta(days=horizon + 1)))
            if censored or incomplete:
                continue
            if observed_days is not None and observed_days <= horizon:
                if second is not None:
                    self.horizon_money[horizon].add(sequence.second_money)
                else:
                    unknown = _Cash(orders=1)
                    unknown.uncertain(sequence.second_issues)
                    self.horizon_money[horizon].add(unknown)
            elif sequence.repeat_risks[horizon]:
                risk = sequence.repeat_risks[horizon]
                self.repeat_risks[month, horizon].add(risk)
                self.horizon_money[horizon].cash.uncertain(risk)

        for offset, issues in sequence.retention_risks.items():
            start, end = (_utc(_month_date(month + i)) for i in (offset, offset + 1))
            censored, incomplete = self.core._eligibility(start, end)
            if not (censored or incomplete or (sequence.dated.months >> offset) & 1):
                self.retention_risks[month, offset].add(issues)

    def result(self, behavior):
        repeat_summary, retention_summary = defaultdict(_Risks), defaultdict(_Risks)
        hist = Counter()
        incomplete = 0
        month_money = {}
        for key, row in behavior["cohorts"].items():
            self.check()
            year, month = map(int, key.split("-"))
            month = _month(date(year, month, 1))
            intervals = self.known_intervals[month]
            interval_incomplete = self.incomplete_intervals[month]
            row["second_purchase_days"] = dict(**_interval_summary(intervals), coverage_incomplete_buyers=interval_incomplete)
            hist.update(intervals)
            incomplete += interval_incomplete
            for horizon in HORIZONS:
                risk = self.repeat_risks[month, horizon]
                risk.apply(row["horizons"][str(horizon)], "repeat_rate")
                repeat_summary[horizon].merge(risk)
            for offset in MONTH_OFFSETS[1:]:
                risk = self.retention_risks[month, offset]
                risk.apply(row["retention"][offset], "retention_rate")
                retention_summary[offset].merge(risk)
            amounts = self.money[month]
            size = row["cohort_size"]
            total = amounts.total.result(size)
            repeated = amounts.repeat.result(size)
            for currency in CURRENCIES:
                amount = total["currencies"][currency]["valued_order_subtotal"]
                repeat_amount = repeated["currencies"][currency]["valued_order_subtotal"]
                total["currencies"][currency]["valued_repeat_amount_rate"] = (
                    _mean(repeat_amount or _ZERO, amount, percent=True))
            month_money[key] = dict(total=total, first=amounts.first.result(size), repeat=repeated)
        summary = behavior["summary"]
        summary["second_purchase_days"] = dict(**_interval_summary(hist), coverage_incomplete_buyers=incomplete)
        for horizon in HORIZONS:
            repeat_summary[horizon].apply(summary["horizons"][str(horizon)], "repeat_rate")
        for offset in MONTH_OFFSETS[1:]:
            retention_summary[offset].apply(summary["retention"][offset], "retention_rate")
        horizon_money = {}
        for horizon, group in self.horizon_money.items():
            row = summary["horizons"][str(horizon)]
            horizon_money[str(horizon)] = dict(**group.result(row["eligible"]),
                                               uncertain_eligible_buyers=row["uncertain_eligible_buyers"])
        return dict(cohorts=month_money, horizons=horizon_money)


def _context(reader, calculated_at, selected_start, selected_end, check):
    if calculated_at.utcoffset() is None:
        raise ValueError("Unified cohorts require an aware calculation timestamp")
    calculated_at = calculated_at.astimezone(timezone.utc)
    provenance = reader.manifest["binding"]["provenance"]
    coverage = _effective(provenance, "orders")
    cutoff = cohort_cutoff(coverage, selected_end, calculated_at)
    start_month = date.fromisoformat(selected_start).isoformat()[:7] if selected_start is not None else None
    core = None if cutoff is None else CohortAggregates(
        cutoff, coverage, identity_space=reader.identity_space, cohort_start=start_month,
        cohort_end=cutoff.isoformat()[:7] if selected_end is not None else None, check=check,
    )
    period = (core.coverage[0][0], core.end) if core is not None and core.coverage else None
    context = AnalyticsContext(
        "cohorts_history", PurchasePolicy(), reader.identity_space, calculated_at, cutoff, period, None,
        coverage, (), "DECLARED_ORDERS_PROXY_V1", False, False, manifest_digest(provenance),
    )
    return context, core


def _summarize(db, context, core, categories, diagnostics, customer_key, check, *, source_db=None):
    start_month = None if core is None or core.cohort_start is None else _month_key(core.cohort_start)
    end_month = None if core is None or core.cohort_end is None else _month_key(core.cohort_end)
    source = db if source_db is None else source_db
    population = source.execute("SELECT COUNT(*) FROM cohort_population").fetchone()[0]
    if core is None:
        check()
        return UnifiedCohortsResult(context, population, {}, {}, {}, {}, {"availability": "NO_ORDERS_COVERAGE"}, None)
    buyers = _Buyers(db, core, categories, diagnostics, customer_key, check, source_db=source_db)
    behavior = core._policy_result(buyers)
    monetary = buyers.result(behavior)
    channels = buyers.channels.result(check)
    purchase_categories = buyers.categories.result(categories, check)
    diagnostics = dict(sorted(diagnostics.items()))
    diagnostics.update(source_coverage=[[a.isoformat(), b.isoformat()] for a, b in core.source_coverage],
                       effective_coverage=[[a.isoformat(), b.isoformat()] for a, b in core.coverage],
                       cutoff_exclusive=core.end.isoformat(), cohort_definition="FIRST_OBSERVED_PURCHASE_UTC_MONTH",
                       cohort_bounds=dict(start=start_month, end=end_month),
                       lifetime_first_purchase_proven=False, gaps_can_hide_earlier_orders=True)
    check()
    return UnifiedCohortsResult(context, population, behavior, monetary, channels, purchase_categories,
                                diagnostics, buyers.customer)


def _calculate(reader, root, scratch_dir, expected_bytes, cancelled, observer, calculated_at,
               selected_start, selected_end, customer_key, item_categories, categories):
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError("Positive cohorts workspace estimate required")
    scratch = reject_reparse(scratch_dir)
    _separate(scratch, root)
    with _staging(scratch, validate_storage_root(scratch)) as (folder, _, _):
        guard = BuildGuard(folder, cancelled, observer, 20 * GIB)
        guard.check(force=True, extra=expected_bytes)
        context, core = _context(reader, calculated_at, selected_start, selected_end, guard.check)
        try:
            with closing(sqlite3.connect(folder / "statistics.sqlite")) as db:
                _configure(db, guard, cache_kib=32768)
                db.executescript(_SCHEMA)
                with closing(_orders(reader, None, None, guard.check)) as orders:
                    diagnostics = _spool(db, orders, core, item_categories, categories, guard.check)
                result = _summarize(db, context, core, categories, diagnostics, customer_key, guard.check)
                guard.check(force=True)
                return result
        except sqlite3.Error:
            if guard.failure is not None:
                raise guard.failure from None
            raise StatisticsIndexError("UNIFIED_COHORTS_FAILED") from None


def cohorts_reference(*, raw_root, catalog_path, scratch_dir, expected_bytes, calculated_at,
                      selected_start=None, selected_end=None, customer_key=None, cancelled=None, observer=None):
    """Calculate cohorts from canonical Orders. Optional customer key is source-local."""
    root, catalog = _inputs(raw_root, catalog_path)
    with reference_order_reader(raw_root=root, catalog_path=catalog, scratch_dir=scratch_dir,
                                expected_bytes=expected_bytes, cancelled=cancelled, observer=observer) as reader:
        _, items, names, paths = reference._catalog_snapshot(catalog)
        return _calculate(reader, root, scratch_dir, expected_bytes, cancelled, observer, calculated_at, selected_start,
                          selected_end, customer_key, {code: item.category for code, item in items.items()},
                          category_reference(names, paths))


def _indexed_categories(db, check):
    names, paths = {}, {}
    for code, name, path in db.execute("SELECT code,name,path FROM category_metadata"):
        check()
        names[code] = name
        paths[code] = json.loads(path)
    return names, paths


def cohorts_indexed(*, raw_root, catalog_path, index_dir, scratch_dir, expected_bytes, calculated_at,
                    selected_start=None, selected_end=None, customer_key=None, cancelled=None, observer=None):
    """Calculate cohorts from schema 4 Orders and its pinned category metadata."""
    def check():
        if cancelled and cancelled():
            raise InterruptedError("Unified cohorts cancelled")

    check()
    with open_unified_index(raw_root=raw_root, catalog_path=catalog_path, index_dir=index_dir, check=check) as reader:
        items = dict(reader.db.execute("SELECT code,category FROM item_metadata"))
        names, paths = _indexed_categories(reader.db, check)
        return _calculate(reader, Path(raw_root).absolute(), scratch_dir, expected_bytes, cancelled, observer,
                          calculated_at, selected_start, selected_end, customer_key, items, category_reference(names, paths))
