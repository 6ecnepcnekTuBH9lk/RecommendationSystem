"""Isolated RFM over the same reconciled Orders stream as unified aggregates."""

from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from itertools import groupby
import json
from pathlib import Path
import sqlite3

from Application import dataset_statistics as legacy
from Application.mindbox.selection import MindboxSelectionConfig
from Application.order_statistics import CURRENCIES
from Application.product_resolution import ProductResolver
from Application.statistics_detail_rfm import decimal_compare
from Application.statistics_index import BuildGuard, GIB, manifest_digest
from Application.statistics_rfm_core import (
    SEGMENTS, _coverage, _histogram, empirical_ranks, rfm_score, rfm_segment, rfm_windows,
)
from Application.statistics_rfm_validation import rfm_cutoff
from Application.statistics_storage_safety import StatisticsIndexError, reject_reparse, validate_storage_root
from Application.statistics_unified_aggregate_input import (
    _effective, _indexed_actions, _indexed_profiles, _orders, _reference_actions, _reference_profiles,
)
from Application.statistics_unified_facts import AnalyticsContext, _issues, _result, _sum
from Application.statistics_unified_index import (
    _configure, _identities, _inputs, _separate, _staging, open_unified_index, reference_order_reader,
)
from Application.statistics_unified_policy import EvidenceQuality as Quality, Measured, PurchasePolicy


_ZERO = Decimal(0)
_SCOPES = (None, *CURRENCIES)
_GAP = (Quality.INSUFFICIENT_COVERAGE, "COVERAGE_GAP_OR_TRUNCATION")
_NO_PURCHASE = (Quality.UNKNOWN, "NO_OBSERVED_PURCHASE")
_OUTSIDE_CURRENCY = (Quality.UNKNOWN, "NOT_IN_CURRENCY_BASE")
_GROUPS = {(True, True): "BOTH", (True, False): "LAST_ONLY",
           (False, True): "PREVIOUS_ONLY", (False, False): "NEITHER"}


@dataclass(frozen=True, slots=True)
class RfmFeatures:
    in_base: bool
    observed_last_order: datetime | None
    last_order: Measured[datetime]
    recency: Measured[int]
    observed_frequency: int
    frequency: Measured[int]
    monetary: Measured[Decimal] | None
    known_subtotal: Decimal | None
    partial_nominal: Decimal | None
    valued_orders: int
    scores: tuple[int | None, int | None, int | None] = (None, None, None)
    segment: str | None = None


@dataclass(frozen=True, slots=True, repr=False)
class UnifiedRfmResult:
    context: AnalyticsContext
    population: int
    base_buyers: int
    dual_currency_buyers: int
    unassigned_currency_buyers: int
    windows: dict
    comparison: dict
    diagnostics: dict[str, int]
    customer: dict[str, dict[str | None, RfmFeatures]] | None


def _add(left, right):
    if right is None:
        return left
    return right if left is None else _sum([left, right])


def _subtract(left, right):
    return _sum([left, right.copy_negate()])


def _ratio(numerator, denominator, *, percent=False):
    if numerator is None or not denominator:
        return None
    denominator = Decimal(denominator)
    with localcontext() as context:
        context.prec = max(28, len(numerator.as_tuple().digits), len(denominator.as_tuple().digits))
        context.rounding = ROUND_HALF_EVEN
        return (numerator.scaleb(2) if percent else numerator) / denominator


def _pack(issues):
    return sorted({(quality.value, reason) for quality, reason in issues})


def _unpack(issues):
    return {(Quality(quality), reason) for quality, reason in issues}


def _covered(date, intervals):
    return any(start <= date < end for start, end in intervals)


def _order_payload(fact, date, possible, windows):
    known = fact.purchase.observed.value is True
    currency = fact.order_currency.value
    uncertain = _issues(fact.order_created_at) + _issues(fact.purchase.observed)
    money = {}
    money_issues = _issues(fact.order_currency)
    for row in fact.money:
        subtotal = row.known_purchase_subtotal.value
        if subtotal == 0 and row.currency != currency and not row.valued_purchase_positions:
            subtotal = None
        nominal = row.partial_nominal.value
        if nominal is None:
            nominal = dict(row.known_nominals).get("partial_nominal")
        money[row.currency] = [None if amount is None else str(amount) for amount in (subtotal, nominal)]
        if currency in (None, row.currency):
            money_issues.extend(_issues(row.recognized_purchase))
    valued = known and currency in CURRENCIES and not money_issues
    return json.dumps(dict(
        occurred=date.isoformat() if date is not None and known else None,
        possible=possible.isoformat(), windows=windows, currency=currency,
        issues=_pack(uncertain), valued=valued, money=money, money_issues=_pack(money_issues),
        # An uncertain purchase could be of either currency. Currency views count
        # only unambiguous observed purchases, even when a known purchase has no currency.
        any_currency=not known,
    ))


def _member(db, owner):
    if owner is not None:
        db.execute("INSERT OR IGNORE INTO rfm_population VALUES(?)", (owner,))


def _spool_order(db, entry, bounds, coverage, diagnostics):
    start, end = bounds.get("LAST_48M", (None, None))
    fact = entry.whole
    if fact is None:
        diagnostics["ORDER_ID_INVALID"] += 1
        return
    owner = fact.customer_key.value
    _member(db, owner)
    evidence = fact.purchase.observed
    if not bounds or evidence.value is False:
        return
    date = fact.order_created_at.value
    date_issues = _issues(fact.order_created_at)
    if date is None:
        diagnostics["purchase_date_unresolved"] += evidence.value is True
        diagnostics.update(reason for _, reason in date_issues)
    if owner is None:
        diagnostics["unattributed_purchase_orders"] += evidence.value is True
        diagnostics.update(fact.customer_key.reasons)
        return
    if date is not None:
        if not start <= date < end:
            return
        if not _covered(date, coverage):
            diagnostics["purchase_outside_orders_coverage"] += evidence.value is True
            return
        candidates = (date,)
    elif fact.observed_creation_dates and all(reason == "CREATED_TIME_CONFLICT" for _, reason in date_issues):
        candidates = tuple(date for date in fact.observed_creation_dates
                           if start <= date < end and _covered(date, coverage))
        if not candidates:
            return
    else:
        candidates = None
    possible = max(candidates) if candidates else end - timedelta(microseconds=1)
    windows = tuple(name for name, (lo, hi) in bounds.items()
                    if candidates is None or any(lo <= candidate < hi for candidate in candidates))
    diagnostics.update(evidence.reasons)
    if evidence.value is True:
        diagnostics.update(fact.order_currency.reasons)
    db.execute("INSERT INTO rfm_orders VALUES(?,?,?)", (owner, fact.order_sort_rank,
               _order_payload(fact, date, possible, windows)))


def _spool(db, orders, profiles, actions, bounds, context, check):
    diagnostics = Counter()
    for owner in profiles:
        check()
        _member(db, owner)
    for action in actions:
        check()
        _member(db, action.customer_key)
    for i, entry in enumerate(orders, 1):
        check()
        _spool_order(db, entry, bounds, context.orders_intervals, diagnostics)
        if i % 1000 == 0:
            db.commit()
    db.commit()
    return diagnostics


@dataclass(slots=True)
class _Window:
    frequency: Counter = field(default_factory=Counter)
    valued: Counter = field(default_factory=Counter)
    unvalued: int = 0
    subtotals: dict = field(default_factory=dict)
    valued_amount: dict = field(default_factory=dict)
    nominals: dict = field(default_factory=dict)
    frequency_issues: dict = field(default_factory=lambda: defaultdict(set))
    money_issues: set = field(default_factory=set)

    def add(self, order, scopes, issues, money_issues):
        if issues:
            for currency in scopes:
                self.frequency_issues[currency].update(issues)
            self.money_issues.update(issues | money_issues)
            return
        self.frequency.update(scopes)
        self.unvalued += not order["valued"]
        self.money_issues.update(money_issues)
        if order["valued"]:
            unit = order["currency"]
            self.valued[unit] += 1
            self.valued_amount[unit] = _add(self.valued_amount.get(unit), Decimal(order["money"][unit][0]))
        for currency, (subtotal, nominal) in order["money"].items():
            self.subtotals[currency] = _add(self.subtotals.get(currency), Decimal(subtotal) if subtotal is not None else None)
            self.nominals[currency] = _add(self.nominals.get(currency), Decimal(nominal) if nominal is not None else None)


class _Buyer:
    def __init__(self, windows):
        self.last = {}
        self.possible_last = defaultdict(dict)
        self.windows = {name: _Window() for name in windows}

    def add(self, order):
        currency = order["currency"]
        scopes = _SCOPES if order["any_currency"] else (None, currency) if currency is not None else (None,)
        issues = _unpack(order["issues"])
        money_issues = _unpack(order["money_issues"])
        date = datetime.fromisoformat(order["possible"])
        for scope in scopes:
            if order["occurred"] is not None:
                self.last[scope] = max(self.last.get(scope, date), date)
            else:
                for issue in issues:
                    dates = self.possible_last[scope]
                    dates[issue] = max(dates.get(issue, date), date)
        for name in order["windows"]:
            self.windows[name].add(order, scopes, issues, money_issues)

    def features(self, name, currency, as_of, reliable_since, complete):
        state = self.windows[name]
        last = self.last.get(currency)
        last_issues = {issue for issue, date in self.possible_last[currency].items() if last is None or date > last}
        if last is None:
            last_issues.add(_NO_PURCHASE)
        if last is not None and (reliable_since is None or last < reliable_since):
            last_issues.add(_GAP)
        gap = set() if complete else {_GAP}
        frequency = state.frequency[currency]
        f_issues = state.frequency_issues[currency] | gap
        subtotal = state.subtotals.get(currency)
        money = None
        if currency is not None:
            m_issues = state.money_issues | gap
            if last is None:
                m_issues = m_issues | {_OUTSIDE_CURRENCY}
            money = _result(subtotal if subtotal is not None else _ZERO, m_issues)
        return RfmFeatures(
            last is not None, last, _result(last, last_issues),
            _result((as_of - last.date()).days if last is not None else None, last_issues),
            frequency, _result(frequency, f_issues), money, subtotal, state.nominals.get(currency), state.valued[currency],
        )


_SCHEMA = """
CREATE TABLE rfm_population(customer TEXT PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE rfm_orders(customer TEXT, rank INTEGER, payload TEXT NOT NULL,
                        PRIMARY KEY(customer, rank)) WITHOUT ROWID;
CREATE TABLE rfm_scoring(id INTEGER PRIMARY KEY, window TEXT, currency TEXT, r INTEGER, f INTEGER, m TEXT, observed INTEGER, valued INTEGER, subtotal TEXT, valued_amount TEXT);
CREATE TABLE rfm_histogram(window TEXT, currency TEXT, metric TEXT, value TEXT COLLATE RFM_DECIMAL,
                           n INTEGER NOT NULL, PRIMARY KEY(window, currency, metric, value)) WITHOUT ROWID;
"""


def _add_scoring(db, name, currency, features, complete, valued_amount):
    values = _scorable(features, complete)
    db.execute("INSERT INTO rfm_scoring(window,currency,r,f,m,observed,valued,subtotal,valued_amount) VALUES(?,?,?,?,?,?,?,?,?)",
               (name, currency, values[0], values[1], None if values[2] is None else str(values[2]),
                features.observed_frequency, features.valued_orders,
                str(features.known_subtotal) if features.known_subtotal is not None else None,
                str(valued_amount) if valued_amount is not None else None))
    for metric, value in zip("RFM", values, strict=True):
        if value is not None:
            db.execute("""INSERT INTO rfm_histogram VALUES(?,?,?,?,1)
                          ON CONFLICT(window,currency,metric,value) DO UPDATE SET n=n+1""",
                       (name, currency, metric, str(value)))


def _scorable(features, complete):
    if not complete:
        return None, None, None
    return features.recency.value, features.frequency.value, features.monetary.value


def _thresholds(db, window, currency, check):
    thresholds = {}
    monetary = {}
    for metric in "RFM":
        key = window, currency, metric
        size = db.execute("SELECT COALESCE(SUM(n),0) FROM rfm_histogram WHERE window=? AND currency=? AND metric=?",
                          key).fetchone()[0]
        ranks = empirical_ranks(size)
        values, median = [], []
        total = _ZERO
        middle = ((size + 1) // 2, (size + 2) // 2)
        seen = 0
        with closing(db.execute("""SELECT value,n FROM rfm_histogram WHERE window=? AND currency=? AND metric=?
                                  ORDER BY value COLLATE RFM_DECIMAL""", key)) as rows:
            for value, count in rows:
                check()
                number = Decimal(value) if metric == "M" else int(value)
                if metric == "M":
                    with localcontext() as context:
                        context.prec = len(number.as_tuple().digits) + len(str(count))
                        total = _sum([total, number * count])
                    median.extend(number for position in middle if seen < position <= seen + count)
                seen += count
                while len(values) < len(ranks) and seen >= ranks[len(values)]:
                    values.append(number)
                if metric != "M" and len(values) == len(ranks):
                    break
        thresholds[metric] = dict(values=tuple(values), buyers=size,
                                  method="NEAREST_RANK_20_40_60_80_STRICT_GREATER")
        if metric == "M":
            monetary = dict(buyers=size, mean=_ratio(total, size), median=_ratio(_sum(median), len(median)))
    return thresholds, monetary


def _score(r, f, m, thresholds):
    scores = (rfm_score(r, thresholds["R"]["values"], recency=True),
              rfm_score(f, thresholds["F"]["values"]), rfm_score(m, thresholds["M"]["values"]))
    return scores, rfm_segment(*scores, f)


@dataclass(slots=True)
class _Slice:
    counts: Counter = field(default_factory=Counter)
    recency: Counter = field(default_factory=Counter)
    frequency: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    subtotal: Decimal | None = None
    partial_nominal: Decimal | None = None

    def add(self, row, currency, as_of):
        # Known subtotals from mixed orders stay visible even without a currency base.
        self.subtotal = _add(self.subtotal, row.known_subtotal)
        self.partial_nominal = _add(self.partial_nominal, row.partial_nominal)
        if not row.in_base:
            return
        self.counts["buyers"] += 1
        self.recency[(as_of - row.observed_last_order.date()).days] += 1
        self.frequency[row.observed_frequency] += 1
        self.counts["active_buyers"] += row.observed_frequency > 0
        self.counts["observed_orders"] += row.observed_frequency
        self.counts["confirmed_inactive_buyers"] += row.frequency.value == 0
        self.counts["reliable_recency_buyers"] += row.recency.value is not None
        self.counts["reliable_frequency_buyers"] += row.frequency.value is not None
        self.reasons.update(set(row.recency.reasons + row.frequency.reasons))
        if currency is not None:
            self.counts["valued_orders"] += row.valued_orders
            self.counts["unavailable_monetary_buyers"] += row.monetary.value is None
            self.reasons.update(set(row.monetary.reasons) - set(row.recency.reasons + row.frequency.reasons))

    def result(self, complete):
        buyers = self.counts["buyers"]
        counts = {name: self.counts[name] for name in (
            "buyers", "active_buyers", "observed_orders", "reliable_recency_buyers", "reliable_frequency_buyers")}
        return dict(**counts, without_observed_orders=buyers-self.counts["active_buyers"],
                    confirmed_inactive_buyers=self.counts["confirmed_inactive_buyers"] if complete else None,
                    valued_orders=self.counts["valued_orders"], known_subtotal=self.subtotal,
                    partial_nominal=self.partial_nominal, unavailable_monetary_buyers=self.counts["unavailable_monetary_buyers"],
                    recency=_histogram(self.recency), frequency=_histogram(self.frequency),
                    reasons=dict(sorted(self.reasons.items())))


@dataclass(slots=True)
class _Comparison:
    groups: Counter = field(default_factory=Counter)
    unreliable_money: Counter = field(default_factory=Counter)
    unreliable_frequency: int = 0

    def add(self, recent, previous, money_unreliable):
        self.groups[_GROUPS[recent[None].observed_frequency > 0, previous[None].observed_frequency > 0]] += 1
        self.unreliable_frequency += any(rows[None].frequency.value is None for rows in (recent, previous))
        self.unreliable_money[None] += money_unreliable
        for currency in CURRENCIES:
            self.unreliable_money[currency] += money_unreliable and recent[currency].in_base

    def result(self, windows):
        recent, previous = (windows[name] for name in ("LAST_12M", "PREVIOUS_12M"))
        reliable = all(window["coverage"]["status"] == "FULL" for window in (recent, previous))
        counts = [window["behavior"]["observed_orders"] for window in (recent, previous)]
        delta = counts[0] - counts[1]
        money = {}
        for currency in CURRENCIES:
            # Zero for an empty covered comparison side only. A side with
            # unvalued observed orders keeps its unavailable known subtotal.
            amounts = []
            for window in (recent, previous):
                row = window["currencies"][currency]
                amount = row["known_subtotal"]
                if (amount is None and window["coverage"]["status"] == "FULL"
                        and not window["quality"].get("unvalued_orders", 0)
                        and not window["quality"].get("uncertain_buyer_windows", 0)):
                    amount = _ZERO
                amounts.append(amount)
            last, prior = amounts
            change = _subtract(last, prior) if last is not None and prior is not None else None
            money[currency] = dict(last_known_amount=last, previous_known_amount=prior,
                                   known_amount_change=change, known_amount_change_percent=_ratio(change, prior, percent=True),
                                   unreliable_buyers=self.unreliable_money[currency])
        return dict(coverage_reliable=reliable, unreliable_frequency_buyers=self.unreliable_frequency,
                    observed_groups={key: self.groups[key] for key in _GROUPS.values()}, last_orders=counts[0],
                    previous_orders=counts[1], order_change=delta,
                    order_change_percent=100 * delta / counts[1] if counts[1] else None,
                    unreliable_monetary_buyers=self.unreliable_money[None], money=money)


def _context(reader, calculated_at, selected_end):
    if calculated_at.utcoffset() is None:
        raise ValueError("Unified RFM requires an aware calculation timestamp")
    calculated_at = calculated_at.astimezone(timezone.utc)
    provenance = reader.manifest["binding"]["provenance"]
    orders = _effective(provenance, "orders")
    cutoff = rfm_cutoff(orders, selected_end, calculated_at)
    windows = rfm_windows(cutoff) if cutoff is not None else {}
    context = AnalyticsContext(
        "rfm_48m", PurchasePolicy(), reader.identity_space, calculated_at, cutoff,
        windows.get("LAST_48M"), None, orders, (), "DECLARED_ORDERS_PROXY_V1", False, False,
        manifest_digest(provenance),
    )
    return context, windows


@dataclass(slots=True)
class _SegmentAmounts:
    orders: int = 0
    valued: int = 0
    complete: int = 0
    known: Decimal | None = None
    valued_amount: Decimal | None = None
    complete_amount: Decimal | None = None

    def add(self, orders, valued, subtotal, valued_amount, monetary):
        self.orders += orders
        self.valued += valued
        self.complete += monetary is not None
        self.known = _add(self.known, Decimal(subtotal) if subtotal is not None else None)
        self.valued_amount = _add(self.valued_amount, Decimal(valued_amount) if valued_amount is not None else None)
        self.complete_amount = _add(self.complete_amount, Decimal(monetary) if monetary is not None else None)

    def result(self, total):
        return dict(observed_orders=self.orders, valued_orders=self.valued, known_subtotal=self.known,
                    share_of_known_subtotal=_ratio(self.known, total, percent=True),
                    mean_valued_order=_ratio(self.valued_amount, self.valued), complete_buyers=self.complete,
                    mean_complete_m=_ratio(self.complete_amount, self.complete))


def _summarize(db, context, bounds, diagnostics, customer_key, check):
    population = db.execute("SELECT COUNT(*) FROM rfm_population").fetchone()[0]
    if not bounds:
        return UnifiedRfmResult(context, population, 0, 0, 0, {}, {}, dict(sorted(diagnostics.items())), None)
    end = bounds["LAST_48M"][1]
    coverage = {name: _coverage(context.orders_intervals, *interval) for name, interval in bounds.items()}
    reliable_since = next((a for a, b in context.orders_intervals if a < end <= b), None)
    complete = {name: row["status"] == "FULL" for name, row in coverage.items()}
    stats = {name: {currency: _Slice() for currency in _SCOPES} for name in bounds}
    quality = defaultdict(Counter)
    comparison = _Comparison()
    base = dual = unassigned = 0
    sample = None
    # Only one buyer's window states live in Python. The join can stream
    # directly in primary-key order with both tables on disk.
    with closing(db.execute("""SELECT p.customer,o.payload FROM rfm_population p
                               LEFT JOIN rfm_orders o USING(customer) ORDER BY p.customer""")) as rows:
        for i, (owner, group) in enumerate(groupby(rows, key=lambda row: row[0]), 1):
            check()
            buyer = _Buyer(bounds)
            for _, payload in group:
                check()
                if payload is not None:
                    buyer.add(json.loads(payload))
            if not buyer.last and not buyer.possible_last and owner != customer_key:
                continue
            in_base = None in buyer.last
            currencies = len(buyer.last) - int(in_base)
            base += in_base
            dual += currencies == 2
            unassigned += in_base and not currencies
            feature_rows = {}
            for name, scopes in stats.items():
                features = {}
                state = buyer.windows[name]
                if state.frequency_issues:
                    quality[name]["uncertain_buyer_windows"] += 1
                if state.unvalued:
                    quality[name]["unvalued_orders"] += state.unvalued
                if state.money_issues:
                    quality[name]["incomplete_money_buyers"] += 1
                    quality[name].update({reason for _, reason in state.money_issues})
                for currency, accumulator in scopes.items():
                    row = buyer.features(name, currency, context.as_of_date, reliable_since, complete[name])
                    accumulator.add(row, currency, context.as_of_date)
                    features[currency] = row
                    if currency is not None and row.in_base:
                        _add_scoring(db, name, currency, row, complete[name], state.valued_amount.get(currency))
                feature_rows[name] = features
            if in_base:
                names = ("LAST_12M", "PREVIOUS_12M")
                money_unreliable = any(not complete[name] or buyer.windows[name].money_issues for name in names)
                comparison.add(*(feature_rows[name] for name in names), money_unreliable)
            if owner == customer_key:
                sample = feature_rows
            if i % 1000 == 0:
                db.commit()
    db.commit()
    windows = {}
    for name, scopes in stats.items():
        windows[name] = dict(coverage=coverage[name], behavior=scopes[None].result(complete[name]),
                             currencies={}, quality=dict(sorted(quality[name].items())))
        for currency in CURRENCIES:
            result = scopes[currency].result(complete[name])
            result["thresholds"], result["monetary"] = _thresholds(db, name, currency, check)
            windows[name]["currencies"][currency] = result
            if sample is not None and sample[name][currency].in_base:
                row = sample[name][currency]
                scores, segment = _score(*_scorable(row, complete[name]), result["thresholds"])
                sample[name][currency] = replace(row, scores=scores, segment=segment)
    segments, scores = defaultdict(Counter), defaultdict(Counter)
    business = defaultdict(lambda: defaultdict(_SegmentAmounts))
    with closing(db.execute("SELECT window,currency,r,f,m,observed,valued,subtotal,valued_amount FROM rfm_scoring")) as rows:
        for name, currency, r, f, m, observed, valued, subtotal, valued_amount in rows:
            check()
            key = name, currency
            cuts = windows[name]["currencies"][currency]["thresholds"]
            values, segment = _score(r, f, None if m is None else Decimal(m), cuts)
            segments[key][segment] += 1
            scores[key][values] += 1
            business[key][segment].add(observed, valued, subtotal, valued_amount, m)
    for name in bounds:
        for currency in CURRENCIES:
            key = name, currency
            result = windows[name]["currencies"][currency]
            count = result["buyers"]
            result["segments"] = tuple((label, segments[key][label], 100 * segments[key][label] / count if count else None)
                                       for label in SEGMENTS)
            result["segment_business"] = {segment: business[key][segment].result(result["known_subtotal"]) for segment in SEGMENTS}
            result["score_counts"] = tuple((*values, scores[key][values])
                                           for values in sorted(scores[key], key=lambda row: tuple(v or 0 for v in row)))
    check()
    return UnifiedRfmResult(context, population, base, dual, unassigned, windows, comparison.result(windows),
                            dict(sorted(diagnostics.items())), sample)


def _calculate(reader, root, scratch_dir, expected_bytes, cancelled, observer, calculated_at, selected_end,
               customer_key, profiles, actions):
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError("Positive RFM workspace estimate required")
    context, bounds = _context(reader, calculated_at, selected_end)
    scratch = reject_reparse(scratch_dir)
    _separate(scratch, root)
    with _staging(scratch, validate_storage_root(scratch)) as (folder, _, _):
        guard = BuildGuard(folder, cancelled, observer, 20 * GIB)
        guard.check(force=True, extra=expected_bytes)
        try:
            with closing(sqlite3.connect(folder / "statistics.sqlite")) as db:
                _configure(db, guard, cache_kib=32768)
                db.create_collation("RFM_DECIMAL", decimal_compare)
                db.executescript(_SCHEMA)
                with closing(_orders(reader, None, None, guard.check, include_products=False)) as orders:
                    diagnostics = _spool(db, orders, profiles, actions, bounds, context, guard.check)
                result = _summarize(db, context, bounds, diagnostics, customer_key, guard.check)
                guard.check(force=True)
                return result
        except sqlite3.Error:
            if guard.failure is not None:
                raise guard.failure from None
            raise StatisticsIndexError("UNIFIED_RFM_FAILED") from None


def rfm_reference(*, raw_root, catalog_path, scratch_dir, expected_bytes, calculated_at,
                  selected_end=None, customer_key=None, cancelled=None, observer=None):
    """Read canonical Orders through the aggregate stream. Optional key is source-local."""
    root, catalog = _inputs(raw_root, catalog_path)
    with reference_order_reader(raw_root=root, catalog_path=catalog, scratch_dir=scratch_dir,
                                expected_bytes=expected_bytes, cancelled=cancelled, observer=observer) as reader:
        provenance = reader.manifest["binding"]["provenance"]
        identities = _identities(root, provenance, reader.check)
        selection = MindboxSelectionConfig(**provenance["canonical"]["selection"])
        resolver = ProductResolver(legacy._catalog_snapshot(catalog)[0])
        with closing(_reference_profiles(root, identities, reader.check)) as profiles, closing(
                _reference_actions(root, provenance, identities, resolver, selection, reader.check)) as actions:
            return _calculate(reader, root, scratch_dir, expected_bytes, cancelled, observer, calculated_at,
                              selected_end, customer_key, profiles, actions)


def rfm_indexed(*, raw_root, catalog_path, index_dir, scratch_dir, expected_bytes, calculated_at,
                selected_end=None, customer_key=None, cancelled=None, observer=None):
    """Read projected Orders, action identities and profile links from schema 4."""
    def check():
        if cancelled and cancelled():
            raise InterruptedError("Unified RFM cancelled")

    check()
    with open_unified_index(raw_root=raw_root, catalog_path=catalog_path, index_dir=index_dir, check=check) as reader:
        provenance = reader.manifest["binding"]["provenance"]
        selection = MindboxSelectionConfig(**provenance["canonical"]["selection"])
        with closing(_indexed_profiles(reader)) as profiles, closing(_indexed_actions(reader, selection, check)) as actions:
            return _calculate(reader, Path(raw_root).absolute(), scratch_dir, expected_bytes, cancelled, observer,
                              calculated_at, selected_end, customer_key, profiles, actions)
