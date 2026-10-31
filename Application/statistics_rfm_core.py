"""Streaming RFM over selected PurchaseOrderFact objects; no I/O or public IDs.

The caller owns canonical identities and first-snapshot deduplication. All
windows are accumulated together; result() revisits compact buyer state only.
"""

from bisect import bisect_left
from calendar import monthrange
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from itertools import accumulate

from Application.order_statistics import CURRENCIES, histogram_median
from Application.statistics_purchase_facts import PurchaseOrderFact, MonetaryIssue
from Application.statistics_purchase_policy import STRICT_COMPLETED, EXTENDED_COMPLETED
from Application.statistics_period import shared_intervals


WINDOWS = ('LAST_12M', 'LAST_24M', 'LAST_48M', 'PREVIOUS_12M')
POLICIES = (STRICT_COMPLETED, EXTENDED_COMPLETED)
SEGMENTS = ('CHAMPIONS', 'LOYAL', 'NEW_OR_RECENT', 'AT_RISK', 'HIBERNATING', 'OTHER', 'UNSCORABLE')


def _shift_months(value, months):
    year, month = divmod(value.year * 12 + value.month - 1 + months, 12)
    return value.replace(year=year, month=month+1, day=min(value.day, monthrange(year, month+1)[1]))


def rfm_windows(as_of_date):
    """Half-open UTC windows: shift the next calendar day by 12/24/48 months."""
    if type(as_of_date) is not date:
        raise ValueError('RFM requires an explicit calendar as_of_date')
    try:
        end = as_of_date + timedelta(days=1)
        starts = [_shift_months(end, -months) for months in (12, 24, 48)]
    except (OverflowError, ValueError):
        raise ValueError('RFM date cannot represent all four windows') from None
    utc = lambda day: datetime.combine(day, time.min, timezone.utc)
    return dict(zip(WINDOWS, [(utc(start), utc(end)) for start in starts] + [(utc(starts[1]), utc(starts[0]))]))


def empirical_thresholds(histogram):
    """Weighted nearest ranks ceil(k*N/5), k=1..4. Keep duplicate cuts."""
    if not histogram:
        return ()
    values = sorted(histogram)
    cumulative = list(accumulate(histogram[value] for value in values))
    total = cumulative[-1]
    return tuple(values[bisect_left(cumulative, rank)] for rank in empirical_ranks(total))


def rfm_score(value, thresholds, *, recency=False):
    if value is None or not thresholds:
        return None
    # Strict > keeps every tie on the same side of every (possibly equal) cut.
    crossed = sum(value > cut for cut in thresholds)
    return 5-crossed if recency else 1+crossed


def rfm_segment(r, f, m, frequency):
    """Ordered, mutually exclusive outcomes; missing M never earns loyalty."""
    if any(value is None for value in (r, f, m)):
        return 'UNSCORABLE'
    if r >= 4 and f >= 4 and m >= 4 and frequency >= 2:
        return 'CHAMPIONS'
    if r >= 3 and f >= 4 and frequency >= 2:
        return 'LOYAL'
    if r >= 4 and frequency == 1:
        return 'NEW_OR_RECENT'
    if r <= 2 and f >= 3 and frequency >= 2:
        return 'AT_RISK'
    if r <= 2 and (f <= 2 or frequency < 2):
        return 'HIBERNATING'
    return 'OTHER'


def _saved(intervals):
    return [[a.isoformat(), b.isoformat()] for a, b in intervals]


def _coverage(intervals, start, end):
    observed = [(max(a, start), min(b, end)) for a, b in intervals if max(a, start) < min(b, end)]
    status = 'FULL' if observed == [(start, end)] else 'PARTIAL' if observed else 'NONE'
    return dict(start=start.isoformat(), end=end.isoformat(), available=_saved(observed), status=status,
                reason=None if status == 'FULL' else 'COVERAGE_GAP_OR_TRUNCATION' if observed else 'NO_ORDERS_COVERAGE')


def _histogram(histogram, *, money=False):
    total = sum(histogram.values())
    summed = sum((v*n for v, n in histogram.items()), Decimal(0) if money else 0)
    median = histogram_median(histogram)
    return dict(buyers=total, distribution=[[str(v) if money else v, histogram[v]] for v in sorted(histogram)],
                mean=str(summed/total) if money and total else summed/total if total else None,
                median=str(median) if money and median is not None else median)


@dataclass(slots=True)
class _Window:
    orders: int = 0
    unvalued: int = 0
    currency_orders: list = field(default_factory=lambda: [0, 0])
    valued_orders: list = field(default_factory=lambda: [0, 0])
    amounts: list = field(default_factory=lambda: [Decimal(0), Decimal(0)])

    def add(self, purchase):
        self.orders += 1
        if purchase.currency in CURRENCIES:
            index = CURRENCIES.index(purchase.currency)
            self.currency_orders[index] += 1
            if purchase.monetary_available:
                self.valued_orders[index] += 1
                self.amounts[index] += purchase.amount
        self.unvalued += int(not purchase.monetary_available)


@dataclass(slots=True)
class _Buyer:
    windows: list = field(default_factory=lambda: [None] * 4)
    last: datetime | None = None
    currency_last: dict = field(default_factory=dict)

    def add(self, purchase, occurred, indices):
        self.last = max(self.last, occurred) if self.last is not None else occurred
        if purchase.currency in CURRENCIES:
            index = CURRENCIES.index(purchase.currency)
            self.currency_last[index] = max(self.currency_last.get(index, occurred), occurred)
        for index in indices:
            if self.windows[index] is None:
                self.windows[index] = _Window()
            self.windows[index].add(purchase)


def _observation(buyer, window, currency, as_of, reliable_since, complete):
    state = buyer.windows[window]
    last = buyer.last if currency is None else buyer.currency_last[currency]
    recency = (as_of-last.date()).days
    reliable = reliable_since is not None and last >= reliable_since
    frequency = (state.orders if currency is None else state.currency_orders[currency]) if state else 0
    # Conservatively match BuyerAggregates: any unvalued policy order taints
    # the full buyer Monetary of both currencies, only inside this window.
    amount = ((state.amounts[currency] if state else Decimal(0))
              if currency is not None and complete and (state is None or not state.unvalued) else None)
    return recency, reliable, frequency, amount


class RfmAggregates:
    def __init__(self, as_of_date, coverage, *, identity_space, policies=POLICIES, check=None):
        if not isinstance(identity_space, str) or not identity_space.strip() or identity_space != identity_space.strip():
            raise ValueError('Invalid RFM identity space')
        if isinstance(policies, str):
            raise ValueError('RFM policies must be a nonempty collection')
        requested = tuple(policies)
        if not requested or any(p not in POLICIES for p in requested) or len(set(requested)) != len(requested):
            raise ValueError('Invalid RFM policies')
        self.as_of_date = as_of_date
        self.windows = rfm_windows(as_of_date)
        self.identity_space = identity_space
        self.policies = tuple(p for p in POLICIES if p in requested)
        self.check = check if check is not None else lambda: None
        intervals = [(a.isoformat() if isinstance(a, datetime) else a,
                      b.isoformat() if isinstance(b, datetime) else b) for a, b in (coverage or ())]
        self.coverage = shared_intervals([], intervals)
        self.observation = {key: _coverage(self.coverage, *value) for key, value in self.windows.items()}
        end = self.windows['LAST_48M'][1]
        # A continuous suffix proves that no more recent order is hidden after
        # a buyer's latest observed purchase. Earlier gaps need not poison R.
        self.reliable_since = next((a for a, b in self.coverage if a < end <= b), None)
        self._buyers = {p: {} for p in self.policies}
        self._issues = {p: [Counter() for _ in WINDOWS] for p in self.policies}

    def add(self, fact: PurchaseOrderFact):
        self.check()
        if not isinstance(fact, PurchaseOrderFact) or fact.identity_space != self.identity_space:
            raise ValueError('RFM requires facts from one explicit identity space')
        occurred = fact.order_datetime_utc
        start, end = self.windows['LAST_48M']
        if not start <= occurred < end:
            return
        if not any(a <= occurred < b for a, b in self.coverage):
            raise ValueError('RFM fact is outside declared Orders coverage')
        indices = [i for i, (a, b) in enumerate(self.windows.values()) if a <= occurred < b]
        for policy, purchase in ((STRICT_COMPLETED, fact.strict), (EXTENDED_COMPLETED, fact.extended)):
            if policy not in self._buyers or not purchase.has_purchase:
                continue
            buyers = self._buyers[policy]
            if fact.customer_id not in buyers:
                buyers[fact.customer_id] = _Buyer()
            buyers[fact.customer_id].add(purchase, occurred, indices)
            for index in indices:
                self._issues[policy][index].update(purchase.monetary_issues)

    def _slice(self, buyers, window, currency, complete):
        recencies, frequencies, monetary = Counter(), Counter(), Counter()
        reliable_recencies = Counter()
        active = orders = valued = money_incomplete = 0
        known_amount = Decimal(0)
        for buyer in buyers.values():
            self.check()
            if currency is not None and currency not in buyer.currency_last:
                continue
            r, reliable, f, m = _observation(buyer, window, currency, self.as_of_date, self.reliable_since, complete)
            recencies[r] += 1
            frequencies[f] += 1
            if reliable:
                reliable_recencies[r] += 1
            active += int(f > 0)
            orders += f
            if currency is not None:
                state = buyer.windows[window]
                if state is not None:
                    valued += state.valued_orders[currency]
                    known_amount += state.amounts[currency]
                    money_incomplete += int(state.unvalued > 0)
                if m is not None:
                    monetary[m] += 1
        count = sum(frequencies.values())
        result = dict(buyers=count, active_buyers=active, without_observed_orders=count-active,
                      confirmed_inactive_buyers=count-active if complete else None,
                      observed_orders=orders, recency=_histogram(recencies), frequency=_histogram(frequencies),
                      reliable_recency_buyers=sum(reliable_recencies.values()))
        if currency is None:
            return result
        r_cuts = empirical_thresholds(reliable_recencies) if complete else ()
        f_cuts = empirical_thresholds(frequencies) if complete else ()
        m_cuts = empirical_thresholds(monetary) if complete else ()
        result.update(known_amount=str(known_amount), valued_orders=valued, monetary=_histogram(monetary, money=True),
                      incomplete_money_buyers=money_incomplete, unavailable_monetary_buyers=count-sum(monetary.values()),
                      thresholds={key: dict(values=[str(v) for v in cuts] if key == 'M' else list(cuts),
                                            buyers=n, method='NEAREST_RANK_20_40_60_80_STRICT_GREATER')
                                  for key, cuts, n in (('R', r_cuts, sum(reliable_recencies.values()) if complete else 0),
                                                       ('F', f_cuts, count if complete else 0),
                                                       ('M', m_cuts, sum(monetary.values())))})
        scores, segments = Counter(), Counter()
        for buyer in buyers.values():
            self.check()
            if currency not in buyer.currency_last:
                continue
            r, reliable, f, m = _observation(buyer, window, currency, self.as_of_date, self.reliable_since, complete)
            rs = rfm_score(r, r_cuts, recency=True) if reliable else None
            fs, ms = rfm_score(f, f_cuts), rfm_score(m, m_cuts)
            scores[rs, fs, ms] += 1
            segments[rfm_segment(rs, fs, ms, f)] += 1
        # Canonical sorting handles optional scores and makes result independent
        # of source arrival order. Only aggregate score combinations leave here.
        result['score_counts'] = [[*key, scores[key]] for key in sorted(scores, key=lambda row: tuple(v or 0 for v in row))]
        result['segments'] = [[key, segments[key], 100*segments[key]/count if count else None] for key in SEGMENTS]
        return result

    def _comparison(self, buyers):
        reliable = all(self.observation[w]['status'] == 'FULL' for w in ('LAST_12M', 'PREVIOUS_12M'))
        groups = Counter()
        orders = [0, 0]
        known = [[Decimal(0), Decimal(0)] for _ in range(2)]
        incomplete = 0
        currency_incomplete = Counter()
        for buyer in buyers.values():
            self.check()
            states = [buyer.windows[i] for i in (0, 3)]
            active = tuple(bool(state and state.orders) for state in states)
            groups[{(True, True): 'BOTH', (True, False): 'LAST_ONLY',
                    (False, True): 'PREVIOUS_ONLY', (False, False): 'NEITHER'}[active]] += 1
            bad = not reliable or any(state and state.unvalued for state in states)
            incomplete += int(bool(bad))
            for currency in buyer.currency_last:
                currency_incomplete[currency] += int(bool(bad))
            for i, state in enumerate(states):
                if state:
                    orders[i] += state.orders
                    for c in range(len(CURRENCIES)):
                        known[i][c] += state.amounts[c]
        money = {}
        for c, currency in enumerate(CURRENCIES):
            last, previous = known[0][c], known[1][c]
            money[currency] = dict(last_known_amount=str(last), previous_known_amount=str(previous),
                                   known_amount_change=str(last-previous),
                                   known_amount_change_percent=str(100*(last-previous)/previous) if previous else None,
                                   unreliable_buyers=currency_incomplete[c])
        return dict(coverage_reliable=reliable, observed_groups={key: groups[key] for key in
                    ('BOTH', 'LAST_ONLY', 'PREVIOUS_ONLY', 'NEITHER')}, last_orders=orders[0], previous_orders=orders[1],
                    order_change=orders[0]-orders[1],
                    order_change_percent=100*(orders[0]-orders[1])/orders[1] if orders[1] else None,
                    unreliable_monetary_buyers=incomplete, money=money)

    def result(self):
        """Build a fresh aggregate result; cancellation never publishes partial data."""
        self.check()
        policies = {}
        for policy, buyers in self._buyers.items():
            windows = {}
            for index, name in enumerate(WINDOWS):
                complete = self.observation[name]['status'] == 'FULL'
                issues = self._issues[policy][index]
                unvalued = incomplete = 0
                for buyer in buyers.values():
                    self.check()
                    state = buyer.windows[index]
                    if state:
                        unvalued += state.unvalued
                        incomplete += int(state.unvalued > 0)
                windows[name] = dict(behavior=self._slice(buyers, index, None, complete),
                    currencies={currency: self._slice(buyers, index, c, complete) for c, currency in enumerate(CURRENCIES)},
                    quality=dict(unvalued_orders=unvalued, incomplete_money_buyers=incomplete,
                                 **{issue.value: issues[issue] for issue in MonetaryIssue}))
            dual = unassigned = 0
            for buyer in buyers.values():
                self.check()
                dual += int(len(buyer.currency_last) == 2)
                unassigned += int(not buyer.currency_last)
            policies[policy] = dict(base_buyers=len(buyers), dual_currency_buyers=dual,
                                    unassigned_currency_buyers=unassigned, windows=windows,
                                    comparison=self._comparison(buyers))
        self.check()
        # Copy metadata, including interval lists; callers cannot mutate state
        # by modifying an earlier result. Nothing reads or writes a cache here.
        return dict(as_of_date=self.as_of_date.isoformat(), windows={
            name: _coverage(self.coverage, *bounds) for name, bounds in self.windows.items()}, policies=policies)



def empirical_ranks(total):
    """The same four weighted nearest ranks for in-memory and disk histograms."""
    if type(total) is not int or total < 0:
        raise ValueError('Invalid RFM population')
    return tuple((k*total+4)//5 for k in range(1,5)) if total else ()


@dataclass(frozen=True, slots=True, repr=False)
class RfmObservation:
    policy: str
    window: str
    currency: str
    in_base: bool
    in_currency_base: bool
    complete_window: bool
    recency: int | None = None
    recency_reliable: bool | None = None
    frequency: int | None = None
    monetary: Decimal | None = None

    def scores(self, r_cuts, f_cuts, m_cuts):
        """Read-only scoring with the core's existing functions and availability."""
        if not self.in_currency_base:
            return None,None,None,None
        rs = rfm_score(self.recency,r_cuts,recency=True) if self.complete_window and self.recency_reliable else None
        fs = rfm_score(self.frequency,f_cuts) if self.complete_window else None
        ms = rfm_score(self.monetary,m_cuts) if self.complete_window else None
        return rs,fs,ms,rfm_segment(rs,fs,ms,self.frequency)


class RfmCustomerState:
    """Bounded one-customer adapter; the aggregate implementation stays private.

    Callers own unique facts, history identity and common population thresholds.
    observations() returns scalar snapshots, never _Buyer/_Window references.
    """
    def __init__(self, as_of_date, coverage, *, identity_space, check=None):
        self._aggregate = RfmAggregates(as_of_date,coverage,identity_space=identity_space,check=check)
        self._customer = None

    def add(self, fact):
        if not isinstance(fact,PurchaseOrderFact):
            raise ValueError('Invalid RFM customer fact')
        if self._customer is not None and self._customer != fact.customer_id:
            raise ValueError('RFM customer state cannot combine customers')
        self._aggregate.add(fact)
        self._customer = fact.customer_id

    def observations(self):
        a = self._aggregate
        result = []
        for policy in POLICIES:
            buyer = a._buyers[policy].get(self._customer)
            for index,window in enumerate(WINDOWS[:3]):
                a.check()
                complete = a.observation[window]['status']=='FULL'
                for currency,unit in enumerate(CURRENCIES):
                    assigned = buyer is not None and currency in buyer.currency_last
                    if assigned:
                        r,reliable,f,m = _observation(buyer,index,currency,a.as_of_date,a.reliable_since,complete)
                        result.append(RfmObservation(policy,window,unit,True,True,complete,r,reliable,f,m))
                    else:
                        result.append(RfmObservation(policy,window,unit,buyer is not None,False,complete))
        a.check()
        return tuple(result)
