"""Private streaming buyer state; aggregate-only results, no I/O or Qt."""

from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from itertools import accumulate

from Application.analysis_filter import AnalysisFilter
from Application.customer_demographics import AGE_LABELS, age_group, calendar_age
from Application.mindbox.adapters._common import AdapterError, birth_date
from Application.order_statistics import CURRENCIES, histogram_median
from Application.statistics_purchase_facts import MonetaryIssue
from Application.statistics_purchase_policy import STRICT_COMPLETED, EXTENDED_COMPLETED


POLICIES = (STRICT_COMPLETED, EXTENDED_COMPLETED)
GENDERS = ("Мужчины", "Женщины", "Не указан")
FREQUENCY_LABELS = ("1", "2", "3–5", "6–10", "11+")
INTERVAL_LABELS = ("0–30", "31–60", "61–90", "91–180", "181–365", "366+")
ANNUAL_MONEY_FIELDS = ("known_orders", "known_buyers", "amount", "complete_buyers", "complete_amount",
                       "mean_complete_per_buyer", "median_complete_per_buyer")


def rate(count, total):
    return 100 * count / total if total else None


def distribution(counts, labels, total):
    return [[label, counts[label], rate(counts[label], total)] for label in labels]


def mean_money(amount, count):
    return str(amount / count) if count else None


def median_money(histogram):
    value = histogram_median(histogram)
    return str(value) if value is not None else None


def clipped_coverage(intervals, selection):
    selected = selection.interval
    result = []
    for start, end in intervals:
        if selected:
            start, end = max(start, selected[0]), min(end, selected[1])
        if start < end:
            result.append((start, end))
    return result


def saved_coverage(intervals):
    return [[start.isoformat(), end.isoformat()] for start, end in intervals]


def covered_years(coverage):
    return sorted({year for start, end in coverage
                   for year in range(start.year, (end-timedelta(microseconds=1)).year+1)})


def quintiles(values, repeats):
    """Weighted nearest-rank cuts. Equal amounts always stay in one group."""
    if not values:
        return []
    ordered = sorted(values)
    cumulative = list(accumulate(values[value] for value in ordered))
    total = cumulative[-1]
    cuts = sorted({ordered[bisect_left(cumulative, (total * k + 4) // 5)] for k in range(1, 5)})
    cuts = [value for value in cuts if value < ordered[-1]]
    groups = []
    for value in ordered:
        index = bisect_left(cuts, value)
        while len(groups) <= index:
            groups.append([])
        groups[index].append(value)
    result = []
    for group in groups:
        if not group:
            continue
        count = sum(values[value] for value in group)
        amount = sum((value * values[value] for value in group), Decimal(0))
        repeat_count = sum(repeats[value] for value in group)
        repeat_amount = sum((value * repeats[value] for value in group), Decimal(0))
        result.append([str(group[0]), str(group[-1]), count, rate(count, total), str(amount),
                       repeat_count, rate(repeat_count, count), str(repeat_amount),
                       float(100 * repeat_amount / amount) if amount else None])
    return result


@dataclass(slots=True)
class _YearMoney:
    orders: int = 0
    unvalued: int = 0
    amounts: dict = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)

    def add(self, result):
        self.orders += 1
        if result.monetary_available:
            self.amounts[result.currency] = self.amounts.get(result.currency, Decimal(0)) + result.amount
            self.counts[result.currency] += 1
        else:
            self.unvalued += 1


@dataclass(slots=True)
class _Purchases:
    orders: int = 0
    positions: int = 0
    units: Decimal = Decimal(0)
    years: dict = field(default_factory=dict)

    def add(self, year, result):
        self.orders += 1
        self.positions += result.line_count
        self.units += result.units
        self.years.setdefault(year, _YearMoney()).add(result)


@dataclass(slots=True)
class _Buyer:
    extended: _Purchases = field(default_factory=_Purchases)
    strict: _Purchases | None = None
    days: array = field(default_factory=lambda: array('I'))
    strict_flags: array = field(default_factory=lambda: array('B'))
    profile: tuple | None = None


class _Finance:
    def __init__(self):
        self.known = Counter()
        self.complete = Counter()
        self.complete_repeats = Counter()
        self.repeat_amount = Decimal(0)

    def add(self, amount, complete, repeat):
        self.known[amount] += 1
        if repeat:
            self.repeat_amount += amount
        if complete:
            self.complete[amount] += 1
            self.complete_repeats[amount] += int(repeat)

    def result(self, orders, *, annual=False):
        n = sum(orders.values())
        amount = sum((value * count for value, count in orders.items()), Decimal(0))
        known = sum(self.known.values())
        complete = sum(self.complete.values())
        complete_amount = sum((value * count for value, count in self.complete.items()), Decimal(0))
        base = dict(known_orders=n, known_buyers=known, amount=str(amount), complete_buyers=complete,
                    complete_amount=str(complete_amount), mean_complete_per_buyer=mean_money(complete_amount, complete),
                    median_complete_per_buyer=median_money(self.complete))
        if annual:
            return base
        repeat_count = sum(self.complete_repeats.values())
        repeat_amount = sum((value * count for value, count in self.complete_repeats.items()), Decimal(0))
        return dict(base, mean_order=mean_money(amount, n), median_order=median_money(orders),
                    mean_known_per_buyer=mean_money(amount, known), median_known_per_buyer=median_money(self.known),
                    repeat_amount=str(self.repeat_amount),
                    repeat_amount_rate=float(100 * self.repeat_amount / amount) if amount else None,
                    complete_repeat_buyers=repeat_count, complete_repeat_rate=rate(repeat_count, complete),
                    complete_repeat_amount=str(repeat_amount),
                    complete_repeat_amount_rate=float(100 * repeat_amount / complete_amount) if complete_amount else None,
                    quintiles=quintiles(self.complete, self.complete_repeats))


class _Summary:
    def __init__(self):
        self.buyers = self.orders = self.positions = 0
        self.units = Decimal(0)
        self.frequencies, self.frequency_orders = Counter(), Counter()
        self.first_months, self.last_months, self.gaps = Counter(), Counter(), Counter()
        self.first = self.last = None
        self.quality = Counter()
        self.money = {currency: _Finance() for currency in CURRENCIES}
        self.annual = {}
        self.profiled = 0
        self.genders, self.ages = Counter(), Counter()

    def add(self, state, days, profile, check):
        self.buyers += 1
        self.orders += state.orders
        self.positions += state.positions
        self.units += state.units
        self.frequencies[state.orders] += 1
        label = FREQUENCY_LABELS[0 if state.orders == 1 else 1 if state.orders == 2 else
                                 2 if state.orders <= 5 else 3 if state.orders <= 10 else 4]
        self.frequency_orders[label] += state.orders
        self.first = min(self.first, days[0]) if self.first is not None else days[0]
        self.last = max(self.last, days[-1]) if self.last is not None else days[-1]
        self.first_months[date.fromordinal(days[0]).strftime('%Y-%m')] += 1
        self.last_months[date.fromordinal(days[-1]).strftime('%Y-%m')] += 1
        for i in range(1, len(days)):
            if i % 1024 == 0:
                check()
            self.gaps[days[i] - days[i-1]] += 1
        unvalued = sum(value.unvalued for value in state.years.values())
        known = state.orders - unvalued
        self.quality['complete_buyers' if not unvalued else 'partial_buyers' if known else 'unvalued_buyers'] += 1
        self.quality['unvalued_orders'] += unvalued
        for currency in CURRENCIES:
            if any(value.counts[currency] for value in state.years.values()):
                amount = sum((value.amounts.get(currency, Decimal(0)) for value in state.years.values()), Decimal(0))
                self.money[currency].add(amount, not unvalued, state.orders >= 2)
        for year, value in state.years.items():
            annual = self.annual.setdefault(year, dict(buyers=0, orders=0, unvalued_orders=0,
                                                       money={c: _Finance() for c in CURRENCIES}))
            annual['buyers'] += 1
            annual['orders'] += value.orders
            annual['unvalued_orders'] += value.unvalued
            for currency, amount in value.amounts.items():
                annual['money'][currency].add(amount, not value.unvalued, value.orders >= 2)
        if profile is not None:
            self.profiled += 1
        gender, age = profile if profile is not None else (GENDERS[-1], None)
        self.genders[gender] += 1
        self.ages[age] += 1

    def result(self, policy, histograms, issues, profiles_available, coverage):
        repeated = self.buyers - self.frequencies[1]
        counts = Counter()
        for frequency, n in self.frequencies.items():
            label = FREQUENCY_LABELS[0 if frequency == 1 else 1 if frequency == 2 else
                                     2 if frequency <= 5 else 3 if frequency <= 10 else 4]
            counts[label] += n
        frequency = [[label, counts[label], rate(counts[label], self.buyers), self.frequency_orders[label]]
                     for label in FREQUENCY_LABELS]
        gap_counts = Counter()
        for days, n in self.gaps.items():
            gap_counts[INTERVAL_LABELS[bisect_left((30, 60, 90, 180, 365), days)]] += n
        gap_n = sum(self.gaps.values())
        gap_total = sum(day * n for day, n in self.gaps.items())
        intervals = dict(buyers=repeated, count=gap_n, total_days=gap_total,
                         mean_days=gap_total / gap_n if gap_n else None,
                         median_days=histogram_median(self.gaps),
                         distribution=distribution(gap_counts, INTERVAL_LABELS, gap_n))
        money = {}
        for currency in CURRENCIES:
            histogram = Counter()
            for (year, unit), values in histograms.items():
                if unit == currency:
                    histogram.update(values)
            money[currency] = self.money[currency].result(histogram)
        quality = {key: self.quality[key] for key in ('complete_buyers', 'partial_buyers', 'unvalued_buyers', 'unvalued_orders')}
        quality.update(unknown_currency_orders=issues[MonetaryIssue.UNKNOWN_CURRENCY],
                       mixed_currency_orders=issues[MonetaryIssue.MIXED_CURRENCY],
                       invalid_price_orders=issues[MonetaryIssue.INVALID_PRICE])
        demography = None
        if profiles_available:
            ages = Counter({age: n for age, n in self.ages.items() if age is not None})
            known_age = sum(ages.values())
            groups = Counter()
            for age, n in self.ages.items():
                groups[age_group(age) if age is not None else AGE_LABELS[-1]] += n
            demography = dict(profiled_buyers=self.profiled, missing_profiles=self.buyers-self.profiled,
                              unknown_gender=self.genders[GENDERS[-1]], unknown_age=self.ages[None],
                              mean_age=sum(age*n for age, n in ages.items()) / known_age if known_age else None,
                              median_age=histogram_median(ages), gender=distribution(self.genders, GENDERS, self.buyers),
                              age=distribution(groups, AGE_LABELS, self.buyers))
        annual = []
        for year in covered_years(coverage):
            values = self.annual.get(year)
            if values is None:
                values = dict(buyers=0, orders=0, unvalued_orders=0, money={c: _Finance() for c in CURRENCIES})
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(year+1, 1, 1, tzinfo=timezone.utc) if year < 9999 else datetime.max.replace(tzinfo=timezone.utc)
            observed = [(max(a, start), min(b, end)) for a, b in coverage if max(a, start) < min(b, end)]
            annual.append(dict(year=f'{year:04d}', coverage=saved_coverage(observed),
                               complete_year=observed == [(start, end)], buyers=values['buyers'], orders=values['orders'],
                               unvalued_orders=values['unvalued_orders'],
                               money={c: values['money'][c].result(histograms.get((year, c), Counter()), annual=True)
                                      for c in CURRENCIES}))
        return dict(policy=policy, buyers=self.buyers, orders=self.orders, positions=self.positions, units=str(self.units),
                    single_buyers=self.frequencies[1], repeat_buyers=repeated, repeat_rate=rate(repeated, self.buyers),
                    mean_orders=self.orders/self.buyers if self.buyers else None,
                    median_orders=histogram_median(self.frequencies), mean_units=mean_money(self.units, self.orders),
                    frequency=frequency, first_date=date.fromordinal(self.first).isoformat() if self.first is not None else None,
                    last_date=date.fromordinal(self.last).isoformat() if self.last is not None else None,
                    first_months=distribution(self.first_months, sorted(self.first_months), self.buyers),
                    last_months=distribution(self.last_months, sorted(self.last_months), self.buyers),
                    intervals=intervals, money=money, quality=quality, demography=demography, annual=annual)


class BuyerAggregates:
    def __init__(self, identity_space, selection: AnalysisFilter, coverage, *, orders_available):
        self.identity_space = identity_space
        self.selection = selection
        self.coverage = clipped_coverage(coverage, selection)
        self.orders_available = orders_available
        self.profiles_available = False
        self.buyers = {}
        self.histograms = {policy: defaultdict(Counter) for policy in POLICIES}
        self.issues = {policy: Counter() for policy in POLICIES}
        self.return_positions = 0

    def add(self, fact):
        if fact.identity_space != self.identity_space:
            raise ValueError('Buyer facts belong to different identity spaces')
        if not self.orders_available or not any(a <= fact.order_datetime_utc < b for a, b in self.coverage):
            raise ValueError('Buyer fact is outside declared observation')
        self.return_positions += fact.return_line_count
        if not fact.extended.has_purchase:
            return
        buyer = self.buyers.setdefault(fact.customer_id, _Buyer())
        buyer.days.append(fact.order_datetime_utc.date().toordinal())
        buyer.strict_flags.append(int(fact.strict.has_purchase))
        for policy, result in ((STRICT_COMPLETED, fact.strict), (EXTENDED_COMPLETED, fact.extended)):
            if not result.has_purchase:
                continue
            if policy == STRICT_COMPLETED:
                if buyer.strict is None:
                    buyer.strict = _Purchases()
                state = buyer.strict
            else:
                state = buyer.extended
            year = fact.order_datetime_utc.year
            state.add(year, result)
            if result.monetary_available:
                self.histograms[policy][year, result.currency][result.amount] += 1
            else:
                self.issues[policy].update(result.monetary_issues)

    def profile(self, customer_id, sex, birthday, as_of):
        buyer = self.buyers.get(customer_id)
        if buyer is None or buyer.profile is not None:
            return
        try:
            born = birth_date({'birthDate': birthday})
        except AdapterError:
            born = None
        buyer.profile = (GENDERS[0] if sex == 'male' else GENDERS[1] if sex == 'female' else GENDERS[2],
                         calendar_age(born, as_of))

    def result(self, as_of, check=lambda: None):
        available = self.orders_available and bool(self.coverage)
        summaries = {policy: _Summary() for policy in POLICIES}
        if available:
            for buyer in self.buyers.values():
                check()
                ordered = sorted(zip(buyer.days, buyer.strict_flags))
                summaries[EXTENDED_COMPLETED].add(buyer.extended, [day for day, _ in ordered], buyer.profile, check)
                if buyer.strict is not None:
                    summaries[STRICT_COMPLETED].add(buyer.strict, [day for day, flag in ordered if flag], buyer.profile, check)
        check()
        complete = bool(self.coverage) and (self.coverage == [self.selection.interval] if self.selection.interval else len(self.coverage) == 1)
        return dict(period=dict(start_date=self.selection.start_date, end_date=self.selection.end_date),
                    coverage=saved_coverage(self.coverage), coverage_complete=complete,
                    orders_available=self.orders_available, profiles_available=self.profiles_available,
                    as_of=as_of.isoformat(), return_positions=self.return_positions if available else None,
                    policies={p: summaries[p].result(p, self.histograms[p], self.issues[p], self.profiles_available, self.coverage)
                              if available else None for p in POLICIES})
