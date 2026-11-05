"""Streaming first-observed purchase cohorts over prepared unique order facts.

No source adaptation, deduplication, identity resolution or I/O. Coverage certifies
complete observation, not the absence of individually observed events outside it.
"""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import re

from Application.order_statistics import histogram_median
from Application.statistics_cohorts_money import BuyerMoney, MonthMoney, SecondMoney
from Application.statistics_cohorts_channels import ChannelCounts, channel_value
from Application.statistics_cohorts_categories import CategoryCounts, category_codes as _category_codes, category_reference
from Application.statistics_period import shared_intervals
from Application.statistics_purchase_facts import PurchaseOrderFact
from Application.statistics_purchase_policy import STRICT_COMPLETED, EXTENDED_COMPLETED


POLICIES = (STRICT_COMPLETED, EXTENDED_COMPLETED)
HORIZONS = (30, 60, 90, 180, 365)
MONTH_OFFSETS = tuple(range(13))


def _utc(day):
    return datetime.combine(day, time.min, timezone.utc)


def _month(value):
    return value.year * 12 + value.month - 1


def _month_date(index):
    year, month = divmod(index, 12)
    return date(year, month + 1, 1)


def _month_key(index):
    day = _month_date(index)
    return f'{day.year:04d}-{day.month:02d}'


def _month_bound(value):
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r'[0-9]{4}-[0-9]{2}', value) is None:
        raise ValueError('Cohort bounds require YYYY-MM')
    try:
        return _month(date.fromisoformat(value + '-01'))
    except ValueError:
        raise ValueError('Invalid cohort month') from None


def _identity(value):
    return type(value) is int and value > 0 or isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _saved(intervals):
    return [[a.isoformat(), b.isoformat()] for a, b in intervals]


def _interval_summary(histogram):
    count = sum(histogram.values())
    return dict(buyers=count, distribution=[[day, histogram[day]] for day in sorted(histogram)],
                mean=sum(day * n for day, n in histogram.items()) / count if count else None,
                median=histogram_median(histogram))


@dataclass(slots=True)
class _Buyer:
    first: tuple | None = None
    second: tuple | None = None
    months: int = 0
    money: BuyerMoney | None = None
    first_channel: tuple | None = None
    second_channel: tuple | None = None
    first_categories: frozenset | None = None
    second_categories: frozenset | None = None

    def add(self, key, purchase=None, channel=None, categories=None):
        payload = None
        if purchase is not None:
            if self.money is None:
                self.money = BuyerMoney()
            payload = self.money.record(purchase)
        month = _month(key[0])
        origin = _month(self.first[0]) if self.first is not None else month
        if month < origin:
            self.months <<= origin - month
            origin = month
        self.months |= 1 << (month - origin)
        # The caller supplies each unique order once. Only the two minima are
        # retained, including deterministic ordering of simultaneous orders.
        if self.first is None or key < self.first:
            self.second, self.first = self.first, key
            self.second_channel, self.first_channel = self.first_channel, channel
            self.second_categories, self.first_categories = self.first_categories, categories
            if self.money is not None:
                self.money.second, self.money.first = self.money.first, payload
        elif self.second is None or key < self.second:
            self.second = key
            self.second_channel = channel
            self.second_categories = categories
            if self.money is not None:
                self.money.second = payload


@dataclass(slots=True)
class _Counts:
    eligible: int = 0
    censored: int = 0
    coverage_incomplete: int = 0
    censored_and_incomplete: int = 0
    observed: int = 0
    qualified: int = 0

    def add(self, censored, incomplete, observed, count=1):
        eligible = not censored and not incomplete
        self.eligible += count * eligible
        self.censored += count * censored
        self.coverage_incomplete += count * incomplete
        self.censored_and_incomplete += count * (censored and incomplete)
        self.observed += count * observed
        self.qualified += count * (eligible and observed)

    def merge(self, other):
        for name in self.__slots__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def result(self, size, *, monthly=False):
        numerator, observed, rate = (('active', 'observed_active', 'retention_rate') if monthly else
                                     ('repeated', 'observed_repeated', 'repeat_rate'))
        reasons = []
        if self.censored:
            reasons.append('RIGHT_CENSORED')
        if self.coverage_incomplete:
            reasons.append('ORDERS_COVERAGE_INCOMPLETE')
        status = ('NO_BUYERS' if not size else 'FULL' if self.eligible == size else
                  'PARTIAL' if self.eligible else 'UNAVAILABLE')
        return dict(cohort_size=size, eligible=self.eligible, censored=self.censored,
                    coverage_incomplete=self.coverage_incomplete,
                    censored_and_incomplete=self.censored_and_incomplete,
                    **{numerator: self.qualified, observed: self.observed,
                       observed + '_ineligible': self.observed - self.qualified,
                       rate: 100 * self.qualified / self.eligible if self.eligible else None},
                    confirmed_without_event=self.eligible - self.qualified, status=status, reasons=reasons)


@dataclass(slots=True)
class _Cohort:
    buyers: int = 0
    second_days: Counter = field(default_factory=Counter)
    second_interval_incomplete: int = 0
    horizons: list = field(default_factory=lambda: [_Counts() for _ in HORIZONS])
    active: list = field(default_factory=lambda: [0] * 13)
    money: MonthMoney | None = None
    channels: ChannelCounts | None = None


class CohortAggregates:
    def __init__(self, as_of_date, coverage, *, identity_space, policies=POLICIES,
                 cohort_start=None, cohort_end=None, check=None, category_names=None, category_paths=None):
        if type(as_of_date) is not date:
            raise ValueError('Cohorts require an explicit calendar as_of_date')
        try:
            self.end = _utc(as_of_date + timedelta(days=1))
            # Ensure even the latest buyer has representable inclusive H365
            # and M12 endpoints. No artificial window truncation at date.max.
            as_of_date + timedelta(days=366)
            _month_date(_month(as_of_date) + 13)
        except (OverflowError, ValueError):
            raise ValueError('Cohort date cannot represent follow-up endpoints') from None
        if not isinstance(identity_space, str) or not identity_space.strip() or identity_space != identity_space.strip():
            raise ValueError('Invalid cohort identity space')
        if isinstance(policies, str):
            raise ValueError('Cohort policies must be a nonempty collection')
        try:
            requested = tuple(policies)
        except TypeError:
            raise ValueError('Invalid cohort policies') from None
        if not requested or any(p not in POLICIES for p in requested) or len(set(requested)) != len(requested):
            raise ValueError('Invalid cohort policies')
        self.cohort_start, self.cohort_end = _month_bound(cohort_start), _month_bound(cohort_end)
        if self.cohort_start is not None and self.cohort_end is not None and self.cohort_start > self.cohort_end:
            raise ValueError('Reversed cohort display bounds')
        if check is not None and not callable(check):
            raise ValueError('Invalid cancellation check')
        if isinstance(coverage, (str, bytes, dict)):
            raise ValueError('Invalid Orders coverage intervals')
        try:
            intervals = [(a.isoformat() if isinstance(a, datetime) else a,
                          b.isoformat() if isinstance(b, datetime) else b) for a, b in (() if coverage is None else coverage)]
            self.source_coverage = shared_intervals([], intervals)
        except (TypeError, ValueError, AttributeError):
            raise ValueError('Invalid Orders coverage intervals') from None
        self.coverage = tuple((a, min(b, self.end)) for a, b in self.source_coverage if a < self.end)
        self._starts = tuple(a for a, _ in self.coverage)
        self.coverage_metadata_available = coverage is not None
        self.as_of_date = as_of_date
        self.identity_space = identity_space
        self.policies = tuple(p for p in POLICIES if p in requested)
        self.check = check if check is not None else lambda: None
        self._buyers = {p: {} for p in self.policies}
        self._future_facts = self._uncovered_purchase_facts = 0
        self._categories = category_reference(category_names,category_paths) if category_names is not None else None

    def _covered(self, start, end):
        index = bisect_right(self._starts, start) - 1
        return index >= 0 and self.coverage[index][1] >= end

    def _eligibility(self, start, end):
        observed_end = min(end, self.end)
        incomplete = start < observed_end and not self._covered(start, observed_end)
        return end > self.end, incomplete

    def add(self, fact: PurchaseOrderFact, *, channel_external_id=None, channel_name=None, order_sort_rank=None, category_codes=None):
        self.check()
        if not isinstance(fact, PurchaseOrderFact) or fact.identity_space != self.identity_space:
            raise ValueError('Cohorts require facts from one explicit identity space')
        occurred = fact.order_datetime_utc
        if (not isinstance(occurred, datetime) or occurred.tzinfo is None or occurred.utcoffset() != timedelta(0)
                or not _identity(fact.customer_id) or not _identity(fact.order_id)):
            raise ValueError('Cohorts require validated UTC order facts and identities')
        if occurred >= self.end:
            self._future_facts += 1
            return
        accepted = [(policy, purchase) for policy, purchase in
                    ((STRICT_COMPLETED, fact.strict), (EXTENDED_COMPLETED, fact.extended))
                    if policy in self._buyers and purchase.has_purchase]
        if not accepted:
            return
        if order_sort_rank is not None and (type(order_sort_rank) is not int or order_sort_rank <= 0):
            raise ValueError('Invalid cohort order sort rank')
        channel = channel_value(channel_external_id, channel_name) if any(
            policy == STRICT_COMPLETED for policy, _ in accepted) else None
        categories = _category_codes(category_codes,self._categories) if self._categories is not None and any(
            policy == STRICT_COMPLETED for policy, _ in accepted) else None
        index = bisect_right(self._starts, occurred) - 1
        self._uncovered_purchase_facts += int(index < 0 or occurred >= self.coverage[index][1])
        # INDEXED supplies the source-ID lexical rank, never SID insertion order.
        # REFERENCE and direct prepared-fact callers retain the original ID key.
        order_key = fact.order_id if order_sort_rank is None else order_sort_rank
        key = occurred, 0 if type(order_key) is int else 1, order_key
        # No check halfway through mutation: cancellation cannot leave policies
        # disagreeing about whether this fact was consumed.
        for policy, purchase in accepted:
            buyers = self._buyers[policy]
            if fact.customer_id not in buyers:
                buyers[fact.customer_id] = _Buyer()
            buyers[fact.customer_id].add(key, purchase if policy == STRICT_COMPLETED else None,
                                         channel if policy == STRICT_COMPLETED else None,
                                         categories if policy == STRICT_COMPLETED else None)

    def _collect(self, buyers, monetary=False, category_counts=None):
        cohorts = {}
        for buyer in buyers.values():
            self.check()
            first = buyer.first[0].date()
            month = _month(first)
            if (self.cohort_start is not None and month < self.cohort_start or
                    self.cohort_end is not None and month > self.cohort_end):
                continue
            if month not in cohorts:
                cohorts[month] = _Cohort()
            cohort = cohorts[month]
            cohort.buyers += 1
            if category_counts is not None:
                category_counts.add(buyer.first_categories,buyer.second_categories,self.check)
            if monetary:
                if cohort.money is None:
                    cohort.money = MonthMoney()
                    cohort.channels = ChannelCounts()
                cohort.money.add(buyer.money)
                cohort.channels.add(buyer.first_channel, buyer.second_channel)
            days = (buyer.second[0].date() - first).days if buyer.second is not None else None
            if days is not None:
                cohort.second_days[days] += 1
                cohort.second_interval_incomplete += int(not self._covered(
                    _utc(first), _utc(buyer.second[0].date() + timedelta(days=1))))
            for i, horizon in enumerate(HORIZONS):
                censored, incomplete = self._eligibility(_utc(first), _utc(first + timedelta(days=horizon + 1)))
                cohort.horizons[i].add(censored, incomplete, days is not None and days <= horizon)
                if monetary and not censored and not incomplete and days is not None and days <= horizon:
                    cohort.money.horizons[i].record(buyer.money.second)
            for offset in MONTH_OFFSETS:
                cohort.active[offset] += (buyer.months >> offset) & 1
        return cohorts

    def _policy_result(self, buyers, monetary=False):
        category_counts = CategoryCounts() if monetary and self._categories is not None else None
        cohorts = self._collect(buyers, monetary, category_counts)
        money_rows = {}
        channels = ChannelCounts() if monetary else None
        money_horizons = [SecondMoney() for _ in HORIZONS] if monetary else None
        rows, histogram = {}, Counter()
        horizons = [_Counts() for _ in HORIZONS]
        retention = [_Counts() for _ in MONTH_OFFSETS[1:]]
        size = interval_incomplete = 0
        for month, cohort in sorted(cohorts.items()):
            self.check()
            size += cohort.buyers
            if monetary:
                money_rows[_month_key(month)] = cohort.money.result(cohort.buyers)
                channels.merge(cohort.channels, self.check)
                for total, value in zip(money_horizons, cohort.money.horizons):
                    total.merge(value)
            histogram.update(cohort.second_days)
            interval_incomplete += cohort.second_interval_incomplete
            cells = [dict(month_offset=0, month=_month_key(month), kind='COHORT_SIZE', buyers=cohort.buyers)]
            for offset in MONTH_OFFSETS[1:]:
                self.check()
                start, end = (_utc(_month_date(month + i)) for i in (offset, offset + 1))
                censored, incomplete = self._eligibility(start, end)
                count = _Counts()
                count.add(censored, incomplete, False, cohort.buyers)
                count.observed = cohort.active[offset]
                count.qualified = count.observed if count.eligible else 0
                retention[offset - 1].merge(count)
                cells.append(dict(month_offset=offset, month=_month_key(month + offset), kind='MONTHLY_ACTIVITY',
                                  **count.result(cohort.buyers, monthly=True)))
            for total, count in zip(horizons, cohort.horizons):
                total.merge(count)
            rows[_month_key(month)] = dict(cohort_size=cohort.buyers,
                observed_repeat_buyers=sum(cohort.second_days.values()),
                second_purchase_days=dict(**_interval_summary(cohort.second_days),
                                          coverage_incomplete_buyers=cohort.second_interval_incomplete),
                horizons={str(h): count.result(cohort.buyers) for h, count in zip(HORIZONS, cohort.horizons)},
                retention=cells)
        result = dict(base_buyers=len(buyers), displayed_buyers=size, hidden_buyers=len(buyers) - size, cohorts=rows,
            summary=dict(cohort_count=len(rows), cohort_size=size, observed_repeat_buyers=sum(histogram.values()),
                second_purchase_days=dict(**_interval_summary(histogram), coverage_incomplete_buyers=interval_incomplete),
                horizons={str(h): count.result(size) for h, count in zip(HORIZONS, horizons)},
                retention=[dict(month_offset=0, kind='COHORT_SIZE', buyers=size)] + [
                    dict(month_offset=offset, kind='MONTHLY_ACTIVITY', **count.result(size, monthly=True))
                    for offset, count in zip(MONTH_OFFSETS[1:], retention)]))
        if category_counts is not None:
            result['purchase_categories'] = category_counts.result(self._categories,self.check)
        if monetary:
            result['purchase_channels'] = channels.result(self.check)
            result['monetary'] = dict(cohorts=money_rows,horizons={
                str(h): value.result(count.eligible,count.qualified)
                for h,value,count in zip(HORIZONS,money_horizons,horizons)})
        return result

    def result(self):
        """Fresh deterministic aggregates only; cancelled results are never published."""
        self.check()
        policies = {policy: self._policy_result(buyers, policy == STRICT_COMPLETED) for policy, buyers in self._buyers.items()}
        self.check()
        return dict(as_of_date=self.as_of_date.isoformat(), cutoff_exclusive=self.end.isoformat(),
            cohort_definition='FIRST_OBSERVED_PURCHASE_UTC_MONTH',
            source_coverage=_saved(self.source_coverage), effective_coverage=_saved(self.coverage),
            cohort_bounds=dict(start=_month_key(self.cohort_start) if self.cohort_start is not None else None,
                               end=_month_key(self.cohort_end) if self.cohort_end is not None else None),
            horizons=list(HORIZONS), month_offsets=list(MONTH_OFFSETS), policies=policies,
            quality=dict(coverage_metadata_available=self.coverage_metadata_available,
                lifetime_first_purchase_proven=False, second_purchase_is_observed=True,
                gaps_can_hide_earlier_orders=True, facts_after_cutoff=self._future_facts,
                observed_purchase_facts_outside_coverage=self._uncovered_purchase_facts))
