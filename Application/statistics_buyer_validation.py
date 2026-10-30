"""Strict aggregate-only buyer DTO validation; no source reads or repair."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, DecimalException
import math
import re

from Application.analysis_filter import AnalysisFilter
from Application.customer_demographics import AGE_LABELS
from Application.order_statistics import CURRENCIES
from Application.statistics_buyer_aggregates import (
    POLICIES, GENDERS, FREQUENCY_LABELS, INTERVAL_LABELS, ANNUAL_MONEY_FIELDS,
    clipped_coverage, saved_coverage, covered_years,
)
from Application.statistics_period import shared_intervals


def _require(value):
    if not value:
        raise ValueError('Invalid buyer statistics')


def _keys(value, fields):
    _require(isinstance(value, dict) and set(value) == set(fields.split() if isinstance(fields, str) else fields))


def _count(value):
    _require(type(value) is int and value >= 0)
    return value


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0)
    return value


def _money(value):
    _require(isinstance(value, str) and bool(value) and value == value.strip())
    try:
        parsed = Decimal(value)
    except DecimalException:
        raise ValueError('Invalid buyer monetary value') from None
    _require(parsed.is_finite() and parsed >= 0)
    return parsed


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _ratio(value, numerator, denominator, *, percent=False, decimal=False):
    if not denominator:
        _require(value is None)
        return
    expected = numerator / denominator * (100 if percent else 1)
    actual = _money(value) if decimal else _number(value)
    if decimal:
        _require(abs(actual - expected) <= max(Decimal('1e-20'), abs(expected) * Decimal('1e-20')))
    else:
        _require(_close(actual, expected) and (not percent or actual <= 100))


def _median(value, n, total, *, decimal=False):
    if not n:
        _require(value is None)
    else:
        actual = _money(value) if decimal else _number(value)
        _require(actual <= total and (n != 1 or actual == total))


def _date(value):
    _require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value)
    return date.fromisoformat(value)


def _rows(value, width):
    _require(isinstance(value, (tuple, list)))
    _require(all(isinstance(row, (tuple, list)) and len(row) == width for row in value))


def _distribution(rows, labels, total):
    _rows(rows, 3)
    _require([r[0] for r in rows] == list(labels))
    _require(sum(_count(r[1]) for r in rows) == total)
    for _, n, percentage in rows:
        _ratio(percentage, n, total, percent=True)


def _median_bounds(value, rows, n, bounds):
    if not n:
        return
    ranks = ((n-1)//2, n//2)
    lower, upper, seen = [], [], 0
    for row, (low, high) in zip(rows, bounds):
        for rank in ranks:
            if seen <= rank < seen + row[1]:
                lower.append(low)
                upper.append(high)
        seen += row[1]
    _require(sum(lower)/2 <= value <= sum(upper)/2)


def _financial(value, buyers, repeat_buyers, *, annual=False):
    full_fields = set(ANNUAL_MONEY_FIELDS) | set('mean_order median_order mean_known_per_buyer median_known_per_buyer '
        'repeat_amount repeat_amount_rate complete_repeat_buyers complete_repeat_rate complete_repeat_amount '
        'complete_repeat_amount_rate quintiles'.split())
    _keys(value, ANNUAL_MONEY_FIELDS if annual else full_fields)
    n, known, complete = (_count(value[k]) for k in ('known_orders', 'known_buyers', 'complete_buyers'))
    _require(complete <= known <= min(n, buyers) and (bool(n) == bool(known)))
    amount, complete_amount = _money(value['amount']), _money(value['complete_amount'])
    _require(complete_amount <= amount)
    if not n:
        _require(amount == 0)
    if not complete:
        _require(complete_amount == 0)
    _ratio(value['mean_complete_per_buyer'], complete_amount, complete, decimal=True)
    _median(value['median_complete_per_buyer'], complete, complete_amount, decimal=True)
    if annual:
        return
    _ratio(value['mean_order'], amount, n, decimal=True)
    _median(value['median_order'], n, amount, decimal=True)
    _ratio(value['mean_known_per_buyer'], amount, known, decimal=True)
    _median(value['median_known_per_buyer'], known, amount, decimal=True)
    repeat_amount = _money(value['repeat_amount'])
    complete_repeat_amount = _money(value['complete_repeat_amount'])
    complete_repeats = _count(value['complete_repeat_buyers'])
    _require(complete_repeats <= min(complete, repeat_buyers))
    _require(complete_repeat_amount <= min(repeat_amount, complete_amount) and repeat_amount <= amount)
    _ratio(value['repeat_amount_rate'], repeat_amount, amount, percent=True)
    _ratio(value['complete_repeat_amount_rate'], complete_repeat_amount, complete_amount, percent=True)
    _ratio(value['complete_repeat_rate'], complete_repeats, complete, percent=True)
    groups = value['quintiles']
    _rows(groups, 9)
    _require(len(groups) <= 5 and bool(groups) == bool(complete))
    previous = None
    counts = repeats = 0
    amounts = repeat_amounts = Decimal(0)
    median_rows, bounds = [], []
    for low, high, count, percentage, subtotal, repeated, repeat_rate, repeated_amount, amount_rate in groups:
        low, high = _money(low), _money(high)
        _require(low <= high and (previous is None or previous < low))
        previous = high
        _require(_count(count) > 0 and _count(repeated) <= count)
        subtotal, repeated_amount = _money(subtotal), _money(repeated_amount)
        _require(low * count <= subtotal <= high * count and low * repeated <= repeated_amount <= high * repeated)
        _require(repeated_amount <= subtotal)
        _ratio(percentage, count, complete, percent=True)
        _ratio(repeat_rate, repeated, count, percent=True)
        _ratio(amount_rate, repeated_amount, subtotal, percent=True)
        counts += count
        repeats += repeated
        amounts += subtotal
        repeat_amounts += repeated_amount
        median_rows.append((None, count))
        bounds.append((low, high))
    _require((counts, repeats, amounts, repeat_amounts) == (complete, complete_repeats, complete_amount, complete_repeat_amount))
    if complete:
        _median_bounds(_money(value['median_complete_per_buyer']), median_rows, complete, bounds)


def _demography(value, buyers, available):
    if not available:
        _require(value is None)
        return
    _keys(value, 'profiled_buyers missing_profiles unknown_gender unknown_age mean_age median_age gender age')
    profiled, missing, unknown_gender, unknown_age = (_count(value[k]) for k in
        ('profiled_buyers', 'missing_profiles', 'unknown_gender', 'unknown_age'))
    _require(profiled + missing == buyers and missing <= unknown_gender <= buyers and missing <= unknown_age <= buyers)
    _distribution(value['gender'], GENDERS, buyers)
    _distribution(value['age'], AGE_LABELS, buyers)
    _require(value['gender'][-1][1] == unknown_gender and value['age'][-1][1] == unknown_age)
    if buyers == unknown_age:
        _require(value['mean_age'] is value['median_age'] is None)
    else:
        _require(_number(value['mean_age']) <= 9998 and _number(value['median_age']) <= 9998)


def _policy(value, policy, profiles, coverage):
    _keys(value, 'policy buyers orders positions units single_buyers repeat_buyers repeat_rate mean_orders median_orders '
          'mean_units frequency first_date last_date first_months last_months intervals money quality demography annual')
    _require(value['policy'] == policy)
    buyers, orders, positions, single, repeated = (_count(value[k]) for k in
                                                  ('buyers', 'orders', 'positions', 'single_buyers', 'repeat_buyers'))
    units = _money(value['units'])
    _require(units == units.to_integral_value() and buyers <= orders <= positions <= units)
    _require(bool(buyers) == bool(orders) and single + repeated == buyers)
    _ratio(value['repeat_rate'], repeated, buyers, percent=True)
    _ratio(value['mean_orders'], orders, buyers)
    _median(value['median_orders'], buyers, orders)
    _ratio(value['mean_units'], units, orders, decimal=True)
    rows = value['frequency']
    _rows(rows, 4)
    _require([r[0] for r in rows] == list(FREQUENCY_LABELS))
    _require(sum(_count(r[1]) for r in rows) == buyers and sum(_count(r[3]) for r in rows) == orders)
    bounds = [(1, 1), (2, 2), (3, 5), (6, 10), (11, max(11, orders))]
    for (_, n, percentage, segment_orders), (lo, hi) in zip(rows, bounds):
        _require(lo * n <= segment_orders <= hi * n)
        _ratio(percentage, n, buyers, percent=True)
    _require(rows[0][1] == single)
    if buyers:
        _median_bounds(value['median_orders'], rows, buyers, bounds)
        first, last = _date(value['first_date']), _date(value['last_date'])
        _require(first <= last)
        for day in (first, last):
            _require(any(start.date() <= day <= (end-timedelta(microseconds=1)).date() for start, end in coverage))
    else:
        _require(value['first_date'] is value['last_date'] is None and positions == units == 0)
        first = last = None
    for key in ('first_months', 'last_months'):
        months = value[key]
        _rows(months, 3)
        labels = [r[0] for r in months]
        _require(all(isinstance(label, str) and re.fullmatch(r'\d{4}-\d{2}', label)
                     and date.fromisoformat(label+'-01').strftime('%Y-%m') == label for label in labels))
        _require(labels == sorted(set(labels)))
        _distribution(months, labels, buyers)
        if buyers:
            _require(labels and first.strftime('%Y-%m') <= labels[0] <= labels[-1] <= last.strftime('%Y-%m'))
    if buyers:
        _require(value['first_months'][0][0] == first.strftime('%Y-%m') and value['last_months'][-1][0] == last.strftime('%Y-%m'))
    intervals = value['intervals']
    _keys(intervals, 'buyers count total_days mean_days median_days distribution')
    _require(_count(intervals['buyers']) == repeated and _count(intervals['count']) == orders-buyers)
    total_days = _count(intervals['total_days'])
    n = intervals['count']
    _ratio(intervals['mean_days'], total_days, n)
    _median(intervals['median_days'], n, total_days)
    _distribution(intervals['distribution'], INTERVAL_LABELS, n)
    span = (last-first).days if buyers else 0
    _require(total_days <= n * span)
    for row, (lo, hi) in zip(intervals['distribution'], [(0,30),(31,60),(61,90),(91,180),(181,365),(366,max(366,span))]):
        if row[1]:
            _require(lo <= span)
    if n:
        _median_bounds(intervals['median_days'], intervals['distribution'], n,
                       [(0,30),(31,60),(61,90),(91,180),(181,365),(366,max(366,span))])
    _keys(value['quality'], 'complete_buyers partial_buyers unvalued_buyers unvalued_orders '
          'unknown_currency_orders mixed_currency_orders invalid_price_orders')
    q = value['quality']
    for count in q.values():
        _count(count)
    _require(q['complete_buyers'] + q['partial_buyers'] + q['unvalued_buyers'] == buyers)
    _require(q['unvalued_buyers'] <= q['unvalued_orders'] <= orders)
    reasons = [q[k] for k in ('unknown_currency_orders', 'mixed_currency_orders', 'invalid_price_orders')]
    _require(max(reasons) <= q['unvalued_orders'] <= sum(reasons))
    _keys(value['money'], CURRENCIES)
    for currency in CURRENCIES:
        _financial(value['money'][currency], buyers, repeated)
        _require(value['money'][currency]['complete_buyers'] <= q['complete_buyers'])
    money = value['money']
    _require(sum(money[c]['known_orders'] for c in CURRENCIES) + q['unvalued_orders'] == orders)
    known_buyers = q['complete_buyers'] + q['partial_buyers']
    _require(max(money[c]['known_buyers'] for c in CURRENCIES) <= known_buyers <= sum(money[c]['known_buyers'] for c in CURRENCIES))
    _require(max(money[c]['complete_buyers'] for c in CURRENCIES) <= q['complete_buyers'] <= sum(money[c]['complete_buyers'] for c in CURRENCIES))
    _demography(value['demography'], buyers, profiles)
    annual = value['annual']
    _require(isinstance(annual, (list, tuple)))
    years = []
    for row in annual:
        _keys(row, 'year coverage complete_year buyers orders unvalued_orders money')
        year = row['year']
        _require(isinstance(year, str) and re.fullmatch(r'\d{4}', year) and 1 <= int(year) <= 9999)
        years.append(year)
        start = datetime(int(year), 1, 1, tzinfo=timezone.utc)
        end = datetime(int(year)+1, 1, 1, tzinfo=timezone.utc) if int(year)<9999 else datetime.max.replace(tzinfo=timezone.utc)
        expected = [(max(a,start),min(b,end)) for a,b in coverage if max(a,start)<min(b,end)]
        _require(row['coverage'] == saved_coverage(expected) and expected)
        _require(type(row['complete_year']) is bool and row['complete_year'] == (expected == [(start,end)]))
        _require(_count(row['buyers']) <= buyers and row['buyers'] <= _count(row['orders']) <= orders)
        _require(bool(row['buyers']) == bool(row['orders']))
        _require(_count(row['unvalued_orders']) <= row['orders'])
        _keys(row['money'], CURRENCIES)
        for currency in CURRENCIES:
            _financial(row['money'][currency], row['buyers'], row['buyers'], annual=True)
        _require(sum(row['money'][c]['known_orders'] for c in CURRENCIES) + row['unvalued_orders'] == row['orders'])
    _require(years == [f'{year:04d}' for year in covered_years(coverage)])
    _require(sum(row['orders'] for row in annual) == orders and sum(row['unvalued_orders'] for row in annual) == q['unvalued_orders'])
    if buyers:
        _require(first.strftime('%Y') in years and last.strftime('%Y') in years)
        _require(max(row['buyers'] for row in annual) <= buyers <= sum(row['buyers'] for row in annual))
    for currency in CURRENCIES:
        _require(sum(row['money'][currency]['known_orders'] for row in annual) == money[currency]['known_orders'])
        _require(sum((_money(row['money'][currency]['amount']) for row in annual), Decimal(0)) == _money(money[currency]['amount']))


def validate_buyer_statistics(value, *, coverage, profiles_available, customer_scope=None):
    if value is None:
        return
    _keys(value, 'period coverage coverage_complete orders_available profiles_available as_of return_positions policies')
    _keys(value['period'], 'start_date end_date')
    selection = AnalysisFilter(**value['period'])
    _date(value['as_of'])
    _require(type(value['orders_available']) is bool and value['orders_available'] == bool(coverage))
    _require(type(value['profiles_available']) is bool and value['profiles_available'] == profiles_available)
    observed = clipped_coverage(shared_intervals([], coverage), selection)
    _require(value['coverage'] == saved_coverage(observed))
    complete = bool(observed) and (observed == [selection.interval] if selection.interval else len(observed) == 1)
    _require(type(value['coverage_complete']) is bool and value['coverage_complete'] == complete)
    if customer_scope is not None:
        _require(value['period'] == customer_scope['period'])
    _keys(value['policies'], POLICIES)
    available = bool(observed) and value['orders_available']
    if not available:
        _require(value['return_positions'] is None and all(v is None for v in value['policies'].values()))
        return
    _count(value['return_positions'])
    for policy in POLICIES:
        _policy(value['policies'][policy], policy, value['profiles_available'], observed)
    strict, extended = (value['policies'][p] for p in POLICIES)
    for key in ('buyers', 'orders', 'positions'):
        _require(strict[key] <= extended[key])
    _require(_money(strict['units']) <= _money(extended['units']))
    if customer_scope is not None and customer_scope['orders'] is not None:
        _require(extended['buyers'] <= customer_scope['orders'])  # Scope O counts clients, not orders.
