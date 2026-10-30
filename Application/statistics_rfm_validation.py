"""RFM snapshot context and closed aggregate validation; never read sources."""

from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import math

from Application.order_statistics import CURRENCIES, histogram_median
from Application.statistics_buyer_validation import _keys, _require, _count, _number, _money, _ratio, _rows, _close
from Application.statistics_period import shared_intervals
from Application.statistics_rfm_core import WINDOWS, POLICIES, SEGMENTS, rfm_windows, empirical_thresholds, rfm_score


def rfm_cutoff(coverage, selected_end, calculated_at):
    if selected_end is not None:
        return date.fromisoformat(selected_end)
    limit = datetime.combine(calculated_at.astimezone(timezone.utc).date()+timedelta(days=1), time(), timezone.utc)
    ends = [min(end, limit) for start, end in coverage if start < min(end, limit)]
    return (max(ends)-timedelta(microseconds=1)).date() if ends else None


def _histogram(value, *, money=False):
    _keys(value, 'buyers distribution mean median')
    n = _count(value['buyers'])
    _rows(value['distribution'], 2)
    histogram = Counter()
    for raw, count in value['distribution']:
        v = _money(raw) if money else _count(raw)
        _require(_count(count) > 0 and v not in histogram)
        histogram[v] = count
    _require(list(histogram) == sorted(histogram) and sum(histogram.values()) == n)
    total = sum((v*c for v, c in histogram.items()), Decimal(0) if money else 0)
    _ratio(value['mean'], total, n, decimal=money)
    median = histogram_median(histogram)
    if median is None:
        _require(value['median'] is None)
    else:
        _require((_money(value['median']) if money else _number(value['median'])) == median)
    return histogram


def _slice(value, complete, *, currency=False):
    fields = 'buyers active_buyers without_observed_orders confirmed_inactive_buyers observed_orders recency frequency reliable_recency_buyers'
    if currency:
        fields += (' known_amount valued_orders monetary incomplete_money_buyers unavailable_monetary_buyers '
                   'thresholds score_counts segments')
    _keys(value, fields)
    n, active, inactive, orders, reliable = (_count(value[k]) for k in
        ('buyers', 'active_buyers', 'without_observed_orders', 'observed_orders', 'reliable_recency_buyers'))
    _require(active+inactive == n and active <= orders and reliable <= n)
    if complete:
        _require(_count(value['confirmed_inactive_buyers']) == inactive)
    else:
        _require(value['confirmed_inactive_buyers'] is None)
    r, f = _histogram(value['recency']), _histogram(value['frequency'])
    _require(sum(r.values()) == sum(f.values()) == n and f[0] == inactive)
    _require(sum(v*c for v,c in f.items()) == orders)
    if not currency:
        return
    amount = _money(value['known_amount'])
    _require(_count(value['valued_orders']) <= orders and (value['valued_orders'] > 0 or amount == 0))
    m = _histogram(value['monetary'], money=True)
    mn = sum(m.values())
    _require(_count(value['unavailable_monetary_buyers'])+mn == n)
    _require(_count(value['incomplete_money_buyers']) <= n-mn)
    _require(sum((v*c for v,c in m.items()), Decimal(0)) <= amount)
    _require(complete or mn == 0)
    _keys(value['thresholds'], ('R', 'F', 'M'))
    for key, histogram, population in (('R', r, reliable), ('F', f, n), ('M', m, mn)):
        threshold = value['thresholds'][key]
        _keys(threshold, 'values buyers method')
        expected_n = population if complete else 0
        _require(_count(threshold['buyers']) == expected_n)
        _require(threshold['method'] == 'NEAREST_RANK_20_40_60_80_STRICT_GREATER')
        cuts = threshold['values']
        _require(isinstance(cuts, (list, tuple)) and len(cuts) == (4 if expected_n else 0))
        cuts = [_money(v) if key == 'M' else _count(v) for v in cuts]
        _require(cuts == sorted(cuts) and all(v in histogram for v in cuts))
        # Reuse the core's pure empirical method, never a second scoring rule.
        if expected_n == sum(histogram.values()):
            _require(tuple(cuts) == empirical_thresholds(histogram))
    _rows(value['score_counts'], 4)
    seen = set()
    score_histograms = [Counter() for _ in range(3)]
    missing = total = 0
    for r_score, f_score, m_score, count in value['score_counts']:
        key = (r_score, f_score, m_score)
        _require(key not in seen and _count(count) > 0)
        seen.add(key)
        _require(all(v is None or type(v) is int and 1 <= v <= 5 for v in key))
        if not complete:
            _require(all(v is None for v in key))
        for i, score in enumerate(key):
            score_histograms[i][score] += count
        total += count
        missing += count if None in key else 0
    _require(total == n)
    if complete:
        _require(score_histograms[0][None] == n-reliable)
    _require(score_histograms[2][None] == n-mn)
    if complete:
        for key, histogram, scores in (('F', f, score_histograms[1]), ('M', m, score_histograms[2])):
            cuts = value['thresholds'][key]['values']
            cuts = tuple(_money(v) for v in cuts) if key == 'M' else cuts
            expected = Counter()
            for v, count in histogram.items():
                expected[rfm_score(v, cuts)] += count
            expected[None] += n-sum(histogram.values())
            _require(scores == expected)
        if reliable == n:
            expected = Counter()
            for v, count in r.items():
                expected[rfm_score(v, value['thresholds']['R']['values'], recency=True)] += count
            _require(score_histograms[0] == expected)
    _rows(value['segments'], 3)
    _require([row[0] for row in value['segments']] == list(SEGMENTS))
    _require(sum(_count(row[1]) for row in value['segments']) == n)
    for _, count, share in value['segments']:
        _ratio(share, count, n, percent=True)
    _require(value['segments'][-1][1] == missing)


def _signed(value):
    _require(isinstance(value, str) and value == value.strip())
    number = Decimal(value)
    _require(number.is_finite())
    return number


def validate_rfm_statistics(value, *, coverage, selected_end, calculated_at):
    if value is None:
        return
    _keys(value, 'as_of_date windows policies')
    parsed = [(a.isoformat() if isinstance(a, datetime) else a, b.isoformat() if isinstance(b, datetime) else b)
              for a, b in coverage]
    coverage = shared_intervals([], parsed)
    cutoff = rfm_cutoff(coverage, selected_end, calculated_at)
    _require(cutoff is not None and value['as_of_date'] == cutoff.isoformat())
    _keys(value['windows'], WINDOWS)
    complete = {}
    for name, (start, end) in rfm_windows(cutoff).items():
        meta = value['windows'][name]
        _keys(meta, 'start end available status reason')
        available = [[max(a,start).isoformat(), min(b,end).isoformat()] for a,b in coverage if max(a,start)<min(b,end)]
        status = 'FULL' if available == [[start.isoformat(),end.isoformat()]] else 'PARTIAL' if available else 'NONE'
        _require(meta == dict(start=start.isoformat(), end=end.isoformat(), available=available, status=status,
                              reason=None if status == 'FULL' else 'COVERAGE_GAP_OR_TRUNCATION' if available else 'NO_ORDERS_COVERAGE'))
        complete[name] = status == 'FULL'
    _keys(value['policies'], POLICIES)
    for policy in POLICIES:
        p = value['policies'][policy]
        _keys(p, 'base_buyers dual_currency_buyers unassigned_currency_buyers windows comparison')
        n = _count(p['base_buyers'])
        dual, unassigned = _count(p['dual_currency_buyers']), _count(p['unassigned_currency_buyers'])
        _require(dual+unassigned <= n)
        _keys(p['windows'], WINDOWS)
        for name in WINDOWS:
            w = p['windows'][name]
            _keys(w, 'behavior currencies quality')
            _slice(w['behavior'], complete[name])
            _require(w['behavior']['buyers'] == n)
            _keys(w['currencies'], CURRENCIES)
            for currency in CURRENCIES:
                c = w['currencies'][currency]
                _slice(c, complete[name], currency=True)
                _require(c['buyers'] <= n and c['observed_orders'] <= w['behavior']['observed_orders'])
                _require(c['buyers'] == p['windows']['LAST_48M']['currencies'][currency]['buyers'])
            _require(sum(c['observed_orders'] for c in w['currencies'].values()) <= w['behavior']['observed_orders'])
            _require(sum(c['buyers'] for c in w['currencies'].values()) == n+dual-unassigned)
            q = w['quality']
            _keys(q, 'unvalued_orders incomplete_money_buyers UNKNOWN_CURRENCY MIXED_CURRENCY INVALID_PRICE')
            for v in q.values():
                _count(v)
            orders = w['behavior']['observed_orders']
            _require(q['incomplete_money_buyers'] <= min(n, q['unvalued_orders']) and q['unvalued_orders'] <= orders)
            _require(max(q[k] for k in ('UNKNOWN_CURRENCY','MIXED_CURRENCY','INVALID_PRICE')) <= q['unvalued_orders'] <=
                     sum(q[k] for k in ('UNKNOWN_CURRENCY','MIXED_CURRENCY','INVALID_PRICE')))
            _require(sum(c['valued_orders'] for c in w['currencies'].values())+q['unvalued_orders'] == orders)
        for group in ('behavior', *CURRENCIES):
            rows = [p['windows'][name]['behavior'] if group=='behavior' else p['windows'][name]['currencies'][group]
                    for name in WINDOWS]
            _require(rows[0]['observed_orders']+rows[3]['observed_orders'] == rows[1]['observed_orders'] <= rows[2]['observed_orders'])
            if group!='behavior':
                _require(_money(rows[0]['known_amount'])+_money(rows[3]['known_amount']) == _money(rows[1]['known_amount']) <=
                         _money(rows[2]['known_amount']))
        comparison = p['comparison']
        _keys(comparison, 'coverage_reliable observed_groups last_orders previous_orders order_change order_change_percent '
                         'unreliable_monetary_buyers money')
        reliable = complete['LAST_12M'] and complete['PREVIOUS_12M']
        _require(type(comparison['coverage_reliable']) is bool and comparison['coverage_reliable'] == reliable)
        groups = comparison['observed_groups']
        _keys(groups, ('BOTH','LAST_ONLY','PREVIOUS_ONLY','NEITHER'))
        _require(sum(_count(v) for v in groups.values()) == n)
        last, previous = (p['windows'][k]['behavior'] for k in ('LAST_12M','PREVIOUS_12M'))
        _require(groups['BOTH']+groups['LAST_ONLY'] == last['active_buyers'])
        _require(groups['BOTH']+groups['PREVIOUS_ONLY'] == previous['active_buyers'])
        _require(_count(comparison['last_orders']) == last['observed_orders'] and
                 _count(comparison['previous_orders']) == previous['observed_orders'])
        delta = last['observed_orders']-previous['observed_orders']
        _require(type(comparison['order_change']) is int and comparison['order_change'] == delta)
        percent = comparison['order_change_percent']
        _require(percent is None if not previous['observed_orders'] else
                 type(percent) in (int,float) and math.isfinite(percent) and _close(percent,100*delta/previous['observed_orders']))
        _require(_count(comparison['unreliable_monetary_buyers']) <= n)
        if not reliable:
            _require(comparison['unreliable_monetary_buyers'] == n)
        _keys(comparison['money'], CURRENCIES)
        for currency in CURRENCIES:
            m = comparison['money'][currency]
            _keys(m, 'last_known_amount previous_known_amount known_amount_change known_amount_change_percent unreliable_buyers')
            current_amount, old_amount = _money(m['last_known_amount']), _money(m['previous_known_amount'])
            for key, amount in (('LAST_12M',current_amount),('PREVIOUS_12M',old_amount)):
                _require(amount == _money(p['windows'][key]['currencies'][currency]['known_amount']))
            _require(_signed(m['known_amount_change']) == current_amount-old_amount)
            _require(m['known_amount_change_percent'] is None if not old_amount else
                     _close(_signed(m['known_amount_change_percent']),100*(current_amount-old_amount)/old_amount))
            _require(_count(m['unreliable_buyers']) <= min(n, comparison['unreliable_monetary_buyers']))
    strict, extended = (value['policies'][p] for p in POLICIES)
    _require(strict['base_buyers'] <= extended['base_buyers'])
    for name in WINDOWS:
        _require(strict['windows'][name]['behavior']['observed_orders'] <= extended['windows'][name]['behavior']['observed_orders'])
