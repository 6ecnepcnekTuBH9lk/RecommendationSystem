"""Closed aggregate cohort contract; context/consistency checks, never raw I/O."""

from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
import re

from Application.order_statistics import CURRENCIES, histogram_median
from Application.statistics_cohorts_money import _add
from Application.statistics_cohorts_channels import UNKNOWN_CHANNEL, channel_key
from Application.statistics_cohorts_categories import (
    UNKNOWN_CATEGORY, category_key, MAX_ORDER_CATEGORIES, MAX_PAIRS_PER_BUYER, MAX_AGGREGATE_PAIRS,
)
from Application.statistics_period import shared_intervals
from Application.statistics_buyer_validation import _keys, _require, _count, _number, _ratio, _rows, _close, _money
from Application.statistics_cohorts_core import (
    CohortAggregates, HORIZONS, MONTH_OFFSETS, POLICIES, _utc, _month_date,
)
from Application.statistics_rfm_validation import rfm_cutoff as cohort_cutoff


COUNT_FIELDS = ('eligible', 'censored', 'coverage_incomplete', 'censored_and_incomplete', 'confirmed_without_event')


def _month(value):
    _require(isinstance(value, str) and re.fullmatch(r'[0-9]{4}-[0-9]{2}', value) is not None)
    parsed = date.fromisoformat(value + '-01')
    return parsed.year * 12 + parsed.month - 1


def _month_key(index):
    year, month = divmod(index, 12)
    return f'{year:04d}-{month+1:02d}'


def _histogram(value, repeats):
    _keys(value, 'buyers distribution mean median coverage_incomplete_buyers')
    _require(_count(value['buyers']) == repeats and _count(value['coverage_incomplete_buyers']) <= repeats)
    _rows(value['distribution'], 2)
    histogram = Counter()
    for day, count in value['distribution']:
        _require(_count(day) not in histogram and _count(count) > 0)
        histogram[day] = count
    _require(list(histogram) == sorted(histogram) and sum(histogram.values()) == repeats)
    _ratio(value['mean'], sum(d*n for d,n in histogram.items()), repeats)
    median = histogram_median(histogram)
    _require(value['median'] is None if median is None else _close(_number(value['median']), median))
    return histogram


def _counts(value, size, *, monthly=False, extra=()):
    event, observed, rate = (('active','observed_active','retention_rate') if monthly else
                             ('repeated','observed_repeated','repeat_rate'))
    _keys(value, (*COUNT_FIELDS, 'cohort_size', event, observed, observed+'_ineligible', rate, 'status', 'reasons', *extra))
    for k in (*COUNT_FIELDS, 'cohort_size', event, observed, observed+'_ineligible'):
        _count(value[k])
    eligible, censored, missing, overlap = (value[k] for k in COUNT_FIELDS[:4])
    _require(value['cohort_size'] == size and overlap <= min(censored, missing))
    _require(eligible + censored + missing - overlap == size)
    _require(0 <= value[event] <= eligible <= size and value[observed] <= size)
    _require(value[observed] == value[event] + value[observed+'_ineligible'])
    _require(value[observed+'_ineligible'] <= size-eligible)
    _require(value['confirmed_without_event'] == eligible-value[event])
    _ratio(value[rate], value[event], eligible, percent=True)
    expected = 'NO_BUYERS' if not size else 'FULL' if eligible == size else 'PARTIAL' if eligible else 'UNAVAILABLE'
    _require(value['status'] == expected)
    reasons = (['RIGHT_CENSORED'] if censored else []) + (['ORDERS_COVERAGE_INCOMPLETE'] if missing else [])
    _require(value['reasons'] == reasons)


def _block(value, size, context, *, month=None):
    repeats = _count(value['observed_repeat_buyers'])
    _require(repeats <= size)
    histogram = _histogram(value['second_purchase_days'], repeats)
    if month is not None and histogram:
        _require(max(histogram) <= (context.as_of_date - _month_date(month)).days)
    _keys(value['horizons'], [str(h) for h in HORIZONS])
    previous_observed, previous_eligible = 0, size
    for h in HORIZONS:
        c = value['horizons'][str(h)]
        _counts(c, size)
        _require(c['observed_repeated'] == sum(n for d,n in histogram.items() if d <= h))
        _require(previous_observed <= c['observed_repeated'] <= repeats and c['eligible'] <= previous_eligible)
        previous_observed, previous_eligible = c['observed_repeated'], c['eligible']
        if month is not None:
            # Aggregate rows omit first dates. Check the bounds possible within
            # this cohort month using the core's own coverage rule, not a second
            # cohort calculation or invented buyer-level reconstruction.
            first, last = _month_date(month), min(_month_date(month+1)-timedelta(days=1), context.as_of_date)
            states = [context._eligibility(_utc(first+timedelta(days=d)),
                       _utc(first+timedelta(days=d+h+1))) for d in range((last-first).days+1)]
            for field, flags in (
                ('eligible',[not young and not gap for young,gap in states]),
                ('censored',[young for young,_ in states]),
                ('coverage_incomplete',[gap for _,gap in states]),
                ('censored_and_incomplete',[young and gap for young,gap in states]),
            ):
                _require(size*all(flags) <= c[field] <= size*any(flags))
    cells = value['retention']
    _require(isinstance(cells, (list,tuple)) and len(cells) == 13)
    _keys(cells[0], ('month_offset','kind','buyers',*(("month",) if month is not None else ())))
    _require(type(cells[0]['month_offset']) is int and cells[0]['month_offset'] == 0)
    _require(cells[0]['kind'] == 'COHORT_SIZE' and _count(cells[0]['buyers']) == size)
    if month is not None:
        _require(cells[0]['month'] == _month_key(month))
    for offset, cell in enumerate(cells[1:],1):
        extra = ('month_offset','kind',*(("month",) if month is not None else ()))
        _counts(cell, size, monthly=True, extra=extra)
        _require(type(cell['month_offset']) is int and cell['month_offset'] == offset and cell['kind'] == 'MONTHLY_ACTIVITY')
        if month is not None:
            _require(cell['month'] == _month_key(month+offset))
            # Reuse core metadata checks; no buyer reconstruction or second math.
            censored, missing = context._eligibility(_utc(_month_date(month+offset)), _utc(_month_date(month+offset+1)))
            _require(cell['eligible'] == size * (not censored and not missing))
            _require(cell['censored'] == size*censored and cell['coverage_incomplete'] == size*missing)
            _require(cell['censored_and_incomplete'] == size*(censored and missing))
    return histogram


def _valued(value, *, monthly=False):
    fields = 'known_orders amount mean_order'
    _keys(value, fields + (' first_orders first_amount repeat_orders repeat_amount repeat_amount_rate' if monthly else ''))
    n, amount = _count(value['known_orders']), _money(value['amount'])
    _require(n > 0 or amount == 0)
    with localcontext():
        _ratio(value['mean_order'], amount, n, decimal=True)
    if monthly:
        first, repeat = _count(value['first_orders']), _count(value['repeat_orders'])
        a, b = _money(value['first_amount']), _money(value['repeat_amount'])
        _require(first + repeat == n and _add(a,b) == amount)
        _require((first > 0 or a == 0) and (repeat > 0 or b == 0))
        with localcontext():
            _ratio(value['repeat_amount_rate'], b, amount, percent=True)
    return n, amount


def _monetary(value, policy):
    """Optional saved extension. Closed keys keep private identities out of DTO."""
    _keys(value, 'cohorts horizons')
    _keys(value['cohorts'], policy['cohorts'])
    totals = {c: [0, Decimal(0)] for c in CURRENCIES}
    unvalued_total = 0
    for month, row in value['cohorts'].items():
        _keys(row, 'cohort_size orders unvalued_orders currencies')
        old = policy['cohorts'][month]
        size = _count(row['cohort_size'])
        orders, unvalued = _count(row['orders']), _count(row['unvalued_orders'])
        _require(size == old['cohort_size'] and orders >= size + old['observed_repeat_buyers'])
        _keys(row['currencies'], CURRENCIES)
        unvalued_total += unvalued
        known = first = 0
        for currency in CURRENCIES:
            money = row['currencies'][currency]
            n, amount = _valued(money, monthly=True)
            _require(money['first_orders'] <= size)
            known += n
            first += money['first_orders']
            totals[currency][0] += money['repeat_orders']
            totals[currency][1] = _add(totals[currency][1], _money(money['repeat_amount']))
        _require(known + unvalued == orders and first <= size and size - first <= unvalued)
    _keys(value['horizons'], [str(h) for h in HORIZONS])
    for horizon, row in value['horizons'].items():
        _keys(row, 'eligible second_buyers unvalued_orders currencies')
        old = policy['summary']['horizons'][horizon]
        _require(_count(row['eligible']) == old['eligible'])
        _require(_count(row['second_buyers']) == old['repeated'])
        unvalued = _count(row['unvalued_orders'])
        _require(unvalued <= unvalued_total)
        _keys(row['currencies'], CURRENCIES)
        known = 0
        for currency in CURRENCIES:
            n, amount = _valued(row['currencies'][currency])
            _require(n <= totals[currency][0] and amount <= totals[currency][1])
            known += n
        _require(known + unvalued == row['second_buyers'])


def _channel_id(value):
    _require(value is None or isinstance(value,str) and bool(value.strip()) and value == value.strip())
    return value


def _purchase_channels(value, policy):
    _keys(value, 'buyers repeat_buyers same_known_buyers other_known_buyers unknown_transition_buyers '
                 'same_known_rate channels transitions')
    buyers, repeated = _count(value['buyers']), _count(value['repeat_buyers'])
    _require(buyers == policy['displayed_buyers'] and repeated == policy['summary']['observed_repeat_buyers'])
    same, other, unknown = (_count(value[k]) for k in (
        'same_known_buyers','other_known_buyers','unknown_transition_buyers'))
    _require(same + other + unknown == repeated)
    _ratio(value['same_known_rate'],same,same+other,percent=True)
    _require(isinstance(value['channels'],(list,tuple)) and bool(value['channels']))
    catalog = {}
    for row in value['channels']:
        _keys(row,'external_id name first_buyers first_rate second_buyers second_rate')
        key = _channel_id(row['external_id'])
        _require(key not in catalog and isinstance(row['name'],str) and bool(row['name'].strip())
                 and row['name'] == row['name'].strip())
        _require(key is not None or row['name'] == UNKNOWN_CHANNEL)
        first, second = _count(row['first_buyers']), _count(row['second_buyers'])
        _require(key is None or first + second > 0)
        _ratio(row['first_rate'],first,buyers,percent=True)
        _ratio(row['second_rate'],second,repeated,percent=True)
        catalog[key] = row
    _require(None in catalog and list(catalog) == sorted(catalog,key=channel_key))
    _require(sum(r['first_buyers'] for r in catalog.values()) == buyers)
    _require(sum(r['second_buyers'] for r in catalog.values()) == repeated)
    _require(isinstance(value['transitions'],(list,tuple)))
    pairs, firsts, seconds = {}, Counter(), Counter()
    totals = Counter()
    for row in value['transitions']:
        _keys(row,'first_external_id second_external_id buyers repeat_rate first_channel_repeat_rate')
        a, b = _channel_id(row['first_external_id']), _channel_id(row['second_external_id'])
        count = _count(row['buyers'])
        _require(a in catalog and b in catalog and (a,b) not in pairs and count > 0)
        pairs[a,b] = row
        firsts[a] += count
        seconds[b] += count
        totals['unknown' if a is None or b is None else 'same' if a == b else 'other'] += count
        _ratio(row['repeat_rate'],count,repeated,percent=True)
    _require(list(pairs) == sorted(pairs,key=lambda pair:(
        -pairs[pair]['buyers'],channel_key(pair[0]),channel_key(pair[1]))))
    _require((totals['same'],totals['other'],totals['unknown']) == (same,other,unknown))
    for key, row in catalog.items():
        _require(firsts[key] <= row['first_buyers'] and seconds[key] == row['second_buyers'])
    for (a,_), row in pairs.items():
        _ratio(row['first_channel_repeat_rate'],row['buyers'],firsts[a],percent=True)


def _purchase_categories(value, policy):
    """Multi-label buyers: validate bounds, never equate label sums to unique buyers."""
    _keys(value,'buyers repeat_buyers unknown_first_buyers unknown_second_buyers categories transitions')
    buyers, repeated = _count(value['buyers']), _count(value['repeat_buyers'])
    _require(buyers == policy['displayed_buyers'] and repeated == policy['summary']['observed_repeat_buyers'])
    _require(_count(value['unknown_first_buyers']) <= buyers and _count(value['unknown_second_buyers']) <= repeated)
    _require(isinstance(value['categories'],(list,tuple)) and bool(value['categories']))
    catalog = {}
    for row in value['categories']:
        _keys(row,'code name path first_buyers repeat_buyers observed_repeat_rate second_buyers second_rate')
        code = _channel_id(row['code'])
        _require(code not in catalog and isinstance(row['name'],str) and bool(row['name'].strip())
                 and row['name'] == row['name'].strip())
        path = row['path']
        if path is not None:
            _require(isinstance(path,(list,tuple)) and bool(path) and path[-1] == row['name'])
            _require(all(isinstance(v,str) and bool(v.strip()) and v == v.strip() for v in path))
        _require(code is not None or row['name'] == UNKNOWN_CATEGORY and path is None)
        first, repeat, second = (_count(row[k]) for k in ('first_buyers','repeat_buyers','second_buyers'))
        _require(repeat <= min(first,repeated) and first <= buyers and second <= repeated)
        _require(first-repeat <= buyers-repeated and (code is None or first+second > 0))
        _ratio(row['observed_repeat_rate'],repeat,first,percent=True)
        _ratio(row['second_rate'],second,repeated,percent=True)
        catalog[code] = row
    _require(None in catalog and list(catalog) == sorted(catalog,key=category_key))
    _require(catalog[None]['first_buyers'] == value['unknown_first_buyers'])
    _require(catalog[None]['second_buyers'] == value['unknown_second_buyers'])
    for field, base in (('first_buyers',buyers),('repeat_buyers',repeated),('second_buyers',repeated)):
        _require(base <= sum(r[field] for r in catalog.values()) <= base*MAX_ORDER_CATEGORIES)
    _require(isinstance(value['transitions'],(list,tuple)) and len(value['transitions']) <= MAX_AGGREGATE_PAIRS)
    pairs, firsts, seconds = {}, Counter(), Counter()
    for row in value['transitions']:
        _keys(row,'first_code second_code buyers repeat_rate')
        a, b = _channel_id(row['first_code']), _channel_id(row['second_code'])
        n = _count(row['buyers'])
        _require(a in catalog and b in catalog and (a,b) not in pairs and n > 0)
        _require(n <= min(catalog[a]['repeat_buyers'],catalog[b]['second_buyers']))
        pairs[a,b] = row
        firsts[a] += n
        seconds[b] += n
        _ratio(row['repeat_rate'],n,repeated,percent=True)
    _require(list(pairs) == sorted(pairs,key=lambda pair:(
        -pairs[pair]['buyers'],category_key(pair[0]),category_key(pair[1]))))
    _require(repeated <= sum(row['buyers'] for row in pairs.values()) <= repeated*MAX_PAIRS_PER_BUYER)
    for code, row in catalog.items():
        _require(row['repeat_buyers'] <= firsts[code] <= row['repeat_buyers']*MAX_ORDER_CATEGORIES)
        _require(row['second_buyers'] <= seconds[code] <= row['second_buyers']*MAX_ORDER_CATEGORIES)


def validate_cohort_statistics(value, *, coverage, selected_start, selected_end, calculated_at):
    if value is None:
        return
    _keys(value, 'as_of_date cutoff_exclusive cohort_definition source_coverage effective_coverage cohort_bounds '
                 'horizons month_offsets policies quality')
    coverage = shared_intervals([], [(a.isoformat() if isinstance(a,datetime) else a,
                                       b.isoformat() if isinstance(b,datetime) else b) for a,b in coverage])
    cutoff = cohort_cutoff(coverage, selected_end, calculated_at)
    _require(cutoff is not None and value['as_of_date'] == cutoff.isoformat())
    context = CohortAggregates(cutoff, coverage, identity_space='VALIDATION',
        cohort_start=selected_start[:7] if selected_start else None, cohort_end=selected_end[:7] if selected_end else None)
    _require(value['cutoff_exclusive'] == context.end.isoformat())
    _require(value['cohort_definition'] == 'FIRST_OBSERVED_PURCHASE_UTC_MONTH')
    _require(value['source_coverage'] == [[a.isoformat(),b.isoformat()] for a,b in context.source_coverage])
    _require(value['effective_coverage'] == [[a.isoformat(),b.isoformat()] for a,b in context.coverage])
    _keys(value['cohort_bounds'], 'start end')
    _require(value['cohort_bounds'] == dict(start=selected_start[:7] if selected_start else None,
                                          end=selected_end[:7] if selected_end else None))
    _require(value['horizons'] == list(HORIZONS) and all(type(v) is int for v in value['horizons']))
    _require(value['month_offsets'] == list(MONTH_OFFSETS) and all(type(v) is int for v in value['month_offsets']))
    _keys(value['quality'], 'coverage_metadata_available lifetime_first_purchase_proven second_purchase_is_observed '
                            'gaps_can_hide_earlier_orders facts_after_cutoff observed_purchase_facts_outside_coverage')
    q = value['quality']
    for name, flag in (('coverage_metadata_available',True), ('lifetime_first_purchase_proven',False),
                       ('second_purchase_is_observed',True), ('gaps_can_hide_earlier_orders',True)):
        _require(q[name] is flag)
    _count(q['facts_after_cutoff'])
    _count(q['observed_purchase_facts_outside_coverage'])
    _keys(value['policies'], POLICIES)
    for policy, p in value['policies'].items():
        has_money = policy == POLICIES[0] and 'monetary' in p
        has_channels = policy == POLICIES[0] and 'purchase_channels' in p
        has_categories = policy == POLICIES[0] and 'purchase_categories' in p
        _keys(p, 'base_buyers displayed_buyers hidden_buyers cohorts summary'
              + (' monetary' if has_money else '') + (' purchase_channels' if has_channels else '')
              + (' purchase_categories' if has_categories else ''))
        base, displayed, hidden = (_count(p[k]) for k in ('base_buyers','displayed_buyers','hidden_buyers'))
        _require(displayed+hidden == base and isinstance(p['cohorts'],dict))
        cohort_keys = list(p['cohorts'])
        _require(cohort_keys == sorted(cohort_keys))
        histograms = []
        for key, row in p['cohorts'].items():
            month = _month(key)
            _require(key <= value['as_of_date'][:7])
            _require((context.cohort_start is None or month >= context.cohort_start) and
                     (context.cohort_end is None or month <= context.cohort_end))
            _keys(row, 'cohort_size observed_repeat_buyers second_purchase_days horizons retention')
            size = _count(row['cohort_size'])
            _require(size > 0)
            histograms.append(_block(row,size,context,month=month))
        summary = p['summary']
        _keys(summary, 'cohort_count cohort_size observed_repeat_buyers second_purchase_days horizons retention')
        _require(_count(summary['cohort_count']) == len(cohort_keys) and _count(summary['cohort_size']) == displayed)
        rows = list(p['cohorts'].values())
        _require(sum(r['cohort_size'] for r in rows) == displayed)
        hist = _block(summary,displayed,context)
        _require(hist == sum(histograms,Counter()))
        _require(summary['second_purchase_days']['coverage_incomplete_buyers'] ==
                 sum(r['second_purchase_days']['coverage_incomplete_buyers'] for r in rows))
        for h in HORIZONS:
            for k in (*COUNT_FIELDS,'repeated','observed_repeated','observed_repeated_ineligible'):
                _require(summary['horizons'][str(h)][k] == sum(r['horizons'][str(h)][k] for r in rows))
        for offset in range(1,13):
            for k in (*COUNT_FIELDS,'active','observed_active','observed_active_ineligible'):
                _require(summary['retention'][offset][k] == sum(r['retention'][offset][k] for r in rows))
        if has_money:
            _monetary(p['monetary'], p)
        if has_channels:
            _purchase_channels(p['purchase_channels'], p)
        if has_categories:
            _purchase_categories(p['purchase_categories'], p)
    _require(value['policies'][POLICIES[0]]['base_buyers'] <= value['policies'][POLICIES[1]]['base_buyers'])
