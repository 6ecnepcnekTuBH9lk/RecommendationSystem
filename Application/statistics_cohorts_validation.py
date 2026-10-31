"""Closed aggregate cohort contract; context/consistency checks, never raw I/O."""

from collections import Counter
from datetime import date, datetime, timedelta
import re

from Application.order_statistics import histogram_median
from Application.statistics_period import shared_intervals
from Application.statistics_buyer_validation import _keys, _require, _count, _number, _ratio, _rows, _close
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
    for p in value['policies'].values():
        _keys(p, 'base_buyers displayed_buyers hidden_buyers cohorts summary')
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
    _require(value['policies'][POLICIES[0]]['base_buyers'] <= value['policies'][POLICIES[1]]['base_buyers'])
