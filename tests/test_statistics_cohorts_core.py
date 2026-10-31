"""Synthetic cohort contracts; no actual sources, GUI or index generations."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.records import CustomerMergeRecord, ProductKey
from Application.statistics_cohorts_core import CohortAggregates, HORIZONS, POLICIES
from Application.statistics_purchase_facts import build_purchase_order_fact


SPACE = 'REFERENCE:synthetic-cohorts'
FULL = [('2020-01-01T00:00:00Z', '2028-01-01T00:00:00Z')]
AS_OF = date(2026, 12, 31)


def fact(when='2024-01-10', *, order='secret-order', customer='secret-buyer', statuses=('F',), space=SPACE):
    when = when + 'T12:00:00Z' if len(when) == 10 else when
    rows = (SimpleNamespace(line_status=status, quantity=Decimal(2), price_of_line=Decimal('12.5'),
                            product=ProductKey('offline1C', 'secret-product')) for status in statuses)
    return build_purchase_order_fact(rows, identity_space=space, customer_id=customer, order_id=order,
                                     order_datetime_utc=datetime.fromisoformat(when.replace('Z', '+00:00')))


def aggregate(records=(), coverage=FULL, *, as_of=AS_OF, **kwargs):
    core = CohortAggregates(as_of, coverage, identity_space=SPACE, **kwargs)
    for record in records:
        core.add(record)
    return core


def policy(core, index=0):
    return core.result()['policies'][POLICIES[index]]


def row(core, month='2024-01', index=0):
    return policy(core, index)['cohorts'][month]


def horizon(core, days=30, month='2024-01', index=0):
    return row(core, month, index)['horizons'][str(days)]


def assert_counts(count, event='repeated', observed='observed_repeated', rate='repeat_rate'):
    assert count['eligible'] + count['censored'] + count['coverage_incomplete'] - count['censored_and_incomplete'] == count['cohort_size']
    assert 0 <= count[event] <= count['eligible'] <= count['cohort_size']
    assert count[observed] == count[event] + count[observed + '_ineligible']
    assert count['confirmed_without_event'] == count['eligible'] - count[event]
    assert count[rate] == (100 * count[event] / count['eligible'] if count['eligible'] else None)


@pytest.mark.parametrize('count', [1, 2, 3, 20])
def test_first_two_unique_orders_not_positions_or_input_order(count):
    records = [fact(f'2024-01-{10+i:02d}', order=f'o{i}', statuses=('F', 'sold', 'Return', 'CD')) for i in range(count)]
    actual = row(aggregate(reversed(records)))
    assert actual['cohort_size'] == 1
    assert actual['observed_repeat_buyers'] == int(count > 1)
    assert actual['second_purchase_days']['distribution'] == ([[1, 1]] if count > 1 else [])
    assert actual['horizons']['30']['repeated'] == int(count > 1)


@pytest.mark.parametrize('statuses,strict,extended', [
    (('F',), 1, 1), (('sold',), 1, 1), (('CP',), 0, 1), (('F', 'CP'), 1, 1),
    (('F', 'Return'), 1, 1), (('Return',), 0, 0), (('CD', 'RJ'), 0, 0),
    (('delivering',), 0, 0), (('ready-for-self-pickup',), 0, 0),
])
def test_prepared_purchase_policies(statuses, strict, extended):
    core = aggregate([fact(statuses=statuses)])
    assert [policy(core, i)['base_buyers'] for i in (0, 1)] == [strict, extended]


def test_strict_and_extended_have_independent_first_and_second_dates():
    core = aggregate([fact('2024-01-30', order='cp', statuses=('CP',)),
                      fact('2024-02-10', order='f'), fact('2024-03-10', order='sold', statuses=('sold',))])
    strict, extended = row(core, '2024-02'), row(core, '2024-01', 1)
    assert strict['second_purchase_days']['distribution'] == [[29, 1]]
    assert extended['second_purchase_days']['distribution'] == [[11, 1]]
    assert extended['retention'][1]['observed_active'] == extended['retention'][2]['observed_active'] == 1


@pytest.mark.parametrize('days', [0, 1, 29, 30, 31, 60, 61, 90, 91, 180, 181, 365, 366])
def test_inclusive_cumulative_calendar_day_horizons(days):
    start = date(2024, 1, 10)
    core = aggregate([fact(start.isoformat(), order='first'),
                      fact((start + timedelta(days=days)).isoformat(), order='second')])
    counts = [horizon(core, h) for h in HORIZONS]
    assert [c['repeated'] for c in counts] == [int(days <= h) for h in HORIZONS]
    assert [c['observed_repeated'] for c in counts] == sorted(c['observed_repeated'] for c in counts)
    for c in counts:
        assert_counts(c)


@pytest.mark.parametrize('first,second,days', [
    ('2024-02-28T23:59:59Z', '2024-02-29T00:00:00Z', 1),
    ('2024-02-28T23:59:59Z', '2024-03-29T00:00:00Z', 30),
    ('2024-12-31T23:59:59Z', '2025-01-01T00:00:00Z', 1),
    ('2024-01-10T00:01:00Z', '2024-01-10T23:59:59Z', 0),
])
def test_utc_dates_not_elapsed_24_hour_periods(first, second, days):
    core = aggregate([fact(first, order='a'), fact(second, order='b')])
    actual = next(iter(policy(core)['cohorts'].values()))
    assert actual['second_purchase_days']['distribution'] == [[days, 1]]


def test_same_timestamp_mixed_order_id_types_and_different_buyers_are_deterministic():
    records = [fact(order='b'), fact(order=2), fact(order=1), fact(order='a'), fact(customer='other')]
    expected = aggregate(records).result()
    for seed in range(8):
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        assert aggregate(shuffled).result() == expected
    actual = row(aggregate(records))
    assert actual['cohort_size'] == 2
    assert actual['second_purchase_days']['distribution'] == [[0, 1]]
    buyer = aggregate(records)._buyers[POLICIES[0]]['secret-buyer']
    assert buyer.first[2] == 1 and buyer.second[2] == 2


def test_canonical_aliases_are_prepared_by_caller_once():
    resolver = CustomerIdResolver([CustomerMergeRecord('merge', datetime(2024, 1, 1, tzinfo=timezone.utc), 'canonical', ('old',))])
    core = aggregate([fact(customer=resolver.resolve('old'), order='a'),
                      fact('2024-01-11', customer=resolver.resolve('canonical'), order='b')])
    assert row(core)['cohort_size'] == row(core)['observed_repeat_buyers'] == 1


def test_cutoff_is_explicit_inclusive_day_and_future_orders_are_ignored():
    core = aggregate([fact('2024-01-10', order='a'), fact('2024-02-09T23:59:59.999999Z', order='b'),
                      fact('2024-02-10T00:00:00Z', order='future')], as_of=date(2024, 2, 9))
    assert horizon(core)['repeated'] == horizon(core)['eligible'] == 1
    assert core.result()['quality']['facts_after_cutoff'] == 1
    assert core.result()['effective_coverage'][-1][1] == '2024-02-10T00:00:00+00:00'


def test_late_positive_events_are_diagnostic_until_full_horizon_is_observed():
    core = aggregate([fact('2024-01-10', order='a'), fact('2024-01-11', order='b')], as_of=date(2024, 1, 15))
    for h in HORIZONS:
        c = horizon(core, h)
        assert (c['eligible'], c['repeated'], c['censored'], c['coverage_incomplete']) == (0, 0, 1, 0)
        assert c['observed_repeated'] == c['observed_repeated_ineligible'] == 1
        assert c['repeat_rate'] is None
        assert_counts(c)


def test_mature_negative_is_proven_only_with_complete_coverage():
    full, gap = FULL, [('2024-01-01T00:00:00Z', '2024-01-20T00:00:00Z'),
                       ('2024-01-21T00:00:00Z', '2028-01-01T00:00:00Z')]
    good, missing = (horizon(aggregate([fact()], coverage)) for coverage in (full, gap))
    assert (good['eligible'], good['confirmed_without_event'], good['repeat_rate']) == (1, 1, 0)
    assert (missing['eligible'], missing['confirmed_without_event'], missing['repeat_rate']) == (0, 0, None)
    assert missing['coverage_incomplete'] == 1


@pytest.mark.parametrize('coverage', [None, [], [('2025-01-01T00:00:00Z', '2028-01-01T00:00:00Z')]])
def test_no_complete_coverage_does_not_erase_observed_orders_or_invent_zero_rates(coverage):
    core = aggregate([fact(order='a'), fact('2024-01-11', order='b')], coverage)
    c = horizon(core)
    assert (c['observed_repeated'], c['repeated'], c['eligible'], c['coverage_incomplete']) == (1, 0, 0, 1)
    assert core.result()['quality']['observed_purchase_facts_outside_coverage'] == 2
    assert c['repeat_rate'] is None
    assert_counts(c)


def test_gaps_can_overlap_right_censoring_but_future_is_not_itself_a_gap():
    coverage = [('2024-01-01T00:00:00Z', '2024-01-11T00:00:00Z'),
                ('2024-01-13T00:00:00Z', '2028-01-01T00:00:00Z')]
    c = horizon(aggregate([fact()], coverage, as_of=date(2024, 1, 15)))
    assert (c['censored'], c['coverage_incomplete'], c['censored_and_incomplete']) == (1, 1, 1)
    assert_counts(c)


def test_one_cohort_can_have_different_eligible_buyers_and_matching_numerator():
    records = [fact('2024-01-01', customer='mature', order='a'),
               fact('2024-01-02', customer='mature', order='b'),
               fact('2024-01-20', customer='young', order='c'),
               fact('2024-01-21', customer='young', order='d')]
    c = horizon(aggregate(records, as_of=date(2024, 1, 31)))
    assert (c['cohort_size'], c['eligible'], c['repeated'], c['observed_repeated'], c['censored']) == (2, 1, 1, 2, 1)
    assert c['repeat_rate'] == 100
    assert_counts(c)


def test_observed_cumulative_counts_not_mixed_eligibility_counts_are_monotonic():
    core = aggregate([fact('2024-01-01', order='a'), fact('2024-01-02', order='b')], as_of=date(2024, 1, 31))
    assert horizon(core, 30)['repeated'] == 1 and horizon(core, 60)['repeated'] == 0
    assert all(horizon(core, h)['observed_repeated'] == 1 for h in HORIZONS)


def test_day_coverage_is_conservative_and_adjacent_intervals_are_merged():
    split = [('2024-01-10T00:00:00Z', '2024-01-20T00:00:00Z'),
             ('2024-01-20T00:00:00Z', '2024-02-10T00:00:00Z')]
    assert horizon(aggregate([fact()], split))['eligible'] == 1
    partial_start = [('2024-01-10T12:00:00Z', '2024-02-10T00:00:00Z')]
    assert horizon(aggregate([fact()], partial_start))['coverage_incomplete'] == 1


def test_observed_interval_does_not_claim_actual_second_purchase_across_gaps():
    coverage = [('2024-01-01T00:00:00Z', '2024-01-11T00:00:00Z'),
                ('2024-01-15T00:00:00Z', '2028-01-01T00:00:00Z')]
    actual = row(aggregate([fact(order='a'), fact('2024-01-16', order='b')], coverage))
    assert actual['second_purchase_days']['distribution'] == [[6, 1]]
    assert actual['second_purchase_days']['coverage_incomplete_buyers'] == 1


def test_retention_is_monthly_not_cumulative_or_order_count():
    records = [fact('2024-01-10', order='a'), fact('2024-03-01', order='b'),
               fact('2024-03-15', order='c'), fact('2024-01-20', customer='other', order='d'),
               fact('2024-02-05', customer='other', order='e')]
    cells = row(aggregate(reversed(records)))['retention']
    assert cells[0] == dict(month_offset=0, month='2024-01', kind='COHORT_SIZE', buyers=2)
    assert [c['observed_active'] for c in cells[1:4]] == [1, 1, 0]
    assert [c['retention_rate'] for c in cells[1:4]] == [50, 50, 0]
    assert cells[12]['month'] == '2025-01'
    for c in cells[1:]:
        assert_counts(c, 'active', 'observed_active', 'retention_rate')


def test_unfinished_and_future_retention_months_have_no_false_zero():
    cells = row(aggregate([fact(order='a'), fact('2024-02-05', order='b')], as_of=date(2024, 2, 15)))['retention']
    for offset in range(1, 13):
        c = cells[offset]
        assert c['eligible'] == c['active'] == 0 and c['censored'] == 1
        assert c['retention_rate'] is None
        assert_counts(c, 'active', 'observed_active', 'retention_rate')
    assert cells[1]['observed_active_ineligible'] == 1
    assert cells[12]['observed_active'] == cells[12]['coverage_incomplete'] == 0


def test_retention_gap_only_invalidates_its_calendar_month_and_m0_stays_size():
    coverage = [('2024-01-01T00:00:00Z', '2024-04-01T00:00:00Z'),
                ('2024-05-01T00:00:00Z', '2028-01-01T00:00:00Z')]
    cells = row(aggregate([fact()], coverage))['retention']
    assert cells[0]['buyers'] == 1
    assert cells[2]['retention_rate'] == cells[4]['retention_rate'] == 0
    assert cells[3]['coverage_incomplete'] == 1 and cells[3]['retention_rate'] is None


def test_display_bounds_do_not_turn_existing_buyers_into_new_cohorts():
    core = aggregate([fact('2023-11-05', order='old'), fact('2024-01-10', order='newer'),
                      fact('2024-02-10', customer='visible', order='visible')], cohort_start='2024-01', cohort_end='2024-02')
    actual = policy(core)
    assert list(actual['cohorts']) == ['2024-02']
    assert (actual['base_buyers'], actual['displayed_buyers'], actual['hidden_buyers']) == (2, 1, 1)


def test_bitmask_rebases_for_early_orders_and_keeps_m12_and_ignores_m13():
    core = aggregate([fact('2025-02-01', order='m13'), fact('2025-01-01', order='m12'),
                      fact('2024-03-01', order='m2'), fact('2024-01-01', order='m0')])
    cells = row(core)['retention']
    assert cells[1]['active'] == 0 and cells[2]['active'] == cells[12]['active'] == 1
    assert row(core)['second_purchase_days']['distribution'] == [[60, 1]]


def test_summary_is_pooled_by_eligible_buyers_not_mean_of_cohort_percentages():
    records = [fact('2024-01-01', customer='a', order='a'), fact('2024-01-02', customer='a', order='aa'),
               fact('2024-02-01', customer='b', order='b'), fact('2024-02-01', customer='c', order='c')]
    actual = policy(aggregate(records))
    assert actual['summary']['horizons']['30']['repeat_rate'] == 100 / 3
    for h in HORIZONS:
        total = actual['summary']['horizons'][str(h)]
        assert_counts(total)
        for field in ('cohort_size', 'eligible', 'repeated', 'observed_repeated', 'censored', 'coverage_incomplete'):
            assert total[field] == sum(c['horizons'][str(h)][field] for c in actual['cohorts'].values())


@pytest.mark.parametrize('coverage', [FULL, [], None])
def test_empty_results_preserve_coverage_and_no_buyers_means_no_rate(coverage):
    actual = aggregate(coverage=coverage).result()
    for value in actual['policies'].values():
        assert not value['cohorts'] and value['base_buyers'] == 0
        for count in value['summary']['horizons'].values():
            assert count['status'] == 'NO_BUYERS' and count['repeat_rate'] is None
            assert_counts(count)
    assert actual['quality']['coverage_metadata_available'] == (coverage is not None)


def test_coverage_is_normalized_sorted_merged_and_clipped_without_actions_intersection():
    coverage = [('2024-02-01T03:00:00+03:00', '2029-01-01T00:00:00Z'),
                ('2024-01-01T00:00:00Z', '2024-02-01T00:00:00Z')]
    actual = aggregate(coverage=coverage, as_of=date(2024, 2, 29)).result()
    assert actual['source_coverage'] == [['2024-01-01T00:00:00+00:00', '2029-01-01T00:00:00+00:00']]
    assert actual['effective_coverage'] == [['2024-01-01T00:00:00+00:00', '2024-03-01T00:00:00+00:00']]


@pytest.mark.parametrize('coverage', [
    [('2024-02-01T00:00:00Z', '2024-01-01T00:00:00Z')],
    [('2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')],
    [('2024-01-01', '2025-01-01')], [('bad', 'date')], [None], [('single',)], 'bad', '', {}, 0, False, 42,
])
def test_invalid_coverage_is_rejected(coverage):
    with pytest.raises(ValueError, match='coverage'):
        aggregate(coverage=coverage)


@pytest.mark.parametrize('kwargs', [
    {'identity_space': ''}, {'identity_space': ' wrong '}, {'identity_space': None},
    {'policies': []}, {'policies': 'STRICT_COMPLETED'}, {'policies': ['other']}, {'policies': None},
    {'policies': [POLICIES[0], POLICIES[0]]}, {'cohort_start': '2024-13'}, {'cohort_start': '2024-1'},
    {'cohort_start': '2025-01', 'cohort_end': '2024-12'}, {'cohort_end': date(2024, 1, 1)}, {'check': 1},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CohortAggregates(AS_OF, FULL, **{'identity_space': SPACE, **kwargs})


@pytest.mark.parametrize('cutoff', [None, '2024-01-01', datetime(2024, 1, 1), date.max])
def test_explicit_representable_cutoff_is_required(cutoff):
    with pytest.raises(ValueError):
        aggregate(as_of=cutoff)


def test_policy_subset_order_is_canonical_and_datetime_coverage_is_accepted():
    coverage = [(datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2028, 1, 1, tzinfo=timezone.utc))]
    both = aggregate([fact()], coverage, policies=list(reversed(POLICIES))).result()
    assert tuple(both['policies']) == POLICIES
    one = aggregate([fact(statuses=('CP',))], coverage, policies=[POLICIES[1]]).result()
    assert tuple(one['policies']) == (POLICIES[1],)


@pytest.mark.parametrize('bad', [
    {'identity_space': 'INDEXED:other'}, {'order_datetime_utc': datetime(2024, 1, 1)},
    {'order_id': True}, {'customer_id': ''}, {'order_id': 0},
])
def test_bad_facts_fail_before_any_mutation(bad):
    core = aggregate([fact()])
    before = core.result()
    with pytest.raises(ValueError):
        core.add(replace(fact(order='other'), **bad))
    assert core.result() == before


def test_cancellation_before_add_and_during_result_is_retryable():
    core = aggregate([fact(order='a'), fact('2024-03-01', order='b')])
    before = core.result()
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        if calls >= 4:
            raise InterruptedError('synthetic cancellation')
    core.check = lambda: (_ for _ in ()).throw(InterruptedError())
    with pytest.raises(InterruptedError):
        core.add(fact(order='c'))
    core.check = cancelled
    with pytest.raises(InterruptedError):
        core.result()
    core.check = lambda: None
    assert core.result() == before


def test_results_are_fresh_private_aggregates_and_reordering_does_not_change_them():
    records = [fact('2024-01-10', order='a'), fact('2023-12-31', order='b'), fact('2024-04-01', order='c'),
               fact('2024-02-01', order='d', customer='other'), fact('2024-03-01', order='e', statuses=('CP',))]
    core = aggregate(records)
    expected = deepcopy(core.result())
    for seed in range(10):
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        assert aggregate(shuffled).result() == expected
    result = core.result()
    assert 'secret' not in json.dumps(result)
    result['source_coverage'].clear()
    result['policies'][POLICIES[0]]['cohorts'].clear()
    result['horizons'].append(999)
    assert core.result() == expected


def test_state_keeps_two_orders_and_a_month_bitmask_not_all_input_facts():
    core = aggregate(fact(order=i + 1) for i in range(3000))
    buyer = core._buyers[POLICIES[0]]['secret-buyer']
    assert buyer.months == 1
    assert set(buyer.__slots__) == {'first', 'second', 'months'}
    assert buyer.first[2] == 1 and buyer.second[2] == 2
    assert row(core)['observed_repeat_buyers'] == 1


def test_calculation_reuses_prepared_facts_without_io_reclassification_or_identity_resolution(monkeypatch):
    import builtins
    from Application import statistics_purchase_facts, statistics_purchase_policy
    records = [fact(order='a'), fact('2024-02-01', order='b', statuses=('CP',))]
    core = aggregate()
    def forbidden(*args, **kwargs):
        raise AssertionError('Cohort core attempted I/O or rebuilt its input')
    with monkeypatch.context() as patch:
        patch.setattr(builtins, 'open', forbidden)
        patch.setattr(Path, 'open', forbidden)
        patch.setattr(CustomerIdResolver, 'resolve', forbidden)
        patch.setattr(statistics_purchase_facts, 'build_purchase_order_fact', forbidden)
        patch.setattr(statistics_purchase_facts, 'eligible_purchase_policies', forbidden)
        patch.setattr(statistics_purchase_policy, 'classify_status', forbidden)
        for record in records:
            core.add(record)
        actual = core.result()
    assert actual['policies'][POLICIES[1]]['cohorts']['2024-01']['observed_repeat_buyers'] == 1


def test_import_and_empty_calculation_need_no_qt_database_or_writes():
    script = """
import os, sys
from datetime import date
def guard(event, args):
    if event == 'open':
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+') or
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            raise AssertionError('Unexpected write')
    if event in ('os.mkdir', 'sqlite3.connect', 'socket.connect'):
        raise AssertionError('Unexpected I/O')
sys.addaudithook(guard)
from Application.statistics_cohorts_core import CohortAggregates
assert CohortAggregates(date(2026, 12, 31), [], identity_space='synthetic').result()['policies']
assert not any(k.startswith(('PyQt', 'qt_material', 'pandas', 'torch')) for k in sys.modules)
"""
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('gap', [False, True])
def test_random_stream_matches_independent_daily_observation_oracle(gap):
    rng = random.Random(108)
    start, cutoff = date(2024, 1, 1), date(2025, 5, 15)
    records = [fact((start + timedelta(days=rng.randrange(550))).isoformat(),
                    order=i + 1, customer=f'buyer-{rng.randrange(20)}',
                    statuses=(rng.choice(('F', 'sold', 'CP', 'Return', 'delivering')),)) for i in range(260)]
    coverage = [('2024-01-01T00:00:00Z', '2026-01-01T00:00:00Z')]
    if gap:
        coverage = [('2024-01-01T00:00:00Z', '2024-07-01T00:00:00Z'),
                    ('2024-07-10T00:00:00Z', '2026-01-01T00:00:00Z')]
    observed_days = {start + timedelta(days=i) for i in range((cutoff - start).days + 1)
                     if not gap or not date(2024, 7, 1) <= start + timedelta(days=i) < date(2024, 7, 10)}
    result = aggregate(records, coverage, as_of=cutoff).result()
    for index, name in enumerate(POLICIES):
        grouped = {}
        for record in records:
            accepted = record.strict if index == 0 else record.extended
            if accepted.has_purchase and record.order_datetime_utc.date() <= cutoff:
                grouped.setdefault(record.customer_id, []).append(record)
        expected = {}
        for orders in grouped.values():
            orders.sort(key=lambda f: (f.order_datetime_utc, f.order_id))
            first = orders[0].order_datetime_utc.date()
            month = f'{first.year:04d}-{first.month:02d}'
            expected.setdefault(month, []).append(orders)
        assert set(result['policies'][name]['cohorts']) == set(expected)
        for month, members in expected.items():
            actual = result['policies'][name]['cohorts'][month]
            assert actual['cohort_size'] == len(members)
            for h in HORIZONS:
                eligible = repeated = censored = incomplete = overlap = observed = 0
                for orders in members:
                    first = orders[0].order_datetime_utc.date()
                    last = first + timedelta(days=h)
                    required = {first + timedelta(days=d) for d in range(h + 1) if first + timedelta(days=d) <= cutoff}
                    young, missing = last > cutoff, not required.issubset(observed_days)
                    repeat = len(orders) > 1 and (orders[1].order_datetime_utc.date() - first).days <= h
                    eligible += not young and not missing
                    repeated += repeat and not young and not missing
                    censored += young
                    incomplete += missing
                    overlap += young and missing
                    observed += repeat
                c = actual['horizons'][str(h)]
                assert [c[k] for k in ('eligible', 'repeated', 'censored', 'coverage_incomplete',
                                       'censored_and_incomplete', 'observed_repeated')] == [
                    eligible, repeated, censored, incomplete, overlap, observed]
            for offset in range(1, 13):
                first = members[0][0].order_datetime_utc.date()
                year, m = divmod(first.year * 12 + first.month - 1 + offset, 12)
                begin = date(year, m + 1, 1)
                year, m = divmod(first.year * 12 + first.month + offset, 12)
                end = date(year, m + 1, 1)
                required = {begin + timedelta(days=d) for d in range((end - begin).days) if begin + timedelta(days=d) <= cutoff}
                eligible = (end - timedelta(days=1) <= cutoff and required.issubset(observed_days))
                active = sum(any(begin <= f.order_datetime_utc.date() < end for f in orders) for orders in members)
                cell = actual['retention'][offset]
                assert cell['eligible'] == len(members) * eligible
                assert cell['observed_active'] == active
                assert cell['active'] == active * eligible
