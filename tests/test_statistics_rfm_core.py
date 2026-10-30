"""RFM uses synthetic prepared facts only; no GUI, source scan or index build."""

from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from itertools import product
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
from Application.statistics_purchase_facts import build_purchase_order_fact
from Application.statistics_rfm_core import (
    RfmAggregates, POLICIES, WINDOWS, SEGMENTS, empirical_thresholds, rfm_score, rfm_segment, rfm_windows,
)


AS_OF = date(2026, 8, 31)
FULL = [('2022-09-01T00:00:00Z', '2026-09-01T00:00:00Z')]
SPACE = 'REFERENCE:synthetic-rfm'


def fact(customer='secret-buyer', order='secret-order', when='2026-08-30T12:00:00Z',
         lines=(('F', 3, '10.25', 'offline1C'),), space=SPACE):
    rows = (SimpleNamespace(line_status=status, quantity=Decimal(qty),
                            price_of_line=Decimal(amount) if amount is not None else None,
                            product=ProductKey(namespace, 'secret-product')) for status, qty, amount, namespace in lines)
    return build_purchase_order_fact(rows, identity_space=space, customer_id=customer, order_id=order,
                                     order_datetime_utc=datetime.fromisoformat(when.replace('Z', '+00:00')))


def aggregate(records=(), coverage=FULL, **kwargs):
    accumulator = RfmAggregates(AS_OF, coverage, identity_space=SPACE, **kwargs)
    for record in records:
        accumulator.add(record)
    return accumulator


def policy(result, index=0):
    return result['policies'][POLICIES[index]]


def window(result, name='LAST_12M', index=0):
    return policy(result, index)['windows'][name]


def currency(result, name='LAST_12M', unit='RUB', index=0):
    return window(result, name, index)['currencies'][unit]


@pytest.mark.parametrize('as_of,lower12,lower24,lower48', [
    ('2026-08-31', '2025-09-01', '2024-09-01', '2022-09-01'),
    ('2024-02-29', '2023-03-01', '2022-03-01', '2020-03-01'),
    ('2025-02-28', '2024-03-01', '2023-03-01', '2021-03-01'),
    ('2024-02-28', '2023-02-28', '2022-02-28', '2020-02-29'),
    ('2026-03-30', '2025-03-31', '2024-03-31', '2022-03-31'),
    ('2026-01-31', '2025-02-01', '2024-02-01', '2022-02-01'),
    ('2026-01-01', '2025-01-02', '2024-01-02', '2022-01-02'),
])
def test_calendar_windows_shift_the_exclusive_endpoint(as_of, lower12, lower24, lower48):
    cutoff = date.fromisoformat(as_of)
    periods = rfm_windows(cutoff)
    for name, expected in zip(WINDOWS[:3], (lower12, lower24, lower48)):
        start, end = periods[name]
        assert start.date().isoformat() == expected
        assert end.date() == cutoff + timedelta(days=1)
        assert start.time() == end.time() and start.hour == 0
        assert start.utcoffset() == end.utcoffset() == timedelta(0)
    assert periods['PREVIOUS_12M'] == (periods['LAST_24M'][0], periods['LAST_12M'][0])


@pytest.mark.parametrize('cutoff', [None, '2026-08-31', datetime(2026, 8, 31), date.min, date.max])
def test_explicit_representable_calendar_cutoff_is_required(cutoff):
    with pytest.raises(ValueError):
        rfm_windows(cutoff)


def test_boundaries_future_and_pre_48_month_orders_are_excluded():
    dates = ['2022-08-31T23:59:59Z', '2022-09-01T00:00:00Z', '2024-08-31T23:59:59Z',
             '2024-09-01T00:00:00Z', '2025-08-31T23:59:59Z', '2025-09-01T00:00:00Z',
             '2026-08-31T23:59:59Z', '2026-09-01T00:00:00Z']
    result = aggregate(fact(order=str(i), when=day) for i, day in enumerate(dates)).result()
    for name, n in zip(WINDOWS, (2, 4, 6, 2)):
        assert window(result, name)['behavior']['observed_orders'] == n
        assert currency(result, name)['frequency']['distribution'] == [[n, 1]]
        assert currency(result, name)['recency']['distribution'] == [[0, 1]]


@pytest.mark.parametrize('months_ago,when', [(36, '2023-08-31T12:00:00Z'), (18, '2025-02-28T12:00:00Z')])
def test_short_windows_keep_inactive_buyers_and_older_recency(months_ago, when):
    result = aggregate([fact(when=when)]).result()
    recent = currency(result)
    days = (AS_OF-datetime.fromisoformat(when.replace('Z', '+00:00')).date()).days
    assert policy(result)['base_buyers'] == recent['buyers'] == 1
    assert recent['frequency']['distribution'] == [[0, 1]]
    assert recent['recency']['distribution'] == [[days, 1]]
    assert recent['monetary']['distribution'] == [['0', 1]]
    assert recent['score_counts'] == [[5, 1, 1, 1]]  # Relative R score, not a fixed-day label.
    assert recent['confirmed_inactive_buyers'] == 1
    assert currency(result, 'LAST_48M')['observed_orders'] == 1
    expected = 'NEITHER' if months_ago == 36 else 'PREVIOUS_ONLY'
    assert policy(result)['comparison']['observed_groups'][expected] == 1


@pytest.mark.parametrize('statuses,strict,extended', [
    (('F',), 1, 1), (('sold',), 1, 1), (('CP',), 0, 1), (('delivering',), 0, 0),
    (('F', 'CP'), 1, 1), (('F', 'Return'), 1, 1), (('Return',), 0, 0),
])
def test_ready_policy_results_count_orders_once_and_do_not_reclassify(statuses, strict, extended):
    record = fact(lines=tuple((s, 3, '10.25', 'offline1C') for s in statuses))
    result = aggregate([record]).result()
    for i, n in enumerate((strict, extended)):
        assert policy(result, i)['base_buyers'] == n
        assert window(result, index=i)['behavior']['observed_orders'] == n
        expected = record.strict if i == 0 else record.extended
        assert Decimal(currency(result, index=i)['known_amount']) == (expected.amount if n else 0)


def test_quantity_is_not_money_multiplier_and_order_lines_are_not_frequency():
    result = aggregate([fact(lines=(('F', 30, '10.25', 'offline1C'), ('sold', 4, '2.75', 'offline1C')))]).result()
    assert currency(result)['observed_orders'] == 1
    assert currency(result)['known_amount'] == '13.00'
    assert currency(result)['monetary']['distribution'] == [['13.00', 1]]
    assert currency(result)['recency']['distribution'] == [[1, 1]]


def test_currency_slices_have_independent_bases_frequency_recency_and_thresholds():
    records = [fact(order='rub-old', when='2023-08-31T12:00:00Z'),
               fact(order='kzt-new', lines=(('F', 1, '5000', 'kanzlerKz'),)),
               fact(customer='rub-only', order='second', lines=(('F', 1, '20', 'offline1C'),))]
    result = aggregate(records).result()
    rub, kzt = currency(result), currency(result, unit='KZT')
    assert policy(result)['base_buyers'] == 2 and policy(result)['dual_currency_buyers'] == 1
    assert (rub['buyers'], kzt['buyers']) == (2, 1)
    assert rub['frequency']['distribution'] == [[0, 1], [1, 1]]
    assert kzt['frequency']['distribution'] == [[1, 1]]
    assert rub['recency']['distribution'][-1][0] > 1000 and kzt['recency']['distribution'] == [[1, 1]]
    assert rub['known_amount'] == '20' and kzt['known_amount'] == '5000'
    assert rub['thresholds']['M']['values'] != kzt['thresholds']['M']['values']
    assert window(result)['behavior']['frequency']['distribution'] == [[1, 2]]


@pytest.mark.parametrize('bad_lines,issue', [
    ((('F', 1, '10', 'unsupported'),), 'UNKNOWN_CURRENCY'),
    ((('F', 1, '10', 'offline1C'), ('F', 1, '20', 'kanzlerKz')), 'MIXED_CURRENCY'),
    ((('F', 1, None, 'offline1C'),), 'INVALID_PRICE'),
    ((('F', 1, '-1', 'offline1C'),), 'INVALID_PRICE'),
])
def test_unknown_money_keeps_behavior_and_known_subtotals_but_no_m_score(bad_lines, issue):
    result = aggregate([fact(order='known'), fact(order='bad', when='2026-08-31T12:00:00Z', lines=bad_lines)]).result()
    assert window(result)['behavior']['observed_orders'] == 2
    assert window(result)['behavior']['recency']['distribution'] == [[0, 1]]
    assert window(result)['quality'][issue] == window(result)['quality']['unvalued_orders'] == 1
    rub = currency(result)
    assert rub['known_amount'] == '10.25' and rub['monetary']['buyers'] == 0
    assert rub['thresholds']['M']['values'] == [] and rub['incomplete_money_buyers'] == 1
    assert all(row[2] is None for row in rub['score_counts'])
    assert dict((name, count) for name, count, _ in rub['segments'])['UNSCORABLE'] == 1


def test_unknown_currency_only_buyer_has_behavior_but_is_not_assigned_a_currency():
    result = aggregate([fact(lines=(('F', 1, '20', 'unsupported'),))]).result()
    assert policy(result)['unassigned_currency_buyers'] == 1
    assert window(result)['behavior']['observed_orders'] == 1
    assert all(value['buyers'] == 0 for value in window(result)['currencies'].values())


def test_completeness_is_per_window_and_conservative_across_currencies():
    result = aggregate([fact(order='bad-old', when='2025-08-31T12:00:00Z', lines=(('F', 1, None, 'offline1C'),)),
                        fact(order='new-rub'), fact(order='new-kzt', lines=(('F', 1, '100', 'kanzlerKz'),))]).result()
    assert currency(result)['monetary']['distribution'] == [['10.25', 1]]
    assert currency(result, unit='KZT')['monetary']['distribution'] == [['100', 1]]
    for unit in ('RUB', 'KZT'):
        assert currency(result, 'LAST_24M', unit)['monetary']['buyers'] == 0
        assert currency(result, 'LAST_24M', unit)['unavailable_monetary_buyers'] == 1
    assert policy(result)['comparison']['unreliable_monetary_buyers'] == 1


def test_real_zero_prices_are_scored_one_and_differ_from_unknown():
    result = aggregate([fact(lines=(('F', 10, '0', 'offline1C'),)),
                        fact(customer='bad', order='bad', lines=(('F', 1, None, 'offline1C'),))]).result()
    rub = currency(result)
    assert rub['monetary']['distribution'] == [['0', 1]]
    assert rub['thresholds']['M']['buyers'] == 1
    assert sorted((row[2] or 0) for row in rub['score_counts']) == [0, 1]


def test_canonical_aliases_are_resolved_by_caller_once():
    resolver = CustomerIdResolver([CustomerMergeRecord('merge', datetime(2026, 1, 1, tzinfo=timezone.utc),
                                                      'canonical', ('old', 'alias'))])
    result = aggregate([fact(resolver.resolve('old'), 'one'), fact(resolver.resolve('alias'), 'two')]).result()
    assert policy(result)['base_buyers'] == 1
    assert currency(result)['frequency']['distribution'] == [[2, 1]]


def test_empirical_nearest_ranks_keep_ties_and_zero_frequency_floor():
    counts = Counter({0: 6, 1: 3, 4: 1})
    cuts = empirical_thresholds(counts)
    assert cuts == (0, 0, 0, 1)
    assert [rfm_score(v, cuts) for v in (0, 1, 4)] == [1, 4, 5]
    records = [fact(str(i), str(i), '2023-01-01T12:00:00Z') for i in range(6)]
    records += [fact(str(i), str(i)) for i in range(6, 9)]
    records += [fact('heavy', str(i)) for i in range(4)]
    result = aggregate(records).result()
    rub = currency(result)
    assert rub['thresholds']['F']['values'] == list(cuts)
    scored = Counter()
    for _, fs, _, n in rub['score_counts']:
        scored[fs] += n
    assert scored == Counter({1: 6, 4: 3, 5: 1})


@pytest.mark.parametrize('value', [0, 1, 100, Decimal('25.50')])
def test_all_equal_values_receive_the_same_score(value):
    cuts = empirical_thresholds(Counter({value: 25}))
    assert cuts == (value,) * 4
    assert rfm_score(value, cuts) == 1
    assert rfm_score(value, cuts, recency=True) == 5


def test_recency_score_is_monotone_and_input_order_does_not_change_result():
    cuts = empirical_thresholds(Counter(range(1, 11)))
    assert [rfm_score(v, cuts, recency=True) for v in range(11)] == sorted(
        (rfm_score(v, cuts, recency=True) for v in range(11)), reverse=True)
    records = [fact(str(i % 5), str(i), f'202{2+i%5}-{1+i%8:02d}-15T12:00:00Z',
                    (('F' if i % 2 else 'CP', 1, str(i*10), 'offline1C' if i % 3 else 'kanzlerKz'),)) for i in range(30)]
    expected = aggregate(records).result()
    random.Random(42).shuffle(records)
    assert aggregate(records).result() == expected
    assert aggregate(reversed(records)).result() == expected


@pytest.mark.parametrize('r,f,m,frequency,expected', [
    (5, 5, 5, 5, 'CHAMPIONS'), (3, 5, 1, 5, 'LOYAL'), (5, 5, 1, 5, 'LOYAL'),
    (5, 1, 1, 1, 'NEW_OR_RECENT'), (5, 5, 5, 1, 'NEW_OR_RECENT'),
    (1, 4, 1, 3, 'AT_RISK'), (1, 1, 1, 0, 'HIBERNATING'),
    (3, 2, 2, 2, 'OTHER'), (5, 5, None, 5, 'UNSCORABLE'),
    (None, 5, 5, 5, 'UNSCORABLE'), (5, None, 5, 5, 'UNSCORABLE'),
])
def test_segment_priority_table(r, f, m, frequency, expected):
    assert rfm_segment(r, f, m, frequency) == expected


def test_segments_partition_every_valid_score_tuple_and_missing_m_is_always_unscorable():
    observed = Counter(rfm_segment(r, f, m, n) for r, f, m, n in product(range(1, 6), range(1, 6), range(1, 6), (0, 1, 2, 10)))
    assert sum(observed.values()) == 500 and set(observed) == set(SEGMENTS)-{'UNSCORABLE'}
    for r, f in product(range(1, 6), repeat=2):
        assert rfm_segment(r, f, None, 10) == 'UNSCORABLE'


@pytest.mark.parametrize('coverage,status', [
    (FULL, 'FULL'),
    ([('2022-09-01T00:00:00Z', '2026-02-01T00:00:00Z'),
      ('2026-03-01T00:00:00Z', '2026-09-01T00:00:00Z')], 'PARTIAL'),
    ([('2026-01-01T00:00:00Z', '2026-09-01T00:00:00Z')], 'PARTIAL'),
    ([('2023-01-01T00:00:00Z', '2024-01-01T00:00:00Z')], 'NONE'),
    ([], 'NONE'), (None, 'NONE'),
])
def test_window_coverage_and_no_final_scores_for_partial_or_missing_history(coverage, status):
    records = [fact()] if status != 'NONE' else []
    result = aggregate(records, coverage).result()
    assert result['windows']['LAST_12M']['status'] == status
    rub = currency(result)
    if status != 'FULL':
        assert result['windows']['LAST_12M']['reason']
        assert rub['confirmed_inactive_buyers'] is None
        assert all(row[:3] == [None, None, None] for row in rub['score_counts'])
        assert all(not value['values'] for value in rub['thresholds'].values())
        assert rub['monetary']['buyers'] == 0
    else:
        assert result['windows']['LAST_12M']['reason'] is None


def test_covered_short_window_does_not_prove_recency_across_an_earlier_gap():
    coverage = [('2022-09-01T00:00:00Z', '2023-01-02T00:00:00Z'),
                ('2025-09-01T00:00:00Z', '2026-09-01T00:00:00Z')]
    result = aggregate([fact(when='2023-01-01T12:00:00Z')], coverage).result()
    rub = currency(result)
    assert result['windows']['LAST_12M']['status'] == 'FULL'
    assert rub['confirmed_inactive_buyers'] == 1 and rub['monetary']['distribution'] == [['0', 1]]
    assert rub['reliable_recency_buyers'] == 0
    assert rub['score_counts'] == [[None, 1, 1, 1]]
    assert dict((s, n) for s, n, _ in rub['segments'])['UNSCORABLE'] == 1


def test_coverage_is_merged_normalized_to_utc_and_full_after_latest_purchase():
    coverage = [('2025-09-01T03:00:00+03:00', '2026-01-01T00:00:00Z'),
                ('2026-01-01T00:00:00Z', '2026-09-01T00:00:00Z')]
    result = aggregate([fact()], coverage).result()
    assert result['windows']['LAST_12M']['status'] == 'FULL'
    assert result['windows']['LAST_48M']['status'] == 'PARTIAL'
    assert currency(result)['reliable_recency_buyers'] == 1
    assert currency(result)['score_counts'][0][:3] == [5, 1, 1]


def test_observed_empty_short_window_with_partial_coverage_does_not_mean_zero_monetary():
    coverage = [('2023-01-01T00:00:00Z', '2023-01-02T00:00:00Z'),
                ('2026-01-01T00:00:00Z', '2026-09-01T00:00:00Z')]
    rub = currency(aggregate([fact(when='2023-01-01T12:00:00Z')], coverage).result())
    assert rub['frequency']['distribution'] == [[0, 1]]
    assert rub['confirmed_inactive_buyers'] is None
    assert rub['monetary']['distribution'] == [] and rub['unavailable_monetary_buyers'] == 1


def test_12_plus_previous_comparison_has_four_groups_exact_deltas_and_currency_bases():
    records = [fact('both', 'p1', '2025-01-01T12:00:00Z'), fact('both', 'p2'),
               fact('previous', 'p3', '2025-01-01T12:00:00Z'), fact('recent', 'p4'),
               fact('neither', 'p5', '2023-01-01T12:00:00Z')]
    result = aggregate(records).result()
    comparison = policy(result)['comparison']
    assert comparison['coverage_reliable']
    assert comparison['observed_groups'] == dict(BOTH=1, LAST_ONLY=1, PREVIOUS_ONLY=1, NEITHER=1)
    assert comparison['last_orders'] == comparison['previous_orders'] == 2
    assert comparison['order_change'] == comparison['order_change_percent'] == 0
    assert comparison['money']['RUB']['last_known_amount'] == comparison['money']['RUB']['previous_known_amount'] == '20.50'
    assert comparison['money']['KZT']['known_amount_change_percent'] is None
    assert comparison['unreliable_monetary_buyers'] == 0


def test_comparison_zero_denominators_and_missing_previous_coverage_are_explicit():
    result = aggregate([fact()], [('2025-09-01T00:00:00Z', '2026-09-01T00:00:00Z')]).result()
    comparison = policy(result)['comparison']
    assert not comparison['coverage_reliable'] and comparison['unreliable_monetary_buyers'] == 1
    assert comparison['order_change_percent'] is None
    assert comparison['money']['RUB']['known_amount_change_percent'] is None
    assert comparison['money']['RUB']['unreliable_buyers'] == 1


@pytest.mark.parametrize('kwargs', [dict(identity_space=''), dict(identity_space=' INDEXED'),
                                  dict(policies=()), dict(policies=('unknown',)),
                                  dict(policies=(POLICIES[0],)*2), dict(policies=POLICIES[0])])
def test_invalid_accumulator_configuration_is_rejected(kwargs):
    options = dict(identity_space=SPACE)
    options.update(kwargs)
    with pytest.raises(ValueError):
        RfmAggregates(AS_OF, FULL, **options)


def test_reference_ids_and_local_indexed_sids_cannot_be_mixed():
    accumulator = aggregate([fact()])
    before = accumulator.result()
    with pytest.raises(ValueError, match='identity space'):
        accumulator.add(fact(customer=1, space='INDEXED:generation'))
    assert accumulator.result() == before
    indexed = RfmAggregates(AS_OF, FULL, identity_space='INDEXED:generation', policies=(POLICIES[0],))
    indexed.add(fact(customer=1, space='INDEXED:generation'))
    assert indexed.result()['policies'] == {POLICIES[0]: policy(before)}


def test_observations_outside_coverage_fail_without_mutation():
    accumulator = aggregate(coverage=[])
    before = accumulator.result()
    with pytest.raises(ValueError, match='coverage'):
        accumulator.add(fact())
    assert accumulator.result() == before


def test_cancellation_before_add_and_during_finalization_is_safe_and_retryable():
    enabled = False
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if enabled and calls % 3 == 0:
            raise InterruptedError('cancelled')
    accumulator = aggregate([fact()], check=check)
    expected = accumulator.result()
    enabled = True
    calls = 2
    with pytest.raises(InterruptedError):
        accumulator.add(fact(customer='must-not-appear'))
    calls = 0
    with pytest.raises(InterruptedError):
        accumulator.result()
    enabled = False
    assert accumulator.result() == expected


def test_no_event_objects_or_identifiers_in_output_and_no_finalization_side_effects():
    accumulator = aggregate(fact(order=f'secret-order-{i}') for i in range(100))
    before = deepcopy(accumulator._buyers)
    result = accumulator.result()
    assert accumulator._buyers == before and accumulator.result() == result
    payload = json.dumps(result, allow_nan=False)
    assert all(value not in payload for value in ('secret-buyer', 'secret-order', 'secret-product', SPACE))
    assert all(not hasattr(buyer, '__dict__') for buyers in accumulator._buyers.values() for buyer in buyers.values())
    assert all(len(buyer.windows) == 4 for buyers in accumulator._buyers.values() for buyer in buyers.values())
    result['windows']['LAST_12M']['available'].clear()
    assert accumulator.result()['windows']['LAST_12M']['available']


def test_import_and_calculation_need_no_qt_raw_io_database_or_new_fact_build():
    script = '''
import os, sys
from datetime import date
def guard(event,args):
    if event == 'open':
        mode,flags=args[1:3]
        if isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
            raise AssertionError('Unexpected write')
    if event in ('os.mkdir','sqlite3.connect','socket.connect'):
        raise AssertionError('Unexpected I/O')
sys.addaudithook(guard)
from Application.statistics_rfm_core import RfmAggregates
result=RfmAggregates(date(2026,8,31),[],identity_space='synthetic').result()
assert not any(k.startswith(('PyQt','qt_material')) for k in sys.modules)
assert result['policies']['STRICT_COMPLETED']['base_buyers'] == 0
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_nonempty_calculation_reuses_prepared_facts_without_io_or_classification(monkeypatch):
    import builtins
    from Application import statistics_purchase_facts, statistics_purchase_policy
    records = [fact(), fact(order='second', lines=(('CP', 5, '20', 'offline1C'),))]
    accumulator = aggregate()
    def forbidden(*args, **kwargs):
        raise AssertionError('RFM attempted I/O, adaptation or reclassification')
    with monkeypatch.context() as patch:
        patch.setattr(builtins, 'open', forbidden)
        patch.setattr(Path, 'open', forbidden)
        patch.setattr(statistics_purchase_facts, 'build_purchase_order_fact', forbidden)
        patch.setattr(statistics_purchase_facts, 'eligible_purchase_policies', forbidden)
        patch.setattr(statistics_purchase_policy, 'classify_status', forbidden)
        for record in records:
            accumulator.add(record)
        result = accumulator.result()
    assert window(result)['behavior']['observed_orders'] == 1
    assert window(result, index=1)['behavior']['observed_orders'] == 2
