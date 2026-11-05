"""Monetary cohorts over synthetic prepared facts; no real history/index access."""
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal, getcontext, localcontext
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from Application import statistics_cache as cache
from Application.mindbox.records import ProductKey
from Application.statistics_cohorts_core import HORIZONS, POLICIES
from Application.statistics_cohorts_money import BuyerMoney
from Application.statistics_cohorts_presentation import cohort_sections, MONEY_LEGACY_TEXT, MONEY_TITLES
from Application.statistics_purchase_facts import build_purchase_order_fact
from tests.test_statistics_cohorts_core import aggregate, SPACE
from tests.test_statistics_cohorts_integration import dataset, calculate

__all__ = ['dataset']


def order(day='2024-01-01', *, id=1, buyer='private-buyer', amount='10', namespace='offline1C', lines=None):
    rows = lines if lines is not None else [(amount,namespace)]
    return build_purchase_order_fact(
        [SimpleNamespace(line_status='F',quantity=Decimal(5),price_of_line=Decimal(a) if isinstance(a,str) else a,
                         product=ProductKey(n,'private-product')) for a,n in rows],
        identity_space=SPACE,customer_id=buyer,order_id=id,
        order_datetime_utc=datetime.fromisoformat(day+'T12:00:00+00:00' if len(day)==10 else day))


def monetary(records, **kwargs):
    return aggregate(records,**kwargs).result()['policies'][POLICIES[0]]['monetary']


def month(records, **kwargs):
    return monetary(records,**kwargs)['cohorts']['2024-01']


def test_first_second_and_all_later_orders_separate_currencies_and_keep_after_365_days():
    rows = [order(amount='10'),order('2024-01-11',id=2,amount='20',namespace='kanzlerKz'),
            order('2024-03-01',id=3,amount='30'),order('2025-03-01',id=4,amount='40')]
    value = monetary(rows)
    cohort = value['cohorts']['2024-01']
    assert (cohort['cohort_size'],cohort['orders'],cohort['unvalued_orders']) == (1,4,0)
    rub,kzt = (cohort['currencies'][c] for c in ('RUB','KZT'))
    assert rub == dict(known_orders=3,amount='80',mean_order=str(Decimal(80)/3),
                      first_orders=1,first_amount='10',repeat_orders=2,repeat_amount='70',repeat_amount_rate=87.5)
    assert kzt['first_orders']==0 and kzt['repeat_amount']=='20' and kzt['repeat_amount_rate']==100
    for h in HORIZONS:
        second = value['horizons'][str(h)]
        assert (second['eligible'],second['second_buyers'],second['unvalued_orders']) == (1,1,0)
        assert second['currencies']['RUB']['known_orders']==0
        assert second['currencies']['KZT']==dict(known_orders=1,amount='20',mean_order='20')


@pytest.mark.parametrize('sequence', list(permutations(range(4))))
def test_late_first_and_second_simultaneous_int_before_string_id_order(sequence):
    rows = [order(id='1',amount='40'),order(id=2,amount='20'),order(id=1,amount='10'),
            order('2025-01-02',id=3,amount='30')]
    value = monetary([rows[i] for i in sequence])
    assert value == monetary(rows)
    row = value['cohorts']['2024-01']['currencies']['RUB']
    assert row['first_amount']=='10' and row['repeat_amount']=='90'
    assert value['horizons']['30']['currencies']['RUB']['amount']=='20'


def test_zero_is_valued_and_missing_currency_or_price_is_not_zero():
    rows = [order(amount='0'), order(id=2,amount=None),order(id=3,namespace='unsupported'),
            order(id=4,lines=[('20','offline1C'),('30','kanzlerKz')]),
            order(id=5,amount='-1'),order(id=6,amount='0',namespace='kanzlerKz'),
            order(id=1,buyer='only-unvalued',amount=None)]
    value = monetary(rows)
    row = value['cohorts']['2024-01']
    assert row['orders']==7 and row['cohort_size']==2 and row['unvalued_orders']==5
    for c in row['currencies'].values():
        assert c['known_orders']==1 and Decimal(c['mean_order'])==0 and c['repeat_amount_rate'] is None
    assert value['horizons']['30']['unvalued_orders']==1
    assert value['horizons']['30']['second_buyers']==1
    empty = month([order(amount=None)])
    assert all(c['known_orders']==0 and c['mean_order'] is None and c['repeat_amount_rate'] is None
               for c in empty['currencies'].values())


@pytest.mark.parametrize('days',HORIZONS)
def test_inclusive_second_purchase_day_boundary_and_cumulative_shorter_horizons(days):
    first = date(2024,1,1)
    rows = [order(), order((first+timedelta(days=days)).isoformat(),id=2,amount='21'),
            order((first+timedelta(days=days+1)).isoformat(),id=3,amount='999')]
    value = monetary(rows)
    for h in HORIZONS:
        c = value['horizons'][str(h)]
        expected = int(h>=days)
        assert c['second_buyers']==expected
        assert c['currencies']['RUB']['known_orders']==expected
        assert Decimal(c['currencies']['RUB']['amount'])==21*expected


@pytest.mark.parametrize('gap',[False,True])
def test_observed_second_orders_outside_eligible_are_not_in_money_base(gap):
    rows = [order('2024-01-01'),order('2024-01-03',id=2,amount='100')]
    coverage = [('2024-01-01T00:00:00Z','2024-01-02T00:00:00Z'),
                ('2024-01-03T00:00:00Z','2025-01-01T00:00:00Z')] if gap else [
                ('2024-01-01T00:00:00Z','2025-01-01T00:00:00Z')]
    core = aggregate(rows,coverage,as_of=date(2024,1,10))
    p = core.result()['policies'][POLICIES[0]]
    for h,c in p['monetary']['horizons'].items():
        old = p['summary']['horizons'][h]
        assert old['observed_repeated']==1 and old['repeated']==0
        assert c['eligible']==c['second_buyers']==c['unvalued_orders']==0
        assert c['currencies']['RUB']['known_orders']==0
    # Observed monthly money still includes both orders and states the known part.
    assert p['monetary']['cohorts']['2024-01']['currencies']['RUB']['amount']=='110'


def test_changing_eligible_population_does_not_require_monotonic_horizon_money():
    rows = [order('2024-01-01',buyer='mature'),order('2024-01-02',buyer='mature',id=2),
            order('2024-06-01',buyer='young'),order('2024-06-02',buyer='young',id=2,amount='100')]
    value = monetary(rows,as_of=date(2024,7,15))
    assert value['horizons']['30']['currencies']['RUB']['amount']=='110'
    assert value['horizons']['60']['currencies']['RUB']['amount']=='10'


def test_single_order_first_is_global_not_first_in_currency_or_display_window():
    rows = [order('2023-12-31',amount='10'),order('2024-01-01',id=2,amount='30',namespace='kanzlerKz'),
            order('2024-01-01',id=1,buyer='single',amount='5')]
    value = monetary(rows,cohort_start='2024-01',cohort_end='2024-01')
    assert list(value['cohorts'])==['2024-01']
    row = value['cohorts']['2024-01']
    assert row['cohort_size']==row['orders']==1
    assert row['currencies']['RUB']['first_amount']=='5'
    assert row['currencies']['RUB']['repeat_amount']=='0'
    assert row['currencies']['KZT']['known_orders']==0


def test_exact_sums_are_arrival_independent_and_leave_decimal_context_untouched():
    rows = [order(id=i+1) for i in range(4)]
    amounts = ['123456789012345678901234567890.12','0.01','0.02','1E-12']
    # Fact construction already follows its own established Decimal contract.
    # Supply exact prepared fact amounts to isolate this accumulator.
    rows = [replace(f,strict=replace(f.strict,amount=Decimal(a))) for f,a in zip(rows,amounts)]
    context = getcontext().copy()
    with localcontext(context) as current:
        current.clear_flags()
        before = current.copy()
        result = monetary(rows)
        assert current.flags==before.flags and current.prec==before.prec
        for sequence in permutations(rows):
            assert monetary(sequence)==result
    rub = result['cohorts']['2024-01']['currencies']['RUB']
    assert rub['amount']=='123456789012345678901234567890.150000000001'
    assert rub['first_amount']==amounts[0]
    assert rub['repeat_amount']=='0.030000000001'


def test_fixed_size_state_results_are_fresh_and_cutoff_still_filters():
    core = aggregate([order(),order('2027-01-01',id=2,amount='999')])
    buyer = core._buyers[POLICIES[0]]['private-buyer']
    assert isinstance(buyer.money,BuyerMoney) and not hasattr(buyer.money,'__dict__')
    assert buyer.money.orders==1
    assert all(not isinstance(getattr(buyer.money,name),(list,dict,set))
               for cls in type(buyer.money).__mro__ for name in getattr(cls,'__slots__',()))
    before = core.result()
    first = core.result()
    first['policies'][POLICIES[0]]['monetary']['cohorts'].clear()
    assert core.result()==before
    assert 'monetary' not in before['policies'][POLICIES[1]]


@pytest.mark.parametrize('damage',[
    'extra_id','currency','negative','nan','bool','orders','size','known','first','repeat','sum',
    'rate','mean','second_base','second_count','second_sum','second_unknown','missing_month','extended'])
def test_closed_monetary_dto_rejects_inconsistent_or_private_data(dataset,damage):
    value = calculate(dataset)
    policy = value['cohort_statistics']['policies'][POLICIES[0]]
    block = policy['monetary']
    row = block['cohorts']['2026-01']
    rub = row['currencies']['RUB']
    second = block['horizons']['60']
    if damage=='extra_id': rub['order_id']='private'
    elif damage=='currency': row['currencies']['USD']=deepcopy(rub)
    elif damage=='negative': rub['amount']='-1'
    elif damage=='nan': rub['amount']='NaN'
    elif damage=='bool': row['orders']=True
    elif damage=='orders': row['orders']+=1
    elif damage=='size': row['cohort_size']+=1
    elif damage=='known': rub['known_orders']+=1
    elif damage=='first': rub['first_orders']+=1
    elif damage=='repeat': rub['repeat_orders']+=1
    elif damage=='sum': rub['repeat_amount']='99999'
    elif damage=='rate': rub['repeat_amount_rate']=101
    elif damage=='mean': rub['mean_order']='99999'
    elif damage=='second_base': second['eligible']+=1
    elif damage=='second_count': second['second_buyers']+=1
    elif damage=='second_sum': second['currencies']['RUB']['amount']='99999'
    elif damage=='second_unknown': second['unvalued_orders']+=1
    elif damage=='missing_month': block['cohorts'].clear()
    elif damage=='extended': value['cohort_statistics']['policies'][POLICIES[1]]['monetary']=block
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(value)


@pytest.mark.parametrize('legacy',[False,True])
def test_aggregate_only_cache_roundtrip_and_explicit_legacy_message(dataset,tmp_path,legacy):
    value = calculate(dataset)
    policy = value['cohort_statistics']['policies'][POLICIES[0]]
    if legacy:
        policy.pop('monetary')
    else:
        assert all(key not in json.dumps(policy['monetary']) for key in ('private','secret','customer_id','order_id'))
    path = tmp_path/'money-cohorts.json'
    cache.save_result(path,value)
    saved = cache.load_result(path)
    assert saved==json.loads(json.dumps(value))
    sections = cohort_sections(saved,compact=True)
    assert [s[0] for s in sections[3:5]]==list(MONEY_TITLES)
    assert all(len(s[2])==2 for s in sections[3:5])
    if legacy:
        assert all(t[1][0][1]==MONEY_LEGACY_TEXT for s in sections[3:5] for t in s[2])


def test_unknown_first_does_not_promote_first_valued_order_to_first_purchase():
    value = monetary([order(id=2,amount='0',namespace='kanzlerKz'),order(id=1,amount=None)])
    row = value['cohorts']['2024-01']
    assert row['unvalued_orders']==1
    kzt = row['currencies']['KZT']
    assert kzt['first_orders']==0 and kzt['repeat_orders']==1
    second = value['horizons']['30']
    assert second['unvalued_orders']==0 and second['currencies']['KZT']['known_orders']==1
    assert second['currencies']['KZT']['mean_order']=='0'


def test_large_decimal_presentation_rounds_without_context_or_saved_value_changes():
    from Application.statistics_cohorts_presentation import cohort_display
    value = Decimal('123456789012345678901234567890.125')
    with localcontext() as context:
        context.clear_flags()
        before = context.copy()
        assert cohort_display(value)=='123 456 789 012 345 678 901 234 567 890.13'
        assert context.flags==before.flags and value==Decimal('123456789012345678901234567890.125')
