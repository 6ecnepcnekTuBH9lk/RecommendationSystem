"""Segment money from existing RFM state: synthetic facts, no real history."""
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal, localcontext
import json

import pytest

from Application import statistics_cache as cache
from Application.statistics_rfm_core import POLICIES, SEGMENTS, WINDOWS
from Application.statistics_rfm_validation import validate_rfm_statistics
from tests.test_statistics_rfm_core import AS_OF, FULL, aggregate, currency, fact
from tests.test_statistics_rfm_integration import dataset, calculate, AT

__all__=['dataset']


def segment_population(namespace='offline1C'):
    groups=[('champ',5,1,10,'100'),('loyal',5,45,9,'0.2'),('new',10,0,1,'10'),
            ('risk',10,250,8,'30'),('hib',10,300,1,'5'),('other',10,100,2,'2'),('unsc',5,0,3,'10')]
    for token,n,age,f,price in groups:
        for i in range(n):
            for j in range(f):
                yield fact(token+str(i),f'{namespace}-{token}-{i}-{j}',
                           (AS_OF-timedelta(days=age)).isoformat()+'T12:00:00Z',
                           lines=(('F',1,None if token=='unsc' and j==2 else price,namespace),))


def check(result,coverage=FULL):
    validate_rfm_statistics(result,coverage=coverage,selected_end=AS_OF.isoformat(),calculated_at=AT)
    for name in WINDOWS[:3]:
        for unit in ('RUB','KZT'):
            c=currency(result,name,unit)
            rows=c['segment_business']
            assert [r['segment'] for r in rows]==list(SEGMENTS)
            for field in ('buyers','observed_orders','valued_orders'):
                assert sum(r[field] for r in rows)==c[field]
            assert sum(Decimal(r['known_amount']) for r in rows)==Decimal(c['known_amount'])
            assert sum(r['complete_m_buyers'] for r in rows)==c['monetary']['buyers']
            assert sum(Decimal(r['complete_m_amount']) for r in rows)==sum(
                Decimal(v)*n for v,n in c['monetary']['distribution'])
            assert [(r['segment'],r['buyers'],r['buyer_rate']) for r in rows]==[tuple(r) for r in c['segments']]
            for r in rows:
                a=Decimal(r['known_amount'])
                assert r['mean_known_order']==(str(a/r['valued_orders']) if r['valued_orders'] else None)
                assert r['mean_complete_m']==(str(Decimal(r['complete_m_amount'])/r['complete_m_buyers'])
                                               if r['complete_m_buyers'] else None)
                assert r['known_amount_rate']==(str(100*a/Decimal(c['known_amount'])) if Decimal(c['known_amount']) else None)
                for key in ('known_amount','known_amount_rate','mean_known_order','complete_m_amount','mean_complete_m'):
                    assert r[key] is None or isinstance(r[key],str)
    for c in result['policies'][POLICIES[1]]['windows'].values():
        assert all('segment_business' not in v for v in c['currencies'].values())
    assert all('segment_business' not in c for c in result['policies'][POLICIES[0]]['windows']['PREVIOUS_12M']['currencies'].values())


@pytest.mark.parametrize('namespace,unit',[('offline1C','RUB'),('kanzlerKz','KZT')])
def test_all_seven_existing_segments_three_windows_and_currency_totals(namespace,unit):
    result=aggregate(segment_population(namespace)).result()
    check(result)
    expected=[5,5,10,10,10,10,5]
    for name in WINDOWS[:3]:
        rows=currency(result,name,unit)['segment_business']
        assert [r['buyers'] for r in rows]==expected
        assert [r['observed_orders'] for r in rows]==[50,45,10,80,10,20,15]
        assert [Decimal(r['known_amount']) for r in rows]==[Decimal(v) for v in ('5000','9.0','100','2400','50','40','100')]
        assert rows[-1]['complete_m_buyers']==0
        assert rows[-1]['valued_orders']==10  # Known subtotal is retained for unscorable buyers.
        assert rows[1]['mean_known_order']=='0.2' and rows[1]['mean_complete_m']=='1.8'
        assert rows[2]['mean_known_order']==rows[2]['mean_complete_m']=='10'


@pytest.mark.parametrize('issue',['bad_rub','bad_kzt','unknown','mixed'])
def test_dual_currency_unknown_or_unvalued_order_taints_both_complete_m_only(issue):
    records=[fact(order='rub',lines=(('F',1,'12.30','offline1C'),)),
             fact(order='kzt',lines=(('F',1,'100','kanzlerKz'),))]
    lines={'bad_rub':(('F',1,None,'offline1C'),),'bad_kzt':(('F',1,None,'kanzlerKz'),),
           'unknown':(('F',1,'7','missing'),),
           'mixed':(('F',1,'7','offline1C'),('F',1,'8','kanzlerKz'))}[issue]
    result=aggregate(records+[fact(order='unvalued',lines=lines)]).result()
    check(result)
    assert result['policies'][POLICIES[0]]['dual_currency_buyers']==1
    for unit,amount in [('RUB','12.30'),('KZT','100')]:
        c=currency(result,unit=unit);r=c['segment_business'][-1]
        assert c['buyers']==r['buyers']==1 and r['complete_m_buyers']==0
        assert r['known_amount']==amount and r['mean_known_order']==amount
        assert r['observed_orders']==1+(issue=='bad_rub' and unit=='RUB' or issue=='bad_kzt' and unit=='KZT')
        assert r['known_amount_rate']=='100'


@pytest.mark.parametrize('price',['0',None,'0.123456789123456789'])
def test_zero_missing_and_precise_money(price):
    result=aggregate([fact(lines=(('F',1,price,'offline1C'),))]).result()
    check(result)
    c=currency(result);r=next(r for r in c['segment_business'] if r['buyers'])
    assert Decimal(r['known_amount'])==Decimal(price or 0)
    assert r['mean_known_order']==price
    assert r['valued_orders']==int(price is not None)
    assert r['known_amount_rate']==('100' if price not in (None,'0') else None)
    assert r['complete_m_buyers']==int(price is not None)


@pytest.mark.parametrize('coverage',[
    [('2025-09-02T00:00:00Z','2026-09-01T00:00:00Z')],
    [('2022-09-01T00:00:00Z','2026-01-01T00:00:00Z'),('2026-01-10T00:00:00Z','2026-09-01T00:00:00Z')],
])
def test_partial_window_preserves_known_sums_without_final_segments(coverage):
    result=aggregate(segment_population(),coverage=coverage).result()
    check(result,coverage)
    for name in WINDOWS[:3]:
        c=currency(result,name);rows=c['segment_business']
        assert not any(r['buyers'] for r in rows[:-1])
        assert rows[-1]['buyers']==55 and rows[-1]['known_amount']==c['known_amount']
        assert not any(r['complete_m_buyers'] for r in rows)


def test_inactive_currency_base_zero_m_and_unknown_recency_are_not_discarded():
    coverage=[('2022-09-01T00:00:00Z','2024-01-01T00:00:00Z'),
              ('2025-01-01T00:00:00Z','2026-09-01T00:00:00Z')]
    result=aggregate([fact(when='2023-01-01T12:00:00Z')],coverage=coverage).result()
    check(result,coverage)
    c=currency(result);r=c['segment_business'][-1]
    assert c['buyers']==r['buyers']==r['complete_m_buyers']==1
    assert r['observed_orders']==r['valued_orders']==0
    assert r['mean_known_order'] is None and r['mean_complete_m']=='0'
    assert c['reliable_recency_buyers']==0  # Full current M does not invent a reliable R.


def test_bad_order_outside_short_window_does_not_poison_current_m():
    result=aggregate([fact(order='now'),fact(order='old',when='2025-01-01T12:00:00Z',
                                            lines=(('F',1,None,'missing'),))]).result()
    check(result)
    assert sum(r['complete_m_buyers'] for r in currency(result)['segment_business'])==1
    assert currency(result,'LAST_24M')['segment_business'][-1]['complete_m_buyers']==0


def test_empty_population_order_independence_and_no_new_decimal_side_effects():
    result=aggregate().result();check(result)
    assert all(r['known_amount_rate'] is None for r in currency(result)['segment_business'])
    facts=list(segment_population())
    with localcontext() as context:
        context.clear_flags()
        core=aggregate(facts)
        result=core.result()
        # Existing result() already does divisions; compare a second finalization, not flag-free assumption.
        flags=context.flags.copy()
        assert core.result()==result and context.flags==flags
    assert aggregate(reversed(facts)).result()==result


@pytest.mark.parametrize('field,value',[
    ('customer_id','private'),('buyers',True),('buyers',99),('buyer_rate',0),
    ('observed_orders',0),('valued_orders',99),('known_amount','999999'),
    ('known_amount','NaN'),('known_amount',1.5),('known_amount_rate','-1'),
    ('known_amount_rate',None),('mean_known_order','999'),('complete_m_buyers',99),
    ('complete_m_amount','999'),('mean_complete_m','999'),('segment','NEW_SEGMENT'),
])
def test_new_aggregate_contract_rejects_wrong_counts_money_identity_and_segments(dataset,field,value):
    result=calculate(dataset)
    rows=currency(result['rfm_statistics'],'LAST_48M')['segment_business']
    row=next(r for r in rows if r['buyers'] and r['observed_orders'] and r['known_amount']!='0')
    row[field]=value
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


@pytest.mark.parametrize('where',['missing_one','extended','previous','swapped'])
def test_block_presence_policy_and_segment_order_are_closed(dataset,where):
    result=calculate(dataset);rfm=result['rfm_statistics']
    c=currency(rfm);rows=c['segment_business']
    if where=='missing_one':c.pop('segment_business')
    elif where=='extended':currency(rfm,index=1)['segment_business']=deepcopy(rows)
    elif where=='previous':currency(rfm,'PREVIOUS_12M')['segment_business']=deepcopy(rows)
    else:rows.reverse()
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


def test_legacy_dto_roundtrips_without_business_recalculation(dataset,tmp_path):
    result=calculate(dataset)
    for name in WINDOWS[:3]:
        for unit in ('RUB','KZT'):
            currency(result['rfm_statistics'],name,unit).pop('segment_business')
    path=tmp_path/'legacy.json';cache.save_result(path,result)
    assert cache.load_result(path)==json.loads(json.dumps(result))
    assert 'customer_id' not in json.dumps(result['rfm_statistics'])
    assert 'order_id' not in json.dumps(result['rfm_statistics'])
    assert Counter(row[0] for row in currency(result['rfm_statistics'])['segments'])==Counter(SEGMENTS)
