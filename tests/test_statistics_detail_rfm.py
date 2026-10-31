"""Personal RFM parity on a complete synthetic population, never real history."""

from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
import hashlib
import json
import sqlite3

import pytest

from Application.statistics_detail_contract import DetailMetadata, DetailPeriod, RfmAvailability
from Application.statistics_detail_rfm import DetailRfmStore
from Application.statistics_rfm_core import RfmAggregates, RfmCustomerState, empirical_thresholds, empirical_ranks, POLICIES
from tests.test_statistics_rfm_core import fact


AS_OF = date(2026,8,31)
SPACE = 'REFERENCE:synthetic-rfm'
FULL = [('2020-01-01T00:00:00Z','2026-09-01T00:00:00Z')]
GAP = [('2020-01-01T00:00:00Z','2026-01-01T00:00:00Z'),('2026-01-10T00:00:00Z','2026-09-01T00:00:00Z')]
GOLDEN = {'full': '7440e184f0b6a9eaca30722faf5b1e31efd983b895815e60f2dc9a56378f6ae0', 'gap': 'a165cde353dbaa147ef50a5af4c48f854373452395e7e594f65c4cf4420489a5'}


def population():
    records = []
    for i in range(18):
        customer = 'synthetic-'+str(i)
        records.append(fact(customer,str(i)+'a','2025-03-15T12:00:00Z',
                            lines=((('CP' if i%5==0 else 'F'),1,str(i%6),'offline1C'),)))
        if i%3:
            records.append(fact(customer,str(i)+'b','2026-02-15T12:00:00Z',
                                lines=(('sold',2,None if i%4==0 else str(i+1),'kanzlerKz' if i%2 else 'offline1C'),)))
        if i%4==0:
            records.append(fact(customer,str(i)+'c','2026-08-20T12:00:00Z',lines=(('F',1,'2','offline1C'),)))
    records += [fact('outside','outside','2020-01-01T12:00:00Z'),
                fact('unknown','unknown','2026-02-15T12:00:00Z',lines=(('F',1,'7','missing'),)),
                fact('return-only','return-only','2026-02-15T12:00:00Z',lines=(('Return',1,'7','offline1C'),))]
    return records


def aggregate(coverage):
    core=RfmAggregates(AS_OF,coverage,identity_space=SPACE)
    for f in population():
        core.add(f)
    return core.result()


@pytest.mark.parametrize('coverage',[FULL,GAP])
def test_aggregate_result_matches_pre_change_golden(coverage):
    data=json.dumps(aggregate(coverage),sort_keys=True,separators=(',',':'))
    assert hashlib.sha256(data.encode()).hexdigest()==GOLDEN['full' if coverage==FULL else 'gap']


@pytest.mark.parametrize('coverage',[FULL,GAP])
def test_disk_thresholds_personal_scores_segments_and_raw_metrics_match_core(coverage):
    metadata=DetailMetadata(AS_OF,DetailPeriod(),SPACE,coverage)
    grouped=defaultdict(list)
    for f in population():
        grouped[f.customer_id].append(f)
    with sqlite3.connect(':memory:') as db:
        store=DetailRfmStore(db,metadata)
        for customer,records in grouped.items():
            state=RfmCustomerState(AS_OF,coverage,identity_space=SPACE)
            for f in records:
                state.add(f)
            store.add_customer(customer,state.observations())
        cuts=store.finalize()
        expected=aggregate(coverage)
        counts=defaultdict(Counter)
        scores=defaultdict(Counter)
        recencies=defaultdict(Counter)
        frequencies=defaultdict(Counter)
        money=defaultdict(Counter)
        for customer in grouped:
            before=db.total_changes
            personal=store.personal(customer)
            assert db.total_changes==before
            assert len(personal)==12 and all(p.status is not RfmAvailability.NOT_COMPUTED for p in personal)
            for p in personal:
                key=p.policy,p.window,p.currency
                if p.status in (RfmAvailability.OUTSIDE_BASE,RfmAvailability.NOT_IN_CURRENCY_BASE):
                    assert p.segment is p.monetary is None
                    continue
                counts[key][p.segment]+=1
                scores[key][p.r_score,p.f_score,p.m_score]+=1
                recencies[key][p.r_days]+=1
                frequencies[key][p.f_orders]+=1
                if p.monetary is not None:
                    money[key][p.monetary]+=1
                if p.status is RfmAvailability.UNSCORABLE:
                    assert None in (p.r_score,p.f_score,p.m_score)
        for policy in POLICIES:
            for window in ('LAST_12M','LAST_24M','LAST_48M'):
                for currency in ('RUB','KZT'):
                    key=policy,window,currency
                    original=expected['policies'][policy]['windows'][window]['currencies'][currency]
                    for component in ('R','F','M'):
                        values=original['thresholds'][component]['values']
                        assert list(cuts[key][component]['values'])==([Decimal(v) for v in values] if component=='M' else values)
                        assert cuts[key][component]['buyers']==original['thresholds'][component]['buyers']
                    assert counts[key]==Counter({segment:n for segment,n,_ in original['segments'] if n})
                    assert scores[key]==Counter({tuple(row[:3]):row[3] for row in original['score_counts']})
                    assert recencies[key]==Counter(dict(original['recency']['distribution']))
                    assert frequencies[key]==Counter(dict(original['frequency']['distribution']))
                    assert money[key]==Counter({Decimal(v):n for v,n in original['monetary']['distribution']})


def test_weighted_disk_nearest_ranks_keep_numeric_decimal_ties_and_exact_order():
    from Application.statistics_rfm_core import RfmObservation
    metadata=DetailMetadata(AS_OF,DetailPeriod(),SPACE,FULL)
    values=['2','10','2.00','100000000000000000000.00000001','10.0','0']
    with sqlite3.connect(':memory:') as db:
        store=DetailRfmStore(db,metadata)
        for i,value in enumerate(values):
            store.add_customer(str(i),(RfmObservation(POLICIES[0],'LAST_12M','RUB',True,True,True,10,True,1,Decimal(value)),))
        result=store.finalize()[POLICIES[0],'LAST_12M','RUB']['M']
        assert result['values']==empirical_thresholds(Counter(Decimal(v) for v in values))
        assert result['buyers']==len(values)
        plan=db.execute("EXPLAIN QUERY PLAN SELECT value,n FROM rfm_histogram WHERE policy=? AND window=? AND currency=? AND component=? ORDER BY value COLLATE DETAIL_DECIMAL",(POLICIES[0],'LAST_12M','RUB','M')).fetchall()
        assert not any('TEMP B-TREE' in row[-1] for row in plan)


def test_customer_state_rejects_multiple_identities_and_result_is_read_only():
    state=RfmCustomerState(AS_OF,FULL,identity_space=SPACE)
    state.add(fact('a'))
    before=state.observations()
    with pytest.raises(ValueError):
        state.add(fact('b'))
    assert state.observations()==before
    assert empirical_ranks(5)==(1,2,3,4)
    assert empirical_ranks(0)==()
    with pytest.raises(ValueError):
        empirical_ranks(-1)
