"""Observed purchase-channel identities, source parity and closed aggregate DTO."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import date
from itertools import permutations
import json

import pytest

from Application import statistics_cache as cache, statistics_slices as slices
from Application.analysis_filter import AnalysisFilter
from Application.statistics_cohorts_core import CohortAggregates, POLICIES
from Application.statistics_cohorts_presentation import cohort_sections, CHANNEL_LEGACY_TEXT, CHANNEL_TITLE
from tests.test_statistics_cohorts_core import FULL, AS_OF, SPACE
from tests.test_statistics_cohorts_money import order
from tests.test_statistics_cohorts_integration import dataset, calculate, SELECTION, AT
from tests.test_statistics_rfm_integration import raw_order
from tests.test_statistics_index import build, query
from tests.test_dataset_statistics import save, put_entry

__all__=['dataset']


def run(records=(), **kwargs):
    core=CohortAggregates(kwargs.pop('as_of',AS_OF),kwargs.pop('coverage',FULL),identity_space=SPACE,**kwargs)
    for fact,key,name in records:
        core.add(fact,channel_external_id=key,channel_name=name)
    return core


def channels(core):
    return core.result()['policies'][POLICIES[0]]['purchase_channels']


def record(*, buyer='buyer', id=1, day='2024-01-01', channel='A', name='Alpha'):
    return order(day,id=id,buyer=buyer),channel,name


def catalog(value):
    return {r['external_id']:r for r in value['channels']}


def assert_consistent(value):
    assert sum(r['first_buyers'] for r in value['channels'])==value['buyers']
    assert sum(r['second_buyers'] for r in value['channels'])==value['repeat_buyers']
    assert sum(r['buyers'] for r in value['transitions'])==value['repeat_buyers']
    assert value['same_known_buyers']+value['other_known_buyers']+value['unknown_transition_buyers']==value['repeat_buyers']
    by_first=Counter()
    for row in value['transitions']:
        by_first[row['first_external_id']]+=row['buyers']
    for row in value['transitions']:
        assert row['first_channel_repeat_rate']==pytest.approx(100*row['buyers']/by_first[row['first_external_id']])


def test_first_second_and_transitions_with_explicit_denominators_and_unknowns():
    core=run([
        record(buyer='same'),record(buyer='same',id=2),
        record(buyer='other'),record(buyer='other',id=2,channel='B',name='Beta'),
        record(buyer='missing_first',channel=None,name='Name must not be identity'),
        record(buyer='missing_first',id=2),
        record(buyer='missing_second'),record(buyer='missing_second',id=2,channel=None),
        record(buyer='both_missing',channel=None),record(buyer='both_missing',id=2,channel=None),
        record(buyer='single',channel='C',name='Single')])
    value=channels(core)
    assert_consistent(value)
    assert (value['buyers'],value['repeat_buyers'])==(6,5)
    assert (value['same_known_buyers'],value['other_known_buyers'],value['unknown_transition_buyers'])==(1,1,3)
    assert value['same_known_rate']==50
    c=catalog(value)
    assert c[None]['name']=='Не указан' and (c[None]['first_buyers'],c[None]['second_buyers'])==(2,2)
    assert c['C']['first_buyers']==1 and c['C']['second_buyers']==0
    assert c['A']['first_rate']==50 and c['A']['second_rate']==40
    pair=next(r for r in value['transitions'] if r['first_external_id']==r['second_external_id']=='A')
    assert pair['repeat_rate']==20 and pair['first_channel_repeat_rate']==pytest.approx(100/3)
    assert 'single' not in json.dumps(value) and 'order_id' not in json.dumps(value)


@pytest.mark.parametrize('sequence',list(permutations(range(4))))
def test_arbitrary_order_same_timestamps_and_late_first_preserve_money_and_first_two(sequence):
    rows=[record(id='1',channel='third',name='Ignored'),record(id=2,channel='B',name='Beta'),
          record(id=1,channel='A'),record(id=3,day='2024-02-01',channel='later')]
    expected=run(rows).result()
    actual=run([rows[i] for i in sequence]).result()
    assert actual==expected
    value=actual['policies'][POLICIES[0]]['purchase_channels']
    assert set(catalog(value))=={'A','B',None}
    assert value['transitions'][0]['first_external_id']=='A'
    assert value['transitions'][0]['second_external_id']=='B'


def test_different_identifiers_with_equal_names_remain_distinct_and_names_do_not_change_identity():
    value=channels(run([record(channel='A',name='Same'),record(id=2,channel='B',name='Same'),
                        record(buyer='rename',channel='C',name='Zulu'),
                        record(buyer='rename',id=2,channel='C',name='Alpha')]))
    assert value['same_known_buyers']==value['other_known_buyers']==1
    assert catalog(value)['C']['name']=='Alpha'


def test_empty_single_and_unknown_only_do_not_invent_retention_rate():
    for records,first,second in (([],0,0),([record(channel=None)],1,0),
            ([record(channel=None),record(id=2,channel=None)],1,1)):
        value=channels(run(records))
        assert_consistent(value)
        assert value['buyers']==first and value['repeat_buyers']==second and value['same_known_rate'] is None
        assert catalog(value)[None]['second_rate']==(100 if second else None)


def test_cohort_filter_cutoff_and_observed_ineligible_population_match_existing_cohorts():
    records=[record(day='2023-12-31'),record(id=2),record(buyer='visible'),
             record(buyer='visible',id=2,day='2024-01-02',channel='B'),
             record(buyer='visible',id=3,day='2024-01-11',channel='future')]
    core=run(records,cohort_start='2024-01',cohort_end='2024-01',as_of=date(2024,1,10),
             coverage=[('2024-01-01T00:00:00Z','2024-01-03T00:00:00Z')])
    value=channels(core)
    assert (value['buyers'],value['repeat_buyers'],value['other_known_buyers'])==(1,1,1)
    strict=core.result()['policies'][POLICIES[0]]
    assert strict['hidden_buyers']==1 and strict['summary']['horizons']['30']['repeated']==0
    assert set(catalog(value))=={'A','B',None}


def test_no_top_limit_sorting_and_fixed_buyer_state():
    records=[]
    for i in range(13):
        records.extend([record(buyer=str(i),channel=f'A{i:02}',name='Repeated name'),
                        record(buyer=str(i),id=2,channel='B')])
    core=run(records)
    value=channels(core)
    assert len(value['transitions'])==13 and len(catalog(value))==15
    assert [r['first_external_id'] for r in value['transitions']]==[f'A{i:02}' for i in range(13)]
    buyer=core._buyers[POLICIES[0]]['0']
    assert buyer.first_channel==('A00','Repeated name') and buyer.second_channel==('B','Alpha')
    assert not hasattr(buyer,'__dict__')
    assert all(not isinstance(getattr(buyer,k),(list,dict,set)) for k in ('first_channel','second_channel'))
    assert 'purchase_channels' not in core.result()['policies'][POLICIES[1]]
    before=core.result()
    core.result()['policies'][POLICIES[0]]['purchase_channels']['channels'].clear()
    assert core.result()==before


@pytest.mark.parametrize('key,name',[(True,'Name'),(1,'Name'),('','Name'),(' A','Name'),('A',{}),(None,3)])
def test_bad_prepared_channel_metadata_rejected_before_mutation(key,name):
    core=run()
    with pytest.raises(ValueError,match='Invalid purchase channel'):
        core.add(order(),channel_external_id=key,channel_name=name)
    assert core.result()['policies'][POLICIES[0]]['displayed_buyers']==0


def raw(key,day,buyer,channel,name='Alpha'):
    item=raw_order(key,day,buyer)
    item['firstAction']['channel']={'ids':{'externalId':channel},'name':name}
    return item


@pytest.fixture
def channel_dataset(dataset):
    root,_,source=dataset
    entry=source['manual_interactions']['orders']
    first=[raw('private-merge-first','2026-01-01','old','A'),
           raw('private-same-first','2026-01-02','same','S','Zulu'),
           raw('private-old-first','2020-01-01','historical','C','Old'),
           raw('private-unknown-first','2026-01-03','unknown',None,'Ignored'),
           raw('private-single','2026-01-05','single',None)]
    duplicate=deepcopy(first[1])
    duplicate['firstAction']['channel']={'ids':{'externalId':'WRONG'},'name':'Later conflicting snapshot'}
    second=[raw('private-merge-second','2026-02-01','new','B','Beta'),
            raw('private-same-second','2026-01-03','same','S','Alpha'),
            raw('private-third','2026-01-04','same','T','Third'),
            raw('private-old-second','2026-02-01','historical','D','New'),duplicate,
            raw('private-unknown-second','2026-01-04','unknown','A'),
            raw('private-future','2026-09-01','future','future')]
    directory=root/entry['directory']
    for i,rows in enumerate((first,second),1):
        (directory/f'orders_part_{i:03}.json').write_text(json.dumps({'orders':rows}),encoding='utf-8')
    save(root,source)
    return dataset


@pytest.mark.parametrize('selection',[SELECTION,AnalysisFilter(),AnalysisFilter('2026-02-01','2026-03-31')])
def test_reference_indexed_channels_preserved_merges_dedup_and_cohort_scope(channel_dataset,selection):
    value=calculate(channel_dataset,selection)
    p=value['cohort_statistics']['policies'][POLICIES[0]]
    c=p['purchase_channels']
    assert_consistent(c)
    assert 'WRONG' not in catalog(c) and 'T' not in catalog(c)
    if selection==SELECTION:
        assert 'future' not in catalog(c)
        assert (c['buyers'],c['repeat_buyers'],c['same_known_buyers'],c['other_known_buyers'],c['unknown_transition_buyers'])==(4,3,1,1,1)
        assert set(catalog(c))=={'A','B','S',None}
        assert c['same_known_rate']==50
    build(channel_dataset)
    assert asdict(query(channel_dataset,selection,calculated_at=AT))==value
    cache.validate_result(value)


def test_indexed_channel_survives_failure_of_unrelated_historical_aggregate_field(channel_dataset):
    root,_,source=channel_dataset
    first=raw('private-old','2020-01-01','historical','C','Old')
    first['payments']='invalid-unrelated-field'
    source['manual_interactions']=None
    source['actions']={'2026-08-31':put_entry(root,'actions',[],306,'2026-08-31T00:00:00Z','2026-09-01T00:00:00Z')}
    source['orders']={
        '2020-01-01':put_entry(root,'orders',[first],304,'2020-01-01T00:00:00Z','2020-01-02T00:00:00Z'),
        '2026-08-31':put_entry(root,'orders',[raw('private-second','2026-08-31','historical','D')],305,
                              '2026-08-31T00:00:00Z','2026-09-01T00:00:00Z')}
    # Only the later day is common; full-history cohorts still observe both.
    save(root,source)
    selected=AnalysisFilter()
    value=calculate(channel_dataset,selected)
    c=value['cohort_statistics']['policies'][POLICIES[0]]['purchase_channels']
    assert catalog(c)['C']['first_buyers']==1
    build(channel_dataset)
    assert asdict(query(channel_dataset,selected,calculated_at=AT))==value


@pytest.mark.parametrize('damage',['id','private','extra','bool','negative','buyers','repeat','same','other','unknown',
    'rate','first_rate','second_rate','first_count','second_count','duplicate','missing_unknown','unknown_name',
    'unknown_id','pair_id','pair_count','pair_rate','row_rate','pair_duplicate','pair_sort','catalog_sort','extended'])
def test_closed_channel_dto_rejects_individual_data_and_inconsistent_aggregates(channel_dataset,damage):
    value=calculate(channel_dataset)
    p=value['cohort_statistics']['policies'][POLICIES[0]]
    c=p['purchase_channels'];row=c['channels'][0];pair=c['transitions'][0]
    if damage=='id':row['customer_id']='private'
    elif damage=='private':pair['order_id']='private'
    elif damage=='extra':c['sequence']=[]
    elif damage=='bool':c['buyers']=True
    elif damage=='negative':row['first_buyers']=-1
    elif damage in ('buyers','repeat'):c['buyers' if damage=='buyers' else 'repeat_buyers']+=1
    elif damage in ('same','other','unknown'):c[{'same':'same_known_buyers','other':'other_known_buyers','unknown':'unknown_transition_buyers'}[damage]]+=1
    elif damage=='rate':c['same_known_rate']=100
    elif damage=='first_rate':row['first_rate']=999
    elif damage=='second_rate':row['second_rate']=999
    elif damage=='first_count':row['first_buyers']+=1
    elif damage=='second_count':row['second_buyers']+=1
    elif damage=='duplicate':c['channels'].append(deepcopy(row))
    elif damage=='missing_unknown':c['channels'].pop()
    elif damage=='unknown_name':c['channels'][-1]['name']='Name is not identity'
    elif damage=='unknown_id':row['external_id']=' A'
    elif damage=='pair_id':pair['second_external_id']='absent'
    elif damage=='pair_count':pair['buyers']+=1
    elif damage=='pair_rate':pair['repeat_rate']=999
    elif damage=='row_rate':pair['first_channel_repeat_rate']=0
    elif damage=='pair_duplicate':c['transitions'].append(deepcopy(pair))
    elif damage=='pair_sort':c['transitions'].reverse()
    elif damage=='catalog_sort':c['channels'].reverse()
    elif damage=='extended':value['cohort_statistics']['policies'][POLICIES[1]]['purchase_channels']=c
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(value)


@pytest.mark.parametrize('legacy',[False,True])
def test_cache_roundtrip_aggregate_only_and_legacy_unavailable_message(channel_dataset,tmp_path,legacy):
    value=calculate(channel_dataset)
    c=value['cohort_statistics']['policies'][POLICIES[0]]
    if legacy:c.pop('purchase_channels')
    path=tmp_path/'channels.json';cache.save_result(path,value)
    saved=cache.load_result(path)
    assert saved==json.loads(json.dumps(value))
    all_sections=cohort_sections(saved,compact=True)
    start=next(i for i,s in enumerate(all_sections) if s[0]==CHANNEL_TITLE)
    sections=all_sections[start:start+4]
    assert sections[0][0]==CHANNEL_TITLE
    assert sum(len(s[2]) for s in sections)==3
    if legacy:
        assert CHANNEL_LEGACY_TEXT in str(sections)
    else:
        assert all(token not in json.dumps(c['purchase_channels']) for token in ('private-','customer_id','order_id'))


def test_new_dependency_is_attested(monkeypatch):
    before=slices._algorithm_version()
    original=slices._hash_file
    monkeypatch.setattr(slices,'_hash_file',lambda path:'changed' if path.name=='statistics_cohorts_channels.py' else original(path))
    assert slices._algorithm_version()!=before


@pytest.mark.parametrize('ids',[('z-order','a-order'),('2','10'),('Я','A'),('😀','А')])
def test_equivalent_sources_choose_same_order_at_equal_timestamp(channel_dataset,ids):
    root,_,source=channel_dataset
    entry=source['manual_interactions']['orders'];directory=root/entry['directory']
    first=raw(ids[0],'2026-01-01','old','Z','Last lexically')
    second=raw(ids[1],'2026-01-01','new','A','First lexically')
    second['lines'][0]['priceOfLine']=99
    for i,rows in enumerate(([first],[second]),1):
        (directory/f'orders_part_{i:03}.json').write_text(json.dumps({'orders':rows}))
    expected=calculate(channel_dataset)
    c=expected['cohort_statistics']['policies'][POLICIES[0]]['purchase_channels']
    assert catalog(c)['A']['first_buyers']==1
    assert catalog(c)['Z']['second_buyers']==1
    build(channel_dataset)
    assert asdict(query(channel_dataset,SELECTION,calculated_at=AT))==expected


@pytest.mark.parametrize('damage',['rank_missing','rank_zero','rank_bool','channel','id','name'])
def test_indexed_missing_projection_is_not_silently_replaced_with_unknown_channel(damage):
    from Application.statistics_query import IndexedInput
    from Application.mindbox.adapters._common import AdapterError
    source=IndexedInput.__new__(IndexedInput);source._batch_lines={}
    details={'firstAction':{'channel':{'ids':{'externalId':None},'name':None}}}
    rank=1
    if damage=='rank_missing':rank=None
    elif damage=='rank_zero':rank=0
    elif damage=='rank_bool':rank=True
    elif damage=='channel':details['firstAction'].pop('channel')
    elif damage=='id':details['firstAction']['channel']['ids'].pop('externalId')
    elif damage=='name':details['firstAction']['channel'].pop('name')
    row=(1,0,0,0,1,1,b'fingerprint',0,0,0,json.dumps(details),rank)
    with pytest.raises(AdapterError,match='unavailable'):
        source._order(row)


def test_channel_order_ranks_are_private_integer_projection_without_original_order_ids(channel_dataset):
    import sqlite3
    from contextlib import closing
    root=channel_dataset[0]
    build(channel_dataset)
    pointer=json.loads((root/'analytics/current.json').read_text())
    generation=root/'analytics'/pointer['generation']
    manifest=json.loads((generation/'manifest.json').read_text())
    assert manifest['schema']==3
    with closing(sqlite3.connect((generation/'statistics.sqlite').resolve().as_uri()+'?mode=ro',uri=True)) as db:
        tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not {'order_ids','customer_ids','cohort_order_ranks'} & tables
        rows=list(db.execute('SELECT details,cohort_order_rank FROM orders WHERE header_bad=0 AND date_bad=0'))
        assert rows and all(type(rank) is int and rank>0 for _,rank in rows)
        assert 'private-' not in str(rows)
