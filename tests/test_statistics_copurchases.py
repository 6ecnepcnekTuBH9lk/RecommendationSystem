"""Exact unordered within-order category counts; synthetic data only."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations,permutations
import json

import pytest

from Application import statistics_cache as cache,statistics_slices as slices
from Application.analysis_filter import AnalysisFilter
from Application import statistics_copurchases as cop
from Application.statistics_cohorts_categories import category_key
from tests.test_statistics_cohorts_categories import category_dataset,dataset,raw,NAMES,PATHS
from tests.test_statistics_buyer_aggregates import purchase,COVERAGE
from tests.test_statistics_cohorts_integration import calculate,SELECTION,AT
from tests.test_statistics_index import build,query

__all__=['dataset','category_dataset']


def record(codes,customer='private-buyer',key='private-order',stamp='2026-01-01T12:00:00Z',lines=None):
    return purchase(customer,key,stamp,**({'lines':lines} if lines is not None else {})),codes


def run(records=(),**kwargs):
    agg=cop.CategoryCoPurchases('synthetic',COVERAGE,kwargs.pop('names',NAMES),kwargs.pop('paths',PATHS),**kwargs)
    for fact,codes in records:agg.add(fact,codes)
    return agg


def categories(value):
    return {r['code']:r for r in value['categories']}


def pairs(value):
    return {(r['category1'],r['category2']):r for r in value['pairs']}


@pytest.mark.parametrize('codes,expected',[(['P'],0),(['P','S','P'],1),(['J','S','P','P'],3)])
def test_unique_categories_and_unordered_pairs_per_unique_order(codes,expected):
    value=run([record(codes)]).result()
    assert value['orders']==value['buyers']==1
    assert len(value['pairs'])==expected
    assert value['multicategory_orders']==value['multicategory_buyers']==int(expected>0)
    assert set(pairs(value))==set(combinations(sorted(set(codes),key=category_key),2))
    assert all(r['orders']==r['buyers']==1 for r in value['categories']+value['pairs'])
    assert all(a!=b for a,b in pairs(value))


def test_repeated_orders_same_pair_do_not_duplicate_exact_buyer_membership():
    value=run([record(['P','S'],key='one'),record(['S','P'],key='two'),
               record(['P','S'],customer='other',key='three')]).result()
    assert value['orders']==3 and value['buyers']==2
    assert value['multicategory_orders']==3 and value['multicategory_buyers']==2
    row=value['pairs'][0]
    assert (row['orders'],row['buyers'],row['order_rate'])==(3,2,100)
    assert row['given_first_rate']==row['given_second_rate']==100


def test_directional_conditional_frequencies_have_category_specific_denominators():
    records=[record({'P','S'},customer=str(i%2),key=str(i)) for i in range(3)]
    records += [record({'P'},key=str(i)) for i in range(3,10)]
    records += [record({'S'},key=str(i)) for i in range(10,15)]
    value=run(records).result()
    assert value['orders']==15
    assert categories(value)['P']['orders']==10 and categories(value)['S']['orders']==8
    pair=pairs(value)['P','S']
    assert pair['orders']==3 and pair['buyers']==2 and pair['order_rate']==20
    assert pair['given_first_rate']==30 and pair['given_second_rate']==37.5


def test_unknown_mixed_and_direct_child_categories_are_not_fabricated_known_groups():
    value=run([record({'CH',None}),record({None},customer='only-unknown',key='two')]).result()
    assert set(categories(value))=={'CH',None} and 'P' not in categories(value) and 'ROOT' not in categories(value)
    assert value['unknown_orders']==2 and value['multicategory_orders']==1
    assert value['pairs']==[dict(category1='CH',category2=None,orders=1,buyers=1,order_rate=50,
                                 given_first_rate=100,given_second_rate=50,incomplete=True)]


def test_equal_names_different_codes_are_distinct_categories_and_form_valid_pair():
    value=run([record({'S','TWIN'})]).result()
    assert set(categories(value))=={'S','TWIN'}
    assert len(value['pairs'])==1 and value['pairs'][0]['category1']!=value['pairs'][0]['category2']


def test_non_strict_only_positions_do_not_create_confirmed_orders_or_categories():
    value=run([record(None,lines=(('CP',1,'10','offline1C'),('Return',1,'10','offline1C')))]).result()
    assert value['orders']==value['buyers']==value['multicategory_orders']==0
    assert value['multicategory_rate'] is None and value['categories']==value['pairs']==[]


@pytest.mark.parametrize('sequence',list(permutations(range(4))))
def test_order_processing_arrival_does_not_change_all_aggregates(sequence):
    records=[record({'P','S'},key='1'),record({'P'},key='2'),
             record({'S',None},key='3',customer='b'),record({'J','S','P'},key='4')]
    assert run([records[i] for i in sequence]).result()==run(records).result()


def test_independent_sets_of_unique_orders_and_buyers_verify_every_counter():
    records=[record({('P','S','J',None)[i%4],('P','S','J',None)[(i//3)%4]},key=f'o{i}',customer=f'b{i%5}')
             for i in range(35)]
    category_orders,category_buyers=defaultdict(set),defaultdict(set)
    pair_orders,pair_buyers=defaultdict(set),defaultdict(set)
    multi_orders,multi_buyers=set(),set()
    for fact,codes in records:
        codes=set(codes)
        if len(codes)>1:multi_orders.add(fact.order_id);multi_buyers.add(fact.customer_id)
        for code in codes:category_orders[code].add(fact.order_id);category_buyers[code].add(fact.customer_id)
        for pair in combinations(sorted(codes,key=category_key),2):
            pair_orders[pair].add(fact.order_id);pair_buyers[pair].add(fact.customer_id)
    value=run(records).result()
    assert value['orders']==35 and value['buyers']==5
    assert value['multicategory_orders']==len(multi_orders) and value['multicategory_buyers']==len(multi_buyers)
    for code,row in categories(value).items():
        assert row['orders']==len(category_orders[code]) and row['buyers']==len(category_buyers[code])
    for pair,row in pairs(value).items():
        assert row['orders']==len(pair_orders[pair]) and row['buyers']==len(pair_buyers[pair])


def test_private_dense_memberships_and_no_order_history_no_top_ten_and_fresh_results():
    names={f'C{i:02}':f'Category{i}' for i in range(14)}
    records=[record({f'C{i:02}','C13'},key=f'private-order{i}') for i in range(13)]
    agg=run(records,names=names,paths={});value=agg.result()
    assert len(value['pairs'])==13 and value['orders']==13 and value['multicategory_buyers']==1
    assert len(agg._buyers)==1 and all(type(sid) is int for g in agg.pairs.values() for sid in g.buyers)
    assert 'private-order' not in str(agg.__dict__)
    assert all(token not in json.dumps(value) for token in ('private-','customer_id','order_id'))
    before=agg.result();agg.result()['pairs'].clear()
    assert agg.result()==before


def test_resource_guards_raise_explicitly_and_never_publish_partial_results(monkeypatch):
    monkeypatch.setattr(cop,'MAX_PAIRS_PER_ORDER',2)
    agg=run()
    with pytest.raises(cop.CoPurchaseLimitError,match='одного заказа'):agg.add(*record({'P','S','J'}))
    with pytest.raises(ValueError,match='incomplete'):agg.result()
    monkeypatch.setattr(cop,'MAX_PAIRS_PER_ORDER',4096)
    monkeypatch.setattr(cop,'MAX_DISTINCT_PAIRS',1)
    agg=run([record({'P','S'})])
    with pytest.raises(cop.CoPurchaseLimitError,match='различных'):agg.add(*record({'J','S'},key='two'))
    with pytest.raises(ValueError,match='incomplete'):agg.result()
    monkeypatch.setattr(cop,'MAX_DISTINCT_PAIRS',100000)
    agg=run();monkeypatch.setattr(cop,'MAX_ADDITIONAL_BYTES',agg._reserved+1)
    with pytest.raises(cop.CoPurchaseLimitError,match='памяти'):agg.add(*record({'P','S'}))
    with pytest.raises(ValueError,match='incomplete'):agg.result()


def test_output_budget_and_cancellation_remain_explicit_and_recoverable_at_finalization(monkeypatch):
    agg=run([record({'P','S'})])
    before=agg.result()
    limit=agg._reserved+512*(len(agg.categories)+len(agg.pairs))-1
    monkeypatch.setattr(cop,'MAX_ADDITIONAL_BYTES',limit)
    with pytest.raises(cop.CoPurchaseLimitError,match='результата'):agg.result()
    monkeypatch.setattr(cop,'MAX_ADDITIONAL_BYTES',512*1024**2)
    def cancel():raise InterruptedError()
    agg.check=cancel
    with pytest.raises(InterruptedError):agg.result()
    agg.check=lambda:None
    assert agg.result()==before


def test_mid_order_cancellation_invalidates_accumulator():
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        if calls==3:raise InterruptedError()
    agg=run(check=cancel)
    with pytest.raises(InterruptedError):agg.add(*record({'P','S','J'}))
    agg.check=lambda:None
    with pytest.raises(ValueError,match='incomplete'):agg.result()


@pytest.mark.parametrize('selection',[SELECTION,AnalysisFilter(),AnalysisFilter('2026-02-01','2026-03-31'),
    AnalysisFilter(SELECTION.start_date,SELECTION.end_date,nomenclature_types=('missing',))])
def test_same_buyer_base_all_orders_reference_indexed_and_product_filter_independence(category_dataset,selection):
    value=calculate(category_dataset,selection)
    strict=value['buyer_statistics']['policies']['STRICT_COMPLETED'];c=strict['category_copurchases']
    assert c['orders']==strict['orders'] and c['buyers']==strict['buyers']
    if selection.start_date==SELECTION.start_date:
        assert (c['orders'],c['buyers'],c['multicategory_orders'],c['multicategory_buyers'],len(c['pairs']))==(11,6,5,3,5)
        assert categories(c)['J']['orders']==3  # Includes the third purchase.
        assert categories(c)['S']['orders']==4 and categories(c)['TWIN']['orders']==2
        assert 'ROOT' not in categories(c) and c['unknown_orders']==2
        assert sum(r['incomplete'] for r in c['pairs'])==2
    build(category_dataset)
    assert asdict(query(category_dataset,selection,calculated_at=AT))==value
    cache.validate_result(value)
    assert calculate(category_dataset,SELECTION)['buyer_statistics']['policies']['STRICT_COMPLETED']['category_copurchases']==calculate(
        category_dataset,AnalysisFilter(SELECTION.start_date,SELECTION.end_date,nomenclature_types=('Костюмные брюки',))
        )['buyer_statistics']['policies']['STRICT_COMPLETED']['category_copurchases']


def test_merged_buyers_repeated_pair_orders_and_existing_cross_part_dedup(category_dataset):
    root,_,source=category_dataset
    directory=root/source['manual_interactions']['orders']['directory']
    path=directory/'orders_part_002.json';payload=json.loads(path.read_text())
    item=raw('private-new-repeat','2026-01-11','new',[('000001','F'),('000002','F')])
    payload['orders'] += [item,deepcopy(item)]
    path.write_text(json.dumps(payload))
    value=calculate(category_dataset)
    c=value['buyer_statistics']['policies']['STRICT_COMPLETED']['category_copurchases']
    assert c['orders']==12 and pairs(c)['P','S']['orders']==2 and pairs(c)['P','S']['buyers']==1
    build(category_dataset)
    assert asdict(query(category_dataset,SELECTION,calculated_at=AT))==value


def test_buyer_scope_preserves_selected_dates_and_is_not_restricted_to_visible_cohorts(category_dataset):
    root,_,source=category_dataset
    path=root/source['manual_interactions']['orders']['directory']/'orders_part_002.json'
    payload=json.loads(path.read_text());payload['orders'].append(raw(
        'private-after-cohort-cutoff','2026-12-01','late',[('000001','F'),('000002','F')]))
    path.write_text(json.dumps(payload))
    selected=AnalysisFilter('2026-01-01','2026-12-31')
    value=calculate(category_dataset,selected)
    strict=value['buyer_statistics']['policies']['STRICT_COMPLETED']
    c=strict['category_copurchases']
    assert value['cohort_statistics']['as_of_date']=='2026-12-31'
    assert c['buyers']>value['cohort_statistics']['policies']['STRICT_COMPLETED']['displayed_buyers']
    assert c['orders']==strict['orders']==13
    assert c['buyers']==strict['buyers']==8
    assert c['multicategory_orders']==6
    build(category_dataset)
    assert asdict(query(category_dataset,selected,calculated_at=AT))==value


@pytest.mark.parametrize('damage',['private','extra','orders','buyers','multi','multi_buyers','multi_rate','unknown',
    'negative','bool','category_count','category_buyers','category_rate','code','name','path','duplicate','sort',
    'pair_missing','pair_duplicate','pair_sort','self_pair','reverse_pair','missing_code','pair_orders','pair_buyers',
    'pair_rate','first_rate','second_rate','incomplete','extended','none'])
def test_closed_dto_rejects_individual_data_and_inconsistent_denominators(category_dataset,damage):
    value=calculate(category_dataset);strict=value['buyer_statistics']['policies']['STRICT_COMPLETED']
    c=strict['category_copurchases'];row=c['categories'][0];pair=c['pairs'][0]
    if damage=='private':pair['customer_id']='private'
    elif damage=='extra':c['order_ids']=[]
    elif damage=='orders':c['orders']+=1
    elif damage=='buyers':c['buyers']+=1
    elif damage=='multi':c['multicategory_orders']=c['orders']+1
    elif damage=='multi_buyers':c['multicategory_buyers']=c['buyers']+1
    elif damage=='multi_rate':c['multicategory_rate']=999
    elif damage=='unknown':c['unknown_orders']+=1
    elif damage=='negative':row['orders']=-1
    elif damage=='bool':row['buyers']=True
    elif damage=='category_count':row['orders']=c['orders']+1
    elif damage=='category_buyers':row['buyers']=row['orders']+1
    elif damage=='category_rate':row['order_rate']=999
    elif damage=='code':row['code']=' bad'
    elif damage=='name':row['name']=''
    elif damage=='path':row['path']=['wrong']
    elif damage=='duplicate':c['categories'].append(deepcopy(row))
    elif damage=='sort':c['categories'].reverse()
    elif damage=='pair_missing':c['pairs'].clear()
    elif damage=='pair_duplicate':c['pairs'].append(deepcopy(pair))
    elif damage=='pair_sort':c['pairs'].reverse()
    elif damage=='self_pair':pair['category2']=pair['category1']
    elif damage=='reverse_pair':pair['category1'],pair['category2']=pair['category2'],pair['category1']
    elif damage=='missing_code':pair['category1']='absent'
    elif damage=='pair_orders':pair['orders']=c['orders']+1
    elif damage=='pair_buyers':pair['buyers']=pair['orders']+1
    elif damage=='pair_rate':pair['order_rate']=999
    elif damage=='first_rate':pair['given_first_rate']=999
    elif damage=='second_rate':pair['given_second_rate']=999
    elif damage=='incomplete':pair['incomplete']=not pair['incomplete']
    elif damage=='extended':value['buyer_statistics']['policies']['EXTENDED_COMPLETED']['category_copurchases']=c
    elif damage=='none':strict['category_copurchases']=None
    with pytest.raises(cache.StatisticsCacheError):cache.validate_result(value)


@pytest.mark.parametrize('legacy',[False,True])
def test_cache_roundtrip_new_block_and_old_optional_absence(category_dataset,tmp_path,legacy):
    value=calculate(category_dataset)
    if legacy:value['buyer_statistics']['policies']['STRICT_COMPLETED'].pop('category_copurchases')
    path=tmp_path/'saved.json';cache.save_result(path,value)
    assert cache.load_result(path)==json.loads(json.dumps(value))


def test_new_calculation_module_is_attested(monkeypatch):
    before=slices._algorithm_version();original=slices._hash_file
    monkeypatch.setattr(slices,'_hash_file',lambda p:'changed' if p.name=='statistics_copurchases.py' else original(p))
    assert slices._algorithm_version()!=before


def test_same_prepared_facts_and_one_category_projection_per_order_are_reused(category_dataset,monkeypatch):
    from Application import dataset_statistics as statistics
    buyer_seen,co_seen,projections=[],[],[]
    buyer_add=statistics.BuyerAggregates.add
    co_add=statistics.CategoryCoPurchases.add
    project=statistics.order_category_codes
    def buyer(self,fact):
        buyer_seen.append(fact)
        return buyer_add(self,fact)
    def joint(self,fact,**kwargs):
        co_seen.append(fact)
        return co_add(self,fact,**kwargs)
    def category_projection(lines,*args):
        projections.append(lines)
        return project(lines,*args)
    monkeypatch.setattr(statistics.BuyerAggregates,'add',buyer)
    monkeypatch.setattr(statistics.CategoryCoPurchases,'add',joint)
    monkeypatch.setattr(statistics,'order_category_codes',category_projection)
    value=calculate(category_dataset)
    assert buyer_seen and len(buyer_seen)==len(co_seen)
    assert all(a is b for a,b in zip(buyer_seen,co_seen))
    assert len(projections)==len({id(lines) for lines in projections})
    strict=value['buyer_statistics']['policies']['STRICT_COMPLETED']
    assert strict['orders']==strict['category_copurchases']['orders']
