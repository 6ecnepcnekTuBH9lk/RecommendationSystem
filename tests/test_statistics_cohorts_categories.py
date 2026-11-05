"""Direct category multi-label buyer analytics on small synthetic data only."""
from copy import deepcopy
from dataclasses import asdict
from datetime import date
from itertools import permutations
from types import SimpleNamespace
from decimal import Decimal
import json

import pytest

from Application import statistics_cache as cache,statistics_slices as slices
from Application.analysis_filter import AnalysisFilter
from Application.statistics_cohorts_core import CohortAggregates,POLICIES
from Application import statistics_cohorts_categories as categories
from Application.statistics_cohorts_categories import CategoryAnalyticsLimitError
from Application.statistics_cohorts_presentation import cohort_sections,CATEGORY_TITLE,CATEGORY_LEGACY_TEXT
from Application.product_resolution import ProductCatalog,CatalogDiagnostics,ProductResolver
from Application.product_statistics import ProductMetadata
from Application.mindbox.records import ProductKey
from tests.test_statistics_cohorts_core import FULL,AS_OF,SPACE
from tests.test_statistics_cohorts_money import order
from tests.test_statistics_cohorts_integration import dataset,calculate,SELECTION,AT
from tests.test_statistics_rfm_integration import raw_order
from tests.test_statistics_index import build,query
from tests.test_dataset_statistics import save

__all__=['dataset']
NAMES={'P':'Брюки','S':'Рубашка','J':'Пиджак','CH':'Костюмные брюки','ROOT':'Одежда','TWIN':'Рубашка'}
PATHS={c:('Одежда',n) for c,n in NAMES.items() if c!='ROOT'}
PATHS['ROOT']=('Одежда',)
PATHS['CH']=('Одежда','Брюки','Костюмные брюки')


def run(records=(),**kwargs):
    core=CohortAggregates(kwargs.pop('as_of',AS_OF),kwargs.pop('coverage',FULL),identity_space=SPACE,
                         category_names=kwargs.pop('names',NAMES),category_paths=kwargs.pop('paths',PATHS),**kwargs)
    for fact,codes in records:
        core.add(fact,category_codes=codes,channel_external_id='A',channel_name='Channel')
    return core


def result(core):
    return core.result()['policies'][POLICIES[0]]['purchase_categories']


def record(codes,*,buyer='private-buyer',id=1,day='2024-01-01'):
    return order(day,id=id,buyer=buyer),codes


def rows(value):
    return {r['code']:r for r in value['categories']}


def test_multicategory_example_four_pairs_single_buyer_bases_and_no_double_count():
    value=result(run([record(['P','S','P']),record(['J','S','S'],id=2),record(['P'],buyer='single')]))
    r=rows(value)
    assert (value['buyers'],value['repeat_buyers'])==(2,1)
    assert (r['P']['first_buyers'],r['P']['repeat_buyers'],r['P']['observed_repeat_rate'])==(2,1,50)
    assert (r['S']['first_buyers'],r['S']['repeat_buyers'],r['S']['observed_repeat_rate'])==(1,1,100)
    assert r['J']['second_buyers']==r['S']['second_buyers']==1
    assert {(r['first_code'],r['second_code']) for r in value['transitions']}=={('P','J'),('P','S'),('S','J'),('S','S')}
    assert all(r['buyers']==1 and r['repeat_rate']==100 for r in value['transitions'])
    assert sum(r['first_buyers'] for r in value['categories'])==3>value['buyers']
    assert sum(r['second_rate'] or 0 for r in value['categories'])==200
    assert sum(r['buyers'] for r in value['transitions'])==4>value['repeat_buyers']


@pytest.mark.parametrize('sequence',list(permutations(range(4))))
def test_first_two_sets_follow_existing_deterministic_order_and_ignore_all_later_orders(sequence):
    records=[record({'S'},id='1'),record({'P',None},id=2),record({'CH'},id=1),
             record({'J'},id=3,day='2025-01-01')]
    expected=run(records).result()
    assert run([records[i] for i in sequence]).result()==expected
    value=expected['policies'][POLICIES[0]]['purchase_categories']
    assert rows(value)['CH']['first_buyers']==1
    assert rows(value)['P']['second_buyers']==1 and rows(value)[None]['second_buyers']==1
    assert 'S' not in rows(value) and 'J' not in rows(value)
    assert 'ROOT' not in rows(value)  # Path is display metadata, not another category.


def test_unknown_mixed_orders_keep_known_and_unknown_parts_and_direct_parent_child_separate():
    value=result(run([record({'P',None}),record({'CH',None},id=2)]))
    assert value['unknown_first_buyers']==value['unknown_second_buyers']==1
    assert rows(value)['P']['first_buyers']==rows(value)[None]['first_buyers']==1
    assert len(value['transitions'])==4
    assert 'ROOT' not in rows(value)
    assert rows(value)['CH']['first_buyers']==0 and rows(value)['CH']['second_buyers']==1
    assert rows(value)['CH']['path']==['Одежда','Брюки','Костюмные брюки']


def test_single_orders_no_second_and_empty_cohorts_have_unavailable_denominators():
    for records,buyers in (([],0),([record({None})],1)):
        value=result(run(records))
        assert value['buyers']==buyers and value['repeat_buyers']==0
        assert not value['transitions']
        assert rows(value)[None]['second_rate'] is None
        assert rows(value)[None]['observed_repeat_rate']==(0 if buyers else None)


def test_display_filter_cutoff_and_ineligible_horizons_preserve_observed_category_base():
    core=run([record({'P'},day='2023-12-31'),record({'S'},id=2),
              record({'J'},buyer='visible'),record({None},buyer='visible',id=2,day='2024-01-02'),
              record({'CH'},buyer='visible',id=3,day='2024-01-11')],
             cohort_start='2024-01',cohort_end='2024-01',as_of=date(2024,1,10))
    value=result(core)
    assert (value['buyers'],value['repeat_buyers'])==(1,1)
    assert set(rows(value))=={'J',None}
    assert rows(value)['J']['observed_repeat_rate']==100
    assert core.result()['policies'][POLICIES[0]]['summary']['horizons']['30']['repeated']==0


def test_prepared_lines_strict_only_resolver_direct_categories_and_unmapped_reference():
    names=NAMES
    metadata={'000001':ProductMetadata(category='CH'),'000002':ProductMetadata(category='P'),
              '000003':ProductMetadata(category=None),'000004':ProductMetadata(category='missing-reference'),
              '000005':ProductMetadata(category='CH')}
    resolver=ProductResolver(ProductCatalog(frozenset(metadata),CatalogDiagnostics()))
    def line(code,status='F',qty=1):
        return SimpleNamespace(product=ProductKey('offline1C',code+'-size'),line_status=status,quantity=qty)
    lines=[line('000001'),line('000001','sold'),line('000005'),line('000002'),line('000003'),line('999999'),
           line('000004'),line('000002','CP'),line('000002','Return'),line('000002','CD'),
           line('000002','F',0),line('000002','F',Decimal('0.5'))]
    before=deepcopy(resolver.__dict__)
    assert categories.order_category_codes(lines,resolver,metadata,names,lambda:None)==frozenset({'CH','P',None})
    assert resolver.__dict__==before  # No interaction diagnostics are incremented.
    assert categories.order_category_codes([line('000001','CP')],resolver,metadata,names,lambda:None)==frozenset()


def test_missing_projection_rejected_and_default_standalone_core_stays_backward_compatible():
    core=run()
    with pytest.raises(ValueError,match='unavailable'):
        core.add(order())
    assert result(core)['buyers']==0
    legacy=CohortAggregates(AS_OF,FULL,identity_space=SPACE)
    legacy.add(order())
    assert 'purchase_categories' not in legacy.result()['policies'][POLICIES[0]]


@pytest.mark.parametrize('codes',[[],set(),{'not-in-reference'},'P',{True}])
def test_invalid_prepared_category_set_does_not_mutate_previous_results(codes):
    core=run([record({'P'})]);before=core.result()
    with pytest.raises(ValueError):
        core.add(order(id=2),category_codes=codes)
    assert core.result()==before


def test_guards_diagnose_without_silent_crop_and_cancellation_is_kept(monkeypatch):
    monkeypatch.setattr(categories,'MAX_ORDER_CATEGORIES',2)
    with pytest.raises(CategoryAnalyticsLimitError,match='одного заказа'):
        run([record({'P','S','J'})])
    monkeypatch.setattr(categories,'MAX_ORDER_CATEGORIES',256)
    monkeypatch.setattr(categories,'MAX_PAIRS_PER_BUYER',3)
    core=run([record({'P','S'}),record({'J','S'},id=2)])
    with pytest.raises(CategoryAnalyticsLimitError,match='одного покупателя'):
        core.result()
    monkeypatch.setattr(categories,'MAX_PAIRS_PER_BUYER',4096)
    monkeypatch.setattr(categories,'MAX_AGGREGATE_PAIRS',3)
    with pytest.raises(CategoryAnalyticsLimitError,match='различных переходов'):
        core.result()
    monkeypatch.setattr(categories,'MAX_AGGREGATE_PAIRS',100000)
    expected=core.result()
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        if calls==4:raise InterruptedError()
    core.check=cancel
    with pytest.raises(InterruptedError):core.result()
    core.check=lambda:None
    assert core.result()==expected


def test_fixed_two_sets_fresh_results_and_no_top_ten_cut():
    names={f'C{i:02}':f'Category{i}' for i in range(13)}
    records=[r for i in range(13) for r in (record({f'C{i:02}'},buyer=str(i)),record({'C00'},buyer=str(i),id=2))]
    core=run(records,names=names,paths={})
    value=result(core)
    assert len(value['transitions'])==13
    buyer=core._buyers[POLICIES[0]]['0']
    assert isinstance(buyer.first_categories,frozenset) and isinstance(buyer.second_categories,frozenset)
    assert not hasattr(buyer,'__dict__')
    before=core.result();result(core)['categories'].clear()
    assert core.result()==before
    assert 'purchase_categories' not in core.result()['policies'][POLICIES[1]]


def raw(key,day,buyer,items):
    item=raw_order(key,day,buyer,statuses=tuple(status for _,status in items))
    for line,(code,_) in zip(item['lines'],items):
        line['product']['ids']={'offline1C':code+'-size'}
    return item


@pytest.fixture
def category_dataset(dataset):
    root,path,source=dataset
    path.write_text('КодНоменклатуры|НазваниеНаСайте|КатегорияНаСайте|ВидНоменклатуры\n'
        '000001|Pants|P|Брюки\n000002|Shirt|S|Рубашки\n000003|Jacket|J|Пиджаки\n'
        '000004|Child|CH|Костюмные брюки\n000005|Parent|P|Брюки\n000006|Twin|TWIN|Рубашки\n'
        '000007|Empty||Другое\n000008|Missing|missing-reference|Другое\n000009|Root|ROOT|Другое\n',encoding='utf-8')
    (root/'site_categories.csv').write_text('КодКатегории|НазваниеКатегории|КодРодительскойКатегории\n'
        'ROOT|Одежда|\nP|Брюки|ROOT\nS|Рубашка|ROOT\nJ|Пиджак|ROOT\nCH|Костюмные брюки|P\nTWIN|Рубашка|ROOT\n',encoding='utf-8')
    first=[raw('private-first','2026-01-01','old',[('000001','F'),('000002','F'),('000001','sold')]),
           raw('private-single','2026-01-05','single',[('000004','F')]),
           raw('private-mixed','2026-01-06','mixed',[('000001','F'),('999999','F'),('000007','F'),('000008','F')]),
           raw('z-tie','2026-01-03','tie',[('000001','F')]),
           raw('private-policy','2026-01-04','policy',[('000004','F'),('000003','CP'),('000009','Return')]),
           raw('private-historical','2020-01-01','historical',[('000005','F')])]
    duplicate=deepcopy(first[0]);duplicate['lines'][0]['product']['ids']={'offline1C':'000003-size'}
    second=[raw('private-second','2026-01-02','new',[('000003','F'),('000002','sold')]),
            raw('private-mixed-second','2026-01-07','mixed',[('999999','F'),('000002','F'),('000002','F')]),
            raw('private-third','2026-01-08','mixed',[('000003','F')]),
            raw('a-tie','2026-01-03','tie',[('000003','F')]),
            raw('private-policy-second','2026-01-05','policy',[('000002','sold'),('000006','F')]),
            raw('private-history-second','2026-01-10','historical',[('000006','F')]),duplicate,
            raw('private-future','2026-09-01','future',[('000003','F')])]
    directory=root/source['manual_interactions']['orders']['directory']
    for i,items in enumerate((first,second),1):
        (directory/f'orders_part_{i:03}.json').write_text(json.dumps({'orders':items}),encoding='utf-8')
    save(root,source)
    return dataset


@pytest.mark.parametrize('selection',[SELECTION,AnalysisFilter(),AnalysisFilter('2026-02-01','2026-03-31'),
    AnalysisFilter(SELECTION.start_date,SELECTION.end_date,nomenclature_types=('missing',))])
def test_reference_indexed_equivalent_inputs_preserve_merges_dedup_ties_strict_and_independent_product_scope(category_dataset,selection):
    value=calculate(category_dataset,selection)
    c=value['cohort_statistics']['policies'][POLICIES[0]]['purchase_categories']
    if selection.start_date==SELECTION.start_date:
        assert (c['buyers'],c['repeat_buyers'],c['unknown_first_buyers'],c['unknown_second_buyers'])==(5,4,1,1)
        r=rows(c)
        assert r['P']['first_buyers']==2 and r['CH']['first_buyers']==2
        assert r['CH']['observed_repeat_rate']==50
        assert r['J']['first_buyers']==1
        assert r['S']['second_buyers']==3 and r['S']['second_rate']==75
        assert 'ROOT' not in r and 'missing-reference' not in r
        assert sum(p['buyers'] for p in c['transitions'])==11
        assert sum(r['first_buyers'] for r in c['categories'])==7
    build(category_dataset)
    assert asdict(query(category_dataset,selection,calculated_at=AT))==value
    cache.validate_result(value)
    baseline=calculate(category_dataset,SELECTION)['cohort_statistics']
    filtered=calculate(category_dataset,AnalysisFilter(SELECTION.start_date,SELECTION.end_date,nomenclature_types=('Костюмные брюки',)))['cohort_statistics']
    assert filtered==baseline


@pytest.mark.parametrize('damage',['private','extra','negative','bool','buyers','repeat','unknown_first','unknown_second',
    'code','name','path','parent_path','duplicate','missing_unknown','first','first_repeat','first_rate','second',
    'second_rate','pair_code','pair_count','pair_rate','pair_duplicate','pair_sort','catalog_sort','missing_pairs','extended'])
def test_closed_multilabel_dto_rejects_damage_without_requiring_label_sums_equal_unique_buyers(category_dataset,damage):
    value=calculate(category_dataset);p=value['cohort_statistics']['policies'][POLICIES[0]];c=p['purchase_categories']
    row=c['categories'][0];pair=c['transitions'][0]
    if damage=='private':row['customer_id']='private'
    elif damage=='extra':c['order_id']='private'
    elif damage=='negative':row['first_buyers']=-1
    elif damage=='bool':row['first_buyers']=True
    elif damage=='buyers':c['buyers']+=1
    elif damage=='repeat':c['repeat_buyers']+=1
    elif damage=='unknown_first':c['unknown_first_buyers']+=1
    elif damage=='unknown_second':c['unknown_second_buyers']+=1
    elif damage=='code':row['code']=' bad'
    elif damage=='name':row['name']=''
    elif damage=='path':row['path']=['wrong']
    elif damage=='parent_path':c['categories'][-1]['path']=['invented']
    elif damage=='duplicate':c['categories'].append(deepcopy(row))
    elif damage=='missing_unknown':c['categories'].pop()
    elif damage=='first':row['first_buyers']=c['buyers']+1
    elif damage=='first_repeat':row['repeat_buyers']=row['first_buyers']+1
    elif damage=='first_rate':row['observed_repeat_rate']=999
    elif damage=='second':row['second_buyers']=c['repeat_buyers']+1
    elif damage=='second_rate':row['second_rate']=999
    elif damage=='pair_code':pair['second_code']='absent'
    elif damage=='pair_count':pair['buyers']=c['repeat_buyers']+1
    elif damage=='pair_rate':pair['repeat_rate']=999
    elif damage=='pair_duplicate':c['transitions'].append(deepcopy(pair))
    elif damage=='pair_sort':c['transitions'].reverse()
    elif damage=='catalog_sort':c['categories'].reverse()
    elif damage=='missing_pairs':c['transitions'].clear()
    elif damage=='extended':value['cohort_statistics']['policies'][POLICIES[1]]['purchase_categories']=c
    with pytest.raises(cache.StatisticsCacheError):cache.validate_result(value)


@pytest.mark.parametrize('legacy',[False,True])
def test_saved_new_and_legacy_dto_open_without_category_recalculation(category_dataset,tmp_path,legacy):
    value=calculate(category_dataset)
    p=value['cohort_statistics']['policies'][POLICIES[0]]
    if legacy:p.pop('purchase_categories')
    else:
        assert all(token not in json.dumps(p['purchase_categories']) for token in ('customer_id','order_id','private-'))
    path=tmp_path/'saved.json';cache.save_result(path,value);saved=cache.load_result(path)
    assert saved==json.loads(json.dumps(value))
    sections=cohort_sections(saved,compact=True)
    assert sections[-4][0]==CATEGORY_TITLE and sum(len(s[2]) for s in sections[-4:])==3
    if legacy:assert CATEGORY_LEGACY_TEXT in str(sections[-4:])


def test_new_algorithm_dependency_changes_fingerprint(monkeypatch):
    before=slices._algorithm_version();original=slices._hash_file
    monkeypatch.setattr(slices,'_hash_file',lambda p:'changed' if p.name=='statistics_cohorts_categories.py' else original(p))
    assert slices._algorithm_version()!=before


def test_counts_match_independent_unique_buyer_sets_for_multicategory_orders():
    from collections import defaultdict
    import random
    records=[]
    for i in range(15):
        for j in range(1+i%4):
            options=['P','S','J','CH',None]
            codes=[options[(i+j)%5],options[(i+2*j)%5]]
            records.append(record(codes,buyer=f'b{i}',id=j+1,day=f'2024-01-{j+1:02}'))
    by_buyer=defaultdict(list)
    for fact,codes in records:
        by_buyer[fact.customer_id].append((fact.order_datetime_utc,fact.order_id,set(codes)))
    first,second,pairs=defaultdict(set),defaultdict(set),defaultdict(set)
    repeats=set()
    for buyer,orders in by_buyer.items():
        orders.sort(key=lambda r:(r[0],r[1]))
        a=orders[0][2]
        for code in a:first[code].add(buyer)
        if len(orders)<2:continue
        repeats.add(buyer);b=orders[1][2]
        for code in b:second[code].add(buyer)
        for x in a:
            for y in b:pairs[x,y].add(buyer)
    expected=None
    for seed in range(6):
        random.Random(seed).shuffle(records)
        value=result(run(records))
        assert value['buyers']==len(by_buyer) and value['repeat_buyers']==len(repeats)
        for row in value['categories']:
            code=row['code']
            assert row['first_buyers']==len(first[code])
            assert row['repeat_buyers']==len(first[code]&repeats)
            assert row['second_buyers']==len(second[code])
        assert {(r['first_code'],r['second_code']):r['buyers'] for r in value['transitions']}=={
            pair:len(buyers) for pair,buyers in pairs.items()}
        if expected is not None:assert value==expected
        expected=value


def test_prepared_order_category_guard_reports_before_truncating(monkeypatch):
    metadata={f'{i:06}':ProductMetadata(category=code) for i,code in enumerate(('P','S','J'),1)}
    resolver=ProductResolver(ProductCatalog(frozenset(metadata),CatalogDiagnostics()))
    lines=[SimpleNamespace(product=ProductKey('offline1C',code),line_status='F',quantity=1) for code in metadata]
    monkeypatch.setattr(categories,'MAX_ORDER_CATEGORIES',2)
    with pytest.raises(CategoryAnalyticsLimitError,match='без усечения'):
        categories.order_category_codes(lines,resolver,metadata,NAMES,lambda:None)


def test_one_direct_category_in_both_orders_yields_one_pair_once():
    value=result(run([record(['P','P']),record(['P','P'],id=2)]))
    assert rows(value)['P']['first_buyers']==rows(value)['P']['second_buyers']==1
    assert rows(value)['P']['observed_repeat_rate']==100
    assert value['transitions']==[dict(first_code='P',second_code='P',buyers=1,repeat_rate=100)]


def test_category_limit_diagnostic_uses_existing_gui_error_protocol_without_private_values(monkeypatch):
    from Application.loading_errors import error_record,format_error
    monkeypatch.setattr(categories,'MAX_PAIRS_PER_BUYER',1)
    with pytest.raises(CategoryAnalyticsLimitError) as caught:
        run([record({'P','S'}),record({'S','J'},id=2)]).result()
    text=format_error(error_record(caught.value),context='Статистика')
    assert 'CategoryAnalyticsLimitError' in text and 'без усечения данных' in text
    assert all(v not in text for v in ('private-buyer','private-order','secret-'))
