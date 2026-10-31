"""Synthetic full-history routing, DTO/cache and reference/indexed parity."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json

import pytest

from Application import dataset_statistics as statistics, statistics_cache as cache, statistics_slices as slices
from Application import statistics_query as query_module
from Application.analysis_filter import AnalysisFilter
from Application.statistics_cohorts_core import POLICIES
from Application.mindbox.adapters._common import AdapterError
from tests.test_analysis_filter import reference, sources
from tests.test_statistics_index import dataset as index_fixture, build, query
from tests.test_statistics_rfm_integration import raw_order
from tests.test_dataset_statistics import action, put_entry, save, DEFAULT_SELECTION


AT = datetime(2026,10,30,12,tzinfo=timezone.utc)
SELECTION = AnalysisFilter('2026-01-01','2026-08-31')


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    data = index_fixture.__wrapped__(tmp_path,monkeypatch)
    reference(data)
    root,_,catalog = data
    first = raw_order('secret-historical','2020-01-05','old')
    boundary = raw_order('secret-boundary','2020-02-01','boundary',('Return',))
    early = [first,boundary]
    late = [deepcopy(first), raw_order('secret-newer','2026-02-01','new'),
            raw_order('secret-boundary','2026-02-10','boundary'),
            raw_order('secret-cp','2026-01-03','visible',('CP',)),
            raw_order('secret-f','2026-01-05','visible',('F','CP','Return')),
            raw_order('secret-second','2026-02-05','visible',('sold',)),
            raw_order('secret-third','2026-03-06','visible'),
            raw_order('secret-partial','2026-01-05','partial',('CP',)),
            raw_order('secret-single','2026-01-06','single'),
            raw_order('secret-return','2026-01-06','return-only',('Return',)),
            raw_order('secret-future','2026-09-01','future')]
    sources(data,[action(DEFAULT_SELECTION.view_action_system_names[0],['000001-size'])],early+late,
            since='2020-01-01T00:00:00Z',until='2027-01-01T00:00:00Z')
    # Two canonical parts in effective source order, duplicate crosses parts.
    entry = catalog['manual_interactions']['orders']
    directory = root/entry['directory']
    (directory/'orders_part_001.json').write_text(json.dumps({'orders':early}))
    (directory/'orders_part_002.json').write_text(json.dumps({'orders':late}))
    entry['parts'] = 2
    save(root,catalog)
    return data


def calculate(dataset, selection=SELECTION):
    return asdict(statistics.calculate_statistics(raw_root=dataset[0],catalog_path=dataset[1],
                                                 analysis_filter=selection,calculated_at=AT))


@pytest.mark.parametrize('selection',[AnalysisFilter(),SELECTION,
    AnalysisFilter('2026-02-01','2026-03-31'),
    AnalysisFilter(SELECTION.start_date,SELECTION.end_date,nomenclature_types=('missing',))])
def test_full_orders_history_short_display_window_product_independence_and_indexed_parity(dataset,selection):
    result = calculate(dataset,selection)
    c = result['cohort_statistics']
    strict = c['policies'][POLICIES[0]]
    assert c['as_of_date'] == (selection.end_date or '2026-10-30')
    if selection.start_date is None:
        assert '2020-01' in strict['cohorts']
        assert strict['cohorts']['2020-01']['cohort_size'] == 1  # old -> new canonical alias.
    else:
        assert strict['hidden_buyers'] == (1 if selection.start_date == SELECTION.start_date else 3)
        assert '2020-01' not in strict['cohorts']
        assert strict['base_buyers'] == 3  # historical/new, visible, single; first Return excludes boundary.
    if selection.start_date == SELECTION.start_date:
        assert strict['cohorts']['2026-01']['cohort_size'] == 2
        assert c['policies'][POLICIES[1]]['cohorts']['2026-01']['cohort_size'] == 3
        assert strict['cohorts']['2026-01']['second_purchase_days']['distribution'] == [[31,1]]
    build(dataset)
    assert asdict(query(dataset,selection,calculated_at=AT)) == result
    cache.validate_result(result)


def test_product_filters_never_change_cohort_data(dataset):
    expected = calculate(dataset)['cohort_statistics']
    for values in (('missing',), ('Рубашки',), (None,)):
        selected = AnalysisFilter(SELECTION.start_date,SELECTION.end_date,nomenclature_types=values)
        assert calculate(dataset,selected)['cohort_statistics'] == expected


def test_single_orders_pass_adaptation_fact_and_shared_objects_in_both_backends(dataset,monkeypatch):
    reads, adapted, built, consumed = [], [], [], {name:[] for name in ('buyers','rfm','cohorts')}
    read, adapt, factory = statistics.iter_export,statistics.adapt_order,statistics.build_purchase_order_fact
    def observe_read(name,**kwargs):
        reads.append(name)
        yield from read(name,**kwargs)
    def observe_adapt(raw,*args,**kwargs):
        adapted.append(raw['ids']['mindboxId'])
        return adapt(raw,*args,**kwargs)
    def observe_fact(*args,**kwargs):
        built.append(kwargs['order_id'])
        return factory(*args,**kwargs)
    monkeypatch.setattr(statistics,'iter_export',observe_read)
    monkeypatch.setattr(statistics,'adapt_order',observe_adapt)
    monkeypatch.setattr(statistics,'build_purchase_order_fact',observe_fact)
    for cls,name in ((statistics.BuyerAggregates,'buyers'),(statistics.RfmAggregates,'rfm'),(statistics.CohortAggregates,'cohorts')):
        original = cls.add
        def observe(self,fact,_add=original,_name=name):
            consumed[_name].append(fact)
            return _add(self,fact)
        monkeypatch.setattr(cls,'add',observe)
    expected = calculate(dataset)
    assert Counter(reads)['orders'] == Counter(reads)['actions'] == Counter(reads)['customer_merges'] == 1
    assert Counter(adapted) == Counter(built)
    assert Counter(built)['secret-historical'] == 1
    assert Counter(built)['secret-boundary'] == 2  # Independent first snapshots of full and short scopes.
    assert len({id(f) for f in consumed['cohorts']} & {id(f) for f in consumed['rfm']} &
               {id(f) for f in consumed['buyers']}) >= 4
    build(dataset)
    statements = []
    records = query_module.IndexedInput.records
    def traced_records(self,name,sources):
        if name=='orders':
            self.db.set_trace_callback(statements.append)
        yield from records(self,name,sources)
    monkeypatch.setattr(query_module.IndexedInput,'records',traced_records)
    assert asdict(query(dataset,SELECTION,calculated_at=AT)) == expected
    assert sum(sql.startswith('SELECT * FROM orders') for sql in statements) == 1
    assert not any(sql.startswith('SELECT * FROM actions') for sql in statements)


@pytest.mark.parametrize('selection',[AnalysisFilter(),SELECTION,
    AnalysisFilter(SELECTION.start_date,SELECTION.end_date,nomenclature_types=('missing',))])
def test_every_existing_value_is_identical_when_cohort_routing_is_disabled(dataset,monkeypatch,selection):
    expected = calculate(dataset,selection)
    expected.pop('cohort_statistics')
    monkeypatch.setattr(statistics,'cohort_cutoff',lambda *args:None)
    baseline = calculate(dataset,selection)
    assert baseline.pop('cohort_statistics') is None
    assert baseline == expected  # Includes every legacy, BuyerAggregates and RFM field.


def test_coverage_gap_keeps_observed_positives_but_never_invents_zero_rates(dataset):
    root,_,catalog = dataset
    catalog['manual_interactions'] = None
    catalog['orders'] = {
        '2020-01-05':put_entry(root,'orders',[raw_order('old','2020-01-05','old')],304,
                              '2020-01-05T00:00:00Z','2020-01-06T00:00:00Z'),
        '2026-08-31':put_entry(root,'orders',[raw_order('later','2026-08-31','new')],305,
                              '2026-08-31T00:00:00Z','2026-09-01T00:00:00Z')}
    catalog['actions'] = {}
    save(root,catalog)
    result = calculate(dataset,AnalysisFilter())
    strict = result['cohort_statistics']['policies'][POLICIES[0]]
    count = strict['cohorts']['2020-01']['horizons']['365']
    assert count['eligible'] == 0 and count['coverage_incomplete'] == 1 and count['repeat_rate'] is None
    assert strict['cohorts']['2020-01']['retention'][3]['retention_rate'] is None
    build(dataset)
    assert asdict(query(dataset,AnalysisFilter(),calculated_at=AT)) == result


@pytest.mark.parametrize('selection',[AnalysisFilter(),SELECTION])
def test_no_orders_has_no_invented_cutoff_or_cohorts(dataset,selection):
    root,_,catalog = dataset
    catalog['orders'] = {}
    catalog['manual_interactions'] = None
    save(root,catalog)
    result = calculate(dataset,selection)
    if selection.end_date is None:
        assert result['cohort_statistics'] is None
    else:
        assert result['cohort_statistics']['policies'][POLICIES[0]]['cohorts'] == {}
    cache.validate_result(result)
    build(dataset)
    assert asdict(query(dataset,selection,calculated_at=AT)) == result


@pytest.mark.parametrize('damage',['id','policy','month','date','cutoff','coverage','negative','bool','eligible',
    'rate','null','nan','censoring','overlap','observed','m0','offset','calendar','retention','summary',
    'histogram','incomplete','missing','bounds','unknown_quality'])
def test_closed_validation_rejects_invalid_or_individual_data(dataset,damage):
    result = calculate(dataset)
    c = result['cohort_statistics']
    p = c['policies'][POLICIES[0]]
    r = p['cohorts']['2026-01']
    h = r['horizons']['30']
    if damage=='id': r['customer_id']='secret'
    elif damage=='policy': c['policies']['unknown']=p
    elif damage=='month': p['cohorts']['2026-13']=p['cohorts'].pop('2026-01')
    elif damage=='date': c['as_of_date']='2026-08-30'
    elif damage=='cutoff': c['cutoff_exclusive']='2026-09-02T00:00:00+00:00'
    elif damage=='coverage': c['effective_coverage']=[]
    elif damage=='negative': h['repeated']=-1
    elif damage=='bool': h['eligible']=True
    elif damage=='eligible': h['repeated']=h['eligible']+1
    elif damage=='rate': h['repeat_rate']=101
    elif damage=='null': h['repeat_rate']=None
    elif damage=='nan': h['repeat_rate']=float('nan')
    elif damage=='censoring': h['censored']+=1
    elif damage=='overlap': h['censored_and_incomplete']=20
    elif damage=='observed': h['observed_repeated']+=1
    elif damage=='m0': r['retention'][0]['buyers']+=1
    elif damage=='offset': r['retention'][12]['month_offset']=13
    elif damage=='calendar': r['retention'][1]['month']='2026-03'
    elif damage=='retention': r['retention'][1]['eligible']=0
    elif damage=='summary': p['summary']['cohort_size']+=1
    elif damage=='histogram': r['second_purchase_days']['distribution'][0][1]+=1
    elif damage=='incomplete': r['retention'][1]['coverage_incomplete']+=1
    elif damage=='missing': del r['horizons']['180']
    elif damage=='bounds': c['cohort_bounds']['start']='2020-01'
    else: c['quality']['customer_ids']=[]
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


def test_schema10_new_and_legacy_results_roundtrip_without_inventing_cohorts(dataset,tmp_path):
    result = calculate(dataset)
    path = tmp_path/'cohort-cache.json'
    cache.save_result(path,result)
    assert cache.load_result(path)==json.loads(json.dumps(result))
    result.pop('cohort_statistics')
    cache.save_result(path,result)
    assert cache.load_result(path)==json.loads(json.dumps(result))
    assert 'cohort_statistics' not in cache.load_result(path)
    assert json.loads(path.read_text(encoding='utf-8'))['schema_version']==10


@pytest.mark.parametrize('name',['statistics_cohorts_core.py','statistics_cohorts_validation.py'])
def test_cohort_algorithm_dependencies_invalidate_provenance(name,monkeypatch):
    before = slices._algorithm_version()
    original = slices._hash_file
    monkeypatch.setattr(slices,'_hash_file',lambda path:'changed' if path.name==name else original(path))
    assert slices._algorithm_version()!=before


def test_indexed_full_history_keeps_adapter_errors_fail_closed(dataset):
    root,_,catalog = dataset
    entry = catalog['manual_interactions']['orders']
    path = root/entry['directory']/'orders_part_001.json'
    raw = json.loads(path.read_text())
    raw['orders'][0]['lines'][0]['priceOfLine']='not-a-number'
    path.write_text(json.dumps(raw))
    with pytest.raises(AdapterError):
        calculate(dataset)
    build(dataset)
    with pytest.raises(AdapterError):
        query(dataset,SELECTION,calculated_at=AT)


@pytest.mark.parametrize('scenario',['not_mature','coverage_gap'])
def test_structurally_consistent_rates_still_require_possible_observation(dataset,scenario):
    if scenario=='coverage_gap':
        manual = dataset[2]['manual_interactions']
        for item in (manual,manual['orders'],manual['actions']):
            item.update(since='2026-01-01T00:00:00Z',until='2026-01-10T00:00:00Z')
        save(dataset[0],dataset[2])
    result = calculate(dataset)
    p = result['cohort_statistics']['policies'][POLICIES[0]]
    h = '365' if scenario=='not_mature' else '30'
    c = p['cohorts']['2026-01']['horizons'][h]
    n, observed = c['cohort_size'], c['observed_repeated']
    c.update(eligible=n,repeated=observed,censored=0,coverage_incomplete=0,censored_and_incomplete=0,
             observed_repeated_ineligible=0,confirmed_without_event=n-observed,
             repeat_rate=100*observed/n,status='FULL',reasons=[])
    p['summary']['horizons'][h] = deepcopy(c)
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)



def test_consistent_second_purchase_interval_cannot_extend_beyond_cutoff(dataset):
    result = calculate(dataset)
    p = result['cohort_statistics']['policies'][POLICIES[0]]
    for block in (p['cohorts']['2026-01'],p['summary']):
        block['second_purchase_days'].update(distribution=[[999999,1]],mean=999999,median=999999)
        for c in block['horizons'].values():
            c.update(repeated=0,observed_repeated=0,observed_repeated_ineligible=0,
                     confirmed_without_event=c['eligible'],repeat_rate=0 if c['eligible'] else None)
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


@pytest.mark.parametrize('name', ['statistics_cohorts_core.py', 'statistics_cohorts_validation.py'])
def test_existing_index_rejects_changed_cohort_dependency_without_publication(dataset, monkeypatch, name):
    from Application import statistics_index as index

    build(dataset)
    pointer = dataset[0]/'analytics/current.json'
    before = pointer.read_bytes()
    original = slices._hash_file
    monkeypatch.setattr(slices, '_hash_file', lambda path: '0'*64 if path.name == name else original(path))
    with pytest.raises(index.StatisticsIndexError, match='STALE_SOURCE'):
        query(dataset, SELECTION, calculated_at=AT)
    assert pointer.read_bytes() == before
