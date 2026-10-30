"""Synthetic driver/SQLite parity, period routing, cache and RFM validation."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json

import pytest

from Application import dataset_statistics as statistics, statistics_cache as cache, statistics_slices as slices
from Application import statistics_index as index
from Application.analysis_filter import AnalysisFilter
from Application.statistics_rfm_core import POLICIES
from Application.statistics_rfm_validation import rfm_cutoff
from tests.test_dataset_statistics import action, order, put_entry, save, DEFAULT_SELECTION
from tests.test_statistics_index import dataset as index_fixture, build, query
from tests.test_analysis_filter import reference, sources


AT = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
SELECTION = AnalysisFilter('2026-01-01','2026-08-31')


def raw_order(key, day, customer='new', statuses=('F',), namespace='offline1C'):
    raw = order(statuses, key)
    raw['firstAction']['dateTimeUtc'] = day+'T12:00:00Z'
    raw['customer']['ids']['mindboxId'] = customer
    for line in raw['lines']:
        line['product']['ids'] = {namespace:'000001-size'}
    return raw


def records():
    first_return = raw_order('boundary','2024-08-31',statuses=('Return',))
    mixed = raw_order('bad-mixed','2026-05-01','bad',('F','F'))
    mixed['lines'][1]['product']['ids'] = {'kanzlerKz':'000001-size'}
    return [raw_order('lower','2022-09-01','old'), raw_order('historic','2023-08-31','historic'),
            first_return, raw_order('boundary','2026-02-01'),
            raw_order('previous','2025-08-31','old'), raw_order('current','2026-08-31','new',('F','CP')),
            raw_order('kzt','2026-07-01','new',namespace='kanzlerKz'),
            raw_order('partial','2026-03-01','other',('CP',)),
            raw_order('bad-known','2026-04-01','bad'), mixed,
            raw_order('future','2027-01-01'), raw_order('too-early','2022-08-31')]


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    data = index_fixture.__wrapped__(tmp_path,monkeypatch)
    reference(data)
    root,_,catalog = data
    events = records()
    events.insert(1,deepcopy(events[0]))
    sources(data,[action(DEFAULT_SELECTION.view_action_system_names[0],['000001-size'])],events,
            since='2022-09-01T00:00:00+00:00',until='2026-09-01T00:00:00+00:00')
    return data


def calculate(dataset, selection=SELECTION):
    return asdict(statistics.calculate_statistics(raw_root=dataset[0],catalog_path=dataset[1],
                                                 analysis_filter=selection,calculated_at=AT))


@pytest.mark.parametrize('product_filter',[{},dict(nomenclature_types=('missing',)),dict(collections=('Осень-зима',))])
def test_short_period_full_48m_product_independence_aliases_first_snapshot_and_indexed_parity(dataset, product_filter):
    selection = AnalysisFilter(SELECTION.start_date,SELECTION.end_date,**product_filter)
    expected = calculate(dataset,selection)
    rfm = expected['rfm_statistics']
    assert rfm == calculate(dataset)['rfm_statistics']
    assert rfm['as_of_date'] == '2026-08-31'
    strict = rfm['policies'][POLICIES[0]]
    assert strict['base_buyers'] == 3  # old/new merge; historic; bad.
    assert strict['windows']['LAST_12M']['behavior']['observed_orders'] == 4
    assert strict['windows']['LAST_48M']['behavior']['observed_orders'] == 7
    assert strict['windows']['LAST_12M']['behavior']['without_observed_orders'] == 1
    assert strict['windows']['LAST_48M']['currencies']['RUB']['buyers'] == 3
    assert strict['windows']['LAST_12M']['currencies']['KZT']['buyers'] == 1
    assert strict['windows']['LAST_12M']['quality']['MIXED_CURRENCY'] == 1
    # First Return in the RFM area does not block the independently selected
    # first F of buyer_statistics' short user period.
    assert expected['buyer_statistics']['policies'][POLICIES[0]]['orders'] == 5
    build(dataset)
    assert asdict(query(dataset,selection,calculated_at=AT)) == expected
    cache.validate_result(expected)


def test_one_source_pass_one_adaptation_and_one_fact_per_selected_occurrence(dataset,monkeypatch):
    reads, adapters, factories, consumers = [], [], [], []
    read,adapt,factory = statistics.iter_export,statistics.adapt_order,statistics.build_purchase_order_fact
    add = statistics.BuyerAggregates.add
    def observed_read(name,**kwargs):
        reads.append(name)
        yield from read(name,**kwargs)
    def observed_adapt(raw,*args,**kwargs):
        adapters.append(raw['ids']['mindboxId'])
        return adapt(raw,*args,**kwargs)
    def observed_fact(*args,**kwargs):
        factories.append(kwargs['order_id'])
        return factory(*args,**kwargs)
    def observed_buyer(self,fact):
        consumers.append(fact)
        return add(self,fact)
    rfm_add = statistics.RfmAggregates.add
    rfm_objects = []
    def observed_rfm(self,fact):
        rfm_objects.append(fact)
        return rfm_add(self,fact)
    monkeypatch.setattr(statistics,'iter_export',observed_read)
    monkeypatch.setattr(statistics,'adapt_order',observed_adapt)
    monkeypatch.setattr(statistics,'build_purchase_order_fact',observed_fact)
    monkeypatch.setattr(statistics.BuyerAggregates,'add',observed_buyer)
    monkeypatch.setattr(statistics.RfmAggregates,'add',observed_rfm)
    calculate(dataset)
    assert Counter(reads)['orders'] == Counter(reads)['actions'] == Counter(reads)['customer_merges'] == 1
    assert Counter(adapters) == Counter(factories)
    assert Counter(factories)['lower'] == 1  # Identical duplicate is not adapted again.
    assert Counter(factories)['boundary'] == 2  # Different first snapshots for two different areas.
    assert len({id(f) for f in consumers}&{id(f) for f in rfm_objects}) >= 4


def test_all_existing_values_including_buyer_statistics_match_without_rfm_routing(dataset,monkeypatch):
    expected = calculate(dataset)
    expected.pop('rfm_statistics')
    with monkeypatch.context() as patch:
        patch.setattr(statistics,'rfm_cutoff',lambda *args:None)
        baseline = calculate(dataset)
    baseline.pop('rfm_statistics')
    assert expected == baseline


@pytest.mark.parametrize('selected,at,coverage,expected',[
    ('2026-08-31','2026-10-30T12:00:00+00:00',[],'2026-08-31'),
    (None,'2026-10-30T12:00:00+00:00',[('2026-01-01','2026-09-01')],'2026-08-31'),
    (None,'2026-10-30T12:00:00+00:00',[('2026-01-01','2027-01-01')],'2026-10-30'),
    (None,'2026-10-30T00:30:00+03:00',[('2026-01-01','2027-01-01')],'2026-10-29'),
    (None,'2026-10-30T12:00:00+00:00',[('2025-01-01','2025-02-01'),('2027-01-01','2028-01-01')],'2025-01-31'),
    (None,'2026-10-30T12:00:00+00:00',[],None),
    (None,'2026-10-30T12:00:00+00:00',[('2027-01-01','2028-01-01')],None),
])
def test_deterministic_cutoff_does_not_use_latest_purchase_or_local_demographic_day(selected,at,coverage,expected):
    observed = [(datetime.fromisoformat(a).replace(tzinfo=timezone.utc),datetime.fromisoformat(b).replace(tzinfo=timezone.utc)) for a,b in coverage]
    actual = rfm_cutoff(observed,selected,datetime.fromisoformat(at))
    assert (actual.isoformat() if actual else None) == expected


@pytest.mark.parametrize('selection',[AnalysisFilter(),SELECTION])
def test_missing_orders_no_invented_cutoff_and_reference_indexed_parity(dataset,selection):
    dataset[2]['orders'] = {}
    dataset[2]['manual_interactions'] = None
    dataset[2]['actions'] = {'2026-01-01':put_entry(dataset[0],'actions',[],205)}
    save(dataset[0],dataset[2])
    result = calculate(dataset,selection)
    if selection.end_date is None:
        assert result['rfm_statistics'] is None
    else:
        assert all(m['status']=='NONE' for m in result['rfm_statistics']['windows'].values())
    cache.validate_result(result)
    build(dataset)
    assert asdict(query(dataset,selection,calculated_at=AT)) == result


def test_gap_keeps_observed_history_and_unscorable_scores(dataset):
    root,_,data = dataset
    data['manual_interactions'] = None
    data['actions'] = {'2026-08-31':put_entry(root,'actions',[
        action(DEFAULT_SELECTION.view_action_system_names[0],['000001-size'],stamp='2026-08-31T12:00:00Z')],206,
        '2026-08-31T00:00:00Z','2026-09-01T00:00:00Z')}
    data['orders'] = {'2023-08-31':put_entry(root,'orders',[raw_order('historical','2023-08-31')],203,
        '2023-08-31T00:00:00Z','2023-09-01T00:00:00Z'),
        '2026-08-31':put_entry(root,'orders',[raw_order('current','2026-08-31')],204,
        '2026-08-31T00:00:00Z','2026-09-01T00:00:00Z')}
    save(root,data)
    result = calculate(dataset)
    rfm = result['rfm_statistics']
    assert rfm['windows']['LAST_12M']['status'] == 'PARTIAL'
    rub = rfm['policies'][POLICIES[0]]['windows']['LAST_12M']['currencies']['RUB']
    assert rub['score_counts'] == [[None,None,None,1]]
    assert rub['monetary']['buyers'] == 0
    cache.validate_result(result)
    build(dataset)
    assert asdict(query(dataset,SELECTION,calculated_at=AT)) == result


@pytest.mark.parametrize('damage',['id','policy','currency','window','date','coverage','count','negative',
                                   'percentage','money','nan','threshold','score','histogram','null','comparison'])
def test_invalid_nested_result_is_rejected_without_repair(dataset,damage):
    result = calculate(dataset)
    r = result['rfm_statistics']
    p = r['policies'][POLICIES[0]]
    c = p['windows']['LAST_12M']['currencies']['RUB']
    if damage=='id': c['customer_id']='secret'
    elif damage=='policy': r['policies']['unknown']=p
    elif damage=='currency': p['windows']['LAST_12M']['currencies']['USD']=c
    elif damage=='window': r['windows']['LAST_12M']['start']='2026-01-01'
    elif damage=='date': r['as_of_date']='2026-08-30'
    elif damage=='coverage': r['windows']['LAST_12M']['status']='FULLY'
    elif damage=='count': c['buyers']+=1
    elif damage=='negative': c['valued_orders']=-1
    elif damage=='percentage': c['segments'][0][2]=101
    elif damage=='money': c['known_amount']='-1'
    elif damage=='nan': c['monetary']['mean']='NaN'
    elif damage=='threshold': c['thresholds']['F']['values']=[999]*4
    elif damage=='score': c['score_counts'][0][0]=6
    elif damage=='histogram': c['frequency']['distribution'][0][1]+=1
    elif damage=='null': c['confirmed_inactive_buyers']=None
    else: p['comparison']['order_change']+=1
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


def test_old_schema10_and_new_snapshot_roundtrip_no_personal_fields(dataset,tmp_path):
    result = calculate(dataset)
    path = tmp_path/'rfm-cache.json'
    cache.save_result(path,result)
    assert cache.load_result(path)==json.loads(json.dumps(result))
    old = deepcopy(result)
    old.pop('rfm_statistics')
    cache.save_result(path,old)
    assert cache.load_result(path)==json.loads(json.dumps(old))
    assert json.loads(path.read_text(encoding='utf-8'))['schema_version']==10
    encoded = json.dumps(result['rfm_statistics'])
    assert all(x not in encoded for x in ('customer_id','order_id','identity_space','historic','boundary'))


def test_utc_cutoff_changes_slice_but_not_index_generation(dataset):
    a,_ = slices.capture_sources(dataset[0],dataset[1],AnalysisFilter(),calculated_at=datetime(2026,8,30,21,30,tzinfo=timezone.utc))
    b,_ = slices.capture_sources(dataset[0],dataset[1],AnalysisFilter(),calculated_at=datetime(2026,8,31,0,30,tzinfo=timezone.utc))
    assert a['rfm_as_of']=='2026-08-30' and b['rfm_as_of']=='2026-08-31'
    assert a!=b and index.source_version(a)==index.source_version(b)


@pytest.mark.parametrize('name',['statistics_rfm_core.py','statistics_rfm_validation.py'])
def test_rfm_algorithm_dependencies_are_attested(name,monkeypatch):
    before = slices._algorithm_version()
    original = slices._hash_file
    monkeypatch.setattr(slices,'_hash_file',lambda path:'changed' if path.name==name else original(path))
    assert slices._algorithm_version()!=before
