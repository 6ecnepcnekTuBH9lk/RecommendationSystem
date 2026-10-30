"""Synthetic buyer metrics, bounded state, strict cache contract and driver parity."""

from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime, timezone
from decimal import Decimal
import json
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

from Application import dataset_statistics as statistics
from Application import statistics_cache as cache
from Application import statistics_slices as slices
from Application.analysis_filter import AnalysisFilter
from Application.statistics_buyer_aggregates import BuyerAggregates, POLICIES, saved_coverage
from Application.statistics_buyer_validation import validate_buyer_statistics
from Application.statistics_purchase_facts import build_purchase_order_fact
from tests.test_dataset_statistics import order, action, put_entry, save, DEFAULT_SELECTION
from tests.test_statistics_index import dataset as index_fixture, build, query
from tests.test_analysis_filter import reference


AT = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
COVERAGE = [(datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2027, 1, 1, tzinfo=timezone.utc))]


def purchase(customer='buyer', key='order', stamp='2026-01-01T12:00:00Z', lines=(('F', 1, '10', 'offline1C'),)):
    items = [SimpleNamespace(line_status=status, quantity=quantity,
                             price_of_line=Decimal(amount) if isinstance(amount, str) else amount,
                             product=SimpleNamespace(namespace=namespace)) for status, quantity, amount, namespace in lines]
    return build_purchase_order_fact(items, identity_space='synthetic', customer_id=customer, order_id=key,
                                      order_datetime_utc=datetime.fromisoformat(stamp.replace('Z', '+00:00')))


def aggregate(facts=(), selection=AnalysisFilter(), coverage=COVERAGE, *, profiles=True):
    result = BuyerAggregates('synthetic', selection, coverage, orders_available=bool(coverage))
    result.profiles_available = profiles
    for fact in facts:
        result.add(fact)
    return result


def result(aggregator):
    value = aggregator.result(AT.date())
    validate_buyer_statistics(value, coverage=saved_coverage(COVERAGE) if aggregator.orders_available else [],
                               profiles_available=aggregator.profiles_available)
    return value


def test_frequency_intervals_calendar_order_money_and_first_last_months():
    data = aggregate([
        purchase('a', 'second', lines=(('F', 1, '0', 'offline1C'), ('CP', 2, '20', 'offline1C'), ('Return', 1, '50', 'offline1C'))),
        purchase('a', 'first', '2025-12-31T23:59:59Z', (('F', 3, '100', 'offline1C'),)),
        purchase('a', 'third', '2026-01-01T22:00:00Z', (('sold', 1, '40', 'kanzlerKz'),)),
        purchase('b', 'partial', '2026-01-02T01:00:00Z', (('CP', 1, '50', 'offline1C'),)),
        purchase('c', 'paid', lines=(('delivering', 1, '90', 'offline1C'),)),
    ])
    data.profile('a', 'male', '2000-02-29', AT.date())
    data.profile('a', 'female', '1990-01-01', AT.date())  # Alias profile never replaces the first.
    data.profile('b', 'female', None, AT.date())
    actual = result(data)
    strict, extended = (actual['policies'][p] for p in POLICIES)
    assert (strict['buyers'], strict['orders'], strict['positions'], strict['units']) == (1, 3, 3, '5')
    assert (extended['buyers'], extended['orders'], extended['positions'], extended['units']) == (2, 4, 5, '8')
    assert strict['repeat_rate'] == 100 and extended['repeat_rate'] == 50
    assert strict['mean_orders'] == strict['median_orders'] == 3
    assert extended['mean_orders'] == extended['median_orders'] == 2
    assert strict['frequency'][2] == ['3–5', 1, 100., 3]
    assert strict['first_date'] == '2025-12-31' and strict['last_date'] == '2026-01-01'
    assert strict['first_months'] == [['2025-12', 1, 100.]]
    assert strict['last_months'] == [['2026-01', 1, 100.]]
    assert strict['intervals']['count'] == 2 and strict['intervals']['total_days'] == 1
    assert strict['intervals']['mean_days'] == strict['intervals']['median_days'] == .5
    rub, kzt = strict['money']['RUB'], strict['money']['KZT']
    assert rub['known_orders'] == 2 and Decimal(rub['amount']) == 100
    assert Decimal(rub['mean_order']) == Decimal(rub['median_order']) == 50
    assert rub['complete_buyers'] == 1 and Decimal(rub['complete_amount']) == 100
    assert kzt['known_orders'] == 1 and Decimal(kzt['amount']) == 40
    assert Decimal(extended['money']['RUB']['amount']) == 170
    assert extended['money']['RUB']['repeat_amount_rate'] == pytest.approx(120/170*100)
    assert [v['year'] for v in strict['annual']] == ['2025', '2026']
    assert strict['annual'][0]['complete_year'] is True
    assert strict['demography']['gender'] == [['Мужчины', 1, 100.], ['Женщины', 0, 0.], ['Не указан', 0, 0.]]
    assert strict['demography']['mean_age'] == 26
    assert extended['demography']['unknown_age'] == 1
    assert actual['return_positions'] == 1
    payload = json.dumps(actual)
    assert '"a"' not in payload and '"second"' not in payload and 'identity_space' not in payload
    assert list(data.buyers['a'].days) == [date(2026,1,1).toordinal(), date(2025,12,31).toordinal(), date(2026,1,1).toordinal()]
    assert data.buyers['a'].days.itemsize == 4 and data.buyers['a'].strict_flags.itemsize == 1


def test_known_order_subtotals_do_not_become_complete_buyer_turnover():
    data = aggregate([
        purchase('partial', 'known'), purchase('partial', 'unknown', lines=(('F', 1, '20', 'unsupported'),)),
        purchase('zero', 'zero', lines=(('F', 1, '0', 'offline1C'),)),
        purchase('bad-price', 'bad', lines=(('F', 1, None, 'offline1C'),)),
        purchase('mixed', 'mixed', lines=(('F', 1, '20', 'offline1C'), ('F', 1, '30', 'kanzlerKz'))),
    ])
    value = result(data)['policies'][POLICIES[0]]
    assert value['buyers'] == 4 and value['orders'] == 5
    assert value['quality'] == dict(complete_buyers=1, partial_buyers=1, unvalued_buyers=2,
                                    unvalued_orders=3, unknown_currency_orders=1, mixed_currency_orders=1, invalid_price_orders=1)
    rub = value['money']['RUB']
    assert rub['known_buyers'] == 2 and rub['known_orders'] == 2
    assert Decimal(rub['amount']) == 10 and Decimal(rub['mean_known_per_buyer']) == 5
    assert rub['complete_buyers'] == 1 and Decimal(rub['complete_amount']) == 0
    assert Decimal(rub['median_complete_per_buyer']) == 0
    assert rub['quintiles'] == [['0', '0', 1, 100., '0', 0, 0., '0', None]]
    assert rub['repeat_amount_rate'] == 100 and rub['complete_repeat_amount_rate'] is None


def test_tie_preserving_quintiles_and_two_distinct_monetary_denominators():
    data = aggregate(purchase(str(i), str(i), lines=(('F', 1, str(value), 'offline1C'),))
                     for i, value in enumerate([0]*6+[10]*3+[100]))
    rub = result(data)['policies'][POLICIES[0]]['money']['RUB']
    assert [(r[0],r[1],r[2]) for r in rub['quintiles']] == [('0','0',6),('10','10',3),('100','100',1)]
    assert Decimal(rub['median_complete_per_buyer']) == 0
    assert Decimal(rub['mean_complete_per_buyer']) == 13


@pytest.mark.parametrize('profiles', [False, True])
def test_empty_available_orders_and_missing_profiles(profiles):
    value = result(aggregate(profiles=profiles))
    for p in value['policies'].values():
        assert p['buyers'] == p['orders'] == 0
        assert p['repeat_rate'] is p['mean_orders'] is p['intervals']['median_days'] is None
        assert (p['demography'] is not None) == profiles
        assert p['money']['RUB']['mean_order'] is None
        assert [row['year'] for row in p['annual']] == ['2025','2026']
        assert all(row['buyers']==row['orders']==0 for row in p['annual'])
    missing = aggregate(coverage=[], profiles=profiles).result(AT.date())
    validate_buyer_statistics(missing, coverage=[], profiles_available=profiles)
    assert all(v is None for v in missing['policies'].values())
    assert missing['return_positions'] is None


def test_no_coverage_is_not_a_measured_zero_and_gaps_are_explicit():
    selected = AnalysisFilter('2024-01-01','2024-12-31')
    value = aggregate(selection=selected).result(AT.date())
    validate_buyer_statistics(value, coverage=saved_coverage(COVERAGE), profiles_available=True)
    assert value['orders_available'] and not value['coverage'] and value['policies'][POLICIES[0]] is None
    coverage = [(datetime(2026,1,1,tzinfo=timezone.utc),datetime(2026,1,2,tzinfo=timezone.utc)),
                (datetime(2026,1,10,tzinfo=timezone.utc),datetime(2026,1,11,tzinfo=timezone.utc))]
    data = aggregate([purchase(key='a'),purchase(key='b',stamp='2026-01-10T00:00:00Z')], coverage=coverage)
    value = data.result(AT.date())
    validate_buyer_statistics(value, coverage=saved_coverage(coverage), profiles_available=True)
    assert not value['coverage_complete'] and value['policies'][POLICIES[0]]['intervals']['mean_days'] == 9


def test_leap_day_and_calendar_year_intervals_not_read_order():
    data = aggregate([purchase(key='later',stamp='2024-03-01T00:00:00Z'),
                      purchase(key='earlier',stamp='2024-02-28T23:59:59Z')],
                     coverage=[(datetime(2024,1,1,tzinfo=timezone.utc),datetime(2025,1,1,tzinfo=timezone.utc))])
    value = data.result(AT.date())
    validate_buyer_statistics(value, coverage=saved_coverage(data.coverage), profiles_available=True)
    assert value['policies'][POLICIES[0]]['intervals']['mean_days'] == 2


def test_private_identity_spaces_and_cancellation_are_not_silent():
    data = aggregate([purchase()])
    with pytest.raises(ValueError, match='identity spaces'):
        data.add(SimpleNamespace(identity_space='other'))
    with pytest.raises(ValueError, match='observation'):
        data.add(purchase(stamp='2020-01-01T00:00:00Z'))
    def stop():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        data.result(AT.date(), stop)
    assert result(data)['policies'][POLICIES[0]]['buyers'] == 1


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    data = index_fixture.__wrapped__(tmp_path, monkeypatch)
    reference(data)
    root, _, catalog = data
    with closing(sqlite3.connect(root/'canonical/customers.sqlite')) as db, db:
        db.execute("UPDATE profiles SET raw=? WHERE id='new'", (json.dumps({'sex':'female','birthDate':'2000-02-29'}),))
        db.execute("INSERT INTO profiles VALUES ('old','',?)", (json.dumps({'sex':'male','birthDate':'1990-01-01'}),))
    first = order(['F','CP','Return'], 'same')
    first['customer']['ids']['mindboxId'] = 'old'
    identical, conflict = deepcopy(first), deepcopy(first)
    conflict['customer']['ids']['mindboxId'] = 'phantom'
    conflict['lines'][0]['status']['ids']['externalId'] = 'Return'
    later = order(['sold'], 'later')
    later['firstAction']['dateTimeUtc'] = '2026-01-02T12:00:00Z'
    later['lines'][0]['priceOfLine'] = 0
    catalog['manual_interactions'] = None
    catalog['actions'] = {'2026-01-01': put_entry(root,'actions',[action(DEFAULT_SELECTION.view_action_system_names[0], ['000001-size'])],121)}
    catalog['orders'] = {'2026-01-01': put_entry(root,'orders',[first,identical,conflict],122),
                         '2026-01-02': put_entry(root,'orders',[later],123,'2026-01-02T00:00:00+00:00','2026-01-03T00:00:00+00:00')}
    save(root,catalog)
    return data


def test_driver_reference_indexed_full_orders_scope_product_independence_and_single_pass(dataset, monkeypatch):
    build(dataset)
    root, catalog, _ = dataset
    baseline = None
    for selection in (AnalysisFilter(), AnalysisFilter(nomenclature_types=('missing',)),
                      AnalysisFilter(collections=('Весна-лето',))):
        calls = []
        profile_queries = []
        fact_calls = []
        read, adapt = statistics.iter_export, statistics.adapt_order
        factory = statistics.build_purchase_order_fact
        def observed_fact(*args,**kwargs):
            fact_calls.append(kwargs['order_id'])
            return factory(*args,**kwargs)
        connect = sqlite3.connect
        def traced_connect(*args, **kwargs):
            db = connect(*args, **kwargs)
            db.set_trace_callback(lambda sql: profile_queries.append(sql) if
                'SELECT id, json_extract(raw' in sql or 'SELECT customer,sex,born FROM demographics.profiles' in sql else None)
            return db
        def observed(name, **kwargs):
            calls.append(name)
            yield from read(name, **kwargs)
        adapted = []
        def adapting(raw, *args, **kwargs):
            adapted.append(raw['ids']['mindboxId'])
            return adapt(raw, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(statistics,'iter_export',observed)
            patch.setattr(statistics,'adapt_order',adapting)
            patch.setattr(sqlite3,'connect',traced_connect)
            patch.setattr(statistics,'build_purchase_order_fact',observed_fact)
            value = statistics.calculate_statistics(raw_root=root,catalog_path=catalog,analysis_filter=selection,calculated_at=AT)
        assert calls.count('orders') == 2  # One pass per two canonical source entries, no second scan.
        assert adapted == ['same','later']
        assert fact_calls == ['same','later']
        assert len(profile_queries) == 1
        assert value.purchase_orders == (0 if selection.nomenclature_types else 1)  # Legacy shared day only.
        assert value.buyer_statistics['policies'][POLICIES[0]]['orders'] == 2
        assert value.buyer_statistics['policies'][POLICIES[0]]['demography']['profiled_buyers'] == 1
        assert value.buyer_statistics['policies'][POLICIES[0]]['demography']['gender'][1][1] == 1
        assert value.buyer_statistics['policies'][POLICIES[0]]['demography']['mean_age'] == 26
        if baseline is not None:
            assert value.buyer_statistics == baseline
        baseline = value.buyer_statistics
        validate_buyer_statistics(value.buyer_statistics,
                                   coverage=next(v.intervals for v in value.coverage if v.source == 'Orders'),
                                   profiles_available=value.customers is not None, customer_scope=value.customer_scope)
        cache.validate_result(asdict(value))
        with monkeypatch.context() as patch:
            profile_queries.clear()
            patch.setattr(sqlite3,'connect',traced_connect)
            patch.setattr(statistics,'iter_export',lambda *a,**kw: pytest.fail('raw read'))
            assert asdict(query(dataset,selection,calculated_at=AT)) == asdict(value)
        assert len(profile_queries) == 1
    selected = AnalysisFilter('2026-01-01','2026-01-01')
    value = statistics.calculate_statistics(raw_root=root,catalog_path=catalog,analysis_filter=selected,calculated_at=AT)
    assert value.analysis_filter.start_date is None  # Legacy normalization does not erase buyer period.
    assert value.buyer_statistics['policies'][POLICIES[0]]['orders'] == 1
    assert value.buyer_statistics['period'] == dict(start_date='2026-01-01',end_date='2026-01-01')
    assert asdict(query(dataset,selected,calculated_at=AT)) == asdict(value)


def test_last_successful_schema10_and_new_result_roundtrip_without_reading_sources(dataset, tmp_path):
    value = asdict(statistics.calculate_statistics(raw_root=dataset[0],catalog_path=dataset[1],calculated_at=AT))
    path = tmp_path/'saved.json'
    cache.save_result(path,value)
    assert cache.load_result(path)['buyer_statistics'] == value['buyer_statistics']
    assert json.loads(path.read_text(encoding='utf-8'))['schema_version'] == 10
    value.pop('buyer_statistics')
    cache.save_result(path,value)
    assert 'buyer_statistics' not in cache.load_result(path)
    assert all(field not in json.dumps(value) for field in ('"customer_id"','"order_id"','"identity_space"'))


def test_first_orders_scope_snapshot_outside_actions_is_not_replaced_inside_legacy_scope(dataset):
    root,catalog,data = dataset
    data['actions'] = {'2026-01-02':put_entry(root,'actions',[],124,'2026-01-02T00:00:00+00:00','2026-01-03T00:00:00+00:00')}
    first,later = order(['Return'],'same'),order(['F'],'same')
    first['customer']['ids']['mindboxId'] = 'first-owner'
    later['customer']['ids']['mindboxId'] = 'second-owner'
    later['firstAction']['dateTimeUtc'] = '2026-01-02T12:00:00Z'
    for day,raw in (('2026-01-01',first),('2026-01-02',later)):
        path = root/data['orders'][day]['directory']/'orders_part_001.json'
        path.write_text(json.dumps({'orders':[raw]}),encoding='utf-8')
    save(root,data)
    value = statistics.calculate_statistics(raw_root=root,catalog_path=catalog,calculated_at=AT)
    assert value.purchase_users == value.purchase_orders == 1  # Legacy shared-coverage first snapshot.
    assert value.customer_scope['orders'] == 1 and value.customer_scope['buyers'] == 0
    assert all(p['buyers']==p['orders']==0 for p in value.buyer_statistics['policies'].values())
    cache.validate_result(asdict(value))
    build(dataset)
    assert asdict(query(dataset,calculated_at=AT)) == asdict(value)
    selected = AnalysisFilter('2026-01-02','2026-01-02')
    day = statistics.calculate_statistics(raw_root=root,catalog_path=catalog,calculated_at=AT,analysis_filter=selected)
    assert day.buyer_statistics['policies'][POLICIES[0]]['buyers'] == 1
    assert asdict(query(dataset,selected,calculated_at=AT)) == asdict(day)


def test_missing_orders_is_not_an_empty_observation_in_both_backends(dataset):
    root,catalog,data = dataset
    data['orders'] = {}
    save(root,data)
    value = statistics.calculate_statistics(raw_root=root,catalog_path=catalog,calculated_at=AT)
    assert not value.buyer_statistics['orders_available']
    assert all(p is None for p in value.buyer_statistics['policies'].values())
    cache.validate_result(asdict(value))
    build(dataset)
    assert asdict(query(dataset,calculated_at=AT)) == asdict(value)


def test_annual_completeness_is_not_confused_with_full_period_completeness():
    data = aggregate([purchase('a','known','2025-12-31T12:00:00Z'),
                      purchase('a','unknown','2026-01-01T12:00:00Z',(('F',1,None,'offline1C'),))])
    value = result(data)['policies'][POLICIES[0]]
    assert value['quality']['partial_buyers'] == 1
    assert value['money']['RUB']['complete_buyers'] == 0
    assert Decimal(value['annual'][0]['money']['RUB']['complete_amount']) == 10
    assert value['annual'][0]['money']['RUB']['complete_buyers'] == 1
    assert value['annual'][1]['money']['RUB']['known_orders'] == 0


def test_original_period_is_attested_in_slices_even_without_customer_scope(dataset):
    full = asdict(statistics.calculate_statistics(raw_root=dataset[0],catalog_path=dataset[1],calculated_at=AT))
    selected = asdict(statistics.calculate_statistics(raw_root=dataset[0],catalog_path=dataset[1],calculated_at=AT,
                                                      analysis_filter=AnalysisFilter('2026-01-01','2026-01-01')))
    full['customer_scope'] = None
    full.pop('rfm_statistics')  # Isolate buyer-period provenance from the new RFM cutoff context.
    provenance, _ = slices.capture_sources(dataset[0],dataset[1],AnalysisFilter(),calculated_at=AT)
    folder = dataset[0]/'buyer-slices'
    slices.save_slice(folder,provenance,full)
    path = next(folder.glob('*.json'))
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['result']['buyer_statistics'] = selected['buyer_statistics']
    payload['result_sha256'] = slices._digest(payload['result'])
    path.write_text(json.dumps(payload),encoding='utf-8')
    assert slices.load_slice(folder,provenance) is None
    with pytest.raises(ValueError,match='Buyer period'):
        slices.save_slice(folder,provenance,payload['result'])


@pytest.mark.parametrize('name',['statistics_purchase_policy.py','statistics_purchase_facts.py','statistics_buyer_aggregates.py'])
def test_new_business_modules_are_in_algorithm_provenance(name, monkeypatch):
    before = slices._algorithm_version()
    original = slices._hash_file
    monkeypatch.setattr(slices,'_hash_file',lambda path: 'modified' if path.name == name else original(path))
    assert slices._algorithm_version() != before


def test_every_legacy_field_equals_published_driver_including_customer_scope_and_demography(dataset):
    # Read the confirmed published implementation, without replacing workspace
    # files or changing Git. Its dependencies retain their legacy code paths.
    source = subprocess.check_output(['git','show','442189b:Application/dataset_statistics.py']).decode('utf-8')
    previous = ModuleType('Application._buyer_legacy_reference')
    previous.__file__ = statistics.__file__
    sys.modules[previous.__name__] = previous
    try:
        exec(compile(source,previous.__file__,'exec'),previous.__dict__)
        for selection in (AnalysisFilter(),AnalysisFilter(nomenclature_types=('missing',)),
                          AnalysisFilter('2026-01-01','2026-01-01')):
            options = dict(raw_root=dataset[0],catalog_path=dataset[1],calculated_at=AT,analysis_filter=selection)
            expected, actual = asdict(previous.calculate_statistics(**options)), asdict(statistics.calculate_statistics(**options))
            actual.pop('buyer_statistics')
            actual.pop('rfm_statistics')
            assert actual == expected
    finally:
        del sys.modules[previous.__name__]


@pytest.mark.parametrize('corrupt', ['extra_id','percentage','amount','nan','frequency','intervals','policy','quality','coverage','null'])
def test_strict_nested_validation_rejects_inconsistent_or_individual_data(corrupt):
    value = result(aggregate([purchase()]))
    p = value['policies'][POLICIES[0]]
    if corrupt == 'extra_id': p['customer_id'] = 'private'
    elif corrupt == 'percentage': p['repeat_rate'] = 101
    elif corrupt == 'amount': p['money']['RUB']['amount'] = '999'
    elif corrupt == 'nan': p['money']['RUB']['amount'] = 'NaN'
    elif corrupt == 'frequency': p['frequency'][0][1] = 0
    elif corrupt == 'intervals': p['intervals']['count'] = 1
    elif corrupt == 'policy': p['policy'] = 'production'
    elif corrupt == 'quality': p['quality']['complete_buyers'] = 0
    elif corrupt == 'coverage': value['coverage'] = []
    else: p['buyers'] = None
    with pytest.raises((ValueError, ArithmeticError, TypeError)):
        validate_buyer_statistics(value,coverage=saved_coverage(COVERAGE),profiles_available=True)
