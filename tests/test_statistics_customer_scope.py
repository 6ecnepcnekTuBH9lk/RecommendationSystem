"""Source-wide customer coverage; tiny synthetic sources and private indexes."""

from copy import deepcopy
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import json
import sqlite3

import pytest

from Application import dataset_statistics as statistics
from Application import statistics_cache as cache
from Application import statistics_slices as slices
from Application.analysis_filter import AnalysisFilter
from Application.customer_statistics_scope import CustomerScope, validate_customer_scope
from Application.statistics_presentation import customer_scope_sections, customer_scope_context
from tests.test_statistics_index import dataset as indexed_fixture, build, query
from tests.test_dataset_statistics import action, order, write_actions, write_orders, save, put_entry, DEFAULT_SELECTION
from tests.test_analysis_filter import reference

AT = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)


@pytest.fixture
def source(tmp_path, monkeypatch):
    data = indexed_fixture.__wrapped__(tmp_path, monkeypatch)
    reference(data)
    with closing(sqlite3.connect(data[0] / 'canonical/customers.sqlite')) as db, db:
        db.execute("INSERT INTO profiles VALUES('old','',?)", (json.dumps({'sex': 'male'}),))
    view, favorite = DEFAULT_SELECTION.view_action_system_names[0], DEFAULT_SELECTION.favorite_action_system_names[0]
    actions = [action(view, customer='old'), action(favorite, customer='new'),
               action(view, ['999999-a'], customer='event-only'),
               action(favorite, ['000001-a'], customer='favorites-only'),
               action(view, ['000001-a', '000002-a'], customer='both-buyer'),
               action(view, ['000001-a'], customer='both-buyer'), action('Unknown', customer='unknown-only')]
    for kind in ('Unknown', view):
        raw = action(kind)
        raw.pop('customer')
        actions.append(raw)
    write_actions(data, actions)
    first = order(['CP'], 'first')
    first['customer']['ids']['mindboxId'] = 'both-buyer'
    first['lines'][0]['product']['ids']['offline1C'] = '000002-a'
    conflict = deepcopy(first)
    conflict['customer']['ids']['mindboxId'] = 'phantom-duplicate'
    returned = order(['Return'], 'returned')
    returned['customer']['ids']['mindboxId'] = 'orders-only'
    zero = order(['F'], 'zero')
    zero['lines'][0]['quantity'] = 0
    zero['lines'][0]['product']['ids']['offline1C'] = '000002-a'
    fraction = order(['CP'], 'fraction')
    fraction['lines'][0]['quantity'] = 1.5
    future = order(['CP'], 'future')
    future['customer']['ids']['mindboxId'] = 'future-only'
    future['firstAction']['dateTimeUtc'] = '2026-01-02T12:00:00Z'
    write_orders(data, [first, first, conflict, returned, zero, fraction, future])
    return data


def calculate(source, selection=AnalysisFilter()):
    return statistics.calculate_statistics(raw_root=source[0], catalog_path=source[1], analysis_filter=selection,
                                           calculated_at=AT)


def test_source_unions_aliases_itemless_and_product_independence_exact_indexed_parity(source, monkeypatch):
    build(source)
    expected_scope = None
    for selected in (AnalysisFilter(), AnalysisFilter(nomenclature_types=('Рубашки',)),
                     AnalysisFilter(collections=('missing',))):
        result = calculate(source, selected)
        scope = result.customer_scope
        assert {k: scope[k] for k in ('profiles', 'actions', 'orders', 'events', 'profile_events', 'identified',
                                      'without_profile', 'profile_only', 'buyers', 'non_buyers')} == dict(
            profiles=3, actions=5, orders=3, events=6, profile_events=1, identified=8,
            without_profile=5, profile_only=2, buyers=2, non_buyers=6)
        assert {k: scope[k] for k in ('views', 'favorites', 'digital', 'view_favorite', 'itemless_digital')} == dict(
            views=3, favorites=2, digital=4, view_favorite=1, itemless_digital=1)
        assert scope['buyer_rate'] == 25. and scope['non_buyer_rate'] == 75.
        assert dict(result.diagnostics)['actions_without_customer_id'] == 2
        assert dict(result.diagnostics)['orders_duplicate_conflicting'] == 1
        assert result.customers == (4 if not selected.active else 0)  # Legacy profile-row/cohort policy.
        if selected.product_restricted:
            assert result.purchase_users == 0  # Both buyers purchased outside this product selection.
        else:
            assert result.purchase_users == 2
        if expected_scope is not None:
            assert scope == expected_scope
        expected_scope = scope
        with monkeypatch.context() as patch:
            patch.setattr(statistics, 'iter_export', lambda *a, **k: pytest.fail('raw read'))
            patch.setattr(statistics, '_customer_snapshot', lambda *a, **k: pytest.fail('raw profile read'))
            actual = query(source, selected, calculated_at=AT)
        assert asdict(actual) == asdict(result)
        cache.validate_result(asdict(result))
        payload = json.dumps(asdict(result))
        assert all('"' + value + '"' not in payload for value in ('old', 'new', 'both-buyer', 'event-only',
                                                                  'phantom-duplicate', 'future-only'))


@pytest.mark.parametrize('missing', ['customers', 'actions', 'orders'])
def test_unavailable_sources_are_not_observed_zeroes(source, missing):
    if missing == 'customers':
        (source[0] / 'canonical/customers.sqlite').unlink()
    else:
        source[2][missing] = {}
        save(source[0], source[2])
    result = calculate(source)
    scope = result.customer_scope
    assert scope['identified'] is scope['non_buyers'] is scope['buyer_rate'] is scope['non_buyer_rate'] is None
    assert scope['profile_only'] is None
    if missing == 'customers':
        assert scope['profiles'] is scope['without_profile'] is None
        assert scope['events'] == 6 and scope['buyers'] == 2
    elif missing == 'actions':
        assert scope['actions'] is scope['digital'] is None
        assert scope['orders'] == 3
    else:
        assert scope['orders'] is scope['buyers'] is None
        assert scope['actions'] == 5
    build(source)
    assert asdict(query(source, calculated_at=AT)) == asdict(result)
    cache.validate_result(asdict(result))
    assert 'Не все источники доступны' in customer_scope_context(asdict(result))


def test_present_empty_sources_and_profiles_are_measured_zero_and_scans_are_single_pass(source, monkeypatch):
    write_actions(source, [])
    write_orders(source, [])
    with closing(sqlite3.connect(source[0] / 'canonical/customers.sqlite')) as db, db:
        db.execute('DELETE FROM profiles')
    scans = []
    reader = statistics.iter_export

    def records(name, **kwargs):
        scans.append(name)
        return reader(name, **kwargs)

    monkeypatch.setattr(statistics, 'iter_export', records)
    result = calculate(source)
    assert scans == ['customer_merges', 'actions', 'orders']
    scope = result.customer_scope
    assert scope['identified'] == scope['profiles'] == scope['events'] == scope['buyers'] == scope['non_buyers'] == 0
    assert scope['buyer_rate'] is scope['non_buyer_rate'] is None
    validate_customer_scope(scope)


def test_orders_own_coverage_original_period_and_cache_keys_do_not_change_legacy_metrics(source):
    # Orders has one extra day beyond shared Actions/Orders coverage.
    first, later = order(['CP'], 'one'), order(['CP'], 'two')
    later['customer']['ids']['mindboxId'] = 'later'
    later['firstAction']['dateTimeUtc'] = '2026-01-02T12:00:00Z'
    write_orders(source, [first])
    source[2]['orders']['2026-01-02'] = put_entry(source[0], 'orders', [later], 29,
                                               '2026-01-02T00:00:00+00:00', '2026-01-03T00:00:00+00:00')
    save(source[0], source[2])
    selected = AnalysisFilter('2026-01-01', '2026-01-01')
    all_history, day = asdict(calculate(source)), asdict(calculate(source, selected))
    assert all_history['analysis_filter'] == day['analysis_filter']  # Existing full-period normalization.
    assert {k: v for k, v in all_history.items() if k not in ('customer_scope','buyer_statistics','rfm_statistics','cohort_statistics')} == {
        k: v for k, v in day.items() if k not in ('customer_scope','buyer_statistics','rfm_statistics','cohort_statistics')}
    assert all_history['cohort_statistics']['as_of_date'] == '2026-01-02'
    assert day['cohort_statistics']['as_of_date'] == '2026-01-01'
    assert all_history['cohort_statistics']['cohort_bounds'] == dict(start=None, end=None)
    assert day['cohort_statistics']['cohort_bounds'] == dict(start='2026-01', end='2026-01')
    assert all_history['cohort_statistics']['policies']['EXTENDED_COMPLETED']['base_buyers'] == 2
    assert day['cohort_statistics']['policies']['EXTENDED_COMPLETED']['base_buyers'] == 1
    assert all_history['buyer_statistics']['policies']['EXTENDED_COMPLETED']['orders'] == 2
    assert day['buyer_statistics']['policies']['EXTENDED_COMPLETED']['orders'] == 1
    assert all_history['customer_scope']['buyers'] == 2 and day['customer_scope']['buyers'] == 1
    build(source)
    assert asdict(query(source, calculated_at=AT)) == all_history
    assert asdict(query(source, selected, calculated_at=AT)) == day
    args = dict(raw_root=source[0], catalog_path=source[1], cache_dir=source[0] / 'slices')
    a = slices.calculate_cached_statistics(**args)
    b = slices.calculate_cached_statistics(**args, analysis_filter=selected)
    assert a['customer_scope']['buyers'] == 2 and b['customer_scope']['buyers'] == 1
    assert len(list((source[0] / 'slices').glob('*.json'))) == 2


def test_old_and_new_cache_roundtrip_without_fabricating_coverage(source, tmp_path):
    result = asdict(calculate(source))
    original = deepcopy(result)
    for legacy in (False, True):
        value = deepcopy(result)
        if legacy:
            value.pop('customer_scope')
        path = tmp_path / ('legacy.json' if legacy else 'new.json')
        cache.save_result(path, value)
        loaded = cache.load_result(path)
        assert loaded == json.loads(json.dumps(value))
        if legacy:
            assert all(v == 'Нет данных в сохранённом результате' for _, rows in customer_scope_sections(loaded) for _, v in rows)
            assert 'customer_scope' not in loaded
        assert json.loads(path.read_text(encoding='utf-8'))['schema_version'] == 10
    assert result == original


@pytest.mark.parametrize('field,value', [('buyers', 4), ('identified', 9), ('actions', -1), ('profiles', True),
    ('digital', 100), ('view_favorite', 3), ('itemless_digital', 5), ('without_profile', 6), ('non_buyers', 7),
    ('buyer_rate', 30.), ('non_buyer_rate', float('nan')), ('profiles_available', 1), ('customer_id', 'PRIVATE')])
def test_scope_validation_rejects_bad_unions_rates_types_and_identity_leaks(source, field, value):
    result = asdict(calculate(source))
    result['customer_scope'][field] = value
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


def test_compact_flags_do_not_materialize_events_or_export_identities():
    scope = CustomerScope(True, True, AnalysisFilter())
    scope.profiles_available = True
    for _ in range(100):
        scope.profile('synthetic')
        scope.order('synthetic', True)
    assert len(scope.flags) == 1
    assert scope.result()['identified'] == 1
    assert 'synthetic' not in json.dumps(scope.result())


def test_first_source_order_owner_is_not_replaced_by_later_snapshot_inside_legacy_coverage(source):
    source[2]['actions'] = {'2026-01-02': put_entry(source[0], 'actions', [], 31,
                              '2026-01-02T00:00:00+00:00', '2026-01-03T00:00:00+00:00')}
    first, second = order(['Return'], 'same'), order(['CP'], 'same')
    first['customer']['ids']['mindboxId'] = 'first-owner'
    second['customer']['ids']['mindboxId'] = 'second-owner'
    second['firstAction']['dateTimeUtc'] = '2026-01-02T12:00:00Z'
    write_orders(source, [first])
    source[2]['orders']['2026-01-02'] = put_entry(source[0], 'orders', [second], 32,
                              '2026-01-02T00:00:00+00:00', '2026-01-03T00:00:00+00:00')
    save(source[0], source[2])
    result = calculate(source)
    assert result.purchase_users == 1  # Existing shared-coverage first snapshot.
    assert result.customer_scope['orders'] == 1 and result.customer_scope['buyers'] == 0
    assert result.customer_scope['identified'] == 4  # Three profiles plus first owner only.
    build(source)
    assert asdict(query(source, calculated_at=AT)) == asdict(result)


def test_customer_period_is_attested_in_slices_not_only_the_normalized_legacy_filter(source):
    result = asdict(calculate(source))
    result.pop('buyer_statistics')  # Isolate the existing schema-10 customer-scope attestation.
    result.pop('rfm_statistics')
    result.pop('cohort_statistics')  # Isolate period attestation from optional cohort cross-validation.
    provenance, _ = slices.capture_sources(source[0], source[1], AnalysisFilter(), calculated_at=AT)
    folder = source[0] / 'slices'
    slices.save_slice(folder, provenance, result)
    path = next(folder.glob('*.json'))
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['result']['customer_scope']['period'] = dict(start_date='2026-01-01', end_date='2026-01-01')
    payload['result_sha256'] = slices._digest(payload['result'])
    path.write_text(json.dumps(payload), encoding='utf-8')
    assert slices.load_slice(folder, provenance) is None
    with pytest.raises(ValueError, match='period'):
        slices.save_slice(folder, provenance, payload['result'])


def test_cancellation_is_checked_while_folding_unique_customer_flags():
    scope = CustomerScope(True, True, AnalysisFilter())
    scope.profiles_available = True
    scope.profile('synthetic')

    def cancel():
        raise InterruptedError('cancelled')

    with pytest.raises(InterruptedError):
        scope.result(cancel)


def test_cohort_snapshot_rejects_customer_period_mismatch_before_slice_attestation(source):
    result = asdict(calculate(source))
    result.pop('buyer_statistics')
    result.pop('rfm_statistics')
    cache.validate_result(result)
    result['customer_scope']['period'] = dict(start_date='2026-01-01', end_date='2026-01-01')
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)
