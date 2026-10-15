import json

import pytest

from Application.evaluation.experiments.gui_history import ExperimentHistory, HistoryError, clean_record
from Application.evaluation.experiments.gui_benchmark import gui_benchmark_metadata, LEGACY_BENCHMARK


def record(id='a', status='completed', date='2026-10-05T10:00:00+00:00'):
    return {'run_id': id * 32, 'status': status, 'started_at': date,
            'hyperparameters': {'w_view_item': .5, 'epochs': 50, 'customer_id': 'SECRET'},
            'metrics': {'overall': {'10': {'ndcg': .01234567, 'recall': .02, 'customer_id': 'SECRET'}},
                        'test': {'10': {'ndcg': 1}}}, 'email': 'SECRET', 'raw_payload': {'phone': 'SECRET'},
            'error_summary': 'SECRET arbitrary exception', 'provenance': {'git_head': 'c' * 40, 'working_tree_dirty': True}}


def test_persistent_newest_deduplicated_aggregate_only(tmp_path):
    store = ExperimentHistory(tmp_path)
    store.upsert(record('a'))
    store.upsert(record('b', 'failed', '2026-10-05T12:00:00+00:00'))
    store.upsert(record('a', 'cancelled'))
    records = ExperimentHistory(tmp_path).read()
    assert [r['run_id'] for r in records] == ['b' * 32, 'a' * 32]
    assert [r['status'] for r in records] == ['failed', 'cancelled']
    text = store.path.read_text()
    assert not any(s in text for s in ['SECRET', 'test', 'customer_id', 'email', 'phone', 'raw_payload'])
    assert records[0]['metrics']['overall']['10']['ndcg'] == .01234567
    assert records[0]['provenance']['working_tree_dirty']


@pytest.mark.parametrize('content', ['{broken', '{"schema_version":999,"runs":[]}', '{"schema_version":1,"runs":[{}]}'])
def test_corrupt_history_never_overwritten(tmp_path, content):
    store = ExperimentHistory(tmp_path)
    store.path.write_text(content)
    with pytest.raises(HistoryError):
        store.read()
    with pytest.raises(HistoryError):
        store.upsert(record())
    assert store.path.read_text() == content


def test_atomic_replace_failure_retains_old_history(tmp_path, monkeypatch):
    from Application.mindbox import canonical_storage as storage
    store = ExperimentHistory(tmp_path)
    store.upsert(record())
    previous = store.path.read_bytes()
    monkeypatch.setattr(storage.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('synthetic failure')))
    with pytest.raises(OSError):
        store.upsert(record('b'))
    assert store.path.read_bytes() == previous
    assert not list(tmp_path.glob('.metadata-*'))


def test_recovery_dead_live_and_finished_artifact(tmp_path):
    from Application.mindbox.canonical_storage import atomic_json
    store = ExperimentHistory(tmp_path)
    store.upsert({**record('a', 'running'), 'pid': 10})
    store.upsert({**record('b', 'running'), 'pid': 20})
    store.upsert({**record('c', 'running'), 'pid': 30})
    completed = clean_record(record('c'))
    atomic_json(tmp_path / completed['artifact'], completed)
    records = {r['run_id']: r for r in store.recover(lambda pid: pid == 20)}
    assert records['a' * 32]['status'] == 'interrupted'
    assert records['b' * 32]['status'] == 'running'
    assert records['c' * 32]['status'] == 'completed'
    assert len(store.recover(lambda pid: pid == 20)) == 3


def test_no_nonfinite_json(tmp_path):
    store = ExperimentHistory(tmp_path)
    store.upsert({**record(), 'total_seconds': float('inf'), 'metrics': {'overall': {'10': {'ndcg': float('nan')}}}})
    parsed = json.loads(store.path.read_text())
    assert 'total_seconds' not in parsed['runs'][0]
    assert not parsed['runs'][0]['metrics']


def test_legacy_and_new_benchmarks_keep_their_own_metadata_and_artifacts(tmp_path):
    store = ExperimentHistory(tmp_path)
    legacy = clean_record(record('a'))
    explicit = {**dict(LEGACY_BENCHMARK), 'test_end': '2025-12-31T00:00:00Z'}
    legacy['benchmark'] = explicit
    original = json.dumps({'schema_version': 1, 'runs': [legacy]})
    store.path.write_text(original, encoding='utf-8')
    assert store.read()[0]['benchmark'] == explicit
    assert store.path.read_text(encoding='utf-8') == original
    artifact = tmp_path / legacy['artifact']
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(legacy), encoding='utf-8')
    before = artifact.read_bytes()
    new = {**record('b'), 'benchmark': gui_benchmark_metadata(),
           'data_coverage': {'declared_start': '2022-01-01T00:00:00+00:00', 'declared_end': '2026-09-01T00:00:00+00:00', 'email': 'SECRET'}}
    store.upsert(new)
    records = {r['run_id']: r for r in store.read()}
    assert records['a' * 32]['benchmark'] == explicit
    assert records['b' * 32]['benchmark'] == gui_benchmark_metadata()
    assert records['b' * 32]['data_coverage'] == {k: v for k, v in new['data_coverage'].items() if k != 'email'}
    assert artifact.read_bytes() == before


def test_missing_benchmark_means_legacy_and_invalid_metadata_is_not_relabelled():
    assert clean_record(record())['benchmark'] == LEGACY_BENCHMARK
    for benchmark in ({'id': 'unknown'}, {**gui_benchmark_metadata(), 'history_end': '2025-11-01T00:00:00Z'},
                      {**gui_benchmark_metadata(), 'min_history_events': True}):
        with pytest.raises(HistoryError):
            clean_record({**record(), 'benchmark': benchmark})
