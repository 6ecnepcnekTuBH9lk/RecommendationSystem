"""GUI dates, selected-source readiness and temporal boundaries; no training."""

from dataclasses import asdict, replace
from datetime import datetime, timedelta
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from Application.evaluation.experiments.gui_benchmark import (
    GUI_BENCHMARK, gui_benchmark_config, gui_benchmark_metadata, require_gui_coverage,
)
from Application.evaluation.experiments.bpr_weights import benchmark_config
from Application.evaluation.temporal import build_temporal_protocol
from Application.tabs.training_workflow import metadata_readiness


def at(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def test_gui_dates_are_separate_and_old_cli_benchmark_is_unchanged():
    gui, legacy = gui_benchmark_config(), benchmark_config()
    assert (gui.validation_start, gui.test_start, gui.test_end) == (
        at('2026-07-01T00:00:00Z'), at('2026-08-01T00:00:00Z'), at('2026-09-01T00:00:00Z'))
    assert (legacy.validation_start, legacy.test_start, legacy.test_end) == (
        at('2025-11-01T00:00:00Z'), at('2025-12-01T00:00:00Z'), at('2025-12-31T00:00:00Z'))
    assert gui.min_history_events == legacy.min_history_events == 10
    metadata = gui_benchmark_metadata()
    assert metadata['id'] == 'mindbox-validation-2026-07'
    assert at(metadata['history_end']) == gui.validation_start
    assert at(metadata['validation_end']) == gui.test_start
    assert at(metadata['test_end']) == gui.test_end


def test_utc_boundaries_whole_history_and_august_cannot_affect_validation(event):
    def make(user, item, stamp):
        original = event(user, item, 0)
        return replace(original, interaction=replace(original.interaction, event_datetime_utc=at(stamp)))
    events = [make('u', 'A', '2022-01-01T00:00:00Z') for _ in range(9)]
    events += [make('u', 'B', '2026-06-30T23:59:59.999999Z'), make('w', 'C', '2022-01-02T00:00:00Z'),
               make('w', 'D', '2026-06-30T00:00:00Z'), make('w', 'E', '2026-06-30T00:00:00Z'),
               make('u', 'C', '2026-07-01T00:00:00Z'),
               make('u', 'D', '2026-07-31T23:59:59.999999Z'), make('u', 'E', '2026-08-01T00:00:00Z'),
               make('u', 'ONLY_AUGUST', '2026-08-31T23:59:59.999999Z'), make('u', 'AFTER_END', '2026-09-01T00:00:00Z')]
    protocol = build_temporal_protocol(events, gui_benchmark_config())
    validation = protocol.validation
    assert len(validation.history) == 13
    assert min(e.interaction.event_datetime_utc for e in validation.history) == GUI_BENCHMARK.history_start
    assert all(e.interaction.event_datetime_utc < GUI_BENCHMARK.validation_start for e in validation.history)
    assert [c.target_item for c in validation.cases] == ['C']  # Existing first-novel-warm semantics.
    assert validation.cases[0].target_timestamp == GUI_BENCHMARK.validation_start
    assert validation.diagnostics.future_events == 2
    assert 'ONLY_AUGUST' not in validation.item_universe
    without_august = [e for e in events if e.interaction.event_datetime_utc < GUI_BENCHMARK.test_start]
    assert validation == build_temporal_protocol(without_august, gui_benchmark_config()).validation
    assert protocol.test.diagnostics.future_events == 2  # Sep 1 excluded from independent test.
    assert protocol.test.cases[0].target_timestamp == GUI_BENCHMARK.test_start


@pytest.mark.parametrize('since,until,ready', [
    ('2022-01-01T00:00:00Z', '2026-09-01T00:00:00Z', True),
    ('2021-01-01T00:00:00Z', '2026-10-01T00:00:00Z', True),
    ('2025-10-01T00:00:00Z', '2026-09-01T00:00:00Z', False),
    ('2022-01-01T00:00:00Z', '2026-08-01T00:00:00Z', False),
    ('2022-01-01T00:00:00.000001Z', '2026-09-01T00:00:00Z', False),
    ('2022-01-01T00:00:00Z', '2026-08-31T23:59:59.999999Z', False),
])
def test_readiness_uses_actual_selected_batch_and_does_not_change_production(tmp_path, monkeypatch, since, until, ready):
    from Application.tabs import training_workflow as workflow
    (tmp_path / 'nomenclature.csv').write_text('synthetic')
    manifest = tmp_path / 'MindboxRaw/canonical/training.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"schema_version":5,"storage":"canonical"}')
    batch = SimpleNamespace(batch_id='synthetic', window=SimpleNamespace(interaction_since=at(since), interaction_until=at(until)))
    monkeypatch.setattr(workflow, 'current_batch', lambda *a: batch)
    assert metadata_readiness(tmp_path, True)[0] is ready
    assert metadata_readiness(tmp_path, False)[0] is True


def test_no_heavy_ml_imports_in_gui_metadata_path():
    code = 'import sys; from Application.tabs import training_workflow; assert "torch" not in sys.modules; assert "numpy" not in sys.modules'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def canonical(tmp_path):
    from Application.mindbox.raw_reader import EXPORT_ROOTS
    from Application.mindbox.selection import DEFAULT_SELECTION
    root = tmp_path / 'MindboxRaw'
    start, end = GUI_BENCHMARK.history_start.isoformat(), GUI_BENCHMARK.test_end.isoformat()
    def raw_action(user, item, stamp, key):
        return {'ids': {'mindboxId': key}, 'customer': {'ids': {'mindboxId': user}},
                'actionTemplate': {'ids': {'systemName': 'ProsmotrProdukta'}}, 'dateTimeUtc': stamp,
                'creationDateTimeUtc': stamp, 'products': [{'ids': {'offline1C': item + '-size'}}]}
    actions = [raw_action('u', '000001', start, str(i)) for i in range(9)]
    actions += [raw_action('u', '000001', '2026-06-30T23:59:59.999999Z', 'june'),
                raw_action('w', '000002', '2022-01-02T00:00:00Z', 'warm'),
                raw_action('u', '000002', '2026-07-01T00:00:00Z', 'july'),
                raw_action('u', '000003', '2026-08-01T00:00:00Z', 'august')]
    entries, directories = {}, {}
    for name, values in (('customer_merges', []), ('actions', actions), ('orders', [])):
        directory = root / 'canonical/objects' / (('1' if name == 'customer_merges' else '2') * 32) / name
        directory.mkdir(parents=True)
        (directory / f'{name}_part_001.json').write_text(json.dumps({EXPORT_ROOTS[name]: values}), encoding='utf-8')
        directories[name] = directory
        entries[name] = {'since': start, 'until': end, 'updated': end, 'directory': directory.relative_to(root).as_posix(),
                         'parts': 1, 'source_kind': 'API' if name == 'customer_merges' else 'MANUAL',
                         'export_id': 'synthetic' if name == 'customer_merges' else None,
                         'operation': 'synthetic' if name == 'customer_merges' else 'MANUAL'}
    metadata = {'schema_version': 2, 'revision': 'synthetic', 'actions': {}, 'orders': {},
                'customer_merges': entries['customer_merges'], 'selection': asdict(DEFAULT_SELECTION),
                'manual_interactions': {'since': start, 'until': end, 'updated': end,
                                        'actions': entries['actions'], 'orders': entries['orders']}}
    (root / 'canonical/catalog.json').write_text(json.dumps(metadata), encoding='utf-8')
    (root / 'canonical/training.json').write_text('{"schema_version":5,"storage":"canonical"}')
    product_catalog = tmp_path / 'nomenclature.csv'
    product_catalog.write_text('КодНоменклатуры\n000001\n000002\n000003\n', encoding='utf-8')
    return root, product_catalog, metadata


def test_existing_loader_supplies_january_2022_history_without_old_batch_limit(canonical):
    from Application.evaluation.audit import load_canonical_events
    root, catalog_path, _ = canonical
    assert metadata_readiness(root.parent, True)[0]
    events, source = load_canonical_events(root, catalog_path)
    require_gui_coverage(source['declared_start'], source['declared_end'])
    snapshot = build_temporal_protocol(events, gui_benchmark_config()).validation
    assert len(snapshot.history) == 11
    assert snapshot.history[0].interaction.event_datetime_utc == GUI_BENCHMARK.history_start
    assert snapshot.cases[0].target_timestamp == GUI_BENCHMARK.validation_start
    assert all(e.item_id != '000003' for e in snapshot.history)


def test_gap_cannot_be_hidden_by_available_early_history_and_later_window(canonical):
    root, _, metadata = canonical
    # Remove the long manual coverage and leave two separated API days. The
    # loader still selects its longest common range; GUI must reject that range.
    metadata['manual_interactions'] = None
    for name in ('actions', 'orders'):
        directory = root / 'canonical/objects' / ('2' * 32) / name
        metadata[name] = {}
        for day in ('2022-01-01', '2026-08-31'):
            since = at(day + 'T00:00:00Z')
            metadata[name][day] = {'since': since.isoformat(), 'until': (since + timedelta(days=1)).isoformat(),
                                  'updated': GUI_BENCHMARK.test_end.isoformat(), 'directory': directory.relative_to(root).as_posix(),
                                  'parts': 1, 'source_kind': 'API', 'export_id': 'synthetic', 'operation': 'synthetic'}
    (root / 'canonical/catalog.json').write_text(json.dumps(metadata), encoding='utf-8')
    ready, reason, _ = metadata_readiness(root.parent, True)
    assert not ready and 'непрерывного покрытия' in reason
    assert metadata_readiness(root.parent, False)[0]
