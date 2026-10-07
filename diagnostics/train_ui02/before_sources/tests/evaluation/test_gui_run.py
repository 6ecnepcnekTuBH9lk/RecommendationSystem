from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Application.evaluation.experiments import gui_run as runner
from Application.evaluation.experiments.bpr_weights import benchmark_config
from Application.evaluation.temporal import TemporalProtocolError, build_temporal_protocol
from Application.interactions import InteractionType
from scripts import run_gui_research as cli


@pytest.fixture
def fixed_events(event):
    def make(user, item, month, kind=InteractionType.VIEW):
        original = event(user, item, 1, kind)
        return replace(original, interaction=replace(original.interaction,
                       event_datetime_utc=datetime(2025, month, 5, tzinfo=timezone.utc)))
    return ([make('u', 'A' if i % 2 else 'B', 10) for i in range(10)]
            + [make('other', ['C', 'D', 'E'][i % 3], 10) for i in range(12)]
            + [make('u', 'C', 11, InteractionType.PURCHASE), make('u', 'D', 12)])


def values():
    return {'w_purchase': 10., 'w_favorite': 2., 'w_view_item': .5, 'epochs': 1}


def test_gui_fixed_config_and_no_caller_hidden_overrides():
    cfg = runner.gui_config({**values(), 'seed': 43, 'early_stop': True, 'use_item_features': True, 'lr': .1})
    assert (cfg.seed, cfg.embedding_dim, cfg.batch_size, cfg.n_neg, cfg.lr, cfg.bpr_reg, cfg.weight_decay) == (42, 128, 256, 10, .0003, .0005, 0)
    assert not cfg.early_stop and not cfg.use_item_features
    assert (cfg.w_purchase, cfg.w_favorite, cfg.w_view_item, cfg.epochs) == (10., 2., .5, 1)


@pytest.mark.parametrize('key,value', [('w_purchase', 0), ('w_view_item', -1), ('w_favorite', float('nan')),
                                       ('w_purchase', float('inf')), ('epochs', 0), ('epochs', 1.5)])
def test_invalid_config_rejected(key, value):
    with pytest.raises(ValueError):
        runner.gui_config({**values(), key: value})


def test_one_training_final_validation_no_publication_or_test(fixed_events, tmp_path, monkeypatch):
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda n: None)
    monkeypatch.setattr(runner.core, '_save_artifacts', lambda *args, **kwargs: pytest.fail('No publication'))
    current = tmp_path / 'current.json'
    current.write_bytes(b'production sentinel')
    temporal = benchmark_config()
    protocol = build_temporal_protocol(fixed_events, temporal)
    cfg = runner.gui_config(values())
    original = runner.core.train_prepared_data_with_metrics
    calls, scored, progress = [], [], []
    def train(cfg, data, device, **kwargs):
        calls.append(cfg)
        assert not len(data.splits.eval_users) and not cfg.early_stop and not cfg.use_item_features
        return original(cfg, data, device, **kwargs)
    monkeypatch.setattr(runner.core, 'train_prepared_data_with_metrics', train)
    evaluate = runner.evaluate_bpr_snapshot
    def score(model, data, k, device):
        assert data.snapshot.cutoff == temporal.validation_start
        assert data.snapshot.future_end == temporal.test_start
        scored.append(k)
        return evaluate(model, data, k, device)
    monkeypatch.setattr(runner, 'evaluate_bpr_snapshot', score)
    result = runner.run_validation(protocol.validation, temporal, cfg, torch.device('cpu'), progress.append, lambda: None)
    assert len(calls) == 1 and result['epochs_completed'] == 1
    assert scored == [5, 10, 20] * 4
    assert 'ndcg' in result['metrics']['overall']['10'] and 'recall' in result['metrics']['PURCHASE']['20']
    assert current.read_bytes() == b'production sentinel'
    assert [p['stage'] for p in progress] == ['preparation', 'training', 'epoch', 'validation']


def test_test_snapshot_rejected_before_training_or_scoring(fixed_events, monkeypatch):
    temporal = benchmark_config()
    protocol = build_temporal_protocol(fixed_events, temporal)
    monkeypatch.setattr(runner.core, 'train_prepared_data_with_metrics', lambda *a, **kw: pytest.fail('Training on test'))
    monkeypatch.setattr(runner, 'evaluate_bpr_snapshot', lambda *a, **kw: pytest.fail('Scoring test'))
    with pytest.raises(TemporalProtocolError):
        runner.run_validation(protocol.test, temporal, runner.gui_config(values()), torch.device('cpu'), lambda _: None, lambda: None)


def test_empty_validation_reports_readiness_failure_before_training(fixed_events, monkeypatch):
    temporal = benchmark_config()
    snapshot = replace(build_temporal_protocol(fixed_events, temporal).validation, cases=())
    monkeypatch.setattr(runner.core, 'train_prepared_data_with_metrics', lambda *a, **kw: pytest.fail('Empty benchmark training'))
    with pytest.raises(runner.BenchmarkUnavailable):
        runner.run_validation(snapshot, temporal, runner.gui_config(values()), torch.device('cpu'), lambda _: None, lambda: None)


def test_cancel_epoch_stops_before_validation(fixed_events, monkeypatch):
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda n: None)
    monkeypatch.setattr(runner, 'evaluate_bpr_snapshot', lambda *a, **kw: pytest.fail('Validation after cancellation'))
    temporal = benchmark_config()
    snapshot = build_temporal_protocol(fixed_events, temporal).validation
    cancelled = []
    def emit(event):
        if event['stage'] == 'epoch':
            cancelled.append(True)
    def check():
        if cancelled:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        runner.run_validation(snapshot, temporal, runner.gui_config(values()), torch.device('cpu'), emit, check)


@pytest.mark.parametrize('outcome', ['completed', 'cancelled', 'failed'])
def test_cli_single_final_artifact_aggregate_only(tmp_path, monkeypatch, capsys, outcome):
    from Application import paths
    from Application.evaluation import audit, temporal as temporal_module
    from Application.evaluation.experiments import bpr_weights
    monkeypatch.setattr(paths, 'USER_SETTINGS_DIR', tmp_path / 'settings')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(bpr_weights, 'git_provenance', lambda root: {'git_head': 'c' * 40, 'working_tree_dirty': True})
    calls = []
    monkeypatch.setattr(audit, 'load_canonical_events', lambda *a, **kw: ([], {'SECRET': 'email'}))
    sentinel = object()
    def protocol(events, config):
        assert config == benchmark_config()
        return SimpleNamespace(validation=sentinel)  # No test attribute: GUI cannot access it.
    monkeypatch.setattr(temporal_module, 'build_temporal_protocol', protocol)
    def run(snapshot, temporal, cfg, device, emit, check):
        calls.append(cfg)
        assert snapshot is sentinel and cfg.seed == 42 and device.type == 'cpu'
        emit({'stage': 'epoch', 'epoch': 1, 'epochs': 1, 'loss': .1, 'device': 'cpu'})
        if outcome == 'cancelled':
            raise KeyboardInterrupt
        if outcome == 'failed':
            raise RuntimeError('SECRET phone/email/customer')
        return {'epochs_completed': 1, 'metrics': {'overall': {'10': {'ndcg': .01, 'recall': .02}}}}
    monkeypatch.setattr(runner, 'run_validation', run)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(values()))
    code = cli.main(['--config', str(config), '--run-id', 'a' * 32, '--cancel-file', str(tmp_path / 'cancel'),
                     '--raw-root', str(tmp_path / 'raw'), '--catalog', str(tmp_path / 'nomenclature.csv')])
    assert code == {'completed': 0, 'cancelled': 130, 'failed': 1}[outcome]
    assert len(calls) == 1
    artifact = tmp_path / 'settings/research_experiments/runs' / ('a' * 32) / 'result.json'
    result = json.loads(artifact.read_text())
    assert result['status'] == outcome and result['epochs_completed'] == 1
    assert result['torch_version'] == str(torch.__version__) and result['device'] == 'cpu'
    output = capsys.readouterr().out
    assert 'SECRET' not in output + artifact.read_text() and 'Traceback' not in output
    assert 'research_finished' in output


def test_real_qprocess_single_research_keeps_event_loop_responsive(tmp_path, monkeypatch, fixed_events):
    """One tiny synthetic epoch in a fresh CPU process; never opens user raw/model."""
    from PyQt6.QtCore import QEventLoop, QTimer, QProcess
    from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget
    from Application.tabs import train_model_tab as tab
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(tab, 'USER_SETTINGS_DIR', tmp_path / 'settings')
    monkeypatch.setattr(tab, 'INPUT_DATA_DIR', tmp_path / 'input_data')
    monkeypatch.setattr(tab, 'metadata_readiness', lambda *a: (True, 'Synthetic ready', 'a' * 32))
    for name in ('set_status_ok', 'set_status_error', 'set_status_processing', 'schedule_status_reset'):
        monkeypatch.setattr(tab, name, lambda *args: None)
    project = Path(tab.PROJECT_ROOT)
    script = tmp_path / 'scripts/run_gui_research.py'
    script.parent.mkdir()
    # Only synthetic domain records are serialized into this test-only wrapper.
    script.write_text(
        f'import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(project)!r})\n'
        'import torch\ntorch.cuda.is_available=lambda: False\n'
        'from datetime import datetime, timezone\n'
        'from Application.interactions import InteractionRecord, InteractionSource, InteractionType\n'
        'from Application.mindbox.records import ProductKey\n'
        'from Application.product_resolution import ResolvedInteraction\n'
        'from Application import paths\nfrom Application.evaluation import audit\n'
        f'paths.USER_SETTINGS_DIR=Path({str(tmp_path / "settings")!r})\n'
        'def event(u,i,m):\n'
        '    return ResolvedInteraction(InteractionRecord(source_customer_id=u,customer_id=u,product=ProductKey("offline1C",i),'
        'interaction_type=InteractionType.VIEW,event_datetime_utc=datetime(2025,m,5,tzinfo=timezone.utc),'
        'source=InteractionSource.ACTION,source_event_id="synthetic"),i)\n'
        'events=[event("u","A" if i%2 else "B",10) for i in range(10)]+'
        '[event("other",["C","D","E"][i%3],10) for i in range(12)]+[event("u","C",11)]\n'
        'audit.load_canonical_events=lambda *a,**kw:(events,{})\n'
        'from scripts.run_gui_research import main\nraise SystemExit(main())\n', encoding='utf-8')
    monkeypatch.setattr(tab, 'PROJECT_ROOT', tmp_path)
    current = tmp_path / 'model/current.json'
    current.parent.mkdir()
    current.write_bytes(b'production sentinel')
    window = QWidget()
    window.tabs = QTabWidget(window)
    tab.create_train_model_widgets_tab(window)
    window.epochs_input.setValue(1)
    loop = QEventLoop()
    timeout, pulse = QTimer(), QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    ticks = []
    pulse.timeout.connect(lambda: ticks.append(True))
    pulse.start(10)
    try:
        tab.start_training_process(window)
        window.train_proc.finished.connect(loop.quit)
        timeout.start(30000)
        loop.exec()
        assert ticks and not window._training_active, window.train_log.toPlainText()
        assert window._train_research_result['status'] == 'completed', window.train_log.toPlainText()
        assert window._train_research_result['epochs_completed'] == 1
        assert window._experiment_store.read()[0]['status'] == 'completed'
        assert current.read_bytes() == b'production sentinel'
        assert 'NDCG@10' in window.training_result.text()
    finally:
        pulse.stop()
        timeout.stop()
        if window.train_proc.state() != QProcess.ProcessState.NotRunning:
            window.train_proc.kill()
            window.train_proc.waitForFinished(5000)
        window._training_active = False
        window.close()
        window.deleteLater()
        app.processEvents()
