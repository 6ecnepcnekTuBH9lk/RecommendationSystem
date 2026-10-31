"""Detail GUI/process tests with synthetic children; never real Mindbox history."""
from datetime import date
import json
import sys

import pytest
from PyQt6.QtCore import QDate, QProcess, QTimer
from PyQt6.QtWidgets import QDialog, QMessageBox

from Application.analysis_filter import AnalysisFilter
from Application.statistics_detail_job import SourceInfo, MAX_MESSAGE
from Application.tabs.statistics_detail_dialog import DetailExportDialog
from Application import statistics_detail_process as process_module
from Application.tabs import dataset_statistics_tab as ui
from tests.tabs.test_dataset_statistics_tab import app as app_fixture, window as window_fixture, wait_until

app = app_fixture
window = window_fixture

pytestmark = pytest.mark.usefixtures('window_settings')


@pytest.fixture
def parameters(tmp_path):
    private = tmp_path/'private'
    private.mkdir()
    return dict(raw_root=tmp_path/'source', destination=tmp_path/'detail.zip',temporary_root=private,
                as_of=date(2026,8,31),start=date(2026,1,1),end=date(2026,8,31),order_mode='ALL_SELECTED')


def synthetic(monkeypatch, script):
    calls=[]
    class Child(QProcess):
        def start(self, program, arguments):
            calls.append((program,arguments,self.workingDirectory()))
            super().start(sys.executable,['-B','-u','-c',script])
    monkeypatch.setattr(process_module,'QProcess',Child)
    return calls


def wire_dialog(monkeypatch, parameters, *, accepted=True):
    monkeypatch.setattr(ui,'inspect_sources',lambda *a:SourceInfo(date(2020,1,1),date(2026,8,31)))
    class Dialog:
        def __init__(self,*args):
            self.options=parameters
        def exec(self):
            return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
    monkeypatch.setattr(ui,'DetailExportDialog',Dialog)


def progress_script(terminal):
    return ("import json,time; print(json.dumps(dict(event='progress',stage='orders',processed=12)),flush=True); "
            "time.sleep(.1); print("+repr(json.dumps(terminal))+",flush=True)")


def test_button_exists_without_displayed_result_and_old_export_stays(window):
    _,tab=window
    assert tab.displayed_result is None and tab.detail_export_button.isEnabled()
    assert tab.detail_export_button.text().strip()=='Экспорт данных покупателей'
    assert tab.export_button.text().strip()=='Экспорт статистики' and not tab.export_button.isEnabled()


def test_dialog_cancel_and_missing_sources_never_launch_worker(window,monkeypatch,parameters):
    _,tab=window
    wire_dialog(monkeypatch,parameters,accepted=False)
    monkeypatch.setattr(ui,'DetailExportProcess',lambda *a:pytest.fail('Unexpected worker'))
    tab.export_details()
    assert tab.detail_process is None and tab.refresh.isEnabled() and tab.detail_export_button.isEnabled()
    def unavailable(*a):raise ValueError('SYNTH_PRIVATE@example.invalid')
    monkeypatch.setattr(ui,'inspect_sources',unavailable)
    tab.export_details()
    assert 'canonical' in tab.detail_status.text() and 'SYNTH_PRIVATE' not in tab.detail_status.text()


def test_dialog_defaults_from_coverage_and_filter_are_independent(window,parameters):
    _,tab=window
    source=SourceInfo(date(2020,1,1),date(2026,8,31))
    selection=AnalysisFilter('2026-02-01','2026-04-30',('A',))
    dialog=DetailExportDialog(tab,source,selection,parameters['raw_root'])
    assert dialog.as_of.date().toPyDate()==date(2026,4,30) and not dialog.all_history.isChecked()
    assert dialog.mode.currentData().value=='ALL_SELECTED'
    assert selection==AnalysisFilter('2026-02-01','2026-04-30',('A',))
    dialog.reject()
    other=DetailExportDialog(tab,source,AnalysisFilter(),parameters['raw_root'])
    assert other.all_history.isChecked() and other.as_of.date().toPyDate()==source.last_date
    assert not other.start_date.isEnabled()
    other.reject()


@pytest.mark.parametrize('kind',['reversed','future','repository'])
def test_invalid_settings_are_rejected(window,parameters,kind):
    _,tab=window
    dialog=DetailExportDialog(tab,SourceInfo(date(2020,1,1),date(2026,8,31)),
                              AnalysisFilter('2026-01-01','2026-08-31'),parameters['raw_root'])
    dialog.destination.setText(str(parameters['destination']))
    dialog.temporary_root.setText(str(parameters['temporary_root']))
    if kind=='reversed':dialog.start_date.setDate(QDate(2026,9,1))
    if kind=='future':dialog.as_of.setDate(QDate(2026,7,1))
    if kind=='repository':dialog.destination.setText(str(ui.PROJECT_ROOT/'forbidden.zip'))
    dialog.accept()
    assert dialog.result()!=QDialog.DialogCode.Accepted and dialog.error.text()
    assert not parameters['destination'].exists()


def test_file_chooser_and_manual_overwrite_do_not_create_zip(window,parameters,monkeypatch):
    _,tab=window
    target=parameters['destination']
    target.write_bytes(b'previous')
    dialog=DetailExportDialog(tab,SourceInfo(date(2020,1,1),date(2026,8,31)),AnalysisFilter(),parameters['raw_root'])
    dialog.temporary_root.setText(str(parameters['temporary_root']))
    captured=[]
    def choose(*args):
        captured.append(args)
        return str(target),'ZIP (*.zip)'
    monkeypatch.setattr(ui.QFileDialog,'getSaveFileName',choose)
    dialog.choose_destination()
    assert 'Данные_покупателей_' in captured[0][2] and captured[0][3]=='ZIP (*.zip)'
    assert target.read_bytes()==b'previous' and dialog.options is None
    dialog.accept()
    assert dialog.result()==QDialog.DialogCode.Accepted and dialog.options['start'] is None
    assert target.read_bytes()==b'previous'


def test_success_real_qprocess_current_python_workdir_and_isolated_state(window,parameters,monkeypatch):
    _,tab=window
    before={'sentinel':'aggregate'}
    tab.displayed_result=before
    tab.cache_path.write_bytes(b'unchanged aggregate cache')
    wire_dialog(monkeypatch,parameters)
    calls=synthetic(monkeypatch,progress_script(dict(event='finished',rows=dict(clients=1,orders=2,order_lines=3))))
    tab.export_details()
    worker=tab.detail_process
    assert tab.process is None and worker is not None and not tab.refresh.isEnabled()
    tab.start()
    tab.export()
    tab.export_details()
    assert tab.detail_process is worker and len(calls)==1 and tab.sections.isEnabled()
    wait_until(lambda:tab.detail_process is None)
    assert calls[0][0]==sys.executable and calls[0][1][:2]==['-u','-B']
    assert calls[0][2]==str(ui.PROJECT_ROOT)
    assert 'scripts' in calls[0][1][2] and 'statistics_detail_export.py' in calls[0][1][2]
    assert 'сохранена' in tab.detail_status.text() and 'Клиентов: 1' in tab.detail_status.text()
    assert tab.displayed_result is before and tab.cache_path.read_bytes()==b'unchanged aggregate cache'
    assert tab.refresh.isEnabled() and tab.export_button.isEnabled() and tab.detail_export_button.isEnabled()


@pytest.mark.parametrize('close',[False,True])
def test_cooperative_cancel_and_close_wait_for_worker_cleanup(window,parameters,monkeypatch,close):
    root,tab=window
    parameters['destination'].write_bytes(b'previous')
    wire_dialog(monkeypatch,parameters)
    synthetic(monkeypatch,"import sys,time; sys.stdin.readline(); time.sleep(.15); "
              "print('{\"event\":\"cancelled\"}',flush=True); raise SystemExit(2)")
    tab.export_details()
    process=tab.detail_process.process
    wait_until(lambda:process.state()==QProcess.ProcessState.Running)
    ticks=[]
    timer=QTimer(tab)
    timer.timeout.connect(lambda:ticks.append(1))
    timer.start(10)
    if close:
        root.close()
        assert root.isVisible()
    else:
        tab.cancel_details()
        tab.cancel_details()
    assert tab.detail_process is not None and not tab.detail_cancel.isEnabled()
    wait_until(lambda:tab.detail_process is None)
    timer.stop()
    assert ticks and 'отменён' in tab.detail_status.text()
    assert parameters['destination'].read_bytes()==b'previous'
    assert tab.detail_export_button.isEnabled() and tab.refresh.isEnabled()
    if close:wait_until(lambda:not root.isVisible())


@pytest.mark.parametrize('kind',['malformed','oversized','extra_field','partial','crash','false_success'])
def test_invalid_or_crashed_worker_is_never_success_and_never_echoes_pii(window,parameters,monkeypatch,kind):
    _,tab=window
    bad={
        'malformed':'SYNTH_PRIVATE@example.invalid',
        'oversized':'x'*(MAX_MESSAGE+1),
        'extra_field':json.dumps(dict(event='progress',stage='orders',processed=1,email='SYNTH_PRIVATE')),
    }
    if kind in bad:
        script="import sys; print("+repr(bad[kind])+",flush=True); sys.stdin.readline(); raise SystemExit(2)"
    elif kind=='partial':
        script="import sys; sys.stdout.write('SYNTH_PRIVATE'); sys.stdout.flush()"
    elif kind=='false_success':
        script="print('{\"event\":\"finished\",\"rows\":{\"clients\":1,\"orders\":1,\"order_lines\":1}}',flush=True); raise SystemExit(1)"
    else:
        script="raise SystemExit(7)"
    synthetic(monkeypatch,script)
    tab._start_details(parameters)
    wait_until(lambda:tab.detail_process is None)
    assert 'сохранена' not in tab.detail_status.text() and 'SYNTH_PRIVATE' not in tab.detail_status.text()
    assert tab.refresh.isEnabled() and tab.detail_export_button.isEnabled()


def test_failed_start_and_backend_error_restore_controls(window,parameters,monkeypatch):
    _,tab=window
    monkeypatch.setattr(process_module.sys,'executable','missing-detail-python-executable')
    tab._start_details(parameters)
    wait_until(lambda:tab.detail_process is None)
    assert 'FAILED_TO_START' in tab.detail_status.text() and tab.refresh.isEnabled()


def test_known_backend_error_is_explained(window,parameters,monkeypatch):
    _,tab=window
    synthetic(monkeypatch,"print('{\"event\":\"error\",\"code\":\"INSUFFICIENT_SPACE\"}',flush=True); raise SystemExit(1)")
    tab._start_details(parameters)
    wait_until(lambda:tab.detail_process is None)
    assert 'Недостаточно' in tab.detail_status.text() and 'INSUFFICIENT_SPACE' in tab.detail_status.text()


def test_force_is_explicit_and_warns_about_leftovers(window,parameters,monkeypatch):
    _,tab=window
    synthetic(monkeypatch,"import time; time.sleep(15)")
    tab._start_details(parameters)
    wait_until(lambda:tab.detail_process.process.state()==QProcess.ProcessState.Running)
    tab.cancel_details()
    tab._detail_stalled()
    assert tab.detail_force.isVisible()
    monkeypatch.setattr(ui.QMessageBox,'warning',lambda *a:QMessageBox.StandardButton.Yes)
    tab.force_details()
    wait_until(lambda:tab.detail_process is None)
    assert 'FORCED_STOP' in tab.detail_status.text() and 'Очистка не гарантируется' in tab.detail_status.text()


def test_aggregate_calculation_blocks_detail_without_changing_legacy_process(window,parameters,monkeypatch):
    _,tab=window
    sentinel=object()
    tab.process=sentinel
    tab._update_export_button()
    tab.export_details()
    assert not tab.detail_export_button.isEnabled() and tab.detail_process is None and tab.process is sentinel
    tab.process=None
    tab._update_export_button()


def test_quit_event_is_deferred_until_cooperative_worker_exit(window,parameters,monkeypatch):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QApplication
    _,tab=window
    synthetic(monkeypatch,"import sys; sys.stdin.readline(); print('{\"event\":\"cancelled\"}',flush=True); raise SystemExit(2)")
    tab._start_details(parameters)
    wait_until(lambda:tab.detail_process.process.state()==QProcess.ProcessState.Running)
    quits=[]
    monkeypatch.setattr(QApplication,'quit',lambda *args:quits.append(True))
    QApplication.sendEvent(QApplication.instance(),QEvent(QEvent.Type.Quit))
    assert tab.detail_process is not None and tab._detail_quitting and not quits
    wait_until(lambda:tab.detail_process is None)
    wait_until(lambda:bool(quits))


def test_progress_is_visible_and_ui_keeps_processing_events(window,parameters,monkeypatch):
    _,tab=window
    synthetic(monkeypatch,"import sys; print('{\"event\":\"progress\",\"stage\":\"rfm\",\"processed\":27}',flush=True); "
              "sys.stdin.readline(); print('{\"event\":\"cancelled\"}',flush=True); raise SystemExit(2)")
    tab._start_details(parameters)
    wait_until(lambda:'RFM' in tab.detail_status.text())
    assert '27' in tab.detail_status.text() and tab.detail_progress.maximum()==0
    tab.cancel_details()
    wait_until(lambda:tab.detail_process is None)


@pytest.mark.parametrize('channel',['stdout','stderr'])
def test_flooding_child_is_stopped_as_emergency_without_freezing_gui(window,parameters,monkeypatch,channel):
    _,tab=window
    code=("import sys,time; s=sys."+channel+"; s.write('SYNTH_PRIVATE'*150000); s.flush(); time.sleep(10)")
    synthetic(monkeypatch,code)
    tab._start_details(parameters)
    wait_until(lambda:tab.detail_process is None)
    assert 'WORKER_PROTOCOL_ERROR' in tab.detail_status.text() and 'SYNTH_PRIVATE' not in tab.detail_status.text()


def test_manual_existing_target_requires_confirmation_without_modification(window,parameters,monkeypatch):
    _,tab=window
    target=parameters['destination']
    target.write_bytes(b'previous')
    dialog=DetailExportDialog(tab,SourceInfo(date(2020,1,1),date(2026,8,31)),AnalysisFilter(),parameters['raw_root'])
    dialog.destination.setText(str(target))
    dialog.temporary_root.setText(str(parameters['temporary_root']))
    monkeypatch.setattr(ui.QMessageBox,'question',lambda *a:QMessageBox.StandardButton.No)
    dialog.accept()
    assert dialog.result()!=QDialog.DialogCode.Accepted and target.read_bytes()==b'previous'


def test_late_cancel_does_not_claim_an_already_published_zip_was_removed(window,parameters,monkeypatch):
    _,tab=window
    synthetic(monkeypatch,progress_script(dict(event='finished',rows=dict(clients=1,orders=1,order_lines=1))))
    tab._start_details(parameters)
    tab.cancel_details()
    wait_until(lambda:tab.detail_process is None)
    assert 'после завершения публикации' in tab.detail_status.text()
