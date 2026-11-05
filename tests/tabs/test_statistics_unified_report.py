"""One report button, immutable context, background metadata and existing worker IPC."""
from copy import deepcopy
from datetime import date
from threading import get_ident

import pytest
from PyQt6.QtCore import QObject, QProcess, pyqtSignal
from PyQt6.QtWidgets import QDialog, QPushButton
from PyQt6.QtTest import QTest

from Application.tabs import dataset_statistics_tab as ui
from Application import statistics_detail_process as process_module
from Application.statistics_detail_job import SourceInfo
from Application.statistics_unified_context import context_for_result
from Application.analysis_filter import AnalysisFilter
from tests.tabs.test_dataset_statistics_tab import app, window, rich_customer_result, wait_until

__all__=["app","window","rich_customer_result"]
pytestmark=pytest.mark.usefixtures("window_settings")


@pytest.fixture
def flow(window,rich_customer_result,tmp_path,monkeypatch):
    host,tab=window
    tab.render(rich_customer_result)
    calls=[]
    main_thread=get_ident()
    options=dict(destination=tmp_path/"report.zip")
    class Dialog:
        accepted=True
    def choose(*args):
        calls.append(("save_dialog",args))
        return (str(options["destination"]) if Dialog.accepted else "", "ZIP-архив (*.zip)")
    monkeypatch.setattr(ui.QFileDialog,"getSaveFileName",choose)
    class Worker(QObject):
        progress=pyqtSignal(dict)
        completed=pyqtSignal(dict)
        stalled=pyqtSignal()
        cancel_requested=False
        def start(self,options,*,report_result=None):
            self.options=deepcopy(options)
            self.snapshot=deepcopy(report_result)
            calls.append(("worker",self))
        def cancel(self):
            self.cancel_requested=True
        def shutdown(self):
            self.cancel()
        def force_stop(self):
            self.cancel()
    monkeypatch.setattr(ui,"DetailExportProcess",Worker)
    return host,tab,rich_customer_result,options,calls,Dialog,main_thread


def test_one_button_requires_displayed_result(window,monkeypatch):
    _,tab=window
    assert not tab.export_button.isEnabled()
    assert tab.export_button.text().strip()=="Экспорт отчета"
    texts=[b.text().strip() for b in tab.findChildren(QPushButton)]
    assert texts.count("Экспорт отчета")==1
    assert "Экспорт статистики" not in texts and "Экспорт данных покупателей" not in texts
    monkeypatch.setattr(ui.QFileDialog,"getSaveFileName",lambda *_:pytest.fail("No result: no dialog"))
    tab.export()
    assert tab.detail_process is None and tab._export_task is None


@pytest.mark.parametrize("configured", [
    AnalysisFilter("2024-01-01","2024-12-31"),
    AnalysisFilter(nomenclature_types=("Новый фильтр",)),
    AnalysisFilter(collections=("Новый сезон",)),
])
def test_one_background_operation_captures_displayed_result_and_blocks_competitors(flow,monkeypatch,configured):
    _,tab,result,options,calls,_,main_thread=flow
    before=deepcopy(result)
    tab.cache_path.write_bytes(b"SYNTH_UNCHANGED_AGGREGATE_CACHE")
    tab.set_analysis_filter(configured)
    tab.export_button.click()
    wait_until(lambda:tab.detail_process is not None)
    worker=tab.detail_process
    assert worker.options==options and tab._export_task is None
    assert next(c[1] for c in calls if c[0]=="save_dialog")[3]=="ZIP-архив (*.zip)"
    assert worker.snapshot==before
    assert not tab.refresh.isEnabled() and not tab.export_button.isEnabled()
    monkeypatch.setattr(ui,"QProcess",lambda *_:pytest.fail("No competing calculation"))
    tab.start();tab.export();tab.export_details()
    assert sum(c[0]=="worker" for c in calls)==1
    updated=deepcopy(result);updated["total_interactions"]+=1
    tab.render(updated)
    assert worker.snapshot==before
    worker.progress.emit(dict(stage="statistics_xlsx",processed=1))
    assert "Excel" in tab.detail_status.text()
    worker.completed.emit(dict(event="finished",files=2))
    assert tab.detail_process is None and tab.export_button.isEnabled() and tab.refresh.isEnabled()
    assert str(options["destination"]) in tab.detail_status.text()
    assert tab.cache_path.read_bytes()==b"SYNTH_UNCHANGED_AGGREGATE_CACHE"


def test_cancel_error_retry_preserves_navigation_and_never_computes(flow):
    _,tab,_,_,calls,_,_=flow
    tab.sections.setCurrentIndex(3);tab.customer_sections.setCurrentIndex(4)
    tab.export()
    wait_until(lambda:tab.detail_process is not None)
    worker=tab.detail_process
    tab.cancel_details()
    assert worker.cancel_requested
    worker.completed.emit(dict(event="cancelled"))
    assert "отменен" in tab.detail_status.text()
    assert tab.sections.currentIndex()==3 and tab.customer_sections.currentIndex()==4
    tab.export()
    wait_until(lambda:tab.detail_process is not None)
    tab.detail_process.completed.emit(dict(event="error",code="XLSX_CELL_TOO_LONG"))
    assert "32767" in tab.detail_status.text() and tab.export_button.isEnabled()
    tab.export()
    wait_until(lambda:tab.detail_process is not None)
    tab.detail_process.completed.emit(dict(event="finished",files=2))
    assert sum(c[0]=="worker" for c in calls)==3


def test_report_starts_without_source_preflight_and_uses_only_destination(flow):
    _,tab,_,options,calls,_,_=flow
    tab.export()
    assert tab._export_task is None and tab.detail_process is not None
    assert [c[0] for c in calls]==["save_dialog","worker"]
    assert tab.detail_process.options==options
    tab.detail_process.completed.emit(dict(event="cancelled"))


def test_modal_cancel_never_starts_worker(flow):
    _,tab,_,_,calls,Dialog,_=flow
    Dialog.accepted=False
    tab.export()
    wait_until(lambda:tab._export_task is None and not tab._detail_dialog_open)
    assert not any(c[0]=="worker" for c in calls)
    assert tab.export_button.isEnabled()


def test_existing_qprocess_receives_report_over_private_stdin_not_arguments(app,rich_customer_result,tmp_path,monkeypatch):
    captured=[]
    script=("import sys,json; message=json.loads(sys.stdin.buffer.readline()); "
            "assert message['command']=='report'; "
            "assert message['snapshot']['total_interactions']=="+str(rich_customer_result["total_interactions"])+"; "
            "print(json.dumps(dict(event='finished',files=2)),flush=True)")
    class Child(QProcess):
        def start(self,program,arguments):
            import sys
            captured.append(arguments)
            super().start(sys.executable,["-B","-u","-c",script])
    monkeypatch.setattr(process_module,"QProcess",Child)
    worker=process_module.DetailExportProcess()
    terminal=[]
    worker.completed.connect(terminal.append)
    worker.start(dict(destination=tmp_path/"report.zip"),report_result=rich_customer_result)
    wait_until(lambda:bool(terminal))
    assert terminal==[dict(event="finished",files=2)]
    assert captured[0][3:]==["--destination",str(tmp_path/"report.zip")]
    assert not any("total_interactions" in arg for arg in captured[0])
    worker.deleteLater();QTest.qWait(10)



def test_shutdown_before_tab_destruction_stops_report_worker(flow):
    from PyQt6.QtCore import QCoreApplication,QEvent
    _,tab,_,_,calls,_,_=flow
    tab.export()
    worker=tab.detail_process
    assert worker is not None and tab._export_task is None
    tab._shutdown()
    tab.deleteLater()
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    assert worker.cancel_requested
    assert sum(c[0]=="worker" for c in calls)==1


@pytest.mark.parametrize("tamper",[None,"as_of","mode"])
def test_unified_dialog_freezes_displayed_context_and_rejects_tampering(
        window,rich_customer_result,tmp_path,tamper):
    from datetime import timedelta
    from PyQt6.QtCore import QDate
    from Application.tabs.statistics_detail_dialog import DetailExportDialog
    _,tab=window
    source=SourceInfo(date(2020,1,1),date(2026,8,31))
    selection=AnalysisFilter.from_dict(rich_customer_result["analysis_filter"])
    context=context_for_result(rich_customer_result)
    dialog=DetailExportDialog(tab,source,selection,tmp_path/"source",report_result=rich_customer_result)
    private=tmp_path/"private";private.mkdir()
    target=tmp_path/"report.zip"
    dialog.destination.setText(str(target));dialog.temporary_root.setText(str(private))
    for control in (dialog.all_history,dialog.start_date,dialog.end_date,dialog.as_of,dialog.mode):
        assert not control.isEnabled()
    assert dialog.as_of.date().toPyDate()==context.as_of
    if tamper=="as_of":
        other=context.as_of-timedelta(days=1)
        dialog.as_of.setDate(QDate(other.year,other.month,other.day))
    elif tamper=="mode":
        dialog.mode.setCurrentIndex(1)
    dialog.accept()
    assert not target.exists()
    if tamper:
        assert dialog.result()!=QDialog.DialogCode.Accepted and dialog.options is None and dialog.error.text()
    else:
        assert dialog.result()==QDialog.DialogCode.Accepted
        assert (dialog.options["start"],dialog.options["end"],dialog.options["as_of"])==(
            context.period.start_date,context.period.end_date,context.as_of)


def test_report_suffix_collision_requires_exact_target_confirmation(window,rich_customer_result,tmp_path,monkeypatch):
    from Application.tabs.statistics_detail_dialog import DetailExportDialog
    from Application.tabs import statistics_detail_dialog as dialogs
    from PyQt6.QtWidgets import QMessageBox
    _,tab=window
    dialog=DetailExportDialog(tab,SourceInfo(date(2020,1,1),date(2026,8,31)),
        AnalysisFilter.from_dict(rich_customer_result["analysis_filter"]),tmp_path/"source",
        report_result=rich_customer_result)
    private=tmp_path/"private";private.mkdir();dialog.temporary_root.setText(str(private))
    target=tmp_path/"report.zip";target.write_bytes(b"OLD ZIP")
    monkeypatch.setattr(dialogs.QFileDialog,"getSaveFileName",lambda *a:(str(target.with_suffix("")),"ZIP (*.zip)"))
    confirmations=[]
    def deny(*a):
        confirmations.append(a);return QMessageBox.StandardButton.No
    monkeypatch.setattr(dialogs.QMessageBox,"question",deny)
    dialog.choose_destination();dialog.accept()
    assert confirmations and dialog.result()!=QDialog.DialogCode.Accepted
    assert target.read_bytes()==b"OLD ZIP" and dialog.options is None


def test_unified_qprocess_cancellation_follows_report_stdin(app,rich_customer_result,tmp_path,monkeypatch):
    import sys
    script=("import sys,json; m=json.loads(sys.stdin.buffer.readline()); assert m['command']=='report'; "
            "print(json.dumps(dict(event='progress',stage='orders',processed=0)),flush=True); "
            "c=json.loads(sys.stdin.buffer.readline()); assert c==dict(command='cancel'); "
            "print(json.dumps(dict(event='cancelled')),flush=True);sys.exit(2)")
    class Child(QProcess):
        def start(self,program,arguments):
            super().start(sys.executable,["-B","-u","-c",script])
    monkeypatch.setattr(process_module,"QProcess",Child)
    worker=process_module.DetailExportProcess()
    terminal=[]
    worker.progress.connect(lambda _:worker.cancel())
    worker.completed.connect(terminal.append)
    worker.start(dict(destination=tmp_path/"report.zip"),report_result=rich_customer_result)
    wait_until(lambda:bool(terminal))
    assert terminal==[dict(event="cancelled")] and worker.process is None
    assert not (tmp_path/"report.zip").exists()
    worker.deleteLater();QTest.qWait(10)

