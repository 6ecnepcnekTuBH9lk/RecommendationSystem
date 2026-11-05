"""Production QProcess, actual two-file ZIP, native Save As wiring and upper cancel."""
from copy import deepcopy
import ctypes
import json
import os
from time import perf_counter
import pytest
from PyQt6.QtCore import QProcess,QProcessEnvironment,QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QProgressBar
from Application import statistics_detail_process as process_module
from Application.tabs import dataset_statistics_tab as ui
from tests.tabs.test_dataset_statistics_tab import app,window,sample_result,rich_customer_result,wait_until
from tests.test_statistics_report_export import assert_archive,clean

__all__=["app","window","sample_result","rich_customer_result"]
pytestmark=[pytest.mark.skipif(os.name!="nt",reason="Windows QProcess/ACL integration"),
            pytest.mark.usefixtures("window_settings")]


def observe(monkeypatch,tmp_path):
    """Run the real script and arguments; only monitor child IO and RAM."""
    monitor=tmp_path/"monitor";monitor.mkdir()
    metrics=tmp_path/"worker-memory.json"
    (monitor/"sitecustomize.py").write_text(
        "import sys,os,atexit,json\nfrom pathlib import Path\n"
        "def guard(event,args):\n"
        " if event=='sqlite3.connect': raise AssertionError('Report attempted SQLite')\n"
        " if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):\n"
        "  parts=Path(os.fsdecode(args[0])).parts\n"
        "  if any(p.lower() in ('canonical','statistics_index') for p in parts):\n"
        "   raise AssertionError('Report attempted source/index IO')\n"
        "sys.addaudithook(guard)\n"
        "@atexit.register\ndef measure():\n"
        " from Application.statistics_index import sample_memory\n"
        " Path("+repr(str(metrics))+").write_text(json.dumps(sample_memory()),encoding='utf-8')\n",
        encoding="utf-8")
    launch=[];stderr=bytearray()
    class Observed(QProcess):
        def start(self,program,arguments):
            env=QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONPATH",str(monitor)+os.pathsep+env.value("PYTHONPATH"))
            self.setProcessEnvironment(env)
            launch.append((program,arguments,self.workingDirectory()))
            super().start(program,arguments)
        def read(self,size):
            data=super().read(size)
            if self.readChannel()==QProcess.ProcessChannel.StandardError:stderr.extend(bytes(data))
            return data
    monkeypatch.setattr(process_module,"QProcess",Observed)
    return launch,stderr,metrics


def test_real_production_entrypoint_creates_unified_zip(app,sample_result,monkeypatch,tmp_path):
    launch,stderr,metrics=observe(monkeypatch,tmp_path)
    worker=process_module.DetailExportProcess();terminal=[];stages=[]
    worker.completed.connect(terminal.append);worker.progress.connect(lambda m:stages.append(m["stage"]))
    target=tmp_path/"report.zip";started=perf_counter()
    worker.start(dict(destination=target),report_result=sample_result)
    try:
        wait_until(lambda:bool(terminal),timeout=60000)
        assert terminal==[dict(event="finished",files=2)],(terminal,stages,stderr.decode(errors="replace")[-2000:],launch)
        elapsed=perf_counter()-started
        assert_archive(target);clean(tmp_path)
        assert launch[0][1][3:]==["--destination",str(target)]
        assert launch[0][2]==str(ui.PROJECT_ROOT)
        assert set(stages)=={"report_preparation","statistics_xlsx","statistics_html","archive"}
        assert worker.diagnostic["exit_code"]==0 and worker.diagnostic["exit_status"]=="NormalExit"
        assert not worker.diagnostic["crash_exit"] and "_enter_buffered_busy" not in stderr.decode(errors="replace")
        assert set(worker.diagnostic["timings"])=={"presentation","excel","html","zip","total"}
        memory=json.loads(metrics.read_text(encoding="utf-8"))
        assert 0<memory["peak_working_set"]<1024**3 and 0<memory["peak_pagefile_usage"]<1024**3
        (tmp_path/"process-measurement.json").write_text(json.dumps(dict(elapsed=elapsed,memory=memory,
            diagnostic=worker.diagnostic),ensure_ascii=False),encoding="utf-8")
    finally:
        worker.shutdown();worker.deleteLater();QTest.qWait(10)


def test_button_native_save_production_worker_and_single_progress(window,rich_customer_result,monkeypatch,tmp_path):
    _,tab=window
    observe(monkeypatch,tmp_path)
    result=deepcopy(rich_customer_result);target=tmp_path/"report.zip";saved=[];ticks=[]
    def choose(*args):
        saved.append(args);return str(target),"ZIP-архив (*.zip)"
    monkeypatch.setattr(ui.QFileDialog,"getSaveFileName",choose)
    timer=QTimer(tab);timer.timeout.connect(lambda:ticks.append(1));timer.start(5)
    tab.render(result);before=deepcopy(tab.displayed_result)
    tab.export_button.click()
    assert len(saved)==1 and saved[0][1]=="Экспорт отчета" and saved[0][3]=="ZIP-архив (*.zip)"
    assert len(tab.findChildren(QProgressBar))==1 and tab._export_task is None
    try:
        wait_until(lambda:tab.detail_process is None,timeout=60000)
        assert "Отчет сохранен" in tab.status.text(),(tab.status.text(),tab.last_export_diagnostic)
        assert ticks and tab.last_export_diagnostic["exit_code"]==0
        assert tab.last_export_diagnostic["exit_status"]=="NormalExit" and not tab.last_export_diagnostic["crash_exit"]
        assert tab.displayed_result==before and tab.progress.maximum()==100 and tab.export_button.isEnabled()
        assert_archive(target);clean(tmp_path)
    finally:
        timer.stop();tab._shutdown()


@pytest.mark.parametrize("kind",["cancel","context_error"])
def test_real_entrypoint_cancel_or_python_error_preserves_existing_archive(window,sample_result,tmp_path,monkeypatch,kind):
    _,tab=window;target=tmp_path/"report.zip";target.write_bytes(b"PREVIOUS ZIP")
    observe(monkeypatch,tmp_path)
    tab.render(sample_result)
    # Missing snapshot is rejected safely in the child, independently of source coverage.
    if kind=="context_error":
        result=deepcopy(sample_result);result.pop("orders")
        worker=process_module.DetailExportProcess();terminal=[]
        worker.completed.connect(terminal.append)
        worker.start(dict(destination=target),report_result=result)
        try:
            wait_until(lambda:bool(terminal),timeout=60000)
            assert terminal==[dict(event="error",code="REPORT_CONTEXT_MISMATCH")]
            assert worker.diagnostic["exit_status"]=="NormalExit" and worker.diagnostic["traceback"]
        finally:
            worker.shutdown();worker.deleteLater();QTest.qWait(10)
    else:
        monkeypatch.setattr(ui.QFileDialog,"getSaveFileName",lambda *a:(str(target),"ZIP-архив (*.zip)"))
        real_start=process_module.DetailExportProcess.start;clicks=[]
        def start(worker,*args,**kwargs):
            def cancel(m):
                if m["stage"]=="statistics_xlsx" and not clicks:
                    clicks.append(True);tab.cancel_button.click()
            worker.progress.connect(cancel)
            return real_start(worker,*args,**kwargs)
        monkeypatch.setattr(process_module.DetailExportProcess,"start",start)
        tab.export_button.click()
        try:
            wait_until(lambda:tab.detail_process is None,timeout=60000)
            assert clicks and "отменен" in tab.status.text(),(tab.status.text(),tab.last_export_diagnostic)
            assert tab.last_export_diagnostic["exit_code"]==2 and not tab.last_export_diagnostic["crash_exit"]
            assert tab.export_button.isEnabled() and tab.refresh.isEnabled()
        finally:tab._shutdown()
    assert target.read_bytes()==b"PREVIOUS ZIP";clean(tmp_path)


def test_production_write_error_keeps_old_zip_and_retry_succeeds(window,sample_result,tmp_path,monkeypatch):
    _,tab=window;target=tmp_path/"report.zip";target.write_bytes(b"PREVIOUS ZIP")
    observe(monkeypatch,tmp_path)
    monkeypatch.setattr(ui.QFileDialog,"getSaveFileName",lambda *a:(str(target),"ZIP-архив (*.zip)"))
    tab.render(sample_result)
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,
                                ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
    kernel.CreateFileW.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    # Read sharing allows admission/version checks, but prevents atomic replacement.
    handle=kernel.CreateFileW(str(target),0x80000000,1,None,3,0,None)
    assert handle not in (None,ctypes.c_void_p(-1).value),ctypes.get_last_error()
    try:
        tab.export_button.click()
        wait_until(lambda:tab.detail_process is None,timeout=60000)
        assert "ACCESS_DENIED" in tab.status.text(),(tab.status.text(),tab.last_export_diagnostic)
        assert "Отчет сохранен" not in tab.status.text()
        assert tab.last_export_diagnostic["exit_code"]==1 and not tab.last_export_diagnostic["crash_exit"]
        assert target.read_bytes()==b"PREVIOUS ZIP" and tab.export_button.isEnabled()
        clean(tmp_path)
    finally:kernel.CloseHandle(handle)
    tab.export_button.click()
    try:
        wait_until(lambda:tab.detail_process is None,timeout=60000)
        assert "Отчет сохранен" in tab.status.text(),(tab.status.text(),tab.last_export_diagnostic)
        assert_archive(target);clean(tmp_path)
    finally:tab._shutdown()

