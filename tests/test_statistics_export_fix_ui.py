"""Actual header drags must not force a fitting table out of its viewport."""
import pytest
from PyQt6.QtCore import QPoint,Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget,QVBoxLayout
from Application.tabs import dataset_statistics_tab as ui
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_dataset_statistics_tab import app,window,rich_customer_result
from tests.tabs.test_statistics_unified_report import flow

__all__=["app","window","rich_customer_result","flow"]
pytestmark=pytest.mark.usefixtures("window_settings")

def drag(table,column,delta):
    header=table.horizontalHeader();viewport=header.viewport()
    x=header.sectionViewportPosition(column)+header.sectionSize(column)-1
    start=QPoint(x,header.height()//2)
    QTest.mousePress(viewport,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
    QTest.mouseMove(viewport,QPoint(x+delta,start.y()),20)
    QTest.mouseRelease(viewport,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(x+delta,start.y()))
    QTest.qWait(20)

@pytest.mark.parametrize("dark",[False,True])
@pytest.mark.parametrize("columns",[2,3,5,9])
def test_real_drag_resize_window_keeps_last_edge_and_no_scroll_if_minima_fit(app,dark,columns):
    apply_app_theme(app,dark)
    host=QWidget();layout=QVBoxLayout(host)
    table=ui._table(layout,["Показатель",*(f"Поле {i}" for i in range(1,columns))],
                    [("Покупатели",*range(1,columns))])
    host.resize(1500,550);host.show();QTest.qWait(30)
    try:
        for action in ((0,1000),(0,-400),(columns//2,1000),(0,300),(max(0,columns-2),500)):
            drag(table,*action)
            assert table.horizontalScrollBar().maximum()==0
            header=table.horizontalHeader()
            assert abs(header.sectionViewportPosition(columns-1)+header.sectionSize(columns-1)-table.viewport().width())<=1
        for width in (1300,1600,1200,1500):
            host.resize(width,550);QTest.qWait(25)
            assert table.horizontalScrollBar().maximum()==0
            header=table.horizontalHeader()
            assert abs(header.sectionViewportPosition(columns-1)+header.sectionSize(columns-1)-table.viewport().width())<=1
    finally:
        host.close();host.deleteLater()



def test_save_cancel_keeps_state_and_never_preflights(window,rich_customer_result,monkeypatch):
    from PyQt6.QtWidgets import QProgressBar
    _,tab=window;tab.render(rich_customer_result)
    before=(tab.status.text(),tab.progress.minimum(),tab.progress.maximum(),tab.progress_text.text())
    calls=[]
    monkeypatch.setattr(ui.QFileDialog,"getSaveFileName",lambda *a:(calls.append(a) or ("","ZIP-архив (*.zip)")))
    monkeypatch.setattr(ui,"validate_report_destination",lambda *a:pytest.fail("Cancel must not validate destination"))
    monkeypatch.setattr(ui,"DetailExportProcess",lambda *a,**k:pytest.fail("Cancel must never create worker"))
    tab.export_button.click()
    assert len(calls)==1 and calls[0][1]=="Экспорт отчета" and calls[0][3]=="ZIP-архив (*.zip)"
    assert tab.export_button.text().strip()=="Экспорт отчета"
    assert len(tab.findChildren(QProgressBar))==1
    assert not hasattr(tab,"detail_panel") and not hasattr(tab,"detail_force")
    assert (tab.status.text(),tab.progress.minimum(),tab.progress.maximum(),tab.progress_text.text())==before
    assert tab._export_task is None and tab.detail_process is None and tab.export_button.isEnabled()


@pytest.mark.parametrize("dark",[False,True])
def test_rerender_and_navigation_keep_fitting_table_edges(window,rich_customer_result,app,dark):
    from PyQt6.QtWidgets import QTableWidget
    apply_app_theme(app,dark)
    host,tab=window;host.resize(1450,1000)
    for _ in range(2):
        tab.render(rich_customer_result)
        for outer in range(5):
            tab.sections.setCurrentIndex(outer);QTest.qWait(20)
            for inner in range(5) if outer==3 else (None,):
                if inner is not None:tab.customer_sections.setCurrentIndex(inner)
                QTest.qWait(20)
                page=tab.customer_sections.widget(inner) if inner is not None else tab.sections.widget(outer)
                for table in page.findChildren(QTableWidget):
                    if not table.rowCount():continue
                    if sum(table._minimum_widths)<=table.viewport().width():
                        drag(table,0,5000)
                        assert table.horizontalScrollBar().maximum()==0
                        h=table.horizontalHeader();last=table.columnCount()-1
                        assert abs(h.sectionViewportPosition(last)+h.sectionSize(last)-table.viewport().width())<=1
                    else:
                        table.horizontalScrollBar().setValue(table.horizontalScrollBar().maximum())
                        assert table.visualItemRect(table.item(0,table.columnCount()-1)).right()<=table.viewport().width()


def test_shared_progress_and_upper_cancel(flow):
    from PyQt6.QtWidgets import QProgressBar
    _,tab,_,_,_,_,_=flow
    tab.export()
    from tests.tabs.test_dataset_statistics_tab import wait_until
    wait_until(lambda:tab.detail_process is not None)
    assert len(tab.findChildren(QProgressBar))==1
    worker=tab.detail_process
    worker.progress.emit(dict(stage="statistics_html",processed=3))
    assert "HTML" in tab.progress_text.text() and tab.progress.maximum()==0
    assert tab.cancel_button.isEnabled()
    tab.cancel_button.click()
    assert worker.cancel_requested and not tab.cancel_button.isEnabled()
    worker.completed.emit(dict(event="cancelled"))
    assert tab.progress.maximum()==100 and tab.progress.value()==0
    assert tab.progress_text.text()=="Прогресс расчета"
    assert tab.export_button.isEnabled() and tab.refresh.isEnabled()



def test_worker_permission_failure_is_safe_and_restores_shared_controls(flow):
    _,tab,_,_,_,_,_=flow
    tab.export()
    tab.detail_process.completed.emit(dict(event="error",code="ACCESS_DENIED"))
    assert tab.detail_process is None and tab.refresh.isEnabled() and tab.export_button.isEnabled()
    assert tab.progress.maximum()==100 and not tab.cancel_button.isEnabled()
    assert "Отказано" in tab.status.text() and "PRIVATE" not in tab.status.text()+tab.status.toolTip()


def test_suffix_collision_uses_native_selection_then_exact_confirmation(flow,monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    _,tab,_,options,_,_,_=flow
    target=options["destination"];target.write_bytes(b"OLD ZIP")
    monkeypatch.setattr(ui.QFileDialog,"getSaveFileName",lambda *a:(str(target.with_suffix("")),"ZIP-архив (*.zip)"))
    confirmations=[]
    monkeypatch.setattr(ui.QMessageBox,"question",lambda *a:(confirmations.append(a) or QMessageBox.StandardButton.No))
    monkeypatch.setattr(ui,"DetailExportProcess",lambda *a:pytest.fail("Declined collision must not start worker"))
    tab.export()
    assert confirmations and tab.detail_process is None and tab._export_task is None
    assert target.read_bytes()==b"OLD ZIP"

