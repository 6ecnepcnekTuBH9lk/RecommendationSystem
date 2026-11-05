"""Geometry regression: the last section fills spare viewport width."""
import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from Application.tabs import dataset_statistics_tab as ui
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_dataset_statistics_tab import app

__all__ = ["app"]
pytestmark = pytest.mark.usefixtures("window_settings")


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("columns", [2, 4])
def test_last_section_tracks_manual_resize_and_viewport_without_blank_space(app, dark, columns):
    apply_app_theme(app, dark)
    host = QWidget()
    layout = QVBoxLayout(host)
    table = ui._table(layout, ["Показатель", *(f"Значение {c}" for c in range(1, columns))],
                      [("Покупатели", *range(1, columns)), ("Другой показатель", *range(1, columns))])
    host.resize(1100, 550)
    host.show()
    try:
        header = table.horizontalHeader()
        for width in (1100, 700, 1250):
            host.resize(width, 550)
            for first in (180, 460, 220):
                header.resizeSection(0, first)
                QTest.qWait(20)
                last = columns-1
                end = header.sectionViewportPosition(last) + header.sectionSize(last)
                if sum(table._minimum_widths) <= table.viewport().width():
                    assert table.horizontalScrollBar().maximum() == 0
                    assert abs(end-table.viewport().width()) <= 1
                else:
                    assert table.horizontalScrollBar().maximum() > 0
                assert table.verticalScrollBar().maximum() == 0
                assert table.item(0, 1).text() == "1"
    finally:
        host.close()
        host.deleteLater()


@pytest.mark.parametrize("dark", [False, True])
def test_wide_table_keeps_last_money_minimum_and_last_section_accessible(app, dark):
    apply_app_theme(app, dark)
    host = QWidget()
    layout = QVBoxLayout(host)
    amount = ui._decimal_display("1234567890123456789012345678901234567890.67")
    table = ui._table(layout, ["Показатель", "Заказы", "Полная сумма, RUB"],
                      [("Длинное название показателя", 1, amount)])
    host.resize(600, 500)
    host.show()
    try:
        QTest.qWait(30)
        minimum = table.fontMetrics().horizontalAdvance(amount)+12
        for first in (300, 700, 180):
            table.horizontalHeader().resizeSection(0, first)
            QTest.qWait(20)
            assert table.columnWidth(2) >= minimum
            assert table.horizontalScrollBar().maximum() > 0
            table.horizontalScrollBar().setValue(table.horizontalScrollBar().maximum())
            QTest.qWait(10)
            end = table.horizontalHeader().sectionViewportPosition(2)+table.columnWidth(2)
            assert end <= table.viewport().width()+1
            assert table.item(0, 2).text() == amount
    finally:
        host.close()
        host.deleteLater()



@pytest.mark.parametrize("dark",[False,True])
def test_long_left_label_keeps_wrapped_lines_when_last_column_fills(app,dark):
    from PyQt6.QtCore import QRect,Qt
    apply_app_theme(app,dark)
    host=QWidget();layout=QVBoxLayout(host)
    label="Клиенты с просмотрами / избранным без товарного идентификатора"
    table=ui._table(layout,["Показатель","Значение"],[(label,123)])
    host.resize(1100,550);host.show()
    try:
        for width in (440,280,180):
            table.horizontalHeader().resizeSection(0,width);QTest.qWait(20)
            text_width=table.columnWidth(0)-2*ui.CONTROL_SPACING-4
            height=table.fontMetrics().boundingRect(QRect(0,0,text_width,100000),
                Qt.TextFlag.TextWordWrap|Qt.TextFlag.TextExpandTabs,label).height()
            assert table.rowHeight(0)>=height+2*ui.CONTROL_SPACING
            assert table.item(0,0).text()==label
            assert table.horizontalHeader().sectionViewportPosition(1)+table.columnWidth(1)==table.viewport().width()
    finally:
        host.close();host.deleteLater()

