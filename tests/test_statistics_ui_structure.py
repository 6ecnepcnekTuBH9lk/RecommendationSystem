"""Structure fixes and two-decimal money tooltips over immutable saved aggregates."""
from copy import deepcopy
from decimal import Decimal, localcontext
from base64 import b64decode
from io import BytesIO
import sqlite3

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QTableWidget,QVBoxLayout,QWidget
from openpyxl import load_workbook

from Application import statistics_chart_data as data,statistics_charts as charts
from Application import statistics_report_presentation as report,statistics_html_export as html,statistics_export as xlsx
from Application import statistics_cache as cache,dataset_statistics as statistics
from Application.statistics_table_explanations import TableExplanation,TABLE_EXPLANATIONS,EXPLANATION_ICON,explanation_icon_data
from Application.statistics_table_explanation_widget import StatisticsTableExplanation
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_dataset_statistics_tab import app,window,sample_result,rich_customer_result,_visual_items

__all__=['app','window','sample_result','rich_customer_result']
pytestmark=pytest.mark.usefixtures('window_settings')


@pytest.mark.parametrize('unit',['RUB','KZT'])
@pytest.mark.parametrize('amount,expected',[
    ('0','0,00'),('0.004','0,00'),('0.125','0,12'),('12511.054200369123456789','12 511,05'),
    ('999999999999999999999999999999.995','1 000 000 000 000 000 000 000 000 000 000,00'),
])
def test_tooltip_money_exactly_two_places_both_currencies_counts_integer_and_original_decimal(unit,amount,expected,app):
    source={'store_statistics':((unit,'a','Магазин',67106,30277,1,'1',amount,amount),)}
    original=deepcopy(source)
    with localcontext() as context:
        flags=context.flags.copy()
        chart=data.store_purchase_chart_data(source,unit)
        assert context.flags==flags
    assert chart.series==((unit,(Decimal(amount),)),)
    tip=chart.tooltips[0][0]
    assert f'Сумма покупок: {expected} {unit}' in tip
    assert f'Средняя сумма заказа: {expected} {unit}' in tip
    assert 'Заказов: 67 106\nПокупателей: 30 277' in tip and source==original
    widget=charts.create_chart(chart);widget.resize(900,320);widget.show();QApplication.processEvents()
    try:
        assert widget.count_axis.titleText()=='' and widget.category_axis.titleText()==''
        widget.show_tooltip(0,0,True)
        assert tip==widget.tooltip.text()
        assert widget.category_axis.count()==1
    finally:widget.close();widget.deleteLater()


def test_only_three_axis_titles_removed_without_changing_labels_ticks_or_ranking(sample_result,app):
    sample_result['store_statistics']=(('RUB','b','Б',1,1,1,'1','100.001','100.001'),
                                       ('RUB','a','А',999,1,1,'1','100.000','100.000'),
                                       ('KZT','k','К',1,1,1,'1','9','9'))
    projects=[data.store_purchase_chart_data(sample_result,'RUB'),data.store_purchase_chart_data(sample_result,'KZT'),
              data.product_category_chart_data(sample_result)]
    for projection in projects:
        assert projection.kind=='horizontal' and projection.value_axis_title==''
        widget=charts.create_chart(projection);widget.resize(900,320);widget.show();QApplication.processEvents()
        try:
            assert not widget.count_axis.titleText() and not widget.category_axis.titleText()
            assert widget.category_axis.count()==len(projection.labels)
            assert widget._value_labels and widget.count_axis.max()>0
            assert [item.text() for item in widget._product_labels]==list(projection.display_labels)
        finally:widget.close();widget.deleteLater()
    assert projects[0].labels==('Б','А') and projects[0].series[0][1]==(Decimal('100.001'),Decimal('100.000'))
    assert data.buyer_frequency_chart_data({}).value_axis_title=='Покупателей'
    assert data.buyer_turnover_chart_data({},'RUB').value_axis_title=='Полный оборот, RUB'


@pytest.mark.parametrize('dark',[False,True])
def test_future_help_renders_after_exact_table_with_bundled_icon_gap_and_safe_html(app,tmp_path,monkeypatch,dark,sample_result):
    from Application.tabs import dataset_statistics_tab as ui
    help_text=TableExplanation('Синтетический пример <script>',(('Сумма','Безопасный текст'),))
    monkeypatch.setitem(TABLE_EXPLANATIONS,'synthetic.orders',help_text)
    blocks=(report.ReportBlock('heading',title='Пример'),
            report.ReportBlock('table',headers=('A','B'),rows=(('Строка',1),),table_id='synthetic.orders'),
            report.ReportBlock('table',headers=('C','D'),rows=(('Другая',2),),table_id='other'))
    host=QWidget();layout=QVBoxLayout(host);apply_app_theme(app,dark)
    ui._render_report_blocks(layout,blocks);host.resize(800,600);host.show();QTest.qWait(30)
    try:
        widgets=[item.widget() for item in _visual_items(layout) if item.widget()]
        assert [type(w) for w in widgets][-3:]==[ui._StatisticsTable,StatisticsTableExplanation,ui._StatisticsTable]
        panel=host.findChildren(StatisticsTableExplanation)[0]
        assert panel.content.isHidden() and panel.toggle.iconSize().width()==20 and not panel.toggle.icon().isNull()
        assert panel.toggle.text_rect().left()-(panel.toggle.icon_rect().right()+1)==8
        assert EXPLANATION_ICON.name=='question.png'
        image=QImage(str(EXPLANATION_ICON));assert image.hasAlphaChannel() and image.pixelColor(0,0).alpha()==0
        panel.toggle.setFocus();QTest.keyClick(panel.toggle,Qt.Key.Key_Space);QTest.qWait(20)
        assert not panel.content.isHidden() and 'развёрнуто' in panel.toggle.accessibleName()
        assert panel.content.height()>=panel.content.heightForWidth(panel.content.width())
        encoded=explanation_icon_data().split(',',1)[1]
        assert b64decode(encoded)==EXPLANATION_ICON.read_bytes()
        model=report.StatisticsReport((),(),'',(report.ReportPage('orders','Заказы',(report.ReportGroup('orders','Заказы',blocks),)),))
        document=html.build_statistics_html(sample_result,presentation=model)
        assert document.index('id="table-0-orders-1"')<document.index('<details class="table-explanation"')<document.index('id="table-0-orders-2"')
        assert document.count('<details class="table-explanation"')==1 and 'data:image/png;base64,' in document
        assert 'Синтетический пример &lt;script&gt;' in document and '<details open' not in document
        assert 'assets/icons' not in document and 'http' not in document.split('<img src="data:image/png;base64,')[1].split('">',1)[0]
    finally:host.close();host.deleteLater()


@pytest.mark.parametrize('scenario',['legacy','rich','empty'])
def test_current_saved_pages_have_no_panels_preserve_notes_tables_and_no_recalculation(
        window,sample_result,rich_customer_result,tmp_path,monkeypatch,scenario):
    from statistics_export_samples import empty_snapshot
    result=deepcopy(rich_customer_result if scenario=='rich' else empty_snapshot(sample_result) if scenario=='empty' else sample_result)
    path=tmp_path/'snapshot.json';cache.save_result(path,result);result=cache.load_result(path);payload=path.read_bytes()
    def forbidden(*a,**kw):pytest.fail('Presentation attempted source access or recalculation')
    monkeypatch.setattr(statistics,'calculate_statistics',forbidden);monkeypatch.setattr(statistics,'iter_export',forbidden)
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    _,tab=window;tab.render(result);QApplication.processEvents()
    assert TABLE_EXPLANATIONS=={} and not tab.findChildren(StatisticsTableExplanation)
    original=deepcopy(result);document=html.build_statistics_html(result)
    assert '<details class="table-explanation"' not in document
    from Application.statistics_presentation import DIRECT_CATEGORY_NOTE
    assert DIRECT_CATEGORY_NOTE in document
    stream=BytesIO();xlsx.write_statistics_xlsx(result,stream)
    book=load_workbook(BytesIO(stream.getvalue()))
    try:
        assert len(book.sheetnames)==4 and 'Как читать таблицу' not in str(list(book['Заказы'].values))
        for name in ['Определения сегментов и возможные бизнес-действия','Денежная ценность месячных когорт']:
            assert name in document  # Independent business sections/warnings remain.
        assert result==original and path.read_bytes()==payload
        for page in tab.report.pages:
            for group in page.groups:
                assert not any(b.kind=='explanation' for b in group.blocks)
        assert len(tab.findChildren(QTableWidget))>0
    finally:book.close()
