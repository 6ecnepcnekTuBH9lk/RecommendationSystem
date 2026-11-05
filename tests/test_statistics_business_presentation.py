"""New demographic sections share the GUI/XLSX/HTML contract and legacy snapshots."""
from copy import deepcopy
from io import BytesIO
from zipfile import ZipFile
import sqlite3
import pytest
from openpyxl import load_workbook
from Application import statistics_cache as cache,statistics_report_presentation as report
from Application import statistics_presentation as presentation,statistics_bundle_export as bundle
from Application.theme.apply_theme import apply_app_theme
from Application.tabs import dataset_statistics_tab as ui
from tests.tabs.test_dataset_statistics_tab import app,window,rich_customer_result
from tests.test_statistics_unified_presentation import Tables
from PyQt6.QtWidgets import QApplication,QTableWidget

__all__=["app","window","rich_customer_result"]
pytestmark=pytest.mark.usefixtures("window_settings")


@pytest.mark.parametrize("dark",[False,True])
@pytest.mark.parametrize("legacy",[False,True])
def test_six_tables_cache_roundtrip_shared_values_and_no_recalculation(window,rich_customer_result,tmp_path,monkeypatch,dark,legacy):
    from Application import dataset_statistics,statistics_slices
    from Application.statistics_rfm_core import RfmAggregates
    result=deepcopy(rich_customer_result)
    strict=result["buyer_statistics"]["policies"]["STRICT_COMPLETED"]
    expected=deepcopy(strict["demographic_analytics"])
    if legacy:strict.pop("demographic_analytics")
    saved=tmp_path/"saved.json";cache.save_result(saved,result)
    payload=saved.read_bytes();result=cache.load_result(saved)
    assert result is not None
    before=deepcopy(result)
    def forbidden(*a,**k):pytest.fail("Saved report must not read sources or recalculate")
    monkeypatch.setattr(dataset_statistics,"calculate_statistics",forbidden)
    monkeypatch.setattr(dataset_statistics,"iter_export",forbidden)
    monkeypatch.setattr(statistics_slices,"calculate_cached_statistics",forbidden)
    monkeypatch.setattr(RfmAggregates,"result",forbidden)
    monkeypatch.setattr(sqlite3,"connect",forbidden)
    monkeypatch.setattr(ui.DatasetStatisticsTab,"start",forbidden)
    host,tab=window;apply_app_theme(QApplication.instance(),dark);tab.render(result)
    sections=presentation.grouped_buyer_sections(result,compact=True)
    overview=[s for g,s in sections if g=="overview"]
    assert [s[0] for s in overview]==["Портрет покупателей — подтверждённые покупки",
        "Покупательская активность по полу","Покупательская активность по возрасту"]
    money=[s for g,s in sections if g=="money"]
    assert [s[0] for s in money][-5:]==["Денежные показатели покупателей по демографии",
        "По полу — RUB","По полу — KZT","По возрасту — RUB","По возрасту — KZT"]
    assert not any(cards for _,cards,_,_ in overview[1:]+money[2:])
    assert "по доступному профилю на дату расчета" in overview[-1][3][0][1]
    assert tab.sections.count()==tab.customer_sections.count()==5
    assert len(tab.customer_sections.findChildren(QTableWidget))==60
    target=tmp_path/"statistics.zip";bundle.export_statistics_bundle(result,target)
    with ZipFile(target) as archive:
        assert archive.namelist()==["Статистика.xlsx","Статистика.html"] and archive.testzip() is None
        book=load_workbook(BytesIO(archive.read("Статистика.xlsx")))
        document=archive.read("Статистика.html").decode("utf-8")
    assert book.sheetnames==list(report.PAGE_LABELS)
    parsed=Tables(document)
    for group_index in (0,2):
        group=tab.report.pages[3].groups[group_index]
        actual_tables=iter(tab.customer_sections.widget(group_index).findChildren(QTableWidget))
        for i,block in enumerate(group.blocks):
            if block.kind!="table":continue
            gui=next(actual_tables)
            expected_rows=[[c.text for c in row] for row in report.table_cells(block)]
            assert [[gui.item(r,c).text() for c in range(gui.columnCount())] for r in range(gui.rowCount())]==expected_rows
            assert parsed.tables[f"table-3-{group.key}-{i}"]==expected_rows if expected_rows else parsed.tables[f"table-3-{group.key}-{i}"]==[["Нет данных для таблицы."]]
    excel_values=[tuple(row) for row in book["Клиенты"].values]
    if legacy:
        assert presentation.BUSINESS_LEGACY_TEXT in document
        assert any(presentation.BUSINESS_LEGACY_TEXT in row for row in excel_values)
        assert any(presentation.BUSINESS_LEGACY_TEXT in (t.item(r,c).text() for r in range(t.rowCount()) for c in range(t.columnCount()))
                   for t in tab.customer_sections.findChildren(QTableWidget))
    else:
        assert [r[0] for r in overview[1][2][0][1]]==[g["label"] for g in expected["gender"]]
        for (dimension,currency),section in zip(
            (("gender","RUB"),("gender","KZT"),("age","RUB"),("age","KZT")),money[-4:]):
            headers,rows=section[2][0]
            assert currency in headers[4] and currency in headers[5] and currency in headers[6]
            for group,row in zip(expected[dimension],rows):
                m=group["money"][currency]
                assert row[1:4]==(m["known_buyers"],m["known_orders"],group["unvalued_orders"])
                assert any(candidate[:4]==tuple(row[:4]) for candidate in excel_values)
        assert len(parsed.tables)==89
    book.close()
    assert result==before and saved.read_bytes()==payload
    assert host.isVisible()


def test_unavailable_profiles_are_distinct_from_old_results_and_not_fake_zero_distribution(rich_customer_result):
    result=deepcopy(rich_customer_result)
    strict=result["buyer_statistics"]["policies"]["STRICT_COMPLETED"]
    strict["demographic_analytics"]=None;strict["demography"]=None
    sections=presentation.grouped_buyer_sections(result,compact=True)
    new=[s for g,s in sections if s[0].startswith("Покупательская активность по")]
    assert len(new)==2
    for _,_,tables,_ in new:
        assert tables[0][1]==[("Демографическая аналитика","Клиентские профили недоступны. Демографическая аналитика не рассчитана")]
    strict.pop("demographic_analytics")
    legacy=presentation.grouped_buyer_sections(result,compact=True)
    assert any(presentation.BUSINESS_LEGACY_TEXT in str(s) for _,s in legacy)


def test_new_calculation_dependency_is_attested_and_cache_schema_stays_readable(monkeypatch):
    from Application import statistics_slices
    before=statistics_slices._algorithm_version()
    old=statistics_slices._hash_file
    monkeypatch.setattr(statistics_slices,"_hash_file",lambda p:"CHANGED" if p.name=="statistics_buyer_demography.py" else old(p))
    assert statistics_slices._algorithm_version()!=before
    assert cache.SCHEMA_VERSION==10
