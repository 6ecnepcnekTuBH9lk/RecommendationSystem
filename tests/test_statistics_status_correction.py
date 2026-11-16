"""Restored report metrics on one synthetic purchase policy and both readers."""

from contextlib import closing
from decimal import Decimal
from io import BytesIO
import sqlite3
from zipfile import ZipFile

import pytest
from openpyxl import load_workbook

from Application import dataset_statistics as legacy
from Application import statistics_unified_index as index
from Application.mindbox.canonical_customers import encode
from Application.mindbox.selection import DEFAULT_SELECTION
from Application.statistics_detail_process import DetailExportProcess
from Application.statistics_index import protect_directory
from Application.statistics_report_presentation import report_presentation
from tests import test_statistics_unified_integration as integration
from tests.tabs import test_dataset_statistics_tab as qt
from tests.test_dataset_statistics import action


@pytest.fixture
def restored(tmp_path, monkeypatch):
    source = integration.source.__wrapped__(tmp_path, monkeypatch)
    orders = integration._write(source)
    orders[0]["payments"] = [{"type": "card"}, {"type": "card"}, {"type": "cash"}]
    integration.rfm_tests._write_orders(source, [*orders[:3], *orders[4:]])
    root, _, data, _, _ = source
    view, favorite = DEFAULT_SELECTION.view_action_system_names[0], DEFAULT_SELECTION.favorite_action_system_names[0]
    channel = {"ids": {"externalId": "shop"}, "name": "Магазин"}
    first, second = integration._FIRST, integration._SECOND
    events = [
        action(view, ["000001-a"], stamp=first, channel=channel, price=Decimal("1234.567"), available=True),
        action(view, ["000001-b"], stamp=second, channel=channel, price=Decimal("0"), available=False),
        action(view, ["000001-b"], "other", stamp=second, price=Decimal("500.005"), namespace="kanzlerKz"),
        action(favorite, ["000001-a"], stamp=first, channel=channel),
        action(view, stamp=second),
        action("Unmapped", ["000001-a"], stamp=second),
    ]
    directory = root / data["manual_interactions"]["actions"]["directory"]
    (directory / "actions_part_001.json").write_text(encode({"customerActions": events}), encoding="utf-8")
    with closing(sqlite3.connect(root / "canonical/customers.sqlite")) as profiles, profiles:
        profiles.execute("UPDATE profiles SET raw=json_set(raw,'$.birthDate','1990-01-01','$.sex','female') WHERE id='new'")
    return source


def _model(result):
    model = report_presentation(result)
    return model, {block.table_id: block for page in model.pages for group in page.groups
                   for block in group.blocks if block.kind == "table"}


def test_restored_dimensions_parity_and_purchase_valuation(restored):
    root, catalog, _, _, target = restored
    index.build_unified_index(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    reference = integration._report(restored)
    indexed = integration._report(restored, reader="INDEXED", index_dir=target)
    assert integration._business(reference) == integration._business(indexed)
    actions = indexed.actions["breakdowns"]
    assert actions["action_monthly_dynamics"] == [["2025-10", 1, 1, 1, 1], ["2025-11", 2, 2, 0, 0]]
    assert actions["view_availability_distribution"][0][:2] == ["Доступен", 1]
    assert Decimal(actions["view_price_statistics"][0][2]) == Decimal("617.2835")
    assert Decimal(actions["view_price_statistics"][1][3]) == Decimal("500.005")
    assert actions["source_events"] == {"all": 6, "VIEW": 4, "FAVORITE": 1, "mapped_without_product": 1, "unmapped": 1}

    order = indexed.orders["breakdowns"]
    assert order["observed_orders"] == 4
    assert Decimal(order["median_check"]["RUB"]["median"]) == Decimal("767.500000000000000000000000005")
    assert order["median_check"]["KZT"]["median"] == "0"
    assert order["payment_types"] == [["card", 4, 100.0], ["cash", 1, 25.0]]
    assert order["delivery"]["RUB"]["evaluated"] == 3
    assert order["basket"]["evaluated"] == 4
    assert order["currencies"]["RUB"]["valued_orders"] == 2
    assert Decimal(order["currencies"]["RUB"]["known_subtotal"]) == Decimal("1535.00000000000000000000000001")
    products = indexed.products["breakdowns"]
    assert products["viewers"]["000001"] == 2
    assert products["favorite_users"]["000001"] == 1
    assert products["namespaces"]
    client = indexed.customers["data"]
    assert client["customer_activity"]["active"] == 2
    assert client["customer_activity"]["repeat_buyers"] == 2
    assert client["customer_demography"]["mean_age"] == 36
    annual = next(row for row in client["buyer_breakdowns"]["annual"] if row["year"] == "2025")
    assert annual["money"]["RUB"]["known_orders"] == 2
    rfm = indexed.rfm["windows"]["LAST_12M"]["currencies"]["RUB"]
    assert "segment_business" in rfm

    model, tables = _model(indexed)
    assert {"unified.actions.source_quality", "unified.orders.financials", "unified.products.resolution",
            "unified.customers.activity", "unified.customers.item_order_distribution",
            "unified.customers.copurchase_base", "unified.customers.copurchase_pairs",
            "unified.customers.money.monthly.RUB", "unified.customers.money.quintiles.KZT",
            "unified.rfm.LAST_12M.RUB.segment_money", "unified.cohorts.repeat"} <= tables.keys()
    assert "Путь" in tables["unified.products.group.category"].headers
    charts = [block.data for page in model.pages for group in page.groups for block in group.blocks if block.kind == "chart"]
    tips = {chart.title: chart.tooltips[0][0] for chart in charts if chart.title.startswith("Магазины и каналы")}
    assert "1 535,00 RUB" in tips["Магазины и каналы — RUB"]
    assert "767,50 RUB" in tips["Магазины и каналы — RUB"]
    assert "0,00 KZT" in tips["Магазины и каналы — KZT"]


def test_status_independent_action_metrics_match_legacy(restored):
    root, catalog, _, _, _ = restored
    integration.rfm_tests._write_orders(restored, [])
    previous = legacy.calculate_statistics(raw_root=root, catalog_path=catalog, calculated_at=integration.rfm_tests._NOW)
    result = integration._report(restored)
    actual = result.actions["breakdowns"]
    for field in (
        "action_monthly_dynamics", "action_channel_statistics", "view_availability_distribution",
        "view_price_statistics", "view_user_activity_distribution", "favorite_user_activity_distribution",
    ):
        assert actual[field] == [list(row) for row in getattr(previous, field)]
    assert result.rfm["as_of_date"] == result.contexts["cohorts_history"]["as_of_date"]


def test_production_qprocess_exports_same_aggregate_dto(restored):
    app = qt.QApplication.instance() or qt.QApplication([])
    result = integration._report(restored)
    root = restored[0]
    folder = root / "report"
    folder.mkdir()
    protect_directory(folder)
    target = folder / "statistics.zip"
    worker = DetailExportProcess()
    outcomes = []
    worker.completed.connect(outcomes.append)
    worker.start(dict(destination=target), report_result=result)
    try:
        qt.wait_until(lambda: bool(outcomes), timeout=70000)
        assert outcomes == [dict(event="finished", files=2)], worker.diagnostic
        with ZipFile(target) as archive:
            assert archive.testzip() is None
            assert archive.namelist() == ["Статистика.xlsx", "Статистика.html"]
            workbook = load_workbook(BytesIO(archive.read("Статистика.xlsx")))
            try:
                assert workbook.sheetnames == ["Действия", "Заказы", "Товары", "Клиенты"]
                orders = workbook["Заказы"]
                assert any(row[0] == "Способ оплаты" for row in orders.values)
                subtotal = next(row[1] for row in orders.iter_rows() if row[0].value == "Сумма покупок")
                assert subtotal.data_type == "n" and subtotal.value == pytest.approx(1535)
                assert subtotal.comment is None
                note = "При неполных денежных данных показан подтвержденный подытог покупок."
                assert not any(row[0] == note for row in orders.values)
                html = archive.read("Статистика.html").decode()
                assert "Распределение покупателей по сумме покупок" in html
                assert note not in html
            finally:
                workbook.close()
    finally:
        if not outcomes:
            worker.shutdown()
        worker.deleteLater()
        app.processEvents()
