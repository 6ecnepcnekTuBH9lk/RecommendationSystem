import numpy as np
import pandas as pd
import pytest

from Application.files import files_processing
from Application.files.files_processing import (
    process_categories_file,
    process_coordinates_file,
    process_favorites_file,
    process_nomenclature_file,
    process_orders_file,
    process_views_file,
)


NOMENCLATURE_COLUMNS = [
    "КодНоменклатуры",
    "Номенклатура",
    "НазваниеНаСайте",
    "ВидНоменклатуры",
    "ВидАссортимента",
    "Марка",
    "Коллекция",
    "СезонНоски",
    "ПолНоменклатуры",
    "ГруппаСоставов",
    "КатегорияНаСайте",
    "СтилеваяГруппа",
    "ТитульнаяФотография",
    "Остаток",
]


def _catalog_row(code="123456", stock="120"):
    return {
        "КодНоменклатуры": code,
        "Номенклатура": "Синтетический товар",
        "НазваниеНаСайте": "Тестовая рубашка",
        "ВидНоменклатуры": "Рубашки",
        "ВидАссортимента": "Основной",
        "Марка": "Тестовая марка",
        "Коллекция": "Весна-Лето 2026",
        "СезонНоски": "Лето",
        "ПолНоменклатуры": "Мужской",
        "ГруппаСоставов": "Хлопок",
        "КатегорияНаСайте": "10.0",
        "СтилеваяГруппа": "Классика",
        "ТитульнаяФотография": "https://example.invalid/item.jpg",
        "Остаток": stock,
    }


def _write_reference_files(tmp_path):
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()
    pd.DataFrame([_catalog_row()]).to_csv(
        input_dir / "Номенклатура.csv", sep="|", index=False
    )
    pd.DataFrame(
        [{"КодКатегории": "77", "НазваниеКатегории": "Тестовая категория"}]
    ).to_csv(input_dir / "КатегорииСайта.csv", sep="|", index=False)


def _raw_customer_fields(prefix):
    return {
        f"{prefix}CustomerLastActivatedCardIdsNumber": "7001.0",
        f"{prefix}CustomerIdsMindboxId": "9001.0",
        f"{prefix}CustomerFirstName": "Иван",
        f"{prefix}CustomerLastName": "Тестов",
        f"{prefix}CustomerMiddleName": "Синтетический",
        f"{prefix}CustomerBirthDate": "1990-06-16",
        f"{prefix}CustomerSex": "male",
        f"{prefix}CustomerEmail": "synthetic@example.invalid",
        f"{prefix}CustomerMobilePhone": np.nan,
        f"{prefix}CustomerPendingMobilePhone": "79990000000.0",
        f"{prefix}CustomerCustomFieldsMostViewedCategory": "10.0",
        f"{prefix}CustomerCustomFieldsMostViewedRootCategory": "20,0",
        f"{prefix}CustomerCustomFieldsMostViewedSubsidiaryCategory": np.nan,
    }


def _raw_interaction_rows(source_name, dates):
    if source_name == "orders":
        base = {
            "OrderIdsMindboxId": "5001.0",
            "OrderLineStatusIdsExternalId": "CP",
            "OrderFirstActionChannelName": "kanzler-style.ru",
            "OrderLineProductIdsOffline1C": "123456.0",
            "OrderLineProductIdsKanzlerKz": np.nan,
            "OrderLineQuantity": "2",
            "OrderLineBasePricePerItem": "100",
            "OrderLinePriceOfLine": "150",
            **_raw_customer_fields("Order"),
        }
        date_column = "OrderFirstActionDateTimeUtc"
        mindbox_column = "OrderCustomerIdsMindboxId"
    else:
        base = {
            "CustomerActionProductsIdsOffline1C": "123456.0",
            "CustomerActionProductsIdsKanzlerKz": np.nan,
            **_raw_customer_fields("CustomerAction"),
        }
        if source_name == "views":
            base["CustomerActionProductCategoriesIdsOffline1C"] = np.nan
        else:
            base["CustomerActionActionTemplateIdsSystemName"] = "AddToWishlist"
        date_column = "CustomerActionDateTimeUtc"
        mindbox_column = "CustomerActionCustomerIdsMindboxId"

    rows = []
    for index, date_value in enumerate(dates, start=1):
        row = dict(base)
        row[date_column] = date_value
        row[mindbox_column] = f"900{index}.0"
        if source_name == "orders":
            row["OrderIdsMindboxId"] = f"500{index}.0"
        rows.append(row)
    return rows


@pytest.mark.parametrize("source_name", ["orders", "views", "favorites"])
def test_interaction_processor_reports_malformed_date_without_dropping_rows(
    tmp_path,
    monkeypatch,
    source_name,
):
    monkeypatch.chdir(tmp_path)
    _write_reference_files(tmp_path)
    statuses = []
    monkeypatch.setattr(
        files_processing,
        "set_status_ok",
        lambda aboba, text: statuses.append(text),
    )
    monkeypatch.setattr(
        files_processing,
        "schedule_status_reset",
        lambda *args: None,
    )
    processor = {
        "orders": files_processing.process_orders_file,
        "views": files_processing.process_views_file,
        "favorites": files_processing.process_favorites_file,
    }[source_name]
    source = pd.DataFrame(
        _raw_interaction_rows(
            source_name,
            ["2024-06-15 10:00:00", "not-a-date", ""],
        )
    )

    result = processor(object(), source)

    assert len(result) == 3
    assert set(result["MindboxID"]) == {"9001", "9002", "9003"}
    assert set(result["КодНоменклатуры"]) == {"123456"}
    assert result.loc[result["MindboxID"] == "9001", "Дата"].notna().all()
    assert result.loc[result["MindboxID"].isin(["9002", "9003"]), "Дата"].isna().all()
    assert statuses == [
        "Обработка завершена. Обнаружено некорректных значений даты: 1"
    ]


@pytest.mark.parametrize("source_name", ["orders", "views", "favorites"])
def test_interaction_processor_does_not_warn_for_valid_or_empty_dates(
    tmp_path,
    monkeypatch,
    source_name,
):
    monkeypatch.chdir(tmp_path)
    _write_reference_files(tmp_path)
    statuses = []
    monkeypatch.setattr(
        files_processing,
        "set_status_ok",
        lambda aboba, text: statuses.append(text),
    )
    monkeypatch.setattr(
        files_processing,
        "schedule_status_reset",
        lambda *args: None,
    )
    processor = {
        "orders": files_processing.process_orders_file,
        "views": files_processing.process_views_file,
        "favorites": files_processing.process_favorites_file,
    }[source_name]
    source = pd.DataFrame(
        _raw_interaction_rows(
            source_name,
            ["2024-06-15 10:00:00", "", "   ", np.nan, None],
        )
    )

    result = processor(object(), source)

    assert len(result) == 5
    assert result["Дата"].notna().sum() == 1
    assert statuses == ["Обработка завершена"]


def test_process_orders_normalizes_filters_and_enriches_catalog(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_reference_files(tmp_path)

    valid = {
        "OrderIdsMindboxId": "5001.0",
        "OrderLineStatusIdsExternalId": " CP ",
        "OrderFirstActionDateTimeUtc": "2024-06-15 12:30:00",
        "OrderFirstActionChannelName": "kanzler-style.ru",
        "OrderLineProductIdsOffline1C": "123456.0",
        "OrderLineProductIdsKanzlerKz": np.nan,
        "OrderLineQuantity": "2",
        "OrderLineBasePricePerItem": "100",
        "OrderLinePriceOfLine": "150",
        **_raw_customer_fields("Order"),
    }
    cancelled = dict(valid)
    cancelled["OrderIdsMindboxId"] = "5002.0"
    cancelled["OrderLineStatusIdsExternalId"] = "cancelled"

    result = process_orders_file(object(), pd.DataFrame([valid, cancelled]))

    assert len(result) == 1
    row = result.iloc[0]
    assert row["НомерЗаказа"] == "5001"
    assert row["КодНоменклатуры"] == "123456"
    assert row["Магазин"] == "ИНТЕРНЕТ-МАГАЗИН"
    assert row["Валюта"] == "RUB"
    assert row["НачальнаяСтоимость"] == 200
    assert row["ПроцентСкидки"] == 25
    assert row["Телефон"] == "79990000000"
    assert row["ФИО"] == "Тестов Иван Синтетический"
    assert row["ПолКлиента"] == "Мужской"
    assert row["Возраст"] == 33
    assert row["ВозрастнаяГруппа"] == "26-35"
    assert row["НазваниеНаСайте"] == "Тестовая рубашка"
    assert row["ЛюбимаяКатегория"] == "10_200"


def test_process_views_keeps_item_and_category_interactions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_reference_files(tmp_path)

    base = {
        "CustomerActionDateTimeUtc": "2024-06-15 10:00:00",
        "CustomerActionProductsIdsOffline1C": "123456.0",
        "CustomerActionProductsIdsKanzlerKz": np.nan,
        "CustomerActionProductCategoriesIdsOffline1C": np.nan,
        **_raw_customer_fields("CustomerAction"),
    }
    category = dict(base)
    category["CustomerActionProductsIdsOffline1C"] = np.nan
    category["CustomerActionProductCategoriesIdsOffline1C"] = "77.0"

    result = process_views_file(object(), pd.DataFrame([base, category]))

    assert result["КодНоменклатуры"].tolist() == ["123456", "77"]
    assert result["ТипТовара"].tolist() == ["Номенклатура", "Категория"]
    assert result.iloc[0]["НазваниеНаСайте"] == "Тестовая рубашка"
    assert result.iloc[1]["НазваниеКатегории"] == "Тестовая категория"
    assert result["MindboxID"].tolist() == ["9001", "9001"]


def test_process_favorites_excludes_currently_ignored_operation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_reference_files(tmp_path)

    kept = {
        "CustomerActionDateTimeUtc": "2024-06-15 10:00:00",
        "CustomerActionProductsIdsOffline1C": "123456.0",
        "CustomerActionProductsIdsKanzlerKz": np.nan,
        "CustomerActionActionTemplateIdsSystemName": "AddToWishlist",
        **_raw_customer_fields("CustomerAction"),
    }
    ignored = dict(kept)
    ignored["CustomerActionActionTemplateIdsSystemName"] = (
        "DobavlenieProduktaVSpisokVOperaciiUstanovka"
    )

    result = process_favorites_file(object(), pd.DataFrame([kept, ignored]))

    assert len(result) == 1
    assert result.iloc[0]["КодНоменклатуры"] == "123456"
    assert result.iloc[0]["MindboxID"] == "9001"
    assert result.iloc[0]["НазваниеНаСайте"] == "Тестовая рубашка"


def test_process_nomenclature_drops_blank_codes_and_rounds_stock():
    valid = _catalog_row(stock="100,6")
    blank = _catalog_row(code="   ", stock="50")

    result = process_nomenclature_file(object(), pd.DataFrame([valid, blank]))

    assert result.columns.tolist() == NOMENCLATURE_COLUMNS
    assert len(result) == 1
    assert result.iloc[0]["КодНоменклатуры"] == "123456"
    assert result.iloc[0]["КатегорияНаСайте"] == "10"
    assert result.iloc[0]["Остаток"] == 101


def test_process_categories_normalizes_identifiers():
    source = pd.DataFrame(
        [
            {
                "КодКатегории": "77.0",
                "НазваниеКатегории": "Тестовая категория",
                "КодРодительскойКатегории": "7.0",
            }
        ]
    )

    result = process_categories_file(object(), source)

    assert result.to_dict(orient="records") == [
        {
            "КодКатегории": "77",
            "НазваниеКатегории": "Тестовая категория",
            "КодРодительскойКатегории": "7",
        }
    ]


def test_process_coordinates_drops_invalid_rows_and_exact_duplicates():
    source = pd.DataFrame(
        [
            {"Город": " Москва ", "Широта": "55,75", "Долгота": "37,61"},
            {"Город": "Москва", "Широта": "55,75", "Долгота": "37,61"},
            {"Город": "Без координат", "Широта": "нет", "Долгота": "37,61"},
        ]
    )

    result = process_coordinates_file(object(), source)

    assert result.to_dict(orient="records") == [
        {"Город": "Москва", "Широта": 55.75, "Долгота": 37.61}
    ]
