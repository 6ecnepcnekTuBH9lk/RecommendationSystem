import os
import json

import pandas as pd
from PyQt6.QtWidgets import (QComboBox, QListWidget)

from Application.settings.set_status import (set_status_processing, schedule_status_reset, show_custom_message,
                                             set_status_error, set_status_ok)


# -------------------------------------------КНОПКА "СОХРАНИТЬ НАСТРОЙКИ"-----------------------------------------------
def save_order_filter_settings(aboba):
    try:

        # 1) сначала забираем значения из UI
        store_city_map = _collect_store_city_map_from_ui(aboba)

        date_from_widget = getattr(aboba, "filter_date_from", None)
        date_to_widget = getattr(aboba, "filter_date_to", None)

        data = {
            "date_from": date_from_widget.text().strip() if date_from_widget is not None else "",
            "date_to": date_to_widget.text().strip() if date_to_widget is not None else "",
            "store_mode": aboba.store_mode.currentText() if hasattr(aboba, "store_mode") else "В группе",
            "kind_mode": aboba.kind_mode.currentText() if hasattr(aboba, "kind_mode") else "В группе",
            "stores_selected": get_selected_list_values(aboba.filter_store)
            if hasattr(aboba, "filter_store") else [],
            "seasons_selected": get_selected_list_values(aboba.filter_season)
            if hasattr(aboba, "filter_season") else [],
            "kinds_selected": get_selected_list_values(aboba.filter_kind)
            if hasattr(aboba, "filter_kind") else [],

            # Количество наиболее лояльных клиентов в выгрузке.
            # 0 означает выгрузку всех клиентов.
            "max_export_users": (
                int(aboba.max_export_users_input.value())
                if hasattr(aboba, "max_export_users_input")
                else 1000
            ),

            # Виды номенклатуры, разрешённые в итоговых рекомендациях
            "export_kinds_selected": (
                get_selected_list_values(aboba.export_kind_filter)
                if hasattr(aboba, "export_kind_filter")
                else []
            ),

            # 2) сохраняем карту склад -> город
            "store_city_map": store_city_map,
        }

        with open(order_filters_settings_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 3) держим в памяти актуальную карту
        aboba._store_city_map = store_city_map

        update_filter_summary(aboba)
        return True

    except Exception as e:
        set_status_error(aboba, "Не удалось сохранить настройки")
        schedule_status_reset(aboba, 5)
        show_custom_message(aboba,
                            title="Ошибка",
                            text=f"Не удалось сохранить настройки фильтров:\n{e}",
                            image_path="Картинки/Неудача.png",
                            )
        return False


# -------------------------------------------ОЧИСТИТЬ ФОРМУ-------------------------------------------------------------
def clear_layout(aboba, layout):
    if layout is None:
        return

    while layout.count():
        item = layout.takeAt(0)

        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
            continue

        child_layout = item.layout()
        if child_layout is not None:
            clear_layout(aboba, child_layout)


# -------------------------------------------ФОРМИРУЕМ ТЕКСТОВУЮ СТРОКУ С НАСТРОЙКАМИ-----------------------------------
def update_filter_summary(aboba):
    # если виджеты ещё не созданы — просто выходим
    targets = []
    for attr in ("filter_summary", "train_filter_summary"):
        w = getattr(aboba, attr, None)
        if w is not None:
            targets.append(w)

    if not targets:
        return

    def _masked_date_is_empty(qle) -> bool:
        if qle is None:
            return True
        t = qle.text()
        if t is None:
            return True
        t = t.replace(" ", "").replace(".", "")
        return t == ""

    # корректно читаем даты: если поле "пустое по маске" — считаем что даты нет
    date_from = ""
    if hasattr(aboba, "filter_date_from") and not _masked_date_is_empty(aboba.filter_date_from):
        date_from = aboba.filter_date_from.text().strip()

    date_to = ""
    if hasattr(aboba, "filter_date_to") and not _masked_date_is_empty(aboba.filter_date_to):
        date_to = aboba.filter_date_to.text().strip()

    def _fmt_many(vals: list[str]) -> str:
        vals = [str(v) for v in vals if str(v).strip()]
        if not vals:
            return "—"
        return ", ".join(vals)

    parts = []

    # Период
    if date_from or date_to:
        if date_from and date_to:
            parts.append(f"Период --> {date_from}–{date_to}")
        elif date_from:
            parts.append(f"Период --> с {date_from}")
        else:
            parts.append(f"Период --> по {date_to}")

    # Вид номенклатуры
    kind_mode = aboba.kind_mode.currentText() if hasattr(aboba, "kind_mode") else "В группе"
    kinds = get_selected_list_values(aboba.filter_kind) if hasattr(aboba, "filter_kind") else []
    if kinds:
        parts.append(f"Вид номенклатуры ({kind_mode}) --> {_fmt_many(kinds)}")

    # Магазин
    store_mode = aboba.store_mode.currentText() if hasattr(aboba, "store_mode") else "В группе"
    stores = get_selected_list_values(aboba.filter_store) if hasattr(aboba, "filter_store") else []
    if stores:
        parts.append(f"Магазин ({store_mode}) --> {_fmt_many(stores)}")

    # Актуальные сезоны (коллекции)
    seasons = get_selected_list_values(aboba.filter_season) if hasattr(aboba, "filter_season") else []
    if seasons:
        parts.append(f"Актуальные сезоны --> {_fmt_many(seasons)}")

    # Виды номенклатуры для итоговой выгрузки
    export_kinds = (
        get_selected_list_values(aboba.export_kind_filter)
        if hasattr(aboba, "export_kind_filter")
        else []
    )

    if export_kinds:
        parts.append(
            "Виды в рекомендациях (только выгрузка) --> "
            f"{_fmt_many(export_kinds)}"
        )

    text_out = "Отбор: " + ("; ".join(parts) if parts else "не установлен")

    for w in targets:
        w.setText(text_out)
        w.setCursorPosition(0)  # курсор в начало


# -------------------------------------------СОБИРАЕМ ТАБЛИЦУ ГОРОДОВ---------------------------------------------------
def _collect_store_city_map_from_ui(aboba) -> dict:
    table = getattr(aboba, "store_city_table", None)
    if table is None:
        return {}

    m = {}

    for r in range(table.rowCount()):
        store_item = table.item(r, 0)
        if not store_item:
            continue

        store = store_item.text().strip()
        cb = table.cellWidget(r, 1)

        if isinstance(cb, QComboBox):
            city = cb.currentText().strip()
            if city and "Сначала загрузите" not in city:
                m[store] = city

    return m


# -------------------------------------------ВОЗВРАЩАЕТ СПИСОК ВЫДЕЛЕННЫХ ЭЛЕМЕНТОВ-------------------------------------
def get_selected_list_values(lw: QListWidget) -> list[str]:
    return [it.text() for it in lw.selectedItems()] if lw is not None else []


# -------------------------------------------ПУТЬ К ДИРЕКТОРИИ НАСТРОЕК-------------------------------------------------
def order_filters_settings_path() -> str:
    cfg_dir = os.path.join(os.getcwd(), "Настройки")
    os.makedirs(cfg_dir, exist_ok=True)
    return os.path.join(cfg_dir, "filter_settings.json")


# -------------------------------------------ПУТИ К ДАТАСЕТАМ-------------------------------------------------------
def dataset_paths() -> dict:
    base = os.path.join(os.getcwd(), "ВходныеДанные")
    return {
        "orders": os.path.join(base, "Заказы.csv"),
        "views": os.path.join(base, "Просмотры.csv"),
        "favs": os.path.join(base, "Избранное.csv"),
    }


# -------------------------------------------ВЕКТОРНО СЧИТАЕМ ВОЗРАСТ---------------------------------------------------
def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce").dt.normalize()
    df["ДатаРождения"] = pd.to_datetime(df["ДатаРождения"], errors="coerce")

    age = df["Дата"].dt.year - df["ДатаРождения"].dt.year

    had_birthday = (
        (df["Дата"].dt.month > df["ДатаРождения"].dt.month) |
        (
            (df["Дата"].dt.month == df["ДатаРождения"].dt.month) &
            (df["Дата"].dt.day >= df["ДатаРождения"].dt.day)
        )
    )

    df["Возраст"] = (age - (~had_birthday).astype(int)).astype("Int64")
    df.loc[df["Дата"].isna() | df["ДатаРождения"].isna(), "Возраст"] = pd.NA

    df["ВозрастнаяГруппа"] = df["Возраст"].apply(get_age_group)

    return df


# -------------------------------------------ВОЗРАСТНАЯ ГРУППА----------------------------------------------------------
def get_age_group(age: int) -> str:
    if pd.isnull(age):
        return "Не указан"
    if age < 14:
        return "до 14"
    elif 14 <= age <= 25:
        return "14-25"
    elif 26 <= age <= 35:
        return "26-35"
    elif 36 <= age <= 45:
        return "36-45"
    elif 46 <= age <= 55:
        return "46-55"
    elif 56 <= age <= 65:
        return "56-65"
    else:
        return "65+"


# -------------------------------------------ОБРАБОТКА ИДЕНТИФИКАТОРОВ--------------------------------------------------
def clean_id_series(s: pd.Series) -> pd.Series:
    out = (
        s.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    out = out.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "none": pd.NA,
        "NULL": pd.NA,
        "null": pd.NA,
        "<NA>": pd.NA,
    })

    return out
