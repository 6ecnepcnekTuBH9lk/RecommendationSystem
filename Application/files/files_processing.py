import os
import numpy as np
import pandas as pd
import requests
import time
import random
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException

from Application.settings.set_status import (schedule_status_reset, show_custom_message,
                                             set_status_error, set_status_ok, set_status_processing)
from Application.settings.settings_and_filter import (add_age_features, get_age_group, clean_id_series)


# -------------------------------------------ОБРАБОТКА HTTP-ЗАПРОСОВ----------------------------------------------------
_HTTP = requests.Session()
_HTTP.headers.update({
    "User-Agent": "KanzlerRecSys/1.0",
    "Accept": "application/json",
    "Connection": "close",  # иногда помогает от ConnectionReset на keep-alive
})


def _get_json_with_retry(
    url: str,
    params: dict,
    attempts: int = 3,
    timeout: tuple[float, float] = (8.0, 15.0),  # (connect, read)
    backoff_base: float = 1.5,
) -> dict | None:

    for i in range(1, attempts + 1):
        try:
            resp = _HTTP.get(url, params=params, timeout=timeout)

            # ретраим временные статусы
            if resp.status_code == 429 or resp.status_code >= 500:
                raise HTTPError(f"HTTP {resp.status_code}", response=resp)

            resp.raise_for_status()
            return resp.json()

        except (Timeout, ConnectionError):
            pass

        except HTTPError as e:
            code = e.response.status_code if e.response is not None else None

            # если это "обычная" 4xx — смысла ретраить нет
            if code is not None and 400 <= code < 500 and code != 429:
                break

        except (ValueError, RequestException):
            # ValueError — на случай json decode error
            pass

        if i < attempts:
            # экспоненциальная пауза + небольшой jitter
            sleep_s = (backoff_base ** (i - 1)) + random.random() * 0.2
            time.sleep(sleep_s)

    return None


def _parse_interaction_date(values: pd.Series) -> tuple[pd.Series, int]:
    raw_text = values.astype("string")
    raw_nonempty = raw_text.notna() & raw_text.str.strip().ne("")
    parsed = pd.to_datetime(values, errors="coerce").dt.normalize()
    malformed_count = int((raw_nonempty & parsed.isna()).sum())
    return parsed, malformed_count


def _set_processing_complete_status(aboba, malformed_date_count: int) -> None:
    message = "Обработка завершена"
    if malformed_date_count:
        message += (
            ". Обнаружено некорректных значений даты: "
            f"{malformed_date_count}"
        )
    set_status_ok(aboba, message)
    schedule_status_reset(aboba, 5)


# -------------------------------------------ОБРАБОТКА ЗАКАЗОВ------------------------------------------------------
def process_orders_file(aboba, df):
    # Список нужных колонок и новые имена
    columns_map = {
        "OrderIdsMindboxId": "НомерЗаказа",
        "OrderLineStatusIdsExternalId": "СтатусЗаказа",
        "OrderFirstActionDateTimeUtc": "Дата",
        "OrderFirstActionChannelName": "Магазин",
        "OrderLineProductIdsOffline1C": "КодНоменклатурыРФ",
        "OrderLineProductIdsKanzlerKz": "КодНоменклатурыКЗ",
        "OrderLineQuantity": "Количество",
        "OrderLineBasePricePerItem": "НачальнаяЦена",
        "OrderLinePriceOfLine": "КонечнаяСтоимость",
        "OrderCustomerLastActivatedCardIdsNumber": "ДисконтнаяКарта",
        "OrderCustomerIdsMindboxId": "MindboxID",
        "OrderCustomerFirstName": "Имя",
        "OrderCustomerLastName": "Фамилия",
        "OrderCustomerMiddleName": "Отчество",
        "OrderCustomerBirthDate": "ДатаРождения",
        "OrderCustomerSex": "ПолКлиента",
        "OrderCustomerEmail": "Почта",
        "OrderCustomerMobilePhone": "ТелефонОсновной",
        "OrderCustomerPendingMobilePhone": "ЗапаснойТелефон",
        "OrderCustomerCustomFieldsMostViewedCategory": "СамаяПросматриваемаяКатегория",
        "OrderCustomerCustomFieldsMostViewedRootCategory": "СамаяПросматриваемаяРодительскаяКатегория",
        "OrderCustomerCustomFieldsMostViewedSubsidiaryCategory": "СамаяПросматриваемаяДочерняяКатегория"
    }

    # Проверка наличия всех колонок
    missing = [col for col in columns_map.keys() if col not in df.columns]

    if missing:
        show_custom_message(aboba, title="Ошибка",
                            text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
                            image_path="Картинки/Неудача.png")
        set_status_error(aboba, "Отсутствуют необходимые колонки")
        schedule_status_reset(aboba, 5)

        return None

    # Оставляем только нужные колонки
    df = df[list(columns_map.keys())]

    # Переименовываем
    df = df.rename(columns=columns_map)

    # ---------- Обработка данных ----------
    # --- Фильтрация по статусам заказа ---
    allowed_statuses = {"CP", "delivering", "F"}

    # приводим к строке, убираем пробелы по краям (на случай " CP " и т.п.)
    df["СтатусЗаказа"] = df["СтатусЗаказа"].astype(str).str.strip()

    # оставляем только нужные статусы
    df = df[df["СтатусЗаказа"].isin(allowed_statuses)]

    # Удаляем строки, где не заполнено НИ одно из полей "КодНоменклатурыРФ" или "КодНоменклатурыКЗ"
    df = df.dropna(subset=["КодНоменклатурыРФ", "КодНоменклатурыКЗ"], how="all")

    # Объединяем номера телефонов
    df["Телефон"] = (df["ТелефонОсновной"].combine_first(df["ЗапаснойТелефон"]).apply(
        lambda x: str(x)[:-2] if pd.notnull(x) and str(x).endswith(".0") else str(x) if pd.notnull(x) else np.nan))
    df = df.drop(columns=["ТелефонОсновной", "ЗапаснойТелефон"])

    # Преобразуем цены и количество в числовой формат
    df["НачальнаяЦена"] = pd.to_numeric(df["НачальнаяЦена"], errors='coerce').fillna(0).astype(int)
    df["КонечнаяСтоимость"] = pd.to_numeric(df["КонечнаяСтоимость"], errors='coerce').fillna(0).astype(int)
    df["Количество"] = pd.to_numeric(df["Количество"], errors='coerce').fillna(0).astype(int)

    df["НачальнаяСтоимость"] = df["НачальнаяЦена"] * df["Количество"]

    df["ПроцентСкидки"] = ((df["НачальнаяСтоимость"] - df["КонечнаяСтоимость"]) / df["НачальнаяСтоимость"] * 100)

    df["ПроцентСкидки"] = (
        df["ПроцентСкидки"]
        .replace([float('inf'), -float('inf')], 0)
        .fillna(0)
        .round()
        .astype(int)
    )

    # Определяем валюту по коду номенклатуры
    df["Валюта"] = np.where(df["КодНоменклатурыРФ"].notna(), "RUB",
                            np.where(df["КодНоменклатурыКЗ"].notna(), "KZT", None)).astype(object)

    # Объединяем коды номенклатуры
    df["КодНоменклатуры"] = df["КодНоменклатурыРФ"].combine_first(df["КодНоменклатурыКЗ"]).astype(str).str[:6]
    df = df.drop(columns=["КодНоменклатурыРФ", "КодНоменклатурыКЗ"])

    # В колонке Дата оставляем только дату (убираем время)
    df["Дата"], malformed_date_count = _parse_interaction_date(df["Дата"])

    df["ДатаРождения"] = pd.to_datetime(df["ДатаРождения"], errors='coerce')

    # Рассчитываем возраст на момент заказа
    df = add_age_features(df)

    # Добавляем колонку с возрастной группой
    df["ВозрастнаяГруппа"] = df["Возраст"].apply(get_age_group)

    # В колонке Магазин заменяем значение
    df["Магазин"] = df["Магазин"].replace({"kanzler-style.ru": "ИНТЕРНЕТ-МАГАЗИН"})
    df["Магазин"] = df["Магазин"].replace({"kanzler-style.kz": "ИНТЕРНЕТ-МАГАЗИН КАЗАХСТАН"})

    # Объединяем Имя, Фамилия, Отчество в ФИО
    df["ФИО"] = df[["Фамилия", "Имя", "Отчество"]].fillna("").agg(" ".join, axis=1).str.strip()
    df = df.drop(columns=["Имя", "Фамилия", "Отчество"])

    # В колонке Пол заменяем значения
    df["ПолКлиента"] = df["ПолКлиента"].replace({"male": "Мужской", "female": "Женский"})

    df["ДисконтнаяКарта"] = clean_id_series(df["ДисконтнаяКарта"])
    df["НомерЗаказа"] = clean_id_series(df["НомерЗаказа"])
    df["MindboxID"] = clean_id_series(df["MindboxID"])

    # Очистка категорий от точек и запятых
    for col in [
        "СамаяПросматриваемаяКатегория",
        "СамаяПросматриваемаяРодительскаяКатегория",
        "СамаяПросматриваемаяДочерняяКатегория"
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)  # убираем только .0 в конце
            .str.replace(r"[.,]", "", regex=True)  # убираем лишние знаки, если вдруг есть
            .str.strip()
        )

    # Объединяем любимые категории в одну колонку через "_"
    df["ЛюбимаяКатегория"] = (df[["СамаяПросматриваемаяКатегория",
                                  "СамаяПросматриваемаяРодительскаяКатегория",
                                  "СамаяПросматриваемаяДочерняяКатегория"]].apply(
        lambda row: "_".join([str(x) for x in row if pd.notnull(x) and x != "nan"]),
        axis=1))

    # Удалим отдельные колонки категорий
    df = df.drop(columns=["СамаяПросматриваемаяКатегория",
                          "СамаяПросматриваемаяРодительскаяКатегория",
                          "СамаяПросматриваемаяДочерняяКатегория"])

    # --- Подтягиваем данные из Номенклатуры.csv ---
    nom_path = "ВходныеДанные/Номенклатура.csv"

    if not os.path.isfile(nom_path):
        show_custom_message(aboba,
                            title="Ошибка",
                            text="Для корректной загрузки необходимо сначала загрузить файл Номенклатура.csv",
                            image_path="Картинки/Неудача.png"
                            )
        set_status_error(aboba, "Отсутствует файл Номенклатура.csv")
        schedule_status_reset(aboba, 5)
        return None

    # Читаем файл номенклатуры
    nom_df = pd.read_csv(nom_path, sep="|", dtype=str)

    # Проверяем обязательные колонки
    required_nom_cols = ["КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте", "Коллекция"]
    missing_nom = [col for col in required_nom_cols if col not in nom_df.columns]

    if missing_nom:
        show_custom_message(aboba,
                            title="Ошибка",
                            text="В файле Номенклатура.csv отсутствуют колонки:\n" + "\n".join(missing_nom),
                            image_path="Картинки/Неудача.png"
                            )
        set_status_error(aboba, "Отсутствуют необходимые колонки")
        schedule_status_reset(aboba, 5)
        return None

    # Оставляем только нужные колонки
    nom_df = nom_df[required_nom_cols]

    # Объединяем основной df с номенклатурой
    df = df.merge(nom_df, on="КодНоменклатуры", how="left")

    # Заменяем все виды пустых значений на np.nan
    df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "nan"], np.nan)

    # Упорядочиваем колонки
    column_order = [
        "Дата", "НомерЗаказа", "Магазин", "КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте",
        "Коллекция", "Количество", "НачальнаяЦена", "НачальнаяСтоимость", "КонечнаяСтоимость", "ПроцентСкидки",
        "Валюта", "ДисконтнаяКарта", "MindboxID", "ФИО", "ДатаРождения", "Возраст", "ВозрастнаяГруппа", "ПолКлиента",
        "Почта", "Телефон", "ЛюбимаяКатегория"
    ]
    df = df[column_order]

    df = df.sort_values(by="Дата", ascending=True)

    _set_processing_complete_status(aboba, malformed_date_count)

    # возвращаем обработанный DataFrame
    return df


# -------------------------------------------ОБРАБОТКА ПРОСМОТРОВ-------------------------------------------------------
def process_views_file(aboba, df):
    # Список нужных колонок и новые имена
    columns_map = {
        "CustomerActionDateTimeUtc": "Дата",
        "CustomerActionProductsIdsOffline1C": "КодНоменклатурыРФ",
        "CustomerActionProductsIdsKanzlerKz": "КодНоменклатурыКЗ",
        "CustomerActionProductCategoriesIdsOffline1C": "КодКатегории",
        "CustomerActionCustomerLastActivatedCardIdsNumber": "ДисконтнаяКарта",
        "CustomerActionCustomerIdsMindboxId": "MindboxID",
        "CustomerActionCustomerFirstName": "Имя",
        "CustomerActionCustomerLastName": "Фамилия",
        "CustomerActionCustomerMiddleName": "Отчество",
        "CustomerActionCustomerBirthDate": "ДатаРождения",
        "CustomerActionCustomerSex": "ПолКлиента",
        "CustomerActionCustomerEmail": "Почта",
        "CustomerActionCustomerMobilePhone": "ТелефонОсновной",
        "CustomerActionCustomerPendingMobilePhone": "ЗапаснойТелефон",
        "CustomerActionCustomerCustomFieldsMostViewedCategory": "СамаяПросматриваемаяКатегория",
        "CustomerActionCustomerCustomFieldsMostViewedRootCategory": "СамаяПросматриваемаяРодительскаяКатегория",
        "CustomerActionCustomerCustomFieldsMostViewedSubsidiaryCategory": "СамаяПросматриваемаяДочерняяКатегория"
    }

    # Проверка наличия всех колонок
    missing = [col for col in columns_map.keys() if col not in df.columns]

    if missing:
        show_custom_message(aboba, title="Ошибка",
                            text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
                            image_path="Картинки/Неудача.png")
        set_status_error(aboba, "Отсутствуют необходимые колонки")
        schedule_status_reset(aboba, 5)

        return None

    # Оставляем только нужные колонки
    df = df[list(columns_map.keys())]

    # Переименовываем
    df = df.rename(columns=columns_map)

    # ---------- Обработка данных ----------

    # Удаляем строки, где не заполнено НИ одно из полей
    # "КодНоменклатурыРФ", "КодНоменклатурыКЗ" или "ПросмотреннаяКатегория"
    df = df.dropna(subset=["КодНоменклатурыРФ", "КодНоменклатурыКЗ", "КодКатегории"], how="all")

    # Объединяем коды номенклатуры
    df["КодНоменклатурыПервый"] = df["КодНоменклатурыРФ"].combine_first(df["КодНоменклатурыКЗ"]).astype(str).str[:6]
    df = df.drop(columns=["КодНоменклатурыРФ", "КодНоменклатурыКЗ"])

    # Объединяем номера телефонов
    df["Телефон"] = (df["ТелефонОсновной"].combine_first(df["ЗапаснойТелефон"])
    .apply(
        lambda x: str(x)[:-2] if pd.notnull(x) and str(x).endswith(".0") else str(x) if pd.notnull(x) else np.nan))
    df = df.drop(columns=["ТелефонОсновной", "ЗапаснойТелефон"])

    # В колонке Дата оставляем только дату (убираем время)
    df["Дата"], malformed_date_count = _parse_interaction_date(df["Дата"])

    df["ДатаРождения"] = pd.to_datetime(df["ДатаРождения"], errors='coerce')

    # Рассчитываем возраст на момент заказа
    df = add_age_features(df)

    # Добавляем колонку с возрастной группой
    df["ВозрастнаяГруппа"] = df["Возраст"].apply(get_age_group)

    # Объединяем Имя, Фамилия, Отчество в ФИО
    df["ФИО"] = df[["Фамилия", "Имя", "Отчество"]].fillna("").agg(" ".join, axis=1).str.strip()
    df = df.drop(columns=["Имя", "Фамилия", "Отчество"])

    # В колонке Пол заменяем значения
    df["ПолКлиента"] = df["ПолКлиента"].replace({"male": "Мужской", "female": "Женский"})

    df["ДисконтнаяКарта"] = clean_id_series(df["ДисконтнаяКарта"])
    df["MindboxID"] = clean_id_series(df["MindboxID"])

    # Очистка категорий от точек и запятых
    for col in [
        "КодКатегории",
        "СамаяПросматриваемаяКатегория",
        "СамаяПросматриваемаяРодительскаяКатегория",
        "СамаяПросматриваемаяДочерняяКатегория"
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)  # убираем только .0 в конце
            .str.replace(r"[.,]", "", regex=True)  # убираем лишние знаки, если вдруг есть
            .str.strip()
        )

    # Объединяем любимые категории в одну колонку через "_"
    df["ЛюбимаяКатегория"] = df[
        ["СамаяПросматриваемаяКатегория", "СамаяПросматриваемаяРодительскаяКатегория",
         "СамаяПросматриваемаяДочерняяКатегория"]
    ].apply(lambda row: "_".join([str(x) for x in row if pd.notnull(x) and x != "nan"]), axis=1)

    # Удалим отдельные колонки категорий
    df = df.drop(columns=["СамаяПросматриваемаяКатегория",
                          "СамаяПросматриваемаяРодительскаяКатегория",
                          "СамаяПросматриваемаяДочерняяКатегория"])

    # Заменяем все виды пустых значений на np.nan
    df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "nan"], np.nan)

    # Определяем тип значения до объединения
    df["ТипТовара"] = df["КодКатегории"].notna().map({
        True: "Категория",
        False: "Номенклатура"
    })

    # Объединяем номенклатуру и категории
    df["КодНоменклатуры"] = df["КодКатегории"].combine_first(
        df["КодНоменклатурыПервый"]).astype(str)
    df = df.drop(columns=["КодКатегории", "КодНоменклатурыПервый"])

    # --- Подтягиваем данные из Номенклатуры.csv ---
    nom_path = "ВходныеДанные/Номенклатура.csv"

    if not os.path.isfile(nom_path):
        show_custom_message(aboba,
                            title="Ошибка",
                            text="Для корректной загрузки необходимо сначала загрузить файл Номенклатура.csv",
                            image_path="Картинки/Неудача.png"
                            )
        set_status_error(aboba, "Отсутствует файл Номенклатура.csv")
        schedule_status_reset(aboba, 5)
        return None

    # Читаем файл номенклатуры
    nom_df = pd.read_csv(nom_path, sep="|", dtype=str)

    # Проверяем обязательные колонки
    required_nom_cols = ["КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте", "Коллекция"]
    missing_nom = [col for col in required_nom_cols if col not in nom_df.columns]

    if missing_nom:
        show_custom_message(aboba,
                            title="Ошибка",
                            text="В файле Номенклатура.csv отсутствуют колонки:\n" + "\n".join(missing_nom),
                            image_path="Картинки/Неудача.png"
                            )
        set_status_error(aboba, "Отсутствуют необходимые колонки")
        schedule_status_reset(aboba, 5)

        return None

    # Оставляем только нужные колонки
    nom_df = nom_df[required_nom_cols]

    # Объединяем основной df с номенклатурой
    df = df.merge(nom_df, on="КодНоменклатуры", how="left")

    # --- Подтягиваем данные из КатегорииСайта.csv ---
    cat_path = "ВходныеДанные/КатегорииСайта.csv"

    if not os.path.isfile(cat_path):
        show_custom_message(aboba,
                            title="Ошибка",
                            text="Для корректной загрузки необходимо сначала загрузить файл КатегорииСайта.csv",
                            image_path="Картинки/Неудача.png"
                            )
        set_status_error(aboba, "Отсутствует файл КатегорииСайта.csv")
        schedule_status_reset(aboba, 5)
        return None

    # Читаем файл категорий
    cat_df = pd.read_csv(cat_path, sep="|", dtype=str)

    # Проверяем обязательные колонки
    required_cat_cols = ["КодКатегории", "НазваниеКатегории"]
    missing_cat = [col for col in required_cat_cols if col not in cat_df.columns]

    if missing_cat:
        show_custom_message(aboba,
                            title="Ошибка",
                            text="В файле КатегорииСайта.csv отсутствуют колонки:\n" + "\n".join(missing_cat),
                            image_path="Картинки/Неудача.png"
                            )
        set_status_error(aboba, "Отсутствуют необходимые колонки")
        schedule_status_reset(aboba, 5)

        return None

    # Оставляем только нужные колонки
    cat_df = cat_df[required_cat_cols]

    # Объединяем основной df с номенклатурой
    df = df.merge(cat_df, left_on="КодНоменклатуры", right_on="КодКатегории", how="left")

    # Упорядочиваем колонки
    column_order = [
        "Дата", "КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте", "Коллекция",
        "НазваниеКатегории", "ТипТовара", "ДисконтнаяКарта", "MindboxID", "ФИО", "ДатаРождения",
        "Возраст", "ВозрастнаяГруппа", "ПолКлиента", "Почта", "Телефон", "ЛюбимаяКатегория"
    ]
    df = df[column_order]

    df = df.sort_values(by="Дата", ascending=True)

    _set_processing_complete_status(aboba, malformed_date_count)

    # возвращаем обработанный DataFrame
    return df


# -------------------------------------------ОБРАБОТКА ИЗБРАННОГО-------------------------------------------------------
def process_favorites_file(aboba, df):
    # Список нужных колонок и новые имена
    columns_map = {
        "CustomerActionDateTimeUtc": "Дата",
        "CustomerActionProductsIdsOffline1C": "КодНоменклатурыРФ",
        "CustomerActionProductsIdsKanzlerKz": "КодНоменклатурыКЗ",
        "CustomerActionCustomerLastActivatedCardIdsNumber": "ДисконтнаяКарта",
        "CustomerActionCustomerIdsMindboxId": "MindboxID",
        "CustomerActionCustomerFirstName": "Имя",
        "CustomerActionCustomerLastName": "Фамилия",
        "CustomerActionCustomerMiddleName": "Отчество",
        "CustomerActionCustomerBirthDate": "ДатаРождения",
        "CustomerActionCustomerSex": "ПолКлиента",
        "CustomerActionCustomerEmail": "Почта",
        "CustomerActionCustomerMobilePhone": "ТелефонОсновной",
        "CustomerActionCustomerPendingMobilePhone": "ЗапаснойТелефон",
        "CustomerActionCustomerCustomFieldsMostViewedCategory": "СамаяПросматриваемаяКатегория",
        "CustomerActionCustomerCustomFieldsMostViewedRootCategory": "СамаяПросматриваемаяРодительскаяКатегория",
        "CustomerActionCustomerCustomFieldsMostViewedSubsidiaryCategory": "СамаяПросматриваемаяДочерняяКатегория",
        "CustomerActionActionTemplateIdsSystemName": "ТипОперации"
    }

    # Проверка наличия всех колонок
    missing = [col for col in columns_map.keys() if col not in df.columns]

    if missing:
        show_custom_message(aboba, title="Ошибка",
                            text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
                            image_path="Картинки/Неудача.png")
        set_status_error(aboba, "Отсутствуют необходимые колонки")
        schedule_status_reset(aboba, 5)

        return None

    # Оставляем только нужные колонки
    df = df[list(columns_map.keys())]

    # Переименовываем
    df = df.rename(columns=columns_map)

    # ---------- Обработка данных ----------

    # Удаляем строки, где ТипОперации = DobavlenieProduktaVSpisokVOperaciiUstanovka
    df = df[df["ТипОперации"] != "DobavlenieProduktaVSpisokVOperaciiUstanovka"]

    # Удаляем строки, где не заполнено НИ одно из полей
    # "КодНоменклатурыРФ", "КодНоменклатурыКЗ" или "ПросмотреннаяКатегория"
    df = df.dropna(subset=["КодНоменклатурыРФ", "КодНоменклатурыКЗ"], how="all")

    # Объединяем коды номенклатуры
    df["КодНоменклатуры"] = df["КодНоменклатурыРФ"].combine_first(df["КодНоменклатурыКЗ"]).astype(str).str[:6]
    df = df.drop(columns=["КодНоменклатурыРФ", "КодНоменклатурыКЗ"])

    # Объединяем номера телефонов
    df["Телефон"] = (df["ТелефонОсновной"].combine_first(df["ЗапаснойТелефон"])
    .apply(
        lambda x: str(x)[:-2] if pd.notnull(x) and str(x).endswith(".0") else str(x) if pd.notnull(x) else np.nan))
    df = df.drop(columns=["ТелефонОсновной", "ЗапаснойТелефон"])

    # В колонке Дата оставляем только дату (убираем время)
    df["Дата"], malformed_date_count = _parse_interaction_date(df["Дата"])

    df["ДатаРождения"] = pd.to_datetime(df["ДатаРождения"], errors='coerce')

    # Рассчитываем возраст на момент заказа
    df = add_age_features(df)

    # Добавляем колонку с возрастной группой
    df["ВозрастнаяГруппа"] = df["Возраст"].apply(get_age_group)

    # Объединяем Имя, Фамилия, Отчество в ФИО
    df["ФИО"] = df[["Фамилия", "Имя", "Отчество"]].fillna("").agg(" ".join, axis=1).str.strip()
    df = df.drop(columns=["Имя", "Фамилия", "Отчество"])

    # В колонке Пол заменяем значения
    df["ПолКлиента"] = df["ПолКлиента"].replace({"male": "Мужской", "female": "Женский"})

    df["ДисконтнаяКарта"] = clean_id_series(df["ДисконтнаяКарта"])
    df["MindboxID"] = clean_id_series(df["MindboxID"])

    # Очистка категорий от точек и запятых
    for col in [
        "СамаяПросматриваемаяКатегория",
        "СамаяПросматриваемаяРодительскаяКатегория",
        "СамаяПросматриваемаяДочерняяКатегория"
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)  # убираем только .0 в конце
            .str.replace(r"[.,]", "", regex=True)  # убираем лишние знаки, если вдруг есть
            .str.strip()
        )

    # Объединяем любимые категории в одну колонку через "_"
    df["ЛюбимаяКатегория"] = (df[["СамаяПросматриваемаяКатегория",
                                  "СамаяПросматриваемаяРодительскаяКатегория",
                                  "СамаяПросматриваемаяДочерняяКатегория"]]
                              .apply(lambda row: "_".join([str(x) for x in row if pd.notnull(x) and x != "nan"]),
                                     axis=1))

    # Удалим отдельные колонки категорий
    df = df.drop(columns=["СамаяПросматриваемаяКатегория",
                          "СамаяПросматриваемаяРодительскаяКатегория",
                          "СамаяПросматриваемаяДочерняяКатегория"])

    # --- Подтягиваем данные из Номенклатуры.csv ---
    fav_path = "ВходныеДанные/Номенклатура.csv"

    if not os.path.isfile(fav_path):
        show_custom_message(aboba,
                            title="Ошибка",
                            text="Для корректной загрузки необходимо сначала загрузить файл Номенклатура.csv",
                            image_path="Картинки/Неудача.png"
                            )
        set_status_error(aboba, "Отсутствует файл Номенклатура.csv")
        schedule_status_reset(aboba, 5)
        return None

    # Читаем файл номенклатуры
    fav_df = pd.read_csv(fav_path, sep="|", dtype=str)

    # Проверяем обязательные колонки
    required_fav_cols = ["КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте", "Коллекция"]
    missing_fav = [col for col in required_fav_cols if col not in fav_df.columns]

    if missing_fav:
        show_custom_message(aboba,
                            title="Ошибка",
                            text="В файле Номенклатура.csv отсутствуют колонки:\n" + "\n".join(missing_fav),
                            image_path="Картинки/Неудача.png"
                            )
        set_status_error(aboba, "Отсутствуют необходимые колонки")
        schedule_status_reset(aboba, 5)

        return None

    # Оставляем только нужные колонки
    fav_df = fav_df[required_fav_cols]

    # Объединяем основной df с номенклатурой
    df = df.merge(fav_df, on="КодНоменклатуры", how="left")

    # Заменяем все виды пустых значений на np.nan
    df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "nan"], np.nan)

    # Упорядочиваем колонки
    column_order = [
        "Дата", "КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте", "Коллекция",
        "ДисконтнаяКарта", "MindboxID", "ФИО", "ДатаРождения", "Возраст", "ВозрастнаяГруппа", "ПолКлиента",
        "Почта", "Телефон", "ЛюбимаяКатегория"
    ]
    df = df[column_order]

    df = df.sort_values(by="Дата", ascending=True)

    _set_processing_complete_status(aboba, malformed_date_count)

    # возвращаем обработанный DataFrame
    return df


# -------------------------------------------ОБРАБОТКА НОМНЕКЛАТУРЫ-----------------------------------------------------
def process_nomenclature_file(aboba, df):
    # Список нужных колонок
    required_columns = [
        "КодНоменклатуры", "Номенклатура", "НазваниеНаСайте", "ВидНоменклатуры",
        "ВидАссортимента", "Марка", "Коллекция", "СезонНоски", "ПолНоменклатуры",
        "ГруппаСоставов", "КатегорияНаСайте", "СтилеваяГруппа",
        "ТитульнаяФотография", "Остаток"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        show_custom_message(
            aboba,
            title="Ошибка",
            text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
            image_path="Картинки/Неудача.png"
        )
        set_status_error(aboba, "Отсутствуют необходимые колонки")
        schedule_status_reset(aboba, 5)

        return None

    # Удаляем строки, где КодНоменклатуры пустой, None, NaN или только пробелы
    df = df[df["КодНоменклатуры"].notna()].copy()
    df = df[df["КодНоменклатуры"].astype(str).str.strip() != ""].copy()

    # Очистка категорий от точек и запятых
    for col in [
        "КатегорияНаСайте",
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[.,]", "", regex=True)
            .str.strip()
        )

    # Заменяем все виды пустых значений на np.nan
    df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "nan"], np.nan)

    # Приводим Остаток к целому числу
    df["Остаток"] = (
        df["Остаток"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"\s+", "", regex=True)
    )

    df["Остаток"] = (
        pd.to_numeric(df["Остаток"], errors="coerce")
        .fillna(0)
        .round()
        .astype("int64")
    )

    # Упорядочиваем колонки
    column_order = [
        "КодНоменклатуры", "Номенклатура", "НазваниеНаСайте", "ВидНоменклатуры",
        "ВидАссортимента", "Марка", "Коллекция", "СезонНоски", "ПолНоменклатуры",
        "ГруппаСоставов", "КатегорияНаСайте", "СтилеваяГруппа",
        "ТитульнаяФотография", "Остаток"
    ]

    df = df[column_order]

    df = df.sort_values(by="КодНоменклатуры", ascending=True)

    set_status_ok(aboba, "Обработка завершена")
    schedule_status_reset(aboba, 5)

    # возвращаем обработанный DataFrame
    return df


# -------------------------------------------ОБРАБОТКА КАТЕГОРИЙ--------------------------------------------------------
def process_categories_file(aboba, df):
    # Список нужных колонок
    required_columns = [
        "КодКатегории", "НазваниеКатегории", "КодРодительскойКатегории"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        show_custom_message(aboba, title="Ошибка",
                            text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
                            image_path="Картинки/Неудача.png")
        set_status_error(aboba, "Отсутствуют необходимые колонки")
        schedule_status_reset(aboba, 5)

        return None

    # Очистка категорий от точек и запятых
    for col in [
        "КодКатегории",
        "КодРодительскойКатегории"
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)  # убираем только .0 в конце
            .str.replace(r"[.,]", "", regex=True)  # убираем лишние знаки, если вдруг есть
            .str.strip()
        )

    # Заменяем все виды пустых значений на np.nan
    df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "nan"], np.nan)

    # Упорядочиваем колонки
    column_order = ["КодКатегории", "НазваниеКатегории", "КодРодительскойКатегории"]

    df = df[column_order]

    df = df.sort_values(by="КодКатегории", ascending=True)

    set_status_ok(aboba, "Обработка завершена")
    schedule_status_reset(aboba, 5)

    # возвращаем обработанный DataFrame
    return df


# -------------------------------------------ОБРАБОТКА КООРДИНАТ--------------------------------------------------------
def process_coordinates_file(aboba, df):

    # 2) Проверяем обязательные колонки
    required_columns = [
        "Город", "Широта", "Долгота"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        show_custom_message(
            aboba,
            title="Ошибка",
            text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
            image_path="Картинки/Неудача.png"
        )
        set_status_error(aboba, "Отсутствуют необходимые колонки")
        schedule_status_reset(aboba, 5)

        return None

    set_status_processing(aboba, "Обработка координат городов...")

    # 3) Очистка / нормализация
    df = df.copy()

    df["Город"] = (
        df["Город"]
        .astype(str)
        .str.strip()
    )

    # Широта/Долгота: убираем .0 в конце, пробелы, приводим десятичную запятую к точке
    for col in ["Широта", "Долгота"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(",", ".", regex=False)
        )

    # Заменяем все виды пустых значений на np.nan
    df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "<NA>", "nan"], np.nan)

    # Приводим координаты к числам
    df["Широта"] = pd.to_numeric(df["Широта"], errors="coerce")
    df["Долгота"] = pd.to_numeric(df["Долгота"], errors="coerce")

    # Удаляем строки без города/координат
    df = df[df["Город"].notna() & df["Широта"].notna() & df["Долгота"].notna()].copy()
    df["Город"] = df["Город"].astype(str).str.strip()
    df = df[df["Город"].str.len() > 0].copy()

    # Убираем дубликаты городов с одинаковыми координатами
    df = df.drop_duplicates(subset=["Город", "Широта", "Долгота"], keep="first")

    if df.empty:
        show_custom_message(
            aboba,
            title="Ошибка",
            text="После очистки файла не осталось строк с корректными городами и координатами",
            image_path="Картинки/Неудача.png"
        )
        set_status_error(aboba, "Нет корректных координат")
        schedule_status_reset(aboba, 5)
        return None

    # Упорядочиваем колонки
    column_order = ["Город", "Широта", "Долгота"]
    df = df[column_order]

    # Сортировка
    df = df.sort_values(by="Город", ascending=True).reset_index(drop=True)

    set_status_ok(aboba, "Обработка завершена")
    schedule_status_reset(aboba, 5)

    return df


# -------------------------------------------ПРОВЕРКА ОТБОРА ПО ДАТЕ----------------------------------------------------
def _masked_date_is_empty(qle) -> bool:

    if qle is None:
        return True

    text = qle.text()
    if text is None:
        return True

    text = text.replace(" ", "").replace(".", "")
    return text == ""


# -------------------------------------------ПРЕОБРАЗОВАНИЕ ДАТЫ ДЛЯ ЗАПРОСА--------------------------------------------
def _get_weather_period_from_filter(aboba):

    date_from_widget = getattr(aboba, "filter_date_from", None)
    date_to_widget = getattr(aboba, "filter_date_to", None)

    if _masked_date_is_empty(date_from_widget) or _masked_date_is_empty(date_to_widget):
        set_status_error(aboba, "Для загрузки погоды необходимо заполнить период")
        schedule_status_reset(aboba, 5)

        show_custom_message(
            aboba,
            title="Ошибка",
            text="Перед загрузкой координат городов и погоды необходимо заполнить период.",
            image_path="Картинки/Неудача.png"
        )
        return None

    date_from_text = date_from_widget.text().strip()
    date_to_text = date_to_widget.text().strip()

    date_from = pd.to_datetime(date_from_text, errors="coerce", dayfirst=True)
    date_to = pd.to_datetime(date_to_text, errors="coerce", dayfirst=True)

    if pd.isna(date_from) or pd.isna(date_to):
        set_status_error(aboba, "Некорректный период отбора")
        schedule_status_reset(aboba, 5)

        show_custom_message(
            aboba,
            title="Ошибка",
            text="Период заполнен некорректно. Ожидаемый формат даты: дд.мм.гггг.",
            image_path="Картинки/Неудача.png"
        )
        return None

    if date_from > date_to:
        set_status_error(aboba, "Дата начала больше даты окончания")
        schedule_status_reset(aboba, 5)

        show_custom_message(
            aboba,
            title="Ошибка",
            text="Дата начала периода не может быть больше даты окончания периода.",
            image_path="Картинки/Неудача.png"
        )
        return None

    return date_from.strftime("%Y-%m-%d"), date_to.strftime("%Y-%m-%d")


# -------------------------------------------СЛОВАРЬ С ПОГОДОЙ----------------------------------------------------------
WEATHER_CODE_RU = {
    0: "Ясно",
    1: "В основном ясно",
    2: "Переменная облачность",
    3: "Пасмурно",
    45: "Туман",
    48: "Туман с изморозью",
    51: "Морось (слабая)",
    53: "Морось (умеренная)",
    55: "Морось (сильная)",
    56: "Переохлаждённая морось (слабая)",
    57: "Переохлаждённая морось (сильная)",
    61: "Дождь (слабый)",
    63: "Дождь (умеренный)",
    65: "Дождь (сильный)",
    66: "Переохлаждённый дождь (слабый)",
    67: "Переохлаждённый дождь (сильный)",
    71: "Снег (слабый)",
    73: "Снег (умеренный)",
    75: "Снег (сильный)",
    77: "Снежные зёрна",
    80: "Ливни (слабые)",
    81: "Ливни (умеренные)",
    82: "Ливни (сильные)",
    85: "Снежные заряды (слабые)",
    86: "Снежные заряды (сильные)",
    95: "Гроза (слабая/умеренная)",
    96: "Гроза с небольшим градом",
    99: "Гроза с сильным градом",
}


# -------------------------------------------СООТВЕТСТВИЕ КОДА С ПОГОДОЙ------------------------------------------------
def _weather_code_to_text(code) -> str:
    if pd.isna(code):
        return pd.NA
    try:
        code_int = int(code)
    except Exception:
        return pd.NA
    return WEATHER_CODE_RU.get(code_int, f"Неизвестно (код {code_int})")


# -------------------------------------------ЗАГРУЗКА ПОГОДЫ ПО КООРДИНАТАМ---------------------------------------------
def _download_daily_weather_by_coordinates(
        city: str,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone: str = "auto",
) -> pd.DataFrame:

    # Всегда делаем полный диапазон дат, чтобы "дырки" стали NA, а не ломали файл
    dates = pd.date_range(start=start_date, end=end_date, freq="D").strftime("%Y-%m-%d").tolist()

    # Пустышка на весь период (если API не ответил/частично ответил)
    result = pd.DataFrame({
        "Дата": dates,
        "Город": city,
        "Широта": latitude,
        "Долгота": longitude,
        "СредняяТемпература": [pd.NA] * len(dates),
        "КоличествоОсадков": [pd.NA] * len(dates),
        "КодПогоды": [pd.NA] * len(dates),
    })

    daily_vars = ["temperature_2m_mean", "precipitation_sum", "weather_code"]

    data = _get_json_with_retry(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(daily_vars),
            "timezone": timezone,
        },
        attempts=3,
        timeout=(8.0, 15.0),  # <- критично: (connect, read)
        backoff_base=1.5,
    )

    if data is None:
        # вообще ничего не получили — вернём пустышку
        result["КодПогоды"] = pd.to_numeric(result["КодПогоды"], errors="coerce").astype("Int64")
        result["ПогодныеУсловия"] = result["КодПогоды"].map(_weather_code_to_text)
        result.attrs["weather_request_status"] = "failed"
        return result

    if not data:
        result["КодПогоды"] = pd.to_numeric(result["КодПогоды"], errors="coerce").astype("Int64")
        result["ПогодныеУсловия"] = result["КодПогоды"].map(_weather_code_to_text)
        result.attrs["weather_request_status"] = "empty"
        return result

    daily = data.get("daily") or {}
    api_dates = daily.get("time", []) or []

    # если daily отсутствует — тоже пустышка
    if not api_dates:
        result["КодПогоды"] = pd.to_numeric(result["КодПогоды"], errors="coerce").astype("Int64")
        result["ПогодныеУсловия"] = result["КодПогоды"].map(_weather_code_to_text)
        result.attrs["weather_request_status"] = "empty"
        return result

    # маппим значения по датам (чтобы частичный ответ не ломал)
    def _to_map(values):
        values = values or []
        return dict(zip(api_dates, values))

    m_temp = _to_map(daily.get("temperature_2m_mean"))
    m_prec = _to_map(daily.get("precipitation_sum"))
    m_code = _to_map(daily.get("weather_code"))

    result["СредняяТемпература"] = [m_temp.get(d, pd.NA) for d in dates]
    result["КоличествоОсадков"] = [m_prec.get(d, pd.NA) for d in dates]
    result["КодПогоды"] = [m_code.get(d, pd.NA) for d in dates]

    result["КодПогоды"] = pd.to_numeric(result["КодПогоды"], errors="coerce").astype("Int64")
    result["ПогодныеУсловия"] = result["КодПогоды"].map(_weather_code_to_text)
    result.attrs["weather_request_status"] = "success"

    return result


# -------------------------------------------ФОРМИРУЕМ ФАЙЛ С ПОГОДОЙ---------------------------------------------------
def _download_weather_for_coordinates_file(aboba, coords_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    total = len(coords_df)
    outcome_counts = {"success": 0, "failed": 0, "empty": 0}

    # (опционально) чтобы UI “дышал” и обновлял статус
    try:
        from PyQt6.QtWidgets import QApplication
        _qt = QApplication
    except Exception:
        _qt = None

    for idx, (_, row) in enumerate(coords_df.iterrows(), start=1):
        city = str(row["Город"]).strip()
        latitude = float(row["Широта"])
        longitude = float(row["Долгота"])

        set_status_processing(aboba, f"Получаем данные...{city} ({idx}/{total})...")

        if _qt is not None:
            _qt.processEvents()

        city_weather = _download_daily_weather_by_coordinates(
            city=city,
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            timezone="auto",
        )
        outcome = city_weather.attrs.get("weather_request_status", "success")
        if outcome not in outcome_counts:
            outcome = "failed"
        outcome_counts[outcome] += 1
        frames.append(city_weather)

    if frames:
        weather_df = pd.concat(frames, ignore_index=True)
    else:
        weather_df = pd.DataFrame(columns=[
            "Дата", "Город", "Широта", "Долгота",
            "СредняяТемпература", "КоличествоОсадков", "КодПогоды", "ПогодныеУсловия",
        ])

    column_order = [
        "Дата", "Город", "ПогодныеУсловия",
        "СредняяТемпература", "КоличествоОсадков", "КодПогоды",
    ]

    weather_df = weather_df[column_order]
    weather_df["Дата"] = pd.to_datetime(weather_df["Дата"], errors="coerce").dt.normalize()
    weather_df = weather_df.sort_values(by=["Город", "Дата"], ascending=True)

    out_path = os.path.join(os.getcwd(), "ВходныеДанные", "Погода.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    weather_df.to_csv(out_path, index=False, sep="|", encoding="utf-8-sig")

    weather_df.attrs.update(
        weather_total_cities=total,
        weather_successful_cities=outcome_counts["success"],
        weather_failed_cities=outcome_counts["failed"],
        weather_empty_cities=outcome_counts["empty"],
    )

    return weather_df


# -------------------------------------------ПОЛУЧАЕМ КООРДИНАТЫ ИЗ ФАЙЛА-----------------------------------------------
def generate_weather_for_saved_coordinates(aboba, start_date: str, end_date: str) -> pd.DataFrame:
    coords_path = os.path.join(os.getcwd(), "ВходныеДанные", "КоординатыГородов.csv")
    if not os.path.isfile(coords_path):
        raise FileNotFoundError("Не найден файл КоординатыГородов.csv")

    coords_df = pd.read_csv(coords_path, sep="|", encoding="utf-8-sig", dtype=str)
    coords_df.columns = [str(c).replace("\ufeff", "").strip() for c in coords_df.columns]

    required = ["Город", "Широта", "Долгота"]
    missing = [c for c in required if c not in coords_df.columns]
    if missing:
        raise ValueError("В файле координат отсутствуют колонки: " + ", ".join(missing))

    coords_df = coords_df[required].copy()
    coords_df["Город"] = coords_df["Город"].astype(str).str.strip()

    for col in ["Широта", "Долгота"]:
        coords_df[col] = (
            coords_df[col].astype(str).str.strip()
            .str.replace(",", ".", regex=False)
            .str.replace(r"\.0$", "", regex=True)
        )

    coords_df["Широта"] = pd.to_numeric(coords_df["Широта"], errors="coerce")
    coords_df["Долгота"] = pd.to_numeric(coords_df["Долгота"], errors="coerce")
    coords_df = coords_df.dropna(subset=["Город", "Широта", "Долгота"])
    coords_df = coords_df[coords_df["Город"].astype(str).str.len() > 0]
    coords_df = coords_df.drop_duplicates(subset=["Город", "Широта", "Долгота"], keep="first")

    if coords_df.empty:
        raise ValueError("Файл координат пустой после очистки")

    return _download_weather_for_coordinates_file(
        aboba=aboba,
        coords_df=coords_df,
        start_date=start_date,
        end_date=end_date,
    )



