"""Read-only report blocks shared by the UI, HTML and Excel."""

from bisect import bisect_left
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal

from Application import statistics_report_presentation as report
from Application.statistics_chart_data import (
    ChartData, RetentionCell, RetentionHeatmapData, RetentionRow, SECONDARY_LIGHT, _tooltip_money, exact_number, month_label,
)
from Application.statistics_cohorts_core import HORIZONS
from Application.statistics_unified_dto import RESULT_SCHEMA
from Application.statistics_presentation import CUSTOMER_GROUP_ORDER
from Application.statistics_rfm_presentation import SEGMENT_LABELS, WINDOW_LABELS

UNIFIED_ERRORS = {
    "STATISTICS_DATA_REQUIRED": "Загрузите действия или заказы Mindbox, затем нажмите «Рассчитать статистику».",
    "SAVED_REPORT_STALE": "Сохраненная статистика устарела или недоступна. Нажмите «Рассчитать статистику».",
    "ANALYTICAL_INDEX_MISSING": "Аналитический индекс не подготовлен. Выполните подготовку статистических данных и повторите расчет.",
    "ANALYTICAL_INDEX_STALE": "Аналитический индекс устарел. Обновите подготовленные статистические данные.",
    "INVALID_ANALYTICAL_INDEX": "Проверка аналитического индекса не пройдена. Подготовьте статистические данные заново.",
    "INDEX_MISSING": "Индекс статистики INDEXED 4 отсутствует. Подготовьте статистические данные и повторите расчет.",
    "INCOMPATIBLE_SCHEMA": "Поколение индекса несовместимо с INDEXED 4.",
    "STALE_SOURCE": "Источник изменился или индекс устарел. Проверьте выгрузки и повторите расчет.",
    "SOURCE_UNAVAILABLE": "Данные Mindbox недоступны. Загрузите действия или заказы и проверьте справочник товаров.",
    "INVALID_UNIFIED_INDEX_INTEGRITY": "Проверка целостности SQLite не пройдена.",
    "UNIFIED_INDEX_UNAVAILABLE": "Не удалось открыть проекцию INDEXED 4. Проверьте поколение, источник и права доступа.",
    "INVALID_MANIFEST": "Описание поколения INDEXED 4 повреждено или не соответствует указателю.",
    "GENERATION_CHANGED": "Поколение INDEXED 4 изменилось во время чтения. Повторите расчет.",
    "INVALID_DEMOGRAPHICS": "Проверка целостности клиентской части индекса не пройдена.",
    "UNIFIED_CACHE_MISS": "Сохраненной статистики нет. Нажмите «Рассчитать статистику».",
    "INSUFFICIENT_DISK": "Недостаточно свободного места для расчета. Освободите место на диске проекта.",
    "INSUFFICIENT_MEMORY": "Недостаточно памяти для расчета. Освободите ресурсы и повторите попытку.",
    "INVALID_UNIFIED_DTO": "Процесс вернул несовместимый отчет статистики.",
    "CACHE_WRITE_FAILED": "Не удалось прочитать или сохранить отчет. Проверьте место и права доступа к кешу.",
    "INVALID_FILTER": "Отбор статистики некорректен. Проверьте выбранные значения.",
    "UNIFIED_FAILED": "Не удалось завершить расчет единой статистики. Проверьте состояние источника.",
    "CANCELLED": "Расчет отменен.",
}
_STORAGE_MESSAGE = "Защищенное хранилище недоступно. Проверьте права на рабочий каталог, кеш и индекс."
for _code in ("PRIVATE_INDEX_ACL_FAILED", "PRIVATE_INDEX_ROOT_REQUIRED", "UNSAFE_STORAGE_ACL",
              "UNSAFE_REPARSE_PATH", "UNTRUSTED_STORAGE_OWNER_OR_ACL", "OWNER_FULL_CONTROL_REQUIRED",
              "UNSUPPORTED_PRIVATE_DENY_ACL", "ACL_INSPECTION_FAILED", "UNSAFE_CACHE_FILE",
              "CACHE_ROOT_CHANGED", "CACHE_FILE_CHANGED", "STAGING_PATH_CHANGED"):
    UNIFIED_ERRORS[_code] = _STORAGE_MESSAGE



_CURRENCIES = ("RUB", "KZT")
_GROUPS = {
    "COMPLETED": "Выполнено", "PARTIALLY_COMPLETED": "Частично выполнено",
    "PAID_OR_IN_PROGRESS": "Оплачено или в процессе", "ORDER_IN_PROGRESS": "В обработке",
    "DELIVERY_IN_PROGRESS": "Доставляется", "CANCELLED": "Отмена", "RETURNED": "Возвратное состояние",
    "UNKNOWN": "Не определена", "COMPLETED_ONLY": "Только выполненные", "PARTIAL_ONLY": "Только частично выполненные",
    "CANCELLED_ONLY": "Только отмененные", "RETURNED_ONLY": "Только возвратные",
    "IN_PROGRESS_ONLY": "Только в процессе", "MIXED": "Смешанное состояние", "EMPTY": "Без позиций",
    "UNAVAILABLE": "Недостаточно данных", "CONFLICT": "Конфликт снимков",
}
_REASONS = {
    "PRICE_MISSING": "Цена строки отсутствует", "PRICE_INVALID": "Цена строки непригодна",
    "STATUS_CONFLICT": "Снимки расходятся по статусу", "PRICE_CONFLICT": "Снимки расходятся по цене",
    "LEDGER_UNAVAILABLE": "Подтвержденный реестр денежных возвратов отсутствует",
    "FULFILLMENT_UNVERIFIED": "Фактически выполненное количество не подтверждено",
    "RETURNED_QUANTITY_UNVERIFIED": "Фактически возвращенное количество не подтверждено",
    "COVERAGE_GAP": "Период покрыт не полностью", "COVERAGE_GAP_OR_TRUNCATION": "Период покрыт не полностью",
    "ORDERS_COVERAGE_INCOMPLETE": "Недостаточное покрытие заказов", "RIGHT_CENSORED": "Период наблюдения еще не завершен",
    "NO_ELIGIBLE_BUYERS": "Нет покупателей с полным периодом наблюдения",
    "NO_FULLY_VALUED_BASE": "Нет пригодных покупочных сумм", "LINES_INCOMPLETE": "Данные о позициях неполны",
    "CREATED_TIME_CONFLICT": "Снимки расходятся по дате оформления", "CREATED_TIME_MISSING": "Дата оформления отсутствует",
    "ORDER_DATE_UNRESOLVED": "Дата оформления не установлена", "OWNER_UNRESOLVED": "Покупатель не установлен",
    "MIXED_ORDER_CURRENCY": "В заказе несколько валют", "CURRENCY_UNKNOWN": "Валюта не установлена",
    "ATTRIBUTE_MISSING": "Реквизит отсутствует", "ATTRIBUTE_INVALID": "Реквизит невалиден",
    "ATTRIBUTE_CONFLICT": "Снимки расходятся по реквизиту",
    "DELIVERY_CURRENCY_UNVERIFIED": "Валюта доставки по всему заказу не установлена",
    "LINE_ID_DUPLICATE": "В снимке заказа повторяется ID позиции (LINE_ID_DUPLICATE)",
}
_ACTIONS = ("VIEW", "FAVORITE", "PURCHASE")


def _number(value, *, money=True, places=None):
    return None if value is None else report.ReportNumber(Decimal(str(value)), money=money, places=places)


def measure(cell, *, money=False, quantity=False, percentage=False):
    value, known = cell["value"], cell.get("known")
    quality, reasons = cell["quality"], tuple(cell["reasons"])
    if money or quantity or percentage:
        value, known = (_number(v, money=not quantity) for v in (value, known))
    return report.ReportMeasure(value, known, quality, reasons)


def _plain_money(value):
    number = _number(value)
    return report.ReportMeasure(number, number, "KNOWN") if number is not None else None


def _age_label(value):
    # The saved age bucket is a report label generated by the app.
    return "Возраст не определен" if value == "Возраст не определ\u0451н" else value


def _date_range(interval):
    if not interval:
        return "—"
    start, end = map(datetime.fromisoformat, interval)
    if start.time() != time.min or end.time() != time.min:
        return f"{start:%d.%m.%Y %H:%M} — {end:%d.%m.%Y %H:%M} UTC"
    last = (end - timedelta(microseconds=1)).date()
    return f"{start:%d.%m.%Y} — {last:%d.%m.%Y}"


def report_header(dto, applied_filter=None):
    params = dto.metadata["parameters"]
    period = params["order_period"] or dto.contexts["order_product_slice"]["period"]
    if period is None:
        coverage = dto.coverage["raw_orders"]
        intervals = (*coverage["orders"], *coverage["actions"])
        if intervals:
            period = min(i[0] for i in intervals), max(i[1] for i in intervals)
    rows = [("Источник", "Mindbox"), ("Период", _date_range(period)),
            ("Дата расчета", datetime.fromisoformat(dto.calculated_at).astimezone().strftime("%d.%m.%Y %H:%M"))]
    if applied_filter is not None:
        from Application.analysis_filter import AnalysisFilter

        if set(applied_filter) != {"selection", "product", "category"} or any(
                not isinstance(applied_filter[key], str) or len(applied_filter[key]) > 1024
                for key in ("product", "category")):
            raise ValueError("Invalid report filter")
        selection = AnalysisFilter.from_dict(applied_filter["selection"])
        if (selection.start_date, selection.end_date) != (params["selected_start"], params["selected_end"]):
            raise ValueError("Report filter period does not match the DTO")
        if bool(params["product_filter_key"]) != bool(
                selection.product_restricted or applied_filter["product"] or applied_filter["category"]):
            raise ValueError("Report product filter does not match the DTO")
        for label, values in (("Вид номенклатуры", selection.nomenclature_types), ("Сезон", selection.collections)):
            if values is not None:
                rows.append((label, ", ".join(value or "Не указано" for value in values)))
        for label, key in (("Товар", "product"), ("Категория", "category")):
            if applied_filter[key]:
                rows.append((label, applied_filter[key]))
    elif params["product_filter_key"]:
        rows.append(("Отбор", "Товарный отбор применен"))
    if dto.rfm["as_of_date"]:
        rows.append(("Дата отсечения покупок и RFM", date.fromisoformat(dto.rfm["as_of_date"]).strftime("%d.%m.%Y")))
    return tuple(rows)


def _summary(header):
    fields = dict(header)
    text = (f"Статистика рассчитана за период: {fields['Период']}. " if fields["Период"] != "—" else "")
    text += f"Дата расчета: {fields['Дата расчета']}."
    for key, title in (("Вид номенклатуры", "Отбор по виду номенклатуры"),
                       ("Сезон", "Отбор по сезону"), ("Товар", "Отбор по товару"),
                       ("Категория", "Отбор по категории")):
        if key in fields:
            text += f" {title}: {fields[key]}."
    if "Отбор" in fields:
        text += f" {fields['Отбор']}."
    return text


def _heading(title):
    return report.ReportBlock("heading", title=title)


def _cards(values):
    return report.ReportBlock("cards", values=tuple(values))


def _table(context, key, headers, rows):
    return report.ReportBlock("table", headers=tuple(headers), rows=tuple(tuple(row) for row in rows),
                              table_id="unified." + key, options={"context_id": context})


def _details(block, title):
    block.options["details"] = title
    return block


def _chart(title, labels, series, *, horizontal=False, line=False, donut=False, pie=False, percentage=False,
           subtotal=False, money=False, secondary=False, height=None, bar_width=None, wide_labels=False):
    labels = tuple(labels)
    def cells(values):
        return tuple(report.cell_presentation(v, percentage=percentage) for v in values)
    rendered = [(name, cells(values)) for name, values in series]
    kind = "horizontal" if horizontal else "line" if line else "donut" if donut else "pie" if pie else "bar"
    chart = ChartData(title, kind, tuple(map(str, labels)),
        tuple((name, tuple(cell.known if subtotal and cell.value is None else cell.value for cell in values))
              for name, values in rendered),
        tuple(tuple(f"{label}\n{name}: " + ("—" if cell.value is None and not subtotal else cell.text)
                    for label, cell in zip(labels, values))
              for name, values in rendered),
        zero_is_data=not pie, percentage=percentage, money=money, secondary=secondary,
        ranked=horizontal and (subtotal or wide_labels), height=height, bar_width=bar_width, wide_labels=wide_labels)
    return report.ReportBlock("chart", data=chart, analytics=True)


def _action_breakdowns(dto):
    data = dto.actions.get("breakdowns")
    if data is None:
        return []
    scope = "item_interactions"
    rows = [("Просмотры", data["mean_views_per_viewer"], data["median_views_per_viewer"]),
            ("Избранное", data["mean_favorites_per_user"], data["median_favorites_per_user"])]
    blocks = [_heading("Активность клиентов по действиям"),
              _table(scope, "actions.user_statistics", ("Действие", "Среднее на клиента", "Медиана на клиента"), rows)]
    for key, title in (("view", "Распределение просмотров на клиента"),
                       ("favorite", "Распределение добавлений в избранное на клиента")):
        distribution = data[key + "_user_activity_distribution"]
        blocks.extend([_heading(title), _chart(title, [r[0] for r in distribution], [("Клиенты", [r[1] for r in distribution])]),
                       _table(scope, "actions." + key + "_distribution", ("Действий", "Клиенты", "Доля, %"), distribution)])
    monthly = data["action_monthly_dynamics"]
    for title, count, people in (("Динамика просмотров", 1, 2), ("Динамика избранного", 3, 4)):
        blocks.extend([_heading(title), _chart(title, [r[0] for r in monthly], [
            ("Действия", [r[count] for r in monthly]), ("Клиенты", [r[people] for r in monthly])], line=True)])
    blocks.append(_table(scope, "actions.monthly", ("Месяц", "Просмотры", "Клиенты с просмотрами", "Избранное",
                                                  "Клиенты с избранным"), monthly))
    blocks.extend([_heading("Каналы действий"), _table(scope, "actions.channels",
        ("Канал", "Просмотры", "Клиенты с просмотрами", "Доля просмотров, %", "Избранное", "Клиенты с избранным", "Доля избранного, %"),
        (r[1:] for r in data["action_channel_statistics"])), _heading("Параметры просмотров"),
        _table(scope, "actions.availability", ("Наличие при просмотре", "Просмотры", "Доля, %"), data["view_availability_distribution"]),
        _table(scope, "actions.view_prices", ("Валюта", "Просмотры с ценой", "Средняя цена", "Медианная цена"),
               ((unit, n, _plain_money(mean) if n else None, _plain_money(median) if n else None)
                for unit, n, mean, median in data["view_price_statistics"]))])
    if "source_events" in data:
        counters = data["source_events"]
        views, favorites = (counters.get(key, 0) for key in ("VIEW", "FAVORITE"))
        total = views + favorites
        missing = counters.get("mapped_without_product", 0)
        unverified = counters.get("mapped_product_unknown", 0)
        blocks.extend([_heading("Качество исходных событий"),
                       _table(scope, "actions.source_quality", ("Показатель", "Значение"), [
                           ("Все исходные события", counters.get("all", 0)),
                           ("Исходные просмотры", views), ("Исходные добавления в избранное", favorites),
                           ("Всего просмотров / избранного", total),
                           ("Просмотры / избранное с товаром", total - missing - unverified),
                           ("Просмотры / избранное без товара", missing),
                           ("Доля без товара, %", None if not total or unverified else 100 * missing / total),
                           ("Нет данных о составе события", unverified),
                           ("Неклассифицированные события", counters.get("unmapped", 0)),
                           ("Классификация неизвестна", counters.get("classification_unknown", 0)),
                       ])])
    return blocks


def _actions(dto):
    scope = "item_interactions"
    data = dto.actions["data"]
    intersections = {
        "VIEW_PURCHASE": "Просмотры и покупки", "FAVORITE_PURCHASE": "Избранное и покупки",
        "VIEW_FAVORITE": "Просмотры и избранное", "VIEW_FAVORITE_PURCHASE": "Все три типа",
    }
    return [_cards([
                ("Просмотры товаров", measure(data["VIEW"])),
                ("Добавления в избранное", measure(data["FAVORITE"])),
                ("Клиенты с просмотрами", measure(data["VIEW_users"])),
                ("Клиенты с избранным", measure(data["FAVORITE_users"])),
                ("Позиции покупок", measure(data["PURCHASE"])),
                ("Клиенты с покупками", measure(data["PURCHASE_users"])),
            ]),
            _heading("Пересечения действий"),
            _chart("Пересечения действий", list(intersections.values()),
                   [("Клиенты", [measure(data[key]) for key in intersections])], horizontal=True),
            _table(scope, "actions.intersections", ("Пересечение", "Уникальные клиенты"),
                   ((label, measure(data[key])) for key, label in intersections.items())),
            *_action_breakdowns(dto)]


def _currency_blocks(context, key, currencies):
    for currency in _CURRENCIES:
        row = currencies[currency]
        yield _heading(f"Денежные показатели — {currency}")
        amounts = [("Сумма покупок", measure(row["amounts"]["recognized_purchase"], money=True)),
                   ("Средний чек", measure(row["mean_order"], money=True))]
        mean = row["mean_per_valued_buyer"]
        if mean is not None and mean["value"] is not None:
            amounts.append(("Средняя сумма покупок на покупателя", measure(mean, money=True)))
        yield _table(context, key + "." + currency, ("Денежный показатель", f"Значение, {currency}"),
                     ((label, amount) for label, amount in amounts if report.cell_presentation(amount).displayed_value is not None))


def _store_chart(stores, currency):
    rows = [r for r in stores if r["known_subtotal"] is not None]
    rows.sort(key=lambda r: (Decimal(r["known_subtotal"]).copy_negate(), r["channel"] or "", r["name"]))
    rows = rows[:10]
    names = [r["name"] for r in rows]
    labels = tuple(name if names.count(name) == 1 else f"{name} [{r['channel'] or 'не указан'}]"
                   for name, r in zip(names, rows))

    def money(amount):
        return f"{_tooltip_money(amount)} {currency}" if amount is not None else "—"

    tips = tuple(f"Магазин / канал: {r['name']}\nСумма покупок: {money(r['known_subtotal'])}\n"
                 f"Заказов: {exact_number(r['orders'])}\nПокупателей: {exact_number(r['buyers'])}\n"
                 f"Средний чек: {money(r['mean_valued_order']['value'])}"
                 for r in rows)
    data = ChartData(f"Магазины и каналы — {currency}", "horizontal", labels,
                     ((currency, tuple(Decimal(r["known_subtotal"]) for r in rows)),), (tips,),
                     ranked=True, money=True, zero_is_data=True)
    return report.ReportBlock("chart", data=data)


def _monthly_amount_chart(months, currency, *, secondary=False, html_line_color=None):
    labels, values = [], []
    for month, amount in months:
        labels.append(month)
        values.append(Decimal(amount) if amount is not None else None)
    tips = tuple(f"{month_label(label, full=True)}\nСумма: "
                 + (f"{_tooltip_money(value)} {currency}" if value is not None else "—")
                 for label, value in zip(labels, values))
    data = ChartData(f"Сумма покупок — {currency}", "line", tuple(labels),
                     (("Сумма", tuple(values)),), (tips,), money=True,
                     secondary=secondary, zero_is_data=True, show_legend=False, html_line_color=html_line_color)
    return report.ReportBlock("chart", data=data, analytics=True)


def _order_breakdowns(dto):
    data = dto.orders.get("breakdowns")
    if data is None:
        return []
    scope = "order_product_slice"
    rows = data["basket_distribution"]
    rates = {label: rate for label, _, rate in rows}
    large_basket = [rates.get(label) for label in ("3–5 позиций", "6–10 позиций", "11+ позиций")]
    blocks = [_heading("Корзина заказа"),
              _cards([
                  ("Среднее число позиций в заказе", _number(data["basket"]["mean"], money=False, places=2)),
                  ("Медиана числа позиций", _number(data["basket"]["median"], money=False, places=2)),
                  ("Доля заказов с 1 позицией, %", rates.get("1 позиция")),
                  ("Доля заказов с 3+ позициями, %",
                   sum(large_basket) if all(rate is not None for rate in large_basket) else None),
              ]),
              _chart("Состав корзины", [r[0] for r in rows], [("Заказы", [r[1] for r in rows])]),
              _table(scope, "orders.basket", ("Количество позиций", "Заказы", "Доля, %"), rows)]
    if "currencies" in data:
        blocks.extend([_heading("Финансовые показатели покупок"),
                       _table(scope, "orders.financials", ("Валюта", "Заказы с покупкой", "Позиции покупок", "Единицы по заказам",
                              "Сумма покупок", "Средний чек", "Медианный чек"),
                              ((currency, r["orders"], r["positions"], _number(r["nominal_units"], money=False),
                                measure(dict(r["recognized"], known=r["known_subtotal"]), money=True),
                                measure(r["mean_valued_order"], money=True), _plain_money(data["median_check"][currency]["median"]))
                               for currency, r in data["currencies"].items()))])
    rows = data["payment_types"]
    blocks.extend([_heading("Способ оплаты"),
                   _chart("Способ оплаты", [r[0] for r in rows], [("Заказы", [r[1] for r in rows])]),
                   _table(scope, "orders.payment_types", ("Способ оплаты", "Заказы", "Доля, %"), rows)])
    for currency in _CURRENCIES:
        months = [r for r in data["months"] if r["currency"] == currency]
        stores = [r for r in data["stores"] if r["currency"] == currency]

        def recognized(row):
            return measure(dict(row["recognized"], known=row["known_subtotal"]), money=True)

        blocks.extend([_heading("Динамика заказов с покупкой — " + currency),
            _chart("Заказы с покупкой — " + currency, [r["month"] for r in months],
                   [("Заказы", [r["orders"] for r in months])], line=True),
            _heading("Динамика суммы покупок — " + currency),
            _monthly_amount_chart(((row["month"], row["recognized"]["value"]) for row in months), currency,
                                  secondary=True, html_line_color=SECONDARY_LIGHT if currency == "RUB" else None),
            _table(scope, "orders.monthly." + currency, ("Месяц", "Заказы с покупкой", f"Сумма покупок, {currency}"),
                   ((r["month"], r["orders"], recognized(r)) for r in months)),
            _heading("Магазины и каналы — " + currency),
            _store_chart(stores, currency),
            _table(scope, "orders.stores." + currency, ("Канал", "Заказы с покупкой", "Покупатели", "Позиции покупок",
                "Единицы по заказам", f"Сумма покупок, {currency}", f"Средний чек, {currency}"),
                ((r["name"], r["orders"], r["buyers"], r["positions"], _number(r["nominal_units"], money=False),
                  recognized(r), measure(r["mean_valued_order"], money=True))
                 for r in stores))])
    return blocks


def _order_status_rows(data, total, *, label, only, share_label):
    counts = {"orders": label, "positions": "Позиции", "only_orders": only,
              "mixed_orders": "Заказы со смешанными статусами", "customers": "Клиенты",
              "returned_units": "Фактически возвращенные единицы"}
    rows = []
    for key, title in counts.items():
        cell = data.get(key)
        if cell and (cell["value"] is not None or cell["known"] is not None):
            rows.append((title, measure(cell, quantity=key == "returned_units")))
    orders = data["orders"]
    partial = orders["value"] is None
    numerator = orders["known"] if partial else orders["value"]
    if total and numerator is not None and (numerator > 0 or not partial):
        rows.insert(1, (share_label, 100 * numerator / total))
    return rows


def _orders(dto):
    scope = "order_product_slice"
    data = dto.orders
    values = {"purchase_orders": "Заказы с покупкой", "purchase_positions": "Позиции покупок",
              "buyers": "Покупатели среза"}
    blocks = [_heading("Заказы и покупки"),
              _cards([*((title, measure(data[key])) for key, title in values.items()),
                      ("Заказы без подтвержденных покупок (отмены, возвраты и т.д.)",
                       measure(data["indicators"]["orders_without_evidence"]))]),
              *_order_breakdowns(dto),
              *_currency_blocks(scope, "orders.money", dto.money["currencies"]),
              _heading("Состояния заказов"),
              _table(scope, "orders.states", ("Состояние", "Заказы"),
                     ((_GROUPS[key], measure(value)) for key, value in data["state_distribution"].items()))]
    for key, title, label, only, share_label in (
            ("cancellations", "Отмены", "Заказы с признаками отмены", "Полностью отмененные заказы",
             "Доля заказов с признаками отмены, %"),
            ("return_states", "Возвраты", "Заказы со статусами возврата", "Заказы только с возвратными позициями",
             "Доля заказов со статусами возврата, %")):
        rows = _order_status_rows(data[key], data["unique_orders"]["value"], label=label, only=only, share_label=share_label)
        if rows:
            blocks.extend([_heading(title), _table(scope, "orders." + key, ("Показатель", "Значение"), rows)])
    statuses = dto.status_catalog["rows"]
    position_base = data["positions"]["known"]
    blocks.extend([_heading("Статусы заказов Mindbox"), _table(scope, "orders.catalog",
        ("Название", "External ID", "Категория Mindbox", "Бизнес-группа", "Позиции",
         "Уникальные заказы", "Доля позиций, %", "Статус покупки"),
        ((r["name"] or "Официальное название не подтверждено", r["external_id"], r["mindbox_category"],
          _GROUPS[r["business_group"]], measure(r["positions"]), measure(r["orders"]),
          100 * r["positions"]["known"] / position_base if position_base else None,
          "Да" if r["external_id"] in ("F", "sold", "CP") else "Нет")
         for r in statuses if r["bucket"] == "CATALOG"))])
    return blocks


def _products(dto):
    scope = "order_product_slice"
    data = dto.products["data"]
    detail = dto.products.get("breakdowns")
    blocks = [_heading("Взаимодействия и покупки по товарам"),
              _cards([
                  ("Товары с взаимодействиями", len(data)),
                  ("Товары с просмотрами", sum(r["views"] > 0 for r in data)),
                  ("Товары в избранном", sum(r["favorites"] > 0 for r in data)),
                  ("Товары с покупками", dto.diagnostics["counters"]["products_with_purchases"]),
              ])]
    details = detail or {}
    for name, label, field, members, companion in (
            ("viewed", "Просмотры", "views", "viewers", "favorites"),
            ("favorited", "Избранное", "favorites", "favorite_users", "views"),
            ("purchased", "Позиции покупок", "purchase_positions", None, None)):
        def value(row):
            if name != "purchased":
                return row[field]
            cell = row[field]
            return cell["value"] if cell["value"] is not None else cell["known"]
        top = sorted((row for row in data if value(row)), key=lambda row: (-value(row), row["item_id"]))[:10]
        if top:
            title = label + " — TOP-10"
            values = [measure(r[field]) if name == "purchased" else r[field] for r in top]
            blocks.extend([_heading(title), _chart(title, [r["name"] or r["item_id"] for r in top],
                                 [(label, values)], horizontal=True, subtotal=True)])
            if members in details:
                blocks.append(_table(scope, "products.top." + name, ("Код", "Товар", label,
                              "Уникальные клиенты", "Избранное" if field == "views" else "Просмотры", "Позиции покупок"),
                             ((r["item_id"], r["name"], r[field], details[members][r["item_id"]], r[companion],
                               measure(r["purchase_positions"])) for r in top)))
            elif name == "purchased":
                blocks.append(_table(scope, "products.top." + name,
                              ("Код", "Товар", label, "Покупатели", "Единицы по заказам", "Просмотры", "Избранное"),
                              ((r["item_id"], r["name"], count, measure(r["buyers"]),
                                measure(r["nominal_units"], quantity=True), r["views"], r["favorites"])
                               for r, count in zip(top, values))))
    if detail is not None:
        for field, label in (("category", "Категории товаров"), ("gender", "Пол товара"),
                             ("collection", "Сезон"), ("style_group", "Стилевая группа")):
            grouped = detail["groups"][field]
            if field == "collection":
                grouped = sorted(grouped, key=lambda r: (r[4] is None, -(r[4] or 0), r[0].casefold(), r[0]))
            headers = (label, "Товары", "Просмотры", "Избранное", "Позиции покупок", "Единицы по заказам")
            blocks.append(_heading(label))
            if field == "category":
                names = detail.get("category_labels", {})
                category_rows = [(r[0], *names.get(r[0], (r[0], None)), *r[1:]) for r in grouped]
                rows = [(code, name, " / ".join(path or ()), *counts[:-1], _number(counts[-1], money=False))
                        for code, name, path, *counts in category_rows]
                headers = ("Код категории", "Категория", "Путь", *headers[1:])
            else:
                rows = [(*r[:-1], _number(r[-1], money=False)) for r in grouped]
                if field != "gender":
                    blocks.append(_chart(label, [r[0] for r in grouped],
                                         [("Позиции покупок", [r[4] for r in grouped])],
                                         horizontal=field == "collection"))
            blocks.append(_table(scope, "products.group." + field, headers, rows))
        if "resolution" in detail:
            resolution = detail["resolution"]
            blocks.extend([_heading("Сводка сопоставления товаров"), _table(scope, "products.resolution",
                           ("Показатель", "Значение"), [
                               ("Уникальные исходные идентификаторы товаров", resolution["source_products"]),
                               ("Уникальные распознанные товары", resolution["resolved_products"]),
                               ("Товары с просмотрами", sum(n > 0 for n in detail["viewers"].values())),
                               ("Товары в избранном", sum(n > 0 for n in detail["favorite_users"].values())),
                               ("Распознанные взаимодействия", resolution["resolved"]),
                               ("Нераспознанные взаимодействия", resolution["unresolved"]),
                               ("Доля распознанных, %", resolution["rate"]),
                           ])])
    return blocks


def _overview(dto):
    scope = "customer_population"
    data = dto.customers["data"]["counts"]
    buyer = dto.customers["data"]["purchases"]["counts"]
    actions = dto.actions["data"]
    names = {"population": "Клиенты", "population_buyers": "Клиенты с покупками",
             "nonpurchase_clients": "Клиенты с заказами без подтвержденных покупок",
             "cancelled_clients": "Клиенты с отменами", "returned_clients": "Клиенты с возвратными состояниями"}
    demo = dto.customers["data"].get("customer_demography")
    blocks = [_heading("Клиентская база"),
            _cards((("Клиенты", measure(data["population"])),
                    ("Покупатели периода", measure(buyer["buyers"])),
                    ("Повторные покупатели", measure(buyer["repeat_buyers"])),
                    ("Клиенты с просмотрами", measure(actions["VIEW_users"])),
                    ("Клиенты с избранным", measure(actions["FAVORITE_users"])),
                    ("Клиенты с просмотрами и покупками", measure(actions["VIEW_PURCHASE"])))),
            _table(scope, "customers.population", ("Показатель", "Значение"),
                   ((title, measure(data[key])) for key, title in names.items()))]
    if demo and demo["profiles"] is not None:
        blocks.extend([_heading("Демография по профилям клиентского среза"),
                       _cards(
                              (("Профили", demo["profiles"]), ("Средний возраст", demo["mean_age"]),
                               ("Медианный возраст", demo["median_age"])))])
        for key, label in (("gender", "Пол"), ("age", "Возрастная группа")):
            rows = demo[key] if key == "gender" else [(_age_label(name), *values) for name, *values in demo[key]]
            blocks.extend([_heading(label), _chart(label, [r[0] for r in rows], [("Клиенты", [r[1] for r in rows])], donut=key == "gender"),
                           _table(scope, "customers.demography." + key, (label, "Профили", "Доля, %"), rows)])
    elif demo:
        blocks.append(report.ReportBlock("text", title="Демографические данные недоступны."))
    activity = dto.customers["data"].get("customer_activity")
    if activity is not None:
        item_scope = "item_interactions"
        rows = activity["interactions"]
        blocks.extend([_heading("Взаимодействия клиентов"),
                       _table(item_scope, "customers.activity",
                              ("Показатель", "Значение"), [
                                  *((name, measure(actions[key]))
                                    for key, name in (("VIEW_users", "Клиенты с просмотрами"),
                                                      ("FAVORITE_users", "Клиенты с избранным"),
                                                      ("PURCHASE_users", "Клиенты с покупками"),
                                                      ("VIEW_PURCHASE", "Клиенты с просмотрами и покупками"),
                                                      ("FAVORITE_PURCHASE", "Клиенты с избранным и покупками"),
                                                      ("VIEW_FAVORITE_PURCHASE", "Клиенты со всеми типами взаимодействий"))),
                                  ("Среднее взаимодействий на клиента", activity["mean"]),
                                  ("Медиана взаимодействий на клиента", activity["median"]),
                                  ("Доля активных клиентов с покупками, %", activity["buyer_rate"])]),
                       _heading("Распределение активности"),
                       _chart("Распределение активности", [r[0] for r in rows], [("Клиенты", [r[1] for r in rows])]),
                       _table(item_scope, "customers.activity_distribution", ("Взаимодействий", "Клиенты", "Доля, %"), rows)])
    return blocks


def _bucket_counts(rows, limits):
    counts = [0] * (len(limits) + 1)
    for value, count in rows:
        counts[bisect_left(limits, value)] += count
    return counts


def _share(count, total):
    if not total or count is None:
        return None
    rate = 100 * count / total
    if 0 < rate < .01:
        # The usual two decimal places would hide small, nonzero shares.
        value = Decimal(str(rate))
        return report.ReportNumber(value, money=False, places=1 - value.adjusted())
    return rate


def _confirmed_count(cell):
    return cell["value"] if cell["value"] is not None else cell["known"]


def _purchase_category_labels(dto, evidence):
    products = dto.products.get("breakdowns") or {}
    cohorts = dto.cohorts.get("categories") or {}
    labels = {}
    for reference in (products.get("category_labels", {}),
                      {row["code"]: (row["name"], row["path"]) for row in cohorts.get("categories", ())},
                      evidence.get("labels", {})):
        for code, (name, path) in reference.items():
            if path or (name and name != code):
                labels[code] = (name, path)
    return labels


def _top_purchases_chart(title, rows, *series, height=None, bar_width=None):
    # The channel comparison ranks by both series combined; the full table retains its original order.
    ranked = sorted(rows, key=lambda row: (-sum(n or 0 for n in row[1:]), row[0].casefold()))[:10]
    return _chart(title, [r[0] for r in ranked],
                  [(name, [r[column] for r in ranked]) for column, name in enumerate(series, 1)],
                  horizontal=True, subtotal=True, height=height, bar_width=bar_width)


def _purchase_categories(scope, field, title, rows, labels):
    table_rows = []
    ranking = []
    for code, count in rows:
        name, path = labels.get(code, ("Категория не определена", None))
        table_rows.append((code, name, " / ".join(path or ()), count))
        ranking.append((name if code in labels else f"{name} ({code})", count))
    return [_heading(title), _top_purchases_chart(title, ranking, "Покупатели"),
            _table(scope, "customers." + field, ("Код категории", "Категория", "Путь", "Покупатели"), table_rows)]


def _copurchase_blocks(scope, data, evidence, labels):
    categories = {row["categories"][0]: row for row in data["categories"]}
    pair_counts = [(a, b, row, _confirmed_count(row["orders"]))
                   for row in data["pairs"] for a, b in (row["categories"],)]
    total = evidence["orders"]["value"]
    counts = {code: _confirmed_count(row["orders"]) for code, row in categories.items()}

    def title(code, *, full=False):
        name, path = labels.get(code, (f"Категория не определена ({code})", None))
        return " / ".join(path) if full and path else name

    return [
        _heading("Совместные покупки категорий"),
        _table(scope, "customers.copurchase_base", ("Показатель", "Значение"), (
            ("Заказы с покупкой", measure(evidence["orders"])),
            ("Заказы с несколькими категориями", measure(evidence["multicategory_orders"])),
            ("Покупатели с заказами в нескольких категориях", measure(evidence["multicategory_buyers"])),
            ("Доля заказов с несколькими категориями, %", _share(_confirmed_count(evidence["multicategory_orders"]), total)),
            ("Заказы с неполным сопоставлением категорий", evidence["unresolved_category_orders"]),
            ("Пары категорий", len(pair_counts)),
        )),
        _heading("Категории по количеству заказов — TOP-10"),
        _top_purchases_chart("Категории по количеству заказов — TOP-10",
                             ((title(code), count) for code, count in counts.items()), "Заказы"),
        _table(scope, "customers.copurchase_categories", ("Категория", "Заказы", "Покупатели", "Доля заказов, %", "Код категории"),
               ((title(code, full=True), measure(row["orders"]), measure(row["buyers"]), _share(counts[code], total), code)
                for code, row in categories.items())),
        _heading("Пары категорий по количеству заказов — TOP-10"),
        _top_purchases_chart("Пары категорий по количеству заказов — TOP-10",
                             ((f"{title(a)} + {title(b)}", count) for a, b, _, count in pair_counts), "Заказы"),
        _table(scope, "customers.copurchase_pairs", ("Категория 1", "Категория 2", "Совместные заказы",
               "Покупатели", "Доля заказов, %", "Категория 2 среди заказов с категорией 1, %",
               "Категория 1 среди заказов с категорией 2, %", "Код категории 1", "Код категории 2"),
               ((title(a, full=True), title(b, full=True), measure(row["orders"]), measure(row["buyers"]), _share(count, total),
                 _share(count, counts[a]), _share(count, counts[b]), a, b) for a, b, row, count in pair_counts)),
    ]


def _purchases(dto):
    scope = "buyer_period"
    data = dto.customers["data"]["purchases"]
    names = {"buyer_purchase_orders": "Заказы с покупкой", "buyers": "Покупатели",
             "single_buyers": "Одна покупка", "repeat_buyers": "Повторные покупатели"}
    order_buckets = ("1 заказ", "2–3 заказа", "4–5 заказов", "6–10 заказов", "11–14 заказов", "15+ заказов")
    counts = _bucket_counts(data["frequency"], (1, 3, 5, 10, 14))
    total = sum(counts)
    distribution_title = "Распределение покупателей по количеству заказов"
    first = data["first_months"]
    second = data["second_intervals"]
    restored = dto.customers["data"].get("buyer_breakdowns") or {}
    evidence = restored.get("copurchases", {})
    labels = _purchase_category_labels(dto, evidence)
    blocks = [_heading("Покупки в периоде"),
        _cards((title, measure(data["counts"][key])) for key, title in names.items()),
        _heading(distribution_title),
        _chart(distribution_title, order_buckets, [("Покупатели", counts)]),
        _table(scope, "customers.frequency", ("Количество заказов", "Покупатели", "Доля, %"),
               ((label, count, _share(count, total)) for label, count in zip(order_buckets, counts))),
        _heading("Первые покупки по месяцам"),
        _chart("Первые покупки по месяцам", [r[0] for r in first], [("Покупатели", [r[1] for r in first])], line=True),
        _table(scope, "customers.first_months", ("Месяц", "Покупатели"), first),
        _heading("Интервалы до второй покупки"),
        _chart("Интервалы до второй покупки", ("0 дней", "1–7 дней", "8–30 дней", "31–60 дней", "61–90 дней", "91+ дней"),
               [("Покупатели", _bucket_counts(second, (0, 7, 30, 60, 90)))]),
        _table(scope, "customers.second_intervals", ("Дни", "Покупатели"), second)]
    for field, title in (("first_categories", "Категории первой покупки"), ("second_categories", "Категории второй покупки")):
        blocks.extend(_purchase_categories(scope, field, title, data[field], labels))
    if not evidence:
        blocks.extend([_heading("Категории и сочетания в одном заказе"),
                       _table(scope, "customers.categories", ("Категория или сочетание", "Заказы", "Покупатели"),
                              ((" / ".join(row["categories"]), measure(row["orders"]), measure(row["buyers"]))
                               for row in (*data["categories"], *data["pairs"])))])
    activity = dto.customers["data"].get("customer_activity")
    if activity is not None:
        blocks.append(_heading("Покупательская активность"))
        blocks.extend([
                       _cards(
                              (("Среднее число заказов на покупателя", activity["order_mean"]),
                               ("Медиана числа заказов на покупателя", activity["order_median"]),
                               ("Доля повторных покупателей, %", activity["repeat_rate"]))),
                       _table("item_interactions", "customers.item_order_distribution",
                              ("Заказов", "Покупатели", "Доля, %"), activity["orders"])])
    if evidence:
        blocks.extend(_copurchase_blocks(scope, data, evidence, labels))
    if restored:
        intervals = restored["intervals"]
        last = restored["last_months"]
        blocks.extend([_heading("Частота и интервалы покупок"),
            _cards(
                (("Средний интервал между покупками, дней", intervals["mean_days"]),
                 ("Медианный интервал между покупками, дней", intervals["median_days"]))),
            _chart("Интервалы между покупками", [r[0] + " дней" for r in intervals["distribution"]],
                   [("Интервалы", [r[1] for r in intervals["distribution"]])]),
            _table(scope, "customers.intervals_distribution", ("Интервал", "Число интервалов", "Доля, %"),
                   intervals["distribution"]),
            _heading("Последняя покупка по месяцам"),
            _chart("Последняя покупка по месяцам", [r[0] for r in last], [("Покупатели", [r[1] for r in last])], line=True),
            _table(scope, "customers.last_months", ("Месяц", "Покупатели", "Доля, %"), last)])
    if dto.cohorts["channels"]:
        channels = [(row["name"], row["first_buyers"], row["second_buyers"])
                    for row in dto.cohorts["channels"]["channels"]]
        blocks.extend([_heading("Каналы первой и второй покупки"),
                       _top_purchases_chart("Каналы первой и второй покупки — TOP-10", channels,
                                            "Первые покупки", "Вторые покупки", height=520, bar_width=.8),
                       _table("cohorts_history", "customers.purchase_channels", ("Канал оформления", "Первые покупки", "Вторые покупки"),
                              channels)])
    return blocks


def _customer_money(dto):
    scope = "buyer_period"
    data = dto.customers["data"].get("buyer_breakdowns")
    if not data:
        return list(_currency_blocks(scope, "customers.money", dto.customers["data"]["purchases"]["buyer_money"]))
    blocks = []
    for currency in _CURRENCIES:
        finance = data["money"][currency]
        groups = finance["quintiles"]
        months = data["monthly_money"][currency]
        total = Decimal(finance["amount"])
        distribution = "Распределение покупателей по сумме покупок — " + currency
        edges = sorted({Decimal(edge) for row in groups for edge in row[:2]})
        places = 2
        if len({_tooltip_money(edge) for edge in edges}) < len(edges):
            places = max(len(format(edge, "f").partition(".")[2].rstrip("0")) for edge in edges)
        edge_text = {edge: report.number_text(_number(edge, places=places)).replace(".", ",") for edge in edges}
        bounds = [f"{edge_text[Decimal(a)]}–{edge_text[Decimal(b)]}" if Decimal(a) != Decimal(b)
                  else edge_text[Decimal(a)] for a, b, *_ in groups]
        summary = (
            ("Сумма покупок", _plain_money(total) if finance["known_orders"] else None),
            ("Средний чек", _plain_money(finance["mean_order"])),
            ("Средняя сумма покупок на покупателя", _plain_money(finance["mean_known_per_buyer"])),
            ("Медианная сумма покупок на покупателя", _plain_money(finance["median_known_per_buyer"])),
            ("Доля суммы покупок повторных покупателей, %", finance["repeat_amount_rate"]),
        )
        blocks.extend([_heading("Денежные показатели — " + currency),
            _table(scope, "customers.money." + currency, ("Денежный показатель", f"Значение, {currency}"),
                   ((label, value) for label, value in summary if value is not None)),
            _heading("Динамика суммы покупок — " + currency),
            _monthly_amount_chart(((month, amount) for month, _, _, amount in months), currency),
            _table(scope, "customers.money.monthly." + currency, ("Месяц", "Покупатели", "Заказы", f"Сумма покупок, {currency}"),
                   ((month, buyers, orders, _plain_money(amount)) for month, buyers, orders, amount in months)),
            _heading(distribution),
            _chart(distribution, bounds, [("Покупатели", [row[2] for row in groups])]),
            _table(scope, "customers.money.quintiles." + currency, (f"От, {currency}", f"До, {currency}", "Покупатели",
                "Доля, %", f"Сумма покупок, {currency}", "Доля общей суммы покупок, %", "Повторные покупатели",
                "Доля повторных, %", f"Сумма покупок повторных, {currency}", "Доля суммы покупок повторных, %"),
                ((_number(a, places=places), _number(b, places=places), n, rate, _plain_money(amount),
                  _number(Decimal(amount) * 100 / total, money=False, places=2) if total else None,
                  repeat, rr, _plain_money(ra), rar) for a, b, n, rate, amount, repeat, rr, ra, rar in groups))])
        for field, label in (("gender", "По полу"), ("age", "По возрасту")):
            if data.get("demographic_analytics"):
                rows = data["demographic_analytics"][field]
                blocks.extend([_heading(label + " — " + currency),
                    _table(scope, "customers.money.demography." + field + "." + currency, (label, "Покупатели",
                        "Заказы", "Повторные покупатели", f"Сумма покупок, {currency}",
                        f"Средний чек, {currency}", f"Медианный чек, {currency}"),
                        (((_age_label(r["label"]) if field == "age" else r["label"]),
                          r["buyers"], r["orders"], r["repeat_buyers"],
                          _plain_money(m["amount"]) if m["known_orders"] else None,
                          _plain_money(m["mean_order"]), _plain_money(m["median_order"]))
                         for r in rows for m in (r["money"][currency],)))])
    return blocks


_RFM_SEGMENT_DEFINITIONS = {
    "CHAMPIONS": "Не менее двух заказов в окне, R и F от 4 баллов, M равен 5.",
    "LOYAL": "Не менее двух заказов в окне, R и F от 3 баллов. Уровень расходов может быть умеренным.",
    "NEW_OR_RECENT": "Одна покупка в окне и R от 4 баллов. Покупка может быть повторной в истории клиента.",
    "AT_RISK": "R до 2 баллов. В окне либо повторные покупки при F от 3, либо до одного заказа после повторных покупок в более ранней истории этой валюты.",
    "HIBERNATING": "R до 2 баллов. F до 2 баллов или не более одного заказа в окне после проверки условий группы риска.",
    "OTHER": "Оценка рассчитана. Условия основных сегментов не выполнены.",
    "UNSCORABLE": "Доступных данных недостаточно для расчета хотя бы одного из баллов R, F или M.",
}


def _rfm_count_chart(title, labels, series):
    data = _chart(title, labels, series, height=285, bar_width=.7).data
    return replace(data, inner_title=True, value_axis_title="Покупателей", zero_is_data=False)


def _rfm_scores_chart_row(rows):
    charts = []
    for index, component in enumerate("RFM"):
        counts = [0] * 5
        has_scores = False
        for *scores, count in rows:
            score = scores[index]
            if score in range(1, 6):
                counts[score - 1] += count
                has_scores = True
        charts.append(_rfm_count_chart(f"Распределение баллов {component}", range(1, 6),
                                       [("Покупатели", counts if has_scores else [None] * 5)]))
    return report.ReportBlock("chart_row", data=tuple(charts))


def _rfm_distributions_chart_row(currencies):
    charts = []
    for key, title, labels, limits in (
        ("recency", "Распределение давности покупок", ("0–30", "31–90", "91–180", "181–365", "366–730", "731+"),
         (30, 90, 180, 365, 730)),
        ("frequency", "Распределение частоты покупок", ("0", "1", "2–3", "4–5", "6–10", "11+"),
         (0, 1, 3, 5, 10)),
    ):
        series = []
        for currency, row in currencies.items():
            distribution = row[key]["distribution"]
            totals = _bucket_counts(distribution, limits)
            observed = {bisect_left(limits, value) for value, _ in distribution}
            series.append((currency, [n if i in observed else None for i, n in enumerate(totals)]))
        if key == "frequency" and all(values[0] is None for _, values in series):
            labels = labels[1:]
            series = [(currency, values[1:]) for currency, values in series]
        if key == "recency":
            labels = tuple(f"{label} дн." for label in labels)
        data = _rfm_count_chart(title, labels, series)
        charts.append(data)
    return report.ReportBlock("chart_row", data=tuple(charts))


def _rfm_segment_money_chart(currency, segments, business):
    rows = [(SEGMENT_LABELS[code][0], _number(business[code]["known_subtotal"]),
             _number(business[code]["share_of_known_subtotal"])) for code, _, _ in segments]
    rows.sort(key=lambda row: (row[1] is None, row[1].value.copy_negate() if row[1] is not None else 0))
    title = f"Сумма покупок по RFM-сегментам — {currency}"
    block = _chart(title, [name for name, _, _ in rows],
                   [(f"Сумма покупок, {currency}", [amount for _, amount, _ in rows])],
                   horizontal=True, money=True, height=345)
    tips = tuple(f"{tip}\nДоля суммы: {report.cell_presentation(share, percentage=True).text}"
                 for tip, (_, _, share) in zip(block.data.tooltips[0], rows))
    block.data = replace(block.data, inner_title=True, tooltips=(tips,))
    return block


def _rfm(dto):
    scope = "rfm_48m"
    data = dto.rfm
    blocks = []
    if not data["windows"]:
        return [*blocks, report.ReportBlock("text", title="RFM недоступен: нет покрытия заказов.")]
    blocks.append(_cards([
        ("Идентифицированные клиенты", data["quality"]["population"]),
        ("Покупатели базы 48 месяцев", data["base_buyers"]),
        ("Покупатели обеих валют", data["quality"]["dual_currency_buyers"]),
    ]))
    choices = []
    quality_rows = []
    for window, payload in data["windows"].items():
        label = WINDOW_LABELS[window]
        for currency, row in payload["currencies"].items():
            labels = [SEGMENT_LABELS[code][0] for code, _, _ in row["segments"]]
            chart = _chart(f"RFM-сегменты — {label}, {currency}", labels,
                           [("Покупатели", [n for _, n, _ in row["segments"]])], horizontal=True)
            choices.append((window, currency, chart.data))
    blocks.extend([_heading("Распределение по RFM-сегментам"),
                   report.ReportBlock("rfm", choices=tuple(choices), options={"context_id": scope})])
    for window, payload in data["windows"].items():
        label = WINDOW_LABELS[window]
        distributions = []
        for currency, row in payload["currencies"].items():
            prefix = "rfm." + window + "." + currency
            blocks.append(_heading(f"{label} — {currency}"))
            blocks.append(_table(scope, prefix + ".metrics", ("Показатель R/F/M", "Значение"), [
                ("Валютная база", row["buyers"]), ("Активные покупатели", row["active_buyers"]),
                ("Заказы в окне", row["observed_orders"]),
                ("Покупатели с надежной давностью (R)", row["reliable_recency_buyers"]),
                ("Покупатели с надежной частотой (F)", row["reliable_frequency_buyers"]),
                ("Средняя наблюдаемая давность, дни", row["recency"]["mean"]),
                ("Медианная наблюдаемая давность, дни", row["recency"]["median"]),
                ("Средняя наблюдаемая частота, заказы", row["frequency"]["mean"]),
                ("Медианная наблюдаемая частота, заказы", row["frequency"]["median"]),
                ("Покупатели с рассчитанной M", row["monetary"]["buyers"]),
                (f"Средняя стоимость (M), {currency}", _plain_money(row["monetary"]["mean"])),
                (f"Медианная стоимость (M), {currency}", _plain_money(row["monetary"]["median"])),
                (f"Сумма покупок, {currency}", _plain_money(row["known_subtotal"])),
                ("Покупатели с недоступной M", row["unavailable_monetary_buyers"]),
            ]))
            if "segment_business" in row:
                blocks.append(_table(scope, prefix + ".segment_money", (f"Сегмент — {currency}", "Заказы в окне", "Заказы с суммой",
                    f"Сумма покупок, {currency}", "Доля суммы покупок, %", f"Средний чек, {currency}",
                    "Покупатели с рассчитанной M", f"Средняя M, {currency}"),
                    ((SEGMENT_LABELS[code][0], v["observed_orders"], v["valued_orders"], _plain_money(v["known_subtotal"]),
                      _plain_money(v["share_of_known_subtotal"]), _plain_money(v["mean_valued_order"]), v["complete_buyers"],
                      _plain_money(v["mean_complete_m"])) for code, v in row["segment_business"].items())))
                blocks.append(_rfm_segment_money_chart(currency, row["segments"], row["segment_business"]))
            blocks.extend([
                _table(scope, prefix + ".segments", (f"Сегмент — {currency}", "Покупатели", "Доля, %", "Что означает"),
                       ((SEGMENT_LABELS[code][0], n, share, _RFM_SEGMENT_DEFINITIONS[code]) for code, n, share in row["segments"])),
                _heading(f"Распределение баллов R/F/M — {currency}"),
                _rfm_scores_chart_row(row["score_counts"]),
                _table(scope, prefix + ".scores", (f"R — {currency}", "F", "M", "Покупатели"), row["score_counts"]),
            ])
            distributions.extend((currency, name, bucket, n)
                                 for key, name in (("recency", "Давность, дни"), ("frequency", "Частота, заказы"))
                                 for bucket, n in row[key]["distribution"])
            quality_rows.extend((label, currency, _REASONS.get(code, code), n) for code, n in row["reasons"].items())
        blocks.extend([_heading("Распределения давности и частоты"),
                       _rfm_distributions_chart_row(payload["currencies"]),
                       _table(scope, f"rfm.{window}.distributions",
                              ("Валюта", "Метрика", "Значение", "Покупатели"), distributions)])
    comparison = data["comparison"]
    groups = {"BOTH": "Покупки в обоих окнах", "LAST_ONLY": "Только в последних 12 месяцах",
              "PREVIOUS_ONLY": "Только в предыдущих 12 месяцах", "NEITHER": "Без покупок в обоих окнах"}
    blocks.extend([_heading("Сравнение последних и предыдущих 12 месяцев"),
        _table(scope, "rfm.comparison", ("Показатель", "Значение"),
            [(groups[key], count) for key, count in comparison["observed_groups"].items()] +
            [("Заказы в последних 12 месяцах", comparison["last_orders"]),
             ("Заказы в предыдущих 12 месяцах", comparison["previous_orders"]),
             ("Изменение числа заказов", comparison["order_change"]),
             ("Изменение числа заказов, %", comparison["order_change_percent"])]),
        _table(scope, "rfm.comparison.money", ("Валюта", "Сумма покупок сейчас", "Сумма покупок ранее",
            "Изменение суммы покупок", "Изменение, %", "Покупатели с неполной историей"),
            ((currency, *(_plain_money(row[key]) for key in ("last_known_amount", "previous_known_amount",
              "known_amount_change", "known_amount_change_percent")), row["unreliable_buyers"])
             for currency, row in comparison["money"].items())),
        ])
    if quality_rows:
        blocks.append(_details(_table(scope, "rfm.quality", ("Период", "Валюта", "Причина", "Покупатели"), quality_rows),
                               "Подробности оценки RFM"))
    return blocks


def _cohort_chart(title, labels, series, **options):
    return _heading(title), _chart(title, labels, series, **options)


def _cohort_retention_chart(heatmap):
    series = tuple((row.month, tuple(cell.value for cell in row.cells)) for row in heatmap.rows)
    tips = tuple(tuple(f"Когорта: {name}\n{month}\nУдержание: " +
                       ("—" if value is None else f"{value:.2f}".replace(".", ",") + " %")
                       for month, value in zip(heatmap.columns, values)) for name, values in series)
    title = "Динамика удержания когорт"
    return _heading(title), report.ReportBlock("chart", analytics=True, data=ChartData(
        title, "line", heatmap.columns, series, tips, zero_is_data=True, percentage=True,
        value_axis_title="Удержание, %", distinct_series=True, adaptive_percentage=True,
        height=370 + 20 * ((len(series) + 7) // 8)))


def _cohort_transitions_chart(title, transitions, names, first, second, same_label):
    top = sorted(transitions, key=lambda row: (-row["buyers"], row[first] or "", row[second] or ""))[:10]
    labels = [f"{names[row[first]]} → {names[row[second]]}" +
              (f" ({same_label})" if row[first] is not None and row[first] == row[second] else "")
              for row in top]
    return _cohort_chart(title, labels, [("Покупатели", [row["buyers"] for row in top])],
                         horizontal=True, wide_labels=True, height=410)


def _cohort_money(blocks, data):
    scope = "cohorts_history"
    labels = {"total": "Все заказы", "first": "Первые заказы", "repeat": "Повторные заказы"}
    for currency in _CURRENCIES:
        blocks.append(_heading(f"Денежные показатели когорт — {currency}"))
        rows = []
        months = sorted(data["cohorts"])
        series = {key: [] for key in ("first", "repeat")}
        for month, scopes in sorted(data["cohorts"].items()):
            for name, row in scopes.items():
                money = row["currencies"][currency]
                rows.append((month, labels[name], measure(money["recognized"], money=True),
                             row["orders"], money["valued_orders"], money["fully_valued_buyers"],
                             _plain_money(money["mean_valued_order"]), _plain_money(money["mean_per_valued_buyer"]),
                             _plain_money(money.get("valued_repeat_amount_rate"))))
            for name, values in series.items():
                values.append(measure(scopes[name]["currencies"][currency]["recognized"], money=True))
        blocks.extend(_cohort_chart(f"Суммы первых и повторных покупок — {currency}", months,
                                    [(f"{labels[k]}, {currency}", v) for k, v in series.items()],
                                    money=True, bar_width=.78, height=380))
        blocks.append(_table(scope, "cohorts.money." + currency, ("Когорта", "База", f"Сумма покупок, {currency}",
            "Заказы", "Заказы с суммой", "Покупатели с суммой", f"Средний чек, {currency}",
            f"Среднее на покупателя, {currency}", "Доля суммы повторных покупок, %"), rows))
        blocks.extend(_cohort_chart(f"Сумма повторных покупок по периодам — {currency}",
                      [f"{h} дней" for h in HORIZONS], [(f"Повторные покупки, {currency}",
                          [measure(data["horizons"][str(h)]["currencies"][currency]["recognized"], money=True)
                           for h in HORIZONS])], line=True, money=True))
        blocks.append(_table(scope, "cohorts.money.horizons." + currency,
            ("Горизонт, дни", "С полным наблюдением", "Повторные покупатели", f"Сумма покупок, {currency}",
             f"Средний чек, {currency}"),
            ((int(h), row["eligible"], row["buyers"], measure(row["currencies"][currency]["recognized"], money=True),
              _plain_money(row["currencies"][currency]["mean_valued_order"])) for h, row in data["horizons"].items())))


def _cohorts(dto):
    scope = "cohorts_history"
    data = dto.cohorts
    blocks = []
    if not data["summary"]:
        return [*blocks, report.ReportBlock("text", title="Когорты недоступны: нет покрытия заказов.")]
    summary = data["summary"]
    blocks.extend([_heading("Когорты первой покупки"),
        _cards((("Когорты", summary["cohort_count"]), ("Покупатели (M0)", summary["cohort_size"]),
                ("Со второй покупкой", summary["observed_repeat_buyers"])))])
    rows = [("Все когорты", summary), *sorted(data["months"].items())]
    repeat_table = _table(scope, "cohorts.repeat", ("Когорта", "Дней", "С полным наблюдением", "Повторили",
        
        "Наблюдаемые повторы вне базы", "Цензурировано", "Покрытие неполно", "Цензурировано с неполным покрытием",
        "Подтверждено без повтора", "Неопределенные в базе", "Доля, %"),
        ((month, h, cell["eligible"], cell["repeated"], cell["observed_repeated_ineligible"], cell["censored"],
          cell["coverage_incomplete"], cell["censored_and_incomplete"], cell["confirmed_without_event"],
          cell["uncertain_eligible_buyers"], measure(cell["rate_quality"], percentage=True))
         for month, row in rows for h in HORIZONS for cell in (row["horizons"][str(h)],)))
    heatmap = RetentionHeatmapData(tuple(RetentionRow(month, row["cohort_size"], tuple(
        RetentionCell(cell["retention_rate"], cell["status"] if cell["retention_rate"] is not None else "UNAVAILABLE",
                      cell["eligible"], cell["active"], cell["censored"], cell["coverage_incomplete"])
        for cell in row["retention"][1:])) for month, row in sorted(data["months"].items())))
    sizes = [(month, row["cohort_size"], row["observed_repeat_buyers"]) for month, row in rows[1:]]
    days = summary["second_purchase_days"]["distribution"]
    intervals = _bucket_counts(days, (0, 7, 30, 60, 90, 180))
    blocks.extend([_heading("Календарное удержание M0–M12"),
                   report.ReportBlock("heatmap", data=heatmap, options={"context_id": scope, "compact": True}),
                   *_cohort_retention_chart(heatmap),
                   *_cohort_chart("Первые и вторые покупки по когортам", [r[0] for r in sizes],
                                  [("Покупатели (M0)", [r[1] for r in sizes]), ("Со второй покупкой", [r[2] for r in sizes])],
                                  bar_width=.78, height=380),
                   _table(scope, "cohorts.sizes", ("Месяц первой покупки", "Покупатели (M0)", "Со второй покупкой"),
                          sizes),
                   *_cohort_chart("Интервалы между покупками", ("0 дней", "1–7 дней", "8–30 дней", "31–60 дней",
                                  "61–90 дней", "91–180 дней", "181+ дней"),
                                  [("Покупатели", intervals if days else [None] * len(intervals))]),
                   _table(scope, "cohorts.second_days", ("Дней между покупками", "Покупатели"), days),
                   *_cohort_chart("Повторные покупки по периодам", [f"{h} дней" for h in HORIZONS],
                                  [("Повторные покупатели", [measure(summary["horizons"][str(h)]["rate_quality"], percentage=True)
                                                            for h in HORIZONS])], percentage=True),
                   repeat_table,
                   _heading("Подробное календарное удержание"),
                   _table(scope, "cohorts.retention", ("Когорта", "Месяц", "С полным наблюдением", "Активны",
                       "Наблюдаемая активность вне базы", "Цензурировано", "Покрытие неполно", "Цензурировано с неполным покрытием",
                       "Подтверждено без активности", "Неопределенные в базе", "Удержание, %"),
                       ((month, f"M{cell['month_offset']}", cell["eligible"], cell["active"], cell["observed_active_ineligible"],
                         cell["censored"], cell["coverage_incomplete"], cell["censored_and_incomplete"],
                         cell["confirmed_without_event"], cell["uncertain_eligible_buyers"],
                         measure(cell["rate_quality"], percentage=True)) for month, row in rows for cell in row["retention"][1:]))])
    _cohort_money(blocks, data["money"])
    channels, categories = data["channels"], data["categories"]
    channel_names = {r["external_id"]: r["name"] for r in channels["channels"]}
    category_names = {r["code"]: r["name"] for r in categories["categories"]}
    blocks.extend([_heading("Каналы первой и второй покупки"),
        _table(scope, "cohorts.channels", ("Канал оформления", "Первые покупки", "Доля первых, %", "Вторые покупки", "Доля вторых, %"),
               ((row["name"], row["first_buyers"], row["first_rate"], row["second_buyers"], row["second_rate"]) for row in channels["channels"])),
        *_cohort_transitions_chart("Популярные переходы между каналами — TOP-10", channels["transitions"], channel_names,
                                   "first_external_id", "second_external_id", "тот же канал"),
        _table(scope, "cohorts.channel_transitions", ("Первый канал", "Второй канал", "Покупатели", "Доля, %", "Доля внутри первого, %"),
               ((channel_names[row["first_external_id"]], channel_names[row["second_external_id"]], row["buyers"],
                 row["repeat_rate"], row["first_channel_repeat_rate"]) for row in channels["transitions"])),
        _heading("Категории первой и второй покупки"),
        _table(scope, "cohorts.categories", ("Категория", "Путь", "Первые покупки", "Повторили", "Доля повторов, %",
                                             "Вторые покупки", "Доля вторых, %"),
               ((row["name"], " / ".join(row["path"] or ()), row["first_buyers"], row["repeat_buyers"], row["observed_repeat_rate"],
                 row["second_buyers"], row["second_rate"]) for row in categories["categories"])),
        *_cohort_transitions_chart("Популярные переходы между категориями — TOP-10", categories["transitions"], category_names,
                                   "first_code", "second_code", "та же категория"),
        _table(scope, "cohorts.transitions", ("Первая категория", "Вторая категория", "Покупатели", "Доля, %"),
               ((category_names[row["first_code"]], category_names[row["second_code"]], row["buyers"], row["repeat_rate"])
                for row in categories["transitions"])),
        _table(scope, "cohorts.quality", ("Качество когорт", "Количество"), [
            ("База покупателей", data["quality"]["base_buyers"]), ("Покупатели в показанных когортах", data["quality"]["displayed_buyers"]),
            ("Покупатели вне выбранных месяцев", data["quality"]["hidden_buyers"]),
            ("Первая покупка не установлена", data["quality"].get("first_purchase_unresolved", 0)),
            ("Вторая покупка не установлена", data["quality"].get("second_purchase_unresolved", 0)),
            ("Интервалы с неполным покрытием", summary["second_purchase_days"]["coverage_incomplete_buyers"]),
        ])])
    return blocks


def unified_presentation(dto, *, applied_filter=None):
    """The DTO must have passed the versioned reader before reaching presentation."""
    header = report_header(dto, applied_filter)
    customer_providers = (_overview, _purchases, _customer_money, _rfm, _cohorts)
    customer_groups = tuple(report.ReportGroup(key, title, tuple(provider(dto))) for key, title, provider in
                            zip(CUSTOMER_GROUP_ORDER, report.CUSTOMER_TAB_LABELS, customer_providers))
    pages = tuple(report.ReportPage(key, title, (report.ReportGroup(key, title, tuple(provider(dto))),))
                  for key, title, provider in zip(("actions", "orders", "products"), report.PAGE_LABELS,
                                                  (_actions, _orders, _products)))
    pages += (report.ReportPage("customers", report.PAGE_LABELS[3], customer_groups),)
    activity = dto.actions["data"]
    interactions = sum(activity[key]["known"] for key in _ACTIONS)
    demography = dto.customers["data"].get("customer_demography")
    cards = (("Взаимодействия", interactions),
             ("Уникальные заказы", measure(dto.orders["unique_orders"])),
             ("Позиции заказов", measure(dto.orders["positions"])),
             ("Профили клиентского среза", demography["profiles"] if demography else None))
    return report.StatisticsReport(header, cards, "Готово", pages,
                                   metadata=dict(policy=dto.policy, source=dto.source, contexts=dto.contexts,
                                                 versions=dto.metadata, schema_version=RESULT_SCHEMA),
                                   summary=_summary(header))
