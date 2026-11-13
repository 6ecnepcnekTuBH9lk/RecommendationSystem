"""Read-only report blocks shared by the UI, HTML and Excel."""

from datetime import datetime
from decimal import Decimal

from Application import statistics_report_presentation as report
from Application.statistics_chart_data import (
    ChartData, RetentionCell, RetentionHeatmapData, RetentionRow, _tooltip_money, exact_number,
)
from Application.statistics_cohorts_core import HORIZONS
from Application.statistics_presentation import CUSTOMER_GROUP_ORDER
from Application.statistics_rfm_presentation import SEGMENT_LABELS, WINDOW_LABELS

UNIFIED_ERRORS = {
    "INDEX_MISSING": "Индекс статистики INDEXED 4 отсутствует. Подготовьте статистические данные и повторите расчёт.",
    "INCOMPATIBLE_SCHEMA": "Поколение индекса несовместимо с INDEXED 4.",
    "STALE_SOURCE": "Источник изменился или индекс устарел. Проверьте выгрузки и повторите расчёт.",
    "SOURCE_UNAVAILABLE": "Данные Mindbox недоступны. Проверьте наличие выгрузок и справочников.",
    "INVALID_UNIFIED_INDEX_INTEGRITY": "Проверка целостности SQLite не пройдена.",
    "UNIFIED_INDEX_UNAVAILABLE": "Не удалось открыть проекцию INDEXED 4. Проверьте поколение, источник и права доступа.",
    "INVALID_MANIFEST": "Описание поколения INDEXED 4 повреждено или не соответствует указателю.",
    "GENERATION_CHANGED": "Поколение INDEXED 4 изменилось во время чтения. Повторите расчёт.",
    "INVALID_DEMOGRAPHICS": "Проверка целостности клиентской части индекса не пройдена.",
    "UNIFIED_CACHE_MISS": "Совместимого сохранённого отчёта нет. Проверьте источник и отбор или запустите расчёт.",
    "INSUFFICIENT_DISK": "Недостаточно свободного места для расчёта. Освободите место на диске проекта.",
    "INSUFFICIENT_MEMORY": "Недостаточно памяти для расчёта. Освободите ресурсы и повторите попытку.",
    "INVALID_UNIFIED_DTO": "Процесс вернул некорректный DTO 11.",
    "CACHE_WRITE_FAILED": "Не удалось прочитать или сохранить отчёт. Проверьте место и права доступа к кешу.",
    "INVALID_FILTER": "Отбор статистики некорректен. Проверьте выбранные значения.",
    "UNIFIED_FAILED": "Не удалось завершить расчёт единой статистики. Проверьте состояние источника.",
    "CANCELLED": "Расчёт отменён.",
}
_STORAGE_MESSAGE = "Защищённое хранилище недоступно. Проверьте права на рабочий каталог, кеш и индекс."
for _code in ("PRIVATE_INDEX_ACL_FAILED", "PRIVATE_INDEX_ROOT_REQUIRED", "UNSAFE_STORAGE_ACL",
              "UNSAFE_REPARSE_PATH", "UNTRUSTED_STORAGE_OWNER_OR_ACL", "OWNER_FULL_CONTROL_REQUIRED",
              "UNSUPPORTED_PRIVATE_DENY_ACL", "ACL_INSPECTION_FAILED", "UNSAFE_CACHE_FILE",
              "CACHE_ROOT_CHANGED", "CACHE_FILE_CHANGED", "STAGING_PATH_CHANGED"):
    UNIFIED_ERRORS[_code] = _STORAGE_MESSAGE



_PROXY = "Расчёт по датам оформления заказов, для которых наблюдались покупочные статусы."
_CURRENCIES = ("RUB", "KZT")
_GROUPS = {
    "COMPLETED": "Выполнено", "PARTIALLY_COMPLETED": "Частично выполнено",
    "PAID_OR_IN_PROGRESS": "Оплачено или в процессе", "ORDER_IN_PROGRESS": "В обработке",
    "DELIVERY_IN_PROGRESS": "Доставляется", "CANCELLED": "Отмена", "RETURNED": "Возвратное состояние",
    "UNKNOWN": "Не определена", "COMPLETED_ONLY": "Только выполненные", "PARTIAL_ONLY": "Только частично выполненные",
    "CANCELLED_ONLY": "Только отменённые", "RETURNED_ONLY": "Только возвратные",
    "IN_PROGRESS_ONLY": "Только в процессе", "MIXED": "Смешанное состояние", "EMPTY": "Без позиций",
    "UNAVAILABLE": "Недостаточно данных", "CONFLICT": "Конфликт снимков",
}
_REASONS = {
    "CP_ALLOCATION_UNKNOWN": "Неизвестна фактически признанная сумма частичного выполнения",
    "PRICE_MISSING": "Цена строки отсутствует", "PRICE_INVALID": "Цена строки непригодна",
    "STATUS_CONFLICT": "Снимки расходятся по статусу", "PRICE_CONFLICT": "Снимки расходятся по цене",
    "LEDGER_UNAVAILABLE": "Подтверждённый реестр денежных возвратов отсутствует",
    "FULFILLMENT_UNVERIFIED": "Фактически выполненное количество не подтверждено",
    "RETURNED_QUANTITY_UNVERIFIED": "Фактически возвращённое количество не подтверждено",
    "COVERAGE_GAP": "Период покрыт не полностью", "COVERAGE_GAP_OR_TRUNCATION": "Период покрыт не полностью",
    "ORDERS_COVERAGE_INCOMPLETE": "Недостаточное покрытие заказов", "RIGHT_CENSORED": "Период наблюдения ещё не завершён",
    "NO_ELIGIBLE_BUYERS": "Нет покупателей с полным периодом наблюдения",
    "NO_FULLY_VALUED_BASE": "Нет полностью оценённой базы", "LINES_INCOMPLETE": "Данные о позициях неполны",
    "CREATED_TIME_CONFLICT": "Снимки расходятся по дате оформления", "CREATED_TIME_MISSING": "Дата оформления отсутствует",
    "ORDER_DATE_UNRESOLVED": "Дата оформления не установлена", "OWNER_UNRESOLVED": "Покупатель не установлен",
    "MIXED_ORDER_CURRENCY": "В заказе несколько валют", "CURRENCY_UNKNOWN": "Валюта не установлена",
    "ATTRIBUTE_MISSING": "Реквизит отсутствует", "ATTRIBUTE_INVALID": "Реквизит невалиден",
    "ATTRIBUTE_CONFLICT": "Снимки расходятся по реквизиту",
    "DELIVERY_CURRENCY_UNVERIFIED": "Валюта доставки по всему заказу не установлена",
}
_QUALITY = {"KNOWN": "Измерено", "UNKNOWN": "Неизвестно", "INVALID": "Невалидные данные",
            "CONFLICT": "Конфликт данных", "INSUFFICIENT_COVERAGE": "Недостаточно данных"}
_COVERAGE = {"FULL": "Полное", "PARTIAL": "Частичное", "NONE": "Отсутствует",
             "UNAVAILABLE": "Недостаточно данных", "NO_BUYERS": "Нет покупателей"}
_CONTEXT_LABELS = {
    "raw_orders": "Исходные Orders", "order_product_slice": "Заказы и товары в отборе",
    "item_interactions": "Взаимодействия в отборе", "customer_population": "Общая клиентская база",
    "buyer_period": "Покупатели периода, все товары", "rfm_48m": "RFM, все товары",
    "cohorts_history": "Когорты по истории, все товары",
}
_ACTIONS = {"VIEW": "Просмотры", "FAVORITE": "Добавления в избранное", "PURCHASE": "Покупочные позиции"}
_MONEY = {
    "source_nominal": "Исходный номинал", "recognized_purchase": "Признанная стоимость",
    "known_purchase_subtotal": "Известный подытог покупок", "partial_nominal": "Номинал частичных выполнений (CP)",
    "cancelled_nominal": "Номинал отменённых позиций", "returned_nominal": "Номинал возвратных позиций",
    "confirmed_refund": "Подтверждённые денежные возвраты", "net_after_refunds": "Сумма после денежных возвратов",
}
_SCOPE_NOTE = "Товарный отбор применяется к заказам, действиям и товарам. Покупатели периода, RFM и когорты используют все товары."


def _number(value, *, money=True):
    return None if value is None else report.ReportNumber(Decimal(str(value)), money=money)


def measure(cell, *, money=False, quantity=False, percentage=False):
    value, known = cell["value"], cell.get("known")
    quality, reasons = cell["quality"], tuple(cell["reasons"])
    if money or quantity or percentage:
        value, known = (_number(v, money=not quantity) for v in (value, known))
    tooltip = _QUALITY[quality]
    if reasons:
        tooltip += "\n" + "\n".join(_REASONS.get(reason, "Код причины: " + reason) for reason in reasons)
    if cell.get("known") is not None and cell["value"] is None:
        text = report.cell_presentation(known, percentage=percentage).text.replace(".", ",")
        tooltip += "\nИзвестная часть: " + text
    return report.ReportMeasure(value, known, quality, reasons, tooltip)


def _plain_money(value):
    number = _number(value)
    return report.ReportMeasure(number, number, "KNOWN") if number is not None else None


def _stamp(value):
    return datetime.fromisoformat(value).strftime("%d.%m.%Y %H:%M") + " UTC" if value else "—"


def _period(value):
    return " → ".join(map(_stamp, value)) + " (конец не включён)" if value else "Вся доступная область"


def _heading(title):
    return report.ReportBlock("heading", title=title)


def _text(text):
    return report.ReportBlock("text", title=text, options={"context": True})


def _table(context, key, headers, rows):
    return report.ReportBlock("table", headers=tuple(headers), rows=tuple(tuple(row) for row in rows),
                              table_id="unified." + key, options={"context_id": context})


def _chart(title, labels, series, *, horizontal=False, percentage=False):
    def cells(values):
        return tuple(report.cell_presentation(v, percentage=percentage) for v in values)
    rendered = [(name, cells(values)) for name, values in series]
    chart = ChartData(title, "horizontal" if horizontal else "bar", tuple(map(str, labels)),
        tuple((name, tuple(cell.value for cell in values)) for name, values in rendered),
        tuple(tuple(f"{label}\n{name}: {cell.text}" for label, cell in zip(labels, values))
              for name, values in rendered),
        zero_is_data=True, percentage=percentage)
    return report.ReportBlock("chart", data=chart, analytics=True)


def _scope(dto, context):
    return _text(_CONTEXT_LABELS[context] + " · " + _period(dto.contexts[context]["period"]))


def _action_breakdowns(dto):
    data = dto.actions.get("breakdowns")
    if data is None:
        return []
    scope = "item_interactions"
    rows = [("Просмотры", data["mean_views_per_viewer"], data["median_views_per_viewer"]),
            ("Избранное", data["mean_favorites_per_user"], data["median_favorites_per_user"])]
    blocks = [_heading("Активность по наблюдаемым действиям"),
              _text("Действия учитываются за выбранный период независимо от покрытия заказов. " + _period(data["period"])),
              _table(scope, "actions.user_statistics", ("Действие", "Среднее на клиента", "Медиана на клиента"), rows)]
    for key, title in (("view", "Распределение просмотров на клиента"),
                       ("favorite", "Распределение добавлений в избранное на клиента")):
        distribution = data[key + "_user_activity_distribution"]
        blocks.extend([_heading(title), _chart(title, [r[0] for r in distribution], [("Клиенты", [r[1] for r in distribution])]),
                       _table(scope, "actions." + key + "_distribution", ("Действий", "Клиенты", "Доля, %"), distribution)])
    monthly = data["action_monthly_dynamics"]
    for title, count, people in (("Динамика просмотров", 1, 2), ("Динамика избранного", 3, 4)):
        blocks.extend([_heading(title), _chart(title, [r[0] for r in monthly], [
            ("Действия", [r[count] for r in monthly]), ("Клиенты", [r[people] for r in monthly])])])
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
    if "event_names" in data:
        blocks.extend([_heading("Системные имена наблюдаемых действий"),
                       _table(scope, "actions.event_names", ("Тип", "Системное имя", "Исходные события"),
                              ((_ACTIONS.get(kind, "Неклассифицировано"), name, count) for kind, name, count in data["event_names"]))])
    return blocks


def _actions(dto):
    scope = "item_interactions"
    data = dto.actions["data"]
    rows = [(title, measure(data[key]), measure(data[key + "_users"])) for key, title in _ACTIONS.items()]
    intersections = {
        "VIEW_PURCHASE": "Просмотры и покупки", "FAVORITE_PURCHASE": "Избранное и покупки",
        "VIEW_FAVORITE": "Просмотры и избранное", "VIEW_FAVORITE_PURCHASE": "Все три типа",
    }
    return [_scope(dto, scope), _heading("Активность клиентов"),
            _chart("Действия в товарном срезе", [r[0] for r in rows], [("Взаимодействия", [r[1] for r in rows])]),
            _table(scope, "actions.totals", ("Тип действия", "Взаимодействия", "Клиенты"), rows),
            _heading("Пересечения действий"),
            _chart("Пересечения действий", list(intersections.values()),
                   [("Клиенты", [measure(data[key]) for key in intersections])], horizontal=True),
            _table(scope, "actions.intersections", ("Пересечение", "Уникальные клиенты"),
                   ((label, measure(data[key])) for key, label in intersections.items())),
            _text("Покупочные взаимодействия соответствуют наблюдаемым покупочным позициям после согласования снимков."),
            *_action_breakdowns(dto)]


def _currency_blocks(context, key, currencies):
    for currency in _CURRENCIES:
        row = currencies[currency]
        yield _heading(f"Денежные показатели — {currency}")
        yield _table(context, key + "." + currency, ("Денежный показатель", f"Значение, {currency}"),
                     [(name, measure(row["amounts"][metric], money=True)) for metric, name in _MONEY.items()] +
                     [("Средняя сумма оценённого заказа", measure(row["mean_order"], money=True)),
                      ("Средняя сумма на полностью оценённого покупателя",
                       measure(row["mean_per_valued_buyer"], money=True) if row["mean_per_valued_buyer"] else None)])
        yield _table(context, key + ".base." + currency, ("База оценки", "Количество"), [
            ("Оценённые заказы", row["valued_orders"]), ("Неоценённые заказы", row["unvalued_orders"]),
            ("Оценённые позиции", row["valued_positions"]), ("Неоценённые позиции", row["unvalued_positions"]),
            ("Полностью оценённые покупатели", row["fully_valued_buyers"]), ("Неоценённые покупатели", row["unvalued_buyers"]),
        ])


def _store_chart(stores, currency):
    rows = [r for r in stores if r["known_subtotal"] is not None]
    rows.sort(key=lambda r: (Decimal(r["known_subtotal"]).copy_negate(), r["channel"] or "", r["name"]))
    rows = rows[:10]
    names = [r["name"] for r in rows]
    labels = tuple(name if names.count(name) == 1 else f"{name} [{r['channel'] or 'не указан'}]"
                   for name, r in zip(names, rows))

    def money(amount):
        return f"{_tooltip_money(amount)} {currency}" if amount is not None else "Недоступно"

    tips = tuple(f"Магазин / канал: {r['name']}\nИзвестный подытог: {money(r['known_subtotal'])}\n"
                 f"Заказов: {exact_number(r['orders'])}\nПокупателей: {exact_number(r['buyers'])}\n"
                 f"Средняя сумма оценённого заказа: {money(r['mean_valued_order']['value'])}"
                 for r in rows)
    data = ChartData(f"Магазины и каналы — {currency}", "horizontal", labels,
                     ((currency, tuple(Decimal(r["known_subtotal"]) for r in rows)),), (tips,),
                     ranked=True, money=True, secondary=currency == "KZT", zero_is_data=True)
    return report.ReportBlock("chart", data=data)


def _order_breakdowns(dto):
    data = dto.orders.get("breakdowns")
    if data is None:
        return []
    scope = "order_product_slice"
    rows = data["basket_distribution"]
    blocks = [_heading("Корзина заказа"),
              _chart("Состав корзины", [r[0] for r in rows], [("Заказы", [r[1] for r in rows])]),
              _table(scope, "orders.basket", ("Количество позиций", "Заказы", "Доля оценённых, %"), rows),
              _table(scope, "orders.basket_summary", ("Показатель", "База", "Среднее", "Медиана"),
                     ((label, row["evaluated"], _number(row["mean"], money=False), _number(row["median"], money=False))
                      for label, row in (("Позиции", data["basket"]), ("Номинальные единицы", data["nominal_units"]))))]
    if "currencies" in data:
        blocks.extend([_heading("Финансовые показатели покупочных заказов"),
                       _table(scope, "orders.financials", ("Валюта", "Заказы", "Позиции", "Известные номинальные единицы",
                              "Оценённые заказы", "Признано", "Среднее по оценённым", "Медиана по оценённым"),
                              ((currency, r["orders"], r["positions"], _number(r["nominal_units"], money=False),
                                r["valued_orders"], measure(dict(r["recognized"], known=r["known_subtotal"]), money=True),
                                measure(r["mean_valued_order"], money=True), _plain_money(data["median_check"][currency]["median"]))
                               for currency, r in data["currencies"].items()))])
    if data["basket_reasons"]:
        blocks.append(_text("Не все корзины измерены. " + "; ".join(_REASONS.get(x, x) for x in data["basket_reasons"])))
    for title, field in (("Способ оформления", "ordering_methods"), ("Способ оплаты", "payment_types"), ("Доставка", "delivery_types")):
        rows = data[field]
        blocks.extend([_heading(title), _chart(title, [r[0] for r in rows], [("Заказы", [r[1] for r in rows])]),
                       _table(scope, "orders." + field, (title, "Заказы", "Доля, %"), rows)])
    for currency in _CURRENCIES:
        n = data["median_check"][currency]
        delivery = data["delivery"][currency]
        months = [r for r in data["months"] if r["currency"] == currency]
        stores = [r for r in data["stores"] if r["currency"] == currency]

        def recognized(row):
            return measure(dict(row["recognized"], known=row["known_subtotal"]), money=True)

        blocks.extend([_heading("Чек и стоимость доставки — " + currency), _table(scope, "orders.check_delivery." + currency,
            ("Показатель", "Оценённые заказы", "Известный подытог", "Среднее по оценённым", "Медиана по оценённым"),
            ((label, value["evaluated"], *(_plain_money(value[x]) for x in ("subtotal", "mean", "median")))
             for label, value in (("Признанный чек", n), ("Стоимость доставки", delivery)))),
            *([_text("Стоимость доставки. " + ". ".join(_REASONS.get(x, x) for x in delivery["reasons"]))]
              if delivery["reasons"] else []),
            _heading("Динамика покупок — " + currency),
            _chart("Покупочные заказы — " + currency, [r["month"] for r in months], [("Заказы", [r["orders"] for r in months])]),
            _chart("Признанная стоимость — " + currency, [r["month"] for r in months], [(currency, [recognized(r) for r in months])]),
            _table(scope, "orders.monthly." + currency, ("Месяц", "Заказы", "Оценено", "Не оценено", f"Признано, {currency}"),
                   ((r["month"], r["orders"], r["valued_orders"], r["unvalued_orders"], recognized(r)) for r in months)),
            _heading("Магазины и каналы — " + currency),
            _store_chart(stores, currency),
            _table(scope, "orders.stores." + currency, ("Канал", "Заказы", "Покупатели", "Позиции", "Номинальные единицы",
                "Оценено", "Не оценено", f"Признано, {currency}", f"Среднее по оценённым, {currency}"),
                ((r["name"], r["orders"], r["buyers"], r["positions"], _number(r["nominal_units"], money=False),
                  r["valued_orders"], r["unvalued_orders"], recognized(r), measure(r["mean_valued_order"], money=True))
                 for r in stores))])
    return blocks


def _orders(dto):
    scope = "order_product_slice"
    data = dto.orders
    values = {"unique_orders": "Уникальные заказы", "positions": "Позиции заказов", "purchase_orders": "Покупочные заказы",
              "purchase_positions": "Покупочные позиции", "buyers": "Покупатели среза"}
    indicators = {"partial_orders": "Заказы с CP", "partial_positions": "Частично выполненные позиции (CP)",
                  "in_progress_positions": "Позиции в процессе", "orders_without_evidence": "Заказы без покупочного свидетельства",
                  "unresolved_purchase_orders": "Неопределённая квалификация покупки", "mixed_slice_orders": "Смешанные заказы в срезе",
                  "conflicting_orders": "Заказы с конфликтом", "unknown_state_orders": "Заказы с неизвестным состоянием",
                  "incomplete_structure_orders": "Заказы с неполным составом", "unresolved_scope_orders": "Принадлежность к срезу неизвестна"}
    blocks = [_scope(dto, scope), _heading("Заказы и покупки"),
              _table(scope, "orders.main", ("Показатель", "Значение"),
                     [(title, measure(data[key])) for key, title in values.items()] +
                     [(title, measure(data["indicators"][key])) for key, title in indicators.items()] +
                     [("Смешанные заказы (весь заказ)", measure(data["mixed_states"]["orders"])),
                      ("Доля заказов с неизвестным состоянием, %", measure(data["mixed_states"]["unresolved_state_share"], percentage=True)),
                      ("Покупочные заказы без однозначной валюты", measure(dto.money["unassigned_currency_orders"]))]),
              _heading("Состояния заказов"),
              _table(scope, "orders.states", ("Состояние", "Заказы"),
                     ((_GROUPS[key], measure(value)) for key, value in data["state_distribution"].items()))]
    for key, title, only in (("cancellations", "Отмены", "Полностью отменённые заказы"),
                              ("return_states", "Возвратные состояния", "Заказы только с возвратными позициями")):
        row = data[key]
        counts = {"positions": "Позиции", "orders": "Заказы", "only_orders": only,
                  "mixed_orders": "Смешанные заказы", "customers": "Клиенты", "returned_units": "Фактически возвращённые единицы"}
        blocks.extend([_heading(title), _table(scope, "orders." + key, ("Показатель", "Значение"),
                       ((label, measure(row[k], quantity=k == "returned_units")) for k, label in counts.items() if k in row))])
    statuses = dto.status_catalog["rows"]
    position_base = data["positions"]["known"]
    for catalog, title, key in ((True, "Каталог статусов Mindbox", "catalog"), (False, "Неизвестные и невалидные статусы", "status_quality")):
        labels = {"UNKNOWN_CODE": "Неизвестный код", "MISSING_CODE": "Код отсутствует",
                  "INVALID_CODE": "Невалидный код", "UNRESOLVED": "Статус не согласован"}
        blocks.extend([_heading(title), _table(scope, "orders." + key,
            ("Название", "External ID", "Категория Mindbox", "Бизнес-группа", "Позиции",
             "Уникальные заказы", "Доля наблюдаемых позиций, %", "Покупочный статус"),
            ((r["name"] or ("Официальное название не подтверждено" if catalog else labels[r["bucket"]]),
              r["external_id"], r["mindbox_category"], _GROUPS[r["business_group"]], measure(r["positions"]), measure(r["orders"]),
              100 * r["positions"]["known"] / position_base if position_base else None,
              "Да" if r["external_id"] in ("F", "sold", "CP") else "Нет" if r["bucket"] == "CATALOG" else None)
             for r in statuses if (r["bucket"] == "CATALOG") == catalog))])
    blocks.extend(_currency_blocks(scope, "orders.money", dto.money["currencies"]))
    blocks.extend(_order_breakdowns(dto))
    return blocks


def _products(dto):
    scope = "order_product_slice"
    data = dto.products["data"]
    keys = ("purchase_orders", "purchase_positions", "buyers", "nominal_units", "fulfilled_units",
            "partial_positions", "cancelled_positions", "returned_positions")
    rows = [(r["item_id"], r["name"], r["category"], r["views"], r["favorites"],
             *(measure(r[key], quantity=key in ("nominal_units", "fulfilled_units")) for key in keys)) for r in data]
    blocks = [_scope(dto, scope), _heading("Взаимодействия и покупки по товарам"),
              _text("По товарам показаны наблюдаемые просмотры и добавления в избранное."),
              _table(scope, "products.quality", ("Показатель", "Количество"), [
                  ("Товары с наблюдаемыми покупками", dto.diagnostics["counters"]["products_with_purchases"]),
                  ("Покупочные позиции без сопоставленного товара", dto.diagnostics["counters"].get("unresolved_purchase_items", 0)),
              ])]
    details = dto.products.get("breakdowns") or {}
    for name, label, column, members, companion in (
            ("viewed", "Просмотры", 3, "viewers", 4), ("favorited", "Избранное", 4, "favorite_users", 3),
            ("purchased", "Покупочные позиции", 6, None, 7)):
        def value(row):
            cell = report.cell_presentation(row[column])
            return cell.value if cell.value is not None else cell.known
        top = sorted((row for row in rows if value(row)), key=lambda row: (-value(row), row[0]))[:15]
        if top:
            blocks.append(_chart(label + " · первые 15 товаров", [r[1] or r[0] for r in top],
                                 [(label, [r[column] if report.cell_presentation(r[column]).value is not None
                                            else value(r) for r in top])], horizontal=True))
            if members in details:
                blocks.append(_table(scope, "products.top." + name, ("Код", "Товар", label,
                              "Уникальные клиенты", "Избранное" if column == 3 else "Просмотры", "Покупочные позиции"),
                             ((r[0], r[1], r[column], details[members][r[0]], r[companion], r[6]) for r in top)))
            elif name == "purchased":
                blocks.append(_table(scope, "products.top." + name,
                              ("Код", "Товар", label, "Покупатели", "Номинальные единицы", "Просмотры", "Избранное"),
                              ((r[0], r[1], r[column], r[companion], r[8], r[3], r[4]) for r in top)))
    blocks.append(_table(scope, "products.items", ("Код товара", "Товар", "Прямая категория", "Просмотры", "Избранное",
        "Покупочные заказы", "Покупочные позиции", "Покупатели", "Исходные единицы", "Выполненные единицы",
        "Частичные позиции", "Отменённые позиции", "Позиции с возвратом"), rows))
    for currency in _CURRENCIES:
        fields = ("source_nominal", "recognized_purchase", "known_purchase_subtotal", "partial_nominal",
                  "cancelled_nominal", "returned_nominal")
        blocks.extend([_heading(f"Стоимость позиций товаров — {currency}"), _table(scope, "products.money." + currency,
            ("Код товара", "Товар", *(_MONEY[key] + f", {currency}" for key in fields), "Неоценённые позиции"),
            ((r["item_id"], r["name"], *(measure(r["money"][currency]["amounts"][key], money=True) for key in fields),
              r["money"][currency]["unvalued_positions"]) for r in data))])
    detail = dto.products.get("breakdowns")
    if detail is not None:
        blocks.extend([_heading("Уникальные клиенты по товарам"), _table(scope, "products.unique_action_users",
            ("Код товара", "Товар", "Клиенты с просмотрами", "Клиенты с избранным"),
            ((row["item_id"], row["name"], detail["viewers"][row["item_id"]], detail["favorite_users"][row["item_id"]]) for row in data))])
        for field, label in (("category", "Прямые категории"), ("gender", "Пол товара"),
                             ("collection", "Сезон"), ("style_group", "Стилевая группа")):
            grouped = detail["groups"][field]
            headers = (label, "Товары", "Просмотры", "Избранное", "Покупочные позиции", "Известные номинальные единицы")
            if field == "category":
                names = detail.get("category_labels", {})
                category_rows = [(r[0], *names.get(r[0], (r[0], None)), *r[1:]) for r in grouped]
                rows = [(code, name, " / ".join(path or ()), *counts[:-1], _number(counts[-1], money=False))
                        for code, name, path, *counts in category_rows]
                chart_labels = [f"{name} [{code}]" for code, name, *_ in category_rows]
                headers = ("Код категории", "Категория", "Путь", *headers[1:])
            else:
                rows = [(*r[:-1], _number(r[-1], money=False)) for r in grouped]
                chart_labels = [r[0] for r in grouped]
            blocks.extend([_heading(label), _chart(label, chart_labels, [("Покупочные позиции", [r[4] for r in grouped])]),
                           _table(scope, "products.group." + field, headers, rows)])
        if "resolution" in detail:
            resolution = detail["resolution"]
            blocks.extend([_heading("Качество сопоставления"), _table(scope, "products.resolution",
                           ("Показатель", "Значение"), [
                               ("Уникальные исходные идентификаторы товаров", resolution["source_products"]),
                               ("Уникальные распознанные товары", resolution["resolved_products"]),
                               ("Товары с просмотрами", sum(n > 0 for n in detail["viewers"].values())),
                               ("Товары в избранном", sum(n > 0 for n in detail["favorite_users"].values())),
                               ("Распознанные взаимодействия", resolution["resolved"]),
                               ("Нераспознанные взаимодействия", resolution["unresolved"]),
                               ("Доля распознанных, %", resolution["rate"]),
                           ])])
        if "namespaces" in detail:
            blocks.extend([_heading("Сопоставление идентификаторов"),
                           _table(scope, "products.namespaces",
                                  ("Система идентификаторов", "Взаимодействия", "Распознано", "Не распознано", "Доля, %"),
                                  (("Не определена" if name == "unsupported" else name, *values)
                                   for name, *values in detail["namespaces"]))])
    return blocks


def _overview(dto):
    scope = "customer_population"
    data = dto.customers["data"]["counts"]
    buyer = dto.customers["data"]["purchases"]["counts"]
    names = {"population": "Идентифицированные клиенты", "population_buyers": "Клиенты с покупками",
             "nonpurchase_clients": "Клиенты с заказами без покупочного свидетельства",
             "cancelled_clients": "Клиенты с отменами", "returned_clients": "Клиенты с возвратными состояниями"}
    demo = dto.customers["data"].get("customer_demography")
    blocks = [_scope(dto, scope), _heading("Клиентская база"),
            _table(scope, "customers.population", ("Показатель", "Клиенты"),
                   ((title, measure(data[key])) for key, title in names.items())),
            _scope(dto, "buyer_period"), _heading("Покупатели и качество идентификации"),
            _table("buyer_period", "customers.buyer_identity", ("Показатель", "Значение"), [
                ("Покупатели", measure(buyer["buyers"])), ("Одна покупка", measure(buyer["single_buyers"])),
                ("Повторные покупатели", measure(buyer["repeat_buyers"])),
                ("Заказы без атрибуции покупателю", measure(buyer["unattributed_orders"])),
                ("Заказы с конфликтом владельца", measure(buyer["owner_conflict_orders"])),
            ])]
    if demo and demo["profiles"] is not None:
        blocks.extend([_heading("Демография по профилям клиентского среза"),
                       _table(scope, "customers.profile_summary", ("Показатель", "Значение"),
                              (("Профили", demo["profiles"]), ("Средний возраст", demo["mean_age"]),
                               ("Медианный возраст", demo["median_age"])))])
        for key, label in (("gender", "Пол"), ("age", "Возрастная группа")):
            rows = demo[key]
            blocks.extend([_chart(label, [r[0] for r in rows], [("Клиенты", [r[1] for r in rows])]),
                           _table(scope, "customers.demography." + key, (label, "Профили", "Доля, %"), rows)])
    elif demo:
        blocks.append(_text("Демография недоступна: нет закреплённой выгрузки Customers."))
    activity = dto.customers["data"].get("customer_activity")
    if activity is not None:
        item_scope = "item_interactions"
        rows = activity["interactions"]
        blocks.extend([_scope(dto, item_scope), _heading("Активность клиентов товарного среза"),
                       _table(item_scope, "customers.activity",
                              ("Показатель", "Значение"), [
                                  ("Активные клиенты", activity["active"]),
                                  ("Повторные покупатели", activity["repeat_buyers"]),
                                  *((_ACTIONS[key] if key in _ACTIONS else name, measure(dto.actions["data"][key]))
                                    for key, name in (("VIEW_users", "Клиенты с просмотрами"),
                                                      ("FAVORITE_users", "Клиенты с избранным"),
                                                      ("PURCHASE_users", "Клиенты с покупками"),
                                                      ("VIEW_PURCHASE", "Клиенты с просмотрами и покупками"),
                                                      ("FAVORITE_PURCHASE", "Клиенты с избранным и покупками"),
                                                      ("VIEW_FAVORITE_PURCHASE", "Клиенты со всеми типами взаимодействий"))),
                                  ("Среднее взаимодействий на клиента", activity["mean"]),
                                  ("Медиана взаимодействий на клиента", activity["median"]),
                                  ("Доля активных клиентов с покупками, %", activity["buyer_rate"])]),
                       _chart("Распределение активности", [r[0] for r in rows], [("Клиенты", [r[1] for r in rows])]),
                       _table(item_scope, "customers.activity_distribution", ("Взаимодействий", "Клиенты", "Доля, %"), rows)])
    return blocks


def _purchases(dto):
    scope = "buyer_period"
    data = dto.customers["data"]["purchases"]
    names = {"buyer_purchase_orders": "Покупочные заказы периода", "buyers": "Покупатели",
             "single_buyers": "Одна покупка", "repeat_buyers": "Повторные покупатели"}
    frequency = data["frequency"]
    blocks = [_scope(dto, scope), _text(_PROXY), _heading("Покупки в периоде"),
        _table(scope, "customers.purchases", ("Показатель", "Значение"),
               ((title, measure(data["counts"][key])) for key, title in names.items())),
        _heading("Распределение по числу наблюдаемых заказов"),
        _chart("Частота наблюдаемых покупок", [r[0] for r in frequency], [("Покупатели", [r[1] for r in frequency])]),
        _table(scope, "customers.frequency", ("Заказов", "Покупателей"), frequency)]
    for field, title, column in (("first_months", "Первые наблюдаемые покупки в выбранном периоде", "Месяц"),
            ("second_intervals", "Интервалы до второй покупки", "Дни"),
            ("first_categories", "Категории первой покупки в периоде", "Категория"),
            ("second_categories", "Категории второй покупки в периоде", "Категория")):
        blocks.extend([_heading(title), _table(scope, "customers." + field, (column, "Покупателей"), data[field])])
    blocks.extend([_heading("Прямые категории и сочетания в одном заказе"),
                   _table(scope, "customers.categories", ("Категория или сочетание", "Заказы", "Покупатели"),
                          ((" / ".join(row["categories"]), measure(row["orders"]), measure(row["buyers"]))
                           for row in (*data["categories"], *data["pairs"])))])
    activity = dto.customers["data"].get("customer_activity")
    if activity is not None:
        blocks.extend([_scope(dto, "item_interactions"), _heading("Покупательская активность отобранных товаров"),
                       _table("item_interactions", "customers.item_order_distribution",
                              ("Заказов", "Покупатели", "Доля, %"), activity["orders"]),
                       _table("item_interactions", "customers.item_order_frequency", ("Показатель", "Значение"),
                              (("Среднее число заказов на покупателя", activity["order_mean"]),
                               ("Медиана числа заказов на покупателя", activity["order_median"]),
                               ("Доля повторных покупателей, %", activity["repeat_rate"])))])
    restored = dto.customers["data"].get("buyer_breakdowns")
    if restored:
        if "copurchases" in restored:
            evidence = restored["copurchases"]
            labels = evidence["labels"]
            categories = {row["categories"][0]: row for row in data["categories"]}

            def title(code):
                name, path = labels.get(code, (code, None))
                return " / ".join(path) if path else name

            def share(num, den):
                return 100 * num["value"] / den["value"] if num["value"] is not None and den["value"] else None

            blocks.extend([
                _heading("Совместные покупки прямых категорий"),
                _table(scope, "customers.copurchase_base", ("Показатель", "Значение"), (
                    ("Заказы в области наблюдения", measure(evidence["orders"])),
                    ("Заказы с двумя или более подтверждёнными категориями", measure(evidence["multicategory_orders"])),
                    ("Покупатели с многокатегорийными заказами", measure(evidence["multicategory_buyers"])),
                    ("Доля многокатегорийных заказов, %", share(evidence["multicategory_orders"], evidence["orders"])),
                    ("Заказы с неполным сопоставлением категорий", evidence["unresolved_category_orders"]),
                    ("Наблюдаемые пары категорий", len(data["pairs"])),
                )),
                _table(scope, "customers.copurchase_categories", ("Категория", "Заказы", "Покупатели", "Доля заказов, %"),
                       ((title(code), measure(row["orders"]), measure(row["buyers"]), share(row["orders"], evidence["orders"]))
                        for code, row in categories.items())),
                _table(scope, "customers.copurchase_pairs", ("Категория 1", "Категория 2", "Заказы вместе",
                       "Покупатели", "Доля заказов, %", "Категория 2 среди заказов с 1, %",
                       "Категория 1 среди заказов с 2, %"),
                       ((title(a), title(b), measure(row["orders"]), measure(row["buyers"]), share(row["orders"], evidence["orders"]),
                         share(row["orders"], categories[a]["orders"]), share(row["orders"], categories[b]["orders"]))
                        for row in data["pairs"] for a, b in (row["categories"],))),
            ])
        blocks.extend([_heading("Частота и интервалы между последовательными покупками"),
            _table(scope, "customers.frequency_metrics", ("Показатель", "Значение"),
                (("Среднее заказов на покупателя", restored["mean_orders"]), ("Медиана заказов на покупателя", restored["median_orders"]),
                 ("Доля повторных покупателей, %", restored["repeat_rate"]), ("Средний интервал, дни", restored["intervals"]["mean_days"]),
                 ("Медианный интервал, дни", restored["intervals"]["median_days"]))),
            _table(scope, "customers.intervals_distribution", ("Интервал", "Число интервалов", "Доля, %"),
                   restored["intervals"]["distribution"]),
            _table(scope, "customers.last_months", ("Последняя наблюдаемая покупка", "Покупатели", "Доля, %"), restored["last_months"])])
    if dto.cohorts["channels"]:
        blocks.extend([_heading("Каналы первой и второй наблюдаемых покупок"),
                       _scope(dto, "cohorts_history"),
                       _table("cohorts_history", "customers.purchase_channels", ("Канал оформления", "Первые покупки", "Вторые покупки"),
                              ((row["name"], row["first_buyers"], row["second_buyers"])
                               for row in dto.cohorts["channels"]["channels"]))])
    return blocks


def _customer_money(dto):
    scope = "buyer_period"
    blocks = [_scope(dto, scope), *_currency_blocks(scope, "customers.money", dto.customers["data"]["purchases"]["buyer_money"])]
    data = dto.customers["data"].get("buyer_breakdowns")
    if not data:
        return blocks
    for currency in _CURRENCIES:
        annual = data["annual"]
        finance = data["money"][currency]
        groups = finance["quintiles"]
        blocks.extend([_heading("Полнота денежной оценки покупателей — " + currency),
            _table(scope, "customers.money.valuation." + currency, ("Показатель", "Значение"), (
                ("Оценённые заказы", finance["known_orders"]),
                ("Покупатели с известной суммой", finance["known_buyers"]),
                ("Покупатели с полной оценкой", finance["complete_buyers"]),
                ("Повторные покупатели с полной оценкой", finance["complete_repeat_buyers"]),
                (f"Известная сумма оценённых заказов, {currency}", _plain_money(finance["amount"]) if finance["known_orders"] else None),
                (f"Оборот покупателей с полной оценкой, {currency}", _plain_money(finance["complete_amount"]) if finance["complete_buyers"] else None),
                (f"Средний оборот на покупателя с полной оценкой, {currency}", _plain_money(finance["mean_complete_per_buyer"])),
                (f"Медианный оборот на покупателя с полной оценкой, {currency}", _plain_money(finance["median_complete_per_buyer"])),
                ("Доля оборота повторных покупателей, %", finance["repeat_amount_rate"]),
                ("Доля оборота повторных покупателей с полной оценкой, %", finance["complete_repeat_amount_rate"]),
            )),
            _heading("Годовая динамика покупателей — " + currency),
            _chart("Годовая сумма оценённых заказов — " + currency, [r["year"] for r in annual],
                   [(currency, [_plain_money(r["money"][currency]["amount"])
                                if r["money"][currency]["known_orders"] else None for r in annual])]),
            _table(scope, "customers.money.annual." + currency, ("Год", "Покупатели", "Заказы", "Неоценённые заказы",
                   f"Сумма оценённых заказов, {currency}", "Покупатели с полной оценкой",
                   f"Оборот полностью оценённых покупателей, {currency}", "Покрытие года", "Наблюдаемые интервалы"),
                   ((row["year"], row["buyers"], row["orders"], row["unvalued_orders"],
                     _plain_money(value["amount"]) if value["known_orders"] else None, value["complete_buyers"],
                     _plain_money(value["complete_amount"]) if value["complete_buyers"] else None,
                     "Полное" if row["complete_year"] else "Частичное",
                     " / ".join(_period(interval) for interval in row["coverage"]))
                    for row in annual for value in (row["money"][currency],))),
            _heading("Квантили оборота полностью оценённых покупателей — " + currency),
            _chart("Квантили оборота — " + currency, list(range(1, len(groups)+1)),
                   [(currency, [_plain_money(row[4]) for row in groups])]),
            _table(scope, "customers.money.quintiles." + currency, (f"От, {currency}", f"До, {currency}", "Покупатели",
                "Доля, %", f"Оборот, {currency}", "Повторные покупатели", "Доля повторных, %", f"Оборот повторных, {currency}",
                "Доля оборота повторных, %"),
                ((_plain_money(a), _plain_money(b), n, rate, _plain_money(amount), repeat, rr, _plain_money(ra), rar)
                 for a, b, n, rate, amount, repeat, rr, ra, rar in groups))])
        for field, label in (("gender", "По полу"), ("age", "По возрасту")):
            if data.get("demographic_analytics"):
                rows = data["demographic_analytics"][field]
                blocks.extend([_heading(label + " — " + currency),
                    _table(scope, "customers.money.demography." + field + "." + currency, (label, "Покупатели",
                        "Заказы", "Повторные покупатели", "Неоценённые заказы", f"Оценённые заказы, {currency}",
                        f"Оценённая сумма, {currency}", f"Среднее по оценённым, {currency}", f"Медиана, {currency}"),
                        ((r["label"], r["buyers"], r["orders"], r["repeat_buyers"], r["unvalued_orders"],
                          m["known_orders"], _plain_money(m["amount"]) if m["known_orders"] else None,
                          _plain_money(m["mean_order"]), _plain_money(m["median_order"]))
                         for r in rows for m in (r["money"][currency],)))])
    return blocks


def _rfm(dto):
    scope = "rfm_48m"
    data = dto.rfm
    blocks = [_scope(dto, scope), _text(_PROXY)]
    if not data["windows"]:
        return [*blocks, _text("RFM недоступен: нет покрытия заказов.")]
    blocks.append(_table(scope, "rfm.base", ("Показатель", "Значение"), [
        ("Идентифицированные клиенты", data["quality"]["population"]),
        ("Покупатели базы 48 месяцев", data["base_buyers"]),
        ("Покупатели обеих валют", data["quality"]["dual_currency_buyers"]),
        ("Покупатели без определённой валютной базы", data["quality"]["unassigned_currency_buyers"]),
    ]))
    choices = []
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
        coverage = payload["coverage"]
        blocks.extend([_heading(label), _text(_COVERAGE[coverage["status"]] + " покрытие · " +
                        _period((coverage["start"], coverage["end"])))])
        for currency, row in payload["currencies"].items():
            prefix = "rfm." + window + "." + currency
            blocks.append(_table(scope, prefix + ".metrics", ("Показатель R/F/M", "Значение"), [
                ("Валютная база", row["buyers"]), ("Активные покупатели", row["active_buyers"]),
                ("Наблюдаемые заказы", row["observed_orders"]),
                ("Покупатели с надёжной давностью (R)", row["reliable_recency_buyers"]),
                ("Покупатели с надёжной частотой (F)", row["reliable_frequency_buyers"]),
                ("Средняя наблюдаемая давность, дни", row["recency"]["mean"]),
                ("Медианная наблюдаемая давность, дни", row["recency"]["median"]),
                ("Средняя наблюдаемая частота, заказы", row["frequency"]["mean"]),
                ("Медианная наблюдаемая частота, заказы", row["frequency"]["median"]),
                ("Покупатели с полностью оценённой стоимостью (M)", row["monetary"]["buyers"]),
                (f"Средняя стоимость (M), {currency}", _plain_money(row["monetary"]["mean"])),
                (f"Медианная стоимость (M), {currency}", _plain_money(row["monetary"]["median"])),
                (f"Известный подытог, {currency}", _plain_money(row["known_subtotal"])),
                (f"Номинал CP, {currency}", _plain_money(row["partial_nominal"])),
                ("Покупатели с недоступной M", row["unavailable_monetary_buyers"]),
            ]))
            if "segment_business" in row:
                blocks.append(_table(scope, prefix + ".segment_money", (f"Сегмент — {currency}", "Наблюдаемые заказы", "Оценённые заказы",
                    f"Известный подытог, {currency}", "Доля подытога, %", f"Средний оценённый заказ, {currency}",
                    "Покупатели с полной M", f"Средняя полная M, {currency}"),
                    ((SEGMENT_LABELS[code][0], v["observed_orders"], v["valued_orders"], _plain_money(v["known_subtotal"]),
                      _plain_money(v["share_of_known_subtotal"]), _plain_money(v["mean_valued_order"]), v["complete_buyers"],
                      _plain_money(v["mean_complete_m"])) for code, v in row["segment_business"].items())))
            blocks.extend([
                _table(scope, prefix + ".segments", (f"Сегмент — {currency}", "Покупатели", "Доля, %"),
                       ((SEGMENT_LABELS[code][0], n, share) for code, n, share in row["segments"])),
                _table(scope, prefix + ".recency", (f"Давность, дни — {currency}", "Покупатели"), row["recency"]["distribution"]),
                _table(scope, prefix + ".frequency", (f"Частота, заказы — {currency}", "Покупатели"), row["frequency"]["distribution"]),
                _table(scope, prefix + ".scores", (f"R — {currency}", "F", "M", "Покупатели"), row["score_counts"]),
                _table(scope, prefix + ".thresholds", (f"Метрика — {currency}", "База", "Порог 20%", "Порог 40%", "Порог 60%", "Порог 80%"),
                       ((key, value["buyers"], *((_plain_money(v) if key == "M" else v) for v in
                         (value["values"] or (None,) * 4))) for key, value in row["thresholds"].items())),
                _table(scope, prefix + ".reasons", ("Качество RFM", "Покупатели"),
                       ((_REASONS.get(code, code), count) for code, count in row["reasons"].items())),
            ])
    comparison = data["comparison"]
    groups = {"BOTH": "Покупки в обоих окнах", "LAST_ONLY": "Только в последних 12 месяцах",
              "PREVIOUS_ONLY": "Только в предыдущих 12 месяцах", "NEITHER": "Без покупок в обоих окнах"}
    blocks.extend([_heading("Сравнение последних и предыдущих 12 месяцев"),
        _text("Покрытие сравнения полное." if comparison["coverage_reliable"] else "Покрытие сравнения неполное."),
        _table(scope, "rfm.comparison", ("Показатель", "Значение"),
            [(groups[key], count) for key, count in comparison["observed_groups"].items()] +
            [("Заказы в последних 12 месяцах", comparison["last_orders"]),
             ("Заказы в предыдущих 12 месяцах", comparison["previous_orders"]),
             ("Изменение числа заказов", comparison["order_change"]),
             ("Изменение числа заказов, %", comparison["order_change_percent"])]),
        _table(scope, "rfm.comparison.money", ("Валюта", "Известный подытог сейчас", "Известный подытог ранее",
            "Изменение известного подытога", "Изменение, %", "Покупатели с неполной оценкой"),
            ((currency, *(_plain_money(row[key]) for key in ("last_known_amount", "previous_known_amount",
              "known_amount_change", "known_amount_change_percent")), row["unreliable_buyers"])
             for currency, row in comparison["money"].items())),
        ])
    return blocks


def _cohort_money(blocks, data):
    scope = "cohorts_history"
    labels = {"total": "Все наблюдаемые заказы", "first": "Первые заказы", "repeat": "Повторные заказы"}
    for currency in _CURRENCIES:
        blocks.append(_heading(f"Денежные показатели когорт — {currency}"))
        rows = []
        for month, scopes in sorted(data["cohorts"].items()):
            for name, row in scopes.items():
                money = row["currencies"][currency]
                rows.append((month, labels[name], measure(money["recognized"], money=True),
                             measure(money["partial_nominal"], money=True), row["orders"], row["unvalued_orders"],
                             money["valued_orders"], money["fully_valued_buyers"], row["unvalued_buyers"],
                             _plain_money(money["mean_valued_order"]), _plain_money(money["mean_per_valued_buyer"]),
                             _plain_money(money.get("valued_repeat_amount_rate"))))
        blocks.append(_table(scope, "cohorts.money." + currency, ("Когорта", "База", f"Признано, {currency}",
            f"Номинал CP, {currency}", "Заказы", "Неоценённые заказы", "Оценённые заказы", "Оценённые покупатели",
            "Неоценённые покупатели", f"Среднее на заказ, {currency}", f"Среднее на покупателя, {currency}",
            "Доля оценённых повторных сумм, %"), rows))
        blocks.append(_table(scope, "cohorts.money.horizons." + currency,
            ("Горизонт, дни", "С полным наблюдением", "Повторные покупатели", f"Признано, {currency}",
             f"Номинал CP, {currency}", "Неоценённые заказы", f"Среднее на оценённый заказ, {currency}"),
            ((int(h), row["eligible"], row["buyers"], measure(row["currencies"][currency]["recognized"], money=True),
              measure(row["currencies"][currency]["partial_nominal"], money=True), row["unvalued_orders"],
              _plain_money(row["currencies"][currency]["mean_valued_order"])) for h, row in data["horizons"].items())))


def _cohorts(dto):
    scope = "cohorts_history"
    data = dto.cohorts
    blocks = [_scope(dto, scope), _text(_PROXY)]
    if not data["summary"]:
        return [*blocks, _text("Когорты недоступны: нет покрытия заказов.")]
    summary = data["summary"]
    blocks.extend([_heading("Когорты первой наблюдаемой покупки"),
        _table(scope, "cohorts.sizes", ("Месяц первой покупки", "Покупатели (M0)", "С наблюдаемой второй покупкой"),
               ((month, row["cohort_size"], row["observed_repeat_buyers"]) for month, row in sorted(data["months"].items()))),
        _table(scope, "cohorts.second_days", ("Дней между покупками", "Покупатели"),
               summary["second_purchase_days"]["distribution"]),
        _text("Повтор по горизонту учитывает дни до второй покупки. Удержание показывает активность в отдельном календарном месяце."),
        _heading("Повторные покупки по горизонтам"),
        _chart("Доля повторных покупателей", [f"{h} дней" for h in HORIZONS],
               [("Повторные покупатели", [measure(summary["horizons"][str(h)]["rate_quality"], percentage=True)
                                         for h in HORIZONS])], percentage=True)])
    rows = [("Все когорты", summary), *sorted(data["months"].items())]
    blocks.append(_table(scope, "cohorts.repeat", ("Когорта", "Дней", "С полным наблюдением", "Повторили",
        
        "Наблюдаемые повторы вне базы", "Цензурировано", "Покрытие неполно", "Цензурировано с неполным покрытием",
        "Подтверждено без повтора", "Неопределённые в базе", "Доля, %"),
        ((month, h, cell["eligible"], cell["repeated"], cell["observed_repeated_ineligible"], cell["censored"],
          cell["coverage_incomplete"], cell["censored_and_incomplete"], cell["confirmed_without_event"],
          cell["uncertain_eligible_buyers"], measure(cell["rate_quality"], percentage=True))
         for month, row in rows for h in HORIZONS for cell in (row["horizons"][str(h)],))))
    heatmap = RetentionHeatmapData(tuple(RetentionRow(month, row["cohort_size"], tuple(
        RetentionCell(cell["retention_rate"], cell["status"] if cell["retention_rate"] is not None else "UNAVAILABLE",
                      cell["eligible"], cell["active"], cell["censored"], cell["coverage_incomplete"])
        for cell in row["retention"][1:])) for month, row in sorted(data["months"].items())))
    blocks.extend([_heading("Календарное удержание M0–M12"),
                   report.ReportBlock("heatmap", data=heatmap, options={"context_id": scope}),
                   _table(scope, "cohorts.retention", ("Когорта", "Месяц", "С полным наблюдением", "Активны",
                       "Наблюдаемая активность вне базы", "Цензурировано", "Покрытие неполно", "Цензурировано с неполным покрытием",
                       "Подтверждено без активности", "Неопределённые в базе", "Удержание, %"),
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
        _table(scope, "cohorts.channel_transitions", ("Первый канал", "Второй канал", "Покупатели", "Доля, %", "Доля внутри первого, %"),
               ((channel_names[row["first_external_id"]], channel_names[row["second_external_id"]], row["buyers"],
                 row["repeat_rate"], row["first_channel_repeat_rate"]) for row in channels["transitions"])),
        _heading("Категории первой и второй покупки"),
        _table(scope, "cohorts.categories", ("Категория", "Путь", "Первые покупки", "Повторили", "Доля повторов, %",
                                             "Вторые покупки", "Доля вторых, %"),
               ((row["name"], " / ".join(row["path"] or ()), row["first_buyers"], row["repeat_buyers"], row["observed_repeat_rate"],
                 row["second_buyers"], row["second_rate"]) for row in categories["categories"])),
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


def _technical(dto, applied_filter):
    meta = dto.metadata
    names = {
        "raw_orders": "Исходные снимки заказов", "raw_lines": "Исходные позиции", "exact_duplicates": "Точные повторы",
        "business_equivalent": "Эквивалентные снимки", "snapshot_conflicts": "Конфликты снимков",
        "undated_orders": "Заказы без согласованной даты", "orders_outside_coverage": "Заказы за пределами покрытия",
        "unassigned_buyers": "Покупочные заказы без атрибуции",
    }
    header = [
        ("Политика покупки", dto.policy["policy_id"]), ("Временная основа", dto.policy["time_basis_id"]),
        ("Источник", dto.source["reader"]), ("DTO", 11), ("Версия проекции", meta["projection_version"]),
        ("Схема индекса", dto.source["projection_schema"]), ("Каталог статусов", dto.policy["status_catalog_id"]),
        ("Правило согласования снимков", dto.policy["snapshot_policy_id"]), ("Денежная политика", dto.policy["money_policy_id"]),
        ("Дата расчёта", _stamp(dto.calculated_at)), ("Дата отсечения", dto.rfm["as_of_date"]),
        ("Применённый отбор", applied_filter), ("Алгоритм", meta["calculation_algorithm"]),
    ]
    diagnostics = dto.diagnostics
    blocks = [_heading("Методология и источник"),
              _table("raw_orders", "technical.metadata", ("Показатель", "Значение"), header),
              _text(_PROXY), _text(_SCOPE_NOTE),
              _heading("Периоды и объявленное покрытие"),
              _table("raw_orders", "technical.coverage", ("Область", "Аналитический период", "Покрытие заказов", "Покрытие действий"),
                ((_CONTEXT_LABELS[key], _period(context["period"]),
                  "; ".join(_period(x) for x in dto.coverage[key]["orders"]) or "Нет покрытия",
                  "; ".join(_period(x) for x in dto.coverage[key]["actions"]) or "Нет покрытия")
                 for key, context in dto.contexts.items())),
              _heading("Диагностика"),
              _table("raw_orders", "technical.diagnostics", ("Диагностика", "Область", "Количество"),
                     ((names.get(key, key), _CONTEXT_LABELS[diagnostics["scope_by_counter"][key]], count)
                      for key, count in diagnostics["counters"].items())),
              _table("raw_orders", "technical.reasons", ("Причина", "Код", "Стадия", "Количество"),
                     ((_REASONS.get(key, "Диагностика исходных данных"), key,
                       "Входные действия" if diagnostics["reason_stage"][key] == "ACTION_INPUT_BEFORE_FILTER" else "Согласование заказов",
                       value) for key, value in diagnostics["reason_counts"].items()))]
    if "event_names" in dto.actions.get("breakdowns", {}):
        names = dto.actions["breakdowns"]["event_names"]
        total = dto.actions["breakdowns"]["source_events"].get("all", 0)
        blocks.extend([_heading("Системные имена исходных событий"),
                       _table("raw_orders", "technical.action_names", ("Системное имя", "События", "Доля, %"),
                              ((name, count, 100 * count / total if total else 0) for _, name, count in names))])
    return tuple(header), blocks


def unified_presentation(dto):
    """The DTO must have passed the versioned reader before reaching presentation."""
    params = dto.metadata["parameters"]
    parts = []
    if params["order_period"]:
        parts.append(_period(params["order_period"]))
    if params["product_filter_key"]:
        parts.append("Товарный отбор применён")
    header, technical = _technical(dto, " · ".join(parts) if parts else "Отбор не установлен")
    customer_providers = (_overview, _purchases, _customer_money, _rfm, _cohorts)
    customer_groups = tuple(report.ReportGroup(key, title, tuple(provider(dto))) for key, title, provider in
                            zip(CUSTOMER_GROUP_ORDER, report.CUSTOMER_TAB_LABELS, customer_providers))
    pages = tuple(report.ReportPage(key, title, (report.ReportGroup(key, title, tuple(provider(dto))),))
                  for key, title, provider in zip(("actions", "orders", "products"), report.PAGE_LABELS,
                                                  (_actions, _orders, _products)))
    pages += (report.ReportPage("customers", report.PAGE_LABELS[3], customer_groups),
              report.ReportPage("technical", report.PAGE_LABELS[4],
                                (report.ReportGroup("technical", report.PAGE_LABELS[4], tuple(technical)),)))
    activity = dto.actions["data"]
    interactions = sum(activity[key]["known"] for key in _ACTIONS)
    demography = dto.customers["data"].get("customer_demography")
    cards = (("Наблюдаемые взаимодействия", interactions),
             ("Уникальные заказы", measure(dto.orders["unique_orders"])),
             ("Позиции заказов", measure(dto.orders["positions"])),
             ("Профили клиентского среза", demography["profiles"] if demography else None))
    return report.StatisticsReport(header, cards, "Статистика Mindbox", pages,
                                   metadata=dict(policy=dto.policy, source=dto.source, contexts=dto.contexts,
                                                 versions=dto.metadata, schema_version=11))
