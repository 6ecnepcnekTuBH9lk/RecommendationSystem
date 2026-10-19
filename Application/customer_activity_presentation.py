"""Activity numbers -> existing card contracts; no I/O or Qt."""

from decimal import Decimal, localcontext
from datetime import timedelta, timezone

from Application.customer_activity_service import ActivityStatus
from Application.customer_card_data import DisplayField, SectionContent, ViewState


ACTIVITY_FIELDS = ("Заказы", "Количество покупок", "Сумма покупок", "Средний чек", "Просмотры", "Избранное",
                   "Дней с последней покупки", "Период анализа")
_REASONS = {
    "coverage": "Покрытие источника отсутствует или имеет разрывы.",
    "order_conflict": "Не все покупочные строки удалось согласовать между снимками заказа.",
    "missing_order_lines": "Состав части заказов недоступен.",
    "quantity_conflict": "Количество различается между покупочными снимками.",
    "amount_conflict": "Есть некорректные или противоречивые суммы товарных строк.",
    "currency": "Валюта части покупочных заказов неизвестна или смешана; RUB и KZT не объединяются.",
    "purchase_time": "Дата покупки недостоверна, находится вне покрытия или в будущем.",
    "action_conflict": "Принадлежность или тип части действий противоречивы.",
    "action_time": "Не все действия относятся к подтверждённому периоду.",
}


def _number(value, *, money=False):
    with localcontext() as context:
        context.prec = max(64, len(Decimal(value).as_tuple().digits) + 4)
        text = format(value, ",.2f" if money else ",f").replace(",", " ")
    if not money and "." in text:
        text = text.rstrip("0").rstrip(".")
    if money and text.endswith(".00"):
        text = text[:-3]
    return text.replace(".", ",")


def _continuous_span(intervals):
    if not intervals:
        return None
    intervals = sorted(intervals)
    start, end = intervals[0]
    if start >= end:
        return None
    for left, right in intervals[1:]:
        if left >= right or left > end:
            return None
        end = max(end, right)
    return start, end


def _analysis_period(result):
    """Envelope of continuous confirmed coverages, preserving exclusive ends."""
    confirmed = []
    for component, complete in (("orders", result.orders_coverage_complete),
                                ("actions", result.actions_coverage_complete)):
        if not complete:
            continue
        intervals = []
        for span in result.coverage:
            if span.component != component:
                continue
            if any(d.tzinfo is None or d.utcoffset() is None for d in (span.since_utc, span.until_utc)):
                return None
            intervals.append((span.since_utc.astimezone(timezone.utc), span.until_utc.astimezone(timezone.utc)))
        interval = _continuous_span(intervals)
        if interval is None:
            return None
        confirmed.append(interval)
    period = _continuous_span(confirmed)
    if period is None:
        return None
    start, end = period
    return f"{start:%d.%m.%Y} - {end - timedelta(microseconds=1):%d.%m.%Y}"


def activity_content(result=None, *, state=ViewState.INITIAL):
    if result is None or result.status not in (ActivityStatus.READY, ActivityStatus.EMPTY):
        if result is not None:
            state = {ActivityStatus.UNAVAILABLE: ViewState.UNAVAILABLE, ActivityStatus.STALE: ViewState.STALE,
                     ActivityStatus.CANCELLED: ViewState.CANCELLED, ActivityStatus.ERROR: ViewState.ERROR}[result.status]
        message = {ViewState.STALE: "Данные устарели", ViewState.CANCELLED: "Загрузка отменена"}.get(state)
        return SectionContent(state, "", tuple(DisplayField(title, state=state, message=message) for title in ACTIVITY_FIELDS))
    period = _analysis_period(result)
    symbols = {"RUB": "₽", "KZT": "₸"}
    amounts = None if result.money is None else "\n".join(f"{_number(row.amount, money=True)} {symbols[row.currency]}" for row in result.money)
    means = None if result.money is None else "\n".join(f"{_number(row.average_check, money=True)} {symbols[row.currency]}" for row in result.money)
    issues = {metric: " ".join(_REASONS[i.reason] for i in result.issues if i.metric == metric)
              for metric in ("orders", "units", "money", "views", "favorites", "recency")}
    values = (result.orders, result.units, amounts, means, result.views, result.favorites,
              result.days_since_purchase, period)
    metrics = ("orders", "units", "money", "money", "views", "favorites", "recency", "period")
    fields = []
    for title, metric, value in zip(ACTIVITY_FIELDS, metrics, values):
        note = None
        if metric == "recency" and result.purchase_as_of_utc is not None:
            note = f"По состоянию на {result.purchase_as_of_utc:%d.%m.%Y}."
        if metric == "period":
            fields.append(DisplayField(title, value, ViewState.READY) if value is not None else
                          DisplayField(title, state=ViewState.UNAVAILABLE))
        elif issues[metric]:
            fields.append(DisplayField(title, state=ViewState.UNAVAILABLE, note=None if metric == "units" else issues[metric]))
        elif value is None or value == "":
            fields.append(DisplayField(title, state=ViewState.EMPTY, message="Нет покупок", note=note))
        else:
            if metric == "recency":
                value = f"{value} дн."
            elif not isinstance(value, str):
                value = _number(value)
            fields.append(DisplayField(title, value, ViewState.READY, note=note))
    return SectionContent(ViewState.READY if result.status is ActivityStatus.READY else ViewState.EMPTY, "", tuple(fields))
