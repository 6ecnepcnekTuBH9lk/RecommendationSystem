"""Research forecast presentation only; no I/O, identity resolution or scoring."""
from datetime import datetime
from Application.customer_card_data import DisplayField, SectionContent, ViewState
from Application.customer_propensity_service import PropensityStatus

FIELDS = ("Вероятность оформления заказа в следующие 30 дней", "Прогнозируемый период", "Модель")
MESSAGES = {
    PropensityStatus.NOT_FOUND: "Для покупателя нет прогноза.",
    PropensityStatus.NOT_READY: "Прогнозный артефакт недоступен.",
    PropensityStatus.STALE: "Данные прогноза устарели или несовместимы. Повторите поиск.",
    PropensityStatus.IDENTITY_NOT_CONFIRMED: "Не удалось однозначно подтвердить покупателя.",
    PropensityStatus.KEY_UNAVAILABLE: "Защищённый ключ прогноза недоступен.",
    PropensityStatus.ERROR: "Ошибка чтения или проверки прогноза.",
    PropensityStatus.CANCELLED: "Поиск отменён.",
}


def propensity_content(result=None, *, state=ViewState.INITIAL):
    if result is None:
        message = "Поиск отменён." if state is ViewState.CANCELLED else ""
        return SectionContent(state, message, tuple(DisplayField(n, state=state,
                              message="Поиск отменён" if state is ViewState.CANCELLED else None) for n in FIELDS))
    if result.status not in (PropensityStatus.READY, PropensityStatus.WARNING):
        state = {PropensityStatus.STALE: ViewState.STALE, PropensityStatus.CANCELLED: ViewState.CANCELLED,
                 PropensityStatus.ERROR: ViewState.ERROR, PropensityStatus.NOT_FOUND: ViewState.EMPTY}.get(result.status, ViewState.UNAVAILABLE)
        message = MESSAGES[result.status]
        return SectionContent(state, message, tuple(DisplayField(n, state=state, message="—") for n in FIELDS))
    warnings = []
    value = f"{result.probability*100:.2f}".replace(".", ",") + "%"
    if result.probability > 0 and result.probability*100 < .005:
        value = "< 0,01%"
    if "NEAR_ZERO_WITH_RECENT_FAVORITE" in result.warnings:
        value = "Оценка ненадёжна"
        warnings.append("Модель может существенно занижать вероятность для покупателей с недавним избранным")
    if "LOW_INDIVIDUALIZATION" in result.warnings:
        warnings.append("Недостаточно индивидуальных сигналов. В доступных исторических окнах не найдено активности, "
                        "позволяющей надежно различать покупателей")
    if "HISTORICAL_FEATURES_UNAVAILABLE" in result.warnings:
        warnings.append("Часть исторических данных недоступна. Оценка ограничена покрытием источников")
    since = datetime.fromisoformat(result.prediction_date.replace("Z", "+00:00")).strftime("%d.%m.%Y")
    until = datetime.fromisoformat(result.target_until.replace("Z", "+00:00")).strftime("%d.%m.%Y")
    values = (value, f"с {since} до {until}", "Customer Propensity")
    return SectionContent(ViewState.READY, ". ".join(warnings), tuple(
        DisplayField(n, v, ViewState.READY) for n, v in zip(FIELDS, values)))
