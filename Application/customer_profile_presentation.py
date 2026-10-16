"""Profile -> existing display contracts. No Qt, SQL, calculations or search."""

from Application.customer_card_data import DisplayField, SectionContent, ViewState
from Application.customer_profile_service import BirthDateStatus, CardStatus, ProfileStatus


PROFILE_FIELDS = ("ФИО", "Пол", "Последнее изменение профиля", "Дата рождения", "Возраст",
                  "Возрастная группа", "Телефон", "Email", "Дисконтная карта")


def profile_content(result):
    messages = {
        ProfileStatus.NOT_FOUND: (ViewState.EMPTY, "Профиль клиента не найден в Customers."),
        ProfileStatus.INVALID_ID: (ViewState.ERROR, "Введите корректный MindboxID клиента."),
        ProfileStatus.SOURCE_UNAVAILABLE: (ViewState.UNAVAILABLE, "Не удалось получить данные профиля. Источник недоступен."),
        ProfileStatus.SOURCE_INCONSISTENT: (ViewState.ERROR, "Данные профиля не удалось обработать."),
        ProfileStatus.STALE: (ViewState.STALE, "Customers или объединения клиентов изменились. Повторите поиск."),
        ProfileStatus.CANCELLED: (ViewState.CANCELLED, "Загрузка профиля отменена."),
    }
    if result.status is not ProfileStatus.FOUND:
        state, message = messages[result.status]
        return SectionContent(state, message)
    p = result.profile
    def value(title, text, note=None):
        return DisplayField(title, text, ViewState.READY if text is not None else ViewState.EMPTY,
                            message="Не указано", note=note)
    card = value("Дисконтная карта", p.contact.discount_card,
                 "Последняя активированная карта" if p.card_status is CardStatus.ACTIVATED else None)
    if p.card_status in (CardStatus.AMBIGUOUS, CardStatus.INVALID):
        card = DisplayField("Дисконтная карта", state=ViewState.UNAVAILABLE,
                            message="Найдено несколько карт" if p.card_status is CardStatus.AMBIGUOUS else "Некорректный номер карты")
    birth = value("Дата рождения", p.record.birth_date.strftime("%d.%m.%Y") if p.record.birth_date else None)
    age = value("Возраст", f"{p.age} полных лет" if p.age is not None else None)
    if p.birth_date_status in (BirthDateStatus.INVALID, BirthDateStatus.FUTURE):
        message = "Некорректная дата" if p.birth_date_status is BirthDateStatus.INVALID else "Дата рождения в будущем"
        birth = DisplayField("Дата рождения", state=ViewState.UNAVAILABLE, message=message)
        age = DisplayField("Возраст", state=ViewState.UNAVAILABLE, message="Возраст не определён")
    phone_notes = []
    if p.contact.mobile_phone is not None and p.phone_invalid is True:
        phone_notes.append("Источник пометил телефон как невалидный")
    if p.contact.mobile_phone is not None and p.phone_confirmed is not None:
        phone_notes.append("Телефон подтверждён" if p.phone_confirmed else "Подтверждение телефона не отмечено")
    fields = (
        value("ФИО", p.full_name), value("Пол", p.sex_label),
        value("Последнее изменение профиля", p.record.change_datetime_utc.astimezone().strftime("%d.%m.%Y %H:%M")
              if p.record.change_datetime_utc else None),
        birth, age, value("Возрастная группа", p.age_group),
        value("Телефон", p.contact.mobile_phone, "; ".join(phone_notes) or None),
        value("Email", p.contact.email, "Источник пометил email как невалидный" if p.email_invalid else None), card,
    )
    return SectionContent(ViewState.READY, f"Профиль получен из Customers. Возраст рассчитан на {p.as_of:%d.%m.%Y}.", fields)
