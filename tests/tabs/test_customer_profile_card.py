import os
from dataclasses import replace
from datetime import date, datetime, timezone
from threading import Event, get_ident
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow

from Application.customer_card_data import CustomerCardResult, CustomerCardService, ProductContent, ViewState
from Application.customer_profile_service import BirthDateStatus, CardStatus, CustomerProfile, CustomerProfileResult, ProfileStatus
from Application.customer_profiles import CustomerContact
from Application.mindbox.records import CustomerRecord
from Application.model.customer_identity_bridge import CustomerModelIdentityResult, CustomerModelIdentityStatus as S
from Application.model.raw_bpr_inference import RawBprRecommendation, RawBprResult, RawBprStatus
from Application.customer_history_service import HistoryPage, HistoryStatus
from Application.customer_profile_presentation import PROFILE_FIELDS
from Application.tabs.customer_results_tab import CustomerResultsTab
from Application.theme.apply_theme import apply_app_theme, prepare_app_theme


def found(*, name="Иван", canonical="C", card=CardStatus.SINGLE, birthday=BirthDateStatus.VALID):
    record = CustomerRecord("C", canonical, datetime(2026, 10, 1, tzinfo=timezone.utc), date(1990, 1, 1),
                            "male", name, "Иванов", "Иванович", "synthetic@example.test", "+7 (900) 123-45-67")
    profile = CustomerProfile(record, CustomerContact(record.email, record.mobile_phone, "000123" if card is not CardStatus.AMBIGUOUS else None),
                              date(2026, 10, 13), 36 if birthday is BirthDateStatus.VALID else None,
                              "36–45 лет" if birthday is BirthDateStatus.VALID else None, birthday, card,
                              phone_invalid=False, phone_confirmed=True)
    return CustomerProfileResult(ProfileStatus.FOUND, profile, canonical, "synthetic-version", "r")


def recommendations():
    identity = CustomerModelIdentityResult("C", "A", S.MATCHED_VIA_MERGE, 1, "g", "r", "v")
    products = tuple(ProductContent(RawBprRecommendation(i, f"SYNTHETIC-{i}", 1.123456789 / i)) for i in range(1, 11))
    return CustomerCardResult(ViewState.READY, "Рекомендации получены", identity, "g", products)


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    prepare_app_theme(app)
    return app


@pytest.fixture
def widget(app):
    host = QMainWindow()
    profile = Mock()
    profile.get_profile.return_value = found()
    profile.is_current.return_value = True
    bridge, inference = Mock(), Mock()
    bridge.resolve.return_value = recommendations().identity
    bridge.is_current.return_value = True
    inference.get_recommendations.return_value = RawBprResult("g", "A", RawBprStatus.OK,
        tuple(p.recommendation for p in recommendations().products))
    service = CustomerCardService(profile_service=profile, bridge=bridge, inference=inference)
    tab = CustomerResultsTab(host, service=service, history_lookup=lambda customer, **kw: HistoryPage(HistoryStatus.EMPTY, customer))
    host.setCentralWidget(tab); host.resize(1250, 900); host.show(); QTest.qWait(20)
    yield host, tab, service
    tab.shutdown()
    if tab._thread is not None:
        assert tab._thread.wait(5000)
    host.close(); host.deleteLater(); QTest.qWait(20)


def wait(predicate, timeout=4000):
    from time import monotonic
    deadline = monotonic() + timeout / 1000
    while not predicate() and monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


def fields(tab):
    return {card.content.title: card.value.text() for card in tab.profile_fields.cards}


def search(tab, value="C", enter=False):
    tab.customer_input.setText(value)
    if enter:
        QTest.keyClick(tab.customer_input, Qt.Key.Key_Return)
    else:
        tab.search_button.click()
    wait(lambda: tab.result is not None)


@pytest.mark.parametrize("enter", [False, True])
def test_profile_fields_and_recommendations_correct_and_off_gui_thread(widget, enter):
    _, tab, service = widget
    main_thread, seen = get_ident(), []
    def profile(*args, **kwargs):
        seen.append(get_ident())
        return found()
    service.profile_service.get_profile.side_effect = profile
    search(tab, enter=enter)
    assert len(seen) == 1 and seen[0] != main_thread
    actual = fields(tab)
    assert actual["ФИО"] == "Иванов Иван Иванович"
    assert actual["Дисконтная карта"] == "000123"
    assert actual["Email"] == "synthetic@example.test"
    assert actual["Телефон"] == "+7 (900) 123-45-67"
    assert actual["Возраст"] == "36 полных лет" and actual["Пол"] == "Мужской"
    assert actual["Дата рождения"] == "01.01.1990"
    assert "Дата регистрации" not in actual and len(actual) == 9
    assert actual["Последнее изменение профиля"] == found().profile.record.change_datetime_utc.astimezone().strftime("%d.%m.%Y %H:%M")
    assert tab.profile.content.state is ViewState.READY
    assert len(tab.raw_grid.cards) == 10
    assert tab.raw_grid.cards[0].content.recommendation.raw_score == 1.123456789
    assert all(not card.value.toolTip() for card in tab.profile_fields.cards)


@pytest.mark.parametrize("status", [S.NOT_IN_MODEL, S.AMBIGUOUS, S.SOURCE_UNAVAILABLE, S.SOURCE_INCONSISTENT])
def test_profile_stays_available_without_bpr(widget, status):
    _, tab, service = widget
    service.bridge.resolve.return_value = CustomerModelIdentityResult(
        "C" if status in (S.NOT_IN_MODEL, S.AMBIGUOUS) else None, None, status, 3 if status is S.AMBIGUOUS else 0)
    search(tab)
    assert fields(tab)["ФИО"] == "Иванов Иван Иванович"
    assert tab.profile.content.state is ViewState.READY
    assert not tab.raw_grid.cards
    service.inference.get_recommendations.assert_not_called()


@pytest.mark.parametrize("status", [ProfileStatus.NOT_FOUND, ProfileStatus.SOURCE_UNAVAILABLE, ProfileStatus.SOURCE_INCONSISTENT])
def test_profile_failure_does_not_destroy_raw_bpr(widget, status):
    _, tab, service = widget
    service.profile_service.get_profile.return_value = CustomerProfileResult(status)
    search(tab)
    assert len(tab.raw_grid.cards) == 10
    assert tab.profile.content.state in (ViewState.EMPTY, ViewState.ERROR, ViewState.UNAVAILABLE)
    assert len(tab.profile_fields.cards) == 9 and all(card.content.state is not ViewState.READY for card in tab.profile_fields.cards)
    assert tab.profile.content.message in tab.search_status.text()


def test_both_absent_do_not_confirm_profile(widget):
    _, tab, service = widget
    service.profile_service.get_profile.return_value = CustomerProfileResult(ProfileStatus.NOT_FOUND)
    service.bridge.resolve.return_value = CustomerModelIdentityResult("unknown", None, S.NOT_IN_MODEL, 0)
    search(tab, "unknown")
    assert "не найден" in tab.profile.message.text()
    assert len(tab.profile_fields.cards) == 9 and not tab.raw_grid.cards
    assert "не найден" in tab.search_status.text()


def test_missing_fields_card_ambiguity_bad_birth_and_registration_are_distinct(widget):
    _, tab, _ = widget
    p = found(card=CardStatus.AMBIGUOUS, birthday=BirthDateStatus.INVALID)
    record = replace(p.profile.record, first_name=None, last_name=None, middle_name=None, email=None, mobile_phone=None)
    p = replace(p, profile=replace(p.profile, record=record, contact=CustomerContact()))
    tab.apply_result(replace(recommendations(), profile=p))
    actual = fields(tab)
    assert actual["ФИО"] == actual["Email"] == actual["Телефон"] == "Не указано"
    assert actual["Дисконтная карта"] == "Найдено несколько карт"
    assert actual["Дата рождения"] == "Некорректная дата"
    assert actual["Возраст"] == "Возраст не определён"
    assert "Дата регистрации" not in actual and actual["Последнее изменение профиля"]
    assert len(tab.raw_grid.cards) == 10


def test_profile_can_be_shown_while_recommendations_are_still_loading(widget):
    _, tab, service = widget
    entered, release = Event(), Event()
    def inference(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        raise ValueError("PRIVATE_MODEL_DETAIL")
    service.inference.get_recommendations.side_effect = inference
    tab.customer_input.setText("C"); tab.search()
    wait(entered.is_set)
    wait(lambda: tab.profile.content.state is ViewState.READY)
    assert fields(tab)["ФИО"] == "Иванов Иван Иванович"
    assert tab.result is None and tab.progress.isVisible()
    release.set(); wait(lambda: tab.result is not None)
    assert tab.result.state is ViewState.ERROR and tab.profile.content.state is ViewState.READY
    assert "PRIVATE" not in tab.search_status.text()


def test_latest_request_never_gets_old_profile_and_cancel_then_repeat(widget):
    _, tab, service = widget
    entered, release = Event(), Event()
    def profile(customer, **kwargs):
        if customer == "old":
            entered.set(); assert release.wait(4)
        return found(name=customer)
    service.profile_service.get_profile.side_effect = profile
    tab.customer_input.setText("old"); tab.search(); wait(entered.is_set)
    tab.cancel_button.click()
    tab.customer_input.setText("new"); tab.search(); release.set()
    wait(lambda: tab.result is not None)
    assert fields(tab)["ФИО"] == "Иванов new Иванович"
    assert all("old" not in card.value.text() for card in tab.profile_fields.cards)


def test_customers_replacement_discards_only_profile(widget):
    _, tab, service = widget
    service.profile_service.is_current.return_value = False
    search(tab)
    assert tab.result.profile.status is ProfileStatus.STALE
    assert tab.profile.content.state is ViewState.STALE and len(tab.profile_fields.cards) == 9
    assert all(card.value.text() == "Данные устарели" for card in tab.profile_fields.cards)
    assert "Повторите поиск" in tab.search_status.text()
    assert len(tab.raw_grid.cards) == 10


def test_inconsistent_confirmed_identities_cannot_show_other_customer_recommendations(widget):
    _, tab, service = widget
    service.profile_service.get_profile.return_value = found(canonical="DIFFERENT")
    search(tab)
    assert tab.result.state is ViewState.STALE and not tab.raw_grid.cards
    assert tab.profile.content.state is ViewState.READY


@pytest.mark.parametrize("dark", [False, True])
def test_long_profile_values_wrap_without_expanding_page(widget, app, dark):
    host, tab, _ = widget
    p = found(name="ОченьДлинноеИмя" * 50)
    contact = replace(p.profile.contact, email="long" * 150 + "@example.test")
    p = replace(p, profile=replace(p.profile, contact=contact))
    tab.apply_result(replace(recommendations(), profile=p))
    apply_app_theme(app, dark); host.resize(550, 850); QTest.qWait(50)
    assert tab.scroll.horizontalScrollBar().maximum() == 0
    assert tab.page.width() <= tab.scroll.viewport().width()
    assert fields(tab)["Email"] == contact.email


def test_close_while_profile_loading_is_safe(widget):
    host, tab, service = widget
    entered, release = Event(), Event()
    def profile(*args, **kwargs):
        entered.set(); assert release.wait(4)
        return found()
    service.profile_service.get_profile.side_effect = profile
    tab.customer_input.setText("C"); tab.search(); wait(entered.is_set)
    host.close(); assert tab._closing
    release.set(); wait(lambda: tab._thread is None)
    assert not host.isVisible() and tab.result is None


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width,columns", [(1250, 3), (680, 2), (480, 1)])
def test_nine_fields_exact_grouped_positions_and_no_duplicate_headers(widget, app, dark, width, columns):
    host, tab, _ = widget
    p = found(card=CardStatus.ACTIVATED)
    tab.apply_result(replace(recommendations(), profile=p))
    apply_app_theme(app, dark)
    host.resize(width, 850)
    QTest.qWait(60)
    grid = tab.profile_fields
    assert grid.columns == columns
    expected = ("ФИО", "Пол", "Последнее изменение профиля", "Дата рождения", "Возраст",
                "Возрастная группа", "Телефон", "Email", "Дисконтная карта")
    assert tuple(card.content.title for card in grid.cards) == PROFILE_FIELDS == expected
    assert len(grid.cards) == len(set(PROFILE_FIELDS)) == 9
    for index, card in enumerate(grid.cards):
        group, row = divmod(index, 3)
        assert grid.grid.itemAtPosition((group // columns) * 3 + row, group % columns).widget() is card
        assert card.value.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    labels = [label for label in tab.profile.findChildren(QLabel) if label.isVisibleTo(tab.profile)]
    texts = [label.text() for label in labels]
    assert texts.count(p.profile.full_name) == 1
    assert "Дата регистрации" not in texts and "Изменение профиля" not in texts
    assert "Телефон подтверждён" in texts and "Последняя активированная карта" in texts
    assert not tab.profile.heading.isVisibleTo(tab.profile) and not tab.profile.message.isVisibleTo(tab.profile)
    assert tab.profile.box.itemAt(tab.profile.box.count() - 1).widget().geometry().top() == 0
    assert not any("Canonical MindboxID" in t or "ID в модели" in t for t in texts)
    assert not any(hasattr(tab, name) for name in ("profile_title", "canonical_id", "model_id", "identity_status"))
    assert tab.recommendations.heading.isVisibleTo(tab.recommendations)
    assert tab.kpis.heading.isVisibleTo(tab.kpis)
    assert tab.scroll.horizontalScrollBar().maximum() == 0
    assert tab.page.width() <= tab.scroll.viewport().width()


def test_initial_loading_cancel_and_error_have_nine_fields_and_visible_search_status(widget):
    _, tab, service = widget
    assert len(tab.profile_fields.cards) == 9
    assert all(card.content.state is ViewState.INITIAL for card in tab.profile_fields.cards)
    entered, release = Event(), Event()
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        return found()
    service.profile_service.get_profile.side_effect = blocked
    tab.customer_input.setText("C")
    tab.search()
    try:
        wait(entered.is_set)
        assert len(tab.profile_fields.cards) == 9
        assert all(card.value.text() == "Загрузка…" for card in tab.profile_fields.cards)
        assert "Ищем клиента" in tab.search_status.text()
        tab.cancel_search()
        assert len(tab.profile_fields.cards) == 9
        assert all(card.value.text() == "Загрузка отменена" for card in tab.profile_fields.cards)
        assert "отменён" in tab.search_status.text()
    finally:
        release.set()
    wait(lambda: tab._active is None)
    service.profile_service.get_profile.side_effect = None
    service.profile_service.get_profile.return_value = CustomerProfileResult(ProfileStatus.SOURCE_INCONSISTENT)
    search(tab)
    assert len(tab.profile_fields.cards) == 9
    assert all(card.value.text() == "Профиль недоступен" for card in tab.profile_fields.cards)
    assert "Данные профиля не удалось обработать" in tab.search_status.text()
    assert len(tab.raw_grid.cards) == 10


def test_user_search_design_and_terminology_preserved(widget):
    _, tab, _ = widget
    assert tab.search_button.text() == " Найти" and not tab.search_button.icon().isNull()
    assert tab.cancel_button.text() == " Отменить поиск" and not tab.cancel_button.icon().isNull()
    assert tab.export_button.text() == " Экспорт рекомендаций" and not tab.export_button.icon().isNull()
    assert tab.search_hint.text() == "Вводить значение без пробелов по краям"
    assert tab.search_status.text() == "Введите значение для выбранного типа поиска, чтобы посмотреть оригинальные рекомендации нейросетевой модели"
    assert tab.customer_input.placeholderText() == "Введите MindboxID клиента..."
    assert tab.search_type.accessibleName() == "Тип поиска клиента"
    for index in range(tab.search_type.count()):
        tab.search_type.setCurrentIndex(index)
        assert "покупател" not in tab.customer_input.placeholderText().lower()
        assert "покупател" not in tab.search_status.text().lower()
    assert not tab.history.content.message
