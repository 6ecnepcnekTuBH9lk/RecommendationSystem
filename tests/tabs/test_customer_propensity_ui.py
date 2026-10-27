"""Optional propensity stage on the existing card worker and request lifecycle."""
from dataclasses import replace
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLabel

from Application.customer_card_data import ViewState
from Application.customer_propensity_service import CustomerPropensity, PropensityStatus as S
from Application.customer_contact_search import ContactType, ContactSearchResult, ContactSearchStatus
from Application.model.customer_identity_bridge import CustomerModelIdentityResult, CustomerModelIdentityStatus as I
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_customer_contact_search_ui import app as app, card as card, search, wait
from tests.test_customer_propensity_service import prediction


@pytest.fixture
def propensity_card(card):
    seen = []
    def get(customer, **kwargs):
        seen.append((customer, get_ident()))
        return replace(prediction(), canonical_customer_id=customer)
    card.service.propensity_service = SimpleNamespace(get=get, validate=lambda result, **kwargs: result)
    return card, seen


@pytest.mark.parametrize("kind", [None,ContactType.PHONE,ContactType.EMAIL,ContactType.DISCOUNT_CARD])
def test_four_modes_confirm_canonical_before_hmac_stage(propensity_card, kind):
    card, seen = propensity_card
    search(card, kind)
    assert seen and all(customer == "C" and thread != get_ident() for customer, thread in seen)
    assert card.tab.result.propensity.canonical_customer_id == "C"
    assert card.tab.propensity_grid.cards[0].value.text() == "12,35%"
    assert card.tab.raw_grid.cards[0].content.recommendation.raw_score == card.raw[0].raw_score
    assert card.tab.profile.content.state is ViewState.READY


@pytest.mark.parametrize("failure", ["not_in_model","inference_error"])
def test_independent_of_bpr(propensity_card, failure):
    card, _ = propensity_card
    if failure == "not_in_model":
        card.service.bridge.resolve.side_effect = lambda *a, **k: CustomerModelIdentityResult("C", None, I.NOT_IN_MODEL, 0)
    else:
        card.service.inference.get_recommendations.side_effect = RuntimeError("PRIVATE")
    search(card)
    assert card.tab.propensity.content.state is ViewState.READY
    assert card.tab.propensity_grid.cards[0].value.text() == "12,35%"
    assert not card.tab.raw_grid.cards


@pytest.mark.parametrize("status", [S.NOT_READY,S.NOT_FOUND,S.KEY_UNAVAILABLE,S.STALE,S.ERROR])
def test_failure_does_not_hide_other_sections(propensity_card, status):
    card, _ = propensity_card
    card.service.propensity_service.get = lambda *a, **k: CustomerPropensity(status, "C" if status is S.NOT_FOUND else None)
    search(card)
    assert card.tab.raw_grid.cards and card.tab.profile.content.state is ViewState.READY
    assert card.tab.history.customer_id == "C"
    assert card.tab.propensity.content.state is not ViewState.READY
    assert all(c.content.value is None for c in card.tab.propensity_grid.cards)


@pytest.mark.parametrize("contact_status", [ContactSearchStatus.AMBIGUOUS,ContactSearchStatus.NOT_FOUND])
def test_no_prediction_for_ambiguous_contacts(propensity_card, contact_status):
    card, seen = propensity_card
    card.backend.return_value = ContactSearchResult(contact_status, candidate_count=2 if contact_status is ContactSearchStatus.AMBIGUOUS else 0)
    search(card)
    assert not seen and card.tab.propensity.content.state is not ViewState.READY
    assert card.tab.result.propensity.status is S.IDENTITY_NOT_CONFIRMED


def test_disagreeing_canonical_ids_fail_closed(propensity_card):
    card, seen = propensity_card
    card.service.bridge.resolve.side_effect = lambda *a, **k: CustomerModelIdentityResult("OTHER", None, I.NOT_IN_MODEL, 0)
    search(card, None)
    assert not seen
    assert card.tab.result.propensity.status is S.IDENTITY_NOT_CONFIRMED


@pytest.mark.parametrize("flag,main,warning", [
    ("NEAR_ZERO_WITH_RECENT_FAVORITE","Оценка ненадёжна","занижать"),
    ("LOW_INDIVIDUALIZATION","12,35%","Недостаточно индивидуальных сигналов"),
    ("HISTORICAL_FEATURES_UNAVAILABLE","12,35%","недоступна")])
def test_warning_presentation(propensity_card, flag, main, warning):
    card, _ = propensity_card
    card.service.propensity_service.get = lambda *a, **k: prediction(warnings=(flag,))
    search(card)
    assert card.tab.propensity_grid.cards[0].value.text() == main
    assert warning in card.tab.propensity.message.text()
    assert not card.tab.propensity.message.isHidden()
    assert card.tab.propensity_grid.cards[1].value.text() == "с 01.09.2026 до 01.10.2026"
    assert card.tab.propensity_grid.cards[2].value.text() == "Customer Propensity"


def test_new_search_cancel_and_late_worker(propensity_card):
    card, _ = propensity_card
    search(card)
    old = card.tab.result
    old_request_id = card.tab._request_id
    entered, release = Event(), Event()
    def blocked(customer, **kwargs):
        entered.set()
        assert release.wait(4)
        return prediction()
    card.service.propensity_service.get = blocked
    card.tab.customer_input.setText("NEXT")
    card.tab.search()
    wait(entered.is_set)
    assert card.tab.propensity.content.state is ViewState.LOADING
    assert card.tab.propensity.message.isHidden() and card.tab.propensity.message.text() == ""
    assert all(c.content.value is None for c in card.tab.propensity_grid.cards)
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(True))
    QTest.qWait(20)
    assert ticks
    card.tab.cancel_search()
    assert card.tab.propensity.content.state is ViewState.CANCELLED
    assert all(c.content.value is None for c in card.tab.propensity_grid.cards)
    release.set()
    wait(lambda: card.tab._active is None)
    card.tab._finished(old_request_id, old)
    assert card.tab.propensity.content.state is ViewState.CANCELLED


def test_latest_search_only_applies_current_propensity(propensity_card):
    card, _ = propensity_card
    entered, release = Event(), Event()
    calls = []
    def blocked(customer, **kwargs):
        calls.append(customer)
        if customer == "C":
            entered.set()
            assert release.wait(4)
        return replace(prediction(probability=.2 if customer == "NEXT" else .1), canonical_customer_id=customer)
    card.service.propensity_service.get = blocked
    card.tab.customer_input.setText("C")
    card.tab.search()
    wait(entered.is_set)
    card.tab.customer_input.setText("NEXT")
    card.tab.search()
    release.set()
    wait(lambda: card.tab.result is not None)
    assert calls == ["C","NEXT"]
    assert card.tab.propensity_grid.cards[0].value.text() == "20,00%"


def test_final_handoff_revalidates_only_propensity(propensity_card):
    card, _ = propensity_card
    card.service.propensity_service.validate = lambda *a, **k: CustomerPropensity(S.STALE)
    search(card)
    assert card.tab.propensity.content.state is ViewState.STALE
    assert card.tab.raw_grid.cards and card.tab.profile.content.state is ViewState.READY


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [480, 600, 900, 1400])
def test_layout_themes_and_existing_section_order(propensity_card, app, dark, width):
    card, _ = propensity_card
    search(card)
    apply_app_theme(app, dark)
    card.host.resize(width,900)
    QTest.qWait(20)
    assert [card.tab.page_layout.itemAt(i).widget().objectName() for i in range(card.tab.page_layout.count())][2:5] == ["customerKpis","customerPropensity","customerRecommendations"]
    grid = card.tab.propensity_grid
    assert card.tab.propensity.heading.text() == "Вероятность следующего оформления заказа".upper()
    assert len(grid.cards) == grid.visible_count == grid.columns == 3
    assert [c.title.text() for c in grid.cards] == [
        "Вероятность оформления заказа в следующие 30 дней", "Прогнозируемый период", "Модель"]
    assert [grid.grid.getItemPosition(i) for i in range(3)] == [(0,0,1,1),(0,1,1,1),(0,2,1,1)]
    assert max(c.width() for c in grid.cards) - min(c.width() for c in grid.cards) <= 1
    assert grid.cards[0].geometry().left() == 0
    assert grid.cards[-1].geometry().right() == grid.width() - 1
    assert grid.width() == card.tab.propensity.width()
    assert card.tab.scroll.horizontalScrollBar().maximum() == 0
    assert all(c.value.wordWrap() for c in grid.cards)
    assert card.tab.propensity.box.count() == 3  # Heading, state message and grid; no footer.
    labels = [label.text() for label in card.tab.propensity.findChildren(QLabel)]
    assert "Дата признаков" not in labels and "Статус" not in labels
    assert not any("Исторический экспериментальный прогноз" in text for text in labels)
    assert card.tab.raw_grid.cards


def test_shutdown_hides_forecast(propensity_card):
    card, _ = propensity_card
    search(card)
    card.tab.shutdown()
    assert all(c.content.value is None for c in card.tab.propensity_grid.cards)


@pytest.mark.parametrize("status", [S.READY, S.ERROR])
def test_loading_message_hidden_but_delivery_and_errors_visible(propensity_card, status):
    card, _ = propensity_card
    entered, release = Event(), Event()
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        return prediction() if status is S.READY else CustomerPropensity(S.ERROR)
    card.service.propensity_service.get = blocked
    card.tab.customer_input.setText("C")
    card.tab.search()
    try:
        wait(entered.is_set)
        assert card.tab.propensity.content.state is ViewState.LOADING
        assert len(card.tab.propensity_grid.cards) == 3
        assert all(c.value.text() == "Загрузка…" for c in card.tab.propensity_grid.cards)
        assert card.tab.propensity.message.isHidden() and card.tab.propensity.message.text() == ""
        assert card.tab.progress.isVisible()
    finally:
        release.set()
    wait(lambda: card.tab.result is not None)
    assert card.tab.propensity.content.state is (ViewState.READY if status is S.READY else ViewState.ERROR)
    assert card.tab.propensity.message.isHidden() is (status is S.READY)
    assert not card.tab.progress.isVisible()
    assert card.tab.raw_grid.cards


def test_field_widgets_are_created_only_on_gui_thread(propensity_card, monkeypatch):
    from Application.tabs.customer_card_widgets import FieldCard
    card, _ = propensity_card
    main_thread = get_ident()
    seen = []
    original = FieldCard.__init__
    def construct(self, *args, **kwargs):
        seen.append(get_ident())
        return original(self, *args, **kwargs)
    monkeypatch.setattr(FieldCard, "__init__", construct)
    search(card)
    assert seen and set(seen) == {main_thread}


def test_close_during_propensity_load_finishes_thread_cooperatively(propensity_card):
    card, _ = propensity_card
    entered, release = Event(), Event()
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        return prediction()
    card.service.propensity_service.get = blocked
    card.tab.customer_input.setText("C")
    card.tab.search()
    try:
        wait(entered.is_set)
        card.host.close()
        assert card.tab._closing and card.tab._active.cancelled.is_set()
        assert all(c.content.value is None for c in card.tab.propensity_grid.cards)
    finally:
        release.set()
    wait(lambda: card.tab._thread is None)
    assert card.tab.result is None and not card.host.isVisible()
