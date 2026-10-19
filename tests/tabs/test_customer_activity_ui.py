"""Activity cards on synthetic SQLite, using the existing serial Qt worker."""

from contextlib import closing, contextmanager
from dataclasses import replace
import sqlite3
from threading import Event, get_ident
from decimal import Decimal
from types import SimpleNamespace

import pytest

from Application import customer_history_index as index
from Application.customer_activity_service import ActivityStatus, CurrencyActivity
from Application.customer_activity_presentation import ACTIVITY_FIELDS
from Application.customer_card_data import ViewState
from Application.customer_contact_search import ContactType
from Application.model.customer_identity_bridge import CustomerModelIdentityResult, CustomerModelIdentityStatus
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_customer_contact_search_ui import app as app, card as card, found, search, wait
from tests.test_customer_activity_service import history as history, prepared, purchase
from tests.test_customer_history_index import action


@pytest.fixture
def activity_card(card, history, monkeypatch):
    source, service = history
    prepared(history, [purchase()], [action(f"a{i}") for i in range(5)])
    card.service.activity_service = service
    card.service.raw_root = source[0]
    main = get_ident()
    reader, threads = index._read_index, []
    @contextmanager
    def checked(path):
        threads.append(get_ident())
        assert get_ident() != main
        with reader(path) as db:
            yield db
    monkeypatch.setattr(index, "_read_index", checked)
    return card, source, service, threads


@pytest.mark.parametrize("kind", [None, ContactType.PHONE, ContactType.EMAIL, ContactType.DISCOUNT_CARD])
def test_four_search_modes_show_same_activity_in_background(activity_card, kind):
    card, _, _, threads = activity_card
    assert [c.value.text() for c in card.tab.kpi_grid.cards] == ["Клиент не выбран"]*8
    search(card, kind)
    result = card.tab.result.activity
    assert result.canonical_customer_id == "C" and result.orders == 1 and result.views == 5
    assert threads and all(thread != get_ident() for thread in threads)
    assert [c.title.text() for c in card.tab.kpi_grid.cards] == list(ACTIVITY_FIELDS)
    assert [c.value.text() for c in card.tab.kpi_grid.cards] == ["1", "2,5", "123,45 ₽", "123,45 ₽", "5", "0", "0 дн.", "01.01.2026 - 01.01.2026"]
    assert card.tab.kpis.message.isHidden() and card.tab.kpis.message.text() == ""
    assert card.tab.kpi_grid.cards[1].title.text() == "Количество покупок"
    assert card.tab.kpi_grid.cards[1].content.note is None
    assert card.tab.kpi_grid.cards[6].content.note == "По состоянию на 01.01.2026."
    assert card.tab.propensity.content.state is ViewState.UNAVAILABLE


@pytest.mark.parametrize("failure", ["not_in_model", "inference_exception"])
def test_activity_does_not_require_bpr_membership_or_success(activity_card, failure):
    card, _, _, _ = activity_card
    if failure == "not_in_model":
        card.service.bridge.resolve.side_effect = lambda c, **kw: CustomerModelIdentityResult(
            "C", None, CustomerModelIdentityStatus.NOT_IN_MODEL, 0)
    else:
        card.service.inference.get_recommendations.side_effect = RuntimeError("PRIVATE")
    search(card)
    assert card.tab.result.activity.status is ActivityStatus.READY
    assert card.tab.kpi_grid.cards[0].value.text() == "1"
    assert card.tab.profile.content.state is ViewState.READY
    assert card.tab.history.customer_id == "C"
    assert not card.tab.raw_grid.cards
    if failure == "not_in_model":
        card.service.inference.get_recommendations.assert_not_called()


def test_activity_error_does_not_destroy_recommendations_profile_or_history(activity_card, monkeypatch):
    card, _, service, _ = activity_card
    monkeypatch.setattr(service, "get", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("PRIVATE")))
    search(card)
    assert card.tab.result.activity.status is ActivityStatus.ERROR
    assert card.tab.raw_grid.cards and card.tab.profile.content.state is ViewState.READY
    assert card.tab.history.customer_id == "C"
    assert all(c.content.value is None for c in card.tab.kpi_grid.cards)


def test_generation_change_before_delivery_invalidates_only_activity(activity_card, monkeypatch):
    card, source, service, _ = activity_card
    validate, calls = service.validate, []
    def changing(result, **kwargs):
        checked = validate(result, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            with closing(sqlite3.connect(source[1].index_path(raw_root=source[0]))) as db, db:
                db.execute("UPDATE metadata SET value=? WHERE key='generation'", ("f"*32,))
        return checked
    monkeypatch.setattr(service, "validate", changing)
    search(card)
    assert len(calls) >= 2
    assert card.tab.result.activity.status is ActivityStatus.STALE
    assert all(c.value.text() == "Данные устарели" for c in card.tab.kpi_grid.cards)
    assert card.tab.raw_grid.cards and card.tab.profile.content.state is ViewState.READY


def test_repeated_search_cancels_old_activity_and_clears_old_numbers(activity_card, monkeypatch):
    card, _, service, _ = activity_card
    started, release = Event(), Event()
    get = service.get
    def blocked(customer, **kwargs):
        if customer == "C":
            started.set()
            assert release.wait(5)
        return get(customer, **kwargs)
    monkeypatch.setattr(service, "get", blocked)
    card.tab.customer_input.setText("C")
    card.tab.search()
    wait(started.is_set)
    card.tab.customer_input.setText("C2")
    card.tab.search()
    assert all(c.value.text() == "Загрузка…" for c in card.tab.kpi_grid.cards)
    assert card.tab.kpis.message.isHidden() and card.tab.kpis.message.text() == ""
    release.set()
    wait(lambda: card.tab.result is not None)
    assert card.tab.result.activity.canonical_customer_id == "C2"
    assert card.tab.kpi_grid.cards[0].value.text() == "0"
    assert card.tab.history.customer_id == "C2"
    assert card.tab.kpis.message.isHidden() and card.tab.kpis.message.text() == ""
    card.tab._clear_result()
    assert card.tab.kpis.message.isHidden() and card.tab.kpis.message.text() == ""
    assert all(c.value.text() == "Клиент не выбран" for c in card.tab.kpi_grid.cards)


def test_wrong_activity_customer_cannot_replace_other_sections(activity_card, monkeypatch):
    card, _, service, _ = activity_card
    get = service.get
    monkeypatch.setattr(service, "get", lambda *a, **kw: replace(get(*a, **kw), canonical_customer_id="OTHER"))
    search(card)
    assert card.tab.result.activity.status is ActivityStatus.STALE
    assert card.tab.raw_grid.cards and card.tab.profile.content.state is ViewState.READY


def test_historical_customer_alias_uses_confirmed_canonical_customer(activity_card):
    card, _, _, _ = activity_card
    card.service.profile_service.get_profile.side_effect = lambda *a, **kw: found("C")
    card.service.bridge.resolve.side_effect = lambda *a, **kw: CustomerModelIdentityResult(
        "C", "A", CustomerModelIdentityStatus.MATCHED_VIA_MERGE, 1, "g", "r", "v")
    search(card, None, value="A")
    assert card.tab.result.activity.canonical_customer_id == "C"
    assert card.tab.result.activity.orders == 1 and card.tab.history.customer_id == "C"


def test_incomplete_product_identity_only_disables_unique_view_count(activity_card):
    card, source, _, _ = activity_card
    with closing(sqlite3.connect(source[1].index_path(raw_root=source[0]))) as db, db:
        db.execute("UPDATE product_occurrences SET selected_namespace=NULL,selected_value=NULL WHERE role='product'")
    search(card)
    fields = card.tab.kpi_grid.cards
    assert fields[0].value.text() == "1" and fields[4].value.text() == "5"
    assert fields[7].value.text() == "01.01.2026 - 01.01.2026"
    assert card.tab.result.activity.unique_viewed_products is None
    assert fields[7].content.note is None and card.tab.raw_grid.cards


@pytest.mark.parametrize("close", [False, True])
def test_cancel_or_close_during_activity_query_discards_response(activity_card, monkeypatch, close):
    card, _, service, _ = activity_card
    started, release = Event(), Event()
    get = service.get
    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return get(*args, **kwargs)
    monkeypatch.setattr(service, "get", blocked)
    card.tab.customer_input.setText("C")
    card.tab.search()
    wait(started.is_set)
    if close:
        card.host.close()
    else:
        card.tab.cancel_search()
        assert all(c.value.text() == "Загрузка отменена" for c in card.tab.kpi_grid.cards)
    release.set()
    wait(lambda: card.tab._thread is None if close else card.tab._active is None)
    assert card.tab.result is None


def test_two_money_lines_use_currency_code_if_font_lacks_tenge(activity_card, monkeypatch):
    from Application.tabs import customer_results_tab as ui
    card, _, service, _ = activity_card
    get = service.get
    def currencies(*a, **kw):
        result = get(*a, **kw)
        return replace(result, money=(*result.money, CurrencyActivity("KZT", 1, Decimal("340000"), Decimal("340000"))))
    monkeypatch.setattr(service, "get", currencies)
    monkeypatch.setattr(ui, "QFontMetrics", lambda font: SimpleNamespace(inFontUcs4=lambda code: False))
    search(card)
    assert card.tab.kpi_grid.cards[2].value.text() == "123,45 ₽\n340 000 KZT"
    assert "340 000 KZT" in card.tab.kpi_grid.cards[3].value.text()
    assert card.tab.result.activity.money[1].currency == "KZT"


def test_unconfirmed_period_is_unavailable_without_separate_source_labels(activity_card, monkeypatch):
    card, _, service, _ = activity_card
    get = service.get
    monkeypatch.setattr(service, "get", lambda *a, **kw: replace(
        get(*a, **kw), orders_coverage_complete=False, actions_coverage_complete=False))
    search(card)
    period = card.tab.kpi_grid.cards[7]
    assert period.title.text() == "Период анализа" and period.value.text() == "Недоступно"
    assert card.tab.kpis.message.isHidden() and card.tab.kpis.message.text() == ""


@pytest.mark.parametrize("dark,width", [(False, 480), (True, 480), (False, 1500), (True, 1500)])
def test_activity_cards_keep_themes_order_and_responsive_layout(activity_card, app, dark, width):
    card, _, _, _ = activity_card
    apply_app_theme(app, dark)
    card.host.resize(width, 1000)
    search(card)
    assert len(card.tab.kpi_grid.cards) == 8
    assert 1 <= card.tab.kpi_grid.columns <= 4
    assert all(c.value.wordWrap() for c in card.tab.kpi_grid.cards)
    assert [c.title.text() for c in card.tab.kpi_grid.cards] == list(ACTIVITY_FIELDS)
    assert card.tab.kpis.message.isHidden()
    assert not any(c.title.text() in ("Приобретённые единицы", "Уникальные просмотренные товары")
                   for c in card.tab.kpi_grid.cards)
