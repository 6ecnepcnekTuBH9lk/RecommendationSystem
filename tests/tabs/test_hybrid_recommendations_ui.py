"""Hybrid delivery through the existing serial worker and settings controls."""

from threading import Event, get_ident
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import QThreadPool

from Application.customer_card_data import ViewState
from Application.hybrid_recommendations import HybridItem, HybridResult, MatchSource
from Application.recommendation_rules import RecommendationRules, load_rules, save_rules
from Application.tabs import recommendation_rules_widget as settings
from Application.tabs.customer_card_widgets import ProductCard
from Application.customer_card_data import ProductContent
from Application.model.raw_bpr_inference import RawBprRecommendation
from tests.test_hybrid_recommendations import write_catalog
from tests.tabs.test_customer_contact_search_ui import app as app, card as card, search, wait, found


def hybrid_result(customer="C", status="READY"):
    source = MatchSource("original-source", "Исходные брюки", -1.123456789, 37, .85, "weighted_attributes_v1", .7)
    item = HybridItem(f"target-{customer}", "Актуальные брюки", "Брюки", "NOS", 10, "", 1, .7, (source,))
    return HybridResult(status, "Рекомендации из актуального ассортимента", (item,) if status == "READY" else ())


@pytest.mark.parametrize("state", ["RULES_NOT_READY", "PURCHASES_NOT_READY", "EMPTY", "READY"])
def test_hybrid_background_and_profile_history_independent(card, state):
    gui = get_ident()
    calls = []
    hybrid = Mock()
    def get(identity, *args, **kwargs):
        calls.append((identity.canonical_customer_id, get_ident()))
        return hybrid_result(identity.canonical_customer_id, state)
    hybrid.get.side_effect = get
    hybrid.is_current.return_value = True
    card.service.hybrid = hybrid
    search(card)
    assert calls and calls[0][1] != gui
    assert card.tab.profile.content.state is ViewState.READY
    assert card.tab.result.profile == found()
    assert card.tab.history.customer_id == "C"
    if state == "READY":
        widget = card.tab.product_grid.cards[0]
        assert widget.code.text() == "Код: target-C"
        assert widget.rank.text() == "№1"
        assert widget.content.recommendation.raw_score == -1.123456789
        assert "Оценка подбора" in widget.score.text()
        assert "-1.123456789" in widget.score.toolTip()
    else:
        assert not card.tab.product_grid.cards
    card.service.inference.get_recommendations.assert_not_called()


def test_switch_client_during_hybrid_cancels_old_and_drops_stale_result(card):
    started, release = Event(), Event()
    hybrid = Mock()
    cancellations = []
    def get(identity, *args, **kwargs):
        if identity.canonical_customer_id == "C":
            started.set()
            release.wait(2)
            cancellations.append(kwargs["cancelled"]())
        return hybrid_result(identity.canonical_customer_id)
    hybrid.get.side_effect = get
    hybrid.is_current.return_value = True
    card.service.hybrid = hybrid
    card.tab.search_type.setCurrentIndex(card.tab.search_type.findData(None))
    card.tab.customer_input.setText("C")
    card.tab.search_button.click()
    wait(started.is_set)
    card.tab.customer_input.setText("NEW")
    card.tab.search_button.click()
    release.set()
    wait(lambda: card.tab.result is not None and card.tab.result.identity.canonical_customer_id == "NEW")
    assert cancellations == [True]
    assert [c.code.text() for c in card.tab.product_grid.cards] == ["Код: target-NEW"]
    assert card.tab.result.profile == found("NEW")


def test_rules_change_at_handoff_blocks_hybrid_without_erasing_profile(card):
    hybrid = Mock()
    hybrid.get.return_value = hybrid_result()
    checks = []
    def current(*args, **kwargs):
        checks.append(get_ident())
        return len(checks) == 1
    hybrid.is_current.side_effect = current
    card.service.hybrid = hybrid
    search(card)
    assert card.tab.result.state is ViewState.STALE
    assert card.tab.result.profile == found()
    assert not card.tab.product_grid.cards
    assert all(t != get_ident() for t in checks)


def test_close_during_hybrid_is_cooperative_and_thread_finishes(card):
    started = Event()
    hybrid = Mock()
    def get(identity, *args, **kwargs):
        started.set()
        while not kwargs["cancelled"]():
            Event().wait(.01)
        return HybridResult("CANCELLED", "Поиск отменён.")
    hybrid.get.side_effect = get
    card.service.hybrid = hybrid
    card.tab.customer_input.setText("C")
    card.tab.search_button.click()
    wait(started.is_set)
    card.tab.shutdown()
    wait(lambda: card.tab._thread is None or not card.tab._thread.isRunning())
    assert card.tab.result is None


def test_target_card_does_not_overwrite_raw_source(app):
    item = hybrid_result().items[0]
    rec = RawBprRecommendation(37, "original-source", -1.123456789)
    widget = ProductCard(ProductContent(rec, item.target_name, collection=item.target_collection, hybrid_item=item))
    assert widget.code.text() == "Код: target-C"
    assert widget.content.recommendation == rec
    assert widget.content.hybrid_item.sources[0].source_raw_rank == 37
    widget.deleteLater()


@pytest.fixture
def rules_widget(app, tmp_path, monkeypatch):
    catalog = write_catalog(tmp_path, [("000001", {"Коллекция": "NOS"}), ("000002", {"Коллекция": "Зима"})])
    threads = []
    def read():
        threads.append(get_ident())
        return catalog
    monkeypatch.setattr(settings, "RecommendationCatalog", lambda: Mock(load=read))
    path = tmp_path / "hybrid_rules.json"
    widget = settings.RecommendationRulesWidget(settings_path=path)
    widget.show()
    wait(lambda: widget._task is None and widget._initialized)
    yield widget, path, catalog, threads
    QThreadPool.globalInstance().waitForDone(3000)
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_settings_no_implicit_unrestricted_first_use_and_background_read(rules_widget):
    widget, path, _, threads = rules_widget
    assert not path.exists() and not widget.kinds.selected() and not widget.collections.selected()
    assert all(t != get_ident() for t in threads)
    widget.save_button.click()
    assert not path.exists() and "Выберите" in widget.message.text()
    widget.kinds.set_checked(True)
    widget.collections.set_checked(True)
    widget.minimum_stock.setValue(17)
    widget.save_button.click()
    wait(lambda: widget._task is None)
    assert load_rules(path) == RecommendationRules(("Брюки",), ("NOS", "Зима"), 17)


def test_settings_refresh_retains_deleted_selection_until_explicit_removal(rules_widget):
    widget, path, catalog, _ = rules_widget
    save_rules(RecommendationRules(("Брюки",), ("Удалённый",), 2), path)
    widget.refresh()
    wait(lambda: widget._task is None)
    assert widget.collections.selected() == ("Удалённый",)
    assert "удалённые" in widget.message.text()
    widget.save()
    wait(lambda: widget._task is None)
    assert load_rules(path).collections == ("Удалённый",)
    assert widget.save_button.isEnabled() and "удалены" in widget.message.text()
    widget.collections.set_checked(False)
    for n in range(widget.collections.items.count()):
        item = widget.collections.items.item(n)
        if item.text() == "NOS":
            from PyQt6.QtCore import Qt
            item.setCheckState(Qt.CheckState.Checked)
    widget.save_button.click()
    wait(lambda: widget._task is None)
    assert load_rules(path).collections == ("NOS",)


def test_settings_draft_survives_catalog_refresh(rules_widget):
    widget, path, _, _ = rules_widget
    widget.kinds.set_checked(True)
    widget.collections.set_checked(True)
    widget.minimum_stock.setValue(23)
    widget.refresh()
    wait(lambda: widget._task is None)
    assert widget.kinds.selected() == ("Брюки",)
    assert widget.collections.selected() == ("NOS", "Зима")
    assert widget.minimum_stock.value() == 23 and not path.exists()
