"""Real synthetic composition shown through the existing customer Qt worker."""

from dataclasses import replace
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QComboBox

from Application.customer_card_data import SectionContent, ViewState
from Application.hybrid_recommendations import HistoricalItem, HybridRecommendationService, HybridResult, compose
from Application.purchased_products import PurchaseEvidence
from Application.recommendation_rules import save_rules
from Application.theme.apply_theme import apply_app_theme
from Application.tabs.customer_card_widgets import StateSection
from tests.test_hybrid_recommendations import BASE, checked_purchases, raw_result, rules, write_catalog
from tests.tabs.test_customer_contact_search_ui import app as app, card as card, search, wait
from tests.test_customer_propensity_service import prediction


@pytest.fixture
def explained(card, tmp_path):
    catalog = write_catalog(tmp_path, [(f"{n:06d}", {"Коллекция": "NOS", "НазваниеНаСайте": f"Брюки хлопок {n}"}) for n in range(1, 13)])
    raw = replace(raw_result(50), model_user_id="A")
    historical = tuple(HistoricalItem(r.model_item_code, f"Историческое имя {r.rank}", "Весна-Лето 2025") for r in raw.recommendations[:10])
    hybrid = Mock()
    hybrid.is_current.return_value = True
    threads = []
    def load(identity, *args, **kwargs):
        threads.append(get_ident())
        items = compose(raw, {f"source-{n}": BASE for n in range(36, 46)}, catalog, rules(catalog), checked_purchases(raw))
        purchases = PurchaseEvidence("READY", historical=PurchaseEvidence(
            "READY", checked_codes=frozenset(r.model_item_code for r in raw.recommendations)))
        return HybridResult("READY", "Готово", items, raw_bpr=raw, purchases=purchases,
                            historical_items=historical, displayed_raw=raw.recommendations[:10], raw_status="READY")
    hybrid.get.side_effect = load
    card.service.hybrid = hybrid
    card.service.propensity_service = SimpleNamespace(
        get=lambda customer, **kwargs: replace(prediction(), canonical_customer_id=customer),
        validate=lambda value, **kwargs: value,
    )
    return card, raw, hybrid, threads


def test_three_sections_real_components_top10_and_row_selection(explained):
    card, raw, hybrid, threads = explained
    search(card)
    tab = card.tab
    assert threads and all(t != get_ident() for t in threads)
    names = [tab.page_layout.itemAt(i).widget().objectName() for i in range(tab.page_layout.count())]
    assert names == ["customerSearch", "customerProfile", "customerKpis", "customerPropensity",
                     "customerRecommendations", "hybridCalculations", "customerHybridRecommendations", "customerHistory"]
    assert all(section.message.isHidden() for section in (tab.recommendations, tab.calculations, tab.hybrid_recommendations))
    assert len(tab.raw_grid.cards) == len(tab.product_grid.cards) == tab.calculations.model.rowCount() == 10
    assert tuple(c.content.recommendation for c in tab.raw_grid.cards) == raw.recommendations[:10]
    assert tab.raw_grid.cards[0].name.text() == "Историческое имя 1"
    assert tab.raw_grid.cards[0].collection.text() == "Коллекция: Весна-Лето 2025"
    assert tab.raw_grid.cards[0].photo.text() == "Нет фото"
    assert tab.propensity.message.isHidden()
    assert len(tab.propensity_grid.cards) == 3
    assert tab.propensity_grid.cards[0].value.text() == "12,35%"
    for widget in tab.raw_grid.cards:
        score = widget.content.recommendation.raw_score
        assert widget.score.text() == f"Исходный BPR-score: {score:.3f}"
        assert repr(score) in widget.score.toolTip()
    assert all(c.content.hybrid_item is not None for c in tab.product_grid.cards)
    assert [c.code.text() for c in tab.product_grid.cards] == [f"Код: {item.target_item_code}" for item in tab.result.hybrid.items]
    assert not any(combo.findText("Топ-10") >= 0 for combo in tab.findChildren(QComboBox))
    c = tab.calculations
    assert c.feature_model.rowCount() == 10
    for row, item in enumerate(tab.result.hybrid.items):
        assert c.model.rows[row][-1] == item.final_score
        source, = item.sources
        assert c.model.rows[row][3] == source.source_raw_rank
        assert c.model.rows[row][5] == 1 - (source.source_raw_rank - 1) / 49
        assert tab.product_grid.cards[row].content.hybrid_item.sources[0].calculation.final_score == item.final_score
        assert tab.product_grid.cards[row].score.text() == f"Итоговый рейтинг: {item.final_score:.3f}"
        assert repr(item.final_score) in tab.product_grid.cards[row].score.toolTip()
    assert c.items[0].sources[0].calculation.source_count == 50
    assert c.model.rows[0][2].startswith("source-36\n")
    assert c.alternative_model.rowCount() == 0
    assert not c.other_sources.isVisible() and not c.alternatives.isVisible()
    c.table.setCurrentIndex(c.model.index(1, 0))
    assert c.table.currentIndex().row() == 1
    assert c.feature_model.rows[0][1] == c.items[1].sources[0].calculation.features[2].historical_value
    assert not c.alternatives.isVisible()
    hybrid.get.assert_called_once()
    card.service.inference.get_recommendations.assert_not_called()


@pytest.mark.parametrize("failure", ["PURCHASES_NOT_READY", "UNAVAILABLE", "EMPTY"])
def test_raw_section_stays_ready_when_hybrid_unavailable(explained, failure):
    card, raw, hybrid, _ = explained
    original = hybrid.get.side_effect
    hybrid.get.side_effect = lambda *a, **kw: replace(original(*a, **kw), status=failure, items=(), message="Причина недоступности")
    search(card)
    tab = card.tab
    assert len(tab.raw_grid.cards) == 10 and not tab.product_grid.cards
    assert tab.recommendations.content.state.value == "ready"
    assert tab.calculations.model.rowCount() == 0 and not tab.calculations.table.isVisible()
    assert tab.profile.content.state.value == "ready"
    assert tuple(c.content.recommendation for c in tab.raw_grid.cards) == raw.recommendations[:10]
    tab.service.bridge.is_current.return_value = False
    stale = tab.service.validate_result(tab.result)
    tab.apply_result(stale)
    assert not tab.raw_grid.cards and not tab.product_grid.cards and not tab.calculations.items


@pytest.mark.parametrize("width", [480, 1600])
@pytest.mark.parametrize("dark", [False, True])
def test_both_grids_and_tables_adapt_to_theme_and_width(explained, app, width, dark):
    card, _, _, _ = explained
    search(card)
    apply_app_theme(app, dark)
    card.host.resize(width, 900)
    tab = card.tab
    wait(lambda: all(view.horizontalHeader().length() == view.viewport().width() and not view._fit_timer.isActive()
                     for view in (tab.calculations.table, tab.calculations.features)))
    assert tab.scroll.horizontalScrollBar().maximum() == 0
    assert tab.raw_grid.columns == tab.product_grid.columns
    assert 1 <= tab.raw_grid.columns <= 5
    if width == 1600:
        assert tab.raw_grid.columns == 5
    assert tab.calculations.table.horizontalScrollBar().maximum() == 0, (
        tab.calculations.table.viewport().width(), tab.calculations.table.horizontalHeader().length(),
        [tab.calculations.table.columnWidth(n) for n in range(10)])
    assert tab.calculations.table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    for table in (tab.calculations.table, tab.calculations.features):
        last = table.model().rowCount() - 1
        assert table.rowViewportPosition(last) + table.rowHeight(last) <= table.viewport().height()


def test_shared_photo_cache_and_cancel_clear_all_sections(explained):
    card, _, hybrid, _ = explained
    pm = QPixmap(4, 4)
    pm.fill(Qt.GlobalColor.green)
    url = "https://kanzler-style.ru/upload/synthetic-shared.jpg"
    card.tab._img_cache[url] = pm
    original = hybrid.get.side_effect
    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, items=tuple(replace(item, photo_url=url) for item in result.items))
    hybrid.get.side_effect = load
    card.service._references.return_value = {f"source-{n}": ("ignored current name", url) for n in range(10)}
    search(card)
    assert all(c.photo.pixmap() is not None for c in card.tab.raw_grid.cards)
    assert all(c.photo.pixmap() is not None for c in card.tab.product_grid.cards)
    card.tab.cancel_search()
    assert not card.tab.raw_grid.cards and not card.tab.product_grid.cards
    assert card.tab.calculations.model.rowCount() == 0
    assert card.tab.calculations.feature_model.rowCount() == 0
    assert not card.tab._img_targets


@pytest.mark.parametrize("purchased_count,uncertain_code", [(5, None), (12, "source-12")])
def test_production_service_filters_raw_in_worker_and_matches_eligible_sources(card, tmp_path, purchased_count, uncertain_code):
    catalog = write_catalog(tmp_path, [("000001", {"Коллекция": "NOS"})])
    raw = replace(raw_result(40), model_user_id="A")
    metadata = {r.model_item_code: {**BASE, "НазваниеНаСайте": f"Исторический товар {r.rank}"}
                for r in raw.recommendations}
    card.service.inference.get_candidate_sources.return_value = raw, metadata
    scope = PurchaseEvidence("READY", frozenset(r.model_item_code for r in raw.recommendations[:purchased_count]),
                             frozenset((uncertain_code,)) if uncertain_code else frozenset(),
                             checked_codes=frozenset(r.model_item_code for r in raw.recommendations))
    purchases = Mock()
    purchases.get.return_value = PurchaseEvidence("READY", historical=scope)
    purchase_threads = []
    def current(*args, **kwargs):
        purchase_threads.append(get_ident())
        return True
    purchases.is_current.side_effect = current
    rules_path = tmp_path / "rules.json"
    save_rules(rules(catalog), rules_path)
    card.service.hybrid = HybridRecommendationService(catalog_path=catalog.path, categories_path=catalog.categories_path,
                                                      rules_path=rules_path, purchases=purchases)
    search(card)
    tab = card.tab
    first_rank = purchased_count + int(uncertain_code is not None) + 1
    assert [w.content.recommendation.rank for w in tab.raw_grid.cards] == list(range(first_rank, first_rank + 10))
    assert tab.raw_grid.cards[0].name.text() == f"Исторический товар {first_rank}"
    assert all(w.content.recommendation is raw.recommendations[w.content.recommendation.rank - 1] for w in tab.raw_grid.cards)
    assert tab.result.hybrid.items == compose(raw, metadata, catalog, rules(catalog), purchases.get.return_value)
    assert tab.result.hybrid.items[0].sources[0].source_raw_rank == first_rank
    assert tab.calculations.model.rows[0][3] == first_rank
    assert [w.rank.text() for w in tab.raw_grid.cards] == [f"№{rank}" for rank in range(1, 11)]
    assert [w.content.display_rank for w in tab.raw_grid.cards] == list(range(1, 11))
    assert tab.result.hybrid.raw_bpr is raw
    assert purchase_threads and all(t != get_ident() for t in purchase_threads)
    purchases.get.assert_called_once()
    card.service.inference.get_candidate_sources.assert_called_once()


def test_history_change_during_worker_handoff_clears_raw_without_erasing_profile(explained):
    card, _, hybrid, _ = explained
    calls = []
    def current(*args, **kwargs):
        calls.append(get_ident())
        return len(calls) == 1
    hybrid.raw_is_current.side_effect = current
    search(card)
    tab = card.tab
    assert tab.result.raw_state.value == "stale" and not tab.raw_grid.cards
    assert tab.result.profile is not None and tab.profile.content.state.value == "ready"
    assert tab.history.customer_id == "C"
    assert calls and all(t != get_ident() for t in calls)


def test_saved_multi_source_result_still_displays_supplied_alternatives(explained):
    card, _, _, _ = explained
    search(card)
    first, second = card.tab.result.hybrid.items[:2]
    legacy = replace(first, sources=(*first.sources, *second.sources))
    section = card.tab.calculations
    section.set_items((legacy,))
    assert section.other_sources.isVisible()
    assert section.alternative_model.rowCount() == 1
    assert section.alternative_model.rows[0][0] == second.sources[0].source_raw_rank
    section.other_sources.click()
    assert section.alternatives.isVisible()
    section.set_items((first,))
    assert not section.other_sources.isVisible() and not section.alternatives.isVisible()
    assert section.alternative_model.rowCount() == 0


def test_saved_result_without_explanation_never_fabricates_calculations(explained):
    card, _, _, _ = explained
    search(card)
    item = card.tab.result.hybrid.items[0]
    section = card.tab.calculations
    section.set_items((replace(item, sources=(replace(item.sources[0], calculation=None),)),))
    assert section.content.state.value == "unavailable"
    assert section.model.rowCount() == section.feature_model.rowCount() == 0
    assert not section.table.isVisible() and not section.other_sources.isVisible()


def test_initial_messages_hidden_without_layout_gaps_or_global_state_change(explained):
    card, _, _, _ = explained
    tab = card.tab
    for section in (tab.recommendations, tab.calculations, tab.hybrid_recommendations):
        assert section.content.state is ViewState.INITIAL
        assert section.content.message and section.message.text() == section.content.message
        assert section.heading.isVisible() and section.message.isHidden()
        assert section.box.itemAt(section.box.indexOf(section.message)).isEmpty()
    assert tab.search_status.isVisible() and tab.search_status.text()
    ordinary = StateSection("Другой раздел", SectionContent(ViewState.LOADING, "Обычное сообщение"), name="ordinary")
    assert not ordinary.message.isHidden()  # Base behavior for profile/history is unchanged.
    ordinary.deleteLater()


@pytest.mark.parametrize("state", [ViewState.INITIAL, ViewState.LOADING, ViewState.READY, ViewState.CANCELLED,
                                 ViewState.EMPTY, ViewState.UNAVAILABLE, ViewState.ERROR, ViewState.STALE])
def test_search_state_controls_only_message_visibility(explained, state):
    card, _, _, _ = explained
    tab = card.tab
    message = "Причина состояния или общий статус поиска"
    tab._display_state(state, message)
    informative = state in (ViewState.EMPTY, ViewState.UNAVAILABLE, ViewState.ERROR, ViewState.STALE)
    for section in (tab.recommendations, tab.calculations, tab.hybrid_recommendations):
        assert section.content.state is state and section.content.message == message
        assert section.message.text() == message
        assert section.message.isVisible() is informative
        assert section.box.itemAt(section.box.indexOf(section.message)).isEmpty() is not informative
    assert tab.search_status.isVisible() and tab.search_status.text() == message
    assert tab.progress.isVisible() is (state is ViewState.LOADING)


@pytest.mark.parametrize("state", [ViewState.EMPTY, ViewState.UNAVAILABLE, ViewState.ERROR, ViewState.STALE])
def test_result_diagnostics_survive_after_successful_recommendations(explained, state):
    card, _, _, _ = explained
    search(card)
    tab = card.tab
    message = "Проверка покупок или источник рекомендаций недоступны"
    result = replace(tab.result, state=state, raw_state=state, message=message, raw_message=message,
                     products=(), raw_products=(), hybrid=replace(tab.result.hybrid, items=()))
    tab.apply_result(result)
    for section in (tab.recommendations, tab.calculations, tab.hybrid_recommendations):
        assert section.content.state is state and section.message.isVisible() and section.message.text()
    assert tab.recommendations.message.text() == tab.hybrid_recommendations.message.text() == message
    assert not tab.raw_grid.cards and not tab.product_grid.cards and not tab.calculations.items


def test_new_search_and_cancel_clear_results_without_duplicate_statuses(explained):
    card, _, hybrid, _ = explained
    search(card)
    tab = card.tab
    original = hybrid.get.side_effect
    started, release = Event(), Event()
    def delayed(identity, *args, **kwargs):
        started.set()
        release.wait(3)
        return original(identity, *args, **kwargs)
    hybrid.get.side_effect = delayed
    try:
        tab.search_type.setCurrentIndex(tab.search_type.findData(None))
        tab.customer_input.setText("NEXT")
        tab.search_button.click()
        wait(started.is_set)
        assert tab.search_status.isVisible() and tab.progress.isVisible()
        for section in (tab.recommendations, tab.calculations, tab.hybrid_recommendations):
            assert section.content.state is ViewState.LOADING and section.message.isHidden()
        assert not tab.raw_grid.cards and not tab.product_grid.cards and not tab.calculations.items
        tab.cancel_search()
        for section in (tab.recommendations, tab.calculations, tab.hybrid_recommendations):
            assert section.content.state is ViewState.CANCELLED and section.message.isHidden()
        assert "отмен" in tab.search_status.text()
    finally:
        release.set()
    wait(lambda: tab._active is None)
    assert tab.result is None and not tab.calculations.items
    hybrid.get.side_effect = original
    search(card, kind=None)
    assert tab.calculations.model.rowCount() == 10
    assert all(section.message.isHidden() for section in (tab.recommendations, tab.calculations, tab.hybrid_recommendations))


def test_filtered_kind_top10_and_rule_change_at_worker_handoff(card, tmp_path, monkeypatch):
    catalog = write_catalog(tmp_path, [(f"{n:06d}", {"Коллекция": "NOS"}) for n in range(12)])
    raw = replace(raw_result(30), model_user_id="A")
    metadata = {rec.model_item_code: {**BASE, "НазваниеНаСайте": f"Исторический товар {rec.rank}",
                                     "Коллекция": "Весна-Лето 2024",
                                     "ВидНоменклатуры": "Носки" if rec.rank <= 10 else "Брюки"}
                for rec in raw.recommendations}
    card.service.inference.get_candidate_sources.return_value = raw, metadata
    purchases = Mock()
    purchases.get.return_value = checked_purchases(raw)
    purchases.is_current.return_value = True
    path = tmp_path / "rules.json"
    save_rules(rules(catalog), path)
    service = HybridRecommendationService(catalog_path=catalog.path, categories_path=catalog.categories_path,
                                          rules_path=path, purchases=purchases)
    card.service.hybrid = service
    from Application import hybrid_recommendations as backend
    load_rules = backend.load_rules
    load_threads = []
    def loaded(path):
        load_threads.append(get_ident())
        return load_rules(path)
    monkeypatch.setattr(backend, "load_rules", loaded)
    search(card)
    tab = card.tab
    assert [w.content.recommendation.rank for w in tab.raw_grid.cards] == list(range(11, 21))
    assert [w.rank.text() for w in tab.raw_grid.cards] == [f"№{n}" for n in range(1, 11)]
    assert all(metadata[w.content.recommendation.model_item_code]["ВидНоменклатуры"] == "Брюки" for w in tab.raw_grid.cards)
    assert tab.raw_grid.cards[0].name.text() == "Исторический товар 11"
    assert tab.raw_grid.cards[0].collection.text() == "Коллекция: Весна-Лето 2024"
    assert tab.raw_grid.cards[0].score.text() == f"Исходный BPR-score: {raw.recommendations[10].raw_score:.3f}"
    assert tab.calculations.model.rows[0][3] == 11
    assert tab.result.hybrid.items[0].sources[0].calculation.source_count == 30
    purchases.get.assert_called_once()
    card.service.inference.get_candidate_sources.assert_called_once()
    assert load_threads and all(t != get_ident() for t in load_threads)
    # load(), worker.run(), then validate_delivery(): change rules at the final handoff.
    original = service.raw_is_current
    changed, checks = [], []
    def at_handoff(result, **kwargs):
        checks.append(get_ident())
        if len(checks) == 3:
            save_rules(replace(rules(catalog), kinds=("Носки",)), path)
            changed.append(get_ident())
        return original(result, **kwargs)
    service.raw_is_current = at_handoff
    search(card)
    assert tab.result.raw_state is ViewState.STALE and not tab.raw_grid.cards and not tab.product_grid.cards
    assert tab.profile.content.state is ViewState.READY and tab.history.customer_id == "C"
    assert changed and all(t != get_ident() for t in changed)
