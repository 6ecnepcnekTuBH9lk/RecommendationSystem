"""Real synthetic composition shown through the existing customer Qt worker."""

from dataclasses import replace
from threading import get_ident
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QComboBox

from Application.hybrid_recommendations import HistoricalItem, HybridRecommendationService, HybridResult, compose
from Application.purchased_products import PurchaseEvidence
from Application.recommendation_rules import save_rules
from Application.theme.apply_theme import apply_app_theme
from tests.test_hybrid_recommendations import BASE, raw_result, rules, write_catalog
from tests.tabs.test_customer_contact_search_ui import app as app, card as card, search


@pytest.fixture
def explained(card, tmp_path):
    catalog = write_catalog(tmp_path, [(f"{n:06d}", {"Коллекция": "NOS", "НазваниеНаСайте": f"Брюки хлопок {n}"}) for n in range(1, 13)])
    raw = replace(raw_result(40), model_user_id="A")
    historical = tuple(HistoricalItem(r.model_item_code, f"Историческое имя {r.rank}", "Весна-Лето 2025") for r in raw.recommendations[:10])
    hybrid = Mock()
    hybrid.is_current.return_value = True
    threads = []
    def load(identity, *args, **kwargs):
        threads.append(get_ident())
        items = compose(raw, {"source-36": BASE, "source-37": BASE}, catalog, rules(catalog), PurchaseEvidence("READY"))
        purchases = PurchaseEvidence("READY", historical=PurchaseEvidence(
            "READY", checked_codes=frozenset(r.model_item_code for r in raw.recommendations)))
        return HybridResult("READY", "Готово", items, raw_bpr=raw, purchases=purchases,
                            historical_items=historical, displayed_raw=raw.recommendations[:10])
    hybrid.get.side_effect = load
    card.service.hybrid = hybrid
    return card, raw, hybrid, threads


def test_three_sections_real_components_top10_and_row_selection(explained):
    card, raw, hybrid, threads = explained
    search(card)
    tab = card.tab
    assert threads and all(t != get_ident() for t in threads)
    names = [tab.page_layout.itemAt(i).widget().objectName() for i in range(tab.page_layout.count())]
    assert names == ["customerSearch", "customerProfile", "customerKpis", "customerPropensity",
                     "customerRecommendations", "customerHybridRecommendations", "customerHistory", "hybridCalculations"]
    assert len(tab.raw_grid.cards) == len(tab.product_grid.cards) == tab.calculations.model.rowCount() == 10
    assert tuple(c.content.recommendation for c in tab.raw_grid.cards) == raw.recommendations[:10]
    assert tab.raw_grid.cards[0].name.text() == "Историческое имя 1"
    assert tab.raw_grid.cards[0].collection.text() == "Коллекция: Весна-Лето 2025"
    assert tab.raw_grid.cards[0].photo.text() == "Нет фото"
    assert tab.propensity.message.isHidden()
    assert tab.propensity.box.itemAt(2).widget().cards and len(tab.propensity.box.itemAt(2).widget().cards) == 4
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
        assert c.model.rows[row][3] == 37
        assert c.model.rows[row][5] == 1 - 36 / 39
        assert tab.product_grid.cards[row].content.hybrid_item.sources[0].calculation.final_score == item.final_score
        assert tab.product_grid.cards[row].score.text() == f"Итоговый рейтинг: {item.final_score:.3f}"
        assert repr(item.final_score) in tab.product_grid.cards[row].score.toolTip()
    assert "40 товаров" in c.bpr_details.text() and "source-36" in c.summary.text()
    assert c.alternative_model.rows[0][0] == 38
    c.other_sources.click()
    assert c.alternatives.isVisible()
    c.table.setCurrentIndex(c.model.index(1, 0))
    assert tab.result.hybrid.items[1].target_item_code in c.summary.text()
    assert not c.alternatives.isVisible()
    hybrid.get.assert_called_once()
    card.service.inference.get_recommendations.assert_not_called()


@pytest.mark.parametrize("failure", ["RULES_NOT_READY", "PURCHASES_NOT_READY", "UNAVAILABLE", "EMPTY"])
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
    QTest.qWait(50)
    tab = card.tab
    assert tab.scroll.horizontalScrollBar().maximum() == 0
    assert tab.raw_grid.columns == tab.product_grid.columns
    assert 1 <= tab.raw_grid.columns <= 5
    if width == 1600:
        assert tab.raw_grid.columns == 5
    if width == 480:
        assert tab.calculations.table.horizontalScrollBar().maximum() > 0
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
    assert card.tab.calculations.summary.text() == ""
    assert not card.tab._img_targets


def test_production_service_filters_raw_in_worker_and_keeps_full_hybrid_sources(card, tmp_path):
    catalog = write_catalog(tmp_path, [("000001", {"Коллекция": "NOS"})])
    raw = replace(raw_result(40), model_user_id="A")
    metadata = {r.model_item_code: {**BASE, "НазваниеНаСайте": f"Исторический товар {r.rank}"}
                for r in raw.recommendations}
    card.service.inference.get_candidate_sources.return_value = raw, metadata
    scope = PurchaseEvidence("READY", frozenset(r.model_item_code for r in raw.recommendations[:12]),
                             frozenset(("source-12",)), checked_codes=frozenset(r.model_item_code for r in raw.recommendations))
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
    assert [w.content.recommendation.rank for w in tab.raw_grid.cards] == list(range(14, 24))
    assert tab.raw_grid.cards[0].name.text() == "Исторический товар 14"
    assert all(w.content.recommendation is raw.recommendations[w.content.recommendation.rank - 1] for w in tab.raw_grid.cards)
    assert tab.result.hybrid.items == compose(raw, metadata, catalog, rules(catalog), purchases.get.return_value)
    assert tab.result.hybrid.items[0].sources[0].source_raw_rank == 1
    assert tab.calculations.model.rows[0][3] == 1
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
