"""Formula layout and adaptive calculation tables, synthetic results only."""

from dataclasses import replace

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QStyleOptionViewItem

from Application.customer_card_data import ViewState
from Application.tabs.hybrid_calculation_widgets import FEATURE_NAMES
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_customer_contact_search_ui import app as app, card as card, search, wait
from tests.tabs.test_recommendation_sections import explained as explained


def test_formula_frame_content_centering_and_exact_section_structure(explained):
    card, _, _, _ = explained
    search(card)
    section = card.tab.calculations
    assert section.heading.text() == "РАСЧЕТ ОЦЕНОК ГИБРИДНОЙ МОДЕЛИ"
    assert section.formula.property("class") == "statisticsCard" and section.formula.isVisible()
    labels = section.formula.findChildren(QLabel)
    text = "\n".join(label.text().replace("<sub>i</sub>", "ᵢ").replace("<br>", "\n") for label in labels)
    for expected in ("S = 0.30 × R + 0.30 × B + 0.40 × M", "S — базовая оценка", "R — нормализованный ранг",
                     "B — нормализованный BPR-score", "M — сходство", "S_final = (S + 0.05 × I) / 1.05",
                     "I = 1", "I = 0", "ко всем кандидатам", "S_final = S", "R = 1 − p / max(1, N − 1)",
                     "начиная с нуля", "N — количество товаров", "не меняет исходные позиции",
                     "B = (score − score_min) / (score_max − score_min)", "score_max = score_min, B = 1",
                     "M = Σ(wᵢ × mᵢ)", "сумма вкладов десяти признаков", "частичным", "вклад равен нулю",
                     "совместимость вида номенклатуры и пола", "не является вероятностью покупки"):
        assert expected in text
    assert all(label.alignment() == Qt.AlignmentFlag.AlignCenter and label.wordWrap() for label in labels)
    assert sum(label.textFormat() is Qt.TextFormat.RichText for label in labels) == 2
    assert labels[0].font().pointSizeF() > labels[1].font().pointSizeF()
    visible = [section.box.itemAt(i).widget() for i in range(section.box.count())
               if section.box.itemAt(i).widget() is not None and section.box.itemAt(i).widget().isVisible()]
    assert visible == [section.heading, section.formula, section.table, section.features]
    for removed in ("summary", "bpr_details", "final_details", "merge_note"):
        assert not hasattr(section, removed)


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [480, 1100, 1800])
def test_tables_fill_viewport_and_wrap_without_scroll_or_lost_rows(explained, app, dark, width):
    card, _, _, _ = explained
    search(card)
    apply_app_theme(app, dark)
    card.host.resize(width, 950)
    section = card.tab.calculations
    wait(lambda: all(view.horizontalHeader().length() == view.viewport().width()
                     and view.horizontalScrollBar().maximum() == 0 for view in (section.table, section.features)))
    QTest.qWait(30)
    assert card.tab.scroll.horizontalScrollBar().maximum() == 0
    for view, count in ((section.table, 10), (section.features, 6)):
        assert view.model().columnCount() == count
        assert sum(view.columnWidth(c) for c in range(count)) == view.viewport().width()
        assert view.horizontalScrollBarPolicy() is Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert view.verticalScrollBarPolicy() is Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert not view.horizontalScrollBar().isVisible() and not view.verticalScrollBar().isVisible()
        last = view.model().rowCount() - 1
        assert view.rowViewportPosition(last) + view.rowHeight(last) <= view.viewport().height()
        for row in range(view.model().rowCount()):
            for col in range(count):
                option = QStyleOptionViewItem()
                option.initFrom(view)
                required = view.itemDelegate().sizeHint(option, view.model().index(row, col)).height()
                assert view.rowHeight(row) >= required
    assert section.table.columnWidth(1) > section.table.columnWidth(0)
    assert section.features.columnWidth(1) > section.features.columnWidth(3)
    for label in section.formula.findChildren(QLabel):
        assert label.height() >= label.heightForWidth(label.width())


def test_selection_resize_and_copy_use_existing_immutable_data_only(explained, monkeypatch, app):
    card, _, _, _ = explained
    search(card)
    section = card.tab.calculations
    saved = section.items
    assert section.table.currentIndex().row() == 0
    def forbidden(*args, **kwargs):
        raise AssertionError("UI must not calculate recommendations or similarity")
    monkeypatch.setattr("Application.hybrid_recommendations.similarity", forbidden)
    monkeypatch.setattr("Application.hybrid_recommendations.compose", forbidden)
    card.service.inference.get_candidate_sources.side_effect = forbidden
    for row, item in enumerate(saved):
        section.table.setCurrentIndex(section.model.index(row, 0))
        source = item.sources[0]
        assert section.model.rows[row][-1] == item.final_score
        assert section.model.rows[row][3:5] == (source.source_raw_rank, source.source_raw_score)
        by_key = {feature.feature: feature for feature in source.calculation.features}
        expected = tuple((name, by_key[key].historical_value or "Не указано", by_key[key].target_value or "Не указано",
                          by_key[key].weight, by_key[key].match, by_key[key].contribution) for key, name in FEATURE_NAMES.items())
        assert section.feature_model.rows == expected
    section.table.setFocus()
    QTest.keyClick(section.table, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    values = section.model.rows[-1]
    assert QApplication.clipboard().text() == "\t".join(repr(v) if isinstance(v, float) else str(v) for v in values)
    card.host.resize(700, 950)
    QTest.qWait(50)
    assert section.items is saved
    card.service.inference.get_recommendations.assert_not_called()


def test_long_names_unbroken_identifiers_and_missing_features(explained):
    card, _, _, _ = explained
    search(card)
    section = card.tab.calculations
    item = section.items[0]
    source = item.sources[0]
    features = tuple(replace(f, historical_value="", target_value="") if n == 0 else f
                     for n, f in enumerate(source.calculation.features))
    source = replace(source, source_name="Историческое название " * 12, source_model_item_code="SOURCE" * 20,
                     calculation=replace(source.calculation, features=features))
    item = replace(item, target_name="Длинное название актуального товара " * 10, sources=(source,))
    section.set_items((item,))
    card.host.resize(480, 950)
    QTest.qWait(60)
    assert section.table.rowHeight(0) > 54
    assert section.feature_model.rowCount() == 10
    assert any(row[1] == row[2] == "Не указано" for row in section.feature_model.rows)
    assert source.source_model_item_code in section.model.rows[0][2]
    assert section.model.data(section.model.index(0, 4), Qt.ItemDataRole.ToolTipRole) == repr(source.source_raw_score)
    assert section.table.horizontalScrollBar().maximum() == 0


@pytest.mark.parametrize("state", [ViewState.EMPTY, ViewState.UNAVAILABLE, ViewState.STALE, ViewState.CANCELLED])
def test_clear_releases_models_and_selection_without_blank_text_blocks(explained, state):
    card, _, _, _ = explained
    search(card)
    section = card.tab.calculations
    section.clear(state, "Состояние раздела")
    assert section.content.state is state and section.message.text() == "Состояние раздела"
    assert not section.items and not section.table.currentIndex().isValid()
    assert section.model.rowCount() == section.feature_model.rowCount() == section.alternative_model.rowCount() == 0
    assert not section.table.isVisible() and not section.features.isVisible()
    assert section.formula.isVisible()


def test_repeated_resize_and_theme_switch_do_not_accumulate_layout_work(explained, app):
    card, _, _, _ = explained
    search(card)
    section = card.tab.calculations
    for width, dark in ((1800, True), (480, False), (1100, True), (480, True), (1800, False)):
        apply_app_theme(app, dark)
        card.host.resize(width, 950)
        QTest.qWait(50)
    wait(lambda: not section.table._fit_timer.isActive() and not section.features._fit_timer.isActive())
    for view in (section.table, section.features):
        assert view.horizontalHeader().length() == view.viewport().width()
        assert view.horizontalScrollBar().maximum() == 0
