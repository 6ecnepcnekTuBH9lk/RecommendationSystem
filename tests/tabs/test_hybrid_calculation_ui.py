"""Formula layout and adaptive calculation tables, synthetic results only."""

from dataclasses import replace
from time import monotonic

import pytest
from PyQt6.QtCore import Qt, QTimer, QT_VERSION_STR, PYQT_VERSION_STR, qVersion
from PyQt6.QtGui import QFontInfo
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QStyleOptionViewItem

from Application.customer_card_data import ViewState
from Application.tabs.hybrid_calculation_widgets import FEATURE_NAMES
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_customer_contact_search_ui import app as app, card as card, search, wait
from tests.tabs.test_recommendation_sections import explained as explained


def formula_geometry(section):
    return tuple((label.geometry().getRect(), label.heightForWidth(label.width()), label.font().toString())
                 for label in section.formula.findChildren(QLabel))


def geometry_diagnostic(section):
    labels = []
    for index, label in enumerate(section.formula.findChildren(QLabel)):
        labels.append({
            "index": index, "text": label.text(), "format": label.textFormat().name,
            "rect": label.geometry().getRect(), "contents": label.contentsRect().getRect(),
            "heightForWidth": label.heightForWidth(label.width()), "font": label.font().toString(),
            "resolved_font": QFontInfo(label.font()).family(),
            "minimum": (label.minimumWidth(), label.minimumHeight()),
            "maximum": (label.maximumWidth(), label.maximumHeight()),
            "size_hint": (label.sizeHint().width(), label.sizeHint().height()),
            "minimum_size_hint": (label.minimumSizeHint().width(), label.minimumSizeHint().height()),
        })
    ancestors = []
    parent = section.formula
    while parent is not None:
        layout = parent.layout()
        ancestors.append({"type": type(parent).__name__, "name": parent.objectName(),
                          "rect": parent.geometry().getRect(), "heightForWidth": parent.heightForWidth(parent.width()),
                          "minimum_height": parent.minimumHeight(), "maximum_height": parent.maximumHeight(),
                          "size_hint": (parent.sizeHint().width(), parent.sizeHint().height()),
                          "layout_rect": layout.geometry().getRect() if layout is not None else None,
                          "layout_height_for_width": layout.totalHeightForWidth(parent.width()) if layout is not None else None})
        parent = parent.parentWidget()
    return {"Qt_build": QT_VERSION_STR, "Qt_runtime": qVersion(), "PyQt": PYQT_VERSION_STR,
            "platform": QApplication.platformName(),
            "dpi": section.formula.logicalDpiY(), "labels": labels, "ancestors": ancestors,
            "tables": tuple((view.size().width(), view.size().height(), view.viewport().width(),
                             view.horizontalHeader().length(), view._fit_timer.isActive())
                            for view in (section.table, section.features))}


def wait_for_calculation_layout(section, *, timeout=5):
    """Observe completed layout turns, without activating or repairing layouts in the test."""
    deadline = monotonic() + timeout
    previous = None
    initial = None
    while monotonic() < deadline:
        QTest.qWait(10)
        geometry = formula_geometry(section)
        tables = tuple((view.geometry().getRect(), view.viewport().size().width(),
                        view.viewport().size().height()) for view in (section.table, section.features))
        current = (section.formula.geometry().getRect(), geometry, tables)
        if initial is None:
            initial = current
        ready = all(not view._fit_timer.isActive()
                    and view.horizontalHeader().length() == view.viewport().width()
                    and view.horizontalScrollBar().maximum() == 0 for view in (section.table, section.features))
        ready = ready and bool(geometry) and all(rect[2] > 0 and rect[3] >= required
                                                 for rect, required, _font in geometry)
        if ready and current == previous:
            return
        previous = current
    pytest.fail(f"Calculation layout did not settle without clipped text within {timeout}s: "
                f"initial={initial!r}; final={geometry_diagnostic(section)!r}")


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
@pytest.mark.parametrize("width", [478, 480, 482, 1100, 1800])
def test_tables_fill_viewport_and_wrap_without_scroll_or_lost_rows(explained, app, dark, width):
    card, _, _, _ = explained
    search(card)
    apply_app_theme(app, dark)
    card.host.resize(width, 950)
    section = card.tab.calculations
    wait_for_calculation_layout(section)
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
        assert label.hasHeightForWidth(), geometry_diagnostic(section)
        assert label.height() >= label.heightForWidth(label.width()), geometry_diagnostic(section)


def test_layout_wait_rejects_persistently_clipped_formula(explained):
    card, _, _, _ = explained
    search(card)
    section = card.tab.calculations
    wait_for_calculation_layout(section)
    label = section.formula.findChildren(QLabel)[1]
    label.setFixedHeight(1)
    with pytest.raises(pytest.fail.Exception, match="without clipped text") as failure:
        wait_for_calculation_layout(section, timeout=0.1)
    assert label.height() < label.heightForWidth(label.width())
    assert label.text() in str(failure.value)
    for field in ("index", "heightForWidth", "resolved_font", "ancestors", "tables", "dpi", "platform"):
        assert field in str(failure.value)


def test_layout_wait_includes_queued_formula_geometry(explained):
    card, _, _, _ = explained
    search(card)
    section = card.tab.calculations
    wait_for_calculation_layout(section)
    label = section.formula.findChildren(QLabel)[1]
    minimum, maximum = label.minimumHeight(), label.maximumHeight()
    label.setFixedHeight(1)
    completed = []

    def finish_layout():
        label.setMinimumHeight(minimum)
        label.setMaximumHeight(maximum)
        completed.append(True)

    QTimer.singleShot(30, finish_layout)
    wait_for_calculation_layout(section)
    assert completed and label.height() >= label.heightForWidth(label.width())


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
        wait_for_calculation_layout(section)
    wait(lambda: not section.table._fit_timer.isActive() and not section.features._fit_timer.isActive())
    for view in (section.table, section.features):
        assert view.horizontalHeader().length() == view.viewport().width()
        assert view.horizontalScrollBar().maximum() == 0
