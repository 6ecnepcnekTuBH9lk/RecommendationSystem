import os
import time
from dataclasses import replace
from threading import Event, get_ident
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtNetwork import QNetworkReply
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget

from Application.customer_card_data import CustomerCardResult, ProductContent, ViewState
from Application.model.customer_identity_bridge import CustomerModelIdentityResult, CustomerModelIdentityStatus as S
from Application.model.raw_bpr_inference import RawBprRecommendation
from Application.photo import photo_processing as photos
from Application.customer_history_service import HistoryPage, HistoryStatus
from Application.tabs.customer_results_tab import CustomerResultsTab
from Application.theme.apply_theme import apply_app_theme, prepare_app_theme


def ready(*, generation="g", name="Синтетический товар", score=1.123456789):
    identity = CustomerModelIdentityResult("C", "A", S.MATCHED_VIA_MERGE, 1, "g", "r", "v")
    return CustomerCardResult(ViewState.READY, "Оригинальные рекомендации BPR-MF.", identity, generation, tuple(
        ProductContent(RawBprRecommendation(i, f"ORIGINAL-{i}", score / i), name) for i in range(1, 11)))


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    prepare_app_theme(app)
    return app


@pytest.fixture
def card(app):
    host = QMainWindow()
    host.status_label, host.status_icon = QLabel(), QLabel()
    host._status_reset_timer = QTimer(host)
    host._status_reset_timer.setSingleShot(True)
    service = Mock()
    service.load.return_value = ready()
    service.is_current.return_value = True
    def validate(result):
        if result.state in (ViewState.READY, ViewState.EMPTY) and not service.is_current(result):
            return replace(result, state=ViewState.STALE, products=())
        return result
    service.validate_result.side_effect = validate
    tab = CustomerResultsTab(host, service=service, history_lookup=lambda customer, **kw: HistoryPage(HistoryStatus.EMPTY, customer))
    host.setCentralWidget(tab)
    host.resize(1250, 900)
    host.show()
    QTest.qWait(20)
    yield host, tab, service
    tab.shutdown()
    if tab._thread is not None:
        assert tab._thread.wait(5000)
    host._status_reset_timer.stop()
    host.close()
    host.deleteLater()
    QTest.qWait(20)


def wait_for(predicate, timeout=4000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


def test_sections_vertical_order_initial_states_and_no_legacy_dependencies(card, monkeypatch):
    _, tab, service = card
    assert [tab.page_layout.itemAt(i).widget().objectName() for i in range(tab.page_layout.count())] == [
        "customerSearch", "customerProfile", "customerKpis", "customerRecommendations", "customerHistory", "customerPropensity"]
    assert tab.recommendations.content.state is ViewState.INITIAL
    assert tab.profile.content.state is ViewState.INITIAL
    assert tab.kpis.content.state is ViewState.NOT_CONNECTED
    assert tab.history.content.state is ViewState.INITIAL
    assert tab.propensity.content.state is ViewState.UNAVAILABLE
    assert not tab.export_button.isEnabled()
    assert not tab.history_filter.isEnabled()
    assert [tab.history_filter.itemText(i) for i in range(4)] == ["Заказы", "Просмотры", "Избранное", "Прочее"]
    assert tab._thread is None
    assert not tab.product_grid.cards
    service.load.assert_not_called()
    assert all(c.value.text() == "Не подключено" for c in tab.kpi_grid.cards)
    assert all(c.value.text() == "Клиент не выбран" for c in tab.profile_fields.cards)


@pytest.mark.parametrize("trigger", ["button", "enter"])
def test_search_button_enter_background_and_original_top10(card, trigger):
    host, tab, service = card
    main_id = get_ident()
    seen = []
    def load(value, cancelled):
        seen.append(get_ident())
        return ready()
    service.load.side_effect = load
    tab.customer_input.setText("current-id")
    if trigger == "button":
        tab.search_button.click()
    else:
        QTest.keyClick(tab.customer_input, Qt.Key.Key_Return)
    wait_for(lambda: tab.result is not None)
    assert seen and seen[0] != main_id
    service.load.assert_called_once()
    assert tab.result.state is ViewState.READY
    assert len(tab.product_grid.cards) == 10
    assert [c.content.recommendation.rank for c in tab.product_grid.cards] == list(range(1, 11))
    assert tab.product_grid.cards[0].code.text() == "Код: ORIGINAL-1"
    assert tab.product_grid.cards[0].content.recommendation.raw_score == 1.123456789
    assert "1.123456789" in tab.product_grid.cards[0].score.toolTip()
    assert tab.canonical_id.text() == "Canonical MindboxID: C"
    assert tab.model_id.text() == "ID в модели BPR-MF: A"
    assert not tab.progress.isVisible() and not tab.cancel_button.isEnabled()
    assert "BPR" in host.status_label.text()


@pytest.mark.parametrize("value", ["", " ", " A", "A "])
def test_invalid_input_is_inline_without_worker_or_fake_data(card, value):
    _, tab, service = card
    tab.customer_input.setText(value)
    tab.search()
    assert tab.recommendations.content.state is ViewState.ERROR
    assert tab._thread is None and not tab.product_grid.cards
    service.load.assert_not_called()


@pytest.mark.parametrize("state", [ViewState.ERROR, ViewState.UNAVAILABLE, ViewState.STALE, ViewState.EMPTY])
def test_independent_future_sections_survive_recommendation_failures(card, state):
    _, tab, _ = card
    tab.apply_result(CustomerCardResult(state, "Синтетическое состояние"))
    assert not tab.product_grid.cards
    assert tab.profile.content.state is ViewState.INITIAL
    assert tab.kpis.content.state is ViewState.NOT_CONNECTED
    assert tab.history.content.state is ViewState.INITIAL
    assert tab.propensity.content.state is ViewState.UNAVAILABLE
    assert all(c.value.text() != "0" for c in tab.kpi_grid.cards)
    assert "недоступен" in tab.propensity.message.text()


def test_topk_uses_local_prefix_and_preserves_card_objects(card):
    _, tab, service = card
    tab.apply_result(ready())
    cards = tuple(tab.product_grid.cards)
    for i, count in enumerate((1, 3, 5, 10)):
        tab.topk.setCurrentIndex(i)
        QTest.qWait(10)
        assert tab.product_grid.visible_count == count
        assert tuple(tab.product_grid.cards) == cards
        assert [c.content.recommendation.rank for c in cards[:count]] == list(range(1, count + 1))
    service.load.assert_not_called()


def test_missing_names_photos_and_catalog_keep_every_recommendation(card):
    _, tab, _ = card
    tab.apply_result(ready(name=""))
    assert len(tab.product_grid.cards) == 10
    assert all(c.name.text() == "Название недоступно" and c.photo.text() == "Нет фото" for c in tab.product_grid.cards)


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [480, 900, 1600])
def test_responsive_layout_both_themes_no_horizontal_page_scroll(card, app, dark, width):
    host, tab, _ = card
    tab.apply_result(ready(name="Очень длинное синтетическое наименование товара " * 6))
    apply_app_theme(app, dark)
    host.resize(width, 850)
    QTest.qWait(60)
    assert tab.scroll.horizontalScrollBar().maximum() == 0
    assert 1 <= tab.product_grid.columns <= 5
    assert 1 <= tab.kpi_grid.columns <= 4
    assert tab.page.width() <= tab.scroll.viewport().width()
    for section in (tab.profile, tab.kpis, tab.recommendations, tab.history, tab.propensity):
        assert section.width() <= tab.page.width()


def test_superseded_search_coalesces_requests_and_never_publishes_old_result(card, app):
    host, tab, service = card
    entered, release = Event(), Event()
    calls = []
    def load(value, cancelled):
        calls.append(value)
        if value == "first":
            entered.set()
            assert release.wait(4)
        return ready(name=value)
    service.load.side_effect = load
    tab.customer_input.setText("first")
    tab.search()
    wait_for(entered.is_set)
    thread = tab._thread
    host.resize(800, 700)
    apply_app_theme(app, True)
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(True))
    QTest.qWait(20)
    assert ticks and tab.result is None
    for name in ("second", "third", "latest"):
        tab.customer_input.setText(name)
        tab.search()
    release.set()
    wait_for(lambda: tab.result is not None)
    assert calls == ["first", "latest"]
    assert tab._thread is thread
    assert tab.product_grid.cards[0].name.text() == "latest"


def test_cancel_and_worker_exception_do_not_leave_loading(card):
    _, tab, service = card
    entered, release = Event(), Event()
    def load(*args):
        entered.set()
        assert release.wait(4)
        raise ValueError("PRIVATE_CUSTOMER PRIVATE_PATH traceback")
    service.load.side_effect = load
    tab.customer_input.setText("A")
    tab.search()
    wait_for(entered.is_set)
    tab.cancel_button.click()
    release.set()
    wait_for(lambda: tab._active is None)
    assert tab.recommendations.content.state is ViewState.CANCELLED
    assert not tab.progress.isVisible() and not tab.product_grid.cards
    service.load.side_effect = ValueError("PRIVATE_PATH")
    tab.search()
    wait_for(lambda: tab.result is not None)
    assert tab.result.state is ViewState.ERROR
    assert "PRIVATE" not in tab.search_status.text()


def test_generation_or_identity_freshness_failure_never_displays_products(card):
    _, tab, service = card
    service.is_current.return_value = False
    tab.customer_input.setText("A")
    tab.search()
    wait_for(lambda: tab.result is not None)
    assert tab.result.state is ViewState.STALE and not tab.product_grid.cards
    tab.apply_result(ready(generation="different"))
    assert tab.result.state is ViewState.STALE and not tab.product_grid.cards


def test_source_change_during_queued_signal_handoff_is_rechecked_on_gui(card):
    _, tab, service = card
    service.is_current.side_effect = [True, False]  # Worker was current; GUI handoff is no longer current.
    tab.customer_input.setText("A")
    tab.search()
    wait_for(lambda: tab.result is not None)
    assert tab.result.state is ViewState.STALE and not tab.product_grid.cards
    assert service.is_current.call_count == 2


def test_close_during_search_waits_cooperatively_without_thread_destruction(card):
    host, tab, service = card
    entered, release = Event(), Event()
    def load(*args):
        entered.set()
        assert release.wait(4)
        return ready()
    service.load.side_effect = load
    tab.customer_input.setText("A")
    tab.search()
    wait_for(entered.is_set)
    thread = tab._thread
    host.close()
    assert tab._closing and thread.isRunning()
    assert tab._active.cancelled.is_set()
    release.set()
    wait_for(lambda: tab._thread is None)
    assert not host.isVisible() and tab.result is None


def test_new_factory_never_connects_legacy_export_or_search(app, monkeypatch):
    from Application.tabs import create_results_tab as legacy
    for name in ("_resolve_mindbox_ids", "_load_client_info", "_load_client_interactions",
                 "_load_recommendations_from_excel", "export_recommendations_to_excel"):
        monkeypatch.setattr(legacy, name, lambda *a: pytest.fail("No legacy data/export path"))
    host = QMainWindow()
    host.tabs = QTabWidget(host)
    legacy.create_result_widgets_tab(host)
    assert isinstance(host.results_tab, CustomerResultsTab)
    assert host.tabs.widget(0) is host.results_tab
    host.results_tab.export_button.click()
    assert host.results_tab._thread is None
    host.close()
    host.deleteLater()
    QTest.qWait(10)


class FakeReply:
    def __init__(self, url, generation, data):
        self.url, self.generation, self.data = url, generation, data
    def property(self, key):
        return self.url if key == "img_url" else self.generation
    def attribute(self, key):
        return 200
    def header(self, key):
        return "image/png"
    def error(self):
        return QNetworkReply.NetworkError.NoError
    def readAll(self):
        return self.data
    def deleteLater(self):
        pass


def png():
    image = QPixmap(100, 200)
    image.fill(Qt.GlobalColor.blue)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return data


def test_late_photo_same_url_cannot_touch_new_client_targets(card):
    _, tab, _ = card
    tab.apply_result(ready())
    label = tab.product_grid.cards[0].photo
    url = "https://example.invalid/photo"
    tab._img_targets[url] = [(label, None)]
    tab._img_inflight.add(url)
    photos._on_image_loaded(tab, FakeReply(url, tab._img_gen - 1, png()))
    assert tab._img_targets[url] == [(label, None)] and url in tab._img_inflight
    assert label.pixmap() is None or label.pixmap().isNull()
    photos._on_image_loaded(tab, FakeReply(url, tab._img_gen, png()))
    assert label.pixmap() is not None and not label.pixmap().isNull()
    assert url not in tab._img_inflight


def test_photo_queue_deduplicates_and_pixmap_preserves_aspect_ratio(card, monkeypatch):
    host, tab, _ = card
    tab.apply_result(ready())
    calls = []
    monkeypatch.setattr(photos, "_request_image", lambda owner, url, gen: calls.append(url))
    first, second = [c.photo for c in tab.product_grid.cards[:2]]
    for label in (first, second):
        photos._set_photo_label(tab, label, "https://example.invalid/photo", tab._img_gen)
    assert calls == ["https://example.invalid/photo"]
    photos._on_image_loaded(tab, FakeReply(calls[0], tab._img_gen, png()))
    host.resize(500, 700)
    QTest.qWait(30)
    pixmap = first.pixmap()
    assert pixmap.width() <= first.width() and pixmap.height() <= first.height()
    assert abs(pixmap.width() / pixmap.height() - 0.5) < 0.02
