"""Real small SQLite indexes only; no production data or model training."""

from copy import deepcopy
import os
from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QPushButton

from Application import customer_history_index as index
from Application.customer_card_data import CustomerCardResult, ProductContent, ViewState
from Application.customer_history_records import HistoryKind as K
from Application.customer_history_service import CustomerHistoryService, HistoryPage, HistoryStatus as S
from Application.customer_profile_service import CustomerProfileResult, ProfileStatus
from Application.model.customer_identity_bridge import CustomerModelIdentityResult, CustomerModelIdentityStatus as I
from Application.model.raw_bpr_inference import RawBprRecommendation
from Application.tabs.customer_history_widgets import CustomerHistorySection
from Application.tabs.customer_results_tab import CustomerResultsTab
from Application.theme.apply_theme import apply_app_theme, prepare_app_theme
from tests.test_customer_history_index import action, build, order, source as source_fixture, update_rows


def wait(predicate):
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


def result(customer="C", *, bpr=True):
    identity = CustomerModelIdentityResult(customer, None, I.NOT_IN_MODEL, 0)
    products = ()
    if bpr:
        identity = CustomerModelIdentityResult(customer, customer, I.MATCHED_EXACT, 1, "g", "r", "v")
        products = (ProductContent(RawBprRecommendation(1, "ORIGINAL", 1.23456789012345), "Товар"),)
    return CustomerCardResult(ViewState.READY if bpr else ViewState.UNAVAILABLE, "BPR", identity, "g", products,
                              CustomerProfileResult(ProfileStatus.NOT_FOUND, canonical_customer_id=customer))


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    prepare_app_theme(app)
    return app


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    source = source_fixture.__wrapped__(tmp_path)
    views = [action(f"view-{i:03d}") for i in range(61)]
    favorite, unknown = action("favorite"), action("other")
    favorite["actionTemplate"]["ids"]["systemName"] = "DobavlenieProduktaVSpisokVOperaciiDobavlenie"
    unknown["actionTemplate"]["ids"]["systemName"] = "UnknownType"
    conflict = action("owner-conflict")
    foreign = action("owner-conflict", customer="OTHER", dateTimeUtc="2026-01-01T14:00:00Z")
    foreign["actionTemplate"]["ids"]["systemName"] = "UnknownType"
    update_rows(source, "actions", views + [favorite, unknown, conflict, foreign])
    first, changed = order(), deepcopy(order())
    changed["lines"][1]["quantity"] = 3
    update_rows(source, "orders", [first, changed, deepcopy(first)])
    assert build(source).status is index.HistoryIndexStatus.READY
    def forbidden(*args, **kwargs):
        pytest.fail("UI must never build, scan raw exports or fully verify the index")
    monkeypatch.setattr(index, "iter_export", forbidden)
    monkeypatch.setattr(index, "build_customer_history_index", forbidden)
    monkeypatch.setattr(index.CustomerHistoryIndexService, "build_index", forbidden)
    monkeypatch.setattr(index, "_validate_storage", forbidden)
    backend = CustomerHistoryService(cache_dir=source[1].cache_dir)
    calls = []
    def lookup(customer, **kwargs):
        calls.append((get_ident(), customer, kwargs))
        return backend.get_customer_history(customer, **kwargs)
    return SimpleNamespace(source=source, lookup=lookup, calls=calls)


@pytest.fixture
def card(app, synthetic):
    host = QMainWindow()
    service = SimpleNamespace(raw_root=synthetic.source[0], profile_observer=None,
                              validate_result=lambda value: value, validate_profile=lambda value: value)
    def load(customer, cancelled):
        value = result(customer)
        if service.profile_observer:
            service.profile_observer(value.profile)
        return value
    service.load = load
    tab = CustomerResultsTab(host, service=service, history_lookup=synthetic.lookup)
    host.setCentralWidget(tab)
    host.resize(1000, 850)
    host.show()
    yield SimpleNamespace(host=host, tab=tab, service=service, data=synthetic)
    tab.shutdown()
    if tab._thread is not None:
        assert tab._thread.wait(5000)
    host.close()
    host.deleteLater()
    QTest.qWait(20)


def loaded(card, customer="C"):
    card.tab.customer_input.setText(customer)
    card.tab.search()
    wait(lambda: card.tab.result is not None and card.tab.history._active is None)
    return card.tab.history


def category(history, kind):
    history.category.setCurrentIndex(history.category.findData(kind))
    wait(lambda: history._active is None)


def test_sqlite_background_profile_and_original_bpr_are_independent(card):
    history = loaded(card)
    assert history.property("historyStatus") == S.PARTIAL.value
    assert len(history.cards) == 1
    assert card.data.calls and all(t != get_ident() for t, _, _ in card.data.calls)
    assert card.data.calls[0][2] == dict(kinds=(K.ORDER,), limit=50, cursor=None, raw_root=card.service.raw_root)
    assert card.tab.result.products == result().products
    assert card.tab.product_grid.cards[0].content.recommendation.raw_score == 1.23456789012345
    assert card.tab.profile.content.state is ViewState.EMPTY
    assert history.generation and history.source_version
    assert "2026-01-01" in history.coverage.text()


def test_pagination_exact_fifty_no_duplicates_or_repeated_page_requests(card):
    history = loaded(card)
    category(history, K.VIEW)
    assert len(history.cards) == 50 and history.cursor
    before = len(card.data.calls)
    history.load_more()
    history.load_more()
    wait(lambda: history._active is None)
    assert len(card.data.calls) == before + 1
    assert len(history.cards) == len({c.event.event_key for c in history.cards}) == 62
    assert history.cursor is None and history.more.isHidden()
    assert history.content.state is ViewState.READY and "Частичная" in history.message.text()
    history.load_more()
    assert len(card.data.calls) == before + 1


def test_all_categories_and_empty_customer_reset_cursor(card):
    history = loaded(card)
    category(history, K.VIEW)
    old_cursor = history.cursor
    category(history, K.FAVORITE)
    assert old_cursor and history.cursor is None
    assert [c.event.source_event_id for c in history.cards] == ["favorite"]
    category(history, K.UNCLASSIFIED)
    # UNCLASSIFIED belongs to the other candidate owner; C matches VIEW only.
    assert {c.event.source_event_id for c in history.cards} == {"other"}
    loaded(card, "EMPTY")
    assert not history.cards and history.property("historyStatus") == S.EMPTY.value
    assert "покрытии" in history.message.text()
    assert history.cursor is None
    assert all(kwargs["cursor"] is None for _, _, kwargs in card.data.calls)


def test_expand_all_order_snapshots_precise_lines_statuses_and_origins(card):
    history = loaded(card)
    event = history.cards[0]
    assert event.details is None and not event.findChildren(QPlainTextEdit)
    event.expand.click()
    assert event.details is not None and not event.details.isHidden()
    buttons = event.details.findChildren(QPushButton)
    assert len(buttons) == 2
    for button in buttons:
        button.click()
    editors = event.details.findChildren(QPlainTextEdit)
    assert len(editors) == 2 and all(editor.isReadOnly() for editor in editors)
    texts = [e.toPlainText() for e in editors]
    assert all(t.index("Количество:") < t.index("Полные исходные поля") for t in texts)
    combined = "\n".join(texts)
    for expected in ("Количество", "-2.500", "CP", "unknown-cancel", "offline1C", "kanzlerKz",
                     "000123456789", "Все исходные записи", "Версия компонента", "Отпечаток версии", "Синтетический ID строки"):
        assert expected in combined
    assert any("\n          3" in t for t in texts)
    assert sum(len(v.origins) for v in event.event.variants) == 3
    assert all(v.event.details.snapshot_key in combined for v in event.event.variants)
    event.expand.click()
    event.expand.click()
    assert len(event.details.findChildren(QPlainTextEdit)) == 2


def test_conflicting_owners_times_and_kinds_preserved_and_marked(card):
    history = loaded(card)
    category(history, K.VIEW)
    history.load_more()
    wait(lambda: history._active is None)
    conflict = next(c for c in history.cards if c.event.source_event_id == "owner-conflict")
    assert conflict.event.owner_conflict and conflict.event.time_conflict and conflict.event.kind_conflict
    conflict.expand.click()
    for button in conflict.details.findChildren(QPushButton):
        button.click()
    text = "\n".join(e.toPlainText() for e in conflict.findChildren(QPlainTextEdit))
    assert "OTHER" in text and "VIEW" in text and "UNCLASSIFIED" in text
    assert "14:00:00" in text and "12:00:00.123456" in text


@pytest.mark.parametrize("status", list(S))
def test_all_backend_states_are_local_and_retry_is_explicit(card, status):
    card.tab._history_lookup = lambda customer, **kwargs: HistoryPage(status, customer)
    history = loaded(card)
    assert history.property("historyStatus") == status.value
    assert card.tab.result.products == result().products
    assert len(card.tab.product_grid.cards) == 1
    assert history.retry.isHidden() == (status in (S.READY, S.PARTIAL, S.EMPTY))
    assert history.more.isHidden()


def test_unexpected_lookup_error_never_hides_bpr_or_exposes_exception(card):
    def failed(*args, **kwargs):
        raise ValueError("private source path")
    card.tab._history_lookup = failed
    history = loaded(card)
    assert history.property("historyStatus") == S.SOURCE_INCONSISTENT.value
    assert "private" not in history.message.text()
    assert card.tab.result.products == result().products


def test_history_works_without_bpr_or_customer_profile_record(card):
    card.service.load = lambda customer, cancelled: result(customer, bpr=False)
    history = loaded(card)
    assert history.cards and not card.tab.product_grid.cards
    assert card.tab.result.state is ViewState.UNAVAILABLE


def test_missing_real_synthetic_index_is_bounded_and_profile_survives(card):
    card.data.source[1].index_path(raw_root=card.service.raw_root).unlink()
    history = loaded(card)
    assert history.property("historyStatus") == S.INDEX_NOT_READY.value
    assert not history.cards and card.tab.result.products == result().products


def test_customer_and_category_switch_discard_inflight_result_and_coalesce(card):
    entered, release = Event(), Event()
    original, calls = card.data.lookup, []
    def blocked(customer, **kwargs):
        calls.append((customer, kwargs["kinds"]))
        if len(calls) == 1:
            entered.set()
            assert release.wait(4)
        return original(customer, **kwargs)
    card.tab._history_lookup = blocked
    card.tab.apply_result(result())
    history = card.tab.history
    wait(entered.is_set)
    old = history._active
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(True))
    history.category.setCurrentIndex(1)
    card.tab.apply_result(result("EMPTY"))
    history.category.setCurrentIndex(2)
    assert old.cancelled.is_set() and not history.cards
    QTest.qWait(20)
    assert ticks
    release.set()
    wait(lambda: history._active is None)
    assert calls == [("C", (K.ORDER,)), ("EMPTY", (K.FAVORITE,))]
    assert history.customer_id == "EMPTY" and not history.cards
    assert history.property("historyStatus") == S.EMPTY.value
    history.receive(old, HistoryPage(S.READY, "C"))
    assert history.customer_id == "EMPTY" and not history.cards


def test_final_card_delivery_does_not_repeat_early_profile_history(card):
    loaded(card)
    assert len(card.data.calls) == 1
    card.tab.apply_result(result())
    QTest.qWait(30)
    assert len(card.data.calls) == 1


def test_stale_generation_rejects_append_and_requires_explicit_restart(app):
    history = CustomerHistorySection()
    requests = []
    history.dispatch.connect(requests.append)
    history.set_customer("C")
    history.receive(requests[-1], HistoryPage(S.READY, "C", next_cursor="cursor", generation="one", source_version="v"))
    history.load_more()
    history.receive(requests[-1], HistoryPage(S.READY, "C", generation="two", source_version="v"))
    assert history.property("historyStatus") == S.STALE_CURSOR.value
    assert history.cursor is None and not history.cards and not history.retry.isHidden()
    before = len(requests)
    QTest.qWait(20)
    assert len(requests) == before
    history.retry.click()
    assert requests[-1].cursor is None
    history.shutdown()
    history.deleteLater()


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [480, 1200])
def test_history_expanded_responsive_and_themed(card, app, dark, width):
    history = loaded(card)
    apply_app_theme(app, dark)
    history.cards[0].expand.click()
    for button in history.cards[0].details.findChildren(QPushButton):
        button.click()
    card.host.resize(width, 850)
    QTest.qWait(60)
    assert card.tab.scroll.horizontalScrollBar().maximum() == 0
    assert history.width() <= card.tab.page.width()
    assert card.tab.page.width() <= card.tab.scroll.viewport().width()
    for editor in history.findChildren(QPlainTextEdit):
        assert editor.width() <= history.width()


def test_window_close_waits_for_history_without_terminate(card):
    entered, release = Event(), Event()
    def blocked(customer, **kwargs):
        entered.set()
        assert release.wait(4)
        return HistoryPage(S.EMPTY, customer)
    card.tab._history_lookup = blocked
    card.tab.apply_result(result())
    wait(entered.is_set)
    thread = card.tab._thread
    card.host.close()
    assert thread.isRunning() and card.tab.history._active.cancelled.is_set()
    release.set()
    wait(lambda: card.tab._thread is None)
    assert not card.host.isVisible()
