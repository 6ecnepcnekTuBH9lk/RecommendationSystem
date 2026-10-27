"""Real small SQLite indexes only; no production data or model training."""

from copy import deepcopy
from dataclasses import replace
import os
from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QPushButton, QTableView, QWidget

from Application import customer_history_index as index
from Application.customer_card_data import CustomerCardResult, CustomerCardService, ProductContent, ViewState
from Application.customer_history_records import HistoryKind as K
from Application.customer_history_service import CustomerHistoryService, HistoryPage, HistoryStatus as S
from Application.customer_profile_service import CustomerProfileResult, ProfileStatus
from Application.model.customer_identity_bridge import CustomerModelIdentityResult, CustomerModelIdentityStatus as I
from Application.model.raw_bpr_inference import RawBprRecommendation
from Application.tabs.customer_history_widgets import CustomerHistorySection
from Application.customer_history_presentation import present_history
from Application.tabs.customer_history_table import HEADERS
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
    unknown["products"] = []
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
    history.buttons[kind].click()
    wait(lambda: history._active is None)


def rows(history):
    model = history.table.rows
    return [model.index(n, 0).data(Qt.ItemDataRole.UserRole) for n in range(model.rowCount())]


def test_find_background_catalog_and_original_bpr_are_independent(card):
    history = loaded(card)
    assert history.property("historyStatus") == S.PARTIAL.value
    assert len(history._seen) == 1 and len(rows(history)) == 2
    assert card.data.calls and all(t != get_ident() for t, _, _ in card.data.calls)
    assert card.data.calls[0][2] == dict(kinds=(K.ORDER,), limit=50, cursor=None, raw_root=card.service.raw_root)
    assert card.tab.result.products == result().products
    assert card.tab.raw_grid.cards[0].content.recommendation.raw_score == 1.23456789012345
    assert history.generation and history.source_version
    before = len(card.data.calls)
    card.tab.apply_result(result())
    QTest.qWait(20)
    assert len(card.data.calls) == before
    loaded(card)
    assert len(card.data.calls) == before + 1  # A new Find explicitly refreshes.


def test_pagination_fifty_events_all_occurrences_no_duplicates(card):
    history = loaded(card)
    category(history, K.VIEW)
    assert len(history._seen) == 50 and history.cursor
    first = list(rows(history))
    before = len(card.data.calls)
    history.load_more()
    history.load_more()
    wait(lambda: history._active is None)
    assert len(card.data.calls) == before + 1
    assert len(history._seen) == 62 and len(rows(history)) == 189
    assert rows(history)[:len(first)] == first
    assert history.cursor is None and history.more.isHidden()
    assert history.property("historyStatus") == S.PARTIAL.value
    history.load_more()
    assert len(card.data.calls) == before + 1


def test_categories_empty_no_product_and_no_current_favorite_claim(card):
    history = loaded(card)
    assert len(history.buttons) == 3 and history.category.group.exclusive()
    category(history, K.FAVORITE)
    assert len(rows(history)) == 3
    assert history.buttons[K.FAVORITE].isChecked()
    assert sum(button.isChecked() for button in history.buttons.values()) == 1
    assert tuple(HEADERS[K.FAVORITE])[-1] == "Дата события"
    assert K.UNCLASSIFIED not in history.buttons
    page = card.data.lookup("C", kinds=(K.UNCLASSIFIED,), raw_root=card.service.raw_root)
    actual = [row for _, items in present_history(page, K.UNCLASSIFIED).events for row in items]
    assert len(actual) == 1 and actual[0].values[1:5] == ("—", "—", "—", "—")
    loaded(card, "EMPTY")
    assert history.property("historyStatus") == S.EMPTY.value and not rows(history)
    assert "Нет событий" in history.table.empty_label.text() and history.more.isHidden()
    assert not history.table.isHidden() and history.message.isHidden()


def test_order_versions_are_flat_product_rows_no_technical_fields(card):
    history = loaded(card)
    model = history.table.rows
    assert isinstance(history.table, QTableView)
    assert model.rowCount() == 2  # Quantity-only snapshot differences add no product rows.
    assert all(model.rowCount(model.index(n, 0)) == 0 for n in range(2))
    assert all(not model.index(n, 0).parent().isValid() for n in range(2))
    assert all(row.values[5] == "01.01.2026 12:00" for row in rows(history))
    assert all(len(row.values) == 6 for row in rows(history))
    texts = " ".join(value for row in rows(history) for value in row.values)
    for forbidden in ("Заказ", "Вариант", "Несколько версий", "000001", "event_key", "snapshot_key",
                      "fingerprint", "provenance", "unknown-cancel", "CP", "MANUAL", "API"):
        assert forbidden not in texts
    assert not history.findChildren(QPlainTextEdit)
    assert not hasattr(history, "retry") and not hasattr(history, "coverage")
    assert not any(button.text() == "Загрузить заново" for button in history.findChildren(QPushButton))
    assert len(history.table.findChildren(QWidget)) < 40


@pytest.mark.parametrize("line_count", [1, 2])
def test_single_version_order_all_lines_available(card, line_count):
    loaded(card)
    backend = CustomerHistoryService(cache_dir=card.data.source[1].cache_dir)
    page = backend.get_customer_history("C", kinds=(K.ORDER,), raw_root=card.service.raw_root)
    logical = page.events[0]
    variant = logical.variants[0]
    event = replace(variant.event, details=replace(variant.event.details, lines=variant.event.details.lines[:line_count]))
    logical = replace(logical, variants=(replace(variant, event=event),))
    display = present_history(replace(page, events=(logical,)), K.ORDER)
    assert len(display.events[0][1]) == line_count
    assert all(len(row.values) == 6 for row in display.events[0][1])


def test_conflicting_owners_keep_flat_rows_and_no_tooltip(card):
    history = loaded(card)
    category(history, K.VIEW)
    history.load_more()
    wait(lambda: history._active is None)
    actual = rows(history)
    assert any("неоднозначна" in row.notice for row in actual)
    assert any(row.values[5] == "01.01.2026 14:00" for row in actual)
    model = history.table.rows
    assert all(model.index(n, 2).data(Qt.ItemDataRole.ToolTipRole) is None for n in range(model.rowCount()))


@pytest.mark.parametrize("status", list(S))
def test_all_statuses_local_short_no_retry_button(card, status):
    card.tab._history_lookup = lambda customer, **kwargs: HistoryPage(status, customer)
    history = loaded(card)
    assert history.property("historyStatus") == status.value
    assert card.tab.result.products == result().products
    assert history.more.isHidden()
    assert not hasattr(history, "retry")
    if status in (S.READY, S.EMPTY):
        assert history.message.isHidden()
        assert not history.table.empty_label.isHidden()
    else:
        assert history.message.text() and "покрыти" not in history.message.text()


def test_unexpected_error_and_missing_index_do_not_hide_bpr(card):
    card.data.source[1].index_path(raw_root=card.service.raw_root).unlink()
    history = loaded(card)
    assert history.property("historyStatus") == S.INDEX_NOT_READY.value
    assert card.tab.result.products == result().products
    def failed(*args, **kwargs):
        raise ValueError("private source path")
    card.tab._worker.history_lookup = failed
    loaded(card)
    assert history.property("historyStatus") == S.SOURCE_INCONSISTENT.value
    assert "private" not in history.message.text()


def test_history_works_without_bpr(card):
    from tests.tabs.test_customer_profile_card import found
    card.service.load = lambda customer, cancelled: replace(result(customer, bpr=False), profile=found())
    history = loaded(card)
    assert rows(history) and not card.tab.raw_grid.cards
    assert card.tab.result.state is ViewState.UNAVAILABLE
    assert card.tab.profile.content.state is ViewState.READY


def test_customer_category_changes_coalesce_and_discard_old_reply(card):
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
    history.buttons[K.VIEW].click()
    card.tab.apply_result(result("EMPTY"))
    history.buttons[K.FAVORITE].click()
    assert old.cancelled.is_set() and not rows(history)
    QTest.qWait(20)
    assert ticks
    release.set()
    wait(lambda: history._active is None)
    assert calls == [("C", (K.ORDER,)), ("EMPTY", (K.FAVORITE,))]
    assert history.customer_id == "EMPTY" and not rows(history)
    history.receive(old, HistoryPage(S.READY, "C"))
    assert not rows(history)


def test_stale_generation_rejects_append_and_requires_explicit_restart(app):
    history = CustomerHistorySection()
    requests = []
    history.dispatch.connect(requests.append)
    history.set_customer("C")
    history.receive(requests[-1], HistoryPage(S.READY, "C", next_cursor="cursor", generation="one", source_version="v"))
    history.load_more()
    history.receive(requests[-1], HistoryPage(S.READY, "C", generation="two", source_version="v"))
    assert history.property("historyStatus") == S.STALE_CURSOR.value
    assert history.cursor is None and not rows(history)
    assert "Повторите поиск" in history.message.text()
    assert len(requests) == 2
    history.set_customer(None)
    history.set_customer("C")
    assert requests[-1].cursor is None
    history.shutdown()
    history.deleteLater()


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [480, 1200])
def test_table_themes_tabs_and_progress_without_topk(card, app, dark, width):
    history = loaded(card)
    apply_app_theme(app, dark)
    card.host.resize(width, 850)
    QTest.qWait(70)
    assert card.tab.scroll.horizontalScrollBar().maximum() == 0
    assert history.width() <= card.tab.page.width()
    assert history.table.width() <= history.width()
    assert card.tab.kpis.message.isHidden()
    assert card.tab.recommendations.message.isHidden()
    assert card.tab.progress.minimum() == card.tab.progress.maximum() == 0
    assert card.tab.progress.text() == "Идет загрузка..."
    assert not hasattr(card.tab, "topk")
    for kind, button in history.buttons.items():
        assert button.isChecked() == (kind is K.ORDER)
        assert button.width() < history.width()
    assert tuple(history.table.rows.headerData(i, Qt.Orientation.Horizontal) for i in range(6)) == HEADERS[K.ORDER]


def test_catalog_names_safe_resolution_duplicate_codes_and_photos(card, tmp_path, monkeypatch):
    from Application.photo import photo_processing as photos
    from tests.tabs.test_customer_results_card import FakeReply, png
    path = tmp_path / "catalog.csv"
    path.write_text("КодНоменклатуры|НазваниеНаСайте|ТитульнаяФотография|ВидНоменклатуры|Коллекция\n000123|Из каталога|https://example.invalid/p|Рубашка|Осень\n000999|Другой товар|||\n", encoding="utf-8")
    catalog = CustomerCardService(catalog_path=path)
    threads = []
    def references(page):
        threads.append(get_ident())
        return catalog.history_product_references(page)
    card.service.history_product_references = references
    calls = []
    monkeypatch.setattr(photos, "_request_image", lambda owner, url, gen: calls.append((url, gen)))
    history = loaded(card)
    category(history, K.VIEW)
    leaves = rows(history)
    assert any(row.values[2:5] == ("Из каталога", "Рубашка", "Осень") and row.photo_url for row in leaves)
    assert threads and all(thread != get_ident() for thread in threads)
    url = "https://example.invalid/p"
    history.table.ensure_photo(url)
    history.table.ensure_photo(url)
    QTest.qWait(20)
    assert calls == [(url, card.tab._img_gen)]
    photos._on_image_loaded(card.tab, FakeReply(url, card.tab._img_gen, png()))
    assert not card.tab._img_cache[url].isNull()
    assert url not in history.table._photo_targets
    assert url not in card.tab._img_targets
    category(history, K.ORDER)
    assert any(row.values[2] == "Товар" for row in rows(history))  # Source name wins.
    path.write_text("КодНоменклатуры|НазваниеНаСайте|ТитульнаяФотография\n000123|Один|https://example.invalid/one\n000123|Другой|https://example.invalid/two\n", encoding="utf-8")
    page = CustomerHistoryService(cache_dir=card.data.source[1].cache_dir).get_customer_history("C", raw_root=card.service.raw_root)
    assert not catalog.history_product_references(page)


def test_close_waits_for_history_without_terminate(card):
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


@pytest.mark.parametrize("kind", [K.ORDER, K.VIEW, K.FAVORITE])
def test_all_category_headers(card, kind):
    history = loaded(card)
    category(history, kind)
    model = history.table.rows
    expected = ("Фото", "Код товара", "Название товара", "Вид номенклатуры", "Коллекция",
                "Дата заказа" if kind is K.ORDER else "Дата события")
    assert tuple(model.headerData(c, Qt.Orientation.Horizontal) for c in range(model.columnCount())) == expected


def test_photo_targets_bounded_and_category_change_drops_unused_queue(card, monkeypatch):
    from Application.photo import photo_processing as photos
    history = loaded(card)
    calls = []
    monkeypatch.setattr(photos, "_request_image", lambda owner, url, gen: calls.append(url))
    for n in range(500):
        history.table.ensure_photo(f"https://example.invalid/photo-{n}")
    QTest.qWait(20)
    assert len(history.table._photo_targets) == 24
    assert len(calls) == len(card.tab._img_inflight) == 3
    assert len(card.tab._img_queue) == 21
    category(history, K.VIEW)
    assert not history.table._photo_targets
    assert not card.tab._img_queue and not card.tab._img_targets


def test_unresolved_product_never_gets_a_catalog_photo(card, tmp_path):
    from Application.customer_history_records import ActionProductOccurrence, HistoryProductReference
    from Application.mindbox.records import ProductKey
    path = tmp_path / "catalog.csv"
    path.write_text("КодНоменклатуры|НазваниеНаСайте|ТитульнаяФотография\n000123|Товар|https://example.invalid/photo\n", encoding="utf-8")
    catalog = CustomerCardService(catalog_path=path)
    page = CustomerHistoryService(cache_dir=card.data.source[1].cache_dir).get_customer_history("C", kinds=(K.VIEW,), raw_root=card.service.raw_root)
    logical = page.events[0]
    variant = logical.variants[0]
    for keys in ((ProductKey("website", "000123456789"),),
                 (ProductKey("offline1C", "000123456789"), ProductKey("kanzlerKz", "000123456789"))):
        ref = HistoryProductReference(keys, None)
        event = replace(variant.event, details=replace(variant.event.details, products=(ActionProductOccurrence(0, ref),)))
        unknown = replace(page, events=(replace(logical, variants=(replace(variant, event=event),)),))
        assert catalog.history_product_references(unknown) == {}
        row = present_history(unknown, K.VIEW).events[0][1][0]
        assert not row.photo_url and row.values[2] == "—"


@pytest.mark.parametrize("state", [ViewState.ERROR, ViewState.UNAVAILABLE, ViewState.STALE, ViewState.EMPTY])
def test_recommendation_problems_remain_visible(card, state):
    loaded(card)
    card.tab.apply_result(replace(result(), state=state, message="Проблема рекомендаций", products=()))
    assert not card.tab.recommendations.message.isHidden()
    assert card.tab.recommendations.message.text() == ("Исходных рекомендаций нет." if state is ViewState.EMPTY else "Проблема рекомендаций")


def test_catalog_failure_keeps_history_readable(card):
    def unavailable(page):
        raise OSError("private catalog path")
    card.service.history_product_references = unavailable
    history = loaded(card)
    category(history, K.VIEW)
    assert rows(history) and history.property("historyStatus") in (S.READY.value, S.PARTIAL.value)
    assert all(row.values[2] == "—" for row in rows(history))
    assert "private" not in history.message.text()


def test_history_photos_share_queue_and_preserve_recommendation_targets(card, monkeypatch):
    from Application.photo import photo_processing as photos
    from tests.tabs.test_customer_results_card import FakeReply, png
    history = loaded(card)
    calls = []
    monkeypatch.setattr(photos, "_request_image", lambda owner, url, gen: calls.append(url))
    url = "https://example.invalid/shared-photo"
    label = card.tab.raw_grid.cards[0].photo
    photos._set_photo_label(card.tab, label, url, card.tab._img_gen)
    history.table.ensure_photo(url)
    QTest.qWait(20)
    assert calls == [url] and len(card.tab._img_targets[url]) == 2
    category(history, K.VIEW)
    assert card.tab._img_targets[url] == [(label, None)]
    photos._on_image_loaded(card.tab, FakeReply(url, card.tab._img_gen, png()))
    assert label.pixmap() is not None and not label.pixmap().isNull()


def test_many_product_rows_do_not_create_per_row_widgets(card):
    history = loaded(card)
    model = history.table.rows
    root = model.index(0, 0).data(Qt.ItemDataRole.UserRole)
    before = len(history.table.findChildren(QWidget))
    history.table.reset_history(K.ORDER)
    history.table.append((root,) * 2000)
    assert model.rowCount() == 2000
    assert model.rowCount(model.index(0, 0)) == 0
    QTest.qWait(20)
    assert len(history.table.findChildren(QWidget)) == before


@pytest.mark.parametrize("empty_reference", [False, True])
def test_views_without_product_are_not_rendered(card, empty_reference):
    from Application.customer_history_records import ActionProductOccurrence, HistoryProductReference
    history = loaded(card)
    page = card.data.lookup("C", kinds=(K.VIEW,), raw_root=card.service.raw_root)
    logical = page.events[0]
    variant = logical.variants[0]
    products = (ActionProductOccurrence(0, HistoryProductReference((), None)),) if empty_reference else ()
    event = replace(variant.event, details=replace(variant.event.details, products=products))
    page = replace(page, events=(replace(logical, variants=(replace(variant, event=event),)),), next_cursor=None)
    display = present_history(page, K.VIEW)
    assert display.events == ((logical.event_key, ()),)
    card.tab._worker.history_lookup = lambda customer, **kwargs: page
    category(history, K.VIEW)
    assert not rows(history)
    assert not history.table.empty_label.isHidden()
    assert history.property("historyStatus") == S.READY.value and history.cursor is None


@pytest.mark.parametrize("status", [S.READY, S.PARTIAL, S.EMPTY])
def test_empty_table_is_centered_and_cursor_can_advance(app, status):
    history = CustomerHistorySection()
    history.resize(1100, 480)
    history.show()
    requests = []
    history.dispatch.connect(requests.append)
    history.set_customer("C")
    history.receive(requests[-1], HistoryPage(status, "C", next_cursor="next", generation="g", source_version="v"))
    QTest.qWait(20)
    assert not history.table.isHidden()
    assert not history.table.empty_label.isHidden()
    assert history.table.empty_label.alignment() == Qt.AlignmentFlag.AlignCenter
    assert history.table.empty_label.geometry().center() == history.table.viewport().rect().center()
    assert history.more.isHidden()
    history.load_more()
    assert len(requests) == 2 and requests[-1].cursor == "next"  # Empty page is skipped automatically.
    history.receive(requests[-1], HistoryPage(S.EMPTY, "C", generation="g", source_version="v"))
    assert history.more.isHidden() and history.cursor is None
    assert history.property("historyStatus") == (S.PARTIAL if status is S.PARTIAL else S.EMPTY).value
    history.shutdown()
    history.close()
    history.deleteLater()


def test_order_versions_keep_different_products_dates_and_repeated_lines(card):
    from datetime import timedelta
    loaded(card)
    page = card.data.lookup("C", kinds=(K.ORDER,), raw_root=card.service.raw_root)
    logical = page.events[0]
    first = logical.variants[0]
    line = first.event.details.lines[0]
    first = replace(first, event=replace(first.event, details=replace(first.event.details, lines=(line, line))))
    second = replace(first, event=replace(first.event, details=replace(first.event.details, lines=(line,))))
    logical = replace(logical, variants=(first, second))
    display = present_history(replace(page, events=(logical,)), K.ORDER)
    assert len(display.events[0][1]) == 2  # Real repeated lines survive merging snapshots.
    stamp = second.event.event_datetime_utc + timedelta(days=1)
    detail = second.event.details
    detail = replace(detail, order=replace(detail.order, order_datetime_utc=stamp))
    second = replace(second, event=replace(second.event, event_datetime_utc=stamp, details=detail))
    third = replace(first, event=replace(first.event, details=replace(first.event.details,
                    lines=(replace(line, product_name="Другое название"),))))
    logical = replace(logical, variants=(first, second, third))
    display = present_history(replace(page, events=(logical,)), K.ORDER)
    assert len(display.events[0][1]) == 4
    assert display.events[0][1][-1].values[2] == "Другое название"
    assert {row.values[-1] for row in display.events[0][1]} == {"01.01.2026 12:00", "02.01.2026 12:00"}


def test_catalog_optional_fields_are_independent_and_bpr_metadata_unchanged(card, tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text("КодНоменклатуры|НазваниеНаСайте|ТитульнаяФотография|ВидНоменклатуры|Коллекция\n"
                    "000123|Рубашка||Рубашки|Весна\n000123|Рубашка||Рубашки|Лето\n", encoding="utf-8")
    service = CustomerCardService(catalog_path=path)
    page = card.data.lookup("C", kinds=(K.VIEW,), raw_root=card.service.raw_root)
    catalog = service.history_product_references(page)
    assert service._references() == {"000123": ("Рубашка", "")}
    assert all(value == ("Рубашка", "", "Рубашки", "") for value in catalog.values())
    display = present_history(page, K.VIEW, catalog)
    assert all(row.values[2:5] == ("Рубашка", "Рубашки", "—") for _, items in display.events for row in items)
    path.write_text("КодНоменклатуры|НазваниеНаСайте\n000123|Новое имя\n", encoding="utf-8")
    display = present_history(page, K.VIEW, service.history_product_references(page))
    assert all(row.values[2:5] == ("Новое имя", "—", "—") for _, items in display.events for row in items)


def test_large_photo_geometry_and_centered_empty_state_in_both_themes(card, app):
    history = loaded(card)
    from PyQt6.QtWidgets import QStyleOptionViewItem
    for dark in (False, True):
        apply_app_theme(app, dark)
        QTest.qWait(20)
        assert history.table.columnWidth(0) >= 156
        assert history.table.rowHeight(0) >= 226
        size = history.table.itemDelegate().sizeHint(QStyleOptionViewItem(), history.table.rows.index(0, 0))
        assert size.height() >= 226
        loaded(card, "EMPTY")
        QTest.qWait(20)
        assert history.table.empty_label.geometry().center() == history.table.viewport().rect().center()
        assert history.table.empty_label.palette().windowText().color() != history.table.palette().base().color()
        loaded(card)


@pytest.mark.parametrize("dark", [False, True])
def test_ui04_alignment_font_headers_and_full_copy(card, app, dark):
    from PyQt6.QtWidgets import QStyleOptionViewItem
    from Application.customer_history_presentation import HistoryRow
    history = loaded(card)
    apply_app_theme(app, dark)
    table, model = history.table, history.table.rows
    value = "Длинное название товара " * 30
    table.append((HistoryRow(("", "000000123456789", value, value, "Осень", "01.01.2026 12:00")),))
    for column in range(6):
        cell = model.index(0, column)
        assert cell.data(Qt.ItemDataRole.TextAlignmentRole) == int(Qt.AlignmentFlag.AlignCenter)
        assert model.headerData(column, Qt.Orientation.Horizontal, Qt.ItemDataRole.TextAlignmentRole) == int(Qt.AlignmentFlag.AlignCenter)
        assert model.headerData(column, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole) is None
        option = QStyleOptionViewItem()
        option.font = table.font()
        table.itemDelegate().initStyleOption(option, cell)
        assert option.font.pointSizeF() == table.font().pointSizeF() + 2
    assert table.horizontalHeader().font().pointSizeF() == table.font().pointSizeF()
    table.selectRow(model.rowCount() - 1)
    QTest.keyClick(table, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert value in QApplication.clipboard().text()
    assert table.textElideMode() == Qt.TextElideMode.ElideRight


@pytest.mark.parametrize("dark", [False, True])
def test_ui04_drag_every_column_boundary_and_preserve_widths(card, app, dark):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QHeaderView
    history = loaded(card)
    table = history.table
    apply_app_theme(app, dark)
    card.tab.scroll.ensureWidgetVisible(table)
    QTest.qWait(30)
    header = table.horizontalHeader()
    for column in range(6):
        assert header.sectionResizeMode(column) == QHeaderView.ResizeMode.Interactive
    assert not header.stretchLastSection()
    assert header.minimumSectionSize() == 64
    # Exercise both directions on all five internal handles, including either
    # side of the nomenclature column. Scroll each handle into view first.
    for column in range(5):
        offset = max(0, header.sectionPosition(column) + header.sectionSize(column) - header.viewport().width() // 2)
        table.horizontalScrollBar().setValue(offset)
        QTest.qWait(10)
        width = table.columnWidth(column)
        x = header.sectionViewportPosition(column) + header.sectionSize(column) - 1
        point = QPoint(x, header.height() // 2)
        QTest.mouseMove(header.viewport(), point)
        QTest.mousePress(header.viewport(), Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseMove(header.viewport(), point + QPoint(25, 0), delay=10)
        QTest.mouseRelease(header.viewport(), Qt.MouseButton.LeftButton, pos=point + QPoint(25, 0))
        assert table.columnWidth(column) == width + 25
        x = header.sectionViewportPosition(column) + header.sectionSize(column) - 1
        point = QPoint(x, header.height() // 2)
        QTest.mousePress(header.viewport(), Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseMove(header.viewport(), point - QPoint(12, 0), delay=10)
        QTest.mouseRelease(header.viewport(), Qt.MouseButton.LeftButton, pos=point - QPoint(12, 0))
        assert table.columnWidth(column) == width + 13
    table.setColumnWidth(5, 205)
    widths = [table.columnWidth(n) for n in range(6)]
    category(history, K.VIEW)
    assert [table.columnWidth(n) for n in range(6)] == widths
    history.load_more()
    wait(lambda: history._active is None)
    assert [table.columnWidth(n) for n in range(6)] == widths
    card.host.resize(480, 850)
    QTest.qWait(20)
    assert table.horizontalScrollBar().maximum() > 0
    assert card.tab.scroll.horizontalScrollBar().maximum() == 0


@pytest.mark.parametrize("count", [0, 1, 4, 5, 20, 50])
def test_ui05_table_height_fits_all_rows_without_inner_vertical_scroll(card, count):
    from Application.customer_history_presentation import HistoryRow
    from Application.tabs.customer_history_table import ROW_HEIGHT
    history = loaded(card)
    table = history.table
    table.reset_history(K.ORDER)
    table.append((HistoryRow(("", "code", "Товар", "Вид", "Коллекция", "Дата")),) * count)
    table.show()
    QTest.qWait(30)
    if count:
        assert table.viewport().height() >= count * ROW_HEIGHT
        assert table.viewport().height() < (count + 1) * ROW_HEIGHT
    else:
        assert table.height() < ROW_HEIGHT
    assert table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert table.verticalScrollBar().maximum() == 0
    assert card.tab.scroll.verticalScrollBar().maximum() > 0


def test_ui04_tooltips_blocked_only_inside_tab_including_dynamic_widgets(card, app):
    from PyQt6.QtCore import QEvent, QPoint
    from PyQt6.QtGui import QHelpEvent
    from PyQt6.QtWidgets import QLabel, QToolTip
    history = loaded(card)
    class Probe(QLabel):
        def __init__(self, parent=None):
            super().__init__("probe", parent)
            self.tooltips = 0
        def event(self, event):
            if event.type() is QEvent.Type.ToolTip:
                self.tooltips += 1
                return True
            return super().event(event)
    dynamic = Probe(card.tab.page)
    outside = Probe()
    targets = [card.tab, card.tab.customer_input, card.tab.search_button, card.tab.export_button,
               card.tab.profile, card.tab.raw_grid.cards[0].photo, history.more,
               history.table.viewport(), history.table.horizontalHeader().viewport(), dynamic]
    QToolTip.hideText()
    for target in targets:
        target.setToolTip("Подсказка для теста")
        point = QPoint(2, 2)
        event = QHelpEvent(QEvent.Type.ToolTip, point, target.mapToGlobal(point))
        QApplication.sendEvent(target, event)
    assert not QToolTip.isVisible() and dynamic.tooltips == 0
    QApplication.sendEvent(outside, QHelpEvent(QEvent.Type.ToolTip, QPoint(2, 2), QPoint(2, 2)))
    assert outside.tooltips == 1
    # Non-tooltip interaction still reaches the dynamically created widget.
    dynamic.setText("Изменено")
    assert dynamic.text() == "Изменено"
    outside.deleteLater()


@pytest.mark.parametrize("width,height,expected_width,expected_height", [(200, 300, 140, 210), (400, 200, 140, 70), (100, 300, 70, 210)])
def test_ui04_photos_keep_aspect_ratio_without_cropping(card, width, height, expected_width, expected_height):
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
    from PyQt6.QtWidgets import QStyleOptionViewItem
    from Application.customer_history_presentation import HistoryRow
    from Application.tabs.customer_history_table import ROW_HEIGHT
    history = loaded(card)
    table = history.table
    table.reset_history(K.ORDER)
    table.append((HistoryRow(("", "code", "Товар", "Вид", "Коллекция", "Дата"), "cached"),))
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#00ff00"))
    card.tab._img_cache["cached"] = pixmap
    canvas = QImage(156, ROW_HEIGHT, QImage.Format.Format_ARGB32)
    canvas.fill(QColor("#ffffff"))
    painter = QPainter(canvas)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 156, ROW_HEIGHT)
    table.itemDelegate().paint(painter, option, table.rows.index(0, 0))
    painter.end()
    points = [(x,y) for y in range(canvas.height()) for x in range(canvas.width()) if canvas.pixelColor(x,y) == QColor("#00ff00")]
    left, right = min(x for x,y in points), max(x for x,y in points)
    top, bottom = min(y for x,y in points), max(y for x,y in points)
    assert right-left+1 == expected_width and bottom-top+1 == expected_height
    assert abs((left+right)/2 - 156/2) <= 1
    assert abs((top+bottom)/2 - ROW_HEIGHT/2) <= 1


@pytest.mark.parametrize("dark", [False, True])
def test_ui05_columns_fill_width_and_respect_manual_sizes(card, app, dark):
    history = loaded(card)
    apply_app_theme(app, dark)
    table = history.table
    for width in (1600, 1200, 1050, 1600):
        card.host.resize(width, 850)
        QTest.qWait(20)
        assert sum(table.columnWidth(n) for n in range(6)) == table.viewport().width()
        assert table.horizontalScrollBar().maximum() == 0
    table.setColumnWidth(2, 380)
    table.setColumnWidth(3, 245)
    for width in (1250, 1800):
        card.host.resize(width, 850)
        QTest.qWait(20)
        assert table.columnWidth(2) == 380 and table.columnWidth(3) == 245
        assert sum(table.columnWidth(n) for n in range(6)) == table.viewport().width()
    category(history, K.VIEW)
    assert table.columnWidth(2) == 380 and table.columnWidth(3) == 245
    height = table.height()
    history.load_more()
    wait(lambda: history._active is None)
    assert table.height() > height
    assert table.viewport().height() >= table.verticalHeader().length()
    assert table.verticalScrollBar().maximum() == 0


def test_ui05_all_manually_set_widths_take_priority_over_automatic_filling(card):
    table = loaded(card).table
    for column, width in enumerate((160, 140, 290, 200, 150, 180)):
        table.setColumnWidth(column, width)
    widths = [table.columnWidth(n) for n in range(6)]
    card.host.resize(1800, 850)
    QTest.qWait(20)
    assert [table.columnWidth(n) for n in range(6)] == widths


@pytest.mark.parametrize("dark", [False, True])
def test_ui05_existing_category_icons_and_centering(card, app, dark):
    from PyQt6.QtCore import QSize
    from PyQt6.QtGui import QIcon
    from Application.paths import ICONS_DIR
    history = loaded(card)
    apply_app_theme(app, dark)
    expected = {K.ORDER: "order", K.VIEW: "actions", K.FAVORITE: "favorites"}
    assert set(history.buttons) == set(expected)
    for kind, name in expected.items():
        button = history.buttons[kind]
        assert button.iconSize() == QSize(20, 20)
        assert not button.icon().isNull()
        assert button.icon().pixmap(20,20).toImage() == QIcon(str(ICONS_DIR / (name + ".png"))).pixmap(20,20).toImage()
        assert button.text().strip() == {K.ORDER: "Заказы", K.VIEW: "Просмотры", K.FAVORITE: "Избранное"}[kind]
    assert history.category.grid.alignment() & Qt.AlignmentFlag.AlignHCenter


def test_ui05_geometry_uses_actual_row_sizes_and_vertical_wheel_is_ignored(card):
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QHeaderView
    history = loaded(card)
    table = history.table
    before = table.height()
    table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.setRowHeight(0, table.rowHeight(0) + 35)
    QTest.qWait(10)
    assert table.height() == before + 35
    event = QWheelEvent(QPointF(2, 2), QPointF(table.mapToGlobal(QPoint(2, 2))), QPoint(), QPoint(0, -120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(table.viewport(), event)
    assert not event.isAccepted()
    assert table.verticalScrollBar().maximum() == 0
