"""Synthetic contacts only; each Qt suite runs in its own pytest process."""

from contextlib import nullcontext
from datetime import date, datetime, timezone
import os
from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMainWindow

from Application import customer_card_data as data
from Application.customer_card_data import CustomerCardService, ViewState
from Application.customer_contact_search import ContactSearchResult, ContactSearchStatus as S, ContactType
from Application.customer_profile_service import BirthDateStatus, CardStatus, CustomerProfile, CustomerProfileResult, ProfileStatus
from Application.customer_profiles import CustomerContact
from Application.mindbox.records import CustomerRecord
from Application.model.customer_identity_bridge import CustomerModelIdentityResult, CustomerModelIdentityStatus as I
from Application.model.raw_bpr_inference import RawBprRecommendation, RawBprResult, RawBprStatus
from Application.tabs import customer_card_worker as worker
from Application.customer_history_service import HistoryPage, HistoryStatus
from Application.tabs.customer_results_tab import CustomerResultsTab
from Application.theme.apply_theme import apply_app_theme, prepare_app_theme


VALUES = {None: "C", ContactType.PHONE: "+7 (900) 111-22-33",
          ContactType.EMAIL: " Person+tag@Example.test ", ContactType.DISCOUNT_CARD: "000123"}


def matched(customer="C", version="synthetic-v1"):
    return ContactSearchResult(S.MATCHED, customer, 1, version)


def found(customer="C"):
    record = CustomerRecord(customer, customer, datetime(2026, 1, 1, tzinfo=timezone.utc), date(1990, 1, 1),
                            "male", "Синтетический", "Покупатель", None, "synthetic@example.test", "+79001112233")
    p = CustomerProfile(record, CustomerContact(record.email, record.mobile_phone, "000123"),
                        date(2026, 10, 13), 36, "36–45 лет", BirthDateStatus.VALID, CardStatus.SINGLE)
    return CustomerProfileResult(ProfileStatus.FOUND, p, customer, "synthetic-profile", "r")


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    prepare_app_theme(app)
    return app


@pytest.fixture
def card(app, monkeypatch):
    backend = Mock(return_value=matched())
    gate = Mock(side_effect=lambda match, **kwargs: match)
    monkeypatch.setattr(worker, "search_customers_by_contact", backend)
    monkeypatch.setattr(data, "validate_contact_match", gate)
    profile, bridge, inference = Mock(), Mock(), Mock()
    profile.get_profile.side_effect = lambda customer, **kwargs: found(customer)
    profile.is_current.return_value = True
    bridge.resolve.side_effect = lambda customer, **kwargs: CustomerModelIdentityResult(customer, "A", I.MATCHED_VIA_MERGE, 1, "g", "r", "v")
    bridge.is_current.return_value = True
    raw = tuple(RawBprRecommendation(i, f"SYNTHETIC-{i}", 1.123456789 / i) for i in range(1, 11))
    inference.get_recommendations.return_value = RawBprResult("g", "A", RawBprStatus.OK, raw)
    service = CustomerCardService(profile_service=profile, bridge=bridge, inference=inference)
    service._references = Mock(return_value={})
    host = QMainWindow()
    tab = CustomerResultsTab(host, service=service, history_lookup=lambda customer, **kw: HistoryPage(HistoryStatus.EMPTY, customer))
    host.setCentralWidget(tab)
    host.resize(1200, 850)
    host.show()
    QTest.qWait(20)
    yield SimpleNamespace(host=host, tab=tab, service=service, backend=backend, gate=gate, raw=raw)
    tab.shutdown()
    if tab._thread is not None:
        assert tab._thread.wait(5000)
    host.close()
    host.deleteLater()
    QTest.qWait(20)


def wait(predicate):
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


def choose(tab, kind):
    tab.search_type.setCurrentIndex(tab.search_type.findData(kind))


def search(card, kind=ContactType.EMAIL, value=None, enter=False):
    choose(card.tab, kind)
    card.tab.customer_input.setText(VALUES[kind] if value is None else value)
    if enter:
        QTest.keyClick(card.tab.customer_input, Qt.Key.Key_Return)
    else:
        card.tab.search_button.click()
    wait(lambda: card.tab.result is not None)


@pytest.mark.parametrize("kind", list(VALUES))
@pytest.mark.parametrize("enter", [False, True])
def test_four_modes_button_enter_background_and_original_topk(card, kind, enter):
    gui_id, search_threads, load_threads = get_ident(), [], []
    def lookup(*args, **kwargs):
        search_threads.append(get_ident())
        return matched()
    def profile(customer, **kwargs):
        load_threads.append(get_ident())
        return found(customer)
    card.backend.side_effect = lookup
    card.service.profile_service.get_profile.side_effect = profile
    search(card, kind, enter=enter)
    assert load_threads and all(t != gui_id for t in load_threads)
    if kind is None:
        card.backend.assert_not_called()
    else:
        card.backend.assert_called_once_with(kind, VALUES[kind], raw_root=card.service.raw_root)
        assert search_threads == load_threads
    assert card.service.profile_service.get_profile.call_args.args == ("C",)
    assert card.service.bridge.resolve.call_args.args == ("C",)
    assert tuple(c.content.recommendation for c in card.tab.product_grid.cards) == card.raw
    for i, count in enumerate((1, 3, 5, 10)):
        card.tab.topk.setCurrentIndex(i)
        assert card.tab.product_grid.visible_count == count
    assert card.service.inference.get_recommendations.call_count == 1
    assert card.tab.profile.content.state is ViewState.READY
    assert not card.tab.export_button.isEnabled()


@pytest.mark.parametrize("status,state,message", [
    (S.AMBIGUOUS, ViewState.UNAVAILABLE, "несколько"),
    (S.NOT_FOUND, ViewState.EMPTY, "не найдены"),
    (S.INVALID_INPUT, ViewState.ERROR, "формат"),
    (S.INDEX_NOT_READY, ViewState.UNAVAILABLE, "построить или обновить"),
    (S.SOURCE_UNAVAILABLE, ViewState.UNAVAILABLE, "недоступен"),
    (S.SOURCE_INCONSISTENT, ViewState.ERROR, "несогласованы"),
])
def test_lookup_failures_never_load_or_confirm_a_customer(card, status, state, message):
    card.backend.return_value = ContactSearchResult(status, candidate_count=3 if status is S.AMBIGUOUS else 0)
    search(card)
    assert card.tab.result.state is state and message in card.tab.search_status.text()
    assert card.tab.result.identity is None and card.tab.result.profile is None
    assert not card.tab.product_grid.cards and not card.tab.profile_fields.cards
    assert "ещё не определён" in card.tab.canonical_id.text()
    card.service.profile_service.get_profile.assert_not_called()
    card.service.bridge.resolve.assert_not_called()
    card.service.inference.get_recommendations.assert_not_called()
    assert not card.tab.progress.isVisible()


@pytest.mark.parametrize("kind,value", [(ContactType.PHONE, "89001112233"), (ContactType.PHONE, "9001112233"),
                                       (ContactType.EMAIL, "bad-email"), (ContactType.DISCOUNT_CARD, ""),
                                       (ContactType.DISCOUNT_CARD, "000123")])
def test_ui_does_not_rewrite_or_guess_contact_inputs(card, kind, value):
    card.backend.return_value = ContactSearchResult(S.INVALID_INPUT)
    search(card, kind, value)
    assert card.backend.call_args.args == (kind, value)
    assert card.tab.customer_input.text() == value
    assert card.tab.result.state is ViewState.ERROR


def test_type_change_invalidates_running_result_and_keeps_gui_responsive(card):
    entered, release = Event(), Event()
    def lookup(kind, value, **kwargs):
        entered.set()
        assert release.wait(4)
        return matched("OLD")
    card.backend.side_effect = lookup
    choose(card.tab, ContactType.EMAIL)
    card.tab.customer_input.setText("old@example.test")
    card.tab.search()
    wait(entered.is_set)
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(True))
    choose(card.tab, ContactType.DISCOUNT_CARD)
    assert card.tab._active.cancelled.is_set()
    card.tab.customer_input.setText("000123")
    card.tab.search()
    choose(card.tab, None)
    card.tab.customer_input.setText("LATEST")
    card.tab.search()
    QTest.qWait(20)
    assert ticks and card.tab.result is None and card.tab._pending.customer_id == "LATEST"
    release.set()
    wait(lambda: card.tab.result is not None)
    card.backend.assert_called_once()
    assert card.service.profile_service.get_profile.call_args.args == ("LATEST",)
    assert "LATEST" in card.tab.canonical_id.text() and "OLD" not in card.tab.canonical_id.text()


def test_latest_contact_request_is_snapshot_of_type_and_value(card):
    entered, release = Event(), Event()
    calls = []
    def lookup(kind, value, **kwargs):
        calls.append((kind, value))
        if len(calls) == 1:
            entered.set()
            assert release.wait(4)
        return matched("LATEST")
    card.backend.side_effect = lookup
    choose(card.tab, ContactType.PHONE)
    card.tab.customer_input.setText("+79001112233")
    card.tab.search()
    wait(entered.is_set)
    for value in ("+79001112234", "+79001112235", "+79001112236"):
        card.tab.customer_input.setText(value)
        card.tab.search()
    release.set()
    wait(lambda: card.tab.result is not None)
    assert calls == [(ContactType.PHONE, "+79001112233"), (ContactType.PHONE, "+79001112236")]
    assert card.service.profile_service.get_profile.call_count == 1


def test_switch_type_clears_previous_card_and_late_photos(card):
    search(card)
    generation = card.tab._img_gen
    choose(card.tab, ContactType.PHONE)
    assert card.tab.result is None and not card.tab.product_grid.cards
    assert card.tab.profile.content.state is ViewState.INITIAL
    assert card.tab._img_gen > generation
    assert "+" in card.tab.customer_input.placeholderText()
    assert "десятизначные" in card.tab.search_hint.text()
    choose(card.tab, ContactType.DISCOUNT_CARD)
    assert "ведущие нули" in card.tab.search_hint.text()


@pytest.mark.parametrize("action", ["cancel", "close"])
def test_cancel_or_close_during_lookup_does_not_load_profile(card, action):
    entered, release = Event(), Event()
    def lookup(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        return matched()
    card.backend.side_effect = lookup
    choose(card.tab, ContactType.EMAIL)
    card.tab.customer_input.setText("synthetic@example.test")
    card.tab.search()
    wait(entered.is_set)
    card.tab.cancel_search() if action == "cancel" else card.host.close()
    release.set()
    wait(lambda: card.tab._active is None if action == "cancel" else card.tab._thread is None)
    card.service.profile_service.get_profile.assert_not_called()
    assert card.tab.result is None and not card.tab.product_grid.cards


def test_contact_profile_can_appear_before_bpr_and_old_profiles_cannot_mix(card):
    entered, release = Event(), Event()
    def inference(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        return RawBprResult("g", "A", RawBprStatus.OK, card.raw)
    card.service.inference.get_recommendations.side_effect = inference
    choose(card.tab, ContactType.EMAIL)
    card.tab.customer_input.setText("synthetic@example.test")
    card.tab.search()
    wait(entered.is_set)
    wait(lambda: card.tab.profile.content.state is ViewState.READY)
    assert card.tab.result is None and card.tab.progress.isVisible()
    choose(card.tab, ContactType.PHONE)
    release.set()
    wait(lambda: card.tab._active is None)
    assert card.tab.profile.content.state is ViewState.INITIAL and card.tab.result is None


def test_contact_freshness_change_during_card_load_clears_profile_and_recommendations(card):
    entered, release = Event(), Event()
    def inference(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        return RawBprResult("g", "A", RawBprStatus.OK, card.raw)
    card.service.inference.get_recommendations.side_effect = inference
    choose(card.tab, ContactType.EMAIL)
    card.tab.customer_input.setText("synthetic@example.test")
    card.tab.search()
    wait(entered.is_set)
    card.gate.side_effect = lambda *args, **kwargs: ContactSearchResult(S.INDEX_NOT_READY)
    release.set()
    wait(lambda: card.tab.result is not None)
    assert card.tab.result.contact_search.status is S.INDEX_NOT_READY
    assert card.tab.result.profile is None and card.tab.result.identity is None
    assert not card.tab.profile_fields.cards and not card.tab.product_grid.cards


def test_contact_freshness_rechecked_at_gui_handoff(card):
    original = card.service.validate_result
    gui_thread = get_ident()
    def validate(result):
        if get_ident() == gui_thread and result.state is ViewState.READY:
            card.gate.side_effect = lambda *args, **kwargs: ContactSearchResult(S.INDEX_NOT_READY)
        return original(result)
    card.service.validate_result = validate
    search(card)
    assert card.tab.result.contact_search.status is S.INDEX_NOT_READY
    assert not card.tab.profile_fields.cards and not card.tab.product_grid.cards


def test_contact_selection_does_not_follow_an_unconfirmed_identity_change(card):
    card.service.profile_service.get_profile.side_effect = lambda *a, **kw: found("DIFFERENT")
    search(card)
    assert card.tab.result.contact_search.status is S.INDEX_NOT_READY
    assert card.tab.result.identity is None and card.tab.result.profile is None


@pytest.mark.parametrize("failure", ["profile", "bpr"])
def test_independent_card_states_are_preserved_for_contacts(card, failure):
    if failure == "profile":
        card.service.profile_service.get_profile.side_effect = None
        card.service.profile_service.get_profile.return_value = CustomerProfileResult(ProfileStatus.SOURCE_UNAVAILABLE)
    else:
        card.service.inference.get_recommendations.side_effect = ValueError("synthetic private detail")
    search(card)
    if failure == "profile":
        assert tuple(c.content.recommendation for c in card.tab.product_grid.cards) == card.raw
        assert card.tab.profile.content.state is ViewState.UNAVAILABLE
    else:
        assert card.tab.profile.content.state is ViewState.READY and not card.tab.product_grid.cards


def test_lookup_exception_is_generic_and_no_pii_is_logged(card, caplog, capsys):
    card.backend.side_effect = ValueError("private-synthetic@example.test")
    search(card)
    assert card.tab.result.state is ViewState.ERROR
    assert "private-synthetic" not in card.tab.search_status.text() + caplog.text + str(capsys.readouterr())
    card.service.profile_service.get_profile.assert_not_called()
    assert "Person+tag" not in repr(card.tab.result)
    assert "Person+tag" not in repr(worker.CustomerCardRequest(1, VALUES[ContactType.EMAIL], contact_type=ContactType.EMAIL))


def test_contact_input_and_mode_are_not_restored_in_another_card(card):
    search(card, ContactType.DISCOUNT_CARD)
    other_host = QMainWindow()
    other_tab = CustomerResultsTab(other_host, service=card.service)
    try:
        assert other_tab.search_type.currentData() is None
        assert other_tab.customer_input.text() == "" and other_tab.result is None
    finally:
        other_tab.shutdown()
        other_host.close()
        other_host.deleteLater()
        QTest.qWait(20)


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [480, 900, 1600])
def test_selector_hints_both_themes_and_adaptive_layout(card, app, dark, width):
    apply_app_theme(app, dark)
    card.host.resize(width, 850)
    for kind in VALUES:
        choose(card.tab, kind)
        QTest.qWait(30)
        assert card.tab.scroll.horizontalScrollBar().maximum() == 0
        assert card.tab.page.width() <= card.tab.scroll.viewport().width()
        assert card.tab.search_type.width() <= card.tab.search_card.width()
        assert card.tab.customer_input.placeholderText() and card.tab.search_hint.text()


@pytest.mark.parametrize("damage,expected", [(None, S.MATCHED), ("version", S.INDEX_NOT_READY),
                                            ("missing", S.INDEX_NOT_READY), ("metadata", S.INDEX_NOT_READY),
                                            ("unavailable", S.SOURCE_UNAVAILABLE), ("corrupt", S.SOURCE_INCONSISTENT)])
def test_metadata_handoff_reuses_backend_gates_without_contact_lookup(tmp_path, monkeypatch, damage, expected):
    backend = data.contact_backend
    path = tmp_path / "synthetic-index.sqlite"
    if damage != "missing":
        path.write_bytes(b"synthetic")
    monkeypatch.setattr(backend, "_reader_lock", lambda *args: nullcontext())
    monkeypatch.setattr(backend, "_sources", lambda *args: SimpleNamespace(version="changed" if damage == "version" else "synthetic-v1"))
    monkeypatch.setattr(backend, "CustomerContactSearchService", lambda: SimpleNamespace(index_path=lambda **kw: path))
    monkeypatch.setattr(backend, "_read_database", lambda *args: nullcontext(Mock()))
    metadata = Mock()
    if damage == "metadata":
        metadata.side_effect = backend._IndexNotReady()
    elif damage == "unavailable":
        metadata.side_effect = OSError("unavailable")
    elif damage == "corrupt":
        metadata.side_effect = ValueError("corrupt")
    monkeypatch.setattr(backend, "_metadata", metadata)
    lookup = Mock(side_effect=AssertionError("No lookup on GUI handoff"))
    monkeypatch.setattr(backend, "search_customers_by_contact", lookup)
    result = data.validate_contact_match(matched(), raw_root=tmp_path)
    assert result.status is expected
    lookup.assert_not_called()
    assert result.canonical_customer_id == ("C" if expected is S.MATCHED else None)
