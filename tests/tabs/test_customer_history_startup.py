"""History handoff races on synthetic metadata locks; no real SQLite or training."""

from pathlib import Path
from threading import Event, get_ident

import pytest
from PyQt6.QtTest import QTest

from Application.customer_card_data import ViewState
from Application.customer_contact_search import ContactType
from Application.customer_history_normalization import normalize_order
from Application.customer_history_records import HistoryKind
from Application.customer_history_service import HistoryPage, HistoryStatus, HistoryVariant, LogicalHistoryEvent
from Application.customer_profile_service import _reader_lock
from Application.mindbox.identity import CustomerIdResolver
from Application.model.customer_identity_bridge import CustomerModelIdentityResult, CustomerModelIdentityStatus
from tests.tabs.test_customer_contact_search_ui import VALUES, app as app, card as card, choose, matched, wait
from tests.test_customer_history_index import order


def order_page(customer):
    event = normalize_order(order(customer=customer), CustomerIdResolver(()))
    logical = LogicalHistoryEvent(event.event_key, event.source, event.source_event_id, customer, (customer,),
                                 event.kind, event.event_datetime_utc, (HistoryVariant(event, (event.provenance,)),),
                                 1, False, False, False, event.quality)
    return HistoryPage(HistoryStatus.READY, customer, (logical,), generation="g", source_version="v")


@pytest.mark.parametrize("kind", list(VALUES))
def test_first_history_waits_for_worker_validation_without_lock_collision(card, tmp_path, kind):
    root = Path(tmp_path)
    (root / "canonical").mkdir()
    (root / "canonical/.writer.lock").write_bytes(b"x")
    profile_ready, finish_bpr = Event(), Event()
    validating, history_attempted = Event(), Event()
    gui = get_ident()
    calls = []
    original_inference = card.service.inference.get_recommendations.return_value
    def infer(*args, **kwargs):
        profile_ready.set()
        assert finish_bpr.wait(4)
        return original_inference
    card.service.inference.get_recommendations.side_effect = infer
    original_validate = card.service.validate_result
    def validate(value):
        assert get_ident() != gui
        with _reader_lock(root):
            validating.set()
            return original_validate(value)
    card.service.validate_result = validate
    def lookup(customer, **kwargs):
        calls.append((customer, kwargs))
        assert validating.wait(2)
        try:
            with _reader_lock(root):
                return order_page(customer)
        except OSError:
            return HistoryPage(HistoryStatus.SOURCE_UNAVAILABLE, customer)
        finally:
            history_attempted.set()
    card.tab._history_lookup = lookup
    choose(card.tab, kind)
    card.tab.customer_input.setText(VALUES[kind])
    card.tab.search()
    try:
        wait(profile_ready.is_set)
        wait(lambda: card.tab.profile.content.state is ViewState.READY)
        assert card.tab.result is None
    finally:
        finish_bpr.set()
    wait(lambda: card.tab.result is not None and card.tab.history._active is None)
    history = card.tab.history
    assert history.property("historyStatus") == HistoryStatus.READY.value
    assert history.table.rows.rowCount() == 2
    assert len(calls) == 1 and calls[0][0] == "C"
    assert calls[0][1]["kinds"] == (HistoryKind.ORDER,) and calls[0][1]["limit"] == 50
    assert tuple(product.content.recommendation for product in card.tab.raw_grid.cards) == card.raw
    page_rows = history.table.rows.rowCount()
    card.tab.apply_result(card.tab.result)
    QTest.qWait(20)
    assert len(calls) == 1 and history.table.rows.rowCount() == page_rows


@pytest.mark.parametrize("status", [HistoryStatus.SOURCE_UNAVAILABLE, HistoryStatus.INDEX_NOT_READY])
def test_real_history_failure_is_not_retried_or_mapped_to_empty(card, status):
    calls = []
    def lookup(customer, **kwargs):
        calls.append(customer)
        return HistoryPage(status, customer)
    card.tab._history_lookup = lookup
    choose(card.tab, ContactType.PHONE)
    card.tab.customer_input.setText(VALUES[ContactType.PHONE])
    card.tab.search()
    wait(lambda: card.tab.result is not None and card.tab.history._active is None)
    assert card.tab.history.property("historyStatus") == status.value
    assert card.tab.history.table.isHidden()
    assert "недоступна" in card.tab.history.message.text()
    QTest.qWait(30)
    assert calls == ["C"]


def test_first_history_without_bpr_and_repeated_find(card):
    card.service.bridge.resolve.side_effect = lambda customer, **kwargs: CustomerModelIdentityResult(
        customer, None, CustomerModelIdentityStatus.NOT_IN_MODEL, 0)
    calls = []
    def lookup(customer, **kwargs):
        calls.append(customer)
        return order_page(customer)
    card.tab._history_lookup = lookup
    choose(card.tab, ContactType.EMAIL)
    card.tab.customer_input.setText(VALUES[ContactType.EMAIL])
    for attempt in range(2):
        card.tab.search()
        wait(lambda: card.tab.result is not None and card.tab.history._active is None)
        assert card.tab.profile.content.state is ViewState.READY
        assert not card.tab.raw_grid.cards
        assert card.tab.history.table.rows.rowCount() == 2
        assert len(calls) == attempt + 1


@pytest.mark.parametrize("replacement", ["customer", "cancel", "invalid"])
def test_cancelled_deferred_history_does_not_block_next_search(card, replacement):
    entered, release = Event(), Event()
    original = card.service.inference.get_recommendations.return_value
    count = []
    def infer(*args, **kwargs):
        count.append(True)
        if len(count) == 1:
            entered.set()
            assert release.wait(4)
        return original
    card.service.inference.get_recommendations.side_effect = infer
    calls = []
    def lookup(customer, **kwargs):
        calls.append(customer)
        return order_page(customer)
    card.tab._history_lookup = lookup
    choose(card.tab, None)
    card.tab.customer_input.setText("C")
    card.tab.search()
    try:
        wait(entered.is_set)
        wait(lambda: card.tab.profile.content.state is ViewState.READY)
        if replacement == "customer":
            card.backend.return_value = matched("D")
            card.tab.customer_input.setText("D")
            card.tab.search()
        elif replacement == "cancel":
            card.tab.cancel_search()
        else:
            card.tab.customer_input.setText(" ")
            card.tab.search()
    finally:
        release.set()
    wait(lambda: card.tab._active is None and card.tab.history._active is None)
    if replacement == "customer":
        assert calls == ["D"] and card.tab.history.customer_id == "D"
        assert card.tab.history.table.rows.rowCount() == 2
    else:
        assert not calls
        card.tab.customer_input.setText("C")
        card.tab.search()
        wait(lambda: card.tab.result is not None and card.tab.history._active is None)
        assert calls == ["C"] and card.tab.history.table.rows.rowCount() == 2
