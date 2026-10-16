"""Repeat lookup and Windows reader-lock races, using synthetic sources only."""

import os
from contextlib import closing, contextmanager
import json
import logging
import sqlite3
from threading import Event, Thread, get_ident
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest

import pytest

from Application import customer_card_data as card_data, customer_contact_search as contacts, customer_profile_service as profiles
from Application.customer_card_data import ViewState
from Application.customer_contact_search import ContactSearchResult, ContactSearchStatus, ContactType
from Application.customer_profile_service import CustomerProfileService, _reader_lock
from Application.customer_history_service import HistoryPage, HistoryStatus
from tests.test_customer_contact_search import source as source_fixture, write_profiles, profile

REAL_VALIDATE_CONTACT = card_data.validate_contact_match
from tests.tabs.test_customer_contact_search_ui import app as app, card as card, choose, search, wait


@pytest.mark.skipif(os.name != "nt", reason="Requires actual Windows msvcrt byte-range locks")
def test_windows_second_contact_search_does_not_race_early_gui_validation(card, tmp_path):
    search(card, ContactType.PHONE)
    wait(lambda: card.tab.history._active is None)
    assert card.tab.result.state is ViewState.READY
    root = tmp_path / "lock-source"
    (root / "canonical").mkdir(parents=True)
    (root / "canonical/.writer.lock").write_bytes(b"x")
    gui = get_ident()
    gui_entered, worker_checked = Event(), Event()
    gui_checks = []
    def gate(match, **kwargs):
        if get_ident() == gui:
            gui_checks.append(True)
            with _reader_lock(root):
                gui_entered.set()
                worker_checked.wait(0.5)
            return match
        try:
            with _reader_lock(root):
                return match
        except OSError:
            return ContactSearchResult(ContactSearchStatus.SOURCE_UNAVAILABLE)
        finally:
            worker_checked.set()
    card.gate.side_effect = gate
    previous = card.service.inference.get_recommendations.return_value
    def inference(*args, **kwargs):
        # Before the fix the early GUI slot enters gate while BPR is running.
        # After the fix no source I/O runs there; the bounded wait simply expires.
        gui_entered.wait(0.2)
        return previous
    card.service.inference.get_recommendations.side_effect = inference
    search(card, ContactType.PHONE, "+79001112234")
    wait(lambda: card.tab.history._active is None)
    assert card.tab.result.state is ViewState.READY
    assert not gui_checks
    with _reader_lock(root):
        pass  # No handle/lock survives the completed request.


@pytest.fixture
def real_contacts(card, tmp_path, monkeypatch):
    # Directly create a tiny synthetic schema. No index builder is invoked.
    root, backend = source_fixture.__wrapped__(tmp_path)
    values = [(f"C{n}", profile(f"C{n}", email=f"repeat{n}@example.test", mobilePhone=f"+790011122{n:02d}",
                              discountCards=[{"ids": {"number": f"000{n}"}}])) for n in range(10)]
    write_profiles(root, values)
    version = contacts._sources(root).version
    path = backend.index_path(raw_root=root)
    path.parent.mkdir(parents=True)
    key = bytes(range(32))
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("CREATE TABLE metadata (key TEXT PRIMARY KEY,value NOT NULL);"
                         "CREATE TABLE contacts (kind TEXT NOT NULL,contact_key BLOB NOT NULL,customer_id TEXT NOT NULL,"
                         "PRIMARY KEY(kind,contact_key,customer_id)) WITHOUT ROWID;")
        db.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", contacts.INDEX_SCHEMA_VERSION), ("normalization_version", contacts.NORMALIZATION_VERSION),
            ("source_version", version), ("hmac_key", key), ("processed_profiles", 10), ("index_entries", 30)])
        for customer, raw in values:
            for kind, normalized in contacts._contacts(raw):
                db.execute("INSERT INTO contacts VALUES (?,?,?)", (kind.value, contacts._contact_key(key, kind, normalized), customer))
    monkeypatch.setattr(card_data, "validate_contact_match", REAL_VALIDATE_CONTACT)
    monkeypatch.setattr(contacts, "CustomerContactSearchService", lambda: backend)
    card.backend.side_effect = lambda kind, value, **kwargs: backend.search(kind, value, **kwargs)
    card.service.raw_root = root
    card.service.profile_service = CustomerProfileService()
    connections, lock_threads, history_calls = [], [], []
    original_connect = sqlite3.connect
    class Tracked(sqlite3.Connection):
        closed = False
        def close(self):
            super().close()
            self.closed = True
    def connect(*args, **kwargs):
        kwargs['factory'] = Tracked
        db = original_connect(*args, **kwargs)
        connections.append((get_ident(), db))
        return db
    monkeypatch.setattr(sqlite3, "connect", connect)
    @contextmanager
    def lock(raw_root):
        lock_threads.append(get_ident())
        with _reader_lock(raw_root):
            yield
    monkeypatch.setattr(contacts, "_reader_lock", lock)
    monkeypatch.setattr(profiles, "_reader_lock", lock)
    def history(customer, **kwargs):
        history_calls.append(customer)
        with lock(root):
            return HistoryPage(HistoryStatus.EMPTY, customer)
    card.tab._history_lookup = history
    return root, backend, connections, lock_threads, history_calls


def test_ten_sequential_real_contacts_and_repeat_keep_worker_and_release_resources(card, real_contacts, caplog):
    root, backend, connections, lock_threads, history = real_contacts
    gui = get_ident()
    caplog.set_level(logging.DEBUG)
    thread = None
    for n in list(range(10)) + [9]:
        search(card, ContactType.PHONE, f"+790011122{n:02d}")
        wait(lambda: card.tab.history._active is None)
        assert card.tab.result.state is ViewState.READY
        assert card.tab.result.profile.canonical_customer_id == f"C{n}"
        assert card.tab.history.customer_id == f"C{n}"
        if thread is None:
            thread = card.tab._thread
        assert card.tab._thread is thread
        assert all(db.closed for _, db in connections)
        assert all(t != gui for t, _ in connections)
        assert all(t != gui for t in lock_threads)
        with _reader_lock(root):
            pass
    assert history == [f"C{n}" for n in range(10)] + ["C9"]
    assert card.backend.call_count == 11
    assert "stage=sqlite_closed" in caplog.text and "stage=validated" in caplog.text
    assert "stage=applied" in caplog.text and "stage=released" in caplog.text
    for forbidden in ("+790011122", "repeat0@", "0000", "hmac_key", str(root)):
        assert forbidden not in "\n".join(record.getMessage() for record in caplog.records if record.name.startswith("Application"))


def test_switch_real_lookup_modes_preserves_canonical_identity(card, real_contacts):
    cases = [(ContactType.PHONE, "+79001112200", "C0"), (ContactType.EMAIL, "repeat1@example.test", "C1"),
             (ContactType.DISCOUNT_CARD, "0002", "C2"), (None, "C3", "C3"),
             (ContactType.PHONE, "+79001112204", "C4")]
    for kind, value, customer in cases:
        search(card, kind, value)
        wait(lambda: card.tab.history._active is None)
        assert card.tab.result.state is ViewState.READY
        assert card.tab.result.profile.canonical_customer_id == customer
        assert card.tab.history.customer_id == customer


@pytest.mark.parametrize("change", ["source", "index"])
def test_actual_source_or_index_change_at_delivery_rejects_old_identity(card, real_contacts, change):
    root, backend, connections, _, _ = real_contacts
    original = card.tab._queue_validation
    def queue(request_id, result):
        if change == "source":
            path = root / "canonical/catalog.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata['customer_merges']['until'] = '2026-10-01T00:00:00+00:00'
            path.write_text(json.dumps(metadata),encoding="utf-8")
        else:
            with closing(sqlite3.connect(backend.index_path(raw_root=root))) as db, db:
                db.execute("UPDATE metadata SET value='changed' WHERE key='source_version'")
        original(request_id, result)
    card.tab._queue_validation = queue
    search(card, ContactType.PHONE, "+79001112200")
    wait(lambda: card.tab.history._active is None)
    assert card.tab.result.contact_search.status is ContactSearchStatus.INDEX_NOT_READY
    assert not card.tab.product_grid.cards and card.tab.history.customer_id is None
    assert all(db.closed for _, db in connections)


def test_actual_source_unavailable_is_logged_without_private_values(card, real_contacts, caplog):
    root, _, connections, _, _ = real_contacts
    (root / "canonical/.writer.lock").unlink()
    caplog.set_level(logging.DEBUG)
    search(card, ContactType.PHONE, "+79001112200")
    assert card.tab.result.contact_search.status is ContactSearchStatus.SOURCE_UNAVAILABLE
    assert "contact_search stage=lock error=FileNotFoundError" in caplog.text
    assert "+79001112200" not in caplog.text and str(root) not in caplog.text
    assert all(db.closed for _, db in connections)



def test_new_contact_waits_for_running_history_without_gui_io_or_stale_delivery(card, real_contacts):
    root, _, connections, lock_threads, _ = real_contacts
    entered, release = Event(), Event()
    history_calls = []
    def history(customer, **kwargs):
        history_calls.append(customer)
        with _reader_lock(root):
            if customer == "C0":
                entered.set()
                assert release.wait(4)
        return HistoryPage(HistoryStatus.EMPTY, customer)
    card.tab._history_lookup = history
    search(card, ContactType.PHONE, "+79001112200")
    wait(entered.is_set)
    worker_thread = card.tab._thread
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(True))
    card.tab.customer_input.setText("+79001112201")
    card.tab.search()
    try:
        QTest.qWait(20)
        assert ticks and card.tab.result is None
    finally:
        release.set()
    wait(lambda: card.tab.result is not None and card.tab.history._active is None)
    assert card.tab.result.profile.canonical_customer_id == "C1"
    assert card.tab.history.customer_id == "C1" and history_calls == ["C0", "C1"]
    assert card.tab._thread is worker_thread
    assert all(db.closed for _, db in connections)
    assert all(t != get_ident() for t in lock_threads)


def test_new_search_cancels_inflight_delivery_validation(card, real_contacts):
    entered, release = Event(), Event()
    stage = []
    original_queue = card.tab._queue_validation
    original_validate = card.service.validate_result
    def queue(request_id, result):
        stage.append(True)
        original_queue(request_id, result)
    def validate(result):
        if len(stage) == 1 and result.state is ViewState.READY:
            entered.set()
            assert release.wait(4)
        return original_validate(result)
    card.tab._queue_validation = queue
    card.service.validate_result = validate
    choose(card.tab, ContactType.PHONE)
    card.tab.customer_input.setText("+79001112200")
    card.tab.search()
    try:
        wait(entered.is_set)
        card.tab.customer_input.setText("+79001112201")
        card.tab.search()
    finally:
        release.set()
    wait(lambda: card.tab.result is not None and card.tab.history._active is None)
    assert card.tab.result.profile.canonical_customer_id == "C1"
    assert card.tab.history.customer_id == "C1"


@pytest.mark.skipif(os.name != "nt", reason="Requires actual Windows msvcrt byte-range locks")
def test_windows_contact_validation_reports_busy_lock_and_recovers_after_release(card, real_contacts, caplog):
    root, backend, _, _, _ = real_contacts
    match = backend.search(ContactType.PHONE, "+79001112200", raw_root=root)
    assert match.status is ContactSearchStatus.MATCHED
    results = []
    caplog.set_level(logging.DEBUG)
    with _reader_lock(root):
        thread = Thread(target=lambda: results.append(REAL_VALIDATE_CONTACT(match, raw_root=root)))
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
    assert results[0].status is ContactSearchStatus.SOURCE_UNAVAILABLE
    assert "contact_validation stage=lock error=PermissionError" in caplog.text
    assert REAL_VALIDATE_CONTACT(match, raw_root=root).status is ContactSearchStatus.MATCHED
    assert "+79001112200" not in caplog.text and str(root) not in caplog.text


@pytest.mark.parametrize("damage", ["metadata", "query"])
def test_failed_real_sqlite_lookup_closes_connection_and_releases_lock(card, real_contacts, damage):
    root, backend, connections, _, _ = real_contacts
    with closing(sqlite3.connect(backend.index_path(raw_root=root))) as db, db:
        if damage == "metadata":
            db.execute("DELETE FROM metadata WHERE key='hmac_key'")
        else:
            db.execute("DROP TABLE contacts")
    search(card, ContactType.PHONE, "+79001112200")
    assert card.tab.result.contact_search.status is ContactSearchStatus.SOURCE_INCONSISTENT
    assert all(db.closed for _, db in connections)
    with _reader_lock(root):
        pass
