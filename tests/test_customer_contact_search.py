"""Synthetic-only contact search, publication and snapshot regressions."""

from contextlib import closing
from dataclasses import FrozenInstanceError, asdict
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import Event

import pytest

from Application import customer_contact_search as module
from Application.customer_contact_search import (
    ContactSearchResult, ContactSearchStatus as S, ContactType,
    CustomerContactSearchService, IndexBuildStatus as B,
    normalize_indexed_contact, normalize_search_input,
)
from Application.mindbox.selection import DEFAULT_SELECTION


def profile(customer, **fields):
    return {"ids": {"mindboxId": customer}, **fields}


def write_profiles(root, rows):
    with closing(sqlite3.connect(root / "canonical/customers.sqlite")) as connection, connection:
        connection.execute("DELETE FROM profiles")
        connection.executemany("INSERT INTO profiles VALUES (?, '', ?)",
                               [(key, json.dumps(raw)) for key, raw in rows])


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "raw"
    directory = root / "canonical/objects" / ("1" * 32) / "customer_merges"
    directory.mkdir(parents=True)
    (directory / "customer_merges_part_001.json").write_text('{"customerMerges": []}', encoding="utf-8")
    (root / "canonical/.writer.lock").write_bytes(b"0")
    (root / "canonical/catalog.json").write_text(json.dumps({
        "schema_version": 1, "revision": "synthetic", "actions": {}, "orders": {},
        "customer_merges": {"directory": directory.relative_to(root).as_posix(), "parts": 1,
                            "since": "2022-01-01T00:00:00+00:00", "until": "2026-09-01T00:00:00+00:00"},
        "selection": asdict(DEFAULT_SELECTION),
    }), encoding="utf-8")
    with closing(sqlite3.connect(root / "canonical/customers.sqlite")) as connection, connection:
        connection.execute("CREATE TABLE profiles (id TEXT PRIMARY KEY, changed TEXT, raw TEXT NOT NULL)")
    write_profiles(root, [("C", profile("C", email=" Person+tag@Example.test ", mobilePhone="+7 (900) 111-22-33",
                                       discountCards=[{"ids": {"number": "000123"}}, {"ids": {"number": "000456"}}],
                                       lastActivatedCard={"ids": {"number": "000789"}}))])
    return root, CustomerContactSearchService(cache_dir=tmp_path / "cache")


def build(source, **kwargs):
    return source[1].build_index(raw_root=source[0], **kwargs)


def search(source, kind="email", value="person+tag@example.test"):
    return source[1].search(kind, value, raw_root=source[0])


@pytest.mark.parametrize("kind,value", [
    ("email", "PERSON+tag@EXAMPLE.test"), (ContactType.PHONE, "79001112233"),
    ("PHONE", "0079001112233"), ("phone", "+7 900 111 22 33"),
    ("discount_card", "000123"), ("DISCOUNT_CARD", " 000456 "), ("discount_card", "000789"),
])
def test_exact_lookup_three_types_and_all_cards(source, kind, value):
    result = build(source)
    assert result.status is B.READY and result.processed_profiles == 1 and result.index_entries == 5
    found = search(source, kind, value)
    assert found.status is S.MATCHED and found.canonical_customer_id == "C" and found.candidate_count == 1
    assert found.lookup_ms is not None and found.source_version == result.source_version
    with pytest.raises(FrozenInstanceError):
        found.candidate_count = 2


@pytest.mark.parametrize("kind,value,normalized", [
    ("email", " A.B+tag@Example.test ", "a.b+tag@example.test"),
    ("phone", "+44 (20) 1234-5678", "+442012345678"),
    ("phone", "0044 20 1234 5678", "+442012345678"),
    ("phone", "79001112233", "+79001112233"),
    ("discount_card", " 000123 ", "000123"), ("discount_card", "ABC-0001", "ABC-0001"),
])
def test_shared_normalization_has_distinct_source_and_query_entrypoints(kind, value, normalized):
    assert normalize_search_input(kind, value) == normalized
    assert normalize_indexed_contact(kind, value) == normalized


@pytest.mark.parametrize("kind,value", [
    ("email", "a..b@example.test"), ("email", ".a@example.test"), ("email", "a.@example.test"),
    ("email", "a@-bad.test"), ("email", "a@bad..test"), ("email", "a@example"),
    ("email", "a b@example.test"), ("email", "a@@example.test"), ("email", "NaN"),
    ("email", {"email": "a@example.test"}), ("phone", "9001112233"), ("phone", "89001112233"),
    ("phone", "+79001112233ext1"), ("phone", "+79001112233.0"), ("phone", "+0123456789"),
    ("phone", "++79001112233"), ("phone", True), ("phone", 79001112233.0),
    ("phone", "+7900111223"), ("phone", "+190011122334"),
    ("discount_card", True), ("discount_card", 123.0), ("discount_card", "000 123"),
    ("discount_card", "null"), ("discount_card", "000\x00123"), ("unknown", "value"),
    ("email", ""), (None, "value"),
])
def test_invalid_input_does_not_read_or_build(source, kind, value, monkeypatch):
    monkeypatch.setattr(module, "_sources", lambda *args: pytest.fail("Invalid query must not touch sources"))
    assert search(source, kind, value).status is S.INVALID_INPUT
    assert not source[1].cache_dir.exists()


def test_index_integer_identifiers_but_queries_require_strings():
    assert normalize_indexed_contact("phone", 79001112233) == "+79001112233"
    assert normalize_indexed_contact("discount_card", 123) == "123"
    assert normalize_search_input("discount_card", 123) is None
    assert normalize_search_input("phone", 79001112233) is None
    assert normalize_indexed_contact("discount_card", -1) is None


def test_no_suffix_matching_or_email_rewriting_or_zero_removal(source):
    write_profiles(source[0], [("C", profile("C", email="a.b+tag@example.test", mobilePhone="+44 20 1234 5678",
                                            discountCards=[{"ids": {"number": "000123"}}])),
                               ("D", profile("D", mobilePhone="+49 20 1234 5678"))])
    assert build(source).status is B.READY
    for email in ("ab+tag@example.test", "a.b@example.test"):
        assert search(source, "email", email).status is S.NOT_FOUND
    assert search(source, "discount_card", "123").status is S.NOT_FOUND
    assert search(source, "phone", "+442012345678").canonical_customer_id == "C"
    assert search(source, "phone", "+492012345678").canonical_customer_id == "D"
    assert search(source, "phone", "2012345678").status is S.INVALID_INPUT


@pytest.mark.parametrize("kind,value,fields", [
    ("email", "same@example.test", {"email": "same@example.test"}),
    ("phone", "+79001112233", {"mobilePhone": "+79001112233"}),
    ("discount_card", "000123", {"discountCards": [{"ids": {"number": "000123"}}]}),
])
def test_shared_contacts_never_choose_a_customer(source, kind, value, fields):
    write_profiles(source[0], [(c, profile(c, **fields)) for c in ("C", "D", "E")])
    assert build(source).status is B.READY
    result = search(source, kind, value)
    assert result.status is S.AMBIGUOUS and result.candidate_count == 3
    assert result.canonical_customer_id is None
    assert not hasattr(result, "candidates")


def write_merges(root, edges):
    path = next((root / "canonical/objects").rglob("customer_merges_part_001.json"))
    rows = [{"id": f"synthetic-{i}", "dateTimeUtc": "2026-01-01T00:00:00Z",
             "resultingCustomer": {"ids": {"mindboxId": target}},
             "mergedCustomers": [{"ids": {"mindboxId": alias}}]} for i, (alias, target) in enumerate(edges)]
    path.write_text(json.dumps({"customerMerges": rows}), encoding="utf-8")


def test_merge_chains_and_duplicate_cards_count_canonical_customers(source):
    root = source[0]
    write_merges(root, [("A", "B"), ("B", "C")])
    card = {"ids": {"number": "000123"}}
    write_profiles(root, [(c, profile(c, email="same@example.test", discountCards=[card, card],
                                      lastActivatedCard=card)) for c in ("A", "B", "C")])
    result = build(source)
    assert result.status is B.READY and result.index_entries == 2
    assert search(source, "email", "same@example.test").canonical_customer_id == "C"
    assert search(source, "discount_card", "000123").candidate_count == 1


def test_damaged_optional_values_do_not_hide_valid_other_contacts(source):
    write_profiles(source[0], [("C", profile("C", email=42, mobilePhone=True,
                                            discountCards=[None, {"ids": "bad"}, {"ids": {"number": True}},
                                                           {"ids": {"number": 12.3}}, {"ids": {"number": "000123"}}],
                                            lastActivatedCard=["bad"])),
                               ("D", profile("D", email="valid@example.test", mobilePhone=[], discountCards="bad")),
                               ("E", profile("E"))])
    assert build(source).status is B.READY
    assert search(source, "discount_card", "000123").canonical_customer_id == "C"
    assert search(source, "email", "valid@example.test").canonical_customer_id == "D"
    assert search(source, "email", "missing@example.test").status is S.NOT_FOUND


@pytest.mark.parametrize("flag", [True, "false", 0, 1])
def test_invalid_or_malformed_contact_flags_do_not_confirm(source, flag):
    write_profiles(source[0], [("C", profile("C", email="person@example.test", isEmailInvalid=flag,
                                            mobilePhone="+79001112233", isMobilePhoneInvalid=flag))])
    assert build(source).status is B.READY
    assert search(source, "email", "person@example.test").status is S.NOT_FOUND
    assert search(source, "phone", "+79001112233").status is S.NOT_FOUND


def test_missing_index_and_stale_index_never_rebuild_on_lookup(source, monkeypatch):
    assert search(source).status is S.INDEX_NOT_READY
    assert not source[1].cache_dir.exists()
    assert build(source).status is B.READY
    write_profiles(source[0], [("D", profile("D", email="person+tag@example.test"))])
    monkeypatch.setattr(source[1], "build_index", lambda **kwargs: pytest.fail("No implicit build"))
    monkeypatch.setattr(module, "_profile_rows", lambda *args: pytest.fail("No source scan"))
    monkeypatch.setattr(module, "iter_export", lambda *args, **kwargs: pytest.fail("No merge scan"))
    assert search(source).status is S.INDEX_NOT_READY


def test_build_reuse_is_persistent_and_lookup_uses_covering_index(source, monkeypatch):
    first = build(source)
    service = CustomerContactSearchService(cache_dir=source[1].cache_dir)
    path = service.index_path(raw_root=source[0])
    before = path.read_bytes()
    monkeypatch.setattr(module, "_profile_rows", lambda *args: pytest.fail("No repeated profile scan"))
    monkeypatch.setattr(module, "iter_export", lambda *args, **kwargs: pytest.fail("No repeated merge scan"))
    reused = service.build_index(raw_root=source[0])
    assert reused.status is B.READY and reused.reused and reused.source_version == first.source_version
    assert service.search("email", "PERSON+TAG@EXAMPLE.TEST", raw_root=source[0]).status is S.MATCHED
    assert path.read_bytes() == before
    with closing(sqlite3.connect(path)) as connection, connection:
        plan = connection.execute("EXPLAIN QUERY PLAN SELECT COUNT(*), MIN(customer_id) FROM contacts "
                                  "WHERE kind=? AND contact_key=?", ("email", b"synthetic")).fetchall()
    assert any("SEARCH contacts USING PRIMARY KEY" in row[3] for row in plan)


def test_orders_actions_revision_updates_do_not_invalidate(source):
    first = build(source)
    path = source[0] / "canonical/catalog.json"
    data = json.loads(path.read_text())
    data.update(revision="different", actions={"unrelated": {}}, orders={"unrelated": {}})
    path.write_text(json.dumps(data), encoding="utf-8")
    assert search(source).status is S.MATCHED
    result = build(source)
    assert result.reused and result.source_version == first.source_version


@pytest.mark.parametrize("change", ["customers", "replacement", "merges", "merge_metadata", "schema", "normalization"])
def test_relevant_versions_invalidate_and_explicit_build_restores(source, change, monkeypatch):
    first = build(source)
    root, service = source
    if change == "customers":
        write_profiles(root, [("C", profile("C", email="changed@example.test"))])
    elif change == "replacement":
        path = root / "canonical/customers.sqlite"
        replacement = root.parent / "replacement.sqlite"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif change == "merges":
        write_merges(root, [("C", "D")])
    elif change == "merge_metadata":
        path = root / "canonical/catalog.json"
        data = json.loads(path.read_text())
        data["customer_merges"]["until"] = "2026-10-01T00:00:00+00:00"
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        monkeypatch.setattr(module, "INDEX_SCHEMA_VERSION" if change == "schema" else "NORMALIZATION_VERSION", 2)
    assert search(source).status is S.INDEX_NOT_READY
    result = build(source)
    assert result.status is B.READY and not result.reused
    if change not in ("schema", "normalization"):
        assert result.source_version != first.source_version
    value = "changed@example.test" if change == "customers" else "person+tag@example.test"
    assert search(source, "email", value).canonical_customer_id == ("D" if change == "merges" else "C")


def test_cancelled_rebuild_preserves_previous_publication_and_cleans_temporary(source):
    assert build(source).status is B.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    cancel = Event()
    events = []
    def progress(event):
        events.append(event)
        if event.phase == "customers":
            cancel.set()
    result = build(source, force=True, cancelled=cancel.is_set, progress=progress, progress_every=1)
    assert result.status is B.CANCELLED and events
    assert path.read_bytes() == before and not list(source[1].cache_dir.glob(".building-*"))
    assert search(source).status is S.MATCHED
    assert build(source, cancelled=lambda: True).status is B.CANCELLED


def test_change_during_build_or_lookup_never_publishes_confirmed_identity(source, monkeypatch):
    assert build(source).status is B.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    real_sources = module._sources
    calls = 0
    def unstable(root):
        nonlocal calls
        calls += 1
        result = real_sources(root)
        return result if calls == 1 else module._Sources("changed", result.merges_directory)
    monkeypatch.setattr(module, "_sources", unstable)
    assert build(source, force=True).status is B.INDEX_NOT_READY
    assert path.read_bytes() == before
    calls = 0
    assert search(source).status is S.INDEX_NOT_READY


def test_concurrent_build_does_not_queue_another_full_scan(source):
    source[1]._build_lock.acquire()
    try:
        assert build(source).status is B.INDEX_NOT_READY
    finally:
        source[1]._build_lock.release()


def test_canonical_unchanged_and_no_pii_in_cache_repr_or_diagnostics(source, caplog, capsys):
    root, service = source
    before = {p.relative_to(root): hashlib.sha256(p.read_bytes()).digest() for p in root.rglob("*") if p.is_file()}
    progress = []
    built = build(source, progress=progress.append, progress_every=1)
    found = search(source)
    after = {p.relative_to(root): hashlib.sha256(p.read_bytes()).digest() for p in root.rglob("*") if p.is_file()}
    assert after == before
    encoded = service.index_path(raw_root=root).read_bytes()
    output = repr(built) + repr(found) + repr(progress) + caplog.text + str(capsys.readouterr())
    for value in ("Person+tag", "person+tag", "Example.test", "example.test", "79001112233", "000123", "000456", "000789"):
        assert value.encode() not in encoded and value not in output
    assert "canonical_customer_id" not in repr(found)


@pytest.mark.parametrize("component", ["customers.sqlite", "catalog.json", ".writer.lock", "merges"])
def test_missing_sources_are_not_not_found(source, component):
    root = source[0]
    path = next((root / "canonical/objects").rglob("customer_merges_part_001.json")) if component == "merges" else root / "canonical" / component
    path.unlink()
    assert build(source).status is B.SOURCE_UNAVAILABLE
    assert search(source).status is S.SOURCE_UNAVAILABLE


@pytest.mark.parametrize("damage", ["identity", "json", "duplicate_json", "cycle", "conflict", "schema", "wal", "journal"])
def test_inconsistent_or_busy_sources_never_publish(source, damage):
    root = source[0]
    if damage == "identity":
        write_profiles(root, [("C", profile("other", email="person+tag@example.test"))])
    elif damage in ("json", "duplicate_json"):
        raw = '{"email": "x", "email": "y"}' if damage == "duplicate_json" else '{"broken":'
        with closing(sqlite3.connect(root / "canonical/customers.sqlite")) as connection, connection:
            connection.execute("UPDATE profiles SET raw=?", (raw,))
    elif damage in ("cycle", "conflict"):
        write_merges(root, [("A", "B"), ("B", "A")] if damage == "cycle" else [("A", "B"), ("A", "C")])
    elif damage == "schema":
        with closing(sqlite3.connect(root / "canonical/customers.sqlite")) as connection, connection:
            connection.execute("DROP TABLE profiles")
            connection.execute("CREATE TABLE profiles (id TEXT, raw TEXT)")
    else:
        Path(str(root / "canonical/customers.sqlite") + "-" + damage).write_bytes(b"pending")
    result = build(source)
    assert result.status is (B.SOURCE_UNAVAILABLE if damage in ("wal", "journal") else B.SOURCE_INCONSISTENT)
    assert not source[1].index_path(raw_root=root).exists()


def test_callback_failure_is_safe_and_old_index_is_retained(source, caplog, capsys):
    assert build(source).status is B.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    def failed(event):
        raise ValueError("synthetic-private-contact@example.test")
    assert build(source, force=True, progress=failed, progress_every=1).status is B.SOURCE_INCONSISTENT
    assert path.read_bytes() == before
    assert "synthetic-private-contact" not in caplog.text + str(capsys.readouterr())
    assert not list(source[1].cache_dir.glob(".building-*"))


def test_public_wrappers_and_result_invariants(source, monkeypatch):
    monkeypatch.setattr(module, "_DEFAULT_SERVICE", source[1])
    assert module.build_customer_contact_index(raw_root=source[0]).status is B.READY
    assert module.search_customers_by_contact(contact_type="email", value="person+tag@example.test", raw_root=source[0]).status is S.MATCHED
    with pytest.raises(ValueError):
        ContactSearchResult(S.MATCHED, "C", 2)
    with pytest.raises(ValueError):
        ContactSearchResult(S.AMBIGUOUS, "C", 2)
    with pytest.raises(ValueError):
        ContactSearchResult(S.AMBIGUOUS, candidate_count=1)
    with pytest.raises(ValueError):
        ContactSearchResult(S.NOT_FOUND, candidate_count=1)


@pytest.mark.parametrize("damage", ["file", "metadata", "table"])
def test_corrupt_cache_is_explicit_and_repairable_without_touching_source(source, damage):
    assert build(source).status is B.READY
    path = source[1].index_path(raw_root=source[0])
    if damage == "file":
        path.write_bytes(b"invalid index")
    else:
        with closing(sqlite3.connect(path)) as connection, connection:
            if damage == "metadata":
                connection.execute("UPDATE metadata SET value=? WHERE key='hmac_key'", (b"bad",))
            else:
                connection.execute("DROP TABLE contacts")
    assert search(source).status is S.SOURCE_INCONSISTENT
    assert build(source).status is B.READY
    assert search(source).status is S.MATCHED


def test_failed_atomic_publication_retains_old_complete_index(source, monkeypatch):
    assert build(source).status is B.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    def failed(*args):
        raise PermissionError("Synthetic publication blocked")
    monkeypatch.setattr(module.os, "replace", failed)
    assert build(source, force=True).status is B.SOURCE_UNAVAILABLE
    assert path.read_bytes() == before and search(source).status is S.MATCHED
    assert not list(source[1].cache_dir.glob(".building-*"))


def test_batches_and_progress_do_not_require_an_all_profiles_collection(source, monkeypatch):
    write_profiles(source[0], [(f"C{i:04d}", profile(f"C{i:04d}", email=f"user{i}@example.test")) for i in range(2101)])
    original = module._profile_rows
    emitted = 0
    def stream(connection):
        nonlocal emitted
        for row in original(connection):
            emitted += 1
            yield row
    monkeypatch.setattr(module, "_profile_rows", stream)
    events = []
    result = build(source, progress=events.append, progress_every=1000)
    assert result.status is B.READY and result.processed_profiles == result.index_entries == emitted == 2101
    assert [(e.phase, e.processed) for e in events] == [("customers", 1000), ("customers", 2000), ("complete", 2101)]
    assert search(source, "email", "user2100@example.test").canonical_customer_id == "C2100"


def test_cancel_during_merge_loading_does_not_publish_partial_graph(source):
    write_merges(source[0], [("A", "B"), ("B", "C")])
    cancel = Event()
    result = build(source, cancelled=cancel.is_set, progress=lambda event: cancel.set(), progress_every=1)
    assert result.status is B.CANCELLED
    assert not source[1].index_path(raw_root=source[0]).exists()


def test_build_refuses_cache_inside_canonical_source(source):
    service = CustomerContactSearchService(cache_dir=source[0] / "canonical/cache")
    assert service.build_index(raw_root=source[0]).status is B.SOURCE_INCONSISTENT
    assert not service.cache_dir.exists()


def test_busy_canonical_writer_never_reuses_contact_identity(source):
    from Application.mindbox.canonical_storage import storage_lock
    assert build(source).status is B.READY
    with storage_lock(source[0]):
        assert search(source).status is S.SOURCE_UNAVAILABLE
        assert build(source).status is B.SOURCE_UNAVAILABLE


def test_source_ro_connections_closed_and_orphan_build_is_never_opened(source, monkeypatch):
    assert build(source).status is B.READY
    path = source[1].index_path(raw_root=source[0])
    # Simulates a process dying before publication.
    orphan = path.parent / ".building-interrupted.sqlite"
    orphan.write_bytes(b"incomplete")
    original = sqlite3.connect
    connections = []
    def connect(path, **kwargs):
        assert str(path).endswith("?mode=ro") and kwargs["uri"] is True
        connection = original(path, **kwargs)
        connections.append(connection)
        return connection
    monkeypatch.setattr(module.sqlite3, "connect", connect)
    assert search(source).status is S.MATCHED
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    path.unlink()
    assert search(source).status is S.INDEX_NOT_READY


def test_different_source_roots_never_share_an_index(source, tmp_path):
    assert source[1].index_path(raw_root=source[0]) != source[1].index_path(raw_root=tmp_path / "other")


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_force_build_does_not_publish_beside_active_cache_sidecars(source, suffix):
    assert build(source).status is B.READY
    path = source[1].index_path(raw_root=source[0])
    before = path.read_bytes()
    sidecar = Path(str(path) + suffix)
    sidecar.write_bytes(b"pending")
    assert build(source, force=True).status is B.SOURCE_UNAVAILABLE
    assert search(source).status is S.SOURCE_UNAVAILABLE
    assert path.read_bytes() == before


def test_sqlite_error_classification_also_handles_python310_exceptions():
    assert module._unavailable(sqlite3.OperationalError("database is locked"))
    assert module._unavailable(sqlite3.OperationalError("unable to open database file"))
    assert not module._unavailable(sqlite3.OperationalError("no such table: contacts"))
    assert not module._unavailable(sqlite3.DatabaseError("database disk image is malformed"))
