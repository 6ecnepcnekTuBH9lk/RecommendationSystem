from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from datetime import date
import json
import sqlite3

import pytest

from Application.customer_profile_service import BirthDateStatus, CardStatus, CustomerProfileService, ProfileStatus
from Application.mindbox.selection import DEFAULT_SELECTION


def write_db(path, raw, *, key="C", schema=None):
    connection = sqlite3.connect(path)
    try:
        connection.execute(schema or "CREATE TABLE profiles (id TEXT PRIMARY KEY, changed TEXT, raw TEXT NOT NULL)")
        connection.execute("INSERT INTO profiles VALUES (?, ?, ?)",
                           (key, raw.get("changeDateTimeUtc") or "", json.dumps(raw)))
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "canonical/objects" / ("1" * 32) / "customer_merges"
    directory.mkdir(parents=True)
    edges = [{"id": "synthetic-merge", "dateTimeUtc": "2026-01-01T00:00:00Z",
              "resultingCustomer": {"ids": {"mindboxId": "C"}},
              "mergedCustomers": [{"ids": {"mindboxId": "A"}}]}]
    (directory / "customer_merges_part_001.json").write_text(json.dumps({"customerMerges": edges}), encoding="utf-8")
    (tmp_path / "canonical/.writer.lock").write_bytes(b"0")
    (tmp_path / "canonical/catalog.json").write_text(json.dumps({
        "schema_version": 1, "revision": "synthetic-revision", "actions": {}, "orders": {},
        "customer_merges": {"directory": directory.relative_to(tmp_path).as_posix(), "parts": 1,
                            "since": "2022-01-01T00:00:00+00:00", "until": "2026-09-01T00:00:00+00:00"},
        "selection": asdict(DEFAULT_SELECTION)}), encoding="utf-8")
    raw = {"ids": {"mindboxId": "A"}, "firstName": " Иван ", "lastName": " Иванов ",
           "middleName": "Иванович", "sex": "male", "birthDate": "1990-10-15",
           "email": "synthetic@example.test", "mobilePhone": "+7 (900) 123-45-67",
           "changeDateTimeUtc": "2026-10-01T00:00:00+00:00",
           "discountCards": [{"ids": {"number": "000123"}}],
           "isMobilePhoneConfirmed": True, "isMobilePhoneInvalid": False,
           "isEmailInvalid": False, "customFields": {"occupation": "synthetic"}}
    path = tmp_path / "canonical/customers.sqlite"
    write_db(path, raw)
    return tmp_path, path, raw


def set_raw(source, raw):
    with sqlite3.connect(source[1]) as connection:
        connection.execute("UPDATE profiles SET changed=?, raw=? WHERE id=?",
                           (raw.get("changeDateTimeUtc") or "", json.dumps(raw), "C"))


def get(source, *, service=None, customer="C", as_of=date(2026, 10, 13)):
    return (service or CustomerProfileService()).get_profile(customer, raw_root=source[0], as_of=as_of)


def test_indexed_read_only_profile_and_no_pii_in_repr(source, monkeypatch):
    from Application import customer_profile_service as module
    before = source[1].read_bytes()
    files = set(source[0].rglob("*"))
    real_connect = sqlite3.connect
    statements, connections = [], []
    def connect(path_uri, **kwargs):
        assert path_uri.endswith("?mode=ro") and kwargs["uri"] is True
        connection = real_connect(path_uri, **kwargs)
        connection.set_trace_callback(statements.append)
        connections.append(connection)
        return connection
    monkeypatch.setattr(module.sqlite3, "connect", connect)
    result = get(source)
    assert result.status is ProfileStatus.FOUND
    assert result.profile.full_name == "Иванов Иван Иванович"
    assert result.profile.contact.mobile_phone == "+7 (900) 123-45-67"
    assert result.profile.contact.email == "synthetic@example.test"
    assert result.profile.age == 35 and result.profile.age_group == "26–35 лет"
    assert result.lookup_ms is not None
    selects = [s for s in statements if s.upper().startswith("SELECT")]
    assert len(selects) == 1 and "WHERE id=" in selects[0] and "LIMIT 1" in selects[0]
    assert "PRAGMA query_only=ON" in statements
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")
    assert source[1].read_bytes() == before and set(source[0].rglob("*")) == files
    assert "Иванов" not in repr(result) + repr(result.profile)
    assert "synthetic@example.test" not in repr(result.profile)
    with pytest.raises(FrozenInstanceError):
        result.profile.age = 0
    with pytest.raises(TypeError):
        result.profile.record.custom_fields["occupation"] = "changed"


def test_primary_key_query_plan(source):
    with sqlite3.connect(source[1]) as connection:
        plan = connection.execute("EXPLAIN QUERY PLAN SELECT id, changed, raw FROM profiles WHERE id=? LIMIT 1", ("C",)).fetchall()
    assert any("SEARCH profiles USING INDEX" in row[3] for row in plan)


@pytest.mark.parametrize("customer", ["C", "A"])
def test_canonical_and_alias_are_confirmed_independently_of_bpr(source, customer):
    assert not (source[0] / "model").exists()
    result = get(source, customer=customer)
    assert result.status is ProfileStatus.FOUND and result.canonical_customer_id == "C"


def test_chain_and_unknown_do_not_guess_profiles(source):
    path = next((source[0] / "canonical/objects").rglob("customer_merges_part_001.json"))
    rows = json.loads(path.read_text())["customerMerges"]
    rows.append({"id": "another", "dateTimeUtc": "2026-01-01T00:00:00Z",
                 "resultingCustomer": {"ids": {"mindboxId": "A"}},
                 "mergedCustomers": [{"ids": {"mindboxId": "B"}}]})
    path.write_text(json.dumps({"customerMerges": rows}), encoding="utf-8")
    assert get(source, customer="B").canonical_customer_id == "C"
    assert get(source, customer="unknown").status is ProfileStatus.NOT_FOUND
    assert get(source, customer="C' OR 1=1 --").status is ProfileStatus.NOT_FOUND


@pytest.mark.parametrize("value", [None, 12, True, "", " ", " C", "C "])
def test_invalid_ids_never_read_sqlite(source, value, monkeypatch):
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("No SQL for invalid ID"))
    assert get(source, customer=value).status is ProfileStatus.INVALID_ID


@pytest.mark.parametrize("component", ["database", "lock", "catalog", "merges"])
def test_unavailable_sources_are_not_not_found(source, component):
    path = {"database": source[1], "lock": source[0] / "canonical/.writer.lock",
            "catalog": source[0] / "canonical/catalog.json",
            "merges": next((source[0] / "canonical/objects").rglob("customer_merges_part_001.json"))}[component]
    path.unlink()
    assert get(source).status is ProfileStatus.SOURCE_UNAVAILABLE


@pytest.mark.parametrize("mutation", ["database", "json", "row_identity", "changed", "firstName", "email", "flags", "duplicate_json"])
def test_corruption_is_safe_and_does_not_disclose_personal_data(source, mutation, caplog, capsys):
    raw = deepcopy(source[2])
    if mutation == "database":
        source[1].write_bytes(b"corrupt database PRIVATE_ID")
    elif mutation in ("json", "changed", "duplicate_json"):
        with sqlite3.connect(source[1]) as connection:
            if mutation == "changed":
                connection.execute("UPDATE profiles SET changed='2025-01-01T00:00:00Z'")
            else:
                text = "PRIVATE_ID broken" if mutation == "json" else '{"ids":{},"ids":{}}'
                connection.execute("UPDATE profiles SET raw=?", (text,))
    else:
        if mutation == "row_identity":
            raw["ids"]["mindboxId"] = "OTHER_PRIVATE_ID"
        elif mutation == "flags":
            raw["isEmailInvalid"] = "true"
        else:
            raw[mutation] = {"PRIVATE_ID": 12}
        set_raw(source, raw)
    result = get(source)
    assert result.status is ProfileStatus.SOURCE_INCONSISTENT
    assert result.profile is None
    assert "PRIVATE_ID" not in repr(result) + caplog.text + str(capsys.readouterr())


@pytest.mark.parametrize("parts,expected", [(("Имя", "Фамилия", "Отчество"), "Фамилия Имя Отчество"),
    ((None, "Фамилия", None), "Фамилия"), (("Имя", None, None), "Имя"),
    (("null", "None", "nan"), None), ((" ", None, None), None)])
def test_partial_name_and_missing_contacts(source, parts, expected):
    raw = deepcopy(source[2])
    raw.update(firstName=parts[0], lastName=parts[1], middleName=parts[2], email=None, mobilePhone=None)
    set_raw(source, raw)
    profile = get(source).profile
    assert profile.full_name == expected
    assert profile.contact.email is profile.contact.mobile_phone is None


@pytest.mark.parametrize("sex,expected", [("male", "Мужской"), ("female", "Женский"), (None, None), ("unknown", None)])
def test_supported_and_unknown_sex(source, sex, expected):
    raw = deepcopy(source[2]); raw["sex"] = sex; set_raw(source, raw)
    assert get(source).profile.sex_label == expected


@pytest.mark.parametrize("born,as_of,age,status", [
    ("1990-10-15", date(2026, 10, 13), 35, BirthDateStatus.VALID),
    ("1990-10-13", date(2026, 10, 13), 36, BirthDateStatus.VALID),
    ("2000-02-29", date(2025, 2, 28), 24, BirthDateStatus.VALID),
    ("2000-02-29", date(2025, 3, 1), 25, BirthDateStatus.VALID),
    ("2000-02-29", date(2024, 2, 29), 24, BirthDateStatus.VALID),
    (None, date(2026, 10, 13), None, BirthDateStatus.MISSING),
    ("2027-01-01", date(2026, 10, 13), None, BirthDateStatus.FUTURE),
    ("2025-02-29", date(2026, 10, 13), None, BirthDateStatus.INVALID),
    (12, date(2026, 10, 13), None, BirthDateStatus.INVALID),
    ("2026-10-13", date(2026, 10, 13), 0, BirthDateStatus.VALID),
])
def test_calendar_age_missing_invalid_future_and_leap_day(source, born, as_of, age, status):
    raw = deepcopy(source[2]); raw["birthDate"] = born; set_raw(source, raw)
    profile = get(source, as_of=as_of).profile
    assert profile.as_of == as_of and profile.age == age and profile.birth_date_status is status
    if age is None:
        assert profile.age_group is None


@pytest.mark.parametrize("cards,activated,number,status", [
    (["000123"], None, "000123", CardStatus.SINGLE),
    (["000123", "000456"], "000456", "000456", CardStatus.ACTIVATED),
    (["000123", "000456"], None, None, CardStatus.AMBIGUOUS),
    ([], None, None, CardStatus.MISSING),
    ([True], None, None, CardStatus.INVALID),
])
def test_card_selection_uses_existing_contract(source, cards, activated, number, status):
    raw = deepcopy(source[2])
    raw["discountCards"] = [{"ids": {"number": n}} for n in cards]
    raw["lastActivatedCard"] = None if activated is None else {"ids": {"number": activated}}
    set_raw(source, raw)
    profile = get(source).profile
    assert profile.contact.discount_card == number and profile.card_status is status


def test_database_replacement_and_no_personal_data_cache(source, monkeypatch):
    from Application import customer_profile_service as module
    service = CustomerProfileService()
    old = get(source, service=service)
    updated = deepcopy(source[2]); updated["firstName"] = "Новое имя"
    replacement = source[1].with_name("replacement.sqlite")
    write_db(replacement, updated)
    replacement.replace(source[1])
    assert not service.is_current(old, raw_root=source[0])
    assert get(source, service=service).profile.record.first_name == "Новое имя"
    assert service._identity is not None and not hasattr(service, "_profile_cache")
    # Replacement during a read is detected and retried from the new database.
    updated["firstName"] = "Замена во время запроса"
    write_db(replacement, updated)
    real_lookup = module._lookup
    calls = []
    def lookup(path, key):
        row = real_lookup(path, key); calls.append(key)
        if len(calls) == 1:
            replacement.replace(path)
        return row
    monkeypatch.setattr(module, "_lookup", lookup)
    assert get(source, service=service).profile.record.first_name == "Замена во время запроса"
    assert calls == ["C", "C"]


def test_registration_is_not_derived_and_source_is_immutable(source):
    before = deepcopy(source[2]); profile = get(source).profile
    assert not hasattr(profile, "registration_date")
    assert profile.record.change_datetime_utc is not None
    assert source[2] == before


@pytest.mark.parametrize("kind", ["journal", "wal"])
def test_no_recovery_or_wal_shared_memory_files_are_created(source, kind, monkeypatch):
    if kind == "wal":
        with sqlite3.connect(source[1]) as db:
            db.execute("PRAGMA journal_mode=WAL")
        expected = ProfileStatus.SOURCE_INCONSISTENT
    else:
        source[1].with_name("customers.sqlite-journal").write_bytes(b"pending recovery")
        expected = ProfileStatus.SOURCE_UNAVAILABLE
    before = set(source[0].rglob("*"))
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: pytest.fail("No SQLite recovery"))
    assert get(source).status is expected
    assert set(source[0].rglob("*")) == before


def test_cancellation_and_long_fields(source):
    service = CustomerProfileService()
    assert service.get_profile("C", raw_root=source[0], cancelled=lambda: True).status is ProfileStatus.CANCELLED
    raw = deepcopy(source[2]); raw["firstName"] = "ДлинноеИмя" * 200; set_raw(source, raw)
    assert get(source).profile.record.first_name == raw["firstName"]


@pytest.mark.parametrize("edges", [[("A", "C"), ("A", "D")], [("A", "B"), ("B", "A")]])
def test_conflicting_or_cyclic_graph_cannot_confirm_profile(source, edges):
    path = next((source[0] / "canonical/objects").rglob("customer_merges_part_001.json"))
    rows = [{"id": f"synthetic-{i}", "dateTimeUtc": "2026-01-01T00:00:00Z",
             "resultingCustomer": {"ids": {"mindboxId": target}},
             "mergedCustomers": [{"ids": {"mindboxId": original}}]} for i, (original, target) in enumerate(edges)]
    path.write_text(json.dumps({"customerMerges": rows}), encoding="utf-8")
    assert get(source).status is ProfileStatus.SOURCE_INCONSISTENT


def test_unindexed_schema_and_unconfirmed_merge_window_are_rejected(source):
    source[1].unlink()
    write_db(source[1], source[2], schema="CREATE TABLE profiles(id TEXT, changed TEXT, raw TEXT)")
    assert get(source).status is ProfileStatus.SOURCE_INCONSISTENT
    path = source[0] / "canonical/catalog.json"
    metadata = json.loads(path.read_text()); metadata["customer_merges"]["parts"] = True
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert get(source).status is ProfileStatus.SOURCE_INCONSISTENT


def test_existing_writer_lock_content_is_unchanged_and_busy_writer_is_safe(source):
    from Application.mindbox.canonical_storage import storage_lock
    lock = source[0] / "canonical/.writer.lock"
    before = lock.read_bytes()
    with storage_lock(source[0]):
        assert get(source).status is ProfileStatus.SOURCE_UNAVAILABLE
    assert lock.read_bytes() == before
    assert get(source).status is ProfileStatus.FOUND
