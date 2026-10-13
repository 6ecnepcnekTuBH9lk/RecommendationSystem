"""Indexed current Customers lookup. Never loads BPR, profiles in bulk or Qt.

Canonical publishers use rollback journals and the existing .writer.lock. We
open that file read-only and hold its read lock only for the SQLite transaction.
No persistent connection or PII cache survives a call. as_of controls age only;
this API does not reconstruct past profile snapshots for temporal evaluation.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from threading import Lock
from time import perf_counter

from Application.customer_demographics import age_group, calendar_age
from Application.customer_profiles import CustomerContact, contact_candidate_from_record
from Application.mindbox.adapters import adapt_customer, adapt_customer_merge
from Application.mindbox.adapters._common import AdapterError, birth_date, boolean, timestamp
from Application.mindbox.canonical_customers import database
from Application.mindbox.canonical_storage import catalog, checked_directory
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, RawExportError, _unique_object, iter_export, part_files
from Application.mindbox.records import CustomerRecord


class ProfileStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    INVALID_ID = "INVALID_ID"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    SOURCE_INCONSISTENT = "SOURCE_INCONSISTENT"
    STALE = "STALE"
    CANCELLED = "CANCELLED"


class BirthDateStatus(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    INVALID = "INVALID"
    FUTURE = "FUTURE"


class CardStatus(str, Enum):
    ACTIVATED = "ACTIVATED"
    SINGLE = "SINGLE"
    AMBIGUOUS = "AMBIGUOUS"
    MISSING = "MISSING"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True, repr=False)
class CustomerProfile:
    record: CustomerRecord
    contact: CustomerContact
    as_of: date
    age: int | None
    age_group: str | None
    birth_date_status: BirthDateStatus
    card_status: CardStatus
    email_invalid: bool | None = None
    phone_invalid: bool | None = None
    phone_confirmed: bool | None = None

    @property
    def canonical_customer_id(self):
        return self.record.customer_id

    @property
    def full_name(self):
        return " ".join(part for part in (self.record.last_name, self.record.first_name, self.record.middle_name) if part) or None

    @property
    def sex_label(self):
        return {"male": "Мужской", "female": "Женский"}.get(self.record.sex)


@dataclass(frozen=True, slots=True, repr=False)
class CustomerProfileResult:
    status: ProfileStatus
    profile: CustomerProfile | None = None
    canonical_customer_id: str | None = None
    source_version: str | None = field(default=None, repr=False)
    canonical_revision: str | None = None
    lookup_ms: float | None = None

    def __post_init__(self):
        if not isinstance(self.status, ProfileStatus):
            raise TypeError("Invalid profile status")
        if self.status is ProfileStatus.FOUND:
            if not isinstance(self.profile, CustomerProfile) or self.profile.canonical_customer_id != self.canonical_customer_id:
                raise ValueError("Invalid confirmed profile")
        elif self.profile is not None:
            raise ValueError("An unavailable profile cannot contain personal data")


def _valid_id(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def _clean(value):
    value = value.strip() if value is not None else None
    return None if not value or value.casefold() in ("null", "none", "nan", "<na>") else value


def _reject_number(value):
    raise ValueError("Invalid profile JSON number")


def _version(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _identity_source(root):
    metadata_path = root / "canonical/catalog.json"
    before = _version(metadata_path)
    with metadata_path.open("rb"):
        pass
    data = catalog(root)
    if type(data["schema_version"]) is not int or not _valid_id(data["revision"]):
        raise ValueError("Invalid identity source metadata")
    entry = data["customer_merges"]
    if entry is None:
        raise FileNotFoundError("No confirmed CustomerMerges")
    if type(entry["parts"]) is not int or entry["parts"] < 1:
        raise ValueError("Invalid merge part count")
    since, until = (datetime.fromisoformat(entry[k]) for k in ("since", "until"))
    if since >= until or any(d.tzinfo is None or d.utcoffset() != timedelta(0) for d in (since, until)):
        raise ValueError("Invalid merge coverage")
    directory = checked_directory(root, entry["directory"], "customer_merges")
    if not tuple(directory.iterdir()):
        raise FileNotFoundError("Missing merge parts")
    parts = part_files(directory, "customer_merges")
    if len(parts) != entry["parts"]:
        raise ValueError("Merge part count mismatch")
    key = root, before, data["revision"], json.dumps(entry, sort_keys=True), tuple((p, _version(p)) for p in parts)
    if _version(metadata_path) != before:
        raise ValueError("Identity metadata changed")
    return key, directory, data["revision"]


@contextmanager
def _reader_lock(root):
    # Coordinate with the existing publisher, without creating/writing its file.
    with (root / "canonical/.writer.lock").open("rb") as stream:
        if stream.read(1) == b"":
            raise OSError("Coordination source unavailable")
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBRLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _database_version(path):
    # Fail closed before SQLite could open/create any WAL shared-memory files.
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) < 20 or header[:16] != b"SQLite format 3\x00" or header[18:20] != b"\x01\x01":
        raise ValueError("Unsupported or corrupt Customers database")
    for suffix in ("-journal", "-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise OSError("Customers update or recovery pending")
    return _version(path)


def _lookup(path, customer_id):
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.5)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        schema = connection.execute("PRAGMA table_info(profiles)").fetchall()
        fields = {row[1]: row for row in schema}
        if (not {"id", "changed", "raw"}.issubset(fields) or fields["id"][2].upper() != "TEXT"
                or fields["id"][5] != 1 or sum(bool(row[5]) for row in schema) != 1):
            raise ValueError("Customers requires indexed canonical IDs")
        start = perf_counter()
        row = connection.execute("SELECT id, changed, raw FROM profiles WHERE id=? LIMIT 1", (customer_id,)).fetchone()
        return row, (perf_counter() - start) * 1000
    finally:
        connection.close()


def _adapt(raw, changed, canonical, resolver, as_of):
    if not isinstance(raw, dict):
        raise ValueError("Invalid profile object")
    birthday_status = BirthDateStatus.MISSING
    try:
        born = birth_date(raw)
        if born is not None:
            birthday_status = BirthDateStatus.FUTURE if born > as_of else BirthDateStatus.VALID
    except AdapterError:
        born = None
        birthday_status = BirthDateStatus.INVALID
    normalized = dict(raw)
    if birthday_status is BirthDateStatus.INVALID:
        normalized["birthDate"] = None
    record = adapt_customer(normalized, resolver)
    if record.customer_id != canonical or resolver.resolve(canonical) != canonical:
        raise ValueError("Profile identity differs from its indexed canonical key")
    if changed is not None and not isinstance(changed, str):
        raise ValueError("Invalid profile change metadata")
    changed_at = timestamp({"changeDateTimeUtc": changed or None}, "changeDateTimeUtc", required=False)
    if changed_at != record.change_datetime_utc:
        raise ValueError("Profile change metadata differs from its raw record")
    record = replace(record, first_name=_clean(record.first_name), last_name=_clean(record.last_name),
                     middle_name=_clean(record.middle_name), email=_clean(record.email), mobile_phone=_clean(record.mobile_phone))
    candidate = contact_candidate_from_record(record)
    activated, single, ambiguous, _, invalid = candidate.card_counts
    card = _clean(candidate.discount_card)
    card_status = (CardStatus.ACTIVATED if activated and card else CardStatus.SINGLE if single and card
                   else CardStatus.AMBIGUOUS if ambiguous else CardStatus.INVALID if invalid or candidate.discount_card
                   else CardStatus.MISSING)
    age = calendar_age(born, as_of)
    return CustomerProfile(record, CustomerContact(record.email, record.mobile_phone, card), as_of, age,
                           age_group(age), birthday_status, card_status,
                           boolean(raw, "isEmailInvalid"), boolean(raw, "isMobilePhoneInvalid"),
                           boolean(raw, "isMobilePhoneConfirmed"))


class CustomerProfileService:
    """Cache only the identity graph; each request uses a new short RO connection."""
    def __init__(self):
        self._lock = Lock()
        self._identity = None

    def get_profile(self, customer_id: str, *, raw_root=DEFAULT_RAW_ROOT, as_of=None,
                    cancelled=lambda: False) -> CustomerProfileResult:
        if not _valid_id(customer_id):
            return CustomerProfileResult(ProfileStatus.INVALID_ID)
        as_of = date.today() if as_of is None else as_of
        if type(as_of) is not date:
            raise TypeError("as_of must be a calendar date")
        with self._lock:
            try:
                root = Path(raw_root).resolve()
                for _ in range(2):
                    if cancelled():
                        return CustomerProfileResult(ProfileStatus.CANCELLED)
                    key, directory, revision = _identity_source(root)
                    if self._identity is None or self._identity[0] != key:
                        self._identity = None
                        resolver = CustomerIdResolver(adapt_customer_merge(row) for row in
                                                      iter_export("customer_merges", input_dir=directory))
                        if _identity_source(root)[0] != key:
                            continue
                        self._identity = key, resolver
                    resolver = self._identity[1]
                    canonical = resolver.resolve(customer_id)
                    path = database(root)
                    with _reader_lock(root):
                        before = _database_version(path)
                        if _identity_source(root)[0] != key:
                            continue
                        row, elapsed = _lookup(path, canonical)
                        if _database_version(path) != before or _identity_source(root)[0] != key:
                            continue
                    if cancelled():
                        return CustomerProfileResult(ProfileStatus.CANCELLED)
                    version = hashlib.sha256(repr((key, before)).encode()).hexdigest()
                    if row is None:
                        return CustomerProfileResult(ProfileStatus.NOT_FOUND, canonical_customer_id=canonical,
                                                     source_version=version, canonical_revision=revision, lookup_ms=elapsed)
                    if row[0] != canonical:
                        raise ValueError("Invalid indexed identity")
                    raw = json.loads(row[2], object_pairs_hook=_unique_object, parse_float=Decimal,
                                     parse_constant=_reject_number)
                    profile = _adapt(raw, row[1], canonical, resolver, as_of)
                    result = CustomerProfileResult(ProfileStatus.FOUND, profile, canonical, version, revision, elapsed)
                    if self.is_current(result, raw_root=root):
                        return result
                return CustomerProfileResult(ProfileStatus.STALE)
            except Exception as exc:
                self._identity = None
                unavailable = isinstance(exc, (OSError, sqlite3.OperationalError)) or (
                    isinstance(exc, RawExportError) and exc.category in ("file_missing", "file_access", "file_read"))
                return CustomerProfileResult(ProfileStatus.SOURCE_UNAVAILABLE if unavailable else ProfileStatus.SOURCE_INCONSISTENT)

    def is_current(self, result, *, raw_root=DEFAULT_RAW_ROOT):
        if result.source_version is None:
            return False
        try:
            root = Path(raw_root).resolve()
            with _reader_lock(root):
                key = _identity_source(root)[0]
                version = hashlib.sha256(repr((key, _database_version(database(root)))).encode()).hexdigest()
                return version == result.source_version
        except Exception:
            return False


_DEFAULT_SERVICE = CustomerProfileService()


def get_customer_profile(customer_id: str, *, raw_root=DEFAULT_RAW_ROOT, as_of=None) -> CustomerProfileResult:
    return _DEFAULT_SERVICE.get_profile(customer_id, raw_root=raw_root, as_of=as_of)
