"""Explicit, streaming contact-index builds and exact read-only lookups, without Qt.

The derived database lives under ignored user_settings/cache. A random per-index
HMAC key avoids storing clear contacts, but is stored in that same database: this
is data minimization, not encryption against someone who can read the cache.
Source readers share the canonical publisher's lock; a build holds it throughout
its snapshot. Interrupted temporary builds never replace the published index.

Phone v1 accepts explicit +/00 international notation and the project's complete
11-digit 7-prefixed notation. Unprefixed 10-digit and 8-prefixed national numbers
are rejected instead of guessing a country. No suffix matching is performed.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from threading import Lock
from time import perf_counter


logger = logging.getLogger(__name__)

from Application.customer_profile_service import _database_version, _reader_lock, _valid_id
from Application.customer_profiles import _card_number
from Application.mindbox.adapters import adapt_customer_merge
from Application.mindbox.adapters._common import AdapterError, identifier, text
from Application.mindbox.canonical_customers import database
from Application.mindbox.canonical_storage import checked_directory
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, RawExportError, _unique_object, iter_export, part_files
from Application.paths import USER_SETTINGS_DIR


INDEX_SCHEMA_VERSION = 1
NORMALIZATION_VERSION = 1
DEFAULT_CACHE_DIR = USER_SETTINGS_DIR / "cache" / "customer_contact_search"
_PLACEHOLDERS = {"null", "none", "nan", "<na>"}
_EMAIL_LOCAL = re.compile(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+")
_DOMAIN_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")


class ContactType(str, Enum):
    PHONE = "phone"
    EMAIL = "email"
    DISCOUNT_CARD = "discount_card"


class ContactSearchStatus(str, Enum):
    MATCHED = "MATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    INDEX_NOT_READY = "INDEX_NOT_READY"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    SOURCE_INCONSISTENT = "SOURCE_INCONSISTENT"


class IndexBuildStatus(str, Enum):
    READY = "READY"
    CANCELLED = "CANCELLED"
    INDEX_NOT_READY = "INDEX_NOT_READY"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    SOURCE_INCONSISTENT = "SOURCE_INCONSISTENT"


@dataclass(frozen=True, slots=True)
class ContactSearchResult:
    status: ContactSearchStatus
    canonical_customer_id: str | None = field(default=None, repr=False)
    candidate_count: int = 0
    source_version: str | None = None
    lookup_ms: float | None = None

    def __post_init__(self):
        if not isinstance(self.status, ContactSearchStatus) or type(self.candidate_count) is not int:
            raise TypeError("Invalid contact search result")
        if self.status is ContactSearchStatus.MATCHED:
            if self.candidate_count != 1 or not _valid_id(self.canonical_customer_id):
                raise ValueError("A match requires exactly one canonical ID")
        elif self.canonical_customer_id is not None:
            raise ValueError("Only an unambiguous match may select a customer")
        elif self.status is ContactSearchStatus.AMBIGUOUS:
            if self.candidate_count < 2:
                raise ValueError("Ambiguity requires multiple canonical customers")
        elif self.candidate_count != 0:
            raise ValueError("An unavailable match cannot contain candidates")


@dataclass(frozen=True, slots=True)
class ContactIndexProgress:
    phase: str
    processed: int


@dataclass(frozen=True, slots=True)
class ContactIndexBuildResult:
    status: IndexBuildStatus
    processed_profiles: int = 0
    index_entries: int = 0
    reused: bool = False
    source_version: str | None = None
    elapsed_ms: float | None = None


def _kind(value):
    if isinstance(value, ContactType):
        return value
    if isinstance(value, str):
        try:
            return ContactType(value.lower())
        except ValueError:
            pass
    return None


def _normalize(kind, value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 254 or value.casefold() in _PLACEHOLDERS:
        return None
    if kind is ContactType.EMAIL:
        if value.count("@") != 1:
            return None
        local, domain = value.split("@")
        if (not _EMAIL_LOCAL.fullmatch(local) or len(local) > 64 or local.startswith(".")
                or local.endswith(".") or ".." in local or "." not in domain
                or any(not _DOMAIN_LABEL.fullmatch(label) for label in domain.split("."))):
            return None
        return value.casefold()
    if kind is ContactType.PHONE:
        # Only harmless formatting is discarded, never arbitrary letters/exts.
        if not re.fullmatch(r"\+?[0-9 ()\-]+", value):
            return None
        number = re.sub(r"[ ()\-]", "", value)
        if number.startswith("00"):
            number = "+" + number[2:]
        if not number.startswith("+"):
            if len(number) != 11 or not number.startswith("7"):
                return None
            number = "+" + number
        digits = number[1:]
        if not re.fullmatch(r"[1-9][0-9]{7,14}", digits):
            return None
        if digits[0] in ("1", "7") and len(digits) != 11:
            return None
        return number
    if kind is ContactType.DISCOUNT_CARD:
        # Existing card numbers need not be numeric. Keep every significant char.
        return value if all(ch.isprintable() and not ch.isspace() for ch in value) else None
    return None


def normalize_search_input(contact_type: ContactType | str, value: str) -> str | None:
    """Queries must be strings, particularly to preserve card leading zeroes."""
    return _normalize(_kind(contact_type), value)


def normalize_indexed_contact(contact_type: ContactType | str, value: object) -> str | None:
    """Source identifiers also support nonnegative built-in ints, never floats/bools."""
    kind = _kind(contact_type)
    try:
        if kind is ContactType.DISCOUNT_CARD:
            value, _ = _card_number({"ids": {"number": value}})
        elif kind is ContactType.PHONE:
            value = identifier({"value": value.strip() if isinstance(value, str) else value}, "value", required=False)
        elif kind is ContactType.EMAIL:
            value = text({"value": value}, "value")
        return _normalize(kind, value)
    except AdapterError:
        return None


class _Cancelled(Exception):
    pass


class _IndexNotReady(Exception):
    pass


def _check_cancel(cancelled):
    if cancelled():
        raise _Cancelled


def _reject_constant(value):
    raise ValueError("Invalid source JSON constant")


@dataclass(frozen=True, slots=True, repr=False)
class _Sources:
    version: str
    merges_directory: Path


def _sources(root):
    """Fingerprint only Customers and merges, never global catalog revision."""
    with (root / "canonical/catalog.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] not in (1, 2):
        raise ValueError("Unsupported canonical catalog")
    entry = metadata["customer_merges"]
    if entry is None:
        raise FileNotFoundError("CustomerMerges unavailable")
    if type(entry["parts"]) is not int or entry["parts"] < 1:
        raise ValueError("Invalid merge parts")
    since, until = (datetime.fromisoformat(entry[key]) for key in ("since", "until"))
    if since >= until or any(d.tzinfo is None or d.utcoffset() != timedelta(0) for d in (since, until)):
        raise ValueError("Invalid merge coverage")
    directory = checked_directory(root, entry["directory"], "customer_merges")
    # Preserve I/O failure classification before the export reader wraps errors.
    if not tuple(directory.iterdir()):
        raise FileNotFoundError("Merge parts unavailable")
    parts = part_files(directory, "customer_merges")
    if len(parts) != entry["parts"]:
        raise ValueError("Invalid merge part count")
    def stat(path):
        s = path.stat()
        return s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    key = (INDEX_SCHEMA_VERSION, NORMALIZATION_VERSION, str(root), _database_version(database(root)),
           entry["directory"], entry["parts"],
           entry["since"], entry["until"], tuple((p.name, stat(p)) for p in parts))
    return _Sources(hashlib.sha256(repr(key).encode()).hexdigest(), directory)


@contextmanager
def _read_database(path):
    logger.debug("contact_database stage=header")
    _database_version(path)  # No WAL/shm creation or hot-journal recovery.
    logger.debug("contact_database stage=sqlite_open")
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.5)
    except sqlite3.Error as exc:
        logger.warning("contact_database stage=sqlite_open error=%s", type(exc).__name__)
        raise
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()
        logger.debug("contact_database stage=sqlite_closed")


def _metadata(connection, sources):
    values = dict(connection.execute("SELECT key, value FROM metadata"))
    if values.get("schema_version") != INDEX_SCHEMA_VERSION or values.get("normalization_version") != NORMALIZATION_VERSION:
        raise _IndexNotReady
    if values.get("source_version") != sources.version:
        raise _IndexNotReady
    if not isinstance(values.get("hmac_key"), bytes) or len(values["hmac_key"]) != 32:
        raise ValueError("Invalid search metadata")
    for name in ("processed_profiles", "index_entries"):
        if type(values.get(name)) is not int or values[name] < 0:
            raise ValueError("Invalid search counts")
    return values


def _contact_key(key, kind, normalized):
    return hmac.digest(key, (kind.value + "\0" + normalized).encode("utf-8"), "sha256")


def _contacts(raw):
    for kind, name, flag in ((ContactType.EMAIL, "email", "isEmailInvalid"),
                             (ContactType.PHONE, "mobilePhone", "isMobilePhoneInvalid")):
        validity = raw.get(flag)
        if validity is not None and validity is not False:
            continue  # Invalid or malformed flags never confirm a contact.
        value = normalize_indexed_contact(kind, raw.get(name))
        if value is not None:
            yield kind, value
    cards = raw.get("discountCards")
    if isinstance(cards, list):
        for card in cards:
            value, _ = _card_number(card)
            value = normalize_indexed_contact(ContactType.DISCOUNT_CARD, value)
            if value is not None:
                yield ContactType.DISCOUNT_CARD, value
    value, _ = _card_number(raw.get("lastActivatedCard"))
    value = normalize_indexed_contact(ContactType.DISCOUNT_CARD, value)
    if value is not None:
        yield ContactType.DISCOUNT_CARD, value


def _profile_rows(connection):
    fields = connection.execute("PRAGMA table_info(profiles)").fetchall()
    by_name = {row[1]: row for row in fields}
    if (not {"id", "raw"}.issubset(by_name) or by_name["id"][2].upper() != "TEXT"
            or by_name["id"][5] != 1 or sum(bool(row[5]) for row in fields) != 1):
        raise ValueError("Customers requires an indexed canonical ID")
    yield from connection.execute("SELECT id, raw FROM profiles")


def _unavailable(exc):
    code = getattr(exc, "sqlite_errorcode", None)
    # Python 3.10 has neither sqlite_errorcode nor sqlite3.SQLITE_* constants.
    if code is None and isinstance(exc, sqlite3.OperationalError):
        return str(exc).startswith(("database is locked", "database table is locked", "unable to open database file",
                                    "attempt to write a readonly database", "disk I/O error"))
    return isinstance(exc, OSError) or (code is not None and code & 255 in (5, 6, 8, 10, 14)) or (
        isinstance(exc, RawExportError) and exc.category in ("file_missing", "file_access", "file_read"))


class CustomerContactSearchService:
    """No implicit rebuilds, profile cache, model dependency or persistent connection."""

    def __init__(self, *, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir).resolve()
        self._build_lock = Lock()

    def index_path(self, *, raw_root: str | Path = DEFAULT_RAW_ROOT) -> Path:
        name = hashlib.sha256(str(Path(raw_root).resolve()).encode()).hexdigest()
        return self.cache_dir / (name + ".sqlite")

    def search(self, contact_type: ContactType | str, value: str, *,
               raw_root: str | Path = DEFAULT_RAW_ROOT) -> ContactSearchResult:
        kind = _kind(contact_type)
        normalized = normalize_search_input(kind, value)
        if normalized is None:
            return ContactSearchResult(ContactSearchStatus.INVALID_INPUT)
        stage = "lock"
        try:
            root = Path(raw_root).resolve()
            with _reader_lock(root):
                stage = "sources"
                sources = _sources(root)
                path = self.index_path(raw_root=root)
                if not path.exists():
                    raise _IndexNotReady
                start = perf_counter()
                stage = "sqlite_open"
                with _read_database(path) as connection:
                    stage = "metadata"
                    metadata = _metadata(connection, sources)
                    stage = "lookup"
                    count, customer = connection.execute(
                        "SELECT COUNT(*), MIN(customer_id) FROM contacts WHERE kind=? AND contact_key=?",
                        (kind.value, _contact_key(metadata["hmac_key"], kind, normalized)),
                    ).fetchone()
                stage = "source_version"
                if _sources(root).version != sources.version:
                    raise _IndexNotReady
                if count == 1 and not _valid_id(customer):
                    raise ValueError("Invalid indexed identity")
                status = (ContactSearchStatus.MATCHED if count == 1 else ContactSearchStatus.AMBIGUOUS
                          if count > 1 else ContactSearchStatus.NOT_FOUND)
                logger.debug("contact_search stage=complete status=%s", status.value)
                return ContactSearchResult(status, customer if count == 1 else None, count,
                                           sources.version, (perf_counter() - start) * 1000)
        except _IndexNotReady:
            logger.debug("contact_search stage=%s status=INDEX_NOT_READY", stage)
            return ContactSearchResult(ContactSearchStatus.INDEX_NOT_READY)
        except Exception as exc:
            logger.warning("contact_search stage=%s error=%s", stage, type(exc).__name__)
            return ContactSearchResult(ContactSearchStatus.SOURCE_UNAVAILABLE if _unavailable(exc)
                                       else ContactSearchStatus.SOURCE_INCONSISTENT)

    def build_index(self, *, raw_root: str | Path = DEFAULT_RAW_ROOT, force: bool = False,
                    cancelled: Callable[[], bool] = lambda: False,
                    progress: Callable[[ContactIndexProgress], None] | None = None,
                    progress_every: int = 1000) -> ContactIndexBuildResult:
        """Build explicitly off the GUI thread. Old complete indices survive failures.

        Memory: SQLite cache (8 MiB), <=1000 insert tuples, one raw profile, plus
        the existing merge resolver. No all-profile/contact Python collection.
        Process death can leave .building-* files, which are never queried.
        """
        if type(progress_every) is not int or progress_every < 1 or type(force) is not bool:
            raise ValueError("Invalid index build options")
        start = perf_counter()
        if not self._build_lock.acquire(blocking=False):
            return ContactIndexBuildResult(IndexBuildStatus.INDEX_NOT_READY)
        temporary = None
        try:
            _check_cancel(cancelled)
            root = Path(raw_root).resolve()
            if self.cache_dir.is_relative_to(root):
                raise ValueError("The contact cache must be outside source data")
            with _reader_lock(root):
                sources = _sources(root)
                path = self.index_path(raw_root=root)
                for suffix in ("-journal", "-wal", "-shm"):
                    sidecar = Path(str(path) + suffix)
                    if sidecar.exists() and sidecar.stat().st_size:
                        raise OSError("Contact index recovery or publication pending")
                if path.exists() and not force:
                    try:
                        with _read_database(path) as connection:
                            metadata = _metadata(connection, sources)
                            # Verify the lookup table also exists before reuse.
                            connection.execute("SELECT kind, contact_key, customer_id FROM contacts LIMIT 0")
                        _check_cancel(cancelled)
                        if _sources(root).version != sources.version:
                            raise _IndexNotReady
                        return ContactIndexBuildResult(IndexBuildStatus.READY, metadata["processed_profiles"],
                                                       metadata["index_entries"], True, sources.version,
                                                       (perf_counter() - start) * 1000)
                    except (_IndexNotReady, ValueError, sqlite3.DatabaseError):
                        pass  # An explicit build can replace a stale/corrupt derived cache.
                def merges() -> Iterator:
                    for number, raw in enumerate(iter_export("customer_merges", input_dir=sources.merges_directory), 1):
                        _check_cancel(cancelled)
                        if progress and number % progress_every == 0:
                            progress(ContactIndexProgress("merges", number))
                        yield adapt_customer_merge(raw)
                resolver = CustomerIdResolver(merges())
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix=".building-", suffix=".sqlite", dir=self.cache_dir)
                os.close(fd)
                temporary = Path(name)
                processed, entries = self._build(temporary, root, resolver, sources, cancelled, progress, progress_every)
                _check_cancel(cancelled)
                if _sources(root).version != sources.version:
                    raise _IndexNotReady
                # Windows _commit/fsync requires a writable file descriptor.
                with temporary.open("r+b") as stream:
                    os.fsync(stream.fileno())
                _check_cancel(cancelled)
                os.replace(temporary, path)
                return ContactIndexBuildResult(IndexBuildStatus.READY, processed, entries, False,
                                               sources.version, (perf_counter() - start) * 1000)
        except _Cancelled:
            return ContactIndexBuildResult(IndexBuildStatus.CANCELLED, elapsed_ms=(perf_counter() - start) * 1000)
        except _IndexNotReady:
            return ContactIndexBuildResult(IndexBuildStatus.INDEX_NOT_READY)
        except Exception as exc:
            return ContactIndexBuildResult(IndexBuildStatus.SOURCE_UNAVAILABLE if _unavailable(exc)
                                           else IndexBuildStatus.SOURCE_INCONSISTENT)
        finally:
            try:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        # Unpublished leftovers are never eligible for lookup.
                        pass
            finally:
                self._build_lock.release()

    @staticmethod
    def _build(path, root, resolver, sources, cancelled, progress, progress_every):
        key = os.urandom(32)
        output = sqlite3.connect(path)
        try:
            output.execute("PRAGMA cache_size=-8192")
            output.execute("PRAGMA temp_store=FILE")
            output.execute("PRAGMA synchronous=FULL")
            output.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value NOT NULL);
                CREATE TABLE contacts (kind TEXT NOT NULL, contact_key BLOB NOT NULL, customer_id TEXT NOT NULL,
                    PRIMARY KEY(kind, contact_key, customer_id)) WITHOUT ROWID;
            """)
            processed, buffer = 0, []
            with _read_database(database(root)) as source:
                for indexed_id, encoded in _profile_rows(source):
                    _check_cancel(cancelled)
                    if not _valid_id(indexed_id):
                        raise ValueError("Invalid profile identity")
                    raw = json.loads(encoded, object_pairs_hook=_unique_object, parse_float=Decimal,
                                     parse_constant=_reject_constant)
                    if not isinstance(raw, dict):
                        raise ValueError("Invalid profile object")
                    canonical = resolver.resolve(indexed_id)
                    if resolver.resolve(identifier(raw, "ids.mindboxId")) != canonical:
                        raise ValueError("Indexed identity differs from profile")
                    for kind, normalized in _contacts(raw):
                        _check_cancel(cancelled)
                        buffer.append((kind.value, _contact_key(key, kind, normalized), canonical))
                        if len(buffer) == 1000:
                            output.executemany("INSERT OR IGNORE INTO contacts VALUES (?, ?, ?)", buffer)
                            output.commit()
                            buffer.clear()
                    processed += 1
                    if progress and processed % progress_every == 0:
                        progress(ContactIndexProgress("customers", processed))
            output.executemany("INSERT OR IGNORE INTO contacts VALUES (?, ?, ?)", buffer)
            entries = output.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
            output.executemany("INSERT INTO metadata VALUES (?, ?)", (
                ("schema_version", INDEX_SCHEMA_VERSION), ("normalization_version", NORMALIZATION_VERSION),
                ("source_version", sources.version), ("hmac_key", key),
                ("processed_profiles", processed), ("index_entries", entries),
            ))
            output.commit()
            if progress:
                progress(ContactIndexProgress("complete", processed))
            return processed, entries
        finally:
            output.close()


_DEFAULT_SERVICE = CustomerContactSearchService()


def search_customers_by_contact(contact_type: ContactType | str, value: str, *,
                                raw_root: str | Path = DEFAULT_RAW_ROOT) -> ContactSearchResult:
    return _DEFAULT_SERVICE.search(contact_type, value, raw_root=raw_root)


def build_customer_contact_index(*, raw_root: str | Path = DEFAULT_RAW_ROOT, force: bool = False,
                                 cancelled: Callable[[], bool] = lambda: False,
                                 progress: Callable[[ContactIndexProgress], None] | None = None,
                                 progress_every: int = 1000) -> ContactIndexBuildResult:
    return _DEFAULT_SERVICE.build_index(raw_root=raw_root, force=force, cancelled=cancelled,
                                        progress=progress, progress_every=progress_every)
