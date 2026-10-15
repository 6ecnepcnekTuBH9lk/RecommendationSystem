"""Explicit derived history builds; no customer queries, Qt, models or financial KPI.

Source origins use a zero-based ordinal across a component's ordered part files:
iter_export() does not expose part boundaries, so part_number is deliberately
unknown for an individual event. Source parts themselves are recorded separately.
Unchanged fingerprints deduplicate variants, never provenance. Changed owners,
timestamps and classifications remain on variants; the logical common field is
NULL when variants disagree. No arbitrary/latest variant is selected.

RAM is bounded by batch_size events, one raw record, SQLite caches and the existing
CustomerIdResolver graph (which scales with merge aliases, not event history).
All deduplication state lives on disk. Source locks are read-only; the independent
cache writer lock also serializes builds across instances and processes.
"""

from collections.abc import Callable
from contextlib import ExitStack, closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import Lock
from time import perf_counter
import uuid

from Application import customer_history_normalization as normalization
from Application import customer_history_records as records
from Application.customer_contact_search import _unavailable
from Application.customer_profile_service import _database_version, _reader_lock
from Application.mindbox.adapters import adapt_customer_merge
from Application.mindbox.canonical_storage import catalog, checked_directory, effective_entries
from Application.mindbox.daily_training_batch import _writer_lock
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, iter_export, part_files
from Application.mindbox.selection import MindboxSelectionConfig
from Application.mindbox.training_batch import TrainingBatchError
from Application.paths import USER_SETTINGS_DIR


INDEX_SCHEMA_VERSION = 2
DEFAULT_CACHE_DIR = USER_SETTINGS_DIR / "cache" / "customer_history_index"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class HistoryIndexStatus(str, Enum):
    READY = "READY"
    MISSING = "MISSING"
    STALE = "STALE"
    CORRUPT = "CORRUPT"
    UNAVAILABLE = "UNAVAILABLE"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    SOURCE_INCONSISTENT = "SOURCE_INCONSISTENT"
    BUILD_IN_PROGRESS = "BUILD_IN_PROGRESS"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class HistoryIndexCounts:
    raw_actions: int = 0
    raw_orders: int = 0
    logical_actions: int = 0
    logical_orders: int = 0
    action_variants: int = 0
    order_snapshots: int = 0
    identical_repeats: int = 0
    conflicting_events: int = 0
    owner_conflicts: int = 0
    time_conflicts: int = 0
    kind_conflicts: int = 0

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ValueError("Invalid history index counts")


@dataclass(frozen=True, slots=True)
class HistoryCoverage:
    component: str
    since_utc: datetime
    until_utc: datetime
    source_kind: str


@dataclass(frozen=True, slots=True)
class HistoryIndexResult:
    status: HistoryIndexStatus
    generation: str | None = None
    source_version: str | None = None
    counts: HistoryIndexCounts | None = None
    coverage: tuple[HistoryCoverage, ...] = ()
    reused: bool = False
    elapsed_ms: float | None = None


@dataclass(frozen=True, slots=True)
class HistoryIndexProgress:
    phase: str
    processed: int


@dataclass(frozen=True, slots=True, repr=False)
class _Component:
    name: str
    key: str
    directory: Path
    since: datetime
    until: datetime
    updated: datetime
    source_kind: str
    export_id: str | None
    operation: str
    parts: tuple


@dataclass(frozen=True, slots=True, repr=False)
class _Sources:
    version: str
    selection: MindboxSelectionConfig
    components: tuple[_Component, ...]


class _Cancelled(Exception):
    pass


class _SourceChanged(Exception):
    pass


class _Busy(Exception):
    pass


class _Stale(Exception):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("ascii")).hexdigest()


def _micros(value):
    if value is None:
        return None
    delta = value.astimezone(timezone.utc) - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Invalid source timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError("Source metadata requires UTC")
    return result.astimezone(timezone.utc)


def _stat(path):
    value = path.stat()
    return value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _capture_sources(root):
    # Probe I/O first: catalog() deliberately wraps its OS errors.
    with (root / "canonical/catalog.json").open("rb"):
        pass
    data = catalog(root)
    selection = MindboxSelectionConfig(**data["selection"])
    merge = data["customer_merges"]
    if merge is None:
        raise FileNotFoundError("CustomerMerges unavailable")
    components = []
    for name, entries in (("customer_merges", (merge,)),
                          ("actions", effective_entries(data, "actions")),
                          ("orders", effective_entries(data, "orders"))):
        for entry in entries:
            since, until, updated = (_timestamp(entry[key]) for key in ("since", "until", "updated"))
            if since >= until or type(entry["parts"]) is not int or entry["parts"] < 1:
                raise ValueError("Invalid source coverage/parts")
            if entry["source_kind"] not in ("API", "MANUAL") or not isinstance(entry["operation"], str):
                raise ValueError("Invalid source origin")
            export_id = entry["export_id"]
            if export_id is not None and (not isinstance(export_id, str) or not export_id.strip()):
                raise ValueError("Invalid export metadata")
            directory = checked_directory(root, entry["directory"], name)
            if not tuple(directory.iterdir()):
                raise FileNotFoundError("Source parts unavailable")
            parts = tuple((path, _stat(path)) for path in part_files(directory, name))
            if len(parts) != entry["parts"]:
                raise ValueError("Source parts differ from metadata")
            payload = [name, entry["directory"], since.isoformat(), until.isoformat(), updated.isoformat(),
                       entry["source_kind"], export_id, entry["operation"], [(p.name, stat) for p, stat in parts]]
            components.append(_Component(name, _hash(payload), directory, since, until, updated,
                                         entry["source_kind"], export_id, entry["operation"], parts))
    merges = components[0]
    if any(c.since < merges.since or c.until > merges.until for c in components[1:]):
        raise ValueError("CustomerMerges does not cover consumed sources")
    # Source code hashes catch normalizer changes even if a version bump is missed.
    code_versions = [hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                     for module in (normalization, records)]
    key = [INDEX_SCHEMA_VERSION, records.HISTORY_SCHEMA_VERSION, code_versions, str(root),
           asdict(selection), [c.key for c in components]]
    return _Sources(_hash(key), selection, tuple(components))


_SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value NOT NULL);
CREATE TABLE sources (
    source_key TEXT PRIMARY KEY, component TEXT NOT NULL, directory TEXT NOT NULL,
    since_us INTEGER NOT NULL, until_us INTEGER NOT NULL, observed_us INTEGER NOT NULL,
    source_kind TEXT NOT NULL, export_id TEXT, operation TEXT NOT NULL, record_count INTEGER NOT NULL DEFAULT 0);
CREATE TABLE source_parts (
    source_key TEXT NOT NULL REFERENCES sources, part_number INTEGER NOT NULL,
    filename TEXT NOT NULL, file_version TEXT NOT NULL, PRIMARY KEY(source_key, part_number)) WITHOUT ROWID;
CREATE TABLE events (
    event_key TEXT PRIMARY KEY, source TEXT NOT NULL, source_event_id TEXT NOT NULL,
    canonical_customer_id TEXT, kind TEXT, event_us INTEGER,
    variant_count INTEGER NOT NULL DEFAULT 0, occurrence_count INTEGER NOT NULL DEFAULT 0,
    owner_conflict INTEGER NOT NULL DEFAULT 0, time_conflict INTEGER NOT NULL DEFAULT 0,
    kind_conflict INTEGER NOT NULL DEFAULT 0, UNIQUE(source, source_event_id));
CREATE INDEX events_customer_time ON events(canonical_customer_id, event_us DESC, event_key DESC);
CREATE INDEX events_customer_kind_time ON events(canonical_customer_id, kind, event_us DESC, event_key DESC);
CREATE TABLE variants (
    variant_key TEXT PRIMARY KEY, event_key TEXT NOT NULL REFERENCES events,
    fingerprint BLOB NOT NULL CHECK(length(fingerprint)=32), canonical_customer_id TEXT NOT NULL,
    source_customer_id TEXT NOT NULL, kind TEXT NOT NULL, event_us INTEGER NOT NULL, quality TEXT NOT NULL,
    UNIQUE(event_key, fingerprint));
CREATE INDEX variants_customer_kind_time ON variants(canonical_customer_id, kind, event_us DESC, variant_key DESC);
CREATE TABLE event_customers (
    event_key TEXT NOT NULL REFERENCES events, canonical_customer_id TEXT NOT NULL,
    event_us INTEGER,
    PRIMARY KEY(event_key, canonical_customer_id)) WITHOUT ROWID;
CREATE INDEX event_customers_customer ON event_customers(canonical_customer_id, event_us DESC, event_key DESC);
CREATE TABLE record_origins (
    variant_key TEXT NOT NULL REFERENCES variants, source_key TEXT NOT NULL REFERENCES sources,
    record_ordinal INTEGER NOT NULL, PRIMARY KEY(source_key, record_ordinal)) WITHOUT ROWID;
CREATE INDEX record_origins_variant ON record_origins(variant_key);
CREATE TABLE actions (
    variant_key TEXT PRIMARY KEY REFERENCES variants, action_id TEXT NOT NULL, system_name TEXT NOT NULL,
    creation_us INTEGER, channel_system_name TEXT, channel_external_id TEXT, channel_mindbox_id TEXT, channel_name TEXT,
    action_name TEXT, brand_system_name TEXT, view_price TEXT, view_available INTEGER);
CREATE TABLE orders (event_key TEXT PRIMARY KEY REFERENCES events, order_id TEXT NOT NULL UNIQUE);
CREATE TABLE order_snapshots (
    variant_key TEXT PRIMARY KEY REFERENCES variants, snapshot_key TEXT NOT NULL UNIQUE,
    retail_order_id TEXT, order_us INTEGER NOT NULL, channel_system_name TEXT, channel_external_id TEXT,
    channel_mindbox_id TEXT, channel_name TEXT, total_price TEXT, delivery_cost TEXT);
CREATE TABLE order_lines (
    line_key TEXT PRIMARY KEY, variant_key TEXT NOT NULL REFERENCES order_snapshots,
    source_line_key TEXT, line_id TEXT NOT NULL, source_line_id TEXT, is_synthetic INTEGER NOT NULL,
    ordinal INTEGER NOT NULL, line_number INTEGER NOT NULL, product_name TEXT, quantity TEXT NOT NULL,
    raw_status TEXT NOT NULL, price_of_line TEXT NOT NULL, base_price_per_item TEXT, quality TEXT NOT NULL,
    UNIQUE(variant_key, ordinal));
CREATE INDEX order_lines_source_identity ON order_lines(source_line_key);
CREATE TABLE product_occurrences (
    variant_key TEXT NOT NULL REFERENCES variants, role TEXT NOT NULL, ordinal INTEGER NOT NULL,
    selected_namespace TEXT, selected_value TEXT, quality TEXT NOT NULL,
    PRIMARY KEY(variant_key, role, ordinal)) WITHOUT ROWID;
CREATE TABLE product_keys (
    variant_key TEXT NOT NULL, role TEXT NOT NULL, ordinal INTEGER NOT NULL, key_ordinal INTEGER NOT NULL,
    namespace TEXT NOT NULL, value TEXT NOT NULL,
    PRIMARY KEY(variant_key, role, ordinal, key_ordinal),
    FOREIGN KEY(variant_key, role, ordinal) REFERENCES product_occurrences) WITHOUT ROWID;
"""
_SCHEMA_FINGERPRINT = hashlib.sha256(_SCHEMA.encode()).hexdigest()


def _schema_signature(connection):
    return _hash(connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master "
                                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall())


def _coverage(sources):
    return tuple(HistoryCoverage(c.name, c.since, c.until, c.source_kind) for c in sources.components if c.name != "customer_merges")


@contextmanager
def _read_index(path):
    _database_version(path)  # Never create WAL/shm files or recover hot journals.
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.5)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()


def _inspect(path, sources, verify_integrity=False, cancelled=lambda: False):
    with _read_index(path) as connection:
        connection.set_progress_handler(lambda: int(bool(cancelled())), 10000)
        return _inspect_connection(connection, sources, verify_integrity)


def _inspect_connection(connection, sources, verify_integrity=False):
    """Validate metadata on the same read snapshot used by a history query."""
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    if (type(metadata.get("schema_version")) is not int or metadata["schema_version"] != INDEX_SCHEMA_VERSION
            or metadata.get("normalizer_version") != records.HISTORY_SCHEMA_VERSION
            or metadata.get("schema_definition") != _SCHEMA_FINGERPRINT):
        raise _Stale
    if metadata.get("schema_signature") != _schema_signature(connection):
        raise ValueError("Damaged history index schema")
    if verify_integrity and (connection.execute("PRAGMA quick_check").fetchone() != ("ok",)
                             or connection.execute("PRAGMA foreign_key_check").fetchone() is not None):
        raise ValueError("Damaged history index data")
    if metadata.get("source_version") != sources.version:
        raise _Stale
    generation = metadata["generation"]
    if not isinstance(generation, str) or len(generation) != 32 or any(c not in "0123456789abcdef" for c in generation):
        raise ValueError("Invalid history generation")
    counts = HistoryIndexCounts(**json.loads(metadata["counts"]))
    if verify_integrity:
        _validate_storage(connection, counts)
    return HistoryIndexResult(HistoryIndexStatus.READY, generation, sources.version, counts, _coverage(sources))


def _text_decimal(value):
    return None if value is None else str(value)


def _quality(flags):
    return _json([flag.value for flag in flags])


def _variant_key(event):
    if event.source is records.HistorySource.ORDER:
        return event.details.snapshot_key
    return _hash(["action-variant", INDEX_SCHEMA_VERSION, event.event_key, event.fingerprint.hex()])


def _write_reference(connection, variant, role, ordinal, reference):
    product = reference.product
    connection.execute("INSERT INTO product_occurrences VALUES (?,?,?,?,?,?)",
                       (variant, role, ordinal, product.namespace if product else None,
                        product.value if product else None, _quality(reference.quality)))
    connection.executemany("INSERT INTO product_keys VALUES (?,?,?,?,?,?)",
                           ((variant, role, ordinal, i, key.namespace, key.value) for i, key in enumerate(reference.keys)))


def _write_variant(connection, event, variant):
    detail = event.details
    if isinstance(detail, records.ActionDetails):
        channel = detail.channel
        connection.execute("INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           (variant, detail.action_id, detail.system_name, _micros(detail.creation_datetime_utc),
                            channel.system_name, channel.external_id, channel.mindbox_id, channel.name,
                            detail.action_name, detail.brand_system_name, _text_decimal(detail.product_view_price),
                            detail.product_view_is_available))
        for role, occurrences in (("product", detail.products), ("category", detail.categories)):
            for item in occurrences:
                _write_reference(connection, variant, role, item.ordinal, item.reference)
    else:
        order, channel = detail.order, detail.order.channel
        connection.execute("INSERT OR IGNORE INTO orders VALUES (?,?)", (event.event_key, order.order_id))
        connection.execute("INSERT INTO order_snapshots VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (variant, detail.snapshot_key, order.retail_order_id, _micros(order.order_datetime_utc),
                            channel.system_name, channel.external_id, channel.mindbox_id, channel.name,
                            _text_decimal(order.total_price), _text_decimal(order.delivery_cost)))
        connection.executemany("INSERT INTO order_lines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            (line.line_key, variant, line.source_line_key, line.line_id, line.source_line_id, line.is_synthetic,
             line.ordinal, line.line_number, line.product_name, str(line.quantity), line.raw_status,
             str(line.price_of_line), _text_decimal(line.base_price_per_item), _quality(line.quality)) for line in detail.lines))
        for line in detail.lines:
            _write_reference(connection, variant, "line", line.ordinal, line.reference)


def _write_batch(connection, batch):
    for event in batch:
        customer, kind, stamp = event.canonical_customer_id, event.kind.value, _micros(event.event_datetime_utc)
        connection.execute("INSERT OR IGNORE INTO events(event_key,source,source_event_id,canonical_customer_id,kind,event_us) "
                           "VALUES (?,?,?,?,?,?)", (event.event_key, event.source.value, event.source_event_id, customer, kind, stamp))
        variant = _variant_key(event)
        inserted = connection.execute("INSERT OR IGNORE INTO variants VALUES (?,?,?,?,?,?,?,?)",
                                      (variant, event.event_key, event.fingerprint, customer, event.source_customer_id,
                                       kind, stamp, _quality(event.quality))).rowcount
        if inserted:
            connection.execute("""UPDATE events SET
                owner_conflict=owner_conflict OR (canonical_customer_id IS NOT NULL AND canonical_customer_id<>?),
                time_conflict=time_conflict OR (event_us IS NOT NULL AND event_us<>?),
                kind_conflict=kind_conflict OR (kind IS NOT NULL AND kind<>?),
                canonical_customer_id=CASE WHEN canonical_customer_id=? THEN canonical_customer_id ELSE NULL END,
                event_us=CASE WHEN event_us=? THEN event_us ELSE NULL END,
                kind=CASE WHEN kind=? THEN kind ELSE NULL END,
                variant_count=variant_count+1 WHERE event_key=?""",
                               (customer, stamp, kind, customer, stamp, kind, event.event_key))
            # The common date is mirrored on EVERY candidate link solely for
            # indexed customer keyset reads, including owner/time conflicts.
            connection.execute("INSERT OR IGNORE INTO event_customers VALUES (?,?,?)", (event.event_key, customer, stamp))
            connection.execute("UPDATE event_customers SET event_us=(SELECT event_us FROM events WHERE event_key=?) "
                               "WHERE event_key=?", (event.event_key, event.event_key))
            _write_variant(connection, event, variant)
        connection.execute("UPDATE events SET occurrence_count=occurrence_count+1 WHERE event_key=?", (event.event_key,))
        connection.execute("INSERT INTO record_origins VALUES (?,?,?)",
                           (variant, event.provenance.component_version, event.provenance.record_ordinal))


def _counts(connection):
    logical = dict(connection.execute("SELECT source,COUNT(*) FROM events GROUP BY source"))
    raw = dict(connection.execute("SELECT component,SUM(record_count) FROM sources GROUP BY component"))
    variants = dict(connection.execute("SELECT e.source,COUNT(*) FROM variants v JOIN events e USING(event_key) GROUP BY e.source"))
    repeats, conflicts, owners, times, kinds = connection.execute(
        "SELECT COALESCE(SUM(occurrence_count-variant_count),0), COALESCE(SUM(variant_count>1),0), "
        "COALESCE(SUM(owner_conflict),0),COALESCE(SUM(time_conflict),0),COALESCE(SUM(kind_conflict),0) FROM events").fetchone()
    return HistoryIndexCounts(raw.get("actions", 0), raw.get("orders", 0), logical.get("ACTION", 0), logical.get("ORDER", 0),
                              variants.get("ACTION", 0), variants.get("ORDER", 0), repeats, conflicts, owners, times, kinds)


def _validate_storage(connection, counts):
    """Check stored accounting/consensus, not business payment or latest-state rules."""
    if counts != _counts(connection):
        raise ValueError("History index counts disagree")
    invalid = connection.execute("""SELECT 1 FROM events e LEFT JOIN (
        SELECT event_key,COUNT(*) AS n,COUNT(DISTINCT canonical_customer_id) AS owners,
          MIN(canonical_customer_id) AS customer,COUNT(DISTINCT event_us) AS times,MIN(event_us) AS stamp,
          COUNT(DISTINCT kind) AS kinds,MIN(kind) AS kind FROM variants GROUP BY event_key
        ) v USING(event_key) WHERE e.variant_count IS NOT v.n
          OR e.owner_conflict IS NOT (v.owners>1) OR e.time_conflict IS NOT (v.times>1)
          OR e.kind_conflict IS NOT (v.kinds>1)
          OR e.canonical_customer_id IS NOT (CASE WHEN v.owners=1 THEN v.customer ELSE NULL END)
          OR e.event_us IS NOT (CASE WHEN v.times=1 THEN v.stamp ELSE NULL END)
          OR e.kind IS NOT (CASE WHEN v.kinds=1 THEN v.kind ELSE NULL END) LIMIT 1""").fetchone()
    origins = connection.execute("""SELECT 1 FROM events e LEFT JOIN (
        SELECT event_key,COUNT(*) AS n FROM record_origins JOIN variants USING(variant_key) GROUP BY event_key
        ) o USING(event_key) WHERE e.occurrence_count IS NOT o.n LIMIT 1""").fetchone()
    payloads = connection.execute("""SELECT 1 FROM variants v JOIN events e USING(event_key)
        LEFT JOIN actions a USING(variant_key) LEFT JOIN order_snapshots o USING(variant_key)
        WHERE (e.source='ACTION' AND (a.variant_key IS NULL OR o.variant_key IS NOT NULL))
           OR (e.source='ORDER' AND (o.variant_key IS NULL OR a.variant_key IS NOT NULL)) LIMIT 1""").fetchone()
    links = connection.execute("""SELECT 1 FROM event_customers ec JOIN events e USING(event_key)
        WHERE ec.event_us IS NOT e.event_us OR NOT EXISTS (
          SELECT 1 FROM variants v WHERE v.event_key=ec.event_key AND v.canonical_customer_id=ec.canonical_customer_id)
        UNION ALL SELECT 1 FROM variants v WHERE NOT EXISTS (
          SELECT 1 FROM event_customers ec WHERE ec.event_key=v.event_key AND ec.canonical_customer_id=v.canonical_customer_id)
        LIMIT 1""").fetchone()
    if invalid or origins or payloads or links:
        raise ValueError("History index relationships disagree")


def _check_cancel(cancelled):
    if cancelled():
        raise _Cancelled


@contextmanager
def _cache_writer(cache_dir):
    with ExitStack() as stack:
        try:
            stack.enter_context(_writer_lock(cache_dir))
        except TrainingBatchError:
            raise _Busy from None
        yield


class CustomerHistoryIndexService:
    def __init__(self, *, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir).resolve()
        self._build_lock = Lock()

    def index_path(self, *, raw_root: str | Path = DEFAULT_RAW_ROOT) -> Path:
        return self.cache_dir / (_hash(str(Path(raw_root).resolve())) + ".sqlite")

    def get_status(self, *, raw_root: str | Path = DEFAULT_RAW_ROOT, verify_integrity: bool = False) -> HistoryIndexResult:
        """Metadata/schema check by default; explicit integrity check scans only the derived DB."""
        if type(verify_integrity) is not bool:
            raise ValueError("Invalid history integrity option")
        root = Path(raw_root).resolve()
        path = self.index_path(raw_root=root)
        if not path.exists():
            return HistoryIndexResult(HistoryIndexStatus.MISSING)
        try:
            with _reader_lock(root):
                sources = _capture_sources(root)
                try:
                    result = _inspect(path, sources, verify_integrity)
                except _Stale:
                    return HistoryIndexResult(HistoryIndexStatus.STALE)
                except Exception as exc:
                    return HistoryIndexResult(HistoryIndexStatus.UNAVAILABLE if _unavailable(exc) else HistoryIndexStatus.CORRUPT)
                if _capture_sources(root).version != sources.version:
                    return HistoryIndexResult(HistoryIndexStatus.STALE)
                return result
        except Exception as exc:
            return HistoryIndexResult(HistoryIndexStatus.SOURCE_UNAVAILABLE if _unavailable(exc) else HistoryIndexStatus.SOURCE_INCONSISTENT)

    def build_index(self, *, raw_root: str | Path = DEFAULT_RAW_ROOT, force: bool = False,
                    progress: Callable[[HistoryIndexProgress], None] | None = None,
                    cancelled: Callable[[], bool] = lambda: False, batch_size: int = 500,
                    progress_every: int = 10000) -> HistoryIndexResult:
        if (type(force) is not bool or type(batch_size) is not int or not 1 <= batch_size <= 10000
                or type(progress_every) is not int or progress_every < 1):
            raise ValueError("Invalid history build options")
        start = perf_counter()
        if not self._build_lock.acquire(blocking=False):
            return HistoryIndexResult(HistoryIndexStatus.BUILD_IN_PROGRESS)
        temporary = None
        try:
            _check_cancel(cancelled)
            root = Path(raw_root).resolve()
            if self.cache_dir.is_relative_to(root):
                raise ValueError("Cache must be outside canonical source data")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with _cache_writer(self.cache_dir), _reader_lock(root):
                sources = _capture_sources(root)
                path = self.index_path(raw_root=root)
                for suffix in ("-journal", "-wal", "-shm"):
                    sidecar = Path(str(path) + suffix)
                    if sidecar.exists() and sidecar.stat().st_size:
                        return HistoryIndexResult(HistoryIndexStatus.UNAVAILABLE)
                if path.exists() and not force:
                    try:
                        ready = _inspect(path, sources)
                    except Exception as exc:
                        if _unavailable(exc):
                            return HistoryIndexResult(HistoryIndexStatus.UNAVAILABLE)
                        # Explicit builds may replace a stale or damaged derived cache.
                    else:
                        _check_cancel(cancelled)
                        if _capture_sources(root).version != sources.version:
                            raise _SourceChanged
                        return HistoryIndexResult(HistoryIndexStatus.READY, ready.generation, ready.source_version,
                                                  ready.counts, ready.coverage, True, (perf_counter() - start) * 1000)
                fd, name = tempfile.mkstemp(prefix=".building-", suffix=".sqlite", dir=self.cache_dir)
                os.close(fd)
                temporary = Path(name)
                result = self._build(temporary, sources, cancelled, progress, batch_size, progress_every)
                _check_cancel(cancelled)
                if _capture_sources(root).version != sources.version:
                    raise _SourceChanged
                # Validate complete derived data before making it eligible for reads.
                _inspect(temporary, sources, verify_integrity=True, cancelled=cancelled)
                with temporary.open("r+b") as stream:
                    os.fsync(stream.fileno())
                _check_cancel(cancelled)
                if _capture_sources(root).version != sources.version:
                    raise _SourceChanged
                os.replace(temporary, path)
                return HistoryIndexResult(HistoryIndexStatus.READY, result.generation, result.source_version,
                                          result.counts, result.coverage, False, (perf_counter() - start) * 1000)
        except _Cancelled:
            return HistoryIndexResult(HistoryIndexStatus.CANCELLED, elapsed_ms=(perf_counter() - start) * 1000)
        except _Busy:
            return HistoryIndexResult(HistoryIndexStatus.BUILD_IN_PROGRESS)
        except _SourceChanged:
            return HistoryIndexResult(HistoryIndexStatus.STALE)
        except sqlite3.OperationalError as exc:
            if getattr(exc, "sqlite_errorcode", None) == 9 or str(exc) == "interrupted":
                return HistoryIndexResult(HistoryIndexStatus.CANCELLED, elapsed_ms=(perf_counter() - start) * 1000)
            return HistoryIndexResult(HistoryIndexStatus.UNAVAILABLE if _unavailable(exc) else HistoryIndexStatus.SOURCE_INCONSISTENT)
        except Exception as exc:
            return HistoryIndexResult(HistoryIndexStatus.SOURCE_UNAVAILABLE if _unavailable(exc) else HistoryIndexStatus.SOURCE_INCONSISTENT)
        finally:
            try:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass  # Crash/locked leftovers are never read as a published index.
            finally:
                self._build_lock.release()

    @staticmethod
    def _build(path, sources, cancelled, progress, batch_size, progress_every):
        connection = sqlite3.connect(path)
        try:
            connection.set_progress_handler(lambda: int(bool(cancelled())), 10000)
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(_SCHEMA)
            for component in sources.components:
                connection.execute("INSERT INTO sources(source_key,component,directory,since_us,until_us,observed_us,source_kind,export_id,operation) "
                                   "VALUES (?,?,?,?,?,?,?,?,?)",
                                   (component.key, component.name, str(component.directory), _micros(component.since),
                                    _micros(component.until), _micros(component.updated), component.source_kind,
                                    component.export_id, component.operation))
                connection.executemany("INSERT INTO source_parts VALUES (?,?,?,?)",
                                       ((component.key, i, p.name, _json(stat)) for i, (p, stat) in enumerate(component.parts, 1)))
            merge = sources.components[0]
            merge_count = 0
            def merge_records():
                nonlocal merge_count
                stream = iter_export("customer_merges", input_dir=merge.directory)
                try:
                    for raw in stream:
                        _check_cancel(cancelled)
                        merge_count += 1
                        if progress and merge_count % progress_every == 0:
                            progress(HistoryIndexProgress("customer_merges", merge_count))
                        yield adapt_customer_merge(raw)
                finally:
                    stream.close()
            with closing(merge_records()) as stream:
                resolver = CustomerIdResolver(stream)
            connection.execute("UPDATE sources SET record_count=? WHERE source_key=?", (merge_count, merge.key))
            batch, processed = [], 0
            for component in sources.components[1:]:
                normalize = normalization.normalize_action if component.name == "actions" else normalization.normalize_order
                stream = iter_export(component.name, input_dir=component.directory)
                count = 0
                try:
                    for ordinal, raw in enumerate(stream):
                        _check_cancel(cancelled)
                        origin = records.HistoryProvenance(component.source_kind, component.key,
                                                           record_ordinal=ordinal, observed_at_utc=component.updated)
                        batch.append(normalize(raw, resolver, selection=sources.selection, provenance=origin))
                        processed += 1
                        count += 1
                        if len(batch) == batch_size:
                            _write_batch(connection, batch)
                            batch.clear()
                        if progress and processed % progress_every == 0:
                            progress(HistoryIndexProgress(component.name, processed))
                    connection.execute("UPDATE sources SET record_count=? WHERE source_key=?", (count, component.key))
                finally:
                    stream.close()
            _write_batch(connection, batch)
            _check_cancel(cancelled)
            counts = _counts(connection)
            generation = uuid.uuid4().hex
            connection.executemany("INSERT INTO metadata VALUES (?,?)", (
                ("schema_version", INDEX_SCHEMA_VERSION), ("normalizer_version", records.HISTORY_SCHEMA_VERSION),
                ("schema_definition", _SCHEMA_FINGERPRINT), ("schema_signature", _schema_signature(connection)),
                ("source_version", sources.version), ("generation", generation), ("counts", _json(asdict(counts))),
                ("selection", _json(asdict(sources.selection))),
                ("built_at_us", _micros(datetime.now(timezone.utc))),
            ))
            connection.commit()
            if progress:
                progress(HistoryIndexProgress("complete", processed))
            return HistoryIndexResult(HistoryIndexStatus.READY, generation, sources.version, counts, _coverage(sources))
        finally:
            connection.close()


_DEFAULT_SERVICE = CustomerHistoryIndexService()


def build_customer_history_index(*, raw_root: str | Path = DEFAULT_RAW_ROOT, force: bool = False,
                                 progress: Callable[[HistoryIndexProgress], None] | None = None,
                                 cancelled: Callable[[], bool] = lambda: False, batch_size: int = 500,
                                 progress_every: int = 10000) -> HistoryIndexResult:
    return _DEFAULT_SERVICE.build_index(raw_root=raw_root, force=force, progress=progress, cancelled=cancelled,
                                        batch_size=batch_size, progress_every=progress_every)


def get_customer_history_index_status(*, raw_root: str | Path = DEFAULT_RAW_ROOT,
                                     verify_integrity: bool = False) -> HistoryIndexResult:
    return _DEFAULT_SERVICE.get_status(raw_root=raw_root, verify_integrity=verify_integrity)
