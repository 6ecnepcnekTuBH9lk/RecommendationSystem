"""One indexed customer dataset; each streamed month is a SQLite transaction."""

from contextlib import contextmanager
from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from .adapters import adapt_customer_merge
from .adapters.customer_contacts import adapt_customer_contact_candidate
from .adapters._common import identifier, timestamp
from .canonical_storage import catalog, checked_directory, stamp, storage_lock
from .customers_stream import iter_json_records
from .identity import CustomerIdResolver
from .manual_sources import normalize_sources
from .raw_reader import iter_export


def database(root):
    return Path(root).resolve() / "canonical/customers.sqlite"


def encode(raw):
    """Lossless JSON numbers from ijson's Decimal, without a new dependency."""
    if isinstance(raw, Decimal):
        if not raw.is_finite():
            raise ValueError("Invalid JSON number")
        return str(raw)
    if isinstance(raw, dict):
        return "{" + ",".join(json.dumps(k) + ":" + encode(v) for k, v in raw.items()) + "}"
    if isinstance(raw, list):
        return "[" + ",".join(encode(v) for v in raw) + "]"
    return json.dumps(raw, ensure_ascii=False, allow_nan=False)


@contextmanager
def connect(path, *, create=False):
    if create:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # mode=rw never creates a missing dataset, but permits hot-journal recovery
    # after QProcess termination during an uncommitted month.
    connection = sqlite3.connect(str(path) if create else Path(path).as_uri() + "?mode=rw", uri=not create, timeout=30)
    try:
        if create:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA cache_size=-8192")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, changed TEXT, raw TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS receipts (job TEXT, component INTEGER, PRIMARY KEY(job, component));
            """)
        yield connection
    finally:
        connection.close()


def resolver_for(root, *, required=False):
    entry = catalog(root)["customer_merges"]
    if entry:
        directory = checked_directory(root, entry["directory"], "customer_merges")
        return CustomerIdResolver(adapt_customer_merge(raw) for raw in iter_export("customer_merges", input_dir=directory))
    # Explicit legacy fallback; invalid saved manifests are not silently ignored.
    if required or any((Path(root) / "training_batches").glob("*/manifest.json")):
        from .manual_import import select_merges_source, _resolver
        return _resolver(select_merges_source(root), Path(root).resolve())
    return CustomerIdResolver(())


def month_committed(root, job, index):
    path = database(root)
    if not path.exists():
        return False
    with connect(path) as connection:
        return connection.execute("SELECT 1 FROM receipts WHERE job=? AND component=?", (job, index)).fetchone() is not None


def _upsert(connection, raw, resolver):
    candidate = adapt_customer_contact_candidate(raw, resolver)
    key = resolver.resolve(identifier(raw, "ids.mindboxId"))
    changed = timestamp(raw, "changeDateTimeUtc", required=False)
    changed = changed.isoformat() if changed else ""
    connection.execute("""INSERT INTO profiles VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE
        SET changed=excluded.changed, raw=excluded.raw
        WHERE excluded.changed >= profiles.changed""", (key, changed, encode(raw)))
    del candidate


def _intervals(old, since, until):
    intervals = sorted([*old, [since.isoformat(), until.isoformat()]])
    result = []
    for a, b in intervals:
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def _source_kinds(summary):
    """Older snapshots contain only source_kind; reads never rewrite their metadata."""
    return sorted(set(summary.get("source_kinds", [summary["source_kind"]] if summary.get("source_kind") else [])))


def customer_summary(root):
    path = database(root)
    if not path.exists():
        return None
    with connect(path) as connection:
        row = connection.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()
        if not row:
            return None
        summary = json.loads(row[0])
        summary["source_kinds"] = _source_kinds(summary)
        return summary


def apply_month(root, directory, since, until, *, job, index):
    """Caller holds global storage lock. Raw, counts, coverage and receipt commit together."""
    resolver = resolver_for(root)
    with connect(database(root), create=True) as connection, connection:
        # Only identities affected by merges need rewriting; untouched raw stays on disk.
        last = ""
        while True:
            keys = connection.execute("SELECT id FROM profiles WHERE id>? ORDER BY id LIMIT 1024", (last,)).fetchall()
            if not keys:
                break
            last = keys[-1][0]
            for (key,) in keys:
                canonical = resolver.resolve(key)
                if key != canonical:
                    row = connection.execute("SELECT raw FROM profiles WHERE id=?", (key,)).fetchone()
                    _upsert(connection, json.loads(row[0], parse_float=Decimal), resolver)
                    connection.execute("DELETE FROM profiles WHERE id=?", (key,))
        for raw in iter_export("customers", input_dir=directory):
            _upsert(connection, raw, resolver)
        row = connection.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()
        old = json.loads(row[0]) if row else {}
        summary = {"updated": stamp(), "intervals": _intervals(old.get("intervals", []), since, until),
                   "count": connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0], "source_kind": "API",
                   "source_kinds": sorted(set(_source_kinds(old)) | {"API"})}
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('summary', ?)", (json.dumps(summary),))
        connection.execute("INSERT OR REPLACE INTO receipts VALUES (?, ?)", (job, index))


def import_full(root, source, *, cancelled=None, progress=None):
    """Manual full snapshot: build a separate DB, then atomically replace the current DB."""
    from .manual_import import check_cancel, VALIDATION_PROGRESS_EVERY

    root = Path(root).resolve()
    sources = normalize_sources(source)

    with storage_lock(root):
        from .canonical_storage import collect_unreferenced

        collect_unreferenced(root)
        resolver = resolver_for(root, required=True)

        if database(root).exists():
            with connect(database(root)) as previous:
                previous.execute("SELECT id FROM profiles LIMIT 1").fetchone()

        fd, temporary = tempfile.mkstemp(
            prefix=".customers-",
            suffix=".sqlite",
            dir=database(root).parent,
        )
        os.close(fd)

        try:
            processed = 0

            with connect(temporary, create=True) as connection, connection:
                if progress:
                    progress("Проверка клиентов...")

                for number, path in enumerate(sources, 1):
                    check_cancel(cancelled)

                    if progress:
                        progress(
                            f"Проверка customers: файл "
                            f"{number} из {len(sources)}"
                        )

                    for raw in iter_json_records(path, "customers"):
                        check_cancel(cancelled)

                        _upsert(
                            connection,
                            raw,
                            resolver,
                        )

                        processed += 1

                        if (
                            progress
                            and processed % VALIDATION_PROGRESS_EVERY == 0
                        ):
                            progress(
                                f"Проверка customers: "
                                f"{processed} записей"
                            )

                if progress:
                    progress(
                        f"Проверка customers завершена: "
                        f"{processed} записей"
                    )

                count = connection.execute(
                    "SELECT COUNT(*) FROM profiles"
                ).fetchone()[0]

                connection.execute(
                    "INSERT INTO metadata VALUES ('summary', ?)",
                    (
                        json.dumps({
                            "updated": stamp(),
                            "intervals": [],
                            "count": count,
                            "source_kind": "MANUAL",
                            "source_kinds": ["MANUAL"],
                        }),
                    ),
                )

            check_cancel(cancelled)
            os.replace(
                temporary,
                database(root),
            )

        finally:
            Path(temporary).unlink(
                missing_ok=True
            )

        return database(root)


def load_contact_index(root, model_mappings=None, *, progress=None, progress_every=100000):
    from Application.customer_profiles import build_customer_contact_index
    with storage_lock(root), connect(database(root)) as connection:
        resolver = resolver_for(root)
        users = None if model_mappings is None else (
            model_mappings["idx2user"] if isinstance(model_mappings, dict) else model_mappings.idx2user)
        wanted = set(users) if users is not None else None
        def records():
            for (text,) in connection.execute("SELECT raw FROM profiles ORDER BY id"):
                yield adapt_customer_contact_candidate(json.loads(text, parse_float=Decimal), resolver, wanted)
        return build_customer_contact_index(records(), users, progress=progress, progress_every=progress_every)
