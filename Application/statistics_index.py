"""Explicit, cancellable full rebuild; immutable generations and atomic pointer.

Never called by opening the statistics tab. The canonical lock is held during
input projection so publication/GC cannot remove source parts underneath it.
Old generations are retained (no unsafe deletion of a reader's open database).
"""

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from time import monotonic
import uuid

from Application import dataset_statistics as reference
from Application.analysis_filter import AnalysisFilter
from Application.mindbox.adapters import adapt_customer_merge
from Application.mindbox.canonical_storage import atomic_json, storage_lock
from Application.mindbox.daily_training_batch import _writer_lock
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, iter_export
from Application.mindbox.selection import MindboxSelectionConfig
from Application.paths import PROJECT_ROOT
from Application.product_resolution import DEFAULT_CATALOG_PATH, ProductResolver
from Application.statistics_facts import SCHEMA, DEMOGRAPHY_SCHEMA, INDEXES, Projector, dump
from Application.statistics_period import shared_intervals
from Application.statistics_slices import capture_sources
from Application.statistics_storage_safety import (
    StatisticsIndexError, check_owned_staging, path_identity, reject_reparse, validate_storage_root,
)

SCHEMA_VERSION = 2
DEFAULT_INDEX_DIR = PROJECT_ROOT / "filtered_data" / "statistics_index"
GIB = 1024**3


def source_version(provenance):
    # Only final results depend on the local demographic day/timezone/filter.
    # Customers and Merges remain bound: independent source refresh is unsafe
    # without a persistent original-ID to SID mapping (intentionally absent).
    return {k: v for k, v in provenance.items() if k not in {"filter", "as_of", "timezone", "customer_period", "rfm_as_of"}}


def component_digest(path, check=lambda: None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            check()
            digest.update(block)
    check()
    return digest.hexdigest()


def protect_directory(path):
    """Narrow access on a newly created owned directory only; fail closed."""
    if os.name != "nt":
        path.chmod(0o700)
        return
    import base64
    import subprocess
    literal = "'" + str(path.resolve()).replace("'", "''") + "'"
    command = (f"$p={literal}; $sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
               '& icacls.exe $p /inheritance:r /grant:r "*$($sid):(OI)(CI)F" '
               "'*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' | Out-Null; exit $LASTEXITCODE")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                             base64.b64encode(command.encode("utf-16le")).decode("ascii")], capture_output=True)
    if result.returncode:
        raise StatisticsIndexError("PRIVATE_INDEX_ACL_FAILED")


def read_pointer(directory):
    try:
        path = Path(directory) / "current.json"
        reject_reparse(path)
        if path.is_symlink() or path.stat().st_size > 1024*1024:
            raise StatisticsIndexError("INVALID_MANIFEST")
        from Application.statistics_cache import _unique_object
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if (set(value) != {"schema", "generation", "manifest_sha256"}
                or type(value["schema"]) is not int or value["schema"] != SCHEMA_VERSION
                or not re.fullmatch(r"[0-9a-f]{32}", value["generation"])):
            raise StatisticsIndexError("INCOMPATIBLE_SCHEMA")
        return value
    except FileNotFoundError:
        raise StatisticsIndexError("INDEX_MISSING") from None
    except StatisticsIndexError:
        raise
    except (OSError, UnicodeError, KeyError, TypeError, ValueError):
        raise StatisticsIndexError("INVALID_MANIFEST") from None


def generation_paths(directory, pointer):
    directory = reject_reparse(directory).resolve()
    folder = directory / pointer["generation"]
    reject_reparse(folder)
    if folder.resolve().parent != directory:
        raise StatisticsIndexError("INVALID_GENERATION_PATH")
    return folder / "statistics.sqlite", folder / "manifest.json"


def manifest_digest(value):
    return hashlib.sha256(dump(value).encode("ascii")).hexdigest()


def sample_memory():
    """Windows OS counters only; independent of training/model modules."""
    import ctypes
    from ctypes import wintypes

    class Memory(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_physical", "available_physical", "commit_limit", "available_commit",
                "total_virtual", "available_virtual", "extended")]

    class Process(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                "peak_nonpaged_pool", "nonpaged_pool", "pagefile_usage", "peak_pagefile_usage", "private_usage")]

    kernel, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Memory)]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Process), wintypes.DWORD]
    memory, process = Memory(), Process()
    memory.length, process.cb = ctypes.sizeof(memory), ctypes.sizeof(process)
    if (not kernel.GlobalMemoryStatusEx(ctypes.byref(memory))
            or not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(process), process.cb)):
        raise StatisticsIndexError("RESOURCE_MONITOR_UNAVAILABLE")
    return {name: getattr(memory, name) for name in ("available_physical", "available_commit")} | {
        name: getattr(process, name) for name in ("private_usage", "working_set", "peak_working_set", "peak_pagefile_usage")}


class BuildGuard:
    def __init__(self, folder, cancelled, observer, reserve_bytes):
        if reserve_bytes < 20*GIB:
            raise ValueError("Statistics disk reserve must be at least 20 GiB")
        self.folder, self.cancelled, self.observer = folder, cancelled, observer
        self.reserve, self.started, self.last = reserve_bytes, monotonic(), -float("inf")
        self.failure = None

    def check(self, *, force=False, extra=0):
        if self.cancelled and self.cancelled():
            raise InterruptedError("Statistics index build cancelled")
        if force or monotonic()-self.last >= 1:
            self.last = monotonic()
            free = shutil.disk_usage(self.folder).free
            sample = {"elapsed_seconds": self.last-self.started, "free_disk_bytes": free,
                      "sqlite_bytes": sum(p.stat().st_size for p in self.folder.glob("*.sqlite*"))}
            if os.name == "nt":
                sample.update(sample_memory())
                if (sample["private_usage"] > 3*GIB or sample["available_physical"] < GIB
                        or sample["available_commit"] < GIB):
                    raise StatisticsIndexError("INSUFFICIENT_MEMORY")
            if self.observer:
                self.observer(sample)
            if free < self.reserve + extra:
                raise StatisticsIndexError("INSUFFICIENT_DISK")

    def sql_check(self):
        try:
            self.check()
            return 0
        except Exception as exc:
            self.failure = exc
            return 1


def build_statistics_index(*, raw_root=DEFAULT_RAW_ROOT, catalog_path=DEFAULT_CATALOG_PATH,
                           index_dir=DEFAULT_INDEX_DIR, cancelled=None, progress=None,
                           observer=None, expected_bytes=10*GIB, reserve_bytes=20*GIB):
    """One explicit full attempt, no retry/time limit. Keep a 20 GiB disk floor.

    expected_bytes must be budgeted from a controlled pilot before a real build;
    it reserves database + journal/sort headroom, beyond all existing generations.
    """
    if expected_bytes <= 0:
        raise ValueError("Positive build size estimate required")
    directory = reject_reparse(index_dir)
    members = tuple(directory / name for name in ("current.json", ".writer.lock")
                    if os.path.lexists(directory / name))
    root_id = validate_storage_root(directory, children=members)
    root = Path(raw_root).resolve()
    if not (root / "canonical/catalog.json").is_file():
        raise StatisticsIndexError("SOURCE_UNAVAILABLE")
    if directory == root or directory.is_relative_to(root / "canonical"):
        raise StatisticsIndexError("INDEX_MUST_BE_SEPARATE")
    generation = uuid.uuid4().hex
    folder = directory / generation
    db = demographics = None
    published = False
    # Order is always index writer -> canonical writer; no nested capture lock.
    with _writer_lock(directory):
        folder.mkdir()
        folder_id = path_identity(folder)
        try:
            protect_directory(folder)
            validate_storage_root(directory, children=(folder,))
            guard = BuildGuard(folder, cancelled, observer, reserve_bytes)
            guard.check(force=True, extra=2*expected_bytes)
            path = folder / "statistics.sqlite"
            db = sqlite3.connect(path)
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA cache_size=-32768")
            # Avoid uncontrolled external temp files; abort on resource limits.
            db.execute("PRAGMA temp_store=MEMORY")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            db.set_progress_handler(guard.sql_check, 10000)
            db.executescript(SCHEMA)
            demographic_path = folder / "demographics.sqlite"
            demographics = sqlite3.connect(demographic_path)
            demographics.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            demographics.execute("PRAGMA synchronous=FULL")
            demographics.execute("PRAGMA cache_size=-8192")
            demographics.set_progress_handler(guard.sql_check, 10000)
            demographics.executescript(DEMOGRAPHY_SCHEMA)
            counts = {}
            with storage_lock(root):
                before, _ = capture_sources(root, catalog_path, AnalysisFilter(), already_locked=True)
                data, entries = reference._statistics_sources(root)

                def records(name, sources):
                    for entry in sources:
                        physical = reference._source_directory(root, name, entry)
                        yield from iter_export(name, input_dir=physical)

                def merges():
                    merge = data["customer_merges"]
                    for raw in records("customer_merges", [merge] if merge else []):
                        guard.check()
                        yield adapt_customer_merge(raw)

                identities = CustomerIdResolver(merges())
                catalog, metadata, _, _ = reference._catalog_snapshot(catalog_path)
                products = ProductResolver(catalog)
                db.executemany("INSERT INTO items VALUES(?,?,?)", ((code, item.nomenclature_type, item.collection)
                                                                  for code, item in metadata.items()))
                period = shared_intervals(*[[(e["since"], e["until"]) for e in entries[n]] for n in ("actions", "orders")])
                projector = Projector(db, identities, MindboxSelectionConfig(**data["selection"]), products, period)
                for name in ("actions", "orders"):
                    count = 0
                    for count, raw in enumerate(records(name, entries[name]), 1):
                        guard.check()
                        (projector.action if name == "actions" else projector.order)(raw, count)
                        if count % 1000 == 0:
                            guard.check(force=True)
                            db.commit()
                            guard.check(force=True)
                        if progress and (count == 1 or count % 100000 == 0):
                            progress(f"Индекс {name}: {count:,}".replace(",", " "))
                    counts[name] = count
                projector.profiles(root, demographics, guard.check)
                counts["profiles"] = demographics.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
                for name in ("actions", "orders"):
                    if db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] != counts[name]:
                        raise StatisticsIndexError("INCOMPLETE_PROJECTION")
                for name in ("action_products", "order_lines"):
                    counts[name] = db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                after, _ = capture_sources(root, catalog_path, AnalysisFilter(), already_locked=True)
                if source_version(before) != source_version(after):
                    raise StatisticsIndexError("STALE_SOURCE")
                # Do not leave original customer/order ID mappings in free pages.
                db.execute("PRAGMA secure_delete=ON")
                db.execute("DROP TABLE customer_ids")
                db.execute("DROP TABLE order_ids")
                db.commit()
                # The merge graph and source projection state are no longer
                # needed during SQLite sorting/integrity verification.
                del projector, identities, products, catalog, metadata
                raw = None
            for command in INDEXES:
                guard.check(force=True)
                if progress:
                    progress("Индекс: построение SQL-индексов")
                db.execute(command)
                db.commit()
            db.execute("ANALYZE")
            binding = {"schema": SCHEMA_VERSION, "generation": generation,
                       "provenance": source_version(before), "profiles": counts["profiles"]}
            demographics.execute("INSERT INTO metadata VALUES('binding',?)", (dump(binding),))
            demographics.execute("ANALYZE")
            demographics.commit()
            if demographics.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise StatisticsIndexError("INTEGRITY_FAILED")
            demographics.close()
            demographics = None
            with demographic_path.open("r+b") as stream:
                os.fsync(stream.fileno())
            manifest = {"schema": SCHEMA_VERSION, "generation": generation, "complete": True,
                        "provenance": source_version(before), "counts": counts,
                        "demographics": {"sha256": component_digest(demographic_path, guard.check),
                                         "bytes": demographic_path.stat().st_size},
                        "elapsed_seconds": monotonic()-guard.started}
            encoded = dump(manifest)
            db.execute("INSERT INTO metadata VALUES('manifest',?)", (encoded,))
            db.commit()
            if progress:
                progress("Индекс: проверка целостности новой SQLite")
            if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)] or db.execute("PRAGMA foreign_key_check").fetchone():
                raise StatisticsIndexError("INTEGRITY_FAILED")
            guard.check(force=True)
            db.close()
            db = None
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
            atomic_json(folder / "manifest.json", manifest)
            with storage_lock(root):
                current, _ = capture_sources(root, catalog_path, AnalysisFilter(), already_locked=True)
                if source_version(current) != manifest["provenance"]:
                    raise StatisticsIndexError("STALE_SOURCE")
                guard.check(force=True)
                check_owned_staging(directory, folder, root_id, folder_id)
                children = (folder, *folder.iterdir())
                if (directory / "current.json").exists():
                    children += (directory / "current.json",)
                validate_storage_root(directory, children=children)
                atomic_json(directory / "current.json", {"schema": SCHEMA_VERSION, "generation": generation,
                                                         "manifest_sha256": manifest_digest(manifest)})
                published = True
            return manifest | {"sqlite_bytes": path.stat().st_size,
                               "demographics_bytes": demographic_path.stat().st_size,
                               "elapsed_seconds": monotonic()-guard.started}
        except sqlite3.OperationalError:
            if 'guard' in locals() and guard.failure is not None:
                raise guard.failure from None
            raise StatisticsIndexError("SQLITE_BUILD_ERROR") from None
        finally:
            if db is not None:
                db.close()
            if demographics is not None:
                demographics.close()
            if not published:
                # Only this UUID's owned staging, never an old generation/source.
                check_owned_staging(directory, folder, root_id, folder_id)
                shutil.rmtree(folder)


def open_generation(directory, provenance, *, connection_factory=sqlite3.Connection, check=lambda: None):
    directory = reject_reparse(directory)
    pointer = read_pointer(directory)
    path, manifest_path = generation_paths(directory, pointer)
    if path.is_symlink() or manifest_path.is_symlink():
        raise StatisticsIndexError("INVALID_GENERATION_PATH")
    try:
        demographic_path = path.with_name("demographics.sqlite")
        validate_storage_root(directory, children=(path.parent, path, manifest_path,
                                                  demographic_path, directory / "current.json"))
        if manifest_path.stat().st_size > 4*1024**2:
            raise StatisticsIndexError("INVALID_MANIFEST")
        from Application.statistics_cache import _unique_object
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if (not isinstance(manifest, dict) or set(manifest) != {
                "schema", "generation", "complete", "provenance", "counts", "elapsed_seconds", "demographics"}
                or manifest_digest(manifest) != pointer["manifest_sha256"] or manifest.get("complete") is not True
                or manifest.get("generation") != pointer["generation"] or manifest.get("schema") != SCHEMA_VERSION):
            raise StatisticsIndexError("INVALID_MANIFEST")
        if manifest["provenance"] != source_version(provenance):
            raise StatisticsIndexError("STALE_SOURCE")
        if (demographic_path.is_symlink()
                or demographic_path.stat().st_size != manifest["demographics"]["bytes"]
                or component_digest(demographic_path, check) != manifest["demographics"]["sha256"]):
            raise StatisticsIndexError("INVALID_DEMOGRAPHICS")
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2, factory=connection_factory)
        try:
            db.execute("PRAGMA trusted_schema=OFF")
            db.execute("PRAGMA cache_size=-32768")
            db.execute("PRAGMA temp_store=MEMORY")
            stored = db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
            if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION or stored is None or json.loads(stored[0]) != manifest:
                raise StatisticsIndexError("INCOMPATIBLE_SCHEMA")
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if (not {"metadata", "items", "actions", "action_products", "orders", "order_lines"} <= tables
                    or {"customer_ids", "order_ids", "profiles"} & tables):
                raise StatisticsIndexError("INCOMPATIBLE_SCHEMA")
            db.execute("ATTACH DATABASE ? AS demographics", (demographic_path.resolve().as_uri() + "?mode=ro",))
            binding = db.execute("SELECT value FROM demographics.metadata WHERE key='binding'").fetchone()
            expected = {"schema": SCHEMA_VERSION, "generation": manifest["generation"],
                        "provenance": manifest["provenance"], "profiles": manifest["counts"]["profiles"]}
            if (db.execute("PRAGMA demographics.user_version").fetchone()[0] != SCHEMA_VERSION
                    or binding is None or json.loads(binding[0]) != expected):
                raise StatisticsIndexError("INVALID_DEMOGRAPHICS")
        except Exception:
            db.close()
            raise
        return db, manifest
    except (StatisticsIndexError, InterruptedError):
        raise
    except (OSError, UnicodeError, sqlite3.Error, KeyError, TypeError, ValueError):
        raise StatisticsIndexError("INDEX_UNAVAILABLE") from None


def index_state(*, raw_root=DEFAULT_RAW_ROOT, catalog_path=DEFAULT_CATALOG_PATH, index_dir=DEFAULT_INDEX_DIR):
    try:
        provenance, _ = capture_sources(raw_root, catalog_path, AnalysisFilter())
        db, _ = open_generation(index_dir, provenance)
        with closing(db):
            return "READY"
    except StatisticsIndexError as exc:
        return str(exc)
    except Exception:
        return "SOURCE_UNAVAILABLE"
