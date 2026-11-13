"""Separate, explicit schema 4 build and read path for UNIFIED_PURCHASE_V1."""

from contextlib import closing, contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import uuid

from Application import dataset_statistics as reference
from Application.analysis_filter import AnalysisFilter
from Application.mindbox.adapters import adapt_customer_merge
from Application.mindbox.adapters._common import AdapterError, birth_date
from Application.mindbox.canonical_storage import atomic_json, storage_lock
from Application.mindbox.customers_stream import iter_json_records
from Application.mindbox.daily_training_batch import _writer_lock
from Application.mindbox.identity import CustomerIdResolver, CustomerIdentityError
from Application.mindbox.raw_reader import EXPORT_ROOTS, RawExportError, _unique_object
from Application.mindbox.selection import MindboxSelectionConfig
from Application.product_resolution import CatalogError, ProductResolver
from Application.statistics_facts import Projector, dump
from Application.statistics_index import BuildGuard, GIB, component_digest, manifest_digest, protect_directory, source_version
from Application.statistics_slices import _hash_file, _stat, capture_sources
from Application.statistics_storage_safety import (
    StatisticsIndexError, check_owned_staging, path_identity, reject_reparse, validate_storage_root,
)
from Application.statistics_unified_facts import normalize_order_observation
from Application.statistics_unified_policy import UNIFIED_PURCHASE_V1
from Application.statistics_unified_projection import (
    OBSERVATION_VERSION, PROFILE_SCHEMA, PROJECTION_ID, SCHEMA, SCHEMA_VERSION,
    ObservationProjector, ObservationReader, validate_layout,
)

POINTER_NAME = "current-unified-v1.json"
_COMPATIBILITY = {
    "projection_id": PROJECTION_ID,
    "observation_version": OBSERVATION_VERSION,
    "policy_id": UNIFIED_PURCHASE_V1,
}
_COMPONENTS = ("statistics.sqlite", "demographics.sqlite")
_CAPABILITIES = ("orders", "actions", "catalog", "profile_links")


def _noop():
    pass


def _inputs(raw_root, catalog_path):
    root, catalog = reject_reparse(raw_root), reject_reparse(catalog_path)
    if not reject_reparse(root / "canonical/catalog.json").is_file():
        raise StatisticsIndexError("SOURCE_UNAVAILABLE")
    for path in (root / "canonical/.writer.lock", root / "canonical/customers.sqlite",
                 catalog, catalog.with_name("site_categories.csv")):
        reject_reparse(path)
    return root, catalog


def _capture(root, catalog):
    """The source lock is owned by the caller."""
    try:
        source, _ = capture_sources(root, catalog, AnalysisFilter(), already_locked=True)
        for _, entry, parts in source["parts"]:
            for name, _ in parts:
                reject_reparse(root / entry["directory"] / name)
        package = Path(__file__).parent
        components = ("statistics_unified_index.py", "statistics_unified_projection.py",
                      "statistics_unified_facts.py", "statistics_unified_policy.py")
        return source_version(source) | {"unified_components": {name: _hash_file(package / name) for name in components}}
    except (OSError, ValueError, RawExportError, CatalogError, sqlite3.Error) as exc:
        if isinstance(exc, StatisticsIndexError):
            raise
        raise StatisticsIndexError("SOURCE_UNAVAILABLE") from None


def _records(root, provenance, name, check):
    for entry_index, (kind, entry, parts) in enumerate(provenance["parts"]):
        if kind != name:
            continue
        for part_index, (filename, signature) in enumerate(parts, 1):
            check()
            path = reject_reparse(root / entry["directory"] / filename)
            if list(_stat(path)) != signature:
                raise StatisticsIndexError("STALE_SOURCE")
            for ordinal, raw in enumerate(iter_json_records(path, EXPORT_ROOTS[name])):
                check()
                yield (entry_index, part_index, ordinal), raw
            if list(_stat(path)) != signature:
                raise StatisticsIndexError("STALE_SOURCE")


def _normalize(raw, source_ref, identities, products):
    return normalize_order_observation(raw, source_ref, identities,
                                       lambda key: products.resolve(key, strict=False).item_id)


def _identities(root, provenance, check):
    return CustomerIdResolver(adapt_customer_merge(raw)
                              for _, raw in _records(root, provenance, "customer_merges", check))


@contextmanager
def reference_observations(*, raw_root, catalog_path, check=_noop):
    """Read canonical Orders in source order; consume the iterator inside this context.

    Each source_ref addresses an effective manifest entry, part and record.
    This independent REFERENCE stream is normalized by CORE before any grouping.
    """
    root, catalog_path = _inputs(raw_root, catalog_path)
    with storage_lock(root):
        before = _capture(root, catalog_path)
        identities = _identities(root, before, check)
        products = ProductResolver(reference._catalog_snapshot(catalog_path)[0])
        stream = (_normalize(raw, ref, identities, products)
                  for ref, raw in _records(root, before, "orders", check))
        with closing(stream):
            yield stream
        if _capture(root, catalog_path) != before:
            raise StatisticsIndexError("STALE_SOURCE")


def _profile_links(root, target, auxiliary, guard):
    """Persist only private SID links, with no demographic values or raw IDs."""
    path = root / "canonical/customers.sqlite"
    if not path.exists():
        return 0
    count = 0
    with closing(_readonly(path)) as profiles:
        for ordinal, (source_id, valid) in enumerate(profiles.execute("SELECT id,json_valid(raw) FROM profiles ORDER BY id")):
            guard.check()
            if not valid:
                raise StatisticsIndexError("INVALID_PROFILE_SOURCE")
            target.execute("INSERT INTO profiles VALUES(?,?)", (ordinal, auxiliary.customer(source_id)))
            count += 1
            if count % 1000 == 0:
                target.commit()
                auxiliary.db.commit()
    return count


def _configure(db, guard, *, cache_kib):
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(f"PRAGMA cache_size=-{cache_kib}")
    # Sorting stays bounded by BuildGuard. No uncontrolled temp files on C:.
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    db.set_progress_handler(guard.sql_check, 10000)


def _project_orders(projector, root, provenance, identities, resolver, guard):
    count = 0
    for count, (source_ref, raw) in enumerate(_records(root, provenance, "orders", guard.check), 1):
        observation = _normalize(raw, source_ref, identities, resolver)
        projector.order(count, observation, raw.get("lines"))
        if count % 1000 == 0:
            guard.check(force=True)
            projector.db.commit()
    return count


def _populate(db, profiles, root, catalog_path, provenance, guard):
    identities = _identities(root, provenance, guard.check)
    products, items, names, paths = reference._catalog_snapshot(catalog_path)
    resolver = ProductResolver(products)
    selection = MindboxSelectionConfig(**provenance["canonical"]["selection"])
    auxiliary = Projector(db, identities, selection, resolver, ())
    projector = ObservationProjector(db, auxiliary, guard.check)
    db.executemany("INSERT INTO items VALUES(?,?,?)",
                   ((code, item.nomenclature_type, item.collection) for code, item in items.items()))
    db.executemany("INSERT INTO item_metadata VALUES(?,?,?,?,?)",
                   ((code, item.name, item.category, item.gender, item.style_group) for code, item in items.items()))
    db.executemany("INSERT INTO category_metadata VALUES(?,?,?)",
                   ((code, name, dump(paths.get(code))) for code, name in names.items()))
    count = 0
    for count, (_, raw) in enumerate(_records(root, provenance, "actions", guard.check), 1):
        auxiliary.action(raw, count)
        if count % 1000 == 0:
            guard.check(force=True)
            db.commit()
    counts = {"actions": count, "orders": _project_orders(projector, root, provenance, identities, resolver, guard)}
    counts["profiles"] = _profile_links(root, profiles, auxiliary, guard)
    projector.finish()
    for count, table in ((counts["orders"], "order_observations"), (counts["actions"], "actions")):
        if db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] != count:
            raise StatisticsIndexError("INCOMPLETE_PROJECTION")
    counts["order_lines"] = db.execute("SELECT COUNT(*) FROM order_line_observations").fetchone()[0]
    return counts


def _seal(db, binding, *, profile=False):
    validate_layout(db, profile=profile)
    db.execute("INSERT INTO metadata VALUES('binding',?)", (dump(binding),))
    db.execute("ANALYZE")
    db.commit()
    if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)] or db.execute("PRAGMA foreign_key_check").fetchone():
        raise StatisticsIndexError("INTEGRITY_FAILED")


def _build_components(folder, root, catalog_path, generation, guard):
    with closing(sqlite3.connect(folder / _COMPONENTS[0])) as db, closing(
            sqlite3.connect(folder / _COMPONENTS[1])) as profiles:
        _configure(db, guard, cache_kib=32768)
        _configure(profiles, guard, cache_kib=8192)
        db.executescript(SCHEMA)
        profiles.executescript(PROFILE_SCHEMA)
        with storage_lock(root):
            before = _capture(root, catalog_path)
            counts = _populate(db, profiles, root, catalog_path, before, guard)
            if _capture(root, catalog_path) != before:
                raise StatisticsIndexError("STALE_SOURCE")
        binding = {"schema": SCHEMA_VERSION, "generation": generation, "compatibility": _COMPATIBILITY,
                   "capabilities": list(_CAPABILITIES), "provenance": before, "counts": counts}
        _seal(db, binding)
        _seal(profiles, binding, profile=True)
    files = {}
    for name in _COMPONENTS:
        guard.check(force=True)
        path = folder / name
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
        files[name] = {"file": list(_stat(path))}
        if name == "demographics.sqlite":
            files[name]["sha256"] = component_digest(path, guard.check)
    return {"binding": binding, "files": files, "complete": True}


def _separate(directory, root):
    if directory == root or directory.is_relative_to(root / "canonical") or root.is_relative_to(directory):
        raise StatisticsIndexError("INDEX_MUST_BE_SEPARATE")


def _check_stage(directory, folder, root_id, folder_id):
    check_owned_staging(directory, folder, root_id, folder_id)
    allowed = {"manifest.json", *(name + suffix for name in _COMPONENTS
                                 for suffix in ("", "-journal", "-wal", "-shm"))}
    if any(path.name not in allowed for path in folder.iterdir()):
        raise StatisticsIndexError("UNEXPECTED_STAGING_CONTENT")


@contextmanager
def _staging(directory, root_id):
    if path_identity(reject_reparse(directory)) != root_id:
        raise StatisticsIndexError("STAGING_PATH_CHANGED")
    folder = directory / uuid.uuid4().hex
    folder.mkdir()
    folder_id = path_identity(folder)
    kept = False

    def retain():
        nonlocal kept
        kept = True

    try:
        protect_directory(folder)
        validate_storage_root(directory, children=(folder,))
        yield folder, folder_id, retain
    finally:
        if not kept:
            _check_stage(directory, folder, root_id, folder_id)
            shutil.rmtree(folder)


def _estimate(expected_bytes):
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError("Positive build size estimate required")


def build_unified_index(*, raw_root, catalog_path, index_dir, expected_bytes,
                        cancelled=None, observer=None, reserve_bytes=20 * GIB):
    """Explicit build into a prepared private root. No automatic build/fallback."""
    _estimate(expected_bytes)
    directory = reject_reparse(index_dir)
    root, catalog_path = _inputs(raw_root, catalog_path)
    _separate(directory, root)
    pointer = directory / POINTER_NAME
    existing = (pointer,) if os.path.lexists(pointer) else ()
    root_id = validate_storage_root(directory, children=existing)
    with _writer_lock(directory), _staging(directory, root_id) as (folder, folder_id, retain):
        guard = BuildGuard(folder, cancelled, observer, reserve_bytes)
        try:
            guard.check(force=True, extra=2 * expected_bytes)
            manifest = _build_components(folder, root, catalog_path, folder.name, guard)
            atomic_json(folder / "manifest.json", manifest)
            with storage_lock(root):
                if _capture(root, catalog_path) != manifest["binding"]["provenance"]:
                    raise StatisticsIndexError("STALE_SOURCE")
                guard.check(force=True)
                _check_stage(directory, folder, root_id, folder_id)
                existing = (pointer,) if os.path.lexists(pointer) else ()
                validate_storage_root(directory, children=(folder, *folder.iterdir(), *existing))
                _check_files(tuple(folder / name for name in _COMPONENTS), manifest)
                guard.check()
                atomic_json(pointer, {"schema": SCHEMA_VERSION, "generation": folder.name,
                                      "manifest_sha256": manifest_digest(manifest)})
                retain()
            return manifest
        except sqlite3.Error:
            if guard.failure is not None:
                raise guard.failure from None
            raise StatisticsIndexError("SQLITE_BUILD_ERROR") from None
        except InterruptedError:
            raise
        except (OSError, RawExportError, AdapterError, CustomerIdentityError, CatalogError):
            raise StatisticsIndexError("SOURCE_OR_BUILD_UNAVAILABLE") from None


def _restore_reference_row(db, row):
    # These dictionaries exist exclusively inside the private REFERENCE spool.
    def restore(value, table):
        if value.value is None:
            return value
        source, = db.execute(f"SELECT source FROM {table} WHERE sid=?", (int(value.value),)).fetchone()
        return replace(value, value=source)

    observation = row.observation
    return replace(row, observation=replace(
        observation, order_key=restore(observation.order_key, "order_ids"),
        customer_key=restore(observation.customer_key, "customer_ids"),
        lines=tuple(replace(line, stable_line_key=restore(line.stable_line_key, "line_ids"))
                    for line in observation.lines),
    ))


@contextmanager
def reference_order_reader(*, raw_root, catalog_path, scratch_dir, expected_bytes,
                           cancelled=None, observer=None, reserve_bytes=20 * GIB):
    """Group the canonical stream in a private, disposable SQLite spool on the caller's volume."""
    _estimate(expected_bytes)
    root, catalog_path = _inputs(raw_root, catalog_path)
    directory = reject_reparse(scratch_dir)
    _separate(directory, root)
    root_id = validate_storage_root(directory)
    with _staging(directory, root_id) as (folder, _, _), storage_lock(root):
        guard = BuildGuard(folder, cancelled, observer, reserve_bytes)
        guard.check(force=True, extra=2 * expected_bytes)
        try:
            with closing(sqlite3.connect(folder / _COMPONENTS[0])) as db:
                _configure(db, guard, cache_kib=32768)
                db.executescript(SCHEMA)
                before = _capture(root, catalog_path)
                identities = _identities(root, before, guard.check)
                resolver = ProductResolver(reference._catalog_snapshot(catalog_path)[0])
                selection = MindboxSelectionConfig(**before["canonical"]["selection"])
                projector = ObservationProjector(db, Projector(db, identities, selection, resolver, ()), guard.check)
                _project_orders(projector, root, before, identities, resolver, guard)
                projector.prepare_read()
                reader = ObservationReader(db, {"binding": {"generation": folder.name, "provenance": before}}, guard.check,
                                           restore=lambda row: _restore_reference_row(db, row))
                reader.identity_space = "REFERENCE:" + folder.name
                yield reader
                guard.check()
                if _capture(root, catalog_path) != before:
                    raise StatisticsIndexError("STALE_SOURCE")
        except sqlite3.Error:
            if guard.failure is not None:
                raise guard.failure from None
            raise StatisticsIndexError("REFERENCE_SPOOL_ERROR") from None
        except InterruptedError:
            raise
        except (OSError, RawExportError, AdapterError, CustomerIdentityError, CatalogError):
            raise StatisticsIndexError("SOURCE_OR_BUILD_UNAVAILABLE") from None


@contextmanager
def reference_facts(*, raw_root, catalog_path, scratch_dir, expected_bytes, product_predicate=None,
                    cancelled=None, observer=None, reserve_bytes=20 * GIB):
    with reference_order_reader(raw_root=raw_root, catalog_path=catalog_path, scratch_dir=scratch_dir,
                                expected_bytes=expected_bytes, cancelled=cancelled, observer=observer,
                                reserve_bytes=reserve_bytes) as reader:
        with closing(reader.facts(product_predicate=product_predicate)) as facts:
            yield facts


def _read_json(path, maximum):
    reject_reparse(path)
    if path.stat().st_size > maximum:
        raise StatisticsIndexError("INVALID_MANIFEST")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def _check_sidecars(path):
    for suffix in ("-journal", "-wal", "-shm"):
        if os.path.lexists(str(path) + suffix):
            raise StatisticsIndexError("UNEXPECTED_SQLITE_SIDECAR")


def _readonly(path):
    path = reject_reparse(path)
    _check_sidecars(path)
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.2)
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("PRAGMA cache_size=-32768")
        db.execute("PRAGMA temp_store=MEMORY")
        return db
    except Exception:
        db.close()
        raise


def _open_manifest(directory, check):
    pointer_path = directory / POINTER_NAME
    if not os.path.lexists(pointer_path):
        raise StatisticsIndexError("INDEX_MISSING")
    validate_storage_root(directory, children=(pointer_path,))
    pointer = _read_json(pointer_path, 1024 * 1024)
    if (not isinstance(pointer, dict) or set(pointer) != {"schema", "generation", "manifest_sha256"}
            or type(pointer["schema"]) is not int or pointer["schema"] != SCHEMA_VERSION
            or not re.fullmatch(r"[0-9a-f]{32}", pointer["generation"])
            or not re.fullmatch(r"[0-9a-f]{64}", pointer["manifest_sha256"])):
        raise StatisticsIndexError("INCOMPATIBLE_SCHEMA")
    folder = directory / pointer["generation"]
    manifest_path = folder / "manifest.json"
    components = tuple(folder / name for name in _COMPONENTS)
    validate_storage_root(directory, children=(pointer_path, folder, manifest_path, *components))
    manifest = _read_json(manifest_path, 4 * 1024**2)
    if not isinstance(manifest, dict) or set(manifest) != {"binding", "files", "complete"}:
        raise StatisticsIndexError("INVALID_MANIFEST")
    binding = manifest["binding"]
    if (not isinstance(binding, dict) or set(binding) != {
            "schema", "generation", "compatibility", "capabilities", "provenance", "counts"}
            or not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(_COMPONENTS)):
        raise StatisticsIndexError("INVALID_MANIFEST")
    if (manifest_digest(manifest) != pointer["manifest_sha256"] or manifest["complete"] is not True
            or binding["generation"] != pointer["generation"]):
        raise StatisticsIndexError("INVALID_MANIFEST")
    if (type(binding["schema"]) is not int or binding["schema"] != SCHEMA_VERSION
            or binding["compatibility"] != _COMPATIBILITY or binding["capabilities"] != list(_CAPABILITIES)):
        raise StatisticsIndexError("INCOMPATIBLE_SCHEMA")
    _check_files(components, manifest)
    profile = components[1]
    if component_digest(profile, check) != manifest["files"][profile.name]["sha256"]:
        raise StatisticsIndexError("INVALID_DEMOGRAPHICS")
    identities = path_identity(directory), path_identity(folder)
    return manifest, components, identities


def _check_files(components, manifest):
    for path in components:
        reject_reparse(path)
        _check_sidecars(path)
        if list(_stat(path)) != manifest["files"][path.name]["file"]:
            raise StatisticsIndexError("GENERATION_CHANGED")


def _check_binding(db, binding, *, schema="main"):
    version = db.execute(f"PRAGMA {schema}.user_version").fetchone()[0]
    row = db.execute(f"SELECT value FROM {schema}.metadata WHERE key='binding'").fetchone()
    if version != SCHEMA_VERSION or row is None or json.loads(row[0]) != binding:
        raise StatisticsIndexError("INCOMPATIBLE_SCHEMA")


def _check_open_generation(directory, components, manifest, path_ids):
    current = path_identity(reject_reparse(directory)), path_identity(reject_reparse(components[0].parent))
    if current != path_ids:
        raise StatisticsIndexError("GENERATION_CHANGED")
    _check_files(components, manifest)


def _read_profile_values(root, catalog_path, provenance, links, check):
    with storage_lock(root):
        if _capture(root, catalog_path) != provenance:
            raise StatisticsIndexError("STALE_SOURCE")
        if not provenance["customers"]["present"]:
            if next(iter(links), None) is not None:
                raise StatisticsIndexError("INVALID_PROFILE_LINKS")
            return
        path = root / "canonical/customers.sqlite"
        try:
            with closing(_readonly(path)) as source:
                values = source.execute("SELECT json_extract(raw,'$.birthDate'),json_extract(raw,'$.sex') "
                                        "FROM profiles ORDER BY id")
                for ordinal, ((linked, sid), (raw_date, sex)) in enumerate(zip(links, values, strict=True)):
                    check()
                    if ordinal != linked or type(sid) is not int or sid < 1:
                        raise StatisticsIndexError("INVALID_PROFILE_LINKS")
                    try:
                        born = birth_date({"birthDate": raw_date})
                    except AdapterError:
                        born = None
                    yield str(sid), born, sex if sex in ("male", "female") else None
        except (sqlite3.Error, ValueError) as exc:
            if isinstance(exc, StatisticsIndexError):
                raise
            raise StatisticsIndexError("INVALID_PROFILE_SOURCE") from None
        if _capture(root, catalog_path) != provenance:
            raise StatisticsIndexError("STALE_SOURCE")


@contextmanager
def open_unified_index(*, raw_root, catalog_path, index_dir, check=_noop):
    """Open only schema 4. The caller consumes facts before publishing its result."""
    root, catalog_path = _inputs(raw_root, catalog_path)
    directory = reject_reparse(index_dir)
    try:
        manifest, components, path_ids = _open_manifest(directory, check)
        binding = manifest["binding"]
        with storage_lock(root):
            if _capture(root, catalog_path) != binding["provenance"]:
                raise StatisticsIndexError("STALE_SOURCE")
        with closing(_readonly(components[0])) as db, closing(_readonly(components[1])) as profiles:
            validate_layout(db)
            validate_layout(profiles, profile=True)
            _check_binding(db, binding)
            _check_binding(profiles, binding)
            db.execute("ATTACH DATABASE ? AS demographics", (components[1].as_uri() + "?mode=ro",))
            _check_binding(db, binding, schema="demographics")
            _check_open_generation(directory, components, manifest, path_ids)
            yield ObservationReader(
                db, manifest, check,
                profile_values=lambda links: _read_profile_values(root, catalog_path, binding["provenance"], links, check),
            )
            check()
            _check_open_generation(directory, components, manifest, path_ids)
            with storage_lock(root):
                if _capture(root, catalog_path) != binding["provenance"]:
                    raise StatisticsIndexError("STALE_SOURCE")
    except (StatisticsIndexError, InterruptedError):
        raise
    except (OSError, sqlite3.Error, UnicodeError, KeyError, ValueError, TypeError):
        raise StatisticsIndexError("UNIFIED_INDEX_UNAVAILABLE") from None
