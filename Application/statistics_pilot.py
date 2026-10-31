"""Explicit isolated real-data pilot infrastructure; never the production root.

Copy complete effective API components, not random rows or manual file fragments.
The source writer lock protects copies against canonical GC. Profiles are a
minimal, indexed selection of referenced canonical customers and ALL their aliases.
All row data stays in a newly created ACL-protected directory. Public outputs are
counts only. The normal statistics builder/query/validation remain unchanged.
"""

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
import uuid

from Application import dataset_statistics as reference
from Application.analysis_filter import AnalysisFilter
from Application.mindbox.adapters import adapt_customer_merge
from Application.mindbox.adapters._common import AdapterError, identifier
from Application.mindbox.canonical_storage import atomic_json, storage_lock
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.raw_reader import iter_export, part_files
from Application.statistics_index import (
    DEFAULT_INDEX_DIR, BuildGuard, GIB, StatisticsIndexError,
    build_statistics_index, protect_directory,
)
from Application.statistics_slices import DEFAULT_CACHE_DIR, _digest, capture_sources
from Application.statistics_index import source_version
from Application.statistics_storage_safety import path_identity, reject_reparse


@dataclass(frozen=True)
class PilotLimits:
    interaction_bytes: int = 128*1024**2
    merges_bytes: int = 200*1024**2
    action_records: int = 250000
    order_records: int = 20000
    profile_candidates: int = 100000
    days: int = 7


def _forbidden(destination, source):
    for path in (Path(source)/"canonical", DEFAULT_INDEX_DIR, DEFAULT_CACHE_DIR):
        path = path.resolve()
        if destination == path or destination.is_relative_to(path) or path.is_relative_to(destination):
            return True
    return destination == Path(source).resolve()


def plan_pilot(*, raw_root, catalog_path, days, limits=PilotLimits()):
    days = tuple(days)
    if not days or len(set(days)) != len(days) or len(days) > limits.days:
        raise StatisticsIndexError("PILOT_INVALID_DAYS")
    from datetime import date, datetime, time, timedelta, timezone
    ordered = sorted(date.fromisoformat(day) for day in days)
    if ordered != [ordered[0]+timedelta(days=n) for n in range(len(ordered))]:
        raise StatisticsIndexError("PILOT_NONCONTIGUOUS_DAYS")
    snapshot, _ = capture_sources(raw_root, catalog_path, AnalysisFilter())
    selected, total, merges_total = {"actions": [], "orders": []}, 0, 0
    for name, entry, files in snapshot["parts"]:
        size = sum(file[1][1] for file in files)
        if name == "customer_merges":
            merges_total += size
        elif entry["source_kind"] == "API" and entry["since"][:10] in days:
            selected[name].append(entry)
            total += size
    for name, entries in selected.items():
        if {e["since"][:10] for e in entries} != set(days) or len(entries) != len(days):
            raise StatisticsIndexError("PILOT_COMPLETE_PAIRED_API_DAYS_REQUIRED")
    for day in ordered:
        first = datetime.combine(day, time(), timezone.utc)
        last = first+timedelta(days=1)
        for entries in selected.values():
            entry = next(e for e in entries if e["since"][:10] == day.isoformat())
            if datetime.fromisoformat(entry["since"]) != first or datetime.fromisoformat(entry["until"]) != last:
                raise StatisticsIndexError("PILOT_COMPLETE_PAIRED_API_DAYS_REQUIRED")
    if (total > limits.interaction_bytes or merges_total > limits.merges_bytes
            or not snapshot["canonical"]["customer_merges"] or not snapshot["customers"]["present"]):
        raise StatisticsIndexError("PILOT_SOURCE_BUDGET_OR_READINESS")
    return {"schema": 1, "scope": "TEST_ONLY", "source": snapshot,
            "selected": selected, "days": sorted(days), "interaction_bytes": total, "merges_bytes": merges_total}


def prepare_pilot(*, plan, raw_root, catalog_path, destination, limits=PilotLimits(),
                  cancelled=None, observer=None):
    """Create a complete source fixture in a fresh private directory. No build."""
    root, destination = Path(raw_root).resolve(), Path(destination)
    if destination.is_symlink() or _forbidden(destination.resolve(), root) or destination.exists():
        raise StatisticsIndexError("PILOT_UNSAFE_DESTINATION")
    if plan.get("scope") != "TEST_ONLY" or plan.get("schema") != 1:
        raise StatisticsIndexError("PILOT_MANIFEST_INCOMPATIBLE")
    # Reconstruct the bounded selection; caller cannot inject directories/budgets.
    current = plan_pilot(raw_root=root, catalog_path=catalog_path, days=plan["days"], limits=limits)
    if current != plan:
        raise StatisticsIndexError("PILOT_STALE_PLAN")
    destination.mkdir()
    destination = destination.resolve()
    parent_id, destination_id = path_identity(destination.parent), path_identity(destination)
    started = perf_counter()
    timings, counts = {}, {}
    try:
        protect_directory(destination)
        guard = BuildGuard(destination, cancelled, observer, 20*GIB)
        guard.check(force=True, extra=2*GIB)
        source_root = destination/"sources"
        source_root.mkdir()
        with storage_lock(root):
            captured, _ = capture_sources(root, catalog_path, AnalysisFilter(), already_locked=True)
            if source_version(captured) != source_version(plan["source"]):
                raise StatisticsIndexError("PILOT_STALE_PLAN")
            data = captured["canonical"]
            copy_started = perf_counter()
            copied = []
            for name, entries in (*plan["selected"].items(), ("customer_merges", [data["customer_merges"]])):
                for entry in entries:
                    physical = reference._source_directory(root, name, entry)
                    target = source_root/entry["directory"]
                    target.mkdir(parents=True)
                    for path in part_files(physical, name):
                        digest = hashlib.sha256()
                        with path.open("rb") as original, (target/path.name).open("xb") as output:
                            for block in iter(lambda: original.read(1024**2), b""):
                                guard.check()
                                output.write(block)
                                digest.update(block)
                        copied.append({"source": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
            refs = destination/"references"
            refs.mkdir()
            for path in (Path(catalog_path), Path(catalog_path).with_name("site_categories.csv")):
                shutil.copyfile(path, refs/path.name)
            timings["copy_seconds"] = perf_counter()-copy_started
            # Full graph validation, not an order-dependent sampled resolver.
            merge_started = perf_counter()
            merge_ids = set()

            def merges():
                count = 0
                for raw in iter_export("customer_merges", input_dir=source_root/data["customer_merges"]["directory"]):
                    guard.check()
                    merged = adapt_customer_merge(raw)
                    merge_ids.add(merged.resulting_customer_id)
                    merge_ids.update(merged.merged_customer_ids)
                    count += 1
                    yield merged
                counts["customer_merges"] = count

            identities = CustomerIdResolver(merges())
            timings["identity_seconds"] = perf_counter()-merge_started
            observed = set()
            for name, entries in plan["selected"].items():
                scan_started = perf_counter()
                count = 0
                bound = limits.action_records if name == "actions" else limits.order_records
                for entry in entries:
                    for raw in iter_export(name, input_dir=source_root/entry["directory"]):
                        guard.check()
                        count += 1
                        if count > bound:
                            raise StatisticsIndexError("PILOT_RECORD_LIMIT_EXCEEDED")
                        try:
                            customer = identifier(raw, "customer.ids.mindboxId", required=False)
                        except AdapterError:
                            customer = None  # Preserve the complete raw record and its deferred error.
                        if customer is not None:
                            observed.add(customer)
                counts[name] = count
                timings[f"{name}_scope_scan_seconds"] = perf_counter()-scan_started
            canonical = {identities.resolve(source) for source in observed}
            candidates = observed | canonical
            for alias in merge_ids:
                guard.check()
                if identities.resolve(alias) in canonical:
                    candidates.add(alias)
            if len(candidates) > limits.profile_candidates:
                raise StatisticsIndexError("PILOT_PROFILE_LIMIT_EXCEEDED")
            counts.update(referenced_canonical_customers=len(canonical), profile_candidate_ids=len(candidates))
            profile_started = perf_counter()
            with closing(sqlite3.connect(reference.database(root).as_uri()+"?mode=ro", uri=True)) as source_db:
                with closing(sqlite3.connect(source_root/"canonical/customers.sqlite")) as target_db:
                    target_db.executescript("CREATE TABLE profiles(id TEXT PRIMARY KEY, changed TEXT, raw TEXT NOT NULL); "
                                            "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);")
                    total_profiles = 0
                    candidate_list = sorted(candidates)
                    for offset in range(0, len(candidate_list), 400):
                        guard.check()
                        batch = candidate_list[offset:offset+400]
                        slots = ",".join("?" for _ in batch)
                        # No real profiles scan: probes the existing PRIMARY KEY.
                        for sid, birthday, sex in source_db.execute(
                                f"SELECT id,json_extract(raw,'$.birthDate'),json_extract(raw,'$.sex') FROM profiles WHERE id IN ({slots})", batch):
                            birthday = birthday if isinstance(birthday, str) else None
                            sex = sex if sex in ("male", "female") else None
                            raw = json.dumps({"birthDate": birthday, "sex": sex}, ensure_ascii=True)
                            target_db.execute("INSERT INTO profiles VALUES(?,'',?)", (sid, raw))
                            total_profiles += 1
                        target_db.commit()
                    summary = json.loads(captured["customers"]["summary"] or "{}")
                    summary.update(count=total_profiles, pilot_scope="REFERENCED_CANONICAL_AND_ALL_ALIASES_ONLY")
                    target_db.execute("INSERT INTO metadata VALUES('summary',?)", (json.dumps(summary),))
                    target_db.commit()
                    counts["profiles"] = total_profiles
            timings["customer_probes_seconds"] = perf_counter()-profile_started
            after, _ = capture_sources(root, catalog_path, AnalysisFilter(), already_locked=True)
            if source_version(after) != source_version(captured):
                raise StatisticsIndexError("PILOT_SOURCE_CHANGED")
            pilot_catalog = dict(data, revision="PILOT-"+uuid.uuid4().hex, manual_interactions=None,
                actions={e["since"][:10]:e for e in plan["selected"]["actions"]},
                orders={e["since"][:10]:e for e in plan["selected"]["orders"]})
            atomic_json(source_root/"canonical/catalog.json", pilot_catalog)
        local, _ = capture_sources(source_root, refs/Path(catalog_path).name, AnalysisFilter())
        marker = {"schema":2, "scope":"TEST_ONLY", "source_digest":_digest(source_version(local)), "counts":counts,
                  "days":plan["days"], "copied":copied, "timings":timings,
                  "elapsed_seconds":perf_counter()-started}
        atomic_json(destination/"PILOT_ONLY.json", marker)
        return marker
    except Exception:
        # Only the directory exclusively created by this call, never user files.
        try:
            reject_reparse(destination)
            if (path_identity(destination.parent) != parent_id
                    or path_identity(destination) != destination_id):
                raise StatisticsIndexError("PILOT_CLEANUP_PATH_CHANGED")
        except (OSError, StatisticsIndexError):
            raise StatisticsIndexError("PILOT_CLEANUP_PATH_CHANGED") from None
        shutil.rmtree(destination)
        raise


def pilot_paths(destination):
    folder = Path(destination).resolve()
    try:
        marker = json.loads((folder/"PILOT_ONLY.json").read_text(encoding="utf-8"))
        if marker.get("schema") != 2 or marker.get("scope") != "TEST_ONLY":
            raise ValueError
        source, reference_path = folder/"sources", folder/"references/nomenclature.csv"
        current, _ = capture_sources(source, reference_path, AnalysisFilter())
        if _digest(source_version(current)) != marker["source_digest"]:
            raise StatisticsIndexError("PILOT_SOURCE_CHANGED")
        if (any(folder == p.resolve() or folder.is_relative_to(p.resolve()) or p.resolve().is_relative_to(folder)
                for p in (DEFAULT_INDEX_DIR, DEFAULT_CACHE_DIR))
                or source.is_symlink() or reference_path.is_symlink()
                or not source.resolve().is_relative_to(folder) or not reference_path.resolve().is_relative_to(folder)):
            raise ValueError
        return {"raw_root":source, "catalog_path":reference_path, "index_dir":folder/"index"}
    except StatisticsIndexError:
        raise
    except (OSError, ValueError, TypeError, AttributeError):
        raise StatisticsIndexError("PILOT_MANIFEST_INCOMPATIBLE") from None


def build_pilot(destination, *, cancelled=None, progress=None, observer=None):
    arguments = pilot_paths(destination)
    # A single attempt; an error/cancel never triggers another build automatically.
    with (Path(destination)/"build-attempted").open("xb"):
        pass
    return build_statistics_index(**arguments, cancelled=cancelled, progress=progress,
                                  observer=observer, expected_bytes=512*1024**2)
