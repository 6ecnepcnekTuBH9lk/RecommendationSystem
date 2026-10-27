"""One bounded EXPLORATORY pilot; indexed canonical IDs, no CustomerMerges parsing.

Parent enforces wall time, Windows Job Object enforces private commit, existing
project ResourceGuard checks system RAM/commit and 20-GiB disk reserves. All
customer data stay in worker RAM; stdout contains aggregate JSON only.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRIVATE_LIMIT = 1536 * 2**20
WALL_SECONDS = 180
OUTPUT_LIMIT = 4 * 2**20


class RunnerError(ValueError):
    """Static early enforcement errors, before heavy research imports."""


def limit_process_memory():
    """Hard Windows private-commit limit, independent of cooperative Python checks."""
    if os.name != "nt":
        raise RunnerError("WINDOWS_JOB_REQUIRED")
    import ctypes
    from ctypes import wintypes
    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", wintypes.DWORD), ("minimum_working_set", ctypes.c_size_t),
                    ("maximum_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(f"value_{n}", ctypes.c_ulonglong) for n in range(6)]
    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t),
                    ("peak_job_memory", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise RunnerError("JOB_CREATION_FAILED")
    limits = Extended()
    limits.basic.flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY (private commit).
    limits.process_memory = PRIVATE_LIMIT
    if (not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())):
        raise RunnerError("JOB_LIMIT_UNAVAILABLE")
    return kernel, job  # Keep the handle alive for the whole worker lifetime.


def index_preflight(service, root):
    from Application import customer_history_index as index
    from Application.evaluation.experiments.customer_interest_index_pilot import PilotError, require_ready
    ready = service.get_status(raw_root=root, verify_integrity=False)
    require_ready(ready)
    path = service.index_path(raw_root=root)
    receipt_root = ROOT / "user_settings/_history_real_index02"
    integrity = json.loads((receipt_root / "integrity_result.json").read_text(encoding="utf-8"))
    final = json.loads((receipt_root / "final_checks.json").read_text(encoding="utf-8"))
    build = json.loads((receipt_root / "build_report.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "filtered_data/.customer-interest-profile-02-20261026/BASELINE.json").read_text(encoding="utf-8"))
    previous = old["protected_stat"].get(str(path.relative_to(ROOT)))
    same_file = previous == [path.stat().st_size, path.stat().st_mtime_ns]
    checks = {"integrity_receipt_ready": integrity.get("integrity_status") == "READY",
              "integrity_generation_matches": integrity.get("generation") == ready.generation,
              "verified_source_matches": final.get("source_version") == ready.source_version,
              "verified_generation_matches": final.get("generation") == ready.generation,
              # final_checks.sqlite_index_bytes means secondary-index pages, NOT
              # the database file length. The publication receipt owns file size.
              "verified_size_matches": build.get("index_bytes") == path.stat().st_size,
              "published_generation_matches": build.get("generation") == ready.generation,
              "published_source_matches": build.get("source_version") == ready.source_version,
              "index_unchanged_since_previous_pilot": same_file,
              "schema_v2": final.get("schema_version") == index.INDEX_SCHEMA_VERSION == 2}
    if not all(checks.values()):
        raise PilotError("PUBLISHED_INTEGRITY_EVIDENCE_MISMATCH:" + ",".join(k for k, valid in checks.items() if not valid))
    return ready, {"status": ready.status.value, "checks": checks, "verify_integrity_executed": False,
                   "coverage": [asdict(c) for c in ready.coverage], "bytes": path.stat().st_size}


def worker(output, *, smoke=False):
    job = limit_process_memory()  # Before Torch/model imports; never loads checkpoint tensors.
    if smoke:
        return {"job_memory_enforced": bool(job), "private_limit_bytes": PRIVATE_LIMIT}
    from Application import customer_history_index as index
    from Application.evaluation.experiments.customer_interest import ResearchContext
    from Application.evaluation.experiments.customer_interest_index_pilot import (
        Bounds, Budget, PilotError, convert_events, evaluate_profiles, open_reader, read_customer, select_customers,
    )
    from Application.model.propensity_ml.resources import ResourceGuard
    from Application.model.propensity_ml.contracts import MLDataError, MLCancelled
    from Application.product_resolution import CatalogDiagnostics, ProductCatalog, ProductResolver, ResolutionStatus
    from Application.recommendation_catalog import RecommendationCatalog
    limits = Bounds()
    cancel = lambda: (output / "CANCEL").exists()
    resource = ResourceGuard((output,), cancelled=cancel, private_limit=PRIVATE_LIMIT)
    def poll():
        try:
            resource.poll(force=True)
        except (MLDataError, MLCancelled) as exc:
            raise PilotError("CANCELLED" if isinstance(exc, MLCancelled) else str(exc)) from None
        if resource.peak_working_set > PRIVATE_LIMIT:
            raise PilotError("WORKING_SET_LIMIT")
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > limits.output_bytes:
            raise PilotError("OUTPUT_LIMIT")
    budget = Budget(limits, poll=poll, cancelled=cancel)
    budget.check()
    service = index.CustomerHistoryIndexService()
    ready, preflight = index_preflight(service, index.DEFAULT_RAW_ROOT)
    paths = [ROOT / "input_data" / name for name in ("nomenclature.csv", "site_categories.csv")]
    if any(p.stat().st_size > 16 * 2**20 for p in paths):
        raise PilotError("CATALOG_INPUT_LIMIT")
    catalog = RecommendationCatalog(*paths).load()
    if len(catalog.all_codes) > 50_000:
        raise PilotError("CATALOG_CODES_LIMIT")
    resolver = ProductResolver(ProductCatalog(frozenset(catalog.items), CatalogDiagnostics()))
    cutoff = min(max(c.until_utc for c in ready.coverage if c.component == name) for name in ("actions", "orders"))
    since = cutoff - timedelta(days=365)
    knowledge = datetime.now(timezone.utc)
    context = ResearchContext(cutoff, datetime.fromtimestamp(paths[0].stat().st_mtime, timezone.utc),
                              mode="EXPLORATORY", dataset_kind="INDEX_EXPLORATORY", knowledge_cutoff=knowledge)
    cohort, diagnostics, strata, channels, observed_codes = [], Counter(), Counter(), Counter(), set()
    panel_codes = tuple(random.Random(20261026).sample(sorted(catalog.items), 256))
    with open_reader(service, index.DEFAULT_RAW_ROOT, budget) as (connection, same):
        # Verify the actual sampling query uses a covering B-tree seek.
        from Application.evaluation.experiments.customer_interest_index_pilot import SEEK_SQL
        plan = [r[-1] for r in connection.execute("EXPLAIN QUERY PLAN " + SEEK_SQL, ("500000",))]
        if not any("SEARCH" in detail and "event_customers_customer" in detail for detail in plan):
            raise PilotError("SAMPLING_QUERY_NOT_INDEXED")
        for customer in select_customers(connection, budget):
            events, read_diagnostics = read_customer(connection, same, customer, since, cutoff, budget)
            diagnostics.update(read_diagnostics)
            if events is None:
                diagnostics["UNUSABLE_CUSTOMER_HISTORY"] += 1
                if read_diagnostics.get("EVENT_BUDGET_EXHAUSTED"):
                    break
                continue
            n = len(events)
            strata["0" if n == 0 else "1-5" if n <= 5 else "6-30" if n <= 30 else "31-200"] += 1
            for logical in events:
                channels[logical.kind.value if logical.kind else "UNKNOWN"] += 1
            adapted, detail = convert_events(customer, events, resolver)
            diagnostics.update(detail)
            partial = bool(read_diagnostics.get("partial_conflict_pages") or any(detail.get(k) for k in (
                "OWNER_CONFLICT", "TIME_OR_KIND_CONFLICT", "ACTION_VARIANTS_CONFLICT", "MISSING_ACTION_PRODUCT",
                "MISSING_PUBLICATION", "ORDER_UNRESOLVED", "ORDER_UNCERTAIN_CODES")))
            cohort.append((customer, adapted, n, partial))
            for logical in events:
                for variant in logical.variants:
                    details = variant.event.details
                    occurrences = details.products if logical.source.value == "ACTION" else details.lines
                    for occurrence in occurrences:
                        product = occurrence.reference.product
                        if product is not None:
                            match = resolver.resolve(product, strict=False)
                            if match.status is ResolutionStatus.RESOLVED:
                                observed_codes.add(match.item_id)
            if len(observed_codes | set(panel_codes)) > limits.max_catalog_items:
                raise PilotError("PROFILE_CATALOG_LIMIT")
        preflight["sampling_query_plan"] = plan
    budget.check()
    selected_catalog = tuple(catalog.items[code] for code in sorted(observed_codes | set(panel_codes)))
    evaluation = evaluate_profiles(cohort, selected_catalog, context, budget, panel_codes=panel_codes)
    after = service.get_status(raw_root=index.DEFAULT_RAW_ROOT, verify_integrity=False)
    if after.status is not index.HistoryIndexStatus.READY or after.generation != ready.generation or after.source_version != ready.source_version:
        raise PilotError("SOURCES_CHANGED")
    budget.check()
    return {"mode": "EXPLORATORY", "status": "COMPLETED", "preflight": preflight,
            "window": [since.isoformat(), cutoff.isoformat()], "knowledge_cutoff": knowledge.isoformat(),
            "sampling": "seeded random numeric lexicographic seeks, gap-weighted, not representative",
            "seed": 20261026, "buyers_probed": budget.customers, "complete_bounded_histories": len(cohort),
            "source_activity_strata": dict(strata), "channels": dict(channels), "diagnostics": dict(diagnostics),
            "event_rows_read_including_lookahead": budget.events, "catalog_items_in_profile_scope": len(selected_catalog),
            "observed_resolved_products": len(observed_codes), "catalog_diagnostics": catalog.diagnostics,
            "evaluation": evaluation, "resources": resource.payload(), "limits": asdict(limits),
            "hard_process_memory_limit_bytes": PRIVATE_LIMIT, "temporary_customer_files_created": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--limits-smoke", action="store_true")
    parser.add_argument("--audit-references", action="store_true", help="Diagnose the unchanged profile-03 cohort; no automatic repair")
    args = parser.parse_args(argv)
    if args.audit_references and args.limits_smoke:
        parser.error("Reference audit and limits-only smoke cannot be combined")
    output = args.output.resolve()
    if ROOT / "filtered_data" not in output.parents:
        parser.error("Use an owned research directory under filtered_data")
    if args._worker:
        try:
            if args.audit_references:
                enforcement = limit_process_memory()
                from Application.evaluation.experiments.customer_interest_reference_audit import audit_worker
                payload = audit_worker(output)
                payload["os_job_enforced"] = bool(enforcement)
                payload["hard_process_memory_limit_bytes"] = PRIVATE_LIMIT
            else:
                payload = worker(output, smoke=args.limits_smoke)
        except BaseException as exc:
            # Never print source/SQL/OS exception messages or traceback payloads.
            safe = isinstance(exc, RunnerError) or (type(exc).__name__ == "PilotError" and
                type(exc).__module__ == "Application.evaluation.experiments.customer_interest_index_pilot")
            reason = str(exc) if safe else type(exc).__name__
            payload = {"status": "BLOCKED_OR_CANCELLED", "reason": reason}
        encoded = json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False)
        if len(encoded.encode("utf-8")) > 512 * 2**10:
            encoded = json.dumps({"status": "BLOCKED", "reason": "RESULT_SIZE_LIMIT"})
        print(encoded, flush=True)
        return 0
    if not output.is_dir() or sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > OUTPUT_LIMIT:
        parser.error("Research directory missing or above output budget")
    receipt_name = "AUDIT.json" if args.audit_references else "PILOT.json"
    if (output / receipt_name).exists() and not args.limits_smoke:
        parser.error("A pilot attempt already exists; no automatic retry")
    command = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--output", str(output), "--_worker"]
    if args.limits_smoke:
        command.append("--limits-smoke")
    if args.audit_references:
        command.append("--audit-references")
    started = monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=ROOT)
    try:
        stdout, _ = process.communicate(timeout=WALL_SECONDS)
        result = json.loads(stdout) if process.returncode == 0 else {"status": "BLOCKED", "reason": "WORKER_FAILED"}
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        result = {"status": "BLOCKED", "reason": "HARD_WALL_TIME_LIMIT"}
    except KeyboardInterrupt:
        process.terminate()
        process.communicate()
        result = {"status": "CANCELLED", "reason": "MANUAL_INTERRUPT"}
    result["parent_wall_seconds"] = monotonic() - started
    if not args.limits_smoke:
        with (output / receipt_name).open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result if args.limits_smoke else {k: result.get(k) for k in (
        "status", "reason", "buyers_probed", "complete_bounded_histories", "event_rows_read_including_lookahead",
        "parent_wall_seconds", "resources")}, ensure_ascii=False))
    return 0 if result.get("status") == "COMPLETED" or result.get("job_memory_enforced") else 1


if __name__ == "__main__":
    raise SystemExit(main())
