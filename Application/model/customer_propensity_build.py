"""Explicit, cancellable research dataset build and atomic local publication."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from itertools import groupby
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
from typing import Callable
import uuid

from Application import customer_history_index as index
from Application import customer_activity_service, product_resolution, purchased_products
from Application.mindbox.daily_training_batch import _writer_lock
from Application.mindbox import selection
from Application.model.customer_propensity_dataset import (
    AS_OF_LEVEL, HORIZON, LIMITATIONS, SCHEMA_VERSION, TARGET, PropensityDatasetConfig,
    DAY, PropensityPlan, PropensitySourceSpan, utc_day,
    covered, customer_rows, feature_schema, iso, latest_prediction_date, make_plan,
)
from Application.model.customer_propensity_history import PropensityCancelled, PropensityDataError, iter_facts
from Application.model.customer_propensity_storage import (
    FILES, PropensityStorage, digest as _digest, iter_validated_file, open_writer,
)
from Application.paths import PROJECT_ROOT


DEFAULT_OUTPUT = PROJECT_ROOT / "filtered_data" / "customer_propensity"


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _config(config: PropensityDatasetConfig) -> dict:
    return {k: iso(v) if isinstance(v, datetime) else v for k, v in asdict(config).items()}


@dataclass(frozen=True, slots=True)
class PropensityProgress:
    phase: str
    logical_event_memberships: int
    customers: int
    rows: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PropensityDatasetSummary:
    status: str
    version: str
    artifact_dir: Path
    counts: dict
    elapsed_seconds: float
    reused: bool = False


def _remove_staging(path: Path, parent: Path) -> None:
    # Only this build's UUID staging directory; verify absolute containment before rmtree.
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or not path.name.startswith(".building-") or path.is_symlink():
        raise PropensityDataError("UNSAFE_STAGING_PATH")
    if path.exists():
        shutil.rmtree(path)


def _validate_rows(directory: Path, expected: dict, cancelled: Callable[[], bool], plan: PropensityPlan,
                   storage: PropensityStorage = PropensityStorage("jsonl")) -> None:
    for name in FILES:
        for _ in iter_validated_file(directory / storage.filename(name), name, plan, expected[name], storage, cancelled):
            pass


@dataclass(frozen=True, slots=True)
class PropensityProbeConfig:
    dates: tuple[datetime, ...]
    max_customers: int = 2000
    max_seconds: float = 600
    max_output_bytes: int = 128 * 1024**2
    max_events: int = 200000

    def __post_init__(self):
        if (not isinstance(self.dates, tuple) or not 1 <= len(self.dates) <= 2
                or tuple(sorted(set(self.dates))) != self.dates):
            raise ValueError("Probe requires one or two distinct ordered dates")
        for date in self.dates:
            utc_day(date)
        if (type(self.max_customers) is not int or not 1 <= self.max_customers <= 10000
                or type(self.max_output_bytes) is not int or not 1 <= self.max_output_bytes <= 2 * 1024**3
                or type(self.max_events) is not int or not 1 <= self.max_events <= 1000000
                or type(self.max_seconds) not in (int, float) or not math.isfinite(self.max_seconds)
                or not 0 < self.max_seconds <= 1800):
            raise ValueError("Invalid bounded probe resources")

    def metadata(self):
        return {**asdict(self), "dates": [iso(d) for d in self.dates],
                "selection": "first canonical IDs in SQLite BINARY order; not representative or label-balanced"}


class _RunGuard:
    """Cheap cancellation, periodic disk checks and hard probe write limits."""
    def __init__(self, builder, cancelled, reserve, started, probe):
        self.builder, self.user_cancelled, self.reserve = builder, cancelled, reserve
        self.started, self.probe = started, probe
        self.written = self.last_bytes = 0
        self.last_disk = float("-inf")

    def poll(self, force=False):
        if self.user_cancelled():
            raise PropensityCancelled
        now = perf_counter()
        if self.probe and now - self.started >= self.probe.max_seconds:
            raise PropensityDataError("PROBE_TIME_LIMIT")
        if force or now - self.last_disk >= 1 or self.written - self.last_bytes >= 8 * 1024**2:
            self.builder._guard(self.user_cancelled, self.reserve)
            self.last_disk, self.last_bytes = now, self.written
        return False

    def before_write(self, size):
        self.poll()
        if self.probe and self.written + size > self.probe.max_output_bytes:
            raise PropensityDataError("PROBE_OUTPUT_LIMIT")
        self.written += size


class PropensityDatasetBuilder:
    def __init__(self, *, raw_root: Path = index.DEFAULT_RAW_ROOT,
                 cache_dir: Path = index.DEFAULT_CACHE_DIR, output_dir: Path = DEFAULT_OUTPUT):
        self.raw_root = Path(raw_root).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.output_dir = Path(output_dir).resolve()

    def _path(self) -> Path:
        return index.CustomerHistoryIndexService(cache_dir=self.cache_dir).index_path(raw_root=self.raw_root)

    def inspect(self, config: PropensityDatasetConfig = PropensityDatasetConfig()) -> dict:
        """Metadata-only preflight: no iter_facts, integrity check or client scan."""
        with index._reader_lock(self.raw_root):
            sources = index._capture_sources(self.raw_root)
            path = self._path()
            version = index._database_version(path)
            with index._read_index(path) as connection:
                status = index._inspect_connection(connection, sources)
                publication = connection.execute(
                    "SELECT component,MIN(observed_us),MAX(observed_us),COUNT(*) FROM sources GROUP BY component"
                ).fetchall()
                plan = make_plan(config, status.coverage)
            self._fresh(sources.version, status.generation, version)
        try:
            inference_date = latest_prediction_date(status.coverage)
            inference_info = {"prediction_utc": iso(inference_date),
                              "last_complete_day_utc": (inference_date - timedelta(days=1)).date().isoformat(),
                              "target_until_utc": iso(inference_date + timedelta(days=HORIZON))}
        except PropensityDataError as exc:
            inference_info = {"unavailable_reason": str(exc)}
        return {
            "status": status.status.value, "schema_version": SCHEMA_VERSION, "index_schema_version": 2,
            "source_version": status.source_version, "generation": status.generation,
            "source_counts": asdict(status.counts), "coverage": self._coverage(status.coverage),
            "publication": [{"component": c, "earliest_us": low, "latest_us": high, "sources": n}
                            for c, low, high, n in publication],
            "as_of_level": AS_OF_LEVEL, "limitations": LIMITATIONS,
            "target": TARGET, "purpose": config.purpose, "latest_inference": inference_info,
            "business_validation": "REQUIRES_BUSINESS_VALIDATION",
            "prediction_dates": [iso(t) for t in plan.dates], "possible_slices": len(plan.dates),
            "slices_by_split": dict(Counter(plan.split(t) for t in plan.dates)),
            "validation_start": iso(plan.validation_start), "test_start": iso(plan.test_start),
            "row_estimate": None, "row_estimate_note": "Requires event scan; metadata cannot establish historical cohort size.",
            "fully_covered_target_slices": (sum(covered(status.coverage, "orders", index._micros(t),
                                                        index._micros(t) + HORIZON * DAY) for t in plan.dates)
                                            if config.purpose == "training" else None),
            "index_bytes": path.stat().st_size,
            "free_bytes": shutil.disk_usage(path.parent).free,
        }

    @staticmethod
    def _coverage(coverage: tuple) -> list[dict]:
        return [{"component": c.component, "since_utc": iso(c.since_utc), "until_utc": iso(c.until_utc),
                 "source_kind": c.source_kind} for c in coverage]

    def _fresh(self, source_version: str, generation: str, database_version: tuple) -> None:
        if index._capture_sources(self.raw_root).version != source_version:
            raise PropensityDataError("SOURCE_CHANGED")
        if index._database_version(self._path()) != database_version:
            raise PropensityDataError("INDEX_CHANGED")
        with index._read_index(self._path()) as connection:
            value = connection.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()
        if not value or value[0] != generation:
            raise PropensityDataError("INDEX_GENERATION_CHANGED")

    def build(self, config: PropensityDatasetConfig, *, pseudonym_key: bytes,
              allow_retrospective: bool = False, cancelled: Callable[[], bool] = lambda: False,
              progress: Callable[[PropensityProgress], None] = lambda value: None,
              min_free_bytes: int = 1024**3, storage: PropensityStorage = PropensityStorage(),
              probe: PropensityProbeConfig | None = None) -> PropensityDatasetSummary:
        if not allow_retrospective:
            raise ValueError("Explicit retrospective research opt-in required")
        if not isinstance(pseudonym_key, bytes) or len(pseudonym_key) < 32:
            raise ValueError("Supply a stable secret pseudonym key of at least 32 bytes")
        if type(min_free_bytes) is not int or min_free_bytes < 0:
            raise ValueError("Invalid free-space reserve")
        if not isinstance(storage, PropensityStorage) or (probe is not None and not isinstance(probe, PropensityProbeConfig)):
            raise ValueError("Invalid storage or probe configuration")
        if probe and config.purpose != "training":
            raise ValueError("Probe supports historical training slices only")
        if probe:
            min_free_bytes = max(min_free_bytes, 20 * 1024**3)
        # Keep generated artifacts away from source/cache trees even for explicit custom destinations.
        if any(self.output_dir == p or self.output_dir.is_relative_to(p) for p in (self.raw_root, self.cache_dir)):
            raise ValueError("Output must be separate from canonical data and index cache")
        started = perf_counter()
        sql_stop_reason = None
        parent = self.output_dir / "probes" if probe else self.output_dir
        parent.mkdir(parents=True, exist_ok=True)
        stage = parent / (".building-" + uuid.uuid4().hex)
        guard = _RunGuard(self, cancelled, min_free_bytes, started, probe)
        try:
            with _writer_lock(parent), index._reader_lock(self.raw_root):
                guard.poll(force=True)
                sources = index._capture_sources(self.raw_root)
                path = self._path()
                version = index._database_version(path)
                with index._read_index(path) as connection:
                    status = index._inspect_connection(connection, sources)
                    plan = make_plan(config, status.coverage)
                    if probe:
                        if not set(probe.dates) <= set(plan.dates) or len(probe.dates) >= len(plan.dates):
                            raise ValueError("Probe dates must be a strict subset of the current training grid")
                        plan = replace(plan, dates=probe.dates)
                    publication = tuple(PropensitySourceSpan(*r) for r in connection.execute(
                        "SELECT component,since_us,until_us,observed_us FROM sources WHERE component IN ('actions','orders') "
                        "ORDER BY component,since_us,until_us,observed_us"))
                    stage.mkdir()
                    # Both global full-build sorting and bounded probe SQL are interruptible.
                    def sql_guard():
                        nonlocal sql_stop_reason
                        try:
                            guard.poll()
                        except PropensityCancelled:
                            return 1
                        except PropensityDataError as exc:
                            sql_stop_reason = str(exc)
                            return 1
                        return 0

                    connection.set_progress_handler(sql_guard, 10000)
                    connection.row_factory = sqlite3.Row
                    counts, distributions, metrics = self._write(
                        stage, connection, plan, status, publication, pseudonym_key,
                        guard, progress, started, storage, probe)
                guard.poll(force=True)
                progress(PropensityProgress("validate_output", counts["event_memberships"], counts["customers"],
                                            counts["rows"], perf_counter() - started))
                phase = perf_counter()
                _validate_rows(stage, counts["files"], guard.poll, plan, storage)
                metrics["validation_seconds"] = perf_counter() - phase
                phase = perf_counter()
                checksums = {storage.filename(name): _digest(stage / storage.filename(name), guard.poll) for name in FILES}
                metrics["sha256_seconds"] = perf_counter() - phase
                sizes = {storage.filename(name): (stage / storage.filename(name)).stat().st_size for name in FILES}
                key_id = hmac.new(pseudonym_key, b"kanzler-propensity-key-id-v1", hashlib.sha256).hexdigest()
                identity = {"schema": SCHEMA_VERSION, "implementation": self._implementation(), "config": _config(config),
                            "source": status.source_version, "generation": status.generation, "key_id": key_id,
                            "checksums": checksums, "storage": storage.metadata(),
                            "probe": probe.metadata() if probe else None}
                dataset_version = hashlib.sha256(_json(identity).encode()).hexdigest()
                usable = counts["positive"] + counts["negative"] + counts["unlabeled"]
                verdict = "RESEARCH_PROBE_ONLY" if probe else "RESEARCH_DATASET_ONLY" if usable else "BLOCKED_BY_DATA_QUALITY"
                manifest = {
                    "schema_version": SCHEMA_VERSION, "version": dataset_version, "status": verdict,
                    "created_utc": iso(datetime.now(timezone.utc)), "identity": identity, "config": _config(config),
                    "as_of_level": AS_OF_LEVEL, "limitations": LIMITATIONS, "coverage": self._coverage(status.coverage),
                    "target": TARGET, "purpose": config.purpose,
                    **storage.metadata(), "file_sizes": sizes,
                    "probe": probe.metadata() if probe else None, "metrics": metrics,
                    "business_validation": "REQUIRES_BUSINESS_VALIDATION",
                    "label_fields_present": config.purpose == "training",
                    "source_publication": [asdict(span) for span in publication],
                    "index_schema_version": 2, "source_version": status.source_version, "generation": status.generation,
                    "source_counts": asdict(status.counts), "prediction_dates": [iso(t) for t in plan.dates],
                    "horizon_days": HORIZON, "validation_start": iso(plan.validation_start), "test_start": iso(plan.test_start),
                    "features": feature_schema(), "counts": counts, "distributions": distributions,
                    "label_semantics": {"target": TARGET, "status_policy": "Ignored, including unknown/cancelled/returned",
                                        "time": "firstAction.dateTimeUtc as placement proxy; completed checkout not independently verified",
                                        "interval": "labels [t,t+30d); features [t-window,t), UTC",
                                        "logical_order": "ORDER event_key derived from ids.mindboxId; orders.order_id agrees with source_event_id",
                                        "snapshots": "Identity, canonical owner and firstAction time consensus; lines/status/price/quantity ignored",
                                        "unknown": "Incomplete target coverage or unreliable identity/owner/time/provenance that could affect existence",
                                        "zero": "No logical order or unresolved candidate under complete declared Orders coverage",
                                        "inference": "No label, target or label_state field; future data is not required"},
                    "cohort": "Reliable interaction event time < t under retrospective current canonical identities",
                    "split_policy": "Chronological; purge target overlap at validation/test boundaries; repeated customers allowed",
                    "sampling": probe.metadata()["selection"] if probe else "No balancing, subsampling, SMOTE or synthetic orders",
                    "within_split_target_overlap_possible": config.step_days < HORIZON,
                    "money": "Omitted; schema v2 counts logical orders, not purchased units or money; no RUB/KZT mixing",
                    "checksums": checksums, "elapsed_seconds": perf_counter() - started,
                }
                if probe:
                    guard.before_write(len((_json(manifest) + "\n").encode("utf-8")))
                self._save(stage / "manifest.json", manifest)
                # Source lock remains held through final freshness validation and pointer switch.
                self._fresh(status.source_version, status.generation, version)
                guard.poll(force=True)
                destination = parent / dataset_version
                reused = destination.exists()
                if reused:
                    prior = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
                    stable = lambda m: {k: v for k, v in m.items() if k not in ("created_utc", "elapsed_seconds", "metrics")}
                    if (_json(stable(prior)) != _json(stable(manifest))
                            or any(_digest(destination / n, guard.poll) != h for n, h in checksums.items())):
                        raise PropensityDataError("EXISTING_VERSION_DAMAGED")
                    _remove_staging(stage, parent)
                else:
                    os.replace(stage, destination)
                if probe is None and verdict == "RESEARCH_DATASET_ONLY":
                    pointer = self.output_dir / (".current-" + uuid.uuid4().hex + ".tmp")
                    try:
                        self._save(pointer, {"version": dataset_version, "status": verdict,
                                             "manifest_sha256": _digest(destination / "manifest.json", cancelled)})
                        self._fresh(status.source_version, status.generation, version)
                        self._guard(cancelled, min_free_bytes)
                        current = "CURRENT_INFERENCE.json" if config.purpose == "inference" else "CURRENT.json"
                        os.replace(pointer, self.output_dir / current)
                    finally:
                        pointer.unlink(missing_ok=True)
                return PropensityDatasetSummary(verdict, dataset_version, destination, counts, perf_counter() - started, reused)
        except sqlite3.OperationalError:
            if cancelled():
                raise PropensityCancelled from None
            if sql_stop_reason:
                raise PropensityDataError(sql_stop_reason) from None
            raise PropensityDataError("HISTORY_SQL_ERROR") from None
        finally:
            if stage.exists():
                _remove_staging(stage, parent)

    @staticmethod
    def _implementation() -> str:
        digest = hashlib.sha256()
        for name in ("customer_propensity_build.py", "customer_propensity_dataset.py", "customer_propensity_history.py", "customer_propensity_storage.py"):
            digest.update(Path(__file__).with_name(name).read_bytes())
        for module in (customer_activity_service, purchased_products, product_resolution, selection):
            digest.update(Path(module.__file__).read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _save(path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _guard(self, cancelled: Callable[[], bool], min_free_bytes: int) -> None:
        if cancelled():
            raise PropensityCancelled
        if shutil.disk_usage(self.output_dir).free < min_free_bytes:
            raise PropensityDataError("INSUFFICIENT_FREE_SPACE")

    def _write(self, stage, connection, plan, status, publication, secret, guard, progress, started, storage, probe):
        counts = {"customers": 0, "customers_processed": 0, "rows": 0, "positive": 0, "negative": 0,
                  "unknown": 0, "embargo": 0, "event_memberships": 0, "repeated_customers": 0,
                  "files": {n: 0 for n in FILES}, "logical_actions": 0, "logical_orders": 0,
                  "action_variants": 0, "order_snapshots": 0, "unlabeled": 0, "uncompressed_bytes": 0}
        metrics = {k: 0.0 for k in ("read_seconds", "feature_seconds", "json_seconds", "write_compress_seconds", "close_fsync_seconds")}
        selected = None
        if probe:
            phase = perf_counter()
            selected = tuple(r[0] for r in connection.execute(
                "SELECT DISTINCT canonical_customer_id FROM event_customers ORDER BY canonical_customer_id LIMIT ?",
                (probe.max_customers,)))
            if selected:
                params = ",".join("?" for _ in selected)
                if len(selected) + 1 > connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER):
                    raise PropensityDataError("INVALID_PROBE_CUSTOMER_SCOPE")
                # Bound sorter input BEFORE starting the grouped provenance query.
                for joins, ceiling in (("JOIN variants v USING(event_key)", probe.max_events),
                                       ("JOIN variants v USING(event_key) LEFT JOIN record_origins r USING(variant_key)",
                                        probe.max_events * 2)):
                    size = connection.execute(
                        "SELECT COUNT(*) FROM (SELECT 1 FROM event_customers ec " + joins +
                        " WHERE ec.canonical_customer_id IN (" + params + ") LIMIT ?)",
                        (*selected, ceiling + 1)).fetchone()[0]
                    if size > ceiling:
                        raise PropensityDataError("PROBE_EVENT_LIMIT")
            metrics["read_seconds"] += perf_counter() - phase
        selected_set = set(selected) if selected is not None else None
        expected_memberships = 0
        reasons, missing, slices, histogram, split_customers = Counter(), Counter(), {}, Counter(), Counter()
        by_split = {n: Counter() for n in ("train", "validation", "test", "embargo", "inference")}

        cutoff = index._micros(plan.dates[0]) if plan.purpose == "inference" else None
        facts = (iter(()) if selected == () else iter_facts(connection, guard.poll, before_us=cutoff, **(
            {"customer_ids": selected} if selected is not None else {})))

        def timed_facts():
            while True:
                phase = perf_counter()
                try:
                    pair = next(facts)
                except StopIteration:
                    metrics["read_seconds"] += perf_counter() - phase
                    return
                metrics["read_seconds"] += perf_counter() - phase
                yield pair

        def counted(events):
            nonlocal expected_memberships
            for customer, event in events:
                guard.poll()
                if probe and counts["event_memberships"] >= probe.max_events:
                    raise PropensityDataError("PROBE_EVENT_LIMIT")
                counts["event_memberships"] += 1
                owners = {v.owner for v in event.variants}
                scoped_owners = owners if selected_set is None else owners & selected_set
                if scoped_owners and customer == min(scoped_owners):
                    expected_memberships += len(scoped_owners)
                    logical, variants = (("logical_orders", "order_snapshots") if event.source == "ORDER"
                                         else ("logical_actions", "action_variants"))
                    counts[logical] += 1
                    counts[variants] += len(event.variants)
                if counts["event_memberships"] % 100000 == 0:
                    progress(PropensityProgress("read", counts["event_memberships"], counts["customers"],
                                                counts["rows"], perf_counter() - started))
                yield event

        with ExitStack() as stack:
            streams = {n: stack.enter_context(open_writer(stage / storage.filename(n), storage, guard.before_write)) for n in FILES}
            for customer, events in groupby(timed_facts(), lambda pair: pair[0]):
                key = hmac.new(secret, b"kanzler-propensity-customer-v1\0" + customer.encode("utf-8"), hashlib.sha256).hexdigest()
                n, splits = 0, set()
                rows = iter(customer_rows(customer, counted(events), plan, status.coverage, key, guard.poll, publication))
                while True:
                    phase, prior_read = perf_counter(), metrics["read_seconds"]
                    try:
                        row = next(rows)
                    except StopIteration:
                        metrics["feature_seconds"] += max(0, perf_counter() - phase - (metrics["read_seconds"] - prior_read))
                        break
                    metrics["feature_seconds"] += max(0, perf_counter() - phase - (metrics["read_seconds"] - prior_read))
                    guard.poll()
                    n += 1
                    counts["rows"] += 1
                    name = ("inference" if row.split == "inference" else "embargo" if row.split == "embargo"
                            else "unknown" if row.target is None else row.split)
                    phase = perf_counter()
                    value = _json(row.payload()) + "\n"
                    metrics["json_seconds"] += perf_counter() - phase
                    counts["uncompressed_bytes"] += len(value.encode("utf-8"))
                    phase = perf_counter()
                    streams[name].write(value)
                    metrics["write_compress_seconds"] += perf_counter() - phase
                    counts["files"][name] += 1
                    category = ("unlabeled" if row.split == "inference" else
                                "unknown" if row.target is None else "positive" if row.target else "negative")
                    by_split[row.split][category] += 1
                    if name == "embargo":
                        counts["embargo"] += 1
                    else:
                        counts[category] += 1
                        splits.add(row.split)
                        if row.split != "inference":
                            reasons.update(row.label_reasons)
                    missing.update(k for k, v in row.available.items() if not v)
                    bucket = slices.setdefault(row.prediction_utc, Counter())
                    bucket[category] += 1
                    bucket["embargo"] += int(name == "embargo")
                counts["customers_processed"] += 1
                if n:
                    counts["customers"] += 1
                    counts["repeated_customers"] += int(n > 1)
                    histogram[n] += 1
                    split_customers.update(splits)
            phase = perf_counter()
        metrics["close_fsync_seconds"] = perf_counter() - phase
        if (expected_memberships != counts["event_memberships"] or (probe is None and plan.purpose == "training" and any(
                counts[name] != getattr(status.counts, name)
                for name in ("logical_actions", "logical_orders", "action_variants", "order_snapshots")))):
            raise PropensityDataError("HISTORY_COVERAGE_COUNTS_DISAGREE")
        counts["scan_scope"] = "bounded_probe" if probe else "pre_cutoff_inference" if plan.purpose == "inference" else "full_index"
        counts["unknown_reasons"] = dict(sorted(reasons.items()))
        labeled = counts["positive"] + counts["negative"]
        counts["positive_rate"] = counts["positive"] / labeled if labeled else None
        return counts, {"by_prediction_date": {k: dict(v) for k, v in sorted(slices.items())},
                        "by_split": {k: dict(v) for k, v in by_split.items()},
                        "rows_per_customer": {str(k): v for k, v in sorted(histogram.items())},
                        "customers_by_split": dict(split_customers),
                        "missing_features": dict(sorted(missing.items()))}, metrics
