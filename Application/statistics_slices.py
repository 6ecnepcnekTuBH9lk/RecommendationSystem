"""Bounded result cache; canonical JSON is never scanned on a cache hit.

Compatibility uses the publication contract (catalog digest/revision and immutable
canonical object generations), part inventory, Customers commit header/summary,
reference content hashes, algorithm sources and the local demographic date.
This is not a tamper audit of raw JSON rewritten outside canonical publishers.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from decimal import getcontext
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
import sys

import ijson

from Application import dataset_statistics as statistics
from Application.analysis_filter import AnalysisFilter, options_from_snapshot
from Application.mindbox.canonical_storage import atomic_json, storage_lock
from Application.mindbox.canonical_customers import database
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, part_files
from Application.paths import PROJECT_ROOT
from Application.product_resolution import DEFAULT_CATALOG_PATH
from Application.statistics_cache import _unique_object, validate_result
from Application.statistics_period import shared_intervals
from Application.statistics_rfm_validation import rfm_cutoff

DEFAULT_CACHE_DIR = PROJECT_ROOT / "filtered_data" / "dataset_statistics_slices"
SCHEMA_VERSION = 1
MAX_ENTRIES = 32
logger = logging.getLogger(__name__)


def _json(value):
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _digest(value):
    return hashlib.sha256(_json(value).encode("ascii")).hexdigest()


def _stat(path):
    value = path.stat()
    return value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _hash_file(path):
    before = _stat(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if _stat(path) != before:
        raise ValueError("Statistics cache sources changed")
    return digest.hexdigest()


def _algorithm_version():
    package = Path(statistics.__file__).parent
    names = (
        "dataset_statistics.py",
        "statistics_slices.py",
        "statistics_index.py",
        "statistics_storage_safety.py",
        "statistics_facts.py",
        "statistics_query.py",
        "statistics_cache.py",
        "statistics_period.py",
        "analysis_filter.py",
        "action_statistics.py",
        "order_statistics.py",
        "product_statistics.py",
        "interactions.py",
        "product_resolution.py",
        "customer_demographics.py",
        "customer_statistics_scope.py",
        "statistics_purchase_policy.py",
        "statistics_purchase_facts.py",
        "statistics_buyer_aggregates.py",
        "statistics_buyer_validation.py",
        "statistics_rfm_core.py",
        "statistics_rfm_validation.py",
        "statistics_cohorts_core.py",
        "statistics_cohorts_validation.py",
    )
    # Include adapters, identity, strict reader and publication rules recursively.
    paths = [package / name for name in names] + sorted(
        (package / "mindbox").rglob("*.py")
    )
    return _digest(
        [(str(path.relative_to(package)), _hash_file(path)) for path in paths]
    )


def _customers(path):
    if not path.exists():
        return {"present": False}
    # Writers use rollback-journal transactions; never attest a live/hot sidecar.
    if any(
        Path(str(path) + suffix).exists() for suffix in ("-journal", "-wal", "-shm")
    ):
        raise ValueError("Customers transaction prevents cache attestation")
    before = _stat(path)
    with path.open("rb") as stream:
        header = stream.read(100)
    if len(header) != 100 or header[:16] != b"SQLite format 3\x00":
        raise ValueError("Invalid Customers header")
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2
    )
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='summary'"
        ).fetchone()
        schema = connection.execute(
            "SELECT type,name,sql FROM sqlite_master ORDER BY type,name"
        ).fetchall()
    finally:
        connection.close()
    if _stat(path) != before:
        raise ValueError("Customers changed")
    return {
        "present": True,
        "file": before,
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "summary": row[0] if row else None,
        "schema": schema,
    }


def capture_sources(raw_root, catalog_path, analysis_filter, *, already_locked=False, calculated_at=None):
    """Short locked metadata/reference snapshot; no Actions/Orders/Merges rows."""
    root, reference = Path(raw_root).resolve(), Path(catalog_path).resolve()
    if not (root / "canonical/catalog.json").is_file():
        raise ValueError("Canonical dataset is missing")
    from contextlib import nullcontext

    with nullcontext() if already_locked else storage_lock(root):
        data, entries = statistics._statistics_sources(root)
        inventory = []
        for name, group in (
            *entries.items(),
            (
                "customer_merges",
                [data["customer_merges"]] if data["customer_merges"] else [],
            ),
        ):
            for entry in group:
                directory = statistics._source_directory(root, name, entry)
                inventory.append(
                    (
                        name,
                        entry,
                        [(p.name, _stat(p)) for p in part_files(directory, name)],
                    )
                )
        hashes = (
            _hash_file(reference),
            _hash_file(reference.with_name("site_categories.csv")),
        )
        metadata = statistics._catalog_metadata(reference)
        period = shared_intervals(
            *[
                [(e["since"], e["until"]) for e in entries[name]]
                for name in ("actions", "orders")
            ]
        )
        effective = (
            options_from_snapshot(metadata, period).normalize(analysis_filter)
            if period
            else analysis_filter
        )
        local = (calculated_at or datetime.now(timezone.utc)).astimezone()
        cutoff = rfm_cutoff(shared_intervals([], [(e['since'], e['until']) for e in entries['orders']]),
                            analysis_filter.end_date, local)
        context = getcontext()
        decimal_config = {
            name: getattr(context, name)
            for name in ("prec", "rounding", "Emin", "Emax", "capitals", "clamp")
        }
        decimal_config["traps"] = {
            signal.__name__: enabled for signal, enabled in context.traps.items()
        }
        source = {
            "schema": SCHEMA_VERSION,
            "root": str(root),
            "canonical": data,
            "parts": inventory,
            "reference_path": str(reference),
            "reference_sha256": hashes,
            "customers": _customers(database(root)),
            "algorithm": _algorithm_version(),
            "runtime": {
                "python": sys.version.split()[0],
                # Sticky Inexact/Rounded flags are outputs, not algorithm inputs.
                "decimal": decimal_config,
                "sqlite": sqlite3.sqlite_version,
                "ijson": ijson.__version__,
                "parser": ijson.backend,
            },
            "as_of": local.date().isoformat(),
            "timezone": str(local.tzinfo),
            "filter": effective.to_dict(),
            "customer_period": {"start_date": analysis_filter.start_date, "end_date": analysis_filter.end_date},
            "rfm_as_of": cutoff.isoformat() if cutoff is not None else None,
        }
        # Freeze to JSON primitives, so comparisons survive disk round trips.
        return json.loads(_json(source)), effective


def load_slice(directory, provenance):
    path = Path(directory) / (_digest(provenance) + ".json")
    try:
        if path.is_symlink() or path.stat().st_size > 16 * 1024**2:
            return None
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream, object_pairs_hook=_unique_object)
        if (
            set(payload) != {"schema", "provenance", "result", "result_sha256"}
            or type(payload["schema"]) is not int
            or payload["schema"] != SCHEMA_VERSION
            or payload["provenance"] != provenance
            or _digest(payload["result"]) != payload["result_sha256"]
        ):
            return None
        result = validate_result(payload["result"])
        if result.get('customer_scope') is not None and result['customer_scope']['period'] != provenance['customer_period']:
            return None
        if result.get('buyer_statistics') is not None and result['buyer_statistics']['period'] != provenance['customer_period']:
            return None
        if result.get('rfm_statistics') is not None and result['rfm_statistics']['as_of_date'] != provenance['rfm_as_of']:
            return None
        if result.get('cohort_statistics') is not None and result['cohort_statistics']['as_of_date'] != provenance['rfm_as_of']:
            return None
        if (
            AnalysisFilter.from_dict(result["analysis_filter"]).to_dict()
            != AnalysisFilter.from_dict(provenance["filter"]).to_dict()
        ):
            return None
        if (
            datetime.fromisoformat(result["calculated_at"])
            .astimezone()
            .date()
            .isoformat()
            != provenance["as_of"]
        ):
            return None
        return result
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        logger.warning(
            "Статистический срез недоступен или повреждён; требуется расчёт."
        )
        return None


def save_slice(directory, provenance, result):
    validate_result(result)
    if result.get('customer_scope') is not None and result['customer_scope']['period'] != provenance['customer_period']:
        raise ValueError('Customer period differs from provenance')
    if result.get('buyer_statistics') is not None and result['buyer_statistics']['period'] != provenance['customer_period']:
        raise ValueError('Buyer period differs from provenance')
    if result.get('rfm_statistics') is not None and result['rfm_statistics']['as_of_date'] != provenance['rfm_as_of']:
        raise ValueError('RFM cutoff differs from provenance')
    if result.get('cohort_statistics') is not None and result['cohort_statistics']['as_of_date'] != provenance['rfm_as_of']:
        raise ValueError('Cohort cutoff differs from provenance')
    path = Path(directory) / (_digest(provenance) + ".json")
    if path.is_symlink():
        raise ValueError("Unsafe statistics slice path")
    if AnalysisFilter.from_dict(result["analysis_filter"]) != AnalysisFilter.from_dict(
        provenance["filter"]
    ):
        raise ValueError("Statistics filter differs from provenance")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        path,
        {
            "schema": SCHEMA_VERSION,
            "provenance": provenance,
            "result": result,
            "result_sha256": _digest(result),
        },
    )
    # Only own hash-named entries, never the existing last-successful JSON.
    entries = sorted(
        (
            p
            for p in path.parent.iterdir()
            if re.fullmatch(r"[0-9a-f]{64}\.json", p.name)
            and p.is_file()
            and not p.is_symlink()
        ),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
        reverse=True,
    )
    for old in entries[MAX_ENTRIES:]:
        old.unlink(missing_ok=True)


def calculate_cached_statistics(
    *,
    raw_root=DEFAULT_RAW_ROOT,
    catalog_path=DEFAULT_CATALOG_PATH,
    analysis_filter=AnalysisFilter(),
    progress=None,
    cancelled=None,
    cache_dir=DEFAULT_CACHE_DIR,
    backend="REFERENCE",
    index_dir=None,
):
    """Return the validated display dict. Existing calculate_statistics stays intact."""
    calculated_at = datetime.now(timezone.utc)

    def check():
        if cancelled and cancelled():
            raise InterruptedError("Расчёт статистики отменён")

    def capture():
        check()
        return capture_sources(raw_root, catalog_path, analysis_filter, calculated_at=calculated_at)

    if backend not in ("REFERENCE", "INDEXED", "AUTO"):
        raise ValueError("Unknown statistics backend")
    check()
    before = None
    if cache_dir is not None:
        try:
            before, effective = capture()
            result = load_slice(cache_dir, before)
            if result is not None and capture()[0] == before:
                check()
                if progress:
                    progress("Кеш: подтверждённый статистический срез")
                return result
        except InterruptedError:
            raise
        except Exception:
            logger.warning(
                "Совместимость кеша не подтверждена; используется эталонный расчёт."
            )
            before = None
    arguments = dict(
            raw_root=raw_root,
            catalog_path=catalog_path,
            analysis_filter=analysis_filter,
            progress=progress,
            cancelled=cancelled,
            calculated_at=calculated_at,
    )
    if backend in ("INDEXED", "AUTO"):
        from Application.statistics_index import StatisticsIndexError
        try:
            result = asdict(statistics.calculate_statistics(**arguments, backend="INDEXED", index_dir=index_dir))
        except StatisticsIndexError as exc:
            if backend == "INDEXED":
                raise
            if progress and str(exc) != "SOURCE_UNAVAILABLE":
                progress(f"REFERENCE: медленный режим ({exc}); аналитический индекс требуется подготовить/обновить отдельно")
            result = asdict(statistics.calculate_statistics(**arguments))
            if progress and str(exc) == "SOURCE_UNAVAILABLE":
                # Preserve the established immediate single-error CLI response
                # when even the reference prerequisites are absent.
                progress("REFERENCE: медленный режим; источник быстрого режима недоступен")
    else:
        if progress:
            progress("REFERENCE: эталонный расчёт по canonical-истории")
        result = asdict(statistics.calculate_statistics(**arguments))
    validate_result(result)
    check()
    if before is not None:
        try:
            after, _ = capture()
            check()
            if after == before:
                save_slice(cache_dir, before, result)
        except InterruptedError:
            raise
        except Exception:
            logger.warning("Кеш среза не сохранён; рассчитанный результат не изменён.")
    return result
