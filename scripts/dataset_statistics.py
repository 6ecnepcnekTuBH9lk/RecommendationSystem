"""Canonical statistics subprocess: small JSON messages, no canonical writes."""

import argparse
import json
import os
from contextlib import suppress
from datetime import datetime, timezone
from threading import Event, Thread
from pathlib import Path
import sys


def _private_directory(directory, *, create):
    from Application.statistics_index import protect_directory
    from Application.statistics_storage_safety import StatisticsIndexError, path_identity, reject_reparse, validate_storage_root

    directory = reject_reparse(directory)
    if not os.path.lexists(directory):
        if not create:
            raise StatisticsIndexError("UNIFIED_CACHE_MISS")
        parent = path_identity(directory.parent)
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass  # Another run owns initialization. Admission still checks its ACL.
        else:
            identity = path_identity(directory)
            try:
                if path_identity(reject_reparse(directory.parent)) != parent:
                    raise StatisticsIndexError("STAGING_PATH_CHANGED")
                protect_directory(directory)
                if path_identity(reject_reparse(directory)) != identity:
                    raise StatisticsIndexError("STAGING_PATH_CHANGED")
            except Exception:
                with suppress(OSError, StatisticsIndexError):
                    if (path_identity(reject_reparse(directory.parent)) == parent and
                            path_identity(reject_reparse(directory)) == identity):
                        directory.rmdir()  # Only an owned, still empty directory.
                raise
    validate_storage_root(directory)
    return directory


def _unified(args, selection, emit):
    from Application.paths import PROJECT_ROOT
    from Application.statistics_index import DEFAULT_INDEX_DIR
    from Application.statistics_storage_safety import StatisticsIndexError, reject_reparse
    from Application.statistics_unified_dto import StatisticsResultError
    from Application.statistics_unified_detail import item_matches
    from Application.statistics_unified_index import POINTER_NAME
    from Application.statistics_unified_presentation import UNIFIED_ERRORS
    from Application.statistics_unified_service import calculate_unified_statistics

    stopped = Event()

    def listen():
        if sys.stdin is None:
            return
        data = b""
        while b"\n" not in data:
            chunk = os.read(sys.stdin.fileno(), 32)
            if not chunk:
                return
            data += chunk
        if data.splitlines()[0].strip() == b"cancel":
            stopped.set()

    Thread(target=listen, daemon=True).start()
    data = PROJECT_ROOT / "filtered_data"
    try:
        reader = args.backend
        if reader not in ("REFERENCE", "INDEXED"):
            raise StatisticsIndexError("INCOMPATIBLE_SCHEMA")
        index = (args.index_dir or DEFAULT_INDEX_DIR) if reader == "INDEXED" else None
        if index is not None and not os.path.lexists(reject_reparse(index) / POINTER_NAME):
            raise StatisticsIndexError("INDEX_MISSING")
        scratch = args.unified_workspace or data / "statistics_unified_workspace"
        target = _private_directory(args.unified_cache or data / "statistics_unified_cache", create=not args.cache_only)
        if not args.cache_only:
            scratch = _private_directory(scratch, create=True)
        emit("progress", "Проверка сохранённого отчёта…" if args.cache_only else "Проверка источника и расчёт единой статистики…")

        def matches(code, item):
            return item_matches(selection, args.product_code, args.category_code, code, item)

        restricted = selection.product_restricted or args.product_code or args.category_code
        result = calculate_unified_statistics(
            raw_root=args.raw_root, catalog_path=args.catalog, scratch_dir=scratch, cache_dir=target,
            expected_bytes=args.expected_bytes, calculated_at=datetime.now(timezone.utc),
            reader=reader, index_dir=index, cache_only=args.cache_only, cancelled=stopped.is_set,
            observer=lambda sample: emit("progress", "Расчёт единой статистики. Обработка данных…"),
            order_period=selection.interval, buyer_period=selection.interval, customer_period=selection.interval,
            selected_start=selection.start_date, selected_end=selection.end_date,
            product_filter=matches if restricted else None,
        )
        if stopped.is_set():
            raise InterruptedError
        emit("result", {"report": result.envelope(), "analysis_filter": selection.to_dict(),
                        "product_code": args.product_code, "category_code": args.category_code})
        return 0
    except InterruptedError:
        emit("unified_error", "CANCELLED")
    except StatisticsIndexError as error:
        code = str(error)
        emit("unified_error", code if code in UNIFIED_ERRORS else "UNIFIED_FAILED")
    except StatisticsResultError:
        emit("unified_error", "INVALID_UNIFIED_DTO")
    except OSError:
        emit("unified_error", "CACHE_WRITE_FAILED")
    except Exception:
        emit("unified_error", "UNIFIED_FAILED")
    return 1


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Application.statistics_slices import (
        calculate_cached_statistics,
        DEFAULT_CACHE_DIR,
    )
    from Application.analysis_filter import AnalysisFilter
    from Application.loading_errors import error_record
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
    from Application.product_resolution import DEFAULT_CATALOG_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--analysis-filter", default=None)
    parser.add_argument("--slice-cache", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--backend", choices=("AUTO", "REFERENCE", "INDEXED"), default="AUTO")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("legacy", "unified"), default="legacy")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--unified-cache", type=Path)
    parser.add_argument("--unified-workspace", type=Path)
    parser.add_argument("--expected-bytes", type=int, default=10 * 1024**3)
    parser.add_argument("--product-code", default="")
    parser.add_argument("--category-code", default="")
    args = parser.parse_args(argv)

    def emit(event, value):
        print(
            json.dumps(
                {"event": event, "value": value}, ensure_ascii=True, allow_nan=False
            ),
            flush=True,
        )

    try:
        selection = (AnalysisFilter.from_dict(json.loads(args.analysis_filter))
                     if args.analysis_filter is not None else AnalysisFilter())
        if args.mode == "unified":
            return _unified(args, selection, emit)
        result = calculate_cached_statistics(
            raw_root=args.raw_root,
            catalog_path=args.catalog,
            cache_dir=args.slice_cache,
            backend=args.backend,
            index_dir=args.index_dir,
            analysis_filter=selection,
            progress=lambda message: emit("progress", message),
        )
        emit("result", result)
        return 0
    except Exception as exc:
        if args.mode == "unified":
            emit("unified_error", "INVALID_FILTER")
        else:
            emit("error", error_record(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
