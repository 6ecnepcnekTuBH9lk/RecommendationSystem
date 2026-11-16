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
    from Application.statistics_storage_safety import StatisticsIndexError, path_identity, reject_reparse, validate_storage_root

    directory = reject_reparse(directory)
    admitted = None
    if not os.path.lexists(directory):
        if not create:
            raise StatisticsIndexError("UNIFIED_CACHE_MISS")
        from Application.paths import PROJECT_ROOT
        if directory.parent == PROJECT_ROOT / "filtered_data":
            directory.parent.mkdir(exist_ok=True)
        parent = path_identity(directory.parent)
        try:
            _create_private_directory(directory)
        except FileExistsError:
            pass  # Another run owns initialization. Admission still checks its ACL.
        else:
            identity = path_identity(directory)
            try:
                if path_identity(reject_reparse(directory.parent)) != parent:
                    raise StatisticsIndexError("STAGING_PATH_CHANGED")
                admitted = validate_storage_root(directory)
                if path_identity(reject_reparse(directory)) != identity:
                    raise StatisticsIndexError("STAGING_PATH_CHANGED")
            except Exception:
                with suppress(OSError, StatisticsIndexError):
                    if (path_identity(reject_reparse(directory.parent)) == parent and
                            path_identity(reject_reparse(directory)) == identity):
                        directory.rmdir()  # Only an owned, still empty directory.
                raise
    if admitted is None:
        validate_storage_root(directory)
    return directory


def _create_private_directory(directory):
    if os.name != "nt":
        directory.mkdir(mode=0o700)
        return
    import ctypes
    from ctypes import wintypes
    from Application.statistics_index import _windows_user_sid

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("descriptor", wintypes.LPVOID), ("inherit", wintypes.BOOL)]

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.LPVOID), wintypes.LPVOID)
    convert.restype = wintypes.BOOL
    create = kernel.CreateDirectoryW
    create.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(SecurityAttributes))
    create.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = (wintypes.LPVOID,)
    kernel.LocalFree.restype = wintypes.LPVOID
    descriptor = wintypes.LPVOID()
    acl = f"D:P(A;OICI;FA;;;{_windows_user_sid()})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
    if not convert(acl, 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
        if not create(str(directory), ctypes.byref(attributes)):
            code = ctypes.get_last_error()
            if code == 183:
                raise FileExistsError(directory)
            raise ctypes.WinError(code)
    finally:
        kernel.LocalFree(descriptor)


def _unified(args, selection, emit):
    from Application.paths import PROJECT_ROOT
    from Application.statistics_index import DEFAULT_INDEX_DIR
    from Application.statistics_storage_safety import StatisticsIndexError
    from Application.statistics_unified_dto import StatisticsResultError
    from Application.statistics_unified_detail import item_matches
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
        phase = lambda message: emit("progress", message)
        if index is not None and not args.cache_only:
            index = _ensure_index(args.raw_root, args.catalog, index, args.expected_bytes, stopped.is_set, phase,
                                  source_changed=lambda: emit("source_changed", None))
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
            cache_mode="slice" if reader == "INDEXED" else "latest", prepared_index=reader == "INDEXED",
            remember_selection=(selection.to_dict(), args.product_code, args.category_code)
                if reader == "INDEXED" and not args.cache_only else None,
            phase=phase,
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


def _ensure_index(raw_root, catalog_path, index_dir, expected_bytes, cancelled, phase, *, source_changed=lambda: None):
    from Application.mindbox.canonical_storage import storage_lock
    from Application.statistics_storage_safety import StatisticsIndexError
    from Application.statistics_unified_analytics import open_analytics, prepare_analytics
    from Application.statistics_unified_index import _capture, _inputs, build_unified_index, open_unified_index

    def check():
        if cancelled():
            raise InterruptedError

    def build_progress(sample):
        elapsed = int(sample["elapsed_seconds"])
        if elapsed and elapsed % 15 == 0:
            phase(f"Подготовка индексированных данных Mindbox · {elapsed} с")

    phase("Проверка загруженных данных Mindbox…")
    check()
    try:
        root, catalog = _inputs(raw_root, catalog_path)
        with storage_lock(root):
            provenance = _capture(root, catalog)
    except StatisticsIndexError:
        source_changed()
        raise
    if not any(kind in {"orders", "actions"} for kind, _, _ in provenance["parts"]):
        source_changed()
        raise StatisticsIndexError("STATISTICS_DATA_REQUIRED")
    directory = _private_directory(index_dir, create=True)
    paths = dict(raw_root=root, catalog_path=catalog, index_dir=directory)
    build = dict(**paths, expected_bytes=expected_bytes, cancelled=cancelled)
    phase("Проверка подготовленных данных…")
    try:
        with open_unified_index(**paths, check=check):
            pass
    except StatisticsIndexError as error:
        if str(error) not in {"INDEX_MISSING", "STALE_SOURCE", "INCOMPATIBLE_SCHEMA", "INVALID_MANIFEST",
                              "INVALID_DEMOGRAPHICS", "INVALID_UNIFIED_INDEX_INTEGRITY"}:
            raise
        source_changed()
        phase("Подготовка индексированных данных Mindbox…")
        build_unified_index(**build, observer=build_progress)
    phase("Проверка аналитических структур…")
    try:
        with open_unified_index(**paths, check=check) as reader:
            with open_analytics(directory, reader, check):
                pass
    except StatisticsIndexError as error:
        if str(error) not in {"ANALYTICAL_INDEX_MISSING", "ANALYTICAL_INDEX_STALE", "INVALID_ANALYTICAL_INDEX"}:
            raise
        prepare_analytics(**build, phase=phase)
        phase("Проверка готовых аналитических данных…")
    return directory


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
