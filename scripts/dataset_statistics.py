"""Canonical statistics subprocess: small JSON messages, no canonical writes."""

import argparse
import json
from pathlib import Path
import sys


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
    args = parser.parse_args(argv)

    def emit(event, value):
        print(
            json.dumps(
                {"event": event, "value": value}, ensure_ascii=True, allow_nan=False
            ),
            flush=True,
        )

    try:
        result = calculate_cached_statistics(
            raw_root=args.raw_root,
            catalog_path=args.catalog,
            cache_dir=args.slice_cache,
            backend=args.backend,
            index_dir=args.index_dir,
            analysis_filter=AnalysisFilter.from_dict(json.loads(args.analysis_filter))
            if args.analysis_filter is not None
            else AnalysisFilter(),
            progress=lambda message: emit("progress", message),
        )
        emit("result", result)
        return 0
    except Exception as exc:
        emit("error", error_record(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
