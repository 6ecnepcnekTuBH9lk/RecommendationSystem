"""Canonical statistics subprocess: small JSON messages, no canonical writes."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Application.dataset_statistics import calculate_statistics
    from Application.analysis_filter import AnalysisFilter
    from Application.loading_errors import error_record
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
    from Application.product_resolution import DEFAULT_CATALOG_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--analysis-filter", default=None)
    args = parser.parse_args(argv)

    def emit(event, value):
        print(json.dumps({"event": event, "value": value}, ensure_ascii=True, allow_nan=False), flush=True)

    try:
        result = calculate_statistics(raw_root=args.raw_root, catalog_path=args.catalog,
                                      analysis_filter=AnalysisFilter.from_dict(json.loads(args.analysis_filter))
                                      if args.analysis_filter is not None else AnalysisFilter(),
                                      progress=lambda message: emit("progress", message))
        emit("result", asdict(result))
        return 0
    except Exception as exc:
        emit("error", error_record(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
