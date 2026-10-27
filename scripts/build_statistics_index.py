"""Explicit analytics rebuild only; --status never scans canonical records."""

import argparse
import json
from pathlib import Path
import signal
import sys


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Application.statistics_index import DEFAULT_INDEX_DIR, GIB, StatisticsIndexError, build_statistics_index, index_state
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
    from Application.product_resolution import DEFAULT_CATALOG_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
                        help="Pre-created private local root under trusted parents; ACLs are checked, never repaired")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--build", action="store_true")
    parser.add_argument("--expected-gib", type=float, help="Pilot-based size budget; double headroom + 20 GiB reserve")
    parser.add_argument("--cancel-file", type=Path, help="Creating this file requests cooperative cancellation")
    args = parser.parse_args(argv)
    inputs = dict(raw_root=args.raw_root, catalog_path=args.catalog, index_dir=args.index_dir)
    if args.status:
        print(json.dumps({"state": index_state(**inputs)}), flush=True)
        return 0
    if args.expected_gib is None or args.expected_gib <= 0:
        parser.error("--build requires a positive --expected-gib based on a controlled pilot")
    stopped = False

    def cancel_signal(*_):
        nonlocal stopped
        stopped = True

    old_handlers = {}
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        value = getattr(signal, name, None)
        if value is not None:
            old_handlers[value] = signal.signal(value, cancel_signal)

    def emit(kind, value):
        print(json.dumps({"event": kind, "value": value}, ensure_ascii=True), flush=True)

    try:
        result = build_statistics_index(**inputs, expected_bytes=int(args.expected_gib*GIB),
            cancelled=lambda: stopped or bool(args.cancel_file and args.cancel_file.exists()),
            progress=lambda message: emit("progress", message), observer=lambda sample: emit("resources", sample))
        emit("result", result)
        return 0
    except InterruptedError:
        emit("cancelled", "CANCELLED: staging removed; previous generation retained")
        return 2
    except StatisticsIndexError as exc:
        emit("error", str(exc))
        return 1
    except Exception:
        # No raw exception/traceback: custom inputs may contain personal values.
        emit("error", "BUILD_FAILED: previous generation retained; no automatic retry")
        return 1
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
