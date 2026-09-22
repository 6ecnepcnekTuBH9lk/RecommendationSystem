"""Subprocess boundary for reference CSV parsing and atomic replacement."""

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from Application.loading_errors import emit_error
    from Application.files.reference_import import REFERENCE_TYPES, import_reference

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--kind", required=True, choices=REFERENCE_TYPES)
    parser.add_argument("--output-dir", type=Path, default=root / "ВходныеДанные")
    args = parser.parse_args(argv)
    try:
        result = import_reference(args.file, args.kind, output_dir=args.output_dir)
        print("Reference: " + json.dumps(result, ensure_ascii=True), flush=True)
        return 0
    except (KeyboardInterrupt, InterruptedError):
        return 130
    except Exception as exc:
        emit_error(exc, source="reference_csv")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
