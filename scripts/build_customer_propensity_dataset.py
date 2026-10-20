"""Metadata inspection by default; explicit research build, never model training."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import signal
import secrets
import sys
from threading import Event


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Application import customer_history_index as index
from Application.model.customer_propensity_build import DEFAULT_OUTPUT, PropensityDatasetBuilder, PropensityProbeConfig
from Application.model.customer_propensity_storage import PropensityStorage
from Application.model.customer_propensity_dataset import AS_OF_LEVEL, STRICT_AS_OF_LEVEL, PropensityDatasetConfig, utc_day
from Application.model.customer_propensity_history import PropensityCancelled, PropensityDataError


def _date(value: str) -> datetime:
    try:
        return utc_day(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        raise argparse.ArgumentTypeError("Use UTC midnight, for example 2024-01-01T00:00:00Z") from None


def _print(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true", help="Explicit full read-only event scan and artifact build")
    mode.add_argument("--probe", action="store_true", help="Bounded research only; never updates CURRENT")
    parser.add_argument("--allow-retrospective", action="store_true", help="Acknowledge B-level temporal limitations")
    parser.add_argument("--as-of-level", choices=(AS_OF_LEVEL, STRICT_AS_OF_LEVEL), default=AS_OF_LEVEL,
                        help="Strict mode fails closed: historical merge availability is not in index v2")
    parser.add_argument("--purpose", choices=("training", "inference"), default="training",
                        help="Inference uses latest common coverage and exports no label fields")
    parser.add_argument("--key-file", type=Path, help="Stable secret binary key (>=32 bytes), outside versioned artifacts")
    parser.add_argument("--raw-root", type=Path, default=index.DEFAULT_RAW_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=index.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=_date)
    parser.add_argument("--end", type=_date, help="Inclusive prediction date; automatically capped by Orders coverage")
    parser.add_argument("--validation-start", type=_date)
    parser.add_argument("--test-start", type=_date)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--min-free-gib", type=int, help="Default 1 for full build; probe enforces at least 20")
    parser.add_argument("--storage-format", choices=("jsonl.gz", "jsonl"), default="jsonl.gz")
    parser.add_argument("--gzip-level", type=int, default=6)
    parser.add_argument("--probe-date", type=_date, action="append")
    parser.add_argument("--probe-max-customers", type=int)
    parser.add_argument("--probe-max-seconds", type=float)
    parser.add_argument("--probe-max-mib", type=int)
    parser.add_argument("--probe-max-events", type=int)
    parser.add_argument("--ephemeral-probe-key", action="store_true", help="Generate a probe-only key in memory; never save or print it")
    args = parser.parse_args(argv)
    probe_args = (args.probe_date, args.probe_max_customers, args.probe_max_seconds,
                  args.probe_max_mib, args.probe_max_events)
    if not args.probe and (any(v is not None for v in probe_args) or args.ephemeral_probe_key):
        parser.error("Probe limits and ephemeral keys require --probe")
    if args.probe and (not args.probe_date or args.purpose != "training"):
        parser.error("--probe requires one or two --probe-date values and training purpose")
    if args.key_file is not None and args.ephemeral_probe_key:
        parser.error("Choose an existing key file or an ephemeral probe key")
    if (args.build or args.probe) and (not args.allow_retrospective or not (args.key_file or args.ephemeral_probe_key)):
        parser.error("Writing requires --allow-retrospective and a key (ephemeral only for probe)")
    if args.min_free_gib is not None and args.min_free_gib < 0:
        parser.error("Invalid disk reserve")
    stop = Event()
    previous = signal.signal(signal.SIGINT, lambda *unused: stop.set())
    try:
        config = PropensityDatasetConfig(start=args.start, end=args.end, step_days=args.step_days,
                                        validation_start=args.validation_start, test_start=args.test_start,
                                        as_of_level=args.as_of_level, purpose=args.purpose)
        storage = PropensityStorage(args.storage_format, args.gzip_level)
        probe = (PropensityProbeConfig(tuple(args.probe_date),
                 max_customers=args.probe_max_customers if args.probe_max_customers is not None else 2000,
                 max_seconds=args.probe_max_seconds if args.probe_max_seconds is not None else 600,
                 max_output_bytes=(args.probe_max_mib if args.probe_max_mib is not None else 128) * 1024**2,
                 max_events=args.probe_max_events if args.probe_max_events is not None else 200000) if args.probe else None)
        output = args.output_dir or (DEFAULT_OUTPUT.parent / "customer_propensity_research" if probe else DEFAULT_OUTPUT)
        builder = PropensityDatasetBuilder(raw_root=args.raw_root, cache_dir=args.cache_dir, output_dir=output)
        _print({"phase": "preflight", **builder.inspect(config)})
        if args.build or args.probe:
            key = secrets.token_bytes(32) if args.ephemeral_probe_key else args.key_file.read_bytes()
            result = builder.build(config, pseudonym_key=key, allow_retrospective=True, storage=storage, probe=probe,
                                   cancelled=stop.is_set, min_free_bytes=(args.min_free_gib if args.min_free_gib is not None else 20 if probe else 1) * 1024**3,
                                   progress=lambda p: _print(asdict(p)))
            # Aggregate only: no customer IDs, raw records, secret or data paths.
            _print({"status": result.status, "version": result.version, "counts": result.counts,
                    "elapsed_seconds": result.elapsed_seconds, "reused": result.reused})
            return 0 if result.status in ("RESEARCH_DATASET_ONLY", "RESEARCH_PROBE_ONLY") else 2
        return 0
    except PropensityCancelled:
        _print({"status": "CANCELLED"})
        return 130
    except Exception as exc:
        # Canonical validation exceptions may contain paths/raw identifiers: never print them.
        category = str(exc) if isinstance(exc, PropensityDataError) else type(exc).__name__
        _print({"status": "FAILED", "category": category})
        return 2
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    raise SystemExit(main())
