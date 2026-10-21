"""Explicit local ZIP export. Defaults to a small trial; never starts a full run implicitly."""

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from threading import Event, Thread

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Application.recommendation_export_records import ExportError, ExportCancelled
from Application.model.propensity_ml.resources import (
    ResourceGuard,
    MLCancelled,
    MLDataError,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--audience", choices=("bpr",), default="bpr")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--limit", type=int, default=50)
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cancelled = Event()

    def listen():
        if sys.stdin is not None:
            for line in sys.stdin:
                if line.strip() == "cancel":
                    cancelled.set()

    logging.disable(logging.INFO)

    def send(**payload):
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    try:
        resource = ResourceGuard(
            (args.destination.absolute().parent,), cancelled=cancelled.is_set
        )
        resource.poll(force=True)
        send(type="progress", phase="prepare", processed=0, total=0)
        from Application.recommendation_export_batch import RecommendationBatchSession
        from Application.recommendation_export import export_recommendations

        # On Windows an already-blocked stdin reader can stall native numeric
        # DLL initialization. Start the control reader AFTER those imports.
        # A GUI cancellation during imports remains queued in the input pipe.
        Thread(target=listen, daemon=True).start()

        with RecommendationBatchSession(
            audience=args.audience,
            limit=None if args.full else args.limit,
            cancelled=cancelled.is_set,
        ) as session:
            resource.poll(force=True)
            send(
                type="progress",
                phase="customers",
                processed=0,
                total=session.metadata.total_candidates,
            )
            result = export_recommendations(
                args.destination,
                session,
                session.metadata,
                cancelled=cancelled.is_set,
                guard=session.guard,
                monitor=resource.poll,
                overwrite=args.overwrite,
                progress=lambda **p: send(type="progress", **p),
            )
        send(type="success", summary=asdict(result), resources=resource.payload())
        return 0
    except (ExportCancelled, MLCancelled):
        send(type="cancelled")
        return 2
    except ExportError as exc:
        send(type="error", code=exc.code)
        return 1
    except MLDataError as exc:
        # These exceptions are static technical categories, no record payloads.
        send(
            type="error", code="RESOURCE_OR_STORAGE_GUARD", category=type(exc).__name__
        )
        return 1
    except Exception as exc:
        send(type="error", code="EXPORT_FAILED", category=type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
