"""SYNTHETIC-ONLY SQL date candidate prototype, never a production analytics index.

Stores raw fixture payloads to prove date pruning without moving dedup/product
selection. Replays the existing calculation; no financial SQL SUM/float coercion.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from Application import dataset_statistics as baseline
from Application.analysis_filter import options_from_snapshot
from Application.mindbox.canonical_customers import encode
from Application.mindbox.adapters._common import timestamp, AdapterError
from Application.mindbox.raw_reader import iter_export
from Application.statistics_period import shared_intervals


def microseconds(value):
    delta = value - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return delta.days * 86400000000 + delta.seconds * 1000000 + delta.microseconds


class CandidatePrototype:
    def __init__(self, dataset, path):
        self.root, self.reference, _ = dataset
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA cache_size=-4096")
        self.db.executescript(
            "CREATE TABLE facts(source TEXT,directory TEXT,ordinal INTEGER,event_us INTEGER,bad_date INTEGER,outside INTEGER,payload TEXT); CREATE INDEX dates ON facts(source,directory,event_us);"
        )
        _, entries = baseline._statistics_sources(self.root)
        self.period = shared_intervals(
            *[
                [(e["since"], e["until"]) for e in entries[name]]
                for name in ("actions", "orders")
            ]
        )
        self.rows = self.bytes = 0
        started = perf_counter()
        for name, group in entries.items():
            for entry in group:
                directory = baseline._source_directory(self.root, name, entry)
                for ordinal, raw in enumerate(iter_export(name, input_dir=directory)):
                    date = None
                    bad = 0
                    try:
                        date = timestamp(
                            raw,
                            "dateTimeUtc"
                            if name == "actions"
                            else "firstAction.dateTimeUtc",
                            required=name == "orders",
                        )
                    except AdapterError:
                        bad = 1  # Keep error rows so the reference still raises when appropriate.
                    outside = int(
                        date is not None
                        and not any(a <= date < b for a, b in self.period)
                    )
                    payload = encode(raw)
                    self.db.execute(
                        "INSERT INTO facts VALUES(?,?,?,?,?,?,?)",
                        (
                            name,
                            str(directory),
                            ordinal,
                            microseconds(date) if date is not None else None,
                            bad,
                            outside,
                            payload,
                        ),
                    )
                    self.rows += 1
                    self.bytes += len(payload.encode())
        self.db.commit()
        self.build_seconds = perf_counter() - started
        self.file_bytes = Path(path).stat().st_size
        self.original = baseline.iter_export
        self.returned = 0

    @contextmanager
    def reader(self, selection):
        metadata = baseline._catalog_metadata(self.reference)
        effective = (
            options_from_snapshot(metadata, self.period).normalize(selection)
            if self.period
            else selection
        )
        self.returned = 0

        def records(name, *, input_dir):
            if name not in ("actions", "orders"):
                yield from self.original(name, input_dir=input_dir)
                return
            query = "SELECT payload FROM facts WHERE source=? AND directory=?"
            args = [name, str(input_dir)]
            if effective.interval is not None:
                first, last = map(microseconds, effective.interval)
                query += " AND (bad_date=1 OR (event_us>=? AND event_us<?)" + (
                    " OR outside=1)" if name == "orders" else ")"
                )
                args.extend((first, last))
            query += " ORDER BY ordinal"
            for (payload,) in self.db.execute(query, args):
                self.returned += 1
                yield json.loads(payload, parse_float=Decimal)

        baseline.iter_export = records
        try:
            yield self
        finally:
            baseline.iter_export = self.original

    def close(self):
        self.db.close()
