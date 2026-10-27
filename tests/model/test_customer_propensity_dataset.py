"""Synthetic facts and tiny v2 databases only. Never opens the real history."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import gzip
import json
from pathlib import Path
import sqlite3
import tracemalloc
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Application import customer_history_index as index
from Application.customer_history_normalization import history_event_key
from Application.customer_history_records import HistorySource
from Application.model import customer_propensity_build as build
from Application.model import customer_propensity_resources as resources
from Application.model.customer_propensity_dataset import (
    AS_OF_LEVEL, DAY, SCHEMA_VERSION, TARGET, PropensityDatasetConfig, PropensityInferenceRow,
    PropensityLabel, PropensityPlan, PropensitySourceSpan, covered, customer_rows, feature_schema,
    latest_prediction_date, make_plan,
)
from Application.model.customer_propensity_history import (
    EventFact, LineFact, PropensityCancelled, PropensityDataError, VariantFact, iter_facts, order_outcome,
)
from Application.model.customer_propensity_storage import PropensityStorage, iter_propensity_rows
from scripts import build_customer_propensity_dataset as cli


T = datetime(2024, 1, 1, tzinfo=timezone.utc)
PLAN = PropensityPlan((T, T + timedelta(days=30), T + timedelta(days=60)),
                      T + timedelta(days=30), T + timedelta(days=60))
COVERAGE = tuple(index.HistoryCoverage(c, T - timedelta(days=800), T + timedelta(days=500), "API")
                 for c in ("orders", "actions"))
PUBLICATION = tuple(PropensitySourceSpan(c.component, index._micros(c.since_utc), index._micros(c.until_utc),
                                        index._micros(T - timedelta(days=900))) for c in COVERAGE)
CONFIG = PropensityDatasetConfig(start=PLAN.dates[0], end=PLAN.dates[-1],
                                validation_start=PLAN.validation_start, test_start=PLAN.test_start)
SECRET = b"synthetic secret, never used in real data" * 2


def variant(day=0, *, kind="ORDER", status="F", qty="1", owner="customer", observed=-1,
            identity="line", code="000001", key="v", created=None):
    return VariantFact(key, owner, kind, index._micros(T) + int(day * DAY),
                       None if observed is None else index._micros(T) + int(observed * DAY),
                       None if created is None else index._micros(T) + int(created * DAY),
                       (LineFact(identity, 0, status, qty, code),) if kind == "ORDER" else ())


def order(*variants, **kwargs):
    return EventFact("ORDER", tuple(variants) or (variant(**kwargs),))


def action(day=-1, kind="VIEW", **kwargs):
    return EventFact("ACTION", (variant(day, kind=kind, **kwargs),))


def rows(events, coverage=COVERAGE):
    return list(customer_rows("customer", iter(events), PLAN, coverage, "b" * 64, publication=PUBLICATION))


@pytest.mark.parametrize("day,status,expected", [
    (0, "F", 1), (29.999, "CP", 1), (10, "delivering", 1), (30, "F", 0), (-.001, "F", 0),
    (10, "Mindbox-Tech-Cancellation", 1), (10, "otmenen", 1), (10, "out-of-stock", 1),
    (10, "Return", 1), (10, "klientskii-vozvrat", 1), (10, "N", 1), (10, "CD", 1),
    (10, "new", 1), (10, "sold", 1), (10, "RJ", 1), (10, "UNRECOGNIZED", 1),
])
def test_binary_labels_boundaries_statuses(day, status, expected):
    row = rows([action(), order(day=day, status=status)])[0]
    assert row.target == expected
    assert row.features["placed_orders_30d"] == int(day < 0)
    assert row.label_state == (PropensityLabel.UNKNOWN if expected is None else
                               PropensityLabel.POSITIVE if expected else PropensityLabel.NEGATIVE)


def test_incomplete_future_even_positive_is_unknown_and_gaps_not_zero():
    coverage = tuple(replace(c, until_utc=T + timedelta(days=20)) if c.component == "orders" else c for c in COVERAGE)
    for events in ([action()], [action(), order(day=1)]):
        row = rows(events, coverage)[0]
        assert row.target is None and "INCOMPLETE_TARGET_COVERAGE" in row.label_reasons
    gaps = (index.HistoryCoverage("orders", T, T + timedelta(days=10), "API"),
            index.HistoryCoverage("orders", T + timedelta(days=11), T + timedelta(days=30), "API"))
    assert not covered(gaps, "orders", index._micros(T), index._micros(T) + 30 * DAY)


@pytest.mark.parametrize("second,identity,expected", [("F", "line", 1), ("Return", "line", 1),
                                                     ("otmenen", "line", 1), ("N", "line", 1),
                                                     ("F", None, 1)])
def test_snapshot_consensus_not_last_write_wins(second, identity, expected):
    variants = (variant(2, identity=identity), variant(2, status=second, identity=identity, key="v2"))
    assert rows([action(), order(*variants)])[0].target == expected
    assert order_outcome(variants) == order_outcome(tuple(reversed(variants)))


def test_positive_order_proves_existence_but_incomplete_coverage_never_does():
    row = rows([action(), order(day=2), replace(order(day=3), identity_valid=False)])[0]
    assert row.target == 1 and row.label_reasons == ()


@pytest.mark.parametrize("change,reason", [
    ({"owner": "someone-else"}, "AMBIGUOUS_OWNER"),
    ({"stamp": None}, "UNRELIABLE_ORDER_TIME"),
    ({"observed": None}, "ORDER_PROVENANCE_MISSING"),
])
def test_unreliable_target_is_not_negative(change, reason):
    first = variant(2)
    row = rows([action(), order(first, replace(first, key="v2", **change))])[0]
    assert row.target is None and reason in row.label_reasons


def test_future_status_change_cannot_rewrite_retro_order_count():
    initial = order(variant(-2, observed=-1, qty="2.500"))
    baseline = rows([action(), initial])[0]
    changed = order(*initial.variants, variant(-2, status="otmenen", observed=10, key="v2"))
    assert rows([action(), changed])[0].features == baseline.features
    assert baseline.features["placed_orders_30d"] == 1
    late_publication = tuple(replace(s, observed_us=index._micros(T + timedelta(days=10))) for s in PUBLICATION)
    late_only = list(customer_rows("customer", [action(), order(day=-2, observed=10)], PLAN, COVERAGE,
                                   "b" * 64, publication=late_publication))[0]
    assert late_only.features["placed_orders_30d"] == 1
    assert late_only.available["placed_orders_30d"]  # Explicit schema-v2 retrospective mode.
    # A backfilled OLD order is allowed in retrospective mode, with the new name.
    old = rows([action()])[0]
    backfill = rows([action(), order(day=-2, observed=10)])[0]
    assert backfill.features["placed_orders_30d"] == old.features["placed_orders_30d"] + 1


@pytest.mark.parametrize("kind", ["VIEW", "FAVORITE", "ORDER"])
def test_future_events_cannot_change_historical_features_or_add_future_cohort(kind):
    baseline = rows([action(), order(day=-2)])[0]
    future = order(day=1) if kind == "ORDER" else action(day=1, kind=kind)
    assert rows([action(), order(day=-2), future])[0].features == baseline.features
    assert rows([future])[0].prediction_utc != baseline.prediction_utc


def test_known_late_action_missing_history_and_observed_absence_are_distinct():
    row = rows([action(day=-1, created=1)])[0]
    assert row.features["views_7d"] is None
    assert row.features["favorites_7d"] == 0
    assert row.features["days_since_favorite_90d"] is None and row.available["days_since_favorite_90d"]
    short = tuple(replace(c, since_utc=T - timedelta(days=60)) for c in COVERAGE)
    row = rows([action(), order(day=-2)], short)[0]
    assert row.features["placed_orders_30d"] == 1
    assert row.features["placed_orders_90d"] is None
    assert not row.available["placed_orders_365d"]


def test_late_export_is_explicitly_allowed_for_retro_placement_features():
    late = tuple(replace(s, observed_us=index._micros(T + timedelta(days=100))) for s in PUBLICATION)
    row = list(customer_rows("customer", [action()], PLAN, COVERAGE, "b" * 64, publication=late))[0]
    assert row.target == 0  # Retrospective label, complete future coverage.
    assert row.features["placed_orders_30d"] == 0
    assert row.available["placed_orders_30d"]


def test_windows_active_days_recency_fractional_units_and_no_money():
    events = [action(-.5), action(-.7), action(-8), action(-40), action(-100), action(-200),
              action(-1, "FAVORITE"), action(-20, "FAVORITE"), action(-70, "FAVORITE"),
              order(day=-1, qty="1.250"), order(day=-40), order(day=-100), order(day=-200)]
    row = rows(events)[0]
    assert [row.features[f"placed_orders_{w}d"] for w in (30, 90, 180, 365)] == [1, 2, 3, 4]
    assert [row.features[f"views_{w}d"] for w in (7, 30, 90, 180)] == [2, 3, 4, 5]
    assert [row.features[f"favorites_{w}d"] for w in (7, 30, 90)] == [1, 2, 3]
    assert row.features["order_active_days_30d"] == 1
    assert row.features["order_active_days_90d"] == 2
    assert row.features["views_active_days_30d"] == 2
    assert row.features["favorites_active_days_30d"] == 2
    assert row.features["days_since_last_order_365d"] == 1
    assert row.features["days_since_view_180d"] == 0
    assert not any("amount" in k or "rub" in k or "kzt" in k for k in row.features)


def test_quantity_uncertainty_does_not_fabricate_zero_or_erase_proven_purchase():
    event = order(variant(-2, qty="1.2"), variant(-2, qty="1.3", key="v2"))
    row = rows([event])[0]
    assert row.features["placed_orders_30d"] == 1
    assert not any("purchase" in key or "units" in key for key in row.features)
    for quantity in ("-1", "0", "NaN", "Infinity", "garbage"):
        assert order_outcome((variant(qty=quantity),)).purchase is None


def test_decimal_addition_is_exact_beyond_default_context_precision():
    row = rows([order(day=-2, qty="123456789012345678901234567890.123456789"), order(day=-1, qty="0.000000001")])[0]
    assert row.features["placed_orders_30d"] == 2  # Quantity is irrelevant, even very large/fractional.


def test_overlapping_grid_targets_are_purged_and_last_date_is_not_today():
    config = replace(CONFIG, step_days=7, end=T + timedelta(days=105),
                     validation_start=T + timedelta(days=35), test_start=T + timedelta(days=70))
    plan = make_plan(config, COVERAGE)
    assert any(plan.split(t) == "embargo" for t in plan.dates)
    by_split = {s: [t for t in plan.dates if plan.split(t) == s] for s in ("train", "validation", "test")}
    assert max(by_split["train"]) + timedelta(days=30) <= min(by_split["validation"])
    assert max(by_split["validation"]) + timedelta(days=30) <= min(by_split["test"])
    capped = make_plan(replace(CONFIG, end=T + timedelta(days=900)), COVERAGE)
    assert max(capped.dates) + timedelta(days=30) <= max(c.until_utc for c in COVERAGE if c.component == "orders")


@pytest.mark.parametrize("kwargs", [{"as_of_level": "POINT_IN_TIME_VERIFIED"}, {"step_days": 0},
                                    {"start": T.replace(tzinfo=None)}, {"start": T + timedelta(hours=1)},
                                    {"test_start": T}, {"max_slices": 0}])
def test_config_refuses_unsupported_dates_and_point_in_time(kwargs):
    with pytest.raises(ValueError):
        PropensityDatasetConfig(**kwargs)


class SyntheticIndex:
    def __init__(self, tmp_path, monkeypatch):
        self.builder = build.PropensityDatasetBuilder(raw_root=tmp_path / "raw", cache_dir=tmp_path / "cache",
                                                       output_dir=tmp_path / "output")
        self.path = self.builder._path()
        self.path.parent.mkdir(parents=True)
        self.sources = SimpleNamespace(version="source-v", components=tuple(SimpleNamespace(
            name=c.component, since=c.since_utc, until=c.until_utc, source_kind=c.source_kind) for c in COVERAGE))
        monkeypatch.setattr(index, "_reader_lock", lambda _: nullcontext())
        monkeypatch.setattr(index, "_capture_sources", lambda _: self.sources)
        monkeypatch.setattr(build, "make_sqlite_temp_locator", lambda _: lambda: tmp_path)
        monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=64 * 1024**3))
        self.db = sqlite3.connect(self.path)
        self.db.executescript(index._SCHEMA)
        self.db.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", 2), ("normalizer_version", index.records.HISTORY_SCHEMA_VERSION),
            ("schema_definition", index._SCHEMA_FINGERPRINT), ("schema_signature", index._schema_signature(self.db)),
            ("source_version", "source-v"), ("generation", "a" * 32),
            ("counts", json.dumps(asdict(index.HistoryIndexCounts())))])
        self.db.commit()

    def add(self, event, key="event", *, aliases=(), namespace="offline1C", commit=True, repeats=1):
        head = event.variants[0]
        stamp = event.stamp()
        event_key = history_event_key(HistorySource(event.source), key)
        self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            event_key, event.source, key, head.owner if event.owned_by(head.owner) else None, head.kind, stamp,
            len(event.variants), len(event.variants) * repeats, int(not event.owned_by(head.owner)), int(stamp is None), 0))
        for owner in sorted({v.owner for v in event.variants}):
            self.db.execute("INSERT INTO event_customers VALUES (?,?,?)", (event_key, owner, stamp))
        for n, v in enumerate(event.variants):
            vk = key + v.key
            fingerprint = hashlib.sha256(vk.encode()).digest()
            self.db.execute("INSERT INTO variants VALUES (?,?,?,?,?,?,?,?)", (
                vk, event_key, fingerprint, v.owner, aliases[n] if n < len(aliases) else v.owner, v.kind,
                v.stamp if v.stamp is not None else head.stamp, "[]"))
            if v.observed is not None:
                self.db.execute("INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    vk, "orders" if event.source == "ORDER" else "actions", "unused", index._micros(COVERAGE[0].since_utc),
                    index._micros(COVERAGE[0].until_utc), v.observed, "API", None, "synthetic", repeats))
                for r in range(repeats):
                    self.db.execute("INSERT INTO record_origins VALUES (?,?,?)", (vk, vk, r))
            if event.source == "ORDER":
                self.db.execute("INSERT INTO orders VALUES (?,?)", (event_key, key)) if n == 0 else None
                self.db.execute("INSERT INTO order_snapshots VALUES (?,?,?,?,?,?,?,?,?,?)",
                                (vk, vk, None, v.stamp if v.stamp is not None else head.stamp, None, None, None, None, None, None))
                for line in v.lines:
                    self.db.execute("INSERT INTO order_lines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        vk + str(line.ordinal), vk, line.identity, str(line.ordinal), line.identity, int(line.identity is None),
                        line.ordinal, line.ordinal + 1, "Synthetic product", line.quantity, line.status, "123.123456789", None, "[]"))
                    self.db.execute("INSERT INTO product_occurrences VALUES (?,?,?,?,?,?)", (
                        vk, "line", line.ordinal, namespace, line.product, "[]"))
            else:
                self.db.execute("INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                    vk, key, "synthetic", v.created, None, None, None, None, None, None, None, None))
        self.db.execute("UPDATE metadata SET value=? WHERE key='counts'", (json.dumps(asdict(index._counts(self.db))),))
        if commit:
            self.db.commit()

    def run(self, **kwargs):
        return self.builder.build(CONFIG, pseudonym_key=SECRET, allow_retrospective=True, min_free_bytes=resources.MIN_FREE_BYTES, **kwargs)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    value = SyntheticIndex(tmp_path, monkeypatch)
    yield value
    value.db.close()


def load_rows(result, name):
    manifest = json.loads((result.artifact_dir / "manifest.json").read_text())
    return list(iter_propensity_rows(result.artifact_dir, name, purpose=manifest["purpose"]))


def test_streaming_sql_deduplicates_origins_variants_merges_and_currencies(tiny):
    tiny.add(action(-1), "a", aliases=("historical-alias",), repeats=3)
    tiny.add(order(day=-2, qty="1.5"), "o-rub", namespace="offline1C", repeats=2)
    tiny.add(order(day=-3, qty="2.25"), "o-kzt", namespace="kanzlerKz")
    with index._read_index(tiny.path) as db:
        db.row_factory = sqlite3.Row
        facts = list(iter_facts(db, lambda: False))
    assert len(facts) == 3 and {customer for customer, _ in facts} == {"customer"}
    result = tiny.run()
    row = load_rows(result, "train")[0]
    assert row["features"]["views_7d"] == 1
    assert row["features"]["placed_orders_30d"] == 2
    assert not any("units" in key for key in row["features"])
    assert all(not event.variants[0].lines for _, event in facts)
    assert set(row["features"]) == set(feature_schema())
    assert "historical-alias" not in json.dumps(row)


def test_build_manifest_synthetic_quality_counts_and_natural_rate(tiny):
    # Same v1 cohort: cancellation now proves placement; 2 positive, 7 negative.
    for customer in ("positive-person", "negative-person", "unknown-person"):
        tiny.add(action(-1, owner=customer), customer)
    tiny.add(order(day=1, owner="positive-person"), "positive")
    tiny.add(order(day=1, owner="unknown-person", status="CD"), "unknown")
    result = tiny.run()
    manifest = json.loads((result.artifact_dir / "manifest.json").read_text())
    assert (result.counts["rows"], result.counts["positive"], result.counts["negative"], result.counts["unknown"]) == (9, 2, 7, 0)
    assert result.counts["positive_rate"] == 2 / 9
    assert result.counts["unknown_reasons"] == {}
    assert manifest["schema_version"] == 2 and manifest["target"] == "order_placed_within_30_days"
    assert manifest["business_validation"] == "REQUIRES_BUSINESS_VALIDATION"
    assert manifest["as_of_level"] == AS_OF_LEVEL
    assert manifest["status"] == "RESEARCH_DATASET_ONLY"
    assert manifest["counts"]["repeated_customers"] == 3
    assert manifest["distributions"]["rows_per_customer"] == {"3": 3}
    for name, checksum in manifest["checksums"].items():
        assert build._digest(result.artifact_dir / name) == checksum
    all_text = "\n".join(gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text() for p in result.artifact_dir.iterdir())
    for forbidden in ("positive-person", "negative-person", "unknown-person", SECRET.decode(), "raw_status", "phone", "email"):
        assert forbidden not in all_text
    assert json.loads((tiny.builder.output_dir / "CURRENT.json").read_text())["version"] == result.version
    assert not list(tiny.builder.output_dir.glob(".building-*"))


def test_preflight_is_metadata_only_and_estimates_dates_not_customers(tiny, monkeypatch):
    monkeypatch.setattr(build, "iter_facts", Mock(side_effect=AssertionError("No event scan")))
    check = Mock(wraps=index._inspect_connection)
    monkeypatch.setattr(index, "_inspect_connection", check)
    report = tiny.builder.inspect(CONFIG)
    assert report["possible_slices"] == 3 and report["row_estimate"] is None
    assert report["slices_by_split"] == {"train": 1, "validation": 1, "test": 1}
    assert check.call_args.kwargs.get("verify_integrity", False) is False
    assert not tiny.builder.output_dir.exists()


def test_deterministic_reordered_variants_repeated_runs_and_distinct_key(tiny):
    first, second = variant(-2, qty="1.25"), variant(-2, qty="1.25", key="v2")
    tiny.add(order(second, first), "o")
    result = tiny.run()
    again = tiny.run()
    assert again.version == result.version and again.reused
    assert rows([order(first, second)])[0].features == rows([order(second, first)])[0].features
    other = tiny.builder.build(CONFIG, pseudonym_key=b"other secret" * 4, allow_retrospective=True, min_free_bytes=resources.MIN_FREE_BYTES)
    assert other.version != result.version
    assert load_rows(other, "train")[0]["customer_key"] != load_rows(result, "train")[0]["customer_key"]


def test_cancel_mid_query_and_reuse_after_cancel(tiny):
    tiny.add(action(), "a")
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 6

    with pytest.raises(PropensityCancelled):
        tiny.run(cancelled=cancelled)
    assert not (tiny.builder.output_dir / "CURRENT.json").exists()
    assert not list(tiny.builder.output_dir.glob(".building-*"))
    assert tiny.run().status == "RESEARCH_DATASET_ONLY"


@pytest.mark.parametrize("mode", ["source", "generation", "disk", "callback", "corrupt_output", "pointer"])
def test_failures_preserve_previous_current_and_cleanup(tiny, monkeypatch, mode):
    tiny.add(action(), "a")
    previous = tiny.run()
    pointer = (tiny.builder.output_dir / "CURRENT.json").read_bytes()
    original = build.os.replace

    def progress(value):
        if value.phase != "validate_output":
            return
        if mode == "source":
            tiny.sources.version = "changed"
        elif mode == "generation":
            tiny.db.execute("UPDATE metadata SET value=? WHERE key='generation'", ("b" * 32,))
            tiny.db.commit()
        elif mode == "callback":
            raise RuntimeError("synthetic failure")
        elif mode == "corrupt_output":
            staging = next(tiny.builder.output_dir.glob(".building-*"))
            (staging / "train.jsonl.gz").write_bytes(gzip.compress(b"{}\n"))

    if mode == "disk":
        monkeypatch.setattr(build.shutil, "disk_usage", lambda _: SimpleNamespace(free=-1))
    if mode == "pointer":
        def replace_pointer(source, destination):
            if Path(destination).name == "CURRENT.json":
                raise OSError("synthetic failure")
            original(source, destination)
        monkeypatch.setattr(build.os, "replace", replace_pointer)
    with pytest.raises((PropensityDataError, RuntimeError, OSError)):
        tiny.run(progress=progress)
    assert (tiny.builder.output_dir / "CURRENT.json").read_bytes() == pointer
    assert previous.artifact_dir.exists()
    assert not list(tiny.builder.output_dir.glob(".building-*"))
    assert not list(tiny.builder.output_dir.glob(".current-*"))


def test_unknown_only_artifact_never_becomes_current(tiny):
    # Unknown date intersects every target window and a separate action supplies cohort.
    tiny.add(action(), "a")
    bad = order(variant(1), replace(variant(1, key="v2"), stamp=index._micros(T + timedelta(days=70))))
    tiny.add(bad, "conflict")
    # For all three windows, at least one variant must potentially fall in it.
    tiny.db.execute("UPDATE order_snapshots SET order_us=order_us+1")
    tiny.db.commit()
    result = tiny.run()
    assert result.status == "BLOCKED_BY_DATA_QUALITY" and result.counts["unknown"] == 3
    assert not (tiny.builder.output_dir / "CURRENT.json").exists()


def test_damaged_schema_missing_index_and_source_inconsistency_fail_closed(tiny, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(tiny.builder, "_path", lambda: tiny.path.with_suffix(".missing"))
        with pytest.raises(FileNotFoundError):
            tiny.builder.inspect(CONFIG)
    tiny.db.execute("UPDATE metadata SET value='bad' WHERE key='schema_signature'")
    tiny.db.commit()
    with pytest.raises(ValueError):
        tiny.run()
    assert not (tiny.builder.output_dir / "CURRENT.json").exists()


@pytest.mark.parametrize("damage", ["missing_event_membership", "missing_variant"])
def test_incomplete_index_content_cannot_publish_false_negative_labels(tiny, damage):
    tiny.add(action(), "cohort")
    tiny.add(order(day=1), "future")
    if damage == "missing_event_membership":
        tiny.db.execute("DELETE FROM event_customers WHERE event_key=?", (history_event_key(HistorySource.ORDER, "future"),))
    else:
        tiny.db.execute("DELETE FROM variants WHERE event_key=?", (history_event_key(HistorySource.ORDER, "future"),))
    tiny.db.commit()
    with pytest.raises(PropensityDataError):
        tiny.run()
    assert not (tiny.builder.output_dir / "CURRENT.json").exists()


def test_bounded_customer_reducer_memory_for_long_history():
    def measure(n):
        event = action(-1)
        tracemalloc.start()
        result = list(customer_rows("customer", (event for _ in range(n)), PLAN, COVERAGE, "b" * 64))
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert result[0].features["views_7d"] == n
        return peak
    small, large = measure(100), measure(10000)
    assert large < 1024**2 and large < small + 250_000


def test_cli_inspect_build_opt_in_safe_errors_and_no_import_build(tiny, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "PropensityDatasetBuilder", lambda **unused: tiny.builder)
    arguments = ["--start", "2024-01-01T00:00:00Z", "--end", "2024-03-01T00:00:00Z"]
    assert cli.main(arguments) == 0
    assert not tiny.builder.output_dir.exists()
    assert "possible_slices" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--build"])
    secret = tmp_path / "key.bin"
    secret.write_bytes(SECRET)
    tiny.add(action(), "a")
    assert cli.main(arguments + ["--build", "--allow-retrospective", "--key-file", str(secret), "--min-free-gib", "20"]) == 0
    monkeypatch.setattr(tiny.builder, "inspect", Mock(side_effect=ValueError("private-data")))
    assert cli.main(arguments) == 2
    assert "private-data" not in capsys.readouterr().out


def test_multiple_product_lines_are_one_placed_order_not_snapshots():
    v = variant(-1, qty="1.2500")
    v = replace(v, lines=v.lines + (LineFact("another-line", 1, "CP", "2.5", "000002"),))
    row = rows([order(v, replace(v, key="different-fingerprint"))])[0]
    assert row.features["placed_orders_30d"] == 1
    assert row.features["has_order_365d"] == 1
    assert order_outcome((v,)).units == order_outcome((v, replace(v, lines=tuple(reversed(v.lines))))).units
    scaled = replace(v, lines=(replace(v.lines[0], quantity="1.25"), v.lines[1]))
    assert rows([order(v, scaled)])[0].features == rows([order(scaled, v)])[0].features


def test_current_kpi_and_profiles_are_not_feature_sources(tiny, monkeypatch):
    from Application.customer_activity_service import CustomerActivityService
    monkeypatch.setattr(CustomerActivityService, "get", Mock(side_effect=AssertionError("No current KPI")))
    tiny.add(action(), "a")
    first = tiny.run()
    # No profile/catalog is consulted, even when unrelated current files change.
    (tiny.builder.raw_root.parent / "profile.csv").write_text("current profile changed")
    assert tiny.run().version == first.version


def test_bounded_sql_reader_query_count_and_cancel_in_sql_sort(tiny):
    for n in range(100):
        tiny.add(action(), f"a-{n}", commit=False)
    tiny.db.commit()
    with index._read_index(tiny.path) as db:
        queries = []
        db.set_trace_callback(queries.append)
        db.row_factory = sqlite3.Row
        assert len(list(iter_facts(db, lambda: False))) == 100
        assert len(queries) == 2  # No per-customer/per-date SQL or raw export reads.
    with index._read_index(tiny.path) as db:
        db.row_factory = sqlite3.Row
        db.set_progress_handler(lambda: 1, 1)
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            list(iter_facts(db, lambda: False))


@pytest.fixture
def canonical_history_with_late_merges(tmp_path):
    """Actual portable adapters/identity/index, independent of native TEMP discovery."""
    from Application.mindbox.selection import DEFAULT_SELECTION
    root, cache = tmp_path / "canonical-input", tmp_path / "index"
    since, until, updated = "2022-01-01T00:00:00Z", "2024-04-01T00:00:00Z", "2024-05-01T00:00:00Z"
    actions = [{"ids": {"mindboxId": "a1"}, "customer": {"ids": {"mindboxId": "old-alias"}},
                "dateTimeUtc": "2023-12-31T12:00:00Z", "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}}}]
    orders = [{"ids": {"mindboxId": "o1"}, "customer": {"ids": {"mindboxId": "old-alias"}},
               "firstAction": {"dateTimeUtc": "2024-01-02T12:00:00Z",
                               "channel": {"ids": {"externalId": "synthetic"}, "name": "Synthetic"}},
               "lines": [{"id": "line1", "number": 1, "quantity": 1.5, "priceOfLine": 10,
                          "product": {"ids": {"offline1C": "000001-size"}}, "status": {"ids": {"externalId": "CP"}}}]}]
    merges = [{"id": "m1", "dateTimeUtc": "2024-03-01T00:00:00Z",
               "resultingCustomer": {"ids": {"mindboxId": "canonical-person"}},
               "mergedCustomers": [{"ids": {"mindboxId": "old-alias"}}]}]

    def source(name, token, data):
        directory = root / "canonical" / "objects" / (token * 32) / name
        directory.mkdir(parents=True)
        collection = {"actions": "customerActions", "orders": "orders", "customer_merges": "customerMerges"}[name]
        (directory / f"{name}_part_001.json").write_text(json.dumps({collection: data}), encoding="utf-8")
        return {"directory": directory.relative_to(root).as_posix(), "since": since, "until": until,
                "updated": updated, "parts": 1, "source_kind": "API" if name == "customer_merges" else "MANUAL",
                "export_id": "synthetic" if name == "customer_merges" else None,
                "operation": "synthetic" if name == "customer_merges" else "MANUAL"}

    catalog = {"schema_version": 2, "revision": "synthetic", "selection": asdict(DEFAULT_SELECTION),
               "customer_merges": source("customer_merges", "1", merges), "actions": {}, "orders": {},
               "manual_interactions": {"since": since, "until": until, "updated": updated,
                                       "actions": source("actions", "2", actions), "orders": source("orders", "2", orders)}}
    (root / "canonical/catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    (root / "canonical/.writer.lock").write_bytes(b"0")
    result = index.CustomerHistoryIndexService(cache_dir=cache).build_index(raw_root=root)
    assert result.status is index.HistoryIndexStatus.READY
    paths = tuple(root.rglob("*.json"))
    before = {str(p): p.read_bytes() for p in paths}
    return root, cache, paths, before


@pytest.mark.skipif(resources.os.name != "nt", reason="Native Windows SQLite TEMP verification is required for publication")
def test_canonical_json_to_v2_to_dataset_with_late_identity_merges(tmp_path, canonical_history_with_late_merges):
    """Windows end-to-end publication; never mock or bypass TEMP verification."""
    root, cache, paths, before = canonical_history_with_late_merges
    builder = build.PropensityDatasetBuilder(raw_root=root, cache_dir=cache, output_dir=tmp_path / "dataset")
    artifact = builder.build(CONFIG, pseudonym_key=SECRET, allow_retrospective=True, min_free_bytes=resources.MIN_FREE_BYTES)
    assert artifact.counts["customers"] == 1 and artifact.counts["rows"] == 3
    assert artifact.counts["positive"] == 1 and artifact.counts["negative"] == 2
    training_row = load_rows(artifact, "train")[0]
    assert training_row["target"] == 1
    assert training_row["features"]["placed_orders_30d"] == 0  # Retrospective: no past order.
    assert {str(p): p.read_bytes() for p in paths} == before
    assert not any(p.exists() for p in (Path(str(builder._path()) + s) for s in ("-wal", "-shm", "-journal")))


def test_canonical_json_to_v2_preserves_late_identity_and_rows(canonical_history_with_late_merges):
    """Portable core contract remains tested even where publication is unsupported."""
    root, cache, paths, before = canonical_history_with_late_merges
    service = index.CustomerHistoryIndexService(cache_dir=cache)
    with index._read_index(service.index_path(raw_root=root)) as connection:
        connection.row_factory = sqlite3.Row
        facts = list(iter_facts(connection, lambda: False))
    assert len(facts) == 2 and {customer for customer, _ in facts} == {"canonical-person"}
    assert all(event.owned_by("canonical-person") for _, event in facts)
    status = service.get_status(raw_root=root, verify_integrity=False)
    plan = make_plan(CONFIG, status.coverage)
    rows = list(customer_rows("canonical-person", (event for _, event in facts), plan, status.coverage, "synthetic-pseudonym"))
    assert len(rows) == 3
    assert [row.target for row in rows] == [1, 0, 0]
    assert rows[0].features["placed_orders_30d"] == 0
    assert {str(p): p.read_bytes() for p in paths} == before


@pytest.mark.skipif(resources.os.name == "nt", reason="Unsupported native TEMP runtime contract is exercised on non-Windows")
def test_canonical_dataset_publication_fails_closed_without_windows(tmp_path, canonical_history_with_late_merges):
    root, cache, paths, before = canonical_history_with_late_merges
    builder = build.PropensityDatasetBuilder(raw_root=root, cache_dir=cache, output_dir=tmp_path / "dataset")
    with pytest.raises(PropensityDataError, match="^SQLITE_TEMP_UNVERIFIED$"):
        builder.build(CONFIG, pseudonym_key=SECRET, allow_retrospective=True, min_free_bytes=resources.MIN_FREE_BYTES)
    assert {str(p): p.read_bytes() for p in paths} == before
    assert not list(builder.output_dir.rglob("manifest.json"))
    assert not list(builder.output_dir.glob(".building-*"))


@pytest.mark.parametrize("change", ["status", "quantity", "price_product", "line_identity"])
def test_order_header_consensus_ignores_all_product_snapshot_differences(tiny, change):
    first = variant(-10)
    other = replace(first, key="v2", observed=100)
    if change == "status":
        other = replace(other, lines=(replace(first.lines[0], status="CD"),))
    elif change == "quantity":
        other = replace(other, lines=(replace(first.lines[0], quantity="-123.456"),))
    elif change == "price_product":
        other = replace(other, lines=(replace(first.lines[0], product="999999"),))
    else:
        other = replace(other, lines=(replace(first.lines[0], identity=None),))
    tiny.add(order(first, other), "snapshots", repeats=3)
    if change == "price_product":
        tiny.db.execute("UPDATE order_lines SET price_of_line='999999.123456789' WHERE variant_key='snapshotsv2'")
        tiny.db.commit()
    result = tiny.run()
    row = load_rows(result, "train")[0]
    assert row["features"]["placed_orders_30d"] == 1
    assert result.counts["logical_orders"] == 1 and result.counts["order_snapshots"] == 2
    assert "purchase" not in json.dumps(row)


@pytest.mark.parametrize("damage", ["order_id", "event_key", "snapshot_key"])
def test_unreliable_order_identity_is_unknown_not_a_placement(tiny, damage):
    tiny.add(action(), "cohort")
    tiny.add(order(day=1), "candidate")
    if damage == "order_id":
        tiny.db.execute("UPDATE orders SET order_id='different-identity'")
    elif damage == "event_key":
        tiny.db.execute("UPDATE events SET source_event_id='different-identity' WHERE source='ORDER'")
    else:
        tiny.db.execute("UPDATE order_snapshots SET snapshot_key='different-identity'")
    tiny.db.commit()
    result = tiny.run()
    unknown = load_rows(result, "unknown")
    assert len(unknown) == 1 and unknown[0]["label_reasons"] == ["INVALID_ORDER_IDENTITY"]
    assert unknown[0]["target"] is None


def test_another_uncertain_order_cannot_remove_proven_positive_and_no_proof_is_unknown(tiny):
    tiny.add(action(), "a")
    tiny.add(order(variant(2), variant(2, owner="another", key="v2")), "conflict")
    assert tiny.run().counts["unknown"] == 1
    tiny.add(order(day=3, status="RJ"), "proof")
    result = tiny.run()
    assert result.counts["positive"] == 1 and result.counts["unknown"] == 0
    assert load_rows(result, "train")[0]["target"] == 1


@pytest.mark.parametrize("days,expected", [([], None), ([-1], None), ([-11, -1], "10"), ([-1, -1], "0")])
def test_average_order_intervals_and_order_active_days(days, expected):
    row = rows([action(), *(order(day=d, status="CD") for d in days)])[0]
    assert row.features["average_days_between_orders_365d"] == expected
    assert row.available["average_days_between_orders_365d"]
    assert row.features["placed_orders_365d"] == len(days)
    assert row.features["has_order_365d"] == int(bool(days))
    assert row.features["order_active_days_30d"] == len(set(days))


@pytest.mark.parametrize("offset", [-5, 0, 3])
def test_first_action_offsets_are_normalized_to_utc_boundary(offset):
    from Application.mindbox.adapters._common import timestamp
    local = T.astimezone(timezone(timedelta(hours=offset)))
    normalized = timestamp({"firstAction": {"dateTimeUtc": local.isoformat()}}, "firstAction.dateTimeUtc")
    fact = order(replace(variant(), stamp=index._micros(normalized)))
    row = rows([action(), fact])[0]
    assert row.target == 1 and row.features["placed_orders_30d"] == 0


def long_coverage(order_until, action_until=None):
    start = order_until - timedelta(days=900)
    return (index.HistoryCoverage("orders", start, order_until, "API"),
            index.HistoryCoverage("actions", start, action_until or order_until, "API"))


def test_inference_date_uses_latest_common_complete_boundary_and_full_lookbacks():
    end = datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert latest_prediction_date(long_coverage(end)) == end
    assert latest_prediction_date(long_coverage(end + timedelta(days=31), end)) == end
    assert latest_prediction_date(long_coverage(end, end - timedelta(days=10))) == end - timedelta(days=10)
    # A stray later partition cannot move the date past another required source.
    coverage = long_coverage(end) + (index.HistoryCoverage("orders", end + timedelta(days=40), end + timedelta(days=41), "API"),)
    assert latest_prediction_date(coverage) == end
    assert latest_prediction_date(long_coverage(end + timedelta(days=30))) == end + timedelta(days=30)
    partial_day = long_coverage(end + timedelta(hours=12))
    assert latest_prediction_date(partial_day) == end


@pytest.mark.parametrize("component,lookback", [("orders", 365), ("actions", 180)])
def test_latest_common_run_with_a_gap_in_required_history_is_rejected(component, lookback):
    end = datetime(2026, 9, 1, tzinfo=timezone.utc)
    coverage = long_coverage(end)
    spans = tuple(s for s in coverage if s.component != component) + (
        index.HistoryCoverage(component, end - timedelta(days=800), end - timedelta(days=lookback - 2), "API"),
        index.HistoryCoverage(component, end - timedelta(days=lookback - 3), end, "API"))
    with pytest.raises(PropensityDataError, match="INFERENCE_HISTORY_INCOMPLETE"):
        latest_prediction_date(spans)


def test_disjoint_source_periods_cannot_create_a_prediction_date():
    end = T + timedelta(days=500)
    spans = (index.HistoryCoverage("orders", T - timedelta(days=900), T, "API"),
             index.HistoryCoverage("actions", T + timedelta(days=1), end, "API"))
    with pytest.raises(PropensityDataError, match="NO_COMMON_COMPLETE_DAY"):
        latest_prediction_date(spans)


def test_inference_reuses_training_features_but_has_no_label():
    config = PropensityDatasetConfig(purpose="inference")
    end = T
    coverage = long_coverage(end)
    inference = make_plan(config, coverage)
    events = [action(-1), order(day=-2, status="otmenen"), order(day=0)]
    feature_row = list(customer_rows("customer", events, inference, coverage, "b" * 64))[0]
    assert type(feature_row) is PropensityInferenceRow
    assert feature_row.features == rows(events, coverage)[0].features
    assert feature_row.features["placed_orders_30d"] == 1
    assert set(feature_row.payload()).isdisjoint({"target", "label_state", "label_reasons"})
    assert feature_row.split == "inference" and feature_row.prediction_utc == "2024-01-01T00:00:00Z"
    assert rows(events, coverage)[0].target is None  # This date cannot be a training negative.
    with pytest.raises(ValueError, match="exclusively"):
        PropensityDatasetConfig(purpose="inference", start=T)


def test_inference_artifact_is_separate_from_training_and_filters_future_events(tiny, monkeypatch):
    tiny.add(action(), "cohort")
    training = tiny.run()
    old_pointer = (tiny.builder.output_dir / "CURRENT.json").read_bytes()
    tiny.add(order(day=499, status="CD"), "last-order")
    tiny.add(order(day=500, owner="future-only"), "future-order")
    config = PropensityDatasetConfig(purpose="inference")
    real_iter = build.iter_facts
    calls = []

    def only_past(connection, cancelled, *, before_us=None):
        calls.append(before_us)
        for customer, event in real_iter(connection, cancelled, before_us=before_us):
            assert customer != "future-only"
            yield customer, event

    monkeypatch.setattr(build, "iter_facts", only_past)
    result = tiny.builder.build(config, pseudonym_key=SECRET, allow_retrospective=True, min_free_bytes=resources.MIN_FREE_BYTES)
    manifest = json.loads((result.artifact_dir / "manifest.json").read_text())
    inference = load_rows(result, "inference")
    assert len(inference) == 1 and calls == [index._micros(T + timedelta(days=500))]
    assert inference[0]["features"]["placed_orders_30d"] == 1
    assert all(name not in inference[0] for name in ("target", "label_state", "label_reasons"))
    assert all(not load_rows(result, name) for name in ("train", "validation", "test", "unknown", "embargo"))
    assert result.counts["unlabeled"] == 1 and result.counts["positive"] == result.counts["negative"] == result.counts["unknown"] == 0
    assert manifest["purpose"] == "inference" and not manifest["label_fields_present"]
    assert (tiny.builder.output_dir / "CURRENT.json").read_bytes() == old_pointer
    assert json.loads((tiny.builder.output_dir / "CURRENT_INFERENCE.json").read_text())["version"] == result.version
    assert training.artifact_dir.exists()


def test_changing_wall_clock_cannot_change_inference_cutoff_or_version(tiny, monkeypatch):
    tiny.add(action(), "cohort")
    config = PropensityDatasetConfig(purpose="inference")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return T.replace(year=2035)

    monkeypatch.setattr(build, "datetime", Clock)
    first = tiny.builder.build(config, pseudonym_key=SECRET, allow_retrospective=True, min_free_bytes=resources.MIN_FREE_BYTES)

    class LaterClock(Clock):
        @classmethod
        def now(cls, tz=None):
            return T.replace(year=2045)

    monkeypatch.setattr(build, "datetime", LaterClock)
    second = tiny.builder.build(config, pseudonym_key=SECRET, allow_retrospective=True, min_free_bytes=resources.MIN_FREE_BYTES)
    assert first.version == second.version and second.reused
    assert load_rows(first, "inference")[0]["prediction_utc"] == (T + timedelta(days=500)).isoformat().replace("+00:00", "Z")


def test_old_purchase_artifacts_are_preserved_and_not_reused_as_v2(tiny):
    tiny.add(action(), "cohort")
    output = tiny.builder.output_dir
    old = output / "schema-one"
    old.mkdir(parents=True)
    old_manifest = {"schema_version": 1, "target": "purchase", "features": {"purchase_orders_30d": {}}}
    (old / "manifest.json").write_text(json.dumps(old_manifest))
    (output / "CURRENT.json").write_text(json.dumps({"version": "schema-one"}))
    result = tiny.run()
    assert result.version != "schema-one" and not result.reused
    assert json.loads((old / "manifest.json").read_text()) == old_manifest
    manifest = json.loads((result.artifact_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION == 2 and manifest["target"] == TARGET
    assert len(manifest["features"]) == 22


def test_purchase_policy_is_unchanged_and_retrospective_opt_in_remains_required(tiny):
    from Application.mindbox.selection import DEFAULT_PURCHASE_STATUSES
    from Application.purchased_products import classify
    assert DEFAULT_PURCHASE_STATUSES == ("CP", "delivering", "F")
    assert classify("N") == classify("CD") == classify("sold") == "UNKNOWN"
    assert classify("otmenen") == "CANCELLED" and classify("Return") == "RETURN"
    assert rows([action(), order(day=1, status="CD")])[0].target == 1
    with pytest.raises(ValueError, match="opt-in"):
        tiny.builder.build(CONFIG, pseudonym_key=SECRET, min_free_bytes=resources.MIN_FREE_BYTES)
    with pytest.raises(PropensityDataError, match="POINT_IN_TIME_IDENTITY_UNVERIFIED"):
        PropensityDatasetConfig(as_of_level="POINT_IN_TIME_VERIFIED")
    assert not tiny.builder.output_dir.exists()


def test_cli_prepares_inference_only_and_explains_strict_rejection(tiny, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "PropensityDatasetBuilder", lambda **unused: tiny.builder)
    tiny.add(action(), "a")
    key = tmp_path / "key.bin"
    key.write_bytes(SECRET)
    assert cli.main(["--purpose", "inference", "--build", "--allow-retrospective", "--key-file", str(key),
                     "--min-free-gib", "20"]) == 0
    assert (tiny.builder.output_dir / "CURRENT_INFERENCE.json").exists()
    assert not (tiny.builder.output_dir / "CURRENT.json").exists()
    assert cli.main(["--as-of-level", "POINT_IN_TIME_VERIFIED"]) == 2
    assert "POINT_IN_TIME_IDENTITY_UNVERIFIED" in capsys.readouterr().out

# Storage/probe regression tests: tiny synthetic indexes only.


def read_manifest(result):
    return json.loads((result.artifact_dir / "manifest.json").read_text())


def probe_config(**kwargs):
    return build.PropensityProbeConfig((PLAN.dates[0], PLAN.dates[-1]), **kwargs)


def reseal_file(result, name, content):
    manifest = read_manifest(result)
    filename = manifest["file_names"][name]
    path = result.artifact_dir / filename
    path.write_bytes(content)
    manifest["file_sizes"][filename] = len(content)
    manifest["checksums"][filename] = build._digest(path)
    (result.artifact_dir / "manifest.json").write_text(json.dumps(manifest))


def test_streaming_plain_gzip_exact_equivalence_and_legacy_reader(tiny):
    for customer in ("p", "n", "u"):
        tiny.add(action(-1, owner=customer), "activity-" + customer)
    tiny.add(order(day=1, owner="p"), "positive")
    tiny.add(order(day=1, owner="u", observed=None), "unknown")
    tiny.add(order(day=-2, owner="p"), "past-1")
    tiny.add(order(day=-3, owner="p"), "past-2")
    plain = tiny.run(storage=PropensityStorage("jsonl"))
    old_bytes = {p.name: p.read_bytes() for p in plain.artifact_dir.iterdir()}
    zipped = tiny.run()
    manifest = read_manifest(zipped)
    assert manifest["schema_version"] == 2 and manifest["storage_contract_version"] == 1
    assert manifest["compression"] == {"codec": "gzip", "level": 6, "mtime": 0, "filename": ""}
    assert set(manifest["checksums"]) == set(manifest["file_sizes"]) == set(manifest["file_names"].values())
    assert zipped.version != plain.version
    labels = set()
    for name in build.FILES:
        decoded = gzip.decompress((zipped.artifact_dir / (name + ".jsonl.gz")).read_bytes())
        assert decoded == old_bytes[name + ".jsonl"]
        loaded = load_rows(zipped, name)
        assert loaded == load_rows(plain, name)
        for row in loaded:
            labels.add(row["label_state"])
            assert len(row["features"]) == len(row["available"]) == 22
            value = row["features"]["placed_orders_30d"]
            assert type(value) is int if row["available"]["placed_orders_30d"] else value is None
            avg = row["features"]["average_days_between_orders_365d"]
            assert avg is None or isinstance(avg, str)
    assert labels == {"POSITIVE", "NEGATIVE", "UNKNOWN_LABEL"}
    assert {p.name: p.read_bytes() for p in plain.artifact_dir.iterdir()} == old_bytes
    # Original v2 manifests had no explicit storage fields.
    legacy = read_manifest(plain)
    for key in ("storage_contract_version", "storage_format", "compression", "file_names", "file_sizes"):
        legacy.pop(key)
    (plain.artifact_dir / "manifest.json").write_text(json.dumps(legacy))
    assert load_rows(plain, "train") == load_rows(zipped, "train")
    assert not list(zipped.artifact_dir.glob("*.jsonl"))


@pytest.mark.parametrize("level", [0, 1, 6, 9])
def test_gzip_determinism_includes_header_and_file_hashes(tiny, level):
    tiny.add(action(), "a")
    first = tiny.run(storage=PropensityStorage(level=level))
    again = tiny.run(storage=PropensityStorage(level=level))
    assert first.version == again.version and again.reused
    for name in build.FILES:
        value = (first.artifact_dir / (name + ".jsonl.gz")).read_bytes()
        assert value[:3] == b"\x1f\x8b\x08" and value[3] & 8 == 0 and value[4:8] == b"\0" * 4


@pytest.mark.parametrize("damage", ["header", "truncated", "crc", "json", "count", "sha", "size", "number", "null"])
def test_stream_reader_rejects_damage(tiny, damage):
    tiny.add(action(), "a")
    result = tiny.run()
    path = result.artifact_dir / "train.jsonl.gz"
    value = path.read_bytes()
    if damage == "header":
        reseal_file(result, "train", b"not gzip")
    elif damage == "truncated":
        reseal_file(result, "train", value[:-5])
    elif damage == "crc":
        reseal_file(result, "train", value[:-8] + bytes([value[-8] ^ 1]) + value[-7:])
    elif damage == "json":
        reseal_file(result, "train", gzip.compress(b"not-json\n"))
    elif damage in ("number", "null"):
        row = load_rows(result, "train")[0]
        row["features"]["views_7d"] = 1.0 if damage == "number" else None
        reseal_file(result, "train", gzip.compress((build._json(row) + "\n").encode()))
    else:
        manifest = read_manifest(result)
        if damage == "count":
            manifest["counts"]["files"]["train"] += 1
        elif damage == "sha":
            manifest["checksums"]["train.jsonl.gz"] = "0" * 64
        elif damage == "size":
            manifest["file_sizes"]["train.jsonl.gz"] += 1
        (result.artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PropensityDataError):
        load_rows(result, "train")


@pytest.mark.parametrize("field,value", [("schema_version", 1), ("schema_version", True), ("target", "purchase"),
    ("purpose", "inference"), ("storage_format", "parquet"), ("compression", {"codec": "none"}),
    ("storage_contract_version", 99), ("label_fields_present", False), ("features", {}), ("horizon_days", 31),
    ("file_names", {"train": "../outside.jsonl.gz"}), ("prediction_dates", [])])
def test_manifest_semantics_and_container_are_strict(tiny, field, value):
    tiny.add(action(), "a")
    result = tiny.run()
    manifest = read_manifest(result)
    manifest[field] = value
    (result.artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PropensityDataError):
        list(iter_propensity_rows(result.artifact_dir, "train", purpose="training"))


def test_reader_checks_expected_source_and_generation(tiny):
    tiny.add(action(), "a")
    result = tiny.run()
    for kwargs in ({"source_version": "wrong"}, {"generation": "b" * 32}):
        with pytest.raises(PropensityDataError):
            list(iter_propensity_rows(result.artifact_dir, "train", purpose="training", **kwargs))


def test_probe_is_bounded_real_algorithm_and_never_changes_current(tiny):
    for customer in ("a", "b", "c"):
        tiny.add(action(owner=customer), "cohort-" + customer)
    tiny.add(order(day=1, owner="c"), "future-positive")
    full = tiny.run()
    pointer = (tiny.builder.output_dir / "CURRENT.json").read_bytes()
    old_files = {p.name: p.read_bytes() for p in full.artifact_dir.iterdir()}
    probe = tiny.run(probe=probe_config(max_customers=2))
    manifest = read_manifest(probe)
    assert probe.status == "RESEARCH_PROBE_ONLY" and probe.counts["customers_processed"] == 2
    assert probe.counts["rows"] == 4 and probe.counts["positive"] == 0
    assert manifest["prediction_dates"] == [d.isoformat().replace("+00:00", "Z") for d in probe_config().dates]
    assert probe.artifact_dir.parent == tiny.builder.output_dir / "probes"
    assert not list((tiny.builder.output_dir / "probes").rglob("CURRENT*.json"))
    assert (tiny.builder.output_dir / "CURRENT.json").read_bytes() == pointer
    assert {p.name: p.read_bytes() for p in full.artifact_dir.iterdir()} == old_files
    for name in build.FILES:
        with pytest.raises(PropensityDataError):
            list(iter_propensity_rows(probe.artifact_dir, name, purpose="training"))
        values = list(iter_propensity_rows(probe.artifact_dir, name, purpose="training", allow_probe=True))
        full_values = load_rows(full, name)
        assert all(v in full_values for v in values)
    assert set(manifest["metrics"]) >= {"read_seconds", "feature_seconds", "json_seconds", "write_compress_seconds", "validation_seconds"}


@pytest.mark.parametrize("kwargs", [{"max_customers": 0}, {"max_customers": 10001}, {"max_events": 0},
    {"max_output_bytes": 2 * 1024**3 + 1}, {"max_output_bytes": 0}, {"max_seconds": 0},
    {"max_seconds": 1801}, {"max_seconds": float("nan")}, {"max_seconds": float("inf")}])
def test_probe_limits_must_be_finite_and_bounded(kwargs):
    with pytest.raises(ValueError):
        probe_config(**kwargs)


@pytest.mark.parametrize("dates", [(), PLAN.dates, (PLAN.dates[0], PLAN.dates[0]), tuple(reversed(PLAN.dates[:2]))])
def test_probe_date_count_and_order(dates):
    with pytest.raises(ValueError):
        build.PropensityProbeConfig(dates)


@pytest.mark.parametrize("mode", ["events", "bytes", "time", "source", "generation", "disk"])
def test_probe_limits_and_failures_cleanup_without_pointer(tiny, monkeypatch, mode):
    tiny.add(action(), "a")
    tiny.add(order(day=1), "o")
    settings = {"max_events": 1} if mode == "events" else {"max_output_bytes": 1024} if mode == "bytes" else {"max_seconds": 1e-9} if mode == "time" else {}
    def progress(value):
        if value.phase == "validate_output" and mode == "source":
            tiny.sources.version = "changed"
        if value.phase == "validate_output" and mode == "generation":
            tiny.db.execute("UPDATE metadata SET value=? WHERE key='generation'", ("b" * 32,))
            tiny.db.commit()
    if mode == "disk":
        monkeypatch.setattr(build.shutil, "disk_usage", lambda _: SimpleNamespace(free=19 * 1024**3))
    with pytest.raises(PropensityDataError):
        tiny.run(probe=probe_config(**settings), progress=progress)
    assert not list(tiny.builder.output_dir.rglob("CURRENT*.json"))
    assert not list(tiny.builder.output_dir.rglob(".building-*"))
    assert not list(tiny.builder.output_dir.rglob("manifest.json"))


def test_probe_rejects_date_outside_grid_and_inference(tiny):
    with pytest.raises(ValueError):
        tiny.run(probe=build.PropensityProbeConfig((T + timedelta(days=1),)))
    with pytest.raises(ValueError):
        tiny.builder.build(PropensityDatasetConfig(purpose="inference"), pseudonym_key=SECRET,
                           allow_retrospective=True, probe=probe_config())


def test_cancel_during_gzip_write_keeps_previous_artifact(tiny, monkeypatch):
    tiny.add(action(), "a")
    previous = tiny.run()
    pointer = (tiny.builder.output_dir / "CURRENT.json").read_bytes()
    original = build.open_writer
    stopped = [False]
    @contextmanager
    def writer(*args, **kwargs):
        with original(*args, **kwargs) as stream:
            class CancellingWriter:
                def write(self, value):
                    stream.write(value)
                    stopped[0] = True
            yield CancellingWriter()
    monkeypatch.setattr(build, "open_writer", writer)
    with pytest.raises(PropensityCancelled):
        tiny.run(cancelled=lambda: stopped[0])
    assert previous.artifact_dir.exists() and (tiny.builder.output_dir / "CURRENT.json").read_bytes() == pointer
    assert not list(tiny.builder.output_dir.glob(".building-*"))


def test_periodic_disk_guard_retains_fast_cancellation(tiny, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(build, "perf_counter", lambda: clock[0])
    check = Mock()
    monkeypatch.setattr(tiny.builder, "_guard", check)
    stop = [False]
    guard = build._RunGuard(tiny.builder, lambda: stop[0], resources.MIN_FREE_BYTES, 0, None)
    guard.temp_locator = lambda: tiny.builder.output_dir
    for _ in range(100):
        guard.poll()
    assert check.call_count == 1
    clock[0] = 1.1
    guard.poll()
    assert check.call_count == 2
    guard.written += 8 * 1024**2
    guard.poll()
    assert check.call_count == 3
    stop[0] = True
    with pytest.raises(PropensityCancelled):
        guard.poll()


@pytest.mark.parametrize("arguments", [["--probe-max-customers", "1"], ["--ephemeral-probe-key"],
    ["--probe", "--build"], ["--probe"], ["--probe", "--purpose", "inference"],
    ["--build", "--allow-retrospective", "--ephemeral-probe-key"]])
def test_cli_refuses_unsafe_probe_combinations(arguments):
    with pytest.raises(SystemExit):
        cli.main(arguments)


def test_cli_probe_ephemeral_key_and_readonly_scope(tiny, monkeypatch, capsys):
    tiny.add(action(), "a")
    monkeypatch.setattr(cli, "PropensityDatasetBuilder", lambda **unused: tiny.builder)
    assert cli.main(["--probe", "--allow-retrospective", "--ephemeral-probe-key", "--probe-date", "2024-01-01T00:00:00Z",
                     "--start", "2024-01-01T00:00:00Z", "--end", "2024-03-01T00:00:00Z", "--probe-max-customers", "1"]) == 0
    assert "RESEARCH_PROBE_ONLY" in capsys.readouterr().out
    assert not list(tiny.builder.output_dir.rglob("CURRENT*.json"))


def test_scoped_facts_keep_all_conflicting_variants(tiny):
    tiny.add(order(variant(-1, owner="a"), variant(1, key="v2", owner="b")), "conflict")
    with index._read_index(tiny.path) as con:
        con.row_factory = sqlite3.Row
        values = list(iter_facts(con, lambda: False, customer_ids=("b",)))
    assert len(values) == 1 and values[0][0] == "b"
    assert len(values[0][1].variants) == 2 and not values[0][1].owned_by("b")

@pytest.mark.parametrize("cap", [1, 64, 512])
def test_probe_physical_byte_cap_includes_headers_and_manifest(tiny, cap):
    tiny.add(action(), "a")
    with pytest.raises(PropensityDataError, match="PROBE_OUTPUT_LIMIT"):
        tiny.run(probe=probe_config(max_output_bytes=cap))
    assert not list(tiny.builder.output_dir.rglob("manifest.json"))
    assert not list(tiny.builder.output_dir.rglob(".building-*"))


def test_probe_snapshot_scope_uses_selected_owner_memberships(tiny):
    tiny.add(action(owner="a"), "a")
    tiny.add(order(variant(-1, owner="0-outside"), variant(-1, key="v2", owner="a")), "ambiguous")
    # Pick a scope containing only the secondary variant owner through a direct reader check.
    with index._read_index(tiny.path) as con:
        con.row_factory = sqlite3.Row
        values = list(iter_facts(con, lambda: False, customer_ids=("a",)))
    assert len(values) == 2 and any(not event.owned_by("a") for _, event in values)
    result = tiny.run(probe=probe_config(max_customers=2))
    assert result.status == "RESEARCH_PROBE_ONLY"
    assert result.counts["event_memberships"] == 3


def test_sql_progress_guard_aborts_probe_deadline_before_first_fact(tiny, monkeypatch):
    for n in range(100):
        tiny.add(action(), f"a-{n}", commit=False)
    tiny.db.commit()
    clock = [0.0]
    monkeypatch.setattr(build, "perf_counter", lambda: clock[0])
    real = build.iter_facts
    def expensive_sql(connection, cancelled, **kwargs):
        clock[0] = 2
        connection.execute("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000) SELECT sum(x) FROM n").fetchone()
        yield from real(connection, cancelled, **kwargs)
    monkeypatch.setattr(build, "iter_facts", expensive_sql)
    with pytest.raises(PropensityDataError, match="PROBE_TIME_LIMIT"):
        tiny.run(probe=probe_config(max_seconds=1))
    assert not list(tiny.builder.output_dir.rglob(".building-*"))
    assert not list(tiny.builder.output_dir.rglob("manifest.json"))

@pytest.mark.parametrize("fmt", ["jsonl", "jsonl.gz"])
def test_valid_container_with_missing_rows_reaches_final_count_check(tiny, fmt):
    tiny.add(action(), "a")
    result = tiny.run(storage=PropensityStorage(fmt))
    reseal_file(result, "train", gzip.compress(b"") if fmt == "jsonl.gz" else b"")
    with pytest.raises(PropensityDataError, match="OUTPUT_COUNT_MISMATCH"):
        load_rows(result, "train")

# SAFETY-01: mocked disks/native APIs, no actual disk-filling or large queries.
@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("reserve", [0, 1, 1024**3, resources.MIN_FREE_BYTES - 1, True])
def test_safe_floor_rejects_unsafe_reserve_before_any_read(tiny, monkeypatch, probe, reserve):
    read = Mock(side_effect=AssertionError("No source read"))
    monkeypatch.setattr(index, "_capture_sources", read)
    with pytest.raises(ValueError, match="20 GiB"):
        tiny.builder.build(CONFIG, pseudonym_key=SECRET, allow_retrospective=True, min_free_bytes=reserve,
                           probe=probe_config() if probe else None)
    assert not read.called and not tiny.builder.output_dir.exists()


@pytest.mark.parametrize("probe", [False, True])
def test_safe_floor_default_and_larger_reserve(tiny, monkeypatch, probe):
    tiny.add(action(), "a")
    original = tiny.builder._guard
    seen = []
    def guard(cancelled, reserve, directory=None):
        seen.append(reserve)
        return original(cancelled, reserve, directory)
    monkeypatch.setattr(tiny.builder, "_guard", guard)
    kwargs = {"probe": probe_config()} if probe else {}
    tiny.builder.build(CONFIG, pseudonym_key=SECRET, allow_retrospective=True, **kwargs)
    assert seen and set(seen) == {resources.MIN_FREE_BYTES}
    seen.clear()
    tiny.builder.build(CONFIG, pseudonym_key=SECRET, allow_retrospective=True, min_free_bytes=25 * 1024**3, **kwargs)
    assert seen and set(seen) == {25 * 1024**3}


@pytest.mark.parametrize("same_volume", [False, True])
def test_disk_monitor_uses_volume_identity_and_deduplicates(tiny, monkeypatch, same_volume):
    output, temp = tiny.builder.output_dir, tiny.builder.output_dir.parent
    calls = []
    monkeypatch.setattr(resources, "volume_identity", lambda p: "volume-a" if p == output or same_volume else "volume-b")
    def usage(directory):
        calls.append(directory)
        return SimpleNamespace(free=resources.MIN_FREE_BYTES)
    monkeypatch.setattr(resources.shutil, "disk_usage", usage)
    resources.check_disk_space(output, temp, resources.MIN_FREE_BYTES)
    assert calls == ([output] if same_volume else [output, temp])


@pytest.mark.parametrize("which", ["output", "temp"])
def test_either_volume_can_stop_the_build(tiny, monkeypatch, which):
    output, temp = tiny.builder.output_dir, tiny.builder.output_dir.parent
    monkeypatch.setattr(resources, "volume_identity", lambda p: "volume-a" if p == output else "volume-b")
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda p: SimpleNamespace(
        free=resources.MIN_FREE_BYTES - 1 if p == (output if which == "output" else temp) else 64 * 1024**3))
    with pytest.raises(PropensityDataError, match="INSUFFICIENT_FREE_SPACE"):
        resources.check_disk_space(output, temp, resources.MIN_FREE_BYTES)


@pytest.mark.parametrize("part", ["identity", "usage"])
def test_disk_information_errors_are_safe_categories(tiny, monkeypatch, part):
    if part == "identity":
        monkeypatch.setattr(resources, "volume_identity", Mock(side_effect=OSError("private location")))
    else:
        monkeypatch.setattr(resources, "volume_identity", lambda _: "volume")
        monkeypatch.setattr(resources.shutil, "disk_usage", Mock(side_effect=OSError("private location")))
    with pytest.raises(PropensityDataError, match="^DISK_INFO_UNAVAILABLE$"):
        resources.check_disk_space(tiny.builder.output_dir, tiny.builder.output_dir.parent, resources.MIN_FREE_BYTES)


def test_unknown_temp_blocks_before_heavy_sql(tiny, monkeypatch):
    tiny.add(action(), "a")
    monkeypatch.setattr(build, "make_sqlite_temp_locator", Mock(side_effect=PropensityDataError("SQLITE_TEMP_UNVERIFIED")))
    facts = Mock(side_effect=AssertionError("No heavy SELECT"))
    monkeypatch.setattr(build, "iter_facts", facts)
    with pytest.raises(PropensityDataError, match="SQLITE_TEMP_UNVERIFIED"):
        tiny.run()
    assert not facts.called and not list(tiny.builder.output_dir.rglob(".building-*"))
    assert not list(tiny.builder.output_dir.rglob("CURRENT*.json"))


def test_changed_temp_location_stops_instead_of_losing_old_volume(tiny, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(build, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(tiny.builder, "_guard", Mock())
    locations = iter((tiny.path.parent, tiny.path.parent.parent))
    guard = build._RunGuard(tiny.builder, lambda: False, resources.MIN_FREE_BYTES, 0, None)
    guard.temp_locator = lambda: next(locations)
    guard.poll()
    clock[0] = 2
    with pytest.raises(PropensityDataError, match="SQLITE_TEMP_CHANGED"):
        guard.poll()


@pytest.mark.parametrize("purpose", ["training", "inference"])
@pytest.mark.parametrize("phase", ["sql", "gzip", "sha", "pointer"])
def test_disk_drop_at_each_phase_preserves_previous_current_and_unlocks(tiny, monkeypatch, purpose, phase):
    from Application.customer_profile_service import _reader_lock as real_reader_lock
    canonical = tiny.builder.raw_root / "canonical"
    canonical.mkdir(parents=True)
    (canonical / ".writer.lock").write_bytes(b"0")
    monkeypatch.setattr(index, "_reader_lock", real_reader_lock)
    tiny.add(action(), "a")
    config = CONFIG if purpose == "training" else PropensityDatasetConfig(purpose="inference")
    # An event immediately before inference boundary supplies its cohort.
    if purpose == "inference":
        tiny.add(action(day=499), "inference-cohort")
    previous = tiny.builder.build(config, pseudonym_key=SECRET, allow_retrospective=True)
    pointer_name = "CURRENT.json" if purpose == "training" else "CURRENT_INFERENCE.json"
    pointer = (tiny.builder.output_dir / pointer_name).read_bytes()
    old_files = {p.name: p.read_bytes() for p in previous.artifact_dir.iterdir()}
    tiny.add(action(day=498 if purpose == "inference" else -2), "new-event")
    clock, free = [0.0], [64 * 1024**3]
    monkeypatch.setattr(build, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=free[0]))
    def drop():
        free[0] = 19 * 1024**3
        clock[0] = 2
    if phase == "sql":
        actual = build.iter_facts
        def sql(connection, cancelled, **kwargs):
            drop()
            connection.execute("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000) SELECT sum(x) FROM n").fetchone()
            yield from actual(connection, cancelled, **kwargs)
        monkeypatch.setattr(build, "iter_facts", sql)
    elif phase == "gzip":
        actual = build.open_writer
        @contextmanager
        def writer(*args, **kwargs):
            with actual(*args, **kwargs) as stream:
                class DroppingWriter:
                    def write(self, value):
                        stream.write(value)
                        stream.flush()  # Physical gzip data exists before the drop.
                        drop()
                yield DroppingWriter()
        monkeypatch.setattr(build, "open_writer", writer)
    elif phase == "sha":
        actual = build._digest
        def digest(path, cancelled=lambda: False):
            if path.name.endswith(".jsonl.gz"):
                drop()
            return actual(path, cancelled)
        monkeypatch.setattr(build, "_digest", digest)
    else:
        actual = tiny.builder._fresh
        n = [0]
        def fresh(*args):
            n[0] += 1
            actual(*args)
            if n[0] == 2:
                drop()
        monkeypatch.setattr(tiny.builder, "_fresh", fresh)
    with pytest.raises(PropensityDataError, match="INSUFFICIENT_FREE_SPACE"):
        tiny.builder.build(config, pseudonym_key=SECRET, allow_retrospective=True)
    assert (tiny.builder.output_dir / pointer_name).read_bytes() == pointer
    assert {p.name: p.read_bytes() for p in previous.artifact_dir.iterdir()} == old_files
    assert not list(tiny.builder.output_dir.rglob(".building-*"))
    assert not list(tiny.builder.output_dir.glob(".current-*"))
    # Both OS-level coordination locks can be acquired again; gzip handles are closed.
    with real_reader_lock(tiny.builder.raw_root), build._writer_lock(tiny.builder.output_dir):
        pass
    for directory in tiny.builder.output_dir.iterdir():
        if directory.is_dir() and (directory / "manifest.json").exists():
            manifest = json.loads((directory / "manifest.json").read_text())
            for name in build.FILES:
                list(iter_propensity_rows(directory, name, purpose=manifest["purpose"]))


@pytest.mark.parametrize("extra", [["--build"], ["--probe"], []])
@pytest.mark.parametrize("reserve", ["0", "1", "19"])
def test_cli_cannot_select_unsafe_reserve(tiny, monkeypatch, capsys, extra, reserve):
    builder = Mock(side_effect=AssertionError("Must reject before builder"))
    monkeypatch.setattr(cli, "PropensityDatasetBuilder", builder)
    with pytest.raises(SystemExit):
        valid = (["--allow-retrospective", "--key-file", "not-read.bin"] if "--build" in extra else
                 ["--allow-retrospective", "--ephemeral-probe-key", "--probe-date", "2024-01-01T00:00:00Z"]
                 if "--probe" in extra else [])
        cli.main(extra + valid + ["--min-free-gib", reserve])
    assert "Disk reserve must be at least 20 GiB" in capsys.readouterr().err
    assert not builder.called


@pytest.mark.parametrize("mode", ["ok", "open", "vfs_rc", "temp_rc", "vfs_name", "relative", "missing", "close"])
def test_native_temp_discovery_frees_allocations_and_closes_handles(tmp_path, monkeypatch, mode):
    import ctypes
    allocated, freed, closed, opened = [], [], [], []
    directory = tmp_path / "native-temp"
    directory.mkdir()
    class Native:
        def sqlite3_open_v2(self, path, out, flags, vfs):
            opened.append(flags)
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p)).contents.value = 1
            return 1 if mode == "open" else 0
        def sqlite3_file_control(self, db, name, opcode, out):
            if opcode == 12:
                value = b"custom-vfs" if mode == "vfs_name" else b"win32"
            else:
                value = ("relative/tempfile" if mode == "relative" else str(tmp_path / "missing" / "tempfile")
                         if mode == "missing" else str(directory / "tempfile")).encode("utf8")
            buffer = ctypes.create_string_buffer(value)
            allocated.append(buffer)
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p)).contents.value = ctypes.addressof(buffer)
            return int((opcode == 12 and mode == "vfs_rc") or (opcode == 16 and mode == "temp_rc"))
        def sqlite3_free(self, pointer):
            freed.append(pointer.value)
        def sqlite3_close(self, db):
            closed.append(db.value)
            return int(mode == "close")
    native = Native()
    monkeypatch.setattr(resources, "_bind_sqlite", lambda: native)
    locator = resources.make_sqlite_temp_locator(tmp_path / "database.sqlite")
    if mode == "ok":
        assert locator() == directory.resolve()
    else:
        with pytest.raises(PropensityDataError, match="SQLITE_TEMP_UNVERIFIED"):
            locator()
    assert freed == [ctypes.addressof(x) for x in allocated] and closed == [1]
    assert opened == [1 | 0x10000]  # Strict READONLY, never CREATE/READWRITE.
    assert not list(directory.iterdir())


def test_native_runtime_mismatch_and_missing_api_fail_closed(monkeypatch):
    native = Mock()
    native.sqlite3_sourceid.return_value = b"different library"
    native.sqlite3_libversion.return_value = sqlite3.sqlite_version.encode()
    monkeypatch.setattr(resources, "_loaded_sqlite_dll", lambda: native)
    with pytest.raises(PropensityDataError, match="SQLITE_TEMP_UNVERIFIED"):
        resources.make_sqlite_temp_locator(Path("unused"))
    monkeypatch.setattr(resources, "_loaded_sqlite_dll", Mock(side_effect=OSError("private DLL path")))
    with pytest.raises(PropensityDataError, match="^SQLITE_TEMP_UNVERIFIED$"):
        resources.make_sqlite_temp_locator(Path("unused"))


def test_nonwindows_temp_runtime_is_rejected_before_native_access(tmp_path, monkeypatch):
    # Patch only this module's OS reference, not Python's global os.name/Path selection.
    monkeypatch.setattr(resources, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(resources, "_extension_imports_sqlite_dll", lambda: pytest.fail("No PE/native access on non-Windows"))
    with pytest.raises(PropensityDataError, match="^SQLITE_TEMP_UNVERIFIED$"):
        resources.make_sqlite_temp_locator(tmp_path / "unused.sqlite")
    assert not list(tmp_path.iterdir())

@pytest.mark.parametrize("magic", [0x10B, 0x20B])
@pytest.mark.parametrize("damage", ["none", "static", "truncated", "magic", "rva", "no_import", "short_optional", "sections"])
def test_runtime_extension_import_check_is_bounded(tmp_path, monkeypatch, magic, damage):
    import struct
    data = bytearray(0x800)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<H", data, 0x86, 1)
    size = 0xE0 if magic == 0x10B else 0xF0
    struct.pack_into("<H", data, 0x94, size)
    optional = 0x98
    struct.pack_into("<H", data, optional, magic)
    directory = optional + (96 if magic == 0x10B else 112) + 8
    struct.pack_into("<II", data, directory, 0x1000, 40)
    struct.pack_into("<IIII", data, optional + size + 8, 0x200, 0x1000, 0x200, 0x400)
    struct.pack_into("<IIIII", data, 0x400, 0, 0, 0, 0x1040, 0)
    name = b"python312.dll\0" if damage == "static" else b"sqlite3.dll\0"
    data[0x440:0x440 + len(name)] = name
    if damage == "truncated":
        data = data[:64]
    elif damage == "magic":
        struct.pack_into("<H", data, optional, 0)
    elif damage == "rva":
        struct.pack_into("<I", data, 0x40C, 0xFFFFFFFF)
    elif damage == "no_import":
        struct.pack_into("<II", data, directory, 0, 0)
    elif damage == "short_optional":
        struct.pack_into("<H", data, 0x94, 16)
    elif damage == "sections":
        struct.pack_into("<H", data, 0x86, 97)
    extension = tmp_path / "fake.pyd"
    extension.write_bytes(data)
    monkeypatch.setattr(resources._sqlite3, "__file__", str(extension))
    assert resources._extension_imports_sqlite_dll() is (damage == "none")


def test_unknown_extension_linkage_is_not_guessed_from_version(monkeypatch):
    monkeypatch.setattr(resources, "_extension_imports_sqlite_dll", lambda: False)
    with pytest.raises(PropensityDataError, match="SQLITE_TEMP_UNVERIFIED"):
        resources.make_sqlite_temp_locator(Path("unused"))


def test_sql_temp_disk_drop_interrupts_while_output_disk_is_healthy(tiny, monkeypatch):
    tiny.add(action(), "a")
    clock, free_temp = [0.0], [64 * 1024**3]
    monkeypatch.setattr(build, "perf_counter", lambda: clock[0])
    output = tiny.builder.output_dir
    monkeypatch.setattr(resources, "volume_identity", lambda p: "output-volume" if p == output else "temp-volume")
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda p: SimpleNamespace(free=64 * 1024**3 if p == output else free_temp[0]))
    original = build.iter_facts
    def sql(connection, cancelled, **kwargs):
        clock[0], free_temp[0] = 2, 19 * 1024**3
        connection.execute("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000) SELECT sum(x) FROM n").fetchone()
        yield from original(connection, cancelled, **kwargs)
    monkeypatch.setattr(build, "iter_facts", sql)
    with pytest.raises(PropensityDataError, match="INSUFFICIENT_FREE_SPACE"):
        tiny.run()
    assert not list(tiny.builder.output_dir.rglob(".building-*"))
    assert not list(tiny.builder.output_dir.rglob("CURRENT*.json"))

def test_initial_low_space_does_not_start_history_queries(tiny, monkeypatch):
    tiny.add(action(), "a")
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=resources.MIN_FREE_BYTES - 1))
    facts = Mock(side_effect=AssertionError("Must not start history queries"))
    monkeypatch.setattr(build, "iter_facts", facts)
    with pytest.raises(PropensityDataError, match="INSUFFICIENT_FREE_SPACE"):
        tiny.run()
    assert not facts.called and not list(tiny.builder.output_dir.rglob(".building-*"))


def test_unverified_temp_never_falls_back_to_output_directory(tiny, monkeypatch):
    usage = Mock(side_effect=AssertionError("No disk guess"))
    monkeypatch.setattr(resources.shutil, "disk_usage", usage)
    with pytest.raises(PropensityDataError, match="SQLITE_TEMP_UNVERIFIED"):
        resources.check_disk_space(tiny.builder.output_dir, None, resources.MIN_FREE_BYTES)
    assert not usage.called


@pytest.mark.skipif(resources.os.name != "nt", reason="Native Windows VFS integration")
def test_native_temp_locator_is_safe_inside_sql_progress_handler(tiny):
    tiny.builder.output_dir.mkdir()
    before = index._database_version(tiny.path)
    locator = resources.make_sqlite_temp_locator(tiny.path)
    guard = build._RunGuard(tiny.builder, lambda: False, resources.MIN_FREE_BYTES, build.perf_counter(), None)
    guard.temp_locator = locator
    calls = []
    with index._read_index(tiny.path) as con:
        def progress():
            calls.append(1)
            guard.poll(force=True)
            return 0
        con.set_progress_handler(progress, 1000)
        assert con.execute("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<3000) SELECT sum(x) FROM n").fetchone()[0] == 4501500
        con.set_progress_handler(None, 0)
    assert calls and index._database_version(tiny.path) == before
