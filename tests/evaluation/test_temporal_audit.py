from dataclasses import replace
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from types import SimpleNamespace
import uuid

import pytest

from Application.evaluation import audit
from Application.mindbox.canonical_storage import atomic_json
from Application.mindbox.raw_reader import EXPORT_ROOTS
from Application.mindbox.selection import DEFAULT_SELECTION
from scripts import audit_temporal_evaluation as cli


def source(start="2026-01-01T00:00:00+00:00", end="2026-05-01T00:00:00+00:00"):
    return {"declared_start": start, "declared_end": end}


def test_snapshot_denominators_distributions_and_history_popularity(protocol):
    result = audit.summarize_snapshot(protocol.validation)
    assert result["rates"]["final_cases_per_future_user"] == {"numerator": 1, "denominator": 1, "percent": 100}
    assert result["target_type_rates"]["VIEW"] == {"numerator": 1, "denominator": 1, "percent": 100}
    assert result["candidate_count"]["median"] == 3
    assert result["benchmark_history_events"]["median"] == 2
    assert result["target_delay_days"]["median"] == pytest.approx(2 / 24)
    # C appears twice in the complete stream, but only once in validation history.
    assert result["target_training_popularity_events"]["median"] == 1
    assert not any(result["invariants"].values())


def test_rates_empty_populations_and_quantile_method():
    assert audit.rate(1, 4) == {"numerator": 1, "denominator": 4, "percent": 25}
    assert audit.rate(0, 0)["percent"] is None
    assert audit.distribution([])["median"] is None
    assert audit.distribution([1, 2, 3, 4])["p25"] == 1.75


def test_invariant_violation_blocks_audit(protocol):
    snap = protocol.validation
    invalid_case = replace(snap.cases[0], target_item="A")
    with pytest.raises(audit.TemporalAuditError, match="blocked"):
        audit.summarize_snapshot(replace(snap, cases=(invalid_case,)))


def dense_events(event):
    records = [event("private_witness", f"I{day}", 0) for day in range(120)]
    records += [event("private_customer", f"I{day}", 24 * day + 1) for day in range(120)]
    return records


def test_scenarios_tail_cutoffs_deterministic_and_no_identifiers(event):
    events = dense_events(event)
    report = audit.audit_scenarios(events, source())
    assert [(s["horizon_days"], s["min_history_events"]) for s in report["scenarios"]] == [(14, 10), (30, 10), (30, 5), (30, 20)]
    assert report["unavailable"][0]["horizon_days"] == 60
    assert report["test_end"] == "2026-04-30T00:00:00+00:00"
    assert report == audit.audit_scenarios(list(reversed(events)), source())
    text = json.dumps(report) + audit.markdown_summary(report)
    assert "private_customer" not in text
    assert "private_witness" not in text
    assert "Cold-item rate uses history-eligible first-novel targets" in text
    assert report["coverage"]["resolved_events"] == len(events)
    assert sum(r["events"] for r in report["coverage"]["weeks"]) == len(events)


def test_insufficient_coverage_and_empty_dataset(event):
    report = audit.audit_scenarios([event("u", "A", 1), event("u", "B", 25)], source())
    assert report["scenarios"] == []
    assert len(report["unavailable"]) == 5
    empty = audit.audit_scenarios([], source())
    assert empty["coverage"]["earliest"] is None
    assert empty["scenarios"] == []
    assert "empty" in empty["unavailable"][0]["reason"]


def test_requested_end_cannot_extend_observed_coverage(event):
    with pytest.raises(audit.TemporalAuditError, match="coverage"):
        audit.audit_scenarios(dense_events(event), source(), test_end=datetime(2026, 5, 1, tzinfo=timezone.utc))


def test_cli_config_and_protected_output(tmp_path):
    args = cli.parse_args(["--horizons", "14", "30", "--thresholds", "5", "20", "--sensitivity-horizon", "14",
                           "--test-end", "2025-12-31T00:00:00Z", "--output-dir", str(tmp_path / "report")])
    assert args.horizons == [14, 30]
    assert args.thresholds == [5, 20]
    assert args.test_end.tzinfo is not None
    for options in (["--horizons", "0"], ["--test-end", "2025-12-31"],
                    ["--output-dir", str(cli.PROJECT_ROOT / "model/audit")]):
        with pytest.raises(SystemExit):
            cli.parse_args(options)


@pytest.fixture
def canonical(tmp_path):
    root = tmp_path / "raw"
    start, end = "2026-01-01T00:00:00+00:00", "2026-05-01T00:00:00+00:00"
    entries, directories = {}, {}
    records = {
        "customer_merges": [{"id": 1, "dateTimeUtc": start, "resultingCustomer": {"ids": {"mindboxId": 300}},
                             "mergedCustomers": [{"ids": {"mindboxId": 100}}]}],
        "actions": [], "orders": [],
    }
    def action(user, item, stamp, system="ProsmotrProdukta"):
        return {"ids": {"mindboxId": 1}, "customer": {"ids": {"mindboxId": user}},
                "dateTimeUtc": stamp, "creationDateTimeUtc": stamp, "actionTemplate": {"ids": {"systemName": system}},
                "products": [{"ids": {"offline1C": item}}]}
    records["actions"] = [action(100, "000001size", start), action(300, "000002size", "2026-04-30T12:00:00Z"),
                          {"actionTemplate": {"ids": {"systemName": "irrelevant"}}},
                          {**action(300, "000002size", start), "products": []}]
    order = {"ids": {"mindboxId": 42}, "firstAction": {"dateTimeUtc": start,
             "channel": {"ids": {"externalId": "test"}, "name": "test"}}, "customer": {"ids": {"mindboxId": 400}},
             "lines": [{"id": "line", "number": 1, "quantity": 1, "basePricePerItem": 10, "priceOfLine": 10,
                        "product": {"ids": {"offline1C": "000002size"}}, "status": {"ids": {"externalId": "CP"}}}]}
    records["orders"] = [order, order]
    manual_object = uuid.uuid4().hex
    for name in records:
        object_id = uuid.uuid4().hex if name == "customer_merges" else manual_object
        directory = root / "canonical/objects" / object_id / name
        directory.mkdir(parents=True)
        atomic_json(directory / f"{name}_part_001.json", {EXPORT_ROOTS[name]: records[name]})
        directories[name] = directory
        entries[name] = {"since": start, "until": end, "updated": end, "directory": directory.relative_to(root).as_posix(),
                         "parts": 1, "source_kind": "API" if name == "customer_merges" else "MANUAL",
                         "export_id": "saved" if name == "customer_merges" else None,
                         "operation": "saved" if name == "customer_merges" else "MANUAL"}
    from dataclasses import asdict
    atomic_json(root / "canonical/catalog.json", {"schema_version": 2, "revision": uuid.uuid4().hex,
                "actions": {}, "orders": {}, "customer_merges": entries["customer_merges"],
                "manual_interactions": {"since": start, "until": end, "updated": end,
                                        "actions": entries["actions"], "orders": entries["orders"]},
                "selection": asdict(DEFAULT_SELECTION)})
    catalog = tmp_path / "catalog/nomenclature.csv"
    catalog.parent.mkdir()
    catalog.write_text("КодНоменклатуры\n000001\n000002\n", encoding="utf-8")
    return root, catalog, directories


def test_loader_reuses_production_semantics_and_keeps_data_readonly(canonical):
    root, catalog, directories = canonical
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.json")}
    events, metadata = audit.load_canonical_events(root, catalog)
    assert len(events) == 3
    assert events[0].interaction.customer_id == "300"
    assert events[0].item_id == "000001"
    assert metadata["ingestion"]["orders_duplicate_identical"] == 1
    assert metadata["ingestion"]["malformed_mapped_actions"] == 1
    assert metadata["ingestion"]["unmapped_actions"] == 1
    # Characterize the same canonical preparation ingestion without training a model.
    from Application.model.mindbox_training_preparation import prepare_training_data_from_mindbox
    prepared = prepare_training_data_from_mindbox(actions_export_dir=directories["actions"],
        orders_export_dir=directories["orders"], customer_merges_export_dir=directories["customer_merges"],
        catalog_path=catalog, train_config=SimpleNamespace(w_view_item=0.1, w_favorite=2, w_purchase=10,
        min_user_interactions_for_eval=10), diagnose=True)
    assert prepared.diagnostics.bpr.events_total == len(events)
    assert prepared.diagnostics.resolution.total.resolved == len(events)
    assert prepared.diagnostics.orders_duplicate_identical == metadata["ingestion"]["orders_duplicate_identical"]
    assert prepared.diagnostics.malformed_actions == metadata["ingestion"]["malformed_mapped_actions"]
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())


def test_cli_main_no_training_publication_or_raw_ids(canonical, tmp_path, monkeypatch):
    from Application.model import BPRMF as core, mindbox_production_training as production
    root, catalog, _ = canonical
    def forbidden(*args, **kwargs):
        pytest.fail("Audit cannot train or publish")
    monkeypatch.setattr(core, "train_prepared_data", forbidden)
    monkeypatch.setattr(core, "_save_artifacts", forbidden)
    monkeypatch.setattr(production, "train_and_publish_production_model", forbidden)
    output = tmp_path / "reports"
    assert cli.main(["--raw-root", str(root), "--catalog", str(catalog), "--output-dir", str(output)]) == 0
    first = (output / "report.json").read_bytes()
    assert cli.main(["--raw-root", str(root), "--catalog", str(catalog), "--output-dir", str(output)]) == 0
    assert first == (output / "report.json").read_bytes()
    document = json.loads(first)
    assert document["method"]["model_training"] is False
    assert "customer_id" not in first.decode()
    assert "source_event_id" not in first.decode()


def test_cli_errors_do_not_expose_raw_exception_values(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise ValueError("PRIVATE_CUSTOMER")
    monkeypatch.setattr(cli, "load_canonical_events", fail)
    assert cli.main(["--output-dir", str(tmp_path / "report")]) == 1
    assert "PRIVATE_CUSTOMER" not in capsys.readouterr().err


def replace_records(canonical, name, records):
    directory = canonical[2][name]
    atomic_json(directory / f"{name}_part_001.json", {EXPORT_ROOTS[name]: records})


def first_order(canonical):
    path = canonical[2]["orders"] / "orders_part_001.json"
    return deepcopy(json.loads(path.read_text(encoding="utf-8"))["orders"][0])


@pytest.mark.parametrize("sequence, identical, conflicting", [
    ((0,), 0, 0), ((0, 0, 0), 2, 0), ((0, 1), 0, 1),
    ((0, 1, 0), 1, 1), ((0, 1, 2), 0, 2), ((2, 1, 0), 0, 2),
])
def test_research_first_snapshot_no_combining_and_preserved_diagnostics(canonical, sequence, identical, conflicting):
    from Application.interactions import InteractionType
    variants = [first_order(canonical) for _ in range(3)]
    for index, raw in enumerate(variants):
        raw["lines"][0]["quantity"] = index + 1
    replace_records(canonical, "orders", [variants[i] for i in sequence])
    events, source = audit.load_canonical_events(*canonical[:2], read_only=True)
    purchases = [e for e in events if e.interaction.interaction_type is InteractionType.PURCHASE]
    assert len(purchases) == 1
    assert purchases[0].interaction.quantity == sequence[0] + 1
    d = source["ingestion"]
    assert d["orders_selected"] == 1
    assert d["orders_duplicate_identical"] == identical
    assert d["orders_duplicate_conflicting"] == conflicting
    repeated, repeated_source = audit.load_canonical_events(*canonical[:2], read_only=True)
    assert repeated == events
    assert repeated_source == source
    # Statistics and production diagnose share the exact same deduplicator.
    from Application.mindbox.order_dedup import OrderSnapshots
    snapshots = OrderSnapshots()
    assert [i for i in sequence if snapshots.accept(variants[i], diagnose=True)] == [sequence[0]]
    assert (snapshots.identical, snapshots.conflicting) == (identical, conflicting)


def test_first_snapshot_uses_numeric_part_order_not_timestamp(canonical):
    root, catalog, directories = canonical
    early = first_order(canonical)
    later = deepcopy(early)
    early["firstAction"]["dateTimeUtc"] = "2026-02-01T00:00:00Z"
    later["firstAction"]["dateTimeUtc"] = "2026-04-01T00:00:00Z"
    later["lines"][0]["quantity"] = 7
    replace_records(canonical, "orders", [later])
    atomic_json(directories["orders"] / "orders_part_002.json", {"orders": [early]})
    metadata = json.loads((root / "canonical/catalog.json").read_text(encoding="utf-8"))
    metadata["manual_interactions"]["orders"]["parts"] = 2
    atomic_json(root / "canonical/catalog.json", metadata)
    events, source = audit.load_canonical_events(root, catalog, read_only=True)
    purchase = next(e for e in events if e.interaction.quantity is not None)
    assert purchase.interaction.quantity == 7
    assert purchase.interaction.event_datetime_utc.month == 4
    assert source["ingestion"]["orders_duplicate_conflicting"] == 1


@pytest.mark.parametrize("duplicate", [False, True])
def test_malformed_first_or_conflicting_order_still_raises(canonical, duplicate):
    from Application.mindbox.adapters._common import AdapterError
    valid = first_order(canonical)
    malformed = deepcopy(valid)
    del malformed["lines"][0]["quantity"]
    replace_records(canonical, "orders", [valid, malformed] if duplicate else [malformed])
    with pytest.raises(AdapterError):
        audit.load_canonical_events(*canonical[:2], read_only=True)


@pytest.mark.parametrize("stamp, included", [
    ("2021-12-31T23:59:59Z", False), ("2022-01-01T00:00:00Z", True),
    ("2023-06-15T12:00:00Z", True), ("2025-12-31T23:59:59Z", True),
    ("2026-01-01T00:00:00Z", False), ("2022-01-01T03:00:00+03:00", True),
    ("2026-01-01T02:59:59+03:00", True), ("2026-01-01T03:00:00+03:00", False),
])
@pytest.mark.parametrize("name", ["actions", "orders"])
def test_dynamic_dataset_bounds_use_actual_interaction_timestamp(canonical, stamp, included, name):
    root, catalog, directories = canonical
    metadata = json.loads((root / "canonical/catalog.json").read_text(encoding="utf-8"))
    for entry in [metadata["manual_interactions"], metadata["manual_interactions"]["actions"],
                  metadata["manual_interactions"]["orders"], metadata["customer_merges"]]:
        entry.update(since="2022-01-01T00:00:00+00:00", until="2026-01-01T00:00:00+00:00")
    atomic_json(root / "canonical/catalog.json", metadata)
    replace_records(canonical, "actions", [])
    replace_records(canonical, "orders", [])
    if name == "actions":
        raw = {"ids": {"mindboxId": 1}, "customer": {"ids": {"mindboxId": 400}},
               "dateTimeUtc": stamp, "creationDateTimeUtc": "2019-01-01T00:00:00Z",
               "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}},
               "products": [{"ids": {"offline1C": "000002size"}}]}
    else:
        # Use a valid synthetic order; auxiliary snapshot/creation dates must not filter PURCHASE.
        raw = {"ids": {"mindboxId": 42}, "customer": {"ids": {"mindboxId": 400}},
               "firstAction": {"dateTimeUtc": stamp, "channel": {"ids": {"externalId": "test"}, "name": "test"}},
               "creationDateTimeUtc": "2019-01-01T00:00:00Z", "dateTimeUtc": "2027-01-01T00:00:00Z",
               "lines": [{"id": "line", "number": 1, "quantity": 1, "priceOfLine": 10,
                          "product": {"ids": {"offline1C": "000002size"}}, "status": {"ids": {"externalId": "CP"}}}]}
    replace_records(canonical, name, [raw])
    events, source = audit.load_canonical_events(root, catalog, read_only=True)
    assert len(events) == int(included)
    if events:
        assert events[0].interaction.event_datetime_utc == datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    excluded = source["ingestion"]["excluded_outside_period"][name]
    assert sum(excluded.values()) == int(not included)


def test_filter_precedes_dedup_and_handles_one_year_dataset(canonical):
    root, catalog, _ = canonical
    metadata = json.loads((root / "canonical/catalog.json").read_text(encoding="utf-8"))
    for entry in [metadata["manual_interactions"], metadata["manual_interactions"]["actions"],
                  metadata["manual_interactions"]["orders"], metadata["customer_merges"]]:
        entry.update(since="2025-01-01T00:00:00+00:00", until="2026-01-01T00:00:00+00:00")
    atomic_json(root / "canonical/catalog.json", metadata)
    orders = [first_order(canonical) for _ in range(3)]
    for raw, stamp, quantity in zip(orders, ["2024-12-31T23:59:59Z", "2025-07-01T00:00:00Z", "2026-01-01T00:00:00Z"], [9, 2, 8]):
        raw["firstAction"]["dateTimeUtc"] = stamp
        raw["lines"][0]["quantity"] = quantity
    replace_records(canonical, "actions", [])
    replace_records(canonical, "orders", orders)
    events, source = audit.load_canonical_events(root, catalog, read_only=True)
    assert len(events) == 1
    assert events[0].interaction.quantity == 2
    assert source["ingestion"]["orders_raw"] == 3
    assert source["ingestion"]["orders_in_period"] == 1
    assert source["ingestion"]["excluded_outside_period"]["orders"] == {"before_start": 1, "at_or_after_end": 1}


@pytest.mark.parametrize("sequence", [(0, 1, 2), (2, 1, 0)])
def test_conflicts_never_move_future_purchase_into_past(canonical, sequence):
    from Application.evaluation.temporal import TemporalConfig, build_temporal_protocol
    orders = [first_order(canonical) for _ in range(3)]
    for raw, stamp in zip(orders, ["2026-01-15T00:00:00Z", "2026-02-15T00:00:00Z", "2026-03-15T00:00:00Z"]):
        raw["firstAction"]["dateTimeUtc"] = stamp
    replace_records(canonical, "orders", [orders[i] for i in sequence])
    events, source = audit.load_canonical_events(*canonical[:2], read_only=True)
    config = TemporalConfig(datetime(2026, 2, 1, tzinfo=timezone.utc), datetime(2026, 3, 1, tzinfo=timezone.utc),
                            datetime(2026, 4, 1, tzinfo=timezone.utc), min_history_events=1)
    protocol = build_temporal_protocol(events, config)
    selected = next(e for e in events if e.interaction.quantity is not None)
    assert selected.interaction.event_datetime_utc.month == sequence[0] + 1
    for snap in [protocol.validation, protocol.test]:
        assert all(e.interaction.event_datetime_utc < snap.cutoff for e in snap.history)
        assert all(snap.cutoff <= c.target_timestamp < snap.future_end for c in snap.cases)
        assert (selected in snap.history) == (selected.interaction.event_datetime_utc < snap.cutoff)
    assert source["ingestion"]["orders_duplicate_conflicting"] == 2


def test_unlocked_read_only_preflight_never_touches_coordination_file(canonical, monkeypatch):
    root, catalog, _ = canonical
    lock = root / "canonical/.writer.lock"
    lock.write_bytes(b"untouched")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(audit, "storage_lock", lambda *_: pytest.fail("Read-only preflight acquired writer lock"))
    audit.load_canonical_events(root, catalog, read_only=True)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_read_only_input_drift_is_rejected(canonical):
    root, catalog, _ = canonical
    def changed(_):
        catalog.write_text(catalog.read_text(encoding="utf-8") + "000003\n", encoding="utf-8")
    with pytest.raises(audit.TemporalAuditError, match="changed"):
        audit.load_canonical_events(root, catalog, progress=changed, read_only=True)


def test_preflight_counts_exact_existing_bpr_aggregation_without_arrays(event, config):
    from Application.evaluation.bpr import prepare_bpr_snapshot
    from Application.evaluation.temporal import build_temporal_protocol
    from Application.model.bpr_preparation import BprWeightConfig
    from scripts.preflight_research import prepare_report
    events = [event("PRIVATE_USER", "A", 1), event("PRIVATE_USER", "A", 2),
              event("PRIVATE_OTHER", "B", 1), event("PRIVATE_OTHER", "B", 2),
              event("PRIVATE_USER", "B", 12), event("PRIVATE_USER", "B", 22)]
    expected = build_temporal_protocol(events, config)
    result = prepare_report(events, source(), config, lambda _: None)
    prepared = prepare_bpr_snapshot(expected.validation, BprWeightConfig(view_weight=0.5))
    assert result["train"]["positive_training_pairs"] == len(prepared.training.splits.train_pairs) == 2
    assert result["future_windows"]["validation"]["interactions"] == 1
    assert not result["bpr_arrays_materialized"]
    assert events == []
    assert not any(result["snapshots"]["validation"]["invariants"].values())
    assert "PRIVATE" not in json.dumps(result)


def test_preflight_rejects_outside_events_and_handles_empty_stream(event, config):
    from scripts.preflight_research import prepare_report
    with pytest.raises(ValueError, match="outside"):
        prepare_report([event("u", "A", -1)], source(), config, lambda _: None)
    result = prepare_report([], source(), config, lambda _: None)
    assert result["history"]["events"] == 0
    assert result["train"]["positive_training_pairs"] == 0


def test_preflight_cli_cannot_train_publish_or_change_working_inputs(canonical, tmp_path, monkeypatch):
    from Application.evaluation.temporal import TemporalConfig
    from Application.model import BPRMF as core, mindbox_production_training as production
    from scripts import preflight_research as preflight
    root, catalog, _ = canonical
    def forbidden(*_args, **_kwargs):
        pytest.fail("Preflight cannot train or publish")
    monkeypatch.setattr(core, "train_prepared_data_with_metrics", forbidden)
    monkeypatch.setattr(core, "_save_artifacts", forbidden)
    monkeypatch.setattr(production, "train_and_publish_production_model", forbidden)
    monkeypatch.setattr(preflight, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preflight, "benchmark_config", lambda: TemporalConfig(
        datetime(2026, 4, 1, tzinfo=timezone.utc), datetime(2026, 5, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 1, tzinfo=timezone.utc), min_history_events=1))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "diagnostics/preflight"
    assert preflight.main(["--raw-root", str(root), "--catalog", str(catalog), "--output-dir", str(output)]) == 0
    result = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert result["protected_files_unchanged"]
    assert not result["model_training"] and not result["model_scoring"] and not result["model_publication"]
    assert result["train"]["positive_training_pairs"] == 2
    assert result["production_policy_projection"]["status"] == "NOT VERIFIED"
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(SystemExit):
        preflight.main(["--output-dir", str(tmp_path / "model")])
