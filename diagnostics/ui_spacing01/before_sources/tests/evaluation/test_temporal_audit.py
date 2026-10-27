from dataclasses import replace
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
