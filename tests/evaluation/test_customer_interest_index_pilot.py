"""Small synthetic SQLite only; no real customers or large canonical reads."""

from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from Application import customer_history_index as index
from Application.evaluation.experiments.customer_interest import (
    InterestConfig, InterestInputError, ResearchContext, build_profile, score_item,
)
from Application.evaluation.experiments.customer_interest_index_pilot import (
    Bounds, Budget, PilotError, convert_events, evaluate_profiles, feature_configs, open_reader,
    read_customer, reference_background, require_ready, select_customers,
)
from Application.product_resolution import CatalogDiagnostics, ProductCatalog, ProductResolver
from scripts.demo_customer_interest import CUTOFF, action as demo_action, fixtures
from tests.test_customer_history_index import action, build, order, source as source_fixture, update_rows
from tests.test_customer_history_service import TIME


@pytest.fixture
def source(tmp_path):
    return source_fixture.__wrapped__(tmp_path)


def resolver():
    return ProductResolver(ProductCatalog(frozenset(("000001", "000002")), CatalogDiagnostics()))


def read_all(source, *, bounds=Bounds()):
    budget = Budget(bounds)
    root, service = source[:2]
    since = TIME.replace(hour=0, minute=0, second=0, microsecond=0)
    with open_reader(service, root, budget) as (connection, ready):
        events, diagnostics = read_customer(connection, ready, "C", since, since + timedelta(days=1), budget)
    return events, diagnostics, budget


def test_explicit_retrospective_mode_preserves_publication_and_never_claims_strict():
    items, bg, _ = fixtures()
    later = CUTOFF + timedelta(days=30)
    context = ResearchContext(CUTOFF, later, mode="EXPLORATORY", dataset_kind="INDEX_EXPLORATORY", knowledge_cutoff=later)
    events = [replace(demo_action(f"10000{i}", i), available_at=later) for i in range(1, 4)]
    profile = build_profile("synthetic", events, items, replace(bg, availability_verified=False), context)
    match = score_item(profile, "100006")
    assert profile.status == "READY" and profile.mode == "EXPLORATORY"
    assert "SYNTHETIC_ONLY" not in match.warnings and "RETROSPECTIVE_KNOWLEDGE_NOT_FORECAST" in match.warnings
    assert match.as_of_utc == CUTOFF.isoformat() and match.knowledge_cutoff_utc == later.isoformat()
    after_knowledge = [replace(e, available_at=later + timedelta(seconds=1)) for e in events]
    assert build_profile("synthetic", after_knowledge, items, bg, context).status == "EMPTY"
    with pytest.raises(InterestInputError):
        replace(context, mode="STRICT", identity_verified=True, event_availability_verified=True)
    with pytest.raises(InterestInputError):
        replace(context, knowledge_cutoff=CUTOFF - timedelta(days=1))


def test_synthetic_future_guard_cannot_be_extended():
    _, _, context = fixtures()
    with pytest.raises(InterestInputError, match="SYNTHETIC_KNOWLEDGE"):
        replace(context, knowledge_cutoff=CUTOFF + timedelta(days=1))
    # Backward-compatible default when callers replace the original event cutoff.
    moved = replace(context, cutoff=CUTOFF + timedelta(days=1))
    assert moved.knowledge_cutoff is None


@pytest.mark.parametrize("status", [s for s in index.HistoryIndexStatus if s is not index.HistoryIndexStatus.READY])
def test_only_ready_index_allowed(status):
    with pytest.raises(PilotError, match="INDEX_" + status.value):
        require_ready(SimpleNamespace(status=status))


def test_missing_real_backend_fails_without_auto_build(tmp_path):
    service = index.CustomerHistoryIndexService(cache_dir=tmp_path)
    with pytest.raises(PilotError, match="INDEX_MISSING"):
        with open_reader(service, tmp_path / "absent", Budget()):
            pytest.fail("Missing index must not open")
    assert not tuple(tmp_path.glob("*.sqlite"))


def test_pages_are_complete_stable_unique_and_read_only(source):
    update_rows(source, "actions", [action(f"event-{n}") for n in range(6)])
    assert build(source).status is index.HistoryIndexStatus.READY
    before = index._stat(source[1].index_path(raw_root=source[0]))
    first, diagnostics, budget = read_all(source, bounds=Bounds(page_size=2))
    second, _, _ = read_all(source, bounds=Bounds(page_size=2))
    assert len(first) == len({e.event_key for e in first}) == 7  # Six actions and one order.
    assert [e.event_key for e in first] == [e.event_key for e in second]
    assert diagnostics["pages"] == 4 and budget.events == 10  # Includes repeated lookahead headers.
    assert index._stat(source[1].index_path(raw_root=source[0])) == before


def test_sampling_uses_only_canonical_index_scope_and_deduplicates(source):
    assert build(source).status is index.HistoryIndexStatus.READY
    budget = Budget()
    with open_reader(source[1], source[0], budget) as (connection, _):
        ids = list(select_customers(connection, budget, anchors=("B", "C", "C")))
    assert ids == ["C"] and budget.customers == 1  # Historical A/B were merged by original indexer.


@pytest.mark.parametrize("statuses,positive", [(('CP',), True), (('F',), True), (('delivering',), True),
    (('otmenen',), False), (('Return',), False), (('N',), False), (('CP', 'otmenen'), False), (('CP', 'Return'), True)])
def test_real_history_adapter_order_semantics_all_versions(source, statuses, positive):
    rows = []
    for status in statuses:
        raw = order()
        for ordinal, line in enumerate(raw["lines"]):
            line["id"] = f"stable-{ordinal}"
            line["quantity"] = 1
            line["product"]["ids"] = {"offline1C": f"{ordinal + 1:06d}"}
            line["status"]["ids"]["externalId"] = status
        rows.append(raw)
    update_rows(source, "actions", [])
    update_rows(source, "orders", rows)
    assert build(source).status is index.HistoryIndexStatus.READY
    events, diagnostics, _ = read_all(source)
    adapted, counters = convert_events("C", events, resolver())
    assert len(adapted) == 1 and len(adapted[0].snapshots) == len(set(statuses))
    assert bool(counters["ORDER_WITH_POSITIVE_CODES"]) is positive
    assert bool(diagnostics["partial_conflict_pages"]) == (len(set(statuses)) > 1)
    assert all(s.available_at is not None for s in adapted[0].snapshots)


def test_incomplete_history_and_conflicting_actions_not_silently_used(source):
    assert build(source).status is index.HistoryIndexStatus.READY
    with open_reader(source[1], source[0], Budget()) as (connection, ready):
        events, diagnostics = read_customer(connection, replace(ready, coverage=()), "C", TIME, TIME + timedelta(days=1), Budget())
    assert events is None and diagnostics == {"INCOMPLETE_SOURCE_COVERAGE": 1}
    first, second = action(), action()
    second["products"] = [{"ids": {"offline1C": "000002"}}]
    update_rows(source, "actions", [first, second])
    assert build(source).status is index.HistoryIndexStatus.READY
    events, _, _ = read_all(source)
    adapted, counters = convert_events("C", events, resolver())
    assert counters["ACTION_VARIANTS_CONFLICT"] == 1
    assert all(not hasattr(e, "products") for e in adapted)


def test_customer_over_limit_and_large_payload_are_not_truncated_profiles(source):
    update_rows(source, "actions", [action(f"e-{n}") for n in range(5)])
    assert build(source).status is index.HistoryIndexStatus.READY
    events, diagnostics, _ = read_all(source, bounds=Bounds(events_per_customer=3, page_size=2))
    assert events is None and diagnostics["CUSTOMER_HISTORY_LIMIT"] == 1
    events, diagnostics, _ = read_all(source, bounds=Bounds(products_per_variant=1))
    assert events is None and diagnostics["EVENT_PAYLOAD_LIMIT"] == 1


def test_owner_conflict_is_not_assigned_to_one_customer(source):
    update_rows(source, "orders", [order(), order(customer="OTHER")])
    assert build(source).status is index.HistoryIndexStatus.READY
    events, _, _ = read_all(source)
    adapted, diagnostics = convert_events("C", events, resolver())
    assert diagnostics["OWNER_CONFLICT"] == 1
    assert all(not hasattr(event, "snapshots") for event in adapted)


def test_limits_time_cancel_memory_poll_and_events():
    now = [0.]
    budget = Budget(Bounds(seconds=1), clock=lambda: now[0])
    now[0] = 1.
    with pytest.raises(PilotError, match="TIME_LIMIT"):
        budget.check()
    with pytest.raises(PilotError, match="CANCELLED"):
        Budget(cancelled=lambda: True).check()
    def memory_failure():
        raise PilotError("MEMORY_LIMIT")
    with pytest.raises(PilotError, match="MEMORY_LIMIT"):
        Budget(poll=memory_failure).check()
    with pytest.raises(PilotError, match="EVENT_LIMIT"):
        Budget(Bounds(events=2)).consume_events(3)
    limited = Budget(Bounds(customers=1))
    limited.consume_customer()
    with pytest.raises(PilotError, match="CUSTOMER_LIMIT"):
        limited.consume_customer()


@pytest.mark.parametrize("field,value", [("events", 10_001), ("customers", 101), ("seconds", 181),
    ("private_bytes", 1536 * 2**20 + 1), ("disk_floor_bytes", 2 * 2**30)])
def test_resource_bounds_cannot_be_weakened(field, value):
    with pytest.raises(PilotError):
        Bounds(**{field: value})


def test_reference_is_user_normalized_and_low_field_support_excluded():
    items, background, context = fixtures()
    few = build_profile("synthetic", [demo_action("100001", 1)], items, background, context)
    many = build_profile("synthetic", [demo_action("200001", i) for i in range(1, 31)], items, background, context)
    reference, support = reference_background([few, many] * 10, CUTOFF)
    kind = reference.distributions["ВидНоменклатуры"]
    assert kind["Пиджак"] == pytest.approx(.5) and kind["Футболка"] == pytest.approx(.5)
    assert support["ВидНоменклатуры"] == 20
    assert not reference_background([few], CUTOFF)[0].distributions


def test_aggregate_evaluation_is_deterministic_disjoint_and_has_no_customer_ids():
    items, _, context = fixtures()
    context = replace(context, mode="EXPLORATORY")
    cohort = []
    ids = [f"PRIVATE_CUSTOMER_{n}" for n in range(26)]
    for customer in ids:
        events = tuple(replace(demo_action(f"{1 + (int(customer.rsplit('_', 1)[1]) % 2)}0000{i}", i), customer_id=customer)
                       for i in range(1, 4))
        cohort.append((customer, events, 3, False))
    left = evaluate_profiles(cohort, items, context, Budget(), panel_codes=("100006", "200006"))
    right = evaluate_profiles(cohort, items, context, Budget(), panel_codes=("100006", "200006"))
    assert left == right and left["reference_users"] == 24 and left["evaluated_users"] == 26
    assert left["heldout_evaluation_users"] == 2 and left["reference_loo_evaluation_users"] == 24
    assert left["quality_metrics"] is None
    encoded = json.dumps(left, ensure_ascii=False)
    assert all(customer not in encoded for customer in ids)
    assert len(encoded.encode()) < 128 * 2**10
    assert set(feature_configs()) == {"KIND", "ASSORTMENT", "BOTH", "WITH_CATEGORY", "CATEGORY"}


def test_observed_masses_are_immutable_and_normalize_to_support():
    items, background, context = fixtures()
    profile = build_profile("synthetic", [demo_action("100001", 1)], items, background, context, InterestConfig())
    for field in profile.fields:
        assert sum(field.counts.values()) == pytest.approx(field.mass)
        with pytest.raises(TypeError):
            field.counts["new"] = 1


def test_leave_one_out_never_counts_own_user_toward_minimum_reference_support():
    items, _, context = fixtures()
    context = replace(context, mode="EXPLORATORY")
    cohort = [(f"person-{n}", tuple(replace(demo_action(f"10000{i}", i), customer_id=f"person-{n}")
                                  for i in range(1, 4)), 3, False) for n in range(20)]
    result = evaluate_profiles(cohort, items, context, Budget(), panel_codes=("100006", "200006"))
    assert result["reference_users"] == result["reference_loo_evaluation_users"] == 20
    assert result["statuses"]["WITH_CATEGORY"] == {"INSUFFICIENT_BACKGROUND": 20}


def test_duplicate_canonical_users_cannot_inflate_reference_support():
    items, _, context = fixtures()
    with pytest.raises(PilotError, match="COHORT_SCOPE"):
        evaluate_profiles([("same", (), 0, False)] * 2, items, replace(context, mode="EXPLORATORY"), Budget(),
                          panel_codes=("100006",))


@pytest.mark.parametrize("panel", [(), ("100006", "100006"), tuple(str(i) for i in range(257))])
def test_invalid_diagnostic_panel_rejected(panel):
    items, _, context = fixtures()
    with pytest.raises(PilotError, match="DIAGNOSTIC_PANEL"):
        evaluate_profiles([], items, replace(context, mode="EXPLORATORY"), Budget(), panel_codes=panel)


def test_integrity_receipt_uses_database_size_not_secondary_index_bytes(tmp_path, monkeypatch):
    from scripts import run_customer_interest_index_pilot as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    receipt = tmp_path / "user_settings/_history_real_index02"
    receipt.mkdir(parents=True)
    old = tmp_path / "filtered_data/.customer-interest-profile-02-20261026"
    old.mkdir(parents=True)
    database = tmp_path / "history.sqlite"
    database.write_bytes(b"test")
    common = {"generation": "synthetic-gen", "source_version": "synthetic-version"}
    (receipt / "integrity_result.json").write_text(json.dumps({**common, "integrity_status": "READY"}))
    (receipt / "final_checks.json").write_text(json.dumps({**common, "schema_version": 2, "sqlite_index_bytes": 9999}))
    (receipt / "build_report.json").write_text(json.dumps({**common, "index_bytes": 4}))
    (old / "BASELINE.json").write_text(json.dumps({"protected_stat": {"history.sqlite": [4, database.stat().st_mtime_ns]}}))
    ready = SimpleNamespace(status=index.HistoryIndexStatus.READY, coverage=(), **common)
    service = SimpleNamespace(get_status=lambda **kwargs: ready, index_path=lambda **kwargs: database)
    _, result = runner.index_preflight(service, tmp_path)
    assert all(result["checks"].values())
    (receipt / "build_report.json").write_text(json.dumps({**common, "index_bytes": 5}))
    with pytest.raises(PilotError, match="verified_size_matches"):
        runner.index_preflight(service, tmp_path)


@pytest.mark.parametrize("interrupt,reason", [("timeout", "HARD_WALL_TIME_LIMIT"), ("manual", "MANUAL_INTERRUPT")])
def test_parent_watchdog_stops_worker_and_only_writes_aggregate_receipt(tmp_path, monkeypatch, interrupt, reason):
    from scripts import run_customer_interest_index_pilot as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "filtered_data/test"
    output.mkdir(parents=True)
    stopped, calls = [], []
    def communicate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            if interrupt == "manual":
                raise KeyboardInterrupt
            raise runner.subprocess.TimeoutExpired("synthetic", 180)
        return b"", b""
    process = SimpleNamespace(communicate=communicate, kill=lambda: stopped.append("kill"),
                              terminate=lambda: stopped.append("terminate"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    assert runner.main(["--output", str(output)]) == 1
    result = json.loads((output / "PILOT.json").read_text())
    assert result["reason"] == reason and len(stopped) == 1
    assert sorted(p.name for p in output.iterdir()) == ["PILOT.json"]
