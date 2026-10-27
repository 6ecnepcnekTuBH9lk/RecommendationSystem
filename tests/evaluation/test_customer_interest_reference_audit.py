"""No real data; strict namespace choice is never relaxed by diagnostics."""

from collections import Counter
from types import SimpleNamespace

import pytest

from Application.customer_history_records import HistoryProductReference, HistoryProvenance, HistoryQuality as Q
from Application.customer_history_service import HistoryVariant, LogicalHistoryEvent
from Application.evaluation.experiments.customer_interest import build_profile
from Application.evaluation.experiments.customer_interest_reference_audit import background_reliability, reference_diagnostic, scope_matches, view_funnel
from Application.mindbox.records import ProductKey
from Application.product_resolution import CatalogDiagnostics, ProductCatalog, ProductResolver
from Application.recommendation_catalog import item_from_metadata
from scripts.demo_customer_interest import CUTOFF, action, fixtures


NAMESPACES = ("offline1C", "kanzlerKz")


def catalog():
    return SimpleNamespace(items={"000001": item_from_metadata("000001", {"ВидНоменклатуры": "Kind"}),
                                  "000002": item_from_metadata("000002", {})},
                           all_codes=frozenset(("000001", "000002", "000003")))


def reference(values, selected=None):
    keys = tuple(ProductKey(ns, value) for ns, value in values)
    return HistoryProductReference(keys, keys[selected] if selected is not None else None)


@pytest.mark.parametrize("ref,reason", [
    (reference([]), "NO_ID"),
    (reference([("website", "000001")]), "UNSUPPORTED_NAMESPACE"),
    (reference([("offline1C", "000001"), ("kanzlerKz", "000001")]), "MULTIPLE_SUPPORTED_NAMESPACES"),
    (reference([("offline1C", "000001"), ("kanzlerKz", "000002")]), "MULTIPLE_SUPPORTED_NAMESPACES"),
    (reference([("offline1C", "999999_size")], 0), "UNKNOWN_CATALOG"),
    (reference([("offline1C", "000003_size")], 0), "CATALOG_ATTRIBUTE_CONFLICT"),
    (reference([("offline1C", "000002_size")], 0), "NO_ACTIVE_ATTRIBUTES"),
    (reference([("offline1C", "000001_size")], 0), "RESOLVED"),
    (reference([("kanzlerKz", "000001_size")], 0), "RESOLVED"),
])
def test_reference_reasons_and_existing_bpr_choice(ref, reason):
    result = reference_diagnostic(ref, catalog(), NAMESPACES)
    assert result["reason"] == reason
    assert result["normalized_code"] == result["bpr_normalized_code"]
    if reason == "MULTIPLE_SUPPORTED_NAMESPACES":
        assert not result["flags"][3] and result["normalized_code"] is None


def test_convergent_dual_namespace_is_still_not_a_verified_selected_product():
    ref = reference([("offline1C", "000001_sizeA"), ("kanzlerKz", "000001_sizeB")])
    before = ref
    result = reference_diagnostic(ref, catalog(), NAMESPACES)
    assert result["ambiguous_convergent_prefix"] and result["normalized_code"] is None
    assert ref is before and ref.product is None


def test_attribute_conflict_is_not_unknown_identity_or_last_row_wins():
    result = reference_diagnostic(reference([("offline1C", "000003")], 0), catalog(), NAMESPACES)
    assert result["flags"][5] and not result["flags"][6]
    assert result["reason"] == "CATALOG_ATTRIBUTE_CONFLICT"


def test_retained_unique_key_without_selected_projection_is_reported_as_inconsistency():
    ref = reference([("offline1C", "000001")])
    result = reference_diagnostic(ref, catalog(), NAMESPACES)
    assert result["reason"] == "SELECTED_KEY_INCONSISTENT" and result["normalized_code"] is None
    assert result["bpr_normalized_code"] == "000001"


def test_history_normalizer_preserves_original_ambiguity_without_dropping_keys():
    from Application.customer_history_normalization import normalize_action
    from Application.mindbox.identity import CustomerIdResolver
    from tests.test_customer_history_index import action as raw_action
    raw = raw_action(products=[{"ids": {"offline1C": "000001", "kanzlerKz": "000002"}}])
    event = normalize_action(raw, CustomerIdResolver([]))
    ref = event.details.products[0].reference
    assert ref.product is None and len(ref.keys) == 2 and Q.AMBIGUOUS_PRODUCT_NAMESPACE in ref.quality
    assert reference_diagnostic(ref, catalog(), NAMESPACES)["reason"] == "MULTIPLE_SUPPORTED_NAMESPACES"


def test_background_jackknife_reports_single_user_values_without_changing_profiles():
    items, bg, context = fixtures()
    profiles = [build_profile("synthetic", [action("100001", 1)], items, bg, context) for _ in range(21)]
    profiles.append(build_profile("synthetic", [action("200001", 1)], items, bg, context))
    before = tuple(dict(p.fields[0].counts) for p in profiles)
    result = background_reliability(profiles, CUTOFF)
    for name in ("ВидНоменклатуры", "ВидАссортимента", "КатегорияНаСайте"):
        values = result[name]
        assert values["value_user_support_histogram"]["1"] == 1
        assert values["loo_values_losing_all_evidence"]["max"] == 1
        assert values["smoothed_floor"] == .025
    assert tuple(dict(p.fields[0].counts) for p in profiles) == before


def logical_view(products, *, favorite=False):
    from Application.customer_history_normalization import normalize_action
    from Application.mindbox.identity import CustomerIdResolver
    from tests.test_customer_history_index import action as raw_action
    origin = HistoryProvenance(observed_at_utc=CUTOFF)
    raw = raw_action(products=products)
    if favorite:
        raw["actionTemplate"]["ids"]["systemName"] = "DobavlenieProduktaVSpisokVOperaciiDobavlenie"
    event = normalize_action(raw, CustomerIdResolver([]), provenance=origin)
    return LogicalHistoryEvent(event.event_key, event.source, event.source_event_id, event.canonical_customer_id,
        (event.canonical_customer_id,), event.kind, event.event_datetime_utc, (HistoryVariant(event, (origin,)),),
        1, False, False, False, event.quality)


def funnel(products):
    resolver = ProductResolver(ProductCatalog(frozenset(catalog().items), CatalogDiagnostics()))
    return view_funnel((logical_view(products),), catalog(), NAMESPACES, resolver, "A")


def test_empty_product_array_is_one_event_loss_not_two_and_no_occurrences():
    result, _, _ = funnel([])
    assert result["events_all"]["VIEW"] == 1 and result["events_all"]["PRODUCT_OBJECT"] == 0
    assert result["reasons"]["EVENT_NO_PRODUCT_OBJECT"] == 1
    assert sum(result["event_reason_combinations"].values()) == 1
    assert result["product_occurrences"] == {} and result["baseline_rejected_views"] == 1


def test_mixed_valid_and_bad_occurrences_report_collateral_without_repairing_data():
    result, _, _ = funnel([{"ids": {"offline1C": "000001"}}, {"ids": {"website": "000001"}}])
    assert result["events_any"]["CATALOG"] == 1 and result["events_all"]["CATALOG"] == 0
    assert result["collateral_events_with_resolved_occurrence"] == 1
    assert result["events_all"]["PROFILE_SIGNAL"] == 0  # No silent partial repair.
    assert result["quality_occurrences"][Q.UNSUPPORTED_PRODUCT_NAMESPACE.value] == 1


def test_event_occurrence_and_unique_product_denominators_are_distinct():
    result, unique, source = funnel([{"ids": {"offline1C": "000001_size"}}] * 2)
    assert result["events_all"]["CATALOG"] == 1 and result["product_occurrences"]["CATALOG"] == 2
    assert len(unique["CATALOG"]) == len(source) == 1
    assert result["baseline_rejected_views"] == 0


def test_scope_does_not_mistake_one_favorite_loss_for_an_extra_view():
    prior = {"source_activity_strata": {"0": 71, "1-5": 28}, "buyers_probed": 100,
             "channels": {"VIEW": 197, "FAVORITE": 1},
             "diagnostics": {"MISSING_ACTION_PRODUCT": 166, "ADAPTED_VIEW": 32, "pages": 111}}
    scalar = Counter(baseline_rejected_views=165)
    totals = {"events_all": Counter(VIEW=197), "conversion": Counter(MISSING_ACTION_PRODUCT=166, ADAPTED_VIEW=32)}
    assert scope_matches(prior, prior["source_activity_strata"], 100, scalar, totals)
    scalar["baseline_rejected_views"] = 166
    assert not scope_matches(prior, prior["source_activity_strata"], 100, scalar, totals)


def test_favorite_missing_product_is_measured_separately():
    event = logical_view([], favorite=True)
    resolver = ProductResolver(ProductCatalog(frozenset(catalog().items), CatalogDiagnostics()))
    result, _, _ = view_funnel((event,), catalog(), NAMESPACES, resolver, "A", kind=event.kind)
    assert result["events_all"]["FAVORITE"] == 1 and result["events_all"]["PRODUCT_OBJECT"] == 0
    assert result["baseline_rejected_favorites"] == 1 and result["reasons"]["EVENT_NO_PRODUCT_OBJECT"] == 1


def test_audit_cli_publishes_separate_receipt_without_running_a_pilot(tmp_path, monkeypatch):
    from scripts import run_customer_interest_index_pilot as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "filtered_data/audit"
    output.mkdir(parents=True)
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, communicate=lambda **kwargs: (b'{"status":"COMPLETED","comparative_pilot_executed":false}', b""))
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    assert runner.main(["--output", str(output), "--audit-references"]) == 0
    assert "--audit-references" in commands[0]
    assert (output / "AUDIT.json").is_file() and not (output / "PILOT.json").exists()


def test_audit_cannot_turn_limits_smoke_into_real_data_read(tmp_path, monkeypatch):
    from scripts import run_customer_interest_index_pilot as runner
    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible flags must fail before starting a worker")
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    with pytest.raises(SystemExit):
        runner.main(["--output", str(tmp_path), "--audit-references", "--limits-smoke"])
