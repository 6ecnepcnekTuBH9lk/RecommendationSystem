from dataclasses import asdict, replace
from datetime import timedelta
from decimal import Decimal
import math

import pytest

from Application.evaluation.experiments.customer_interest import (
    InterestAction, InterestConfig, InterestInputError, InterestLine, build_profile, score_item,
)
from Application.interactions import InteractionType as Kind
from Application.mindbox.records import ProductKey
from Application.recommendation_catalog import item_from_metadata
from scripts.demo_customer_interest import CUTOFF, action, demo, fixtures, main, order


def profile(events, *, background=None, config=InterestConfig(), context=None, catalog=None):
    items, reference, ctx = fixtures()
    return build_profile("synthetic", events, items if catalog is None else catalog,
                          background or reference, context or ctx, config)


def favorites(prefix=1):
    return [action(f"{prefix}0000{i}", i, Kind.FAVORITE) for i in range(1, 4)]


def test_preferences_change_item_correspondence_and_explanations_sum():
    formal, casual = profile(favorites(1)), profile(favorites(2))
    for p, preferred, other in ((formal, "100006", "200006"), (casual, "200006", "100006")):
        good, bad = score_item(p, preferred), score_item(p, other)
        assert p.status == good.status == "READY"
        assert good.score > 0 > bad.score
        assert good.score == pytest.approx(sum(row["contribution"] for row in good.contributions))
        assert -1 <= bad.score <= 1 and 0 < good.confidence < 1
        assert all(row["contribution"] == 0 for row in good.contributions if row["weight"] == 0)


@pytest.mark.parametrize("events,status", [([], "EMPTY"), ([action("100001", 1)], "INSUFFICIENT_HISTORY"),
    ([action("100001", 1), action("100002", 2)], "INSUFFICIENT_HISTORY")])
def test_cold_start_does_not_invent_personalization(events, status):
    p = profile(events)
    result = score_item(p, "100006")
    assert result.status == status and result.score == result.confidence == 0
    assert sum(row["contribution"] for row in result.contributions) == 0


def test_repeated_views_deduplicate_product_day_and_saturate_across_days():
    one = profile([action("100001", 1)])
    repeated = profile([action("100001", 1, identifier=f"view-{i}") for i in range(1000)])
    assert repeated.diagnostics["total_mass"] == one.diagnostics["total_mass"]
    several_days = profile([action("100001", i + 1) for i in range(300)])
    assert several_days.diagnostics["total_mass"] < 2.
    assert several_days.status == "INSUFFICIENT_HISTORY"  # One distinct product is still one product.


def test_same_logical_event_not_counted_twice_and_conflicting_id_is_omitted():
    events = favorites()
    original = profile(events)
    repeated = profile(events * 3)
    assert score_item(original, "100006").score == score_item(repeated, "100006").score
    assert repeated.diagnostics["duplicate_events"] == 6
    conflicting = replace(events[0], products=(ProductKey("offline1C", "200001"),))
    changed = profile(events + [conflicting])
    assert changed.diagnostics["conflicting_event_ids"] == 1
    assert changed.diagnostics["distinct_items"] == 2 and changed.status == "INSUFFICIENT_HISTORY"


def test_cross_channel_max_does_not_triple_count_conversion_funnel():
    purchase = order("100001", 1)
    purchase_only = profile([purchase])
    all_channels = profile([purchase, action("100001", 1), action("100001", 1, Kind.FAVORITE)])
    assert all_channels.diagnostics["total_mass"] == purchase_only.diagnostics["total_mass"]


def test_multi_product_budget_is_split_before_missing_product_filter():
    event = action("100001", 1)
    mixed = replace(event, products=(ProductKey("offline1C", "100001"), ProductKey("offline1C", "999999")))
    p = profile([mixed])
    assert p.diagnostics["unresolved_product_slots"] == 1
    decay = 2 ** (-1 / 30)
    assert p.diagnostics["total_mass"] == pytest.approx(2 * -math.expm1(-decay / 4))
    assert p.diagnostics["total_mass"] < profile([event]).diagnostics["total_mass"]
    sized = replace(event, products=(ProductKey("offline1C", "100001_sizeA"), ProductKey("offline1C", "100001_sizeB")))
    assert profile([sized]).diagnostics["total_mass"] == profile([event]).diagnostics["total_mass"]


@pytest.mark.parametrize("statuses,positive", [(("CP",), True), (("delivering",), True), (("F",), True),
    (("otmenen",), False), (("out-of-stock",), False), (("Return",), False), (("klientskii-vozvrat",), False),
    (("CD",), False), (("N",), False), (("CP", "otmenen"), False), (("CP", "Return"), True)])
def test_purchase_uses_existing_consensus_not_order_placement(statuses, positive):
    p = profile([order("100001", 1, statuses)])
    assert (p.diagnostics["total_mass"] > 0) is positive


def test_snapshots_and_units_do_not_multiply_purchase_signal():
    once = profile([order("100001", 1, ("CP",))])
    many = profile([order("100001", 1, ("CP", "CP", "CP"), quantity=Decimal(10))])
    assert once.diagnostics["total_mass"] == many.diagnostics["total_mass"]
    missing_line = order("100001", 1, ("CP", "CP"))
    missing_line = replace(missing_line, snapshots=tuple(replace(s, lines=tuple(replace(line, source_line_key=None)
                                      for line in s.lines)) for s in missing_line.snapshots))
    assert profile([missing_line]).diagnostics["total_mass"] == 0
    assert profile([replace(order("100001", 1), owner_conflict=True)]).diagnostics["total_mass"] == 0


def test_uncertainty_for_same_code_overrides_other_confirmed_line():
    event = order("100001", 1)
    snapshot = event.snapshots[0]
    doubtful = InterestLine(ProductKey("offline1C", "100001"), "N", Decimal(1), "other-line")
    p = profile([replace(event, snapshots=(replace(snapshot, lines=snapshot.lines + (doubtful,)),))])
    assert p.diagnostics["total_mass"] == 0


def test_insufficient_or_balanced_reference_returns_neutral_component():
    _, background, _ = fixtures()
    p = profile(favorites(), background=replace(background, reference_users=2))
    assert p.status == "INSUFFICIENT_BACKGROUND" and score_item(p, "100006").score == 0
    balanced = profile(favorites(1) + favorites(2))
    assert balanced.status == "NO_DISTINCT_PREFERENCE"
    assert score_item(balanced, "100006").score == 0


def test_popularity_correction_reduces_common_value_bonus():
    _, balanced, _ = fixtures()
    distributions = {name: dict(values) for name, values in balanced.distributions.items()}
    for name in ("ВидНоменклатуры", "ВидАссортимента", "КатегорияНаСайте"):
        first, second = distributions[name]
        distributions[name] = {first: .9, second: .1}
    popular = replace(balanced, distributions=distributions)
    assert 0 < score_item(profile(favorites(), background=popular), "100006").score < score_item(profile(favorites()), "100006").score


def test_missing_fields_fixed_denominator_and_unseen_vocabulary():
    p = profile(favorites())
    complete, partial, missing = (score_item(p, code) for code in ("100006", "300001", "300002"))
    assert 0 < partial.score < complete.score
    assert partial.attribute_coverage == .5
    assert missing.status == "INSUFFICIENT_ITEM_ATTRIBUTES" and missing.score == 0
    assert score_item(p, "999999").status == "UNKNOWN_ITEM"
    items, background, _ = fixtures()
    items = [item_from_metadata(item.code, {name: "unseen" for name in item.attributes}) for item in items]
    unknown = profile(favorites(), catalog=items, background=background)
    assert unknown.status == "NO_ATTRIBUTE_EVIDENCE"


def test_smoothed_distributions_normalize_and_rare_lifts_are_bounded():
    p = profile(favorites())
    for attribute in p.fields:
        assert sum(attribute.prior.values()) == pytest.approx(1.)
        assert sum(attribute.posterior.values()) == pytest.approx(1.)
        assert all(-1 <= value <= 1 for value in attribute.lift.values())
        assert all(value > 0 for value in attribute.prior.values())


def test_score_matches_independent_closed_form_and_incomplete_order_is_not_purchase_signal():
    masses = [3 * 2 * -math.expm1(-2 ** (-day / 90) / 2) for day in (1, 2, 3)]
    total = math.fsum(masses)
    effective = total ** 2 / math.fsum(value ** 2 for value in masses)
    posterior = (total + 3 * .5) / (total + 3)
    expected = total / (total + 3) * effective / 5 * math.log(posterior / .5) / math.log(4)
    assert score_item(profile(favorites()), "100006").score == pytest.approx(expected, abs=1e-12)
    event = order("100001", 1)
    snapshot = event.snapshots[0]
    missing = InterestLine(None, "CP", Decimal(1), "missing-line")
    p = profile([replace(event, snapshots=(replace(snapshot, lines=snapshot.lines + (missing,)),))])
    assert p.diagnostics["unresolved_order_events"] == 1 and p.diagnostics["total_mass"] == 0


def test_recency_shift_and_future_data_do_not_rewrite_past():
    old = [action(f"20000{i}", 200 + i, Kind.FAVORITE) for i in range(1, 4)]
    changed = profile(old + favorites())
    assert score_item(changed, "100006").score > score_item(changed, "200006").score
    future = replace(action("200001", 1, Kind.FAVORITE), when=CUTOFF, available_at=CUTOFF)
    assert score_item(profile(favorites() + [future]), "100006") == score_item(profile(favorites()), "100006")
    late = replace(action("200001", 2, Kind.FAVORITE), available_at=CUTOFF + timedelta(days=1))
    assert score_item(profile(favorites() + [late]), "100006") == score_item(profile(favorites()), "100006")
    earlier = order("100001", 1, ("CP",))
    later_cancel = replace(earlier.snapshots[0], lines=(replace(earlier.snapshots[0].lines[0], status="otmenen"),),
                           available_at=CUTOFF + timedelta(days=1))
    assert profile([replace(earlier, snapshots=earlier.snapshots + (later_cancel,))]).diagnostics["total_mass"] == profile([earlier]).diagnostics["total_mass"]


@pytest.mark.parametrize("change", [{"catalog_available_at": None}, {"catalog_available_at": CUTOFF + timedelta(days=1)},
    {"identity_verified": False}, {"event_availability_verified": False}, {"dataset_kind": "REAL"}])
def test_strict_requires_evidence_and_real_inputs_are_not_supported(change):
    _, _, context = fixtures()
    with pytest.raises(InterestInputError):
        replace(context, **change)


def test_unknown_event_availability_fails_strict_but_exploratory_is_labeled():
    events = [replace(event, available_at=None) for event in favorites()]
    with pytest.raises(InterestInputError, match="AVAILABILITY"):
        profile(events)
    _, _, context = fixtures()
    result = profile(events, context=replace(context, mode="EXPLORATORY", identity_verified=False,
                                              catalog_available_at=CUTOFF + timedelta(days=1)))
    assert "AS_OF_UNVERIFIED" in score_item(result, "100006").warnings


def test_background_cannot_use_future_data_and_ambiguous_catalog_not_chosen():
    items, reference, _ = fixtures()
    with pytest.raises(InterestInputError, match="REFERENCE"):
        profile(favorites(), background=replace(reference, event_until=CUTOFF + timedelta(seconds=1)))
    with pytest.raises(InterestInputError, match="AMBIGUOUS_CATALOG"):
        profile(favorites(), catalog=items + [item_from_metadata(items[0].code, {"ВидНоменклатуры": "other"})])


def test_inputs_immutable_deterministic_and_explanations_do_not_contain_customer_id():
    items, background, context = fixtures()
    events = favorites()
    before = [asdict(event) for event in events]
    first = profile(events, catalog=items, background=background, context=context)
    second = profile(reversed(events), catalog=reversed(items), background=background, context=context)
    assert score_item(first, "100006") == score_item(second, "100006")
    assert score_item(first, "100006").as_of_utc == CUTOFF.isoformat()
    assert before == [asdict(event) for event in events]
    assert "synthetic" not in repr(score_item(first, "100006"))
    with pytest.raises(TypeError):
        first.catalog["new"] = items[0]


def test_invalid_inputs_and_bounded_streams():
    with pytest.raises(InterestInputError, match="CONSENSUS"):
        InterestAction("id", "synthetic", CUTOFF, Kind.PURCHASE, ())
    with pytest.raises(InterestInputError, match="CUSTOMER"):
        profile([replace(action("100001", 1), customer_id="other")])
    with pytest.raises(InterestInputError, match="BOUNDED_EVENTS"):
        profile(favorites() * 2, config=InterestConfig(max_events=3))
    with pytest.raises(InterestInputError, match="UNBOUNDED_CATALOG"):
        profile([], catalog=[fixtures()[0][0]] * 11, config=InterestConfig(max_catalog_items=10))
    with pytest.raises(InterestInputError):
        InterestConfig(prior_mass=float("nan"))


def test_demo_is_synthetic_and_output_guard():
    data = demo()
    assert data["dataset_kind"] == "SYNTHETIC" and len(data["cases"]) == 10
    assert all(case["matches"]["Пиджак"]["score"] == 0 for case in data["cases"] if case["profile_status"] != "READY")
    from scripts.demo_customer_interest import ROOT
    with pytest.raises(SystemExit):
        main(["--output", str(ROOT / "model")])
