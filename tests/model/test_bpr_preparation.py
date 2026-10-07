from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from Application.interactions import InteractionRecord, InteractionSource, InteractionType as Kind
from Application.mindbox.records import ProductKey
from Application.model import BPRMF as legacy
from Application.model.bpr_preparation import (
    BprEvent, BprPreparationConfig, BprPreparationError, BprWeightConfig, DateMode, prepare_bpr, to_bpr_event,
)
from Application.product_resolution import ResolvedInteraction


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def resolved(kind=Kind.VIEW, user="SECRET_USER", item="SECRET_ITEM", timestamp=BASE, quantity=None):
    return ResolvedInteraction(InteractionRecord(
        source_customer_id="SECRET_SOURCE", customer_id=user,
        product=ProductKey("offline1C", "SECRET_FULL_PRODUCT"), interaction_type=kind,
        event_datetime_utc=timestamp, source=InteractionSource.ORDER if kind is Kind.PURCHASE else InteractionSource.ACTION,
        source_event_id="SECRET_EVENT", quantity=quantity,
    ), item)


def legacy_input(records, mode):
    frames = {}
    for kind in Kind:
        rows = []
        for record in records:
            event = record.interaction
            if event.interaction_type is not kind:
                continue
            timestamp = event.event_datetime_utc.astimezone(timezone.utc)
            if mode is DateMode.LEGACY_DATE:
                timestamp = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            rows.append({"MindboxID": event.customer_id, "КодНоменклатуры": record.item_id,
                         "Дата": timestamp.isoformat(), "ТипТовара": "Номенклатура",
                         "Количество": None if event.quantity is None else str(event.quantity)})
        frames[kind] = pd.DataFrame(rows, columns=["MindboxID", "КодНоменклатуры", "Дата", "ТипТовара", "Количество"])
    return frames[Kind.PURCHASE], frames[Kind.VIEW], frames[Kind.FAVORITE]


def assert_parity(records, config=BprPreparationConfig()):
    orders, views, favorites = legacy_input(records, config.date_mode)
    cfg = legacy.TrainConfig(w_view_item=config.weights.view_weight, w_favorite=config.weights.favorite_weight,
                             w_purchase=config.weights.purchase_weight,
                             min_user_interactions_for_eval=config.min_user_interactions_for_eval)
    maps = legacy._build_mappings(orders, views, favorites)
    old_events = legacy._collect_user_item_events(orders, views, favorites, maps, cfg)
    old_split = legacy._train_test_split_last_per_user(old_events, cfg, len(maps.idx2user))
    events = [to_bpr_event(record, config.weights) for record in records]
    new = prepare_bpr(iter(events), config)
    for name in ("user2idx", "idx2user", "item2idx", "idx2item"):
        assert getattr(new.mappings, name) == getattr(maps, name)
    ordered = [event for kind in (Kind.PURCHASE, Kind.FAVORITE, Kind.VIEW)
               for event in events if event.interaction_type is kind]
    np.testing.assert_array_equal(old_events["w"].to_numpy(), [event.weight for event in ordered])
    for name in ("train_pairs", "train_weights", "eval_users", "eval_items"):
        np.testing.assert_array_equal(getattr(new.splits, name), getattr(old_split, name))
    assert new.splits.user_pos_train == old_split.user_pos_train
    assert new.diagnostics.events_total == len(records)
    assert new.diagnostics.train_events_before_aggregation + new.diagnostics.eval_events == len(records)
    return new


def test_defaults_match_production_train_config():
    old, new = legacy.TrainConfig(), BprPreparationConfig()
    assert new.weights.view_weight == old.w_view_item == 0.1
    assert new.weights.favorite_weight == old.w_favorite == 2.0
    assert new.weights.purchase_weight == old.w_purchase == 10.0
    assert new.min_user_interactions_for_eval == old.min_user_interactions_for_eval == 10
    assert new.weights.min_purchase_quantity == 1
    assert new.weights.max_purchase_quantity == 10
    assert new.date_mode is DateMode.LEGACY_DATE
    with pytest.raises(FrozenInstanceError):
        new.weights.view_weight = 3


@pytest.mark.parametrize("quantity,weight", [
    (None, 10), (Decimal("0"), 10), (Decimal("1"), 10), (Decimal("1.5"), 15),
    (Decimal("2"), 20), (Decimal("15"), 100), (Decimal("-5"), 10),
    (Decimal("NaN"), 10), (Decimal("Infinity"), 100), (Decimal("-Infinity"), 10),
    ("bad numeric", 10), (Decimal("1.23456789"), 12.3456789),
])
def test_purchase_quantity_legacy_parity(quantity, weight):
    record = resolved(Kind.PURCHASE, quantity=quantity)
    assert to_bpr_event(record).weight == pytest.approx(weight)
    assert_parity([record])


@pytest.mark.parametrize("kind,weight", [(Kind.VIEW, 0.1), (Kind.FAVORITE, 2.0)])
def test_non_purchase_ignores_quantity_and_event_is_minimal(kind, weight):
    record = resolved(kind, quantity=Decimal("100"))
    event = to_bpr_event(record)
    assert event.weight == weight
    assert event.customer_id == record.interaction.customer_id
    assert event.item_id == record.item_id
    assert event.timestamp is record.interaction.event_datetime_utc
    assert set(vars(event)) == {"customer_id", "item_id", "timestamp", "interaction_type", "weight"}
    assert "SECRET" not in repr(event)
    with pytest.raises(FrozenInstanceError):
        event.weight = 42
    assert_parity([record])


@pytest.mark.parametrize("mode", list(DateMode))
@pytest.mark.parametrize("same_timestamp", [False, True])
def test_mixed_events_mapping_split_aggregation_parity(mode, same_timestamp):
    records = []
    for user, count in (("z-user", 9), ("a-user", 10), ("m-user", 13)):
        for index in range(count):
            records.append(resolved(
                list(Kind)[index % 3], user, f"item-{index % 4}",
                BASE if same_timestamp else BASE + timedelta(hours=(count - index) * 5),
                [None, Decimal("1.5"), Decimal("15")][index % 3],
            ))
    new = assert_parity(records, BprPreparationConfig(date_mode=mode))
    assert new.diagnostics.interaction_count_eligible_users == 2
    assert new.diagnostics.eligible_eval_users == new.diagnostics.eval_events == 0
    assert new.diagnostics.excluded_seen_eval_users == 2
    assert new.diagnostics.train_events_before_aggregation == 32
    assert new.diagnostics.train_pairs_after_aggregation < 32


def test_legacy_mapping_orders_differ_for_users_and_items():
    records = [resolved(Kind.VIEW, "v-user", "v-item"), resolved(Kind.FAVORITE, "f-user", "f-item"),
               resolved(Kind.PURCHASE, "p-user", "p-item")]
    new = assert_parity(records)
    assert new.mappings.idx2user == ["p-user", "v-user", "f-user"]
    assert new.mappings.idx2item == ["p-item", "f-item", "v-item"]


def test_stable_order_of_equal_type_and_timestamp_selects_last_input():
    records = [resolved(item=f"item-{index}") for index in range(10)]
    first = assert_parity(records)
    reversed_result = assert_parity(list(reversed(records)))
    assert first.mappings.idx2item[first.splits.eval_items[0]] == "item-9"
    assert reversed_result.mappings.idx2item[reversed_result.splits.eval_items[0]] == "item-0"


def test_date_granularity_changes_holdout_and_full_timestamp_matches_chronology():
    from Application.files.files_processing import _parse_interaction_date

    records = [resolved(item=f"early-view-{index}", timestamp=BASE + timedelta(hours=8)) for index in range(9)]
    records.append(resolved(Kind.PURCHASE, item="late-purchase", timestamp=BASE + timedelta(hours=22)))
    parsed, malformed = _parse_interaction_date(pd.Series(["2026-01-01 08:00:00", "2026-01-01 22:00:00"]))
    assert malformed == 0
    assert parsed.iloc[0] == parsed.iloc[1] == pd.Timestamp("2026-01-01")
    dated = assert_parity(records)
    precise = assert_parity(records, BprPreparationConfig(date_mode=DateMode.FULL_TIMESTAMP))
    assert dated.mappings.idx2item[dated.splits.eval_items[0]] == "early-view-8"
    assert precise.mappings.idx2item[precise.splits.eval_items[0]] == "late-purchase"
    assert to_bpr_event(records[-1]).timestamp.hour == 22


def test_repeated_last_item_is_not_an_evaluation_target():
    records = [resolved(item="A", timestamp=BASE), *[
        resolved(item="B", timestamp=BASE + timedelta(days=1)) for _ in range(8)],
        resolved(Kind.PURCHASE, item="A", timestamp=BASE + timedelta(days=2))]
    result = assert_parity(records)
    assert result.splits.eval_users.size == result.splits.eval_items.size == 0
    assert result.diagnostics.train_events_before_aggregation == 10
    weights = {result.mappings.idx2item[item]: weight for (_, item), weight
               in zip(result.splits.train_pairs, result.splits.train_weights)}
    assert weights == pytest.approx({"A": 10.1, "B": 0.8})


def test_timezone_offsets_represent_same_utc_instant():
    utc = resolved(timestamp=BASE + timedelta(hours=23))
    offset = replace(utc, interaction=replace(utc.interaction, event_datetime_utc=
                     utc.interaction.event_datetime_utc.astimezone(timezone(timedelta(hours=3)))))
    for mode in DateMode:
        assert_parity([utc, offset], BprPreparationConfig(date_mode=mode))


def test_empty_and_custom_config():
    result = prepare_bpr([])
    assert result.splits.train_pairs.shape == (0, 2)
    assert result.splits.train_weights.dtype == np.float64
    assert result.diagnostics.total_train_weight == 0
    cfg = BprPreparationConfig(BprWeightConfig(0.3, 4, 7), min_user_interactions_for_eval=2)
    assert_parity([resolved(kind) for kind in Kind], cfg)


@pytest.mark.parametrize("kwargs", [{"view_weight": -1}, {"favorite_weight": float("nan")},
                                    {"min_purchase_quantity": 20}, {"purchase_weight": True}])
def test_bad_weights(kwargs):
    with pytest.raises(BprPreparationError):
        BprWeightConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"min_user_interactions_for_eval": 0}, {"min_user_interactions_for_eval": True},
                                    {"date_mode": "unknown"}])
def test_bad_preparation_config(kwargs):
    with pytest.raises(BprPreparationError):
        BprPreparationConfig(**kwargs)


def test_naive_timestamp_is_safe_error():
    with pytest.raises(BprPreparationError) as exc:
        BprEvent("SECRET_USER", "SECRET_ITEM", datetime(2026, 1, 1), Kind.VIEW, 0.1)
    assert "SECRET" not in str(exc.value)


@pytest.mark.parametrize("latest", [Kind.PURCHASE, Kind.FAVORITE, Kind.VIEW])
def test_same_day_full_timestamp_holdout_follows_time_with_legacy_characterization(latest):
    records = [resolved(kind, item=kind.value, timestamp=BASE + timedelta(hours=20 if kind is latest else 8))
               for kind in Kind]
    dated = assert_parity(records, BprPreparationConfig(min_user_interactions_for_eval=3))
    precise = assert_parity(records, BprPreparationConfig(min_user_interactions_for_eval=3,
                                                        date_mode=DateMode.FULL_TIMESTAMP))
    assert dated.mappings == precise.mappings
    assert dated.mappings.idx2item[dated.splits.eval_items[0]] == Kind.VIEW.value
    assert precise.mappings.idx2item[precise.splits.eval_items[0]] == latest.value
    assert precise.diagnostics.train_events_before_aggregation == 2
    assert precise.splits.user_pos_train[0] == set(range(3)) - {precise.splits.eval_items[0]}


def test_full_timestamp_exact_ties_keep_type_and_source_row_tiebreak():
    records = [resolved(Kind.VIEW, item="first-view"), resolved(Kind.PURCHASE, item="purchase"),
               resolved(Kind.FAVORITE, item="favorite"), resolved(Kind.VIEW, item="last-view")]
    config = BprPreparationConfig(min_user_interactions_for_eval=3, date_mode=DateMode.FULL_TIMESTAMP)
    first = assert_parity(records, config)
    repeated = assert_parity(records, config)
    assert first.mappings.idx2item[first.splits.eval_items[0]] == "last-view"
    np.testing.assert_array_equal(first.splits.eval_items, repeated.splits.eval_items)
    reversed_result = assert_parity(list(reversed(records)), config)
    assert reversed_result.mappings.idx2item[reversed_result.splits.eval_items[0]] == "first-view"


@pytest.mark.parametrize("mode", list(DateMode))
def test_events_across_midnight_keep_chronological_holdout(mode):
    records = [resolved(Kind.PURCHASE, item="after-midnight", timestamp=BASE + timedelta(days=1, minutes=10)),
               resolved(Kind.VIEW, item="before-midnight", timestamp=BASE + timedelta(hours=23, minutes=50))]
    result = assert_parity(records, BprPreparationConfig(min_user_interactions_for_eval=2, date_mode=mode))
    assert result.mappings.idx2item[result.splits.eval_items[0]] == "after-midnight"


def test_full_timestamp_orders_offsets_by_utc_instant():
    records = [resolved(Kind.PURCHASE, item="latest", timestamp=BASE + timedelta(hours=21)),
               resolved(Kind.VIEW, item="earlier", timestamp=datetime(2026, 1, 1, 23, tzinfo=timezone(timedelta(hours=3))))]
    result = assert_parity(records, BprPreparationConfig(min_user_interactions_for_eval=2,
                                                       date_mode=DateMode.FULL_TIMESTAMP))
    assert result.mappings.idx2item[result.splits.eval_items[0]] == "latest"


@pytest.mark.parametrize("mode", list(DateMode))
@pytest.mark.parametrize("history,target", [("ABC", "C"), ("ABA", None), ("ABCB", None), ("AABAC", "C")])
def test_final_novel_target_only_and_non_eval_history_retention(mode, history, target):
    records = [resolved(item=item, timestamp=BASE + timedelta(hours=n)) for n, item in enumerate(history)]
    result = assert_parity(records, BprPreparationConfig(min_user_interactions_for_eval=3, date_mode=mode))
    assert result.splits.eval_users.tolist() == ([] if target is None else [0])
    if target is not None:
        assert result.mappings.idx2item[result.splits.eval_items[0]] == target
    retained = history if target is None else history[:-1]
    weights = {result.mappings.idx2item[item]: weight for (_, item), weight
               in zip(result.splits.train_pairs, result.splits.train_weights)}
    assert weights == pytest.approx({item: retained.count(item) * 0.1 for item in set(retained)})
    assert result.diagnostics.train_events_before_aggregation == len(retained)
    for user, item in zip(result.splits.eval_users, result.splits.eval_items):
        assert item not in result.splits.user_pos_train[user]


@pytest.mark.parametrize("threshold", [2, 3, 4])
def test_novel_final_item_still_requires_event_count_threshold(threshold):
    result = assert_parity([resolved(item=item, timestamp=BASE + timedelta(hours=n)) for n, item in enumerate("ABC")],
                           BprPreparationConfig(min_user_interactions_for_eval=threshold, date_mode=DateMode.FULL_TIMESTAMP))
    assert len(result.splits.eval_users) == int(threshold <= 3)
    assert result.diagnostics.train_events_before_aggregation == (2 if threshold <= 3 else 3)


def test_evaluation_eligibility_diagnostics_count_excluded_seen_final_items():
    records = [resolved(user=user, item=item, timestamp=BASE + timedelta(hours=n))
               for user, history in (("novel", "ABC"), ("repeat", "ABA"), ("leak", "ABCB"), ("short", "AB"))
               for n, item in enumerate(history)]
    result = assert_parity(records, BprPreparationConfig(min_user_interactions_for_eval=3,
                                                        date_mode=DateMode.FULL_TIMESTAMP))
    d = result.diagnostics
    assert d.interaction_count_eligible_users == 3
    assert d.eligible_eval_users == d.eval_events == 1
    assert d.excluded_seen_eval_users == 2
    assert d.excluded_seen_eval_rate == pytest.approx(2 / 3)
    assert d.train_events_before_aggregation == 11


def test_empty_eval_diagnostics_have_zero_exclusion_rate():
    result = prepare_bpr([])
    assert result.diagnostics.interaction_count_eligible_users == 0
    assert result.diagnostics.excluded_seen_eval_users == 0
    assert result.diagnostics.excluded_seen_eval_rate == 0
