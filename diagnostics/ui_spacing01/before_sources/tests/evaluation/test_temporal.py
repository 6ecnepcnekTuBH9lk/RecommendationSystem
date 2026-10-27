from dataclasses import FrozenInstanceError, replace
from datetime import timedelta, timezone

import numpy as np
import pytest

from Application.evaluation.ranking import evaluate_snapshot
from Application.evaluation.temporal import TemporalConfig, TemporalProtocolError, build_temporal_protocol
from Application.interactions import InteractionBuilder, InteractionType as Kind
from Application.mindbox.adapters.actions import adapt_action
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.records import CustomerMergeRecord
from Application.product_resolution import CatalogDiagnostics, ProductCatalog, ProductResolver


def test_two_snapshots_and_fixed_seen(protocol):
    validation, test = protocol.validation, protocol.test
    assert [(c.customer_id, c.target_item) for c in validation.cases] == [("u", "C")]
    assert [(c.customer_id, c.target_item) for c in test.cases] == [("u", "D")]
    assert validation.seen_at_cutoff["u"] == {"A", "B"}
    assert test.seen_at_cutoff["u"] == {"A", "B", "C"}
    assert len(validation.history) == 5
    assert len(test.history) == 6
    assert all(e.interaction.event_datetime_utc < validation.cutoff for e in validation.history)
    assert all(e.interaction.event_datetime_utc < test.cutoff for e in test.history)
    assert validation.diagnostics.mean_target_delay_seconds == 7200


@pytest.mark.parametrize("offset,validation_history,validation_future,test_history,test_future", [
    (-0.000001, 2, 0, 2, 0), (0, 1, 1, 2, 0), (0.000001, 1, 1, 2, 0),
    (10 - 0.000001, 1, 1, 2, 0), (10, 1, 0, 1, 1), (10 + 0.000001, 1, 0, 1, 1),
    (20 - 0.000001, 1, 0, 1, 1), (20, 1, 0, 1, 0), (20 + 0.000001, 1, 0, 1, 0),
])
def test_exact_boundaries(event, config, offset, validation_history, validation_future, test_history, test_future):
    boundary = event("u", "B", 10)
    boundary = replace(boundary, interaction=replace(boundary.interaction,
                       event_datetime_utc=config.validation_start + timedelta(seconds=offset * 3600)))
    result = build_temporal_protocol([event("u", "A", 1), boundary], replace(config, min_history_events=1))
    assert len(result.validation.history) == validation_history
    assert result.validation.diagnostics.future_events == validation_future
    assert len(result.test.history) == test_history
    assert result.test.diagnostics.future_events == test_future


def test_open_test_window(event, config):
    result = build_temporal_protocol([event("u", "A", 1), event("u", "B", 100)],
                                     replace(config, test_end=None, min_history_events=1))
    assert result.test.diagnostics.future_events == 1
    assert result.validation.diagnostics.future_events == 0


@pytest.mark.parametrize("future,expected", [("AABC", "B"), ("BBBC", "B"), ("AAAC", "C")])
def test_first_novel_ignores_repeats_and_later_events(event, config, future, expected):
    records = [event("u", "A", 1), event("other", "B", 1), event("other", "C", 2)]
    records += [event("u", item, 11 + index) for index, item in enumerate(future)]
    result = build_temporal_protocol(records, replace(config, min_history_events=1)).validation
    assert len(result.cases) == 1
    assert result.cases[0].target_item == expected
    assert result.cases[0].target_timestamp == records[3 + future.index(expected)].interaction.event_datetime_utc
    assert result.seen_at_cutoff["u"] == {"A"}


def test_population_diagnostics_and_no_cold_item_substitution(event, config):
    records = [event(user, "A", hour) for user in ("warm", "no_novel", "cold_item") for hour in (1, 2)]
    records += [event("sparse", "A", 1), event("witness", "B", 1)]
    records += [event("warm", "B", 11), event("no_novel", "A", 12), event("cold_item", "X", 13),
                event("cold_item", "B", 14), event("cold", "B", 11)]
    records += [event("sparse", "B", 11 + index / 100) for index in range(20)]
    snap = build_temporal_protocol(records, config).validation
    d = snap.diagnostics
    assert [(c.customer_id, c.target_item) for c in snap.cases] == [("warm", "B")]
    assert (d.training_events, d.training_users, d.training_items) == (8, 5, 2)
    assert (d.future_events, d.future_users) == (25, 5)
    assert (d.users_passing_history_threshold, d.eligible_warm_user_targets) == (3, 2)
    assert (d.excluded_cold_users, d.excluded_sparse_users, d.excluded_cold_item_targets,
            d.no_novel_target_users, d.evaluation_targets) == (1, 1, 1, 1, 1)
    assert d.future_users == (d.excluded_cold_users + d.excluded_sparse_users + d.excluded_cold_item_targets
                              + d.no_novel_target_users + d.evaluation_targets)
    assert d.history_event_count_distribution == ((1, 2), (2, 3))


def test_default_threshold_counts_only_past_events(event, config):
    assert TemporalConfig(config.validation_start, config.test_start).min_history_events == 10
    records = [event("u", "A", index / 2) for index in range(9)] + [event("witness", "B", 1)]
    records += [event("u", "B", 11 + index / 100) for index in range(20)]
    result = build_temporal_protocol(records, TemporalConfig(config.validation_start, config.test_start))
    assert result.validation.cases == ()
    assert result.validation.diagnostics.excluded_sparse_users == 1
    records.insert(0, event("u", "A", 0))
    assert len(build_temporal_protocol(records, TemporalConfig(config.validation_start, config.test_start)).validation.cases) == 1


@pytest.mark.parametrize("kind,field", [(Kind.VIEW, "view_targets"), (Kind.FAVORITE, "favorite_targets"),
                                        (Kind.PURCHASE, "purchase_targets")])
def test_target_interaction_type_diagnostics(event, config, kind, field):
    records = [event("u", "A", 1), event("other", "B", 2), event("u", "B", 12, kind)]
    snap = build_temporal_protocol(records, replace(config, min_history_events=1)).validation
    assert snap.cases[0].interaction_type is kind
    assert getattr(snap.diagnostics, field) == 1
    assert snap.diagnostics.view_targets + snap.diagnostics.favorite_targets + snap.diagnostics.purchase_targets == 1


def test_exact_ties_follow_canonical_type_and_source_order(event, config):
    past = [event("u", "A", 1)] + [event("other", item, 1) for item in "BCDE"]
    future = [event("u", "B", 12, Kind.VIEW), event("u", "C", 12, Kind.FAVORITE),
              event("u", "E", 12, Kind.PURCHASE), event("u", "D", 12, Kind.PURCHASE)]
    snap = build_temporal_protocol(past + future, replace(config, min_history_events=1)).validation
    assert snap.cases[0].target_item == "E"
    assert snap.cases[0].interaction_type is Kind.PURCHASE


def test_future_cannot_change_validation_history_or_seen(event, config):
    past = [event("u", "A", 1), event("u", "A", 2), event("witness", "B", 1)]
    validation = [event("u", "B", 11)]
    first = build_temporal_protocol(past + validation, config)
    second = build_temporal_protocol(past + validation + [event("future_user", "future_item", 20),
                                      event("u", "future_item", 22)], config)
    assert first.validation == second.validation
    assert first.test.history == second.test.history
    assert first.test.item_universe == second.test.item_universe
    assert first.test.seen_at_cutoff == second.test.seen_at_cutoff
    assert "future_item" not in second.test.item_universe
    assert "future_user" not in second.test.seen_at_cutoff


def test_all_validation_events_become_test_history(event, config):
    records = [event("u", "A", 1), event("u", "A", 2), event("witness", "B", 1),
               event("u", "A", 11), event("u", "B", 12), event("u", "B", 13), event("u", "C", 14)]
    snap = build_temporal_protocol(records, config)
    assert len(snap.test.history) == len(records)
    assert snap.test.seen_at_cutoff["u"] == {"A", "B", "C"}
    assert snap.validation.seen_at_cutoff["u"] == {"A"}


def test_offset_timestamps_normalize_to_utc_without_mutating_input(event, config):
    record = event("u", "A", 1)
    offset = timezone(timedelta(hours=3))
    record = replace(record, interaction=replace(record.interaction,
                     event_datetime_utc=record.interaction.event_datetime_utc.astimezone(offset)))
    result = build_temporal_protocol([record], replace(config, validation_start=config.validation_start.astimezone(offset)))
    assert result.validation.cutoff == config.validation_start
    assert result.validation.history[0].interaction.event_datetime_utc.tzinfo is timezone.utc
    assert record.interaction.event_datetime_utc.tzinfo is offset


@pytest.mark.parametrize("overrides", [{"min_history_events": 0}, {"min_history_events": True},
                                       {"min_history_events": 1.5}, {"validation_start": None}])
def test_invalid_config(config, overrides):
    with pytest.raises(TemporalProtocolError):
        replace(config, **overrides)


def test_invalid_cutoff_order_and_naive_times(config):
    for overrides in ({"test_start": config.validation_start}, {"test_end": config.test_start},
                      {"test_start": config.validation_start - timedelta(seconds=1)},
                      {"validation_start": config.validation_start.replace(tzinfo=None)}):
        with pytest.raises(TemporalProtocolError):
            replace(config, **overrides)


@pytest.mark.parametrize("field,value", [("customer_id", " PRIVATE "), ("event_datetime_utc", None),
                                        ("interaction_type", "PRIVATE")])
def test_invalid_events_are_rejected_without_exposing_values(event, config, field, value):
    record = event("PRIVATE", "PRIVATE", 1)
    record = replace(record, interaction=replace(record.interaction, **{field: value}))
    with pytest.raises(TemporalProtocolError) as caught:
        build_temporal_protocol([record], config)
    assert "PRIVATE" not in str(caught.value)


def test_raw_events_and_unresolved_ids_are_rejected(event, config):
    for record in ({"private": "value"}, replace(event("u", "A", 1), item_id="")):
        with pytest.raises(TemporalProtocolError):
            build_temporal_protocol([record], config)


def test_immutable_shared_cases(protocol):
    with pytest.raises(FrozenInstanceError):
        protocol.validation.cases[0].target_item = "different"
    with pytest.raises(TypeError):
        protocol.validation.seen_at_cutoff["u"] = frozenset()
    with pytest.raises(AttributeError):
        protocol.validation.seen_at_cutoff["u"].add("C")
    with pytest.raises(TemporalProtocolError):
        replace(protocol.validation, history=protocol.test.history)
    with pytest.raises(TemporalProtocolError):
        replace(protocol.validation, cases=protocol.test.cases)


class Scorer:
    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def score(self, user, candidates):
        self.calls.append((user, candidates))
        return [self.scores[item] for item in candidates]


@pytest.mark.parametrize("k,recall,ndcg", [(1, 0, 0), (2, 1, 1 / np.log2(3)), (100, 1, 1 / np.log2(3))])
def test_shared_scoring_and_metrics(protocol, k, recall, ndcg):
    scorer = Scorer({"C": 2, "D": 3, "E": 1})
    cases = protocol.validation.cases
    result = evaluate_snapshot(protocol.validation, scorer, k)
    assert (result.recall, result.ndcg, result.evaluated_users) == (recall, ndcg, 1)
    assert scorer.calls == [("u", ("C", "D", "E"))]
    assert protocol.validation.cases is cases


def test_multiple_backends_share_cases_and_deterministic_score_ties(protocol):
    cases = protocol.validation.cases
    for scorer in (Scorer({"C": 0, "D": 0, "E": 0}), Scorer({"C": 10, "D": 1, "E": 1})):
        assert evaluate_snapshot(protocol.validation, scorer, 1).recall == 1
        assert protocol.validation.cases is cases


@pytest.mark.parametrize("scores", [[0], [0, np.inf, 1], [np.nan, 1, 0], [[1, 2, 3]]])
def test_invalid_scorer_output(protocol, scores):
    class InvalidScorer:
        def score(self, user, candidates):
            return scores
    with pytest.raises(TemporalProtocolError):
        evaluate_snapshot(protocol.validation, InvalidScorer())


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_ranking_k(protocol, k):
    with pytest.raises(TemporalProtocolError):
        evaluate_snapshot(protocol.validation, Scorer({}), k)


def test_empty_benchmark(event, config):
    snap = build_temporal_protocol([], config).validation
    result = evaluate_snapshot(snap, Scorer({}))
    assert (result.evaluated_users, result.recall, result.ndcg) == (0, 0, 0)
    assert snap.diagnostics.mean_target_delay_seconds is None
    assert snap.diagnostics.history_event_count_distribution == ()


def test_canonical_customer_and_product_resolvers_are_reused(config):
    identity = CustomerIdResolver([CustomerMergeRecord(1, config.validation_start - timedelta(days=1), "300", ("100", "200"))])
    resolver = ProductResolver(ProductCatalog(frozenset({"000001", "000002"}), CatalogDiagnostics(unique_items=2)))
    builder = InteractionBuilder()
    resolved = []
    for user, product, hour in [(100, "000001_size1", 1), (200, "000001_size2", 2),
                                (400, "000002_size1", 1), (200, "000002_size2", 11)]:
        stamp = config.validation_start + timedelta(hours=hour - 10)
        raw = {"ids": {"mindboxId": 1}, "customer": {"ids": {"mindboxId": user}},
               "dateTimeUtc": stamp.isoformat(), "creationDateTimeUtc": stamp.isoformat(),
               "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}},
               "products": [{"ids": {"offline1C": product}}]}
        for interaction in builder.from_action(adapt_action(raw, identity)):
            resolved.append(resolver.resolve_interaction(interaction))
    snap = build_temporal_protocol(resolved, config).validation
    assert list(snap.seen_at_cutoff) == ["300", "400"]
    assert snap.seen_at_cutoff["300"] == {"000001"}
    assert [(case.customer_id, case.target_item) for case in snap.cases] == [("300", "000002")]
