from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from Application.model.training_data import Mappings, Splits, PreparedBprData
from Application.model.training_quality import (
    QualityLevel, TrainingQualityConfig, TrainingQualityDiagnostics, evaluate_training_quality,
)


@pytest.fixture
def prepared():
    return PreparedBprData(
        Mappings({"SECRET_CUSTOMER": 0}, ["SECRET_CUSTOMER"], {"SECRET_ITEM": 0}, ["SECRET_ITEM"]),
        Splits(np.array([[0, 0]]), np.array([10.]), np.array([], dtype=int), np.array([], dtype=int), [{0}]))


def test_clean_and_unmapped_not_quality_issue(prepared):
    for unmapped in (0, 127092):
        report = evaluate_training_quality(prepared, TrainingQualityDiagnostics(unmapped_actions=unmapped))
        assert report.level is QualityLevel.PASS
        assert report.training_allowed
        assert report.issues == ()
        assert report.metrics["malformed_rate"] == 0


@pytest.mark.parametrize("views,favorites,breakdown", [
    (10, 0, {"ProsmotrProdukta": 1}),
    (0, 10, {"DobavlenieProduktaVSpisokVOperaciiDobavlenieTovara": 1}),
    (8, 2, {"ProsmotrProdukta": 1, "ProsmotrProduktaVApiMethod": 2,
            "DobavlenieProduktaVSpisokVOperaciiDobavlenieTovara": 1}),
])
def test_malformed_warning_and_denominator(prepared, views, favorites, breakdown):
    count = sum(breakdown.values())
    report = evaluate_training_quality(prepared, TrainingQualityDiagnostics(
        actions_view=views, actions_favorite=favorites, malformed_mapped_actions=count,
        malformed_action_system_names=breakdown, unmapped_actions=100000, bpr_events=999))
    assert report.level is QualityLevel.WARN
    assert report.training_allowed
    assert report.metrics["mapped_actions"] == 10
    assert report.metrics["malformed_rate"] == count / 10
    issue, = report.issues
    assert issue.code == "MAPPED_ACTION_WITHOUT_PRODUCT"
    assert issue.breakdown == breakdown


def test_even_all_mapped_malformed_is_not_rate_block(prepared):
    report = evaluate_training_quality(prepared, TrainingQualityDiagnostics(actions_view=10, malformed_mapped_actions=10))
    assert report.level is QualityLevel.WARN and report.training_allowed


@pytest.mark.parametrize("kwargs,code", [({"unresolved_products": 1}, "UNRESOLVED_PRODUCT"),
                                        ({"unsupported_products": 1}, "UNSUPPORTED_PRODUCT_NAMESPACE")])
def test_identity_loss_blocks_even_valid_prepared(prepared, kwargs, code):
    report = evaluate_training_quality(prepared, TrainingQualityDiagnostics(**kwargs))
    assert report.level is QualityLevel.BLOCK
    assert not report.training_allowed
    assert report.issues[0].code == code


@pytest.mark.parametrize("problem", ["none", "empty_pairs", "no_users", "no_items", "nan", "inf", "negative", "bad_index"])
def test_invalid_prepared_block(prepared, problem):
    if problem == "none":
        prepared = None
    elif problem == "empty_pairs":
        prepared.splits = replace(prepared.splits, train_pairs=np.empty((0, 2), dtype=int),
                                   train_weights=np.array([]), user_pos_train=[set()])
    elif problem == "no_users":
        prepared.mappings = replace(prepared.mappings, user2idx={}, idx2user=[])
    elif problem == "no_items":
        prepared.mappings = replace(prepared.mappings, item2idx={}, idx2item=[])
    elif problem == "bad_index":
        prepared.splits.train_pairs[0, 1] = 3
    else:
        prepared.splits.train_weights[0] = {"nan": np.nan, "inf": np.inf, "negative": -1}[problem]
    report = evaluate_training_quality(prepared, TrainingQualityDiagnostics())
    assert report.level is QualityLevel.BLOCK
    assert not report.training_allowed
    assert report.issues[0].code == "INVALID_PREPARED_DATA"
    assert report.metrics["users"] is None
    assert "SECRET" not in repr(report)


def test_report_issues_and_input_snapshot_immutable(prepared):
    breakdown = {"ProsmotrProdukta": 1}
    diagnostics = TrainingQualityDiagnostics(actions_view=2, malformed_mapped_actions=1,
                                              malformed_action_system_names=breakdown)
    report = evaluate_training_quality(prepared, diagnostics)
    breakdown.clear()
    assert report.issues[0].breakdown == {"ProsmotrProdukta": 1}
    with pytest.raises(FrozenInstanceError):
        report.level = QualityLevel.PASS
    with pytest.raises(FrozenInstanceError):
        report.issues[0].count = 0
    with pytest.raises(TypeError):
        report.metrics["users"] = 9
    with pytest.raises(TypeError):
        report.issues[0].breakdown["ProsmotrProdukta"] = 99
    assert "SECRET" not in repr(report)


@pytest.mark.parametrize("rate", [-1, 2, float("nan"), float("inf"), True])
def test_invalid_policy(rate):
    with pytest.raises(ValueError):
        TrainingQualityConfig(rate)


def test_invalid_diagnostics_rejected(prepared):
    with pytest.raises(ValueError):
        TrainingQualityDiagnostics(malformed_mapped_actions=1)
    with pytest.raises(ValueError):
        TrainingQualityDiagnostics(actions_view=-1)
    with pytest.raises(ValueError):
        TrainingQualityDiagnostics(actions_view=2, malformed_mapped_actions=1,
                                  malformed_action_system_names={"ProsmotrProdukta": 2})
