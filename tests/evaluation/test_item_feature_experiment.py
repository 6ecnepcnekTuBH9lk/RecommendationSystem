from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import hashlib

import pytest
import torch

from Application.evaluation.bpr import prepare_bpr_snapshot
from Application.evaluation.experiments import item_features as experiment
from Application.evaluation.ranking import evaluate_snapshot
from Application.model import BPRMF as core
from scripts.run_bpr_feature_ab import main, synthetic_protocol


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # The real research runner uses fresh processes; unit tests share this one.
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(core, "_save_artifacts", lambda *a, **k: pytest.fail("research must not publish"))
    protocol, source = synthetic_protocol(tmp_path / "catalog")
    cfg = core.TrainConfig(epochs=2, embedding_dim=4, batch_size=128, n_neg=2, early_stop=False)
    data = prepare_bpr_snapshot(protocol.validation).training
    return protocol, source, cfg, data


def run(setup, arm, seed=42):
    protocol, source, cfg, data = setup
    return experiment._run_arm(protocol.validation, data, cfg, source.path.read_bytes(), arm, seed)


def test_common_initialization_sampler_and_real_feature_participation(setup):
    a, b = run(setup, "A"), run(setup, "B")
    experiment.check_pair(a, b)
    assert a["initial_hashes"]["users"] == b["initial_hashes"]["users"]
    assert a["initial_hashes"]["items"] == b["initial_hashes"]["items"]
    assert a["sampler_sha256"] == b["sampler_sha256"]
    assert not a["features_participate"] and a["feature_diagnostics"]["unique_tokens"] == 0
    assert b["features_participate"] and b["feature_weights_changed"]
    assert b["feature_diagnostics"]["unique_tokens"] > 0
    users, items = experiment.common_initial_weights(64, 64, 4, 42)
    assert a["initial_hashes"]["users"] == experiment.tensor_hash(users)
    assert a["initial_hashes"]["items"] == experiment.tensor_hash(items)
    assert a["candidate_item_coverage"] == b["candidate_item_coverage"] == 1
    assert a["evaluated_users"] == b["targets"] == 64
    assert a["hitrate10"] == a["recall10"]
    assert a["epochs"] == b["epochs"] == 2


def test_reproducible_results_without_timing_and_input_mutation(setup):
    protocol, _, _, data = setup
    before = experiment.fingerprints(protocol.validation, data)
    first, second = run(setup, "B"), run(setup, "B")
    first.pop("training_seconds")
    second.pop("training_seconds")
    assert first == second
    assert experiment.fingerprints(protocol.validation, data) == before
    other = run(setup, "B", 43)
    assert other["sampler_sha256"] != first["sampler_sha256"]
    assert other["initial_hashes"]["items"] != first["initial_hashes"]["items"]


@pytest.mark.parametrize("problem", ["missing", "future", "checksum", "unreviewed", "wrong_basis", "naive"])
def test_strict_rejects_unproven_catalog_before_training(setup, monkeypatch, problem):
    protocol, source, cfg, _ = setup
    evidence = source.evidence
    if problem == "missing":
        evidence = None
    elif problem == "future":
        evidence = replace(evidence, available_at=protocol.validation.cutoff + timedelta(seconds=1))
    elif problem == "checksum":
        evidence = replace(evidence, sha256="0" * 64)
    elif problem == "unreviewed":
        evidence = replace(evidence, reviewed_by="")
    elif problem == "wrong_basis":
        evidence = replace(evidence, basis="FILE_MTIME")
    else:
        evidence = replace(evidence, available_at=evidence.available_at.replace(tzinfo=None))
    bad = replace(source, evidence=evidence)
    monkeypatch.setattr(experiment, "isolated_arm", lambda *a: pytest.fail("gate must precede training"))
    with pytest.raises(experiment.FeatureExperimentError):
        experiment.run_experiment(protocol, cfg, bad, source, dataset_kind="SYNTHETIC")


def test_strict_checks_test_catalog_before_any_validation_training(setup, monkeypatch):
    protocol, source, cfg, _ = setup
    bad_test = replace(source, evidence=None)
    monkeypatch.setattr(experiment, "isolated_arm", lambda *a: pytest.fail("must preflight both catalogs"))
    with pytest.raises(experiment.FeatureExperimentError, match="STRICT blocked"):
        experiment.run_experiment(protocol, cfg, source, bad_test, dataset_kind="SYNTHETIC")


def test_exploratory_explicitly_marks_metadata_leakage_risk(setup):
    protocol, source, _, _ = setup
    unproven = replace(source, evidence=None)
    content, report = unproven.freeze(protocol.validation.cutoff, "EXPLORATORY", "REAL_BOUNDED")
    assert report["metadata_leakage_risk"] and not report["as_of_confirmed"]
    assert hashlib.sha256(content).hexdigest() == report["sha256"]
    with pytest.raises(experiment.FeatureExperimentError, match="STRICT"):
        unproven.freeze(protocol.validation.cutoff, "STRICT", "REAL_BOUNDED")
    with pytest.raises(experiment.FeatureExperimentError):
        source.freeze(protocol.validation.cutoff, "STRICT", "REAL_BOUNDED")


def test_exploratory_run_keeps_warning_in_every_result(setup, monkeypatch):
    protocol, source, cfg, _ = setup
    monkeypatch.setattr(experiment, "isolated_arm", experiment._run_arm)
    source = replace(source, evidence=None)
    result = experiment.run_experiment(protocol, cfg, source, source, mode="EXPLORATORY",
                                        dataset_kind="SYNTHETIC", seeds=(42,))
    assert result["plan"]["mode"] == "EXPLORATORY"
    assert all(run["metadata_leakage_risk"] for run in result["runs"])
    assert all(not source["as_of_confirmed"] for source in result["plan"]["catalogs"].values())


def test_existing_ranking_metrics_observer_and_seen_exclusion(setup):
    protocol, _, _, _ = setup
    snapshot = protocol.validation

    class Scorer:
        def score(self, user, candidates):
            case = next(case for case in snapshot.cases if case.customer_id == user)
            assert not set(candidates) & case.seen_items
            assert len(candidates) > 10
            return [1. if item == case.target_item else 0. for item in candidates]

    observed = []
    metrics = evaluate_snapshot(snapshot, Scorer(), 10, case_observer=lambda case, values: observed.append(values))
    assert metrics.recall == metrics.ndcg == 1.
    assert len(observed) == 64 and all(values == (1., 1.) for values in observed)
    assert all(event.interaction.event_datetime_utc < snapshot.cutoff for event in snapshot.history)
    assert all(snapshot.cutoff <= case.target_timestamp < protocol.test.cutoff for case in snapshot.cases)
    assert all(event.interaction.event_datetime_utc < protocol.test.cutoff for event in protocol.test.history)


def test_plan_frozen_before_validation_and_test_and_original_config_isolated(setup, monkeypatch):
    protocol, source, cfg, _ = setup
    phases = []
    original = experiment._run_arm
    captured = {}

    def inline(snapshot, training, current, content, arm, seed):
        assert captured and phases != ["test"]
        assert current.lr == cfg_initial_lr
        return original(snapshot, training, current, content, arm, seed)

    def on_plan(plan):
        captured.update(deepcopy(plan))
        cfg.item_feature_cols.clear()

    def on_phase(phase, *_):
        phases.append(phase)
        cfg.lr = 99

    cfg_initial_lr = cfg.lr
    monkeypatch.setattr(experiment, "isolated_arm", inline)
    result = experiment.run_experiment(protocol, cfg, source, source, seeds=(42, 43), dataset_kind="SYNTHETIC",
                                        on_plan=on_plan, on_phase=on_phase)
    assert phases == ["validation", "test"]
    assert len(result["runs"]) == 8
    assert result["plan_sha256"] == captured["sha256"]
    assert result["plan"]["parameters"]["item_feature_cols"]
    assert all(r["plan_sha256"] == result["plan_sha256"] for r in result["runs"])
    assert result["plan"]["test_tuning"] is False


def test_paired_bootstrap_known_delta_and_small_sample_guard():
    rows = []
    for seed in (42, 43, 44):
        for arm in ("A", "B"):
            value = float(arm == "B")
            rows.append({"arm": arm, "seed": seed, "contributions": [(value, value)] * 40,
                         "recall10": value, "hitrate10": value, "ndcg10": value, "training_seconds": 1.,
                         "fingerprints": {}, "sampler_sha256": str(seed), "batches": 1,
                         "evaluated_users": 40, "targets": 40, "candidate_items": 64,
                         "initial_hashes": {"users": str(seed), "items": str(seed)}})
    summary = experiment.paired_summary(rows, bootstrap_samples=100)
    assert summary["delta_B_minus_A"]["recall10"] == 1.
    assert summary["uncertainty"]["interval95"]["ndcg10"] == [1., 1.]
    for row in rows:
        row["contributions"] = row["contributions"][:10]
    assert experiment.paired_summary(rows)["uncertainty"]["interval95"] is None


def test_pair_mismatch_is_an_error(setup):
    a, b = run(setup, "A"), run(setup, "B")
    b["initial_hashes"]["users"] = "changed"
    with pytest.raises(experiment.FeatureExperimentError, match="initialization"):
        experiment.check_pair(a, b)


@pytest.mark.parametrize("epochs,seeds", [(51, (42,)), (2, (42, 42)), (2, ())])
def test_controlled_budget_and_predeclared_seeds(setup, epochs, seeds):
    protocol, source, cfg, _ = setup
    with pytest.raises(experiment.FeatureExperimentError):
        experiment.run_experiment(protocol, replace(cfg, epochs=epochs), source, source,
                                  seeds=seeds, dataset_kind="SYNTHETIC")


def test_batch_observer_cannot_change_training_samples(setup):
    _, source, cfg, data = setup
    cfg = replace(cfg, data_dir=str(source.path.parent), use_item_features=False)
    torch.manual_seed(42)
    first, _, _ = core.train_prepared_data_with_metrics(cfg, data, torch.device("cpu"))

    def corrupt_copies(*arrays):
        for array in arrays:
            array.fill(0)

    torch.manual_seed(42)
    second, _, _ = core.train_prepared_data_with_metrics(cfg, data, torch.device("cpu"), batch_observer=corrupt_copies)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


def test_output_protection_rejects_production_model_directory():
    from scripts.run_bpr_feature_ab import ROOT
    with pytest.raises(SystemExit):
        main(["--synthetic", "--output", str(ROOT / "model")])
