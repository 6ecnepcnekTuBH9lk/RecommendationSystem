from dataclasses import asdict, replace
from decimal import Decimal
import json

import numpy as np
import pytest
import torch

from Application.evaluation.experiments import bpr_weights as exp
from Application.evaluation.temporal import TemporalProtocolError, build_temporal_protocol
from Application.interactions import InteractionType
from Application.model.bpr_preparation import BprWeightConfig


def tiny_config():
    return exp.core.TrainConfig(embedding_dim=4, epochs=2, batch_size=4, n_neg=2,
                                early_stop=False, use_item_features=False)


def fake_run(snapshot, training, cfg, temporal, weights, seed, device, plan):
    value = .2 + weights.view_weight / 100 + weights.favorite_weight / 10000 + (seed - 43) / 1000
    metrics = {label: {str(k): {"cases": len(snapshot.cases) if label == "overall" else
                              sum(case.interaction_type.value == label for case in snapshot.cases),
                              "recall": value, "ndcg": value / 2} for k in exp.KS}
               for label in ("overall", *(kind.value for kind in InteractionType))}
    return {"seed": seed, "weights": asdict(weights), "metrics": metrics}


def test_frozen_config_existing_epochs_and_required_flags(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"epochs": 200, "lr": .0003, "seed": 42,
                                   "early_stop": True, "use_item_features": True, "filter_summary": "private"}))
    cfg = exp.frozen_config(settings)
    assert cfg.epochs == 200 and cfg.lr == .0003 and cfg.seed == 42
    assert not cfg.early_stop and not cfg.use_item_features
    plan = exp.freeze_plan(cfg, exp.benchmark_config(), "cpu", {"working_tree_dirty": True})
    assert plan["seeds"] == [42, 43, 44]
    assert plan["std_ddof"] == 1 and "NDCG@10" in plan["primary_metric"]
    assert plan["provenance"]["working_tree_dirty"] is True
    assert "private" not in json.dumps(plan)


def test_test_snapshot_rejected_before_preparation(protocol, config, tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "prepare_bpr_snapshot", lambda *args: pytest.fail("test preparation"))
    with pytest.raises(TemporalProtocolError, match="only.*validation"):
        exp.run_experiment(protocol.test, config, tiny_config(), "cpu", {}, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["early_stop", "use_item_features"])
def test_invalid_flags_rejected(field, config):
    with pytest.raises(ValueError):
        exp.freeze_plan(replace(tiny_config(), **{field: True}), config, "cpu", {})


def test_whole_history_and_supplied_weights_reach_adapter(protocol, config, tmp_path, monkeypatch):
    original = exp.prepare_bpr_snapshot
    preparations, calls = [], []
    def prepare(snapshot, weights):
        assert snapshot is protocol.validation
        data = original(snapshot, weights)
        assert data.training.diagnostics.train_events_before_aggregation == len(snapshot.history)
        assert len(data.training.splits.eval_users) == 0
        preparations.append(weights)
        return data
    def run(*args):
        snapshot, training, cfg, temporal, weights, seed, device, plan = args
        assert snapshot is protocol.validation and temporal == config
        calls.append((asdict(cfg), seed, exp.mapping_fingerprint(training)))
        return fake_run(*args)
    monkeypatch.setattr(exp, "prepare_bpr_snapshot", prepare)
    cfg = tiny_config()
    report = exp.run_experiment(protocol.validation, config, cfg, "cpu",
                                exp.freeze_plan(cfg, config, "cpu", {}), tmp_path,
                                run=run, progress=lambda _: None)
    assert len(preparations) == 10 and len(calls) == 30
    assert [w.view_weight for w in preparations[:6]] == list(exp.VIEW_GRID)
    assert len({json.dumps(c[0], sort_keys=True) for c in calls}) == 1
    assert len({c[2] for c in calls}) == 1
    assert [c[1] for c in calls] == list(exp.SEEDS) * 10
    assert report["stage_b"][2]["reused_from_stage_a"]
    assert report["status"] == "complete" and not report["test_performance_computed"]
    content = (tmp_path / "results.json").read_text()
    assert '"customer_id"' not in content and '"target_item"' not in content
    assert '"u"' not in content and '"other"' not in content


def test_existing_quantity_mass_and_repeat_sum(event, config):
    protocol = build_temporal_protocol([
        event("u", "A", 1, InteractionType.PURCHASE, Decimal("20")),
        event("u", "A", 2, InteractionType.PURCHASE, Decimal("0")),
        event("u", "B", 3), event("other", "C", 1), event("other", "D", 2),
        event("u", "C", 12),
    ], config)
    units = exp.confidence_units(protocol.validation)
    assert units["PURCHASE"] == {"events": 2, "confidence_units": 11}
    weights = BprWeightConfig(view_weight=.5)
    training = exp.prepare_bpr_snapshot(protocol.validation, weights).training
    mass = exp.mass_diagnostics(units, weights, training)
    assert mass["types"]["PURCHASE"]["weight_mass"] == 110
    assert mass["aggregated_pairs"] == 4
    assert mass["aggregated_weight_sum"] == 111.5


def test_mean_sample_std(protocol, config):
    data = exp.prepare_bpr_snapshot(protocol.validation).training
    runs = [fake_run(protocol.validation, data, tiny_config(), config, BprWeightConfig(), seed, "cpu", {})
            for seed in exp.SEEDS]
    result = exp.aggregate(runs)["metrics"]["overall"]["10"]["ndcg"]
    values = [r["metrics"]["overall"]["10"]["ndcg"] for r in runs]
    assert result["mean"] == np.mean(values)
    assert result["std"] == np.std(values, ddof=1)
    with pytest.raises(ValueError):
        exp.aggregate(runs[:2])
    with pytest.raises(ValueError):
        exp.aggregate([runs[0]] * 3)


def test_conservative_tie_full_precision_selection():
    configs = [{"weights": {"view_weight": v}, "metrics": {"overall": {"10": {
        "ndcg": {"mean": mean, "std": std}}}}} for v, mean, std in (
            (.1, .20000001, .002), (2, .2002, .002), (.05, .19, .002))]
    chosen = exp.select_representative(configs, "view_weight", .1)
    assert chosen["formal_maximum"]["view_weight"] == 2
    assert chosen["representative"]["view_weight"] == .1
    assert not chosen["clear_winner"]


def test_seed_reproducibility_real_isolated_training_no_publication(protocol, config, tmp_path):
    cfg = tiny_config()
    plan = exp.freeze_plan(cfg, config, "cpu", {"working_tree_dirty": True, "git_head": "synthetic"})
    training = exp.prepare_bpr_snapshot(protocol.validation).training
    first = exp.isolated_run(protocol.validation, training, cfg, config, BprWeightConfig(), 42, "cpu", plan)
    second = exp.isolated_run(protocol.validation, training, cfg, config, BprWeightConfig(), 42, "cpu", plan)
    assert first["metrics"] == second["metrics"]
    assert first["epochs_completed"] == cfg.epochs
    assert first["provenance"] == plan["provenance"]
    assert not first["publication_executed"] and not first["test_performance_computed"]
    assert not list(tmp_path.iterdir())


def test_source_provenance_dirty_and_digests(tmp_path, monkeypatch):
    (tmp_path / "Application").mkdir()
    (tmp_path / "Application/a.py").write_text("x = 1\n")
    answers = iter(["feature/training-settings", "head", " M file"])
    monkeypatch.setattr(exp.subprocess, "check_output", lambda *a, **k: next(answers))
    result = exp.git_provenance(tmp_path)
    assert result["working_tree_dirty"] is True
    assert len(result["source_sha256"]["Application/a.py"]) == 64


def test_cli_protects_production_and_settings(tmp_path):
    from scripts.run_bpr_weight_experiment import parse_args, PROJECT_ROOT
    for directory in ("model", "user_settings", "input_data"):
        with pytest.raises(SystemExit):
            parse_args(["--output-dir", str(PROJECT_ROOT / directory / "research")])
    assert parse_args(["--output-dir", str(tmp_path / "research")]).output_dir == tmp_path / "research"


def test_single_baseline_exactly_one_seed_no_sweep(protocol, config, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(exp, "run_experiment", lambda *a, **k: pytest.fail("sweep executed"))
    monkeypatch.setattr(exp, "select_representative", lambda *a, **k: pytest.fail("selection executed"))
    monkeypatch.setattr(exp, "aggregate", lambda *a, **k: pytest.fail("aggregation executed"))
    def run(snapshot, training, cfg, temporal, weights, seed, device, plan):
        calls.append((weights, seed, asdict(cfg)))
        persisted = json.loads((tmp_path / "frozen_config.json").read_text())
        assert persisted["seeds"] == [42] and persisted["planned_training_runs"] == 1
        value = fake_run(snapshot, training, cfg, temporal, weights, seed, device, plan)
        return {**value, "training_seconds": 100., "total_seconds": 110., "device": device,
                "training_users": len(training.mappings.idx2user), "training_items": len(training.mappings.idx2item),
                "training_pairs": len(training.splits.train_pairs), "training_memory": {"available": False}}
    cfg = tiny_config()
    report = exp.run_single_baseline(protocol.validation, config, cfg, "cpu",
                                     exp.freeze_plan(cfg, config, "cpu", {}), tmp_path,
                                     run=run, progress=lambda _: None)
    assert calls == [(BprWeightConfig(), 42, asdict(cfg))]
    assert report["status"] == "complete" and not report["test_performance_computed"]
    assert report["projections"]["30"]["training_seconds"] == 3000
    assert not report["plan"]["selection_performed"]
    assert "No second seed" in (tmp_path / "summary.md").read_text()
    with pytest.raises(ValueError, match="already exists"):
        exp.run_single_baseline(protocol.validation, config, cfg, "cpu", {}, tmp_path)


def test_single_baseline_rejects_test_snapshot(protocol, config, tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "prepare_bpr_snapshot", lambda *a: pytest.fail("test preparation"))
    with pytest.raises(TemporalProtocolError):
        exp.run_single_baseline(protocol.test, config, tiny_config(), "cpu", {}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_cli_default_single_and_explicit_sweep(tmp_path, monkeypatch):
    from scripts import run_bpr_weight_experiment as cli
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    default = cli.parse_args([])
    assert default.mode == "baseline" and default.output_dir.name == "bpr_exp01_baseline_cost"
    sweep = cli.parse_args(["--mode", "sweep"])
    assert sweep.mode == "sweep" and sweep.output_dir.name == "bpr_exp01_weights"
    default.output_dir.mkdir(parents=True)
    (default.output_dir / "results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.parse_args([])


def test_cost_projection_and_memory_scope():
    assert exp.cost_projections(10., 12.)["18"] == {
        "training_seconds": 180., "training_and_validation_seconds": 216.}
    counters = exp.process_memory()
    assert counters["scope"] == "this process only"
    if "peak_working_set_bytes" in counters:
        assert counters["peak_working_set_bytes"] >= counters["current_working_set_bytes"] > 0


def test_runner_external_validation_receives_final_epoch(protocol, config, monkeypatch):
    cfg = replace(tiny_config(), epochs=4, lr=.03)
    training = exp.prepare_bpr_snapshot(protocol.validation).training
    assert len(training.splits.eval_users) == 0
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *args, **kwargs: self.clone())
    epochs, evaluations = [], []
    internal_eval = exp.core._eval_bprmf_recall_ndcg
    external_eval = exp.evaluate_bpr_snapshot

    def state(model):
        return {name: value.detach().clone() for name, value in model.state_dict().items()}

    def capture(model, *args):
        epochs.append(state(model))
        return internal_eval(model, *args)

    def validate(model, *args):
        assert len(epochs) == cfg.epochs
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, epochs[-1][name], rtol=0, atol=0)
        evaluations.append(args[1])
        return external_eval(model, *args)

    monkeypatch.setattr(exp.core, "_eval_bprmf_recall_ndcg", capture)
    monkeypatch.setattr(exp, "evaluate_bpr_snapshot", validate)
    plan = exp.freeze_plan(cfg, config, "cpu", {})
    result = exp.run_seed(protocol.validation, training, cfg, config, BprWeightConfig(), 42, "cpu", plan)
    assert not torch.equal(epochs[0]["user_emb.weight"], epochs[-1]["user_emb.weight"])
    assert len(evaluations) == 12
    assert result["trainer_best_epoch"] == -1 and result["epochs_completed"] == 4
    assert not result["test_performance_computed"] and not result["publication_executed"]


def test_interrupted_cost_pilot_summary_is_explicit():
    report = {"status": "interrupted_cost_pilot", "plan": {}, "run": None,
              "requested_epochs": 200, "completed_epochs": 3}
    summary = exp.single_summary(report)
    assert "Status: interrupted_cost_pilot" in summary
    assert "Requested epochs: 200; completed epochs: 3" in summary
    assert "No validation metrics were computed" in summary
    assert "must not be used for model-quality comparison" in summary
    assert "| Slice |" not in summary and "| Runs |" not in summary
