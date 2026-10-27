"""Controlled validation-only BPR weight sweep with isolated training processes."""

from collections import Counter
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from Application.evaluation.bpr import BprTemporalData, evaluate_bpr_snapshot, prepare_bpr_snapshot
from Application.evaluation.temporal import TemporalConfig, TemporalProtocolError, TemporalSnapshot
from Application.interactions import InteractionType
from Application.model import BPRMF as core
from Application.model.bpr_preparation import BprWeightConfig, to_bpr_event

EXPERIMENT = "BPR-EXP-01"
SEEDS = (42, 43, 44)
VIEW_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
FAVORITE_GRID = (0.5, 1.0, 2.0, 5.0, 10.0)
KS = (5, 10, 20)
SELECTION_POLICY = (
    "Maximize full-precision mean overall validation NDCG@10. Treat a configuration "
    "as tied when its gap to the maximum is <= max(sample std of both configurations). "
    "Among tied configurations choose the swept weight closest to the historical "
    "baseline in absolute log ratio; then prefer the smaller weight. This is a "
    "conservative descriptive rule, not a statistical significance test."
)


def benchmark_config():
    return TemporalConfig(*(datetime.fromisoformat(s) for s in (
        "2025-11-01T00:00:00+00:00", "2025-12-01T00:00:00+00:00",
        "2025-12-31T00:00:00+00:00")), min_history_events=10)


def temporal_dict(config):
    return {key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in asdict(config).items()}


def frozen_config(settings_path):
    """Use existing settings/defaults; only the two explicitly required flags differ."""
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8-sig"))
    names = {field.name for field in fields(core.TrainConfig)}
    cfg = core.TrainConfig(**{key: value for key, value in settings.items() if key in names})
    cfg.early_stop = False
    cfg.use_item_features = False
    return cfg


def git_provenance(root):
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={Path(root).as_posix()}", *args],
            cwd=root, text=True, encoding="utf-8").strip()
    sources = {}
    for directory in ("Application", "scripts"):
        for path in sorted((Path(root) / directory).rglob("*.py")):
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"git_branch": git("branch", "--show-current"), "git_head": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git("status", "--porcelain")), "source_sha256": sources}


def freeze_plan(cfg, temporal, device, provenance):
    if cfg.early_stop or cfg.use_item_features:
        raise ValueError("Research requires early stopping and item features disabled")
    return {"experiment": EXPERIMENT, "frozen_at": datetime.now(timezone.utc).isoformat(),
            "temporal_config": temporal_dict(temporal), "hyperparameters": asdict(cfg),
            "optimizer": "torch.optim.Adam", "device": str(device), "seeds": list(SEEDS),
            "primary_metric": "overall.validation.NDCG@10.mean", "std_ddof": 1,
            "selection_policy": SELECTION_POLICY, "stage_a_view_grid": list(VIEW_GRID),
            "stage_b_favorite_grid": list(FAVORITE_GRID), "test_performance_computed": False,
            "preparation": "TRAIN-02E prepare_bpr_snapshot; FULL_TIMESTAMP; whole history; no internal holdout",
            "provenance": provenance}


def require_validation(snapshot, config):
    if snapshot.cutoff != config.validation_start or snapshot.future_end != config.test_start:
        raise TemporalProtocolError("Experiment accepts only the configured validation snapshot")


def confidence_units(snapshot):
    """Exact existing quantity coercion/clip, no alternative confidence transformation."""
    counts = Counter()
    masses = {kind.value: 0.0 for kind in InteractionType}
    base = BprWeightConfig()
    for event in snapshot.history:
        kind = event.interaction.interaction_type
        counts[kind.value] += 1
        masses[kind.value] += to_bpr_event(event, base).weight
    multipliers = {"VIEW": base.view_weight, "FAVORITE": base.favorite_weight,
                   "PURCHASE": base.purchase_weight}
    return {kind: {"events": counts[kind], "confidence_units": masses[kind] / multipliers[kind]}
            for kind in masses}


def mass_diagnostics(units, weights, training):
    factors = {"VIEW": weights.view_weight, "FAVORITE": weights.favorite_weight,
               "PURCHASE": weights.purchase_weight}
    masses = {kind: value["confidence_units"] * factors[kind] for kind, value in units.items()}
    total = math.fsum(masses.values())
    values = training.splits.train_weights
    return {"types": {kind: {**units[kind], "weight_mass": mass, "share": mass / total}
                      for kind, mass in masses.items()}, "total_event_weight_mass": total,
            "aggregated_pairs": len(values), "aggregated_weight_sum": float(values.sum()),
            "aggregated_weight_mean": float(values.mean()),
            "aggregated_weight_quantiles": dict(zip(
                ("min", "p25", "median", "p75", "p90", "p99", "max"),
                map(float, np.quantile(values, (0, .25, .5, .75, .9, .99, 1)))))}


def mapping_fingerprint(training):
    """Hash identifiers rather than writing identifiers to experiment results."""
    digest = hashlib.sha256()
    maps = training.mappings
    for group in (maps.idx2user, maps.idx2item):
        for value in group:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    digest.update(training.splits.train_pairs.tobytes())
    return digest.hexdigest()


def snapshot_payload(snapshot):
    # MappingProxyType is not picklable. Only validation travels to the child.
    return {field.name: dict(snapshot.seen_at_cutoff) if field.name == "seen_at_cutoff"
            else getattr(snapshot, field.name) for field in fields(TemporalSnapshot)}


def process_memory():
    """OS process counters only; no sampling thread or dependency."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *((name, ctypes.c_size_t) for name in (
                            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                            "PagefileUsage", "PeakPagefileUsage"))]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {"scope": "this process only", "current_working_set_bytes": counters.WorkingSetSize,
                    "peak_working_set_bytes": counters.PeakWorkingSetSize,
                    "peak_commit_bytes": counters.PeakPagefileUsage}
    return {"scope": "this process only", "available": False}


def run_seed(snapshot, training, cfg, temporal, weights, seed, device, plan):
    """One in-memory model; no save/publish API is invoked. Called in a fresh process."""
    require_validation(snapshot, temporal)
    if cfg.early_stop or cfg.use_item_features or len(training.splits.eval_users):
        raise ValueError("Invalid controlled training configuration")
    current = replace(cfg, seed=seed, w_view_item=weights.view_weight,
                      w_favorite=weights.favorite_weight, w_purchase=weights.purchase_weight)
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.monotonic()
    core._set_seed(seed)
    model, _, observed = core.train_prepared_data_with_metrics(current, training, torch.device(device))
    if observed.early_stopped or observed.epochs_completed != current.epochs:
        raise ValueError("Training did not complete the frozen epoch budget")
    training_seconds = time.monotonic() - start_clock
    training_memory = process_memory()
    metrics = {}
    for label, kind in (("overall", None), *( (k.value, k) for k in InteractionType)):
        selected = snapshot if kind is None else replace(
            snapshot, cases=tuple(case for case in snapshot.cases if case.interaction_type is kind))
        metrics[label] = {}
        for k in KS:
            score = evaluate_bpr_snapshot(model, BprTemporalData(selected, training), k, torch.device(device))
            metrics[label][str(k)] = {"cases": score.evaluated_users, "recall": score.recall, "ndcg": score.ndcg}
    return {"experiment": EXPERIMENT, "started_at": started, "seed": seed,
            "weights": asdict(weights), "hyperparameters": asdict(current),
            "optimizer": plan["optimizer"], "device": str(device),
            "temporal_config": temporal_dict(temporal), "provenance": plan["provenance"],
            "training_events": len(snapshot.history), "training_users": len(training.mappings.idx2user),
            "training_items": len(training.mappings.idx2item), "mapping_sha256": mapping_fingerprint(training),
            "training_pairs": len(training.splits.train_pairs), "training_memory": training_memory,
            "epochs_completed": observed.epochs_completed, "trainer_best_epoch": observed.best_epoch,
            "training_seconds": training_seconds, "total_seconds": time.monotonic() - start_clock,
            "metrics": metrics, "test_performance_computed": False, "publication_executed": False}


def _worker(connection, payload, training, cfg, temporal, weights, seed, device, plan):
    try:
        # Spawn does not inherit the parent's -u flag. Preserve visible epoch progress.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        connection.send({"result": run_seed(TemporalSnapshot(**payload), training, cfg,
                                             temporal, weights, seed, device, plan)})
    except Exception as exc:
        # Never serialize exception text: input values might contain PII.
        connection.send({"error_type": type(exc).__name__})
    finally:
        connection.close()


def isolated_run(snapshot, training, cfg, temporal, weights, seed, device, plan):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, snapshot_payload(snapshot), training,
                                                   cfg, temporal, weights, seed, device, plan))
    try:
        process.start()
        sender.close()
        while not receiver.poll(30):
            if not process.is_alive():
                raise RuntimeError("Research worker exited without results")
        message = receiver.recv()
        process.join()
        if process.exitcode != 0 or "error_type" in message:
            raise RuntimeError("Research worker failed: " + message.get("error_type", "PROCESS_EXIT"))
        return message["result"]
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        receiver.close()
        sender.close()


def aggregate(runs):
    seeds = [run["seed"] for run in runs]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("At least three distinct fixed seeds are required")
    metrics = {}
    for label in runs[0]["metrics"]:
        metrics[label] = {}
        for k in runs[0]["metrics"][label]:
            cells = [run["metrics"][label][k] for run in runs]
            if len({cell["cases"] for cell in cells}) != 1:
                raise ValueError("All seeds require the same evaluation cases")
            metrics[label][k] = {"cases": cells[0]["cases"]}
            for metric in ("recall", "ndcg"):
                values = np.array([cell[metric] for cell in cells], dtype=np.float64)
                metrics[label][k][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return {"weights": runs[0]["weights"], "seeds": seeds, "metrics": metrics}


def select_representative(configurations, swept_field, historical):
    best = max(configurations, key=lambda c: c["metrics"]["overall"]["10"]["ndcg"]["mean"])
    best_metric = best["metrics"]["overall"]["10"]["ndcg"]
    tied = [c for c in configurations if best_metric["mean"] - c["metrics"]["overall"]["10"]["ndcg"]["mean"]
            <= max(best_metric["std"], c["metrics"]["overall"]["10"]["ndcg"]["std"])]
    representative = min(tied, key=lambda c: (abs(math.log(c["weights"][swept_field] / historical)),
                                             c["weights"][swept_field]))
    return {"formal_maximum": best["weights"], "representative": representative["weights"],
            "tied_weights": [c["weights"] for c in tied], "clear_winner": len(tied) == 1,
            "policy": SELECTION_POLICY}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summary_markdown(report):
    lines = ["# BPR-EXP-01: validation-only interaction weights", "",
             f"Status: {report['status']}. Test performance computed: **false**. Publication: **false**.", "",
             "Primary: overall validation NDCG@10 mean; standard deviation: sample, ddof=1.", "",
             SELECTION_POLICY, "", "Frozen configuration:", "```json",
             json.dumps({k: report['plan'][k] for k in ('temporal_config', 'hyperparameters', 'seeds', 'device')},
                        ensure_ascii=False, indent=2), "```", "",
             "FAVORITE validation slice has one real case: diagnostic only, unsuitable for selection.", ""]
    baseline = next((c for c in report.get("stage_a", []) if c["weights"]["view_weight"] == .1), None)
    for stage in ("stage_a", "stage_b"):
        lines += [f"## {stage}", "", "| VIEW | FAVORITE | PURCHASE | Seeds | NDCG@10 mean ± std | Recall@10 mean ± std | VIEW NDCG@10 | PURCHASE NDCG@10 | Δ NDCG (abs; %) | Δ Recall (abs; %) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in report.get(stage, []):
            weights = c["weights"]
            overall = c["metrics"]["overall"]["10"]
            def cell(metric):
                return f"{metric['mean']:.8f} ± {metric['std']:.8f}"
            deltas = []
            for metric in ("ndcg", "recall"):
                reference = baseline["metrics"]["overall"]["10"][metric]["mean"] if baseline else 0
                delta = overall[metric]["mean"] - reference
                percent = f"{100 * delta / reference:+.3f}%" if reference else "undefined"
                deltas.append(f"{delta:+.8f}; {percent}")
            lines.append(f"| {weights['view_weight']} | {weights['favorite_weight']} | 10 | {len(c['seeds'])} | "
                         f"{cell(overall['ndcg'])} | {cell(overall['recall'])} | "
                         f"{cell(c['metrics']['VIEW']['10']['ndcg'])} | {cell(c['metrics']['PURCHASE']['10']['ndcg'])} | "
                         + " | ".join(deltas) + " |")
        lines += ["", "Selection: " + json.dumps(report.get(stage + "_selection"), ensure_ascii=False), ""]
    lines += ["## Historical baseline confidence", "", "| Type | Events | Confidence units (quantity included) | Baseline weight mass |",
              "|---|---|---|---|"]
    factors = {"VIEW": .1, "FAVORITE": 2, "PURCHASE": 10}
    for kind, units in report.get("confidence_units", {}).items():
        lines.append(f"| {kind} | {units['events']} | {units['confidence_units']:.8f} | {units['confidence_units'] * factors[kind]:.8f} |")
    lines += ["", "## Weight mass", "", "| VIEW | FAVORITE | VIEW mass share | FAVORITE mass share | PURCHASE mass share | Pairs | Pair mean | Pair median |",
              "|---|---|---|---|---|---|---|---|"]
    for c in report.get("preparation_diagnostics", []):
        mass = c["mass"]
        lines.append(f"| {c['weights']['view_weight']} | {c['weights']['favorite_weight']} | "
                     + " | ".join(f"{100 * mass['types'][kind]['share']:.4f}%" for kind in ("VIEW", "FAVORITE", "PURCHASE"))
                     + f" | {mass['aggregated_pairs']} | {mass['aggregated_weight_mean']:.5f} | {mass['aggregated_weight_quantiles']['median']:.5f} |")
    lines += ["", "## Ranking diagnostics", "", "| Stage | VIEW | FAVORITE | Slice | K | Cases | NDCG mean ± std | Recall mean ± std |",
              "|---|---|---|---|---|---|---|---|"]
    for stage in ("stage_a", "stage_b"):
        for c in report.get(stage, []):
            for label, ks in c["metrics"].items():
                for k, values in ks.items():
                    lines.append(f"| {stage} | {c['weights']['view_weight']} | {c['weights']['favorite_weight']} | "
                                 f"{label} | {k} | {values['cases']} | {cell(values['ndcg'])} | {cell(values['recall'])} |")
    lines += ["", "Full precision, run-level metrics, confidence quantiles and provenance: results.json.",
              "", "## State selection and remaining technical debt", "",
              "- Trainer calls torch.set_num_interop_threads for every training call. Fresh spawned processes avoid repeated calls without modifying trainer.",
              "- With early_stop=False the trainer returns the final executed epoch without restoring an internal best checkpoint. Empty internal evaluation is not monitored; trainer_best_epoch=-1 denotes no internal best observation. External temporal validation evaluates the returned final model.",
              "- Dense Adam updates the full user embedding table every step; CPU runtime may dominate. No optimizer, model, sampling or budget changes were made.",
              "- As-of catalog contract is missing: features explicitly disabled. Quantity clip, repeat sum, recency and target protocol unchanged.", ""]
    return "\n".join(lines)


def run_experiment(snapshot, temporal, cfg, device, plan, output_dir, *, run=isolated_run, progress=print):
    require_validation(snapshot, temporal)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "results.json").exists():
        raise ValueError("Experiment output already exists; use a new directory")
    write_json(output_dir / "frozen_config.json", plan)
    report = {"status": "running", "plan": plan, "runs": [], "stage_a": [], "stage_b": [],
              "test_performance_computed": False, "publication_executed": False}
    def save():
        write_json(output_dir / "results.json", report)
        (output_dir / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    save()
    progress("Computing exact historical confidence mass (existing quantity semantics).")
    units = confidence_units(snapshot)
    report["confidence_units"] = units
    save()
    fingerprint = None
    reused = {}
    def configuration(weights, stage):
        nonlocal fingerprint
        key = (weights.view_weight, weights.favorite_weight, weights.purchase_weight)
        if key in reused:
            return {**reused[key], "reused_from_stage_a": True}
        progress(f"Preparing {stage}: VIEW={weights.view_weight}, FAVORITE={weights.favorite_weight}, PURCHASE=10")
        data = prepare_bpr_snapshot(snapshot, weights)
        actual_fingerprint = mapping_fingerprint(data.training)
        if fingerprint is not None and actual_fingerprint != fingerprint:
            raise ValueError("Weight configurations changed mappings or positive pairs")
        fingerprint = actual_fingerprint
        mass = mass_diagnostics(units, weights, data.training)
        # Save diagnostics before this configuration's first training run.
        report.setdefault("preparation_diagnostics", []).append({"weights": asdict(weights), "mass": mass,
                                                                  "mapping_sha256": fingerprint})
        save()
        runs = []
        for seed in SEEDS:
            progress(f"Training {stage}: VIEW={weights.view_weight}, FAVORITE={weights.favorite_weight}, seed={seed}, epochs={cfg.epochs}")
            result = run(snapshot, data.training, cfg, temporal, weights, seed, device, plan)
            runs.append(result)
            report["runs"].append(result)
            save()
        value = {**aggregate(runs), "mass": mass}
        reused[key] = value
        return value
    for view in VIEW_GRID:
        report["stage_a"].append(configuration(BprWeightConfig(view_weight=view), "A"))
        save()
    report["stage_a_selection"] = select_representative(report["stage_a"], "view_weight", .1)
    report["status"] = "stage_a_complete"
    save()
    selected_view = report["stage_a_selection"]["representative"]["view_weight"]
    for favorite in FAVORITE_GRID:
        report["stage_b"].append(configuration(BprWeightConfig(view_weight=selected_view, favorite_weight=favorite), "B"))
        save()
    report["stage_b_selection"] = select_representative(report["stage_b"], "favorite_weight", 2)
    report["status"] = "complete"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    return report


def cost_projections(training_seconds, run_seconds):
    """Sequential extrapolation; shared ingestion/preparation overhead is additional."""
    return {str(count): {"training_seconds": count * training_seconds,
                         "training_and_validation_seconds": count * run_seconds}
            for count in (3, 18, 30, 33)}


def single_summary(report):
    lines = ["# BPR-EXP-01: one baseline cost measurement", "",
             f"Status: {report['status']}. Exactly one planned training run. No model selection.", "",
             "Weights VIEW=0.1, FAVORITE=2, PURCHASE=10; seed=42.",
             "Test performance computed: false. Publication: false. Automatic follow-up runs: false.", "",
             "Frozen configuration:", "```json", json.dumps(report["plan"], ensure_ascii=False, indent=2), "```", ""]
    if report["status"] == "interrupted_cost_pilot":
        lines[7:7] = [f"Requested epochs: {report['requested_epochs']}; completed epochs: {report['completed_epochs']}.",
                      "No validation metrics were computed. This incomplete cost pilot must not be used for model-quality comparison.",
                      "No completed-run timing or cost extrapolation is available. Artifacts are retained.", ""]
    if report.get("run"):
        run = report["run"]
        lines += [f"Training wall-clock: {run['training_seconds']:.3f} seconds; device: {run['device']}.",
                  f"Training users/items/pairs: {run['training_users']}/{run['training_items']}/{run['training_pairs']}.",
                  f"Preparation: {report['preparation_seconds']:.3f} seconds; isolated worker including validation: {report['worker_wall_seconds']:.3f} seconds.",
                  "", "Memory counters (worker process only): " + json.dumps(run["training_memory"]), "",
                  "| Slice | K | Cases | NDCG | Recall |", "|---|---|---|---|---|"]
        for label, ks in run["metrics"].items():
            for k, values in ks.items():
                lines.append(f"| {label} | {k} | {values['cases']} | {values['ndcg']:.10f} | {values['recall']:.10f} |")
        lines += ["", "FAVORITE slice has one real validation case; no inference about weight sensitivity is justified.",
                  "One seed cannot estimate seed variance or select a winner.", "",
                  "| Runs | Training hours | Training + validation hours |", "|---|---|---|"]
        for count, estimate in report["projections"].items():
            lines.append(f"| {count} | {estimate['training_seconds'] / 3600:.3f} | {estimate['training_and_validation_seconds'] / 3600:.3f} |")
        lines += ["", "Sequential extrapolations assume the same device and unchanged throughput; shared ingestion and per-configuration preparation add overhead.",
                  "Original Stage A + B: 33 logical runs, 30 unique runs after reusing the three identical FAVORITE=2 runs.",
                  "No second seed, Stage A sweep or Stage B was executed. Await user decision."]
    return "\n".join(lines) + "\n"


def run_single_baseline(snapshot, temporal, cfg, device, plan, output_dir, *, run=isolated_run, progress=print):
    """One run only: deliberately no sweep/seed loop, aggregation or selection."""
    require_validation(snapshot, temporal)
    if cfg.early_stop or cfg.use_item_features:
        raise ValueError("Invalid single baseline configuration")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "results.json").exists():
        raise ValueError("Experiment output already exists; use a new directory")
    weights = BprWeightConfig()
    plan = {**plan, "execution_mode": "single_baseline", "seeds": [42], "planned_training_runs": 1,
            "automatic_follow_up_runs": False, "weights": asdict(weights), "selection_performed": False}
    report = {"status": "preparing", "plan": plan, "run": None,
              "test_performance_computed": False, "publication_executed": False}
    def save():
        write_json(output_dir / "results.json", report)
        (output_dir / "summary.md").write_text(single_summary(report), encoding="utf-8")
    write_json(output_dir / "frozen_config.json", plan)
    save()
    clock = time.monotonic()
    progress("Preparing exactly one baseline: VIEW=0.1, FAVORITE=2, PURCHASE=10, seed=42.")
    data = prepare_bpr_snapshot(snapshot, weights)
    report["preparation_seconds"] = time.monotonic() - clock
    report["training_users"] = len(data.training.mappings.idx2user)
    report["training_items"] = len(data.training.mappings.idx2item)
    report["training_pairs"] = len(data.training.splits.train_pairs)
    report["mapping_sha256"] = mapping_fingerprint(data.training)
    report["parent_memory_before_worker"] = process_memory()
    report["status"] = "training"
    save()
    progress(f"Training one baseline, {cfg.epochs} epochs: users={report['training_users']}, items={report['training_items']}, pairs={report['training_pairs']}.")
    clock = time.monotonic()
    result = run(snapshot, data.training, cfg, temporal, weights, 42, device, plan)
    report["worker_wall_seconds"] = time.monotonic() - clock
    report["run"] = result
    report["projections"] = cost_projections(result["training_seconds"], result["total_seconds"])
    report["status"] = "complete"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    return report
