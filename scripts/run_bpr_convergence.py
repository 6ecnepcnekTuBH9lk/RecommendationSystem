"""CONV-01A: one continuous baseline, eight validation observations, no publication."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from Application.evaluation.audit import load_canonical_events
from Application.evaluation.bpr import BprTemporalData, evaluate_bpr_snapshot, prepare_bpr_snapshot
from Application.evaluation.experiments.bpr_weights import (
    benchmark_config, git_provenance, mapping_fingerprint, process_memory,
    require_validation, snapshot_payload, temporal_dict, write_json,
)
from Application.evaluation.temporal import TemporalSnapshot, build_temporal_protocol
from Application.interactions import InteractionType
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
from Application.model import BPRMF as core
from Application.model.bpr_preparation import BprWeightConfig
from Application.product_resolution import DEFAULT_CATALOG_PATH

OUTPUT_DIR = PROJECT_ROOT / "diagnostics/bpr_conv01a"
REFERENCE_DIR = PROJECT_ROOT / "diagnostics/bpr_exp01_baseline_cost"
CHECKPOINTS = (1, 2, 3, 5, 8, 12, 20, 30)


def baseline_config(plan):
    original = core.TrainConfig(**plan["hyperparameters"])
    cfg = replace(original, epochs=30)
    if (original.epochs != 200 or cfg.seed != 42 or cfg.early_stop or cfg.use_item_features
            or (cfg.w_view_item, cfg.w_favorite, cfg.w_purchase) != (.1, 2., 10.)
            or (cfg.embedding_dim, cfg.batch_size, cfg.n_neg, cfg.lr, cfg.bpr_reg,
                cfg.weight_decay) != (128, 256, 10, .0003, .0005, 0.)):
        raise ValueError("CONV-01A requires the frozen historical baseline")
    if temporal_dict(benchmark_config()) != plan["temporal_config"]:
        raise ValueError("Historical temporal configuration mismatch")
    return cfg


def validate_run(snapshot, training, cfg, temporal, checkpoints):
    # This guard runs before any trainer/scoring invocation, including in the child.
    require_validation(snapshot, temporal)
    if (cfg.early_stop or cfg.use_item_features or cfg.seed != 42
            or len(training.splits.eval_users) or len(training.splits.eval_items)
            or not 1 <= cfg.epochs <= 30):
        raise ValueError("Convergence requires a fixed final-state baseline with empty internal eval")
    if (not checkpoints or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
            or any(type(e) is not int or not 1 <= e <= cfg.epochs for e in checkpoints)
            or checkpoints[-1] != cfg.epochs):
        raise ValueError("Checkpoints must be ordered unique completed epochs including the final epoch")


def snapshot_fingerprint(snapshot, training):
    """Hash historical weighted data and targets; never write customer/item identities."""
    digest = hashlib.sha256(mapping_fingerprint(training).encode("ascii"))
    digest.update(training.splits.train_weights.tobytes())
    for event in snapshot.history:
        r = event.interaction
        digest.update(json.dumps([r.customer_id, event.item_id, r.interaction_type.value,
                                  r.event_datetime_utc.isoformat(), str(r.quantity)],
                                 ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    for case in snapshot.cases:
        digest.update(json.dumps([case.customer_id, case.target_item, case.target_timestamp.isoformat(),
                                  case.interaction_type.value, sorted(case.seen_items)],
                                 ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def cuda_memory(device):
    if device.type != "cuda":
        return {"available": False}
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def score_checkpoint(model, snapshot, training, temporal, device):
    require_validation(snapshot, temporal)
    versions = tuple(p._version for p in model.parameters())
    mode = model.training
    synchronize(device)
    started = time.perf_counter()
    metrics = {}
    for label, kind in (("overall", None), *( (k.value, k) for k in InteractionType)):
        selected = snapshot if kind is None else replace(
            snapshot, cases=tuple(case for case in snapshot.cases if case.interaction_type is kind))
        metrics[label] = {}
        for k in ((5, 10, 20) if kind is None else (10,)):
            score = evaluate_bpr_snapshot(model, BprTemporalData(selected, training), k, device)
            metrics[label][str(k)] = {"cases": score.evaluated_users,
                                     "ndcg": score.ndcg, "recall": score.recall}
    synchronize(device)
    seconds = time.perf_counter() - started
    if versions != tuple(p._version for p in model.parameters()) or mode != model.training:
        raise RuntimeError("Validation must not mutate model state or mode")
    return {"metrics": metrics, "scoring_seconds": seconds, "scoring_device": str(device),
            "cases": len(snapshot.cases), "cuda_memory_after_scoring": cuda_memory(device)}


def train_trajectory(snapshot, training, cfg, temporal, device, checkpoints, progress):
    """One trainer/Adam lifecycle. progress receives scalars only, never model tensors."""
    validate_run(snapshot, training, cfg, temporal, checkpoints)
    counts = [len(snapshot.item_universe) - len(case.seen_items) for case in snapshot.cases]
    candidate_counts = ({"min": min(counts), "max": max(counts), "mean": sum(counts) / len(counts)}
                        if counts else {"min": None, "max": None, "mean": None})
    core._set_seed(cfg.seed)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epoch_start = time.perf_counter()
    trajectory = []
    previous_ndcg = None
    cumulative_training = 0.0

    def observe(model, epoch):
        nonlocal epoch_start, previous_ndcg, cumulative_training
        synchronize(device)
        seconds = time.perf_counter() - epoch_start
        cumulative_training += seconds
        row = {"epoch": epoch.epoch, "loss": epoch.loss, "training_seconds": seconds,
               "cumulative_training_seconds": cumulative_training,
               "cuda_memory_before_scoring": cuda_memory(device)}
        if epoch.epoch in checkpoints:
            row["validation"] = score_checkpoint(model, snapshot, training, temporal, device)
            row["validation"]["candidate_counts"] = candidate_counts
            ndcg = row["validation"]["metrics"]["overall"]["10"]["ndcg"]
            row["ndcg10_delta"] = None if previous_ndcg is None else ndcg - previous_ndcg
            row["ndcg10_relative_delta"] = (None if not previous_ndcg
                                             else (ndcg - previous_ndcg) / previous_ndcg)
            previous_ndcg = ndcg
            print(f"Checkpoint {epoch.epoch}: NDCG@10={ndcg:.10f}, "
                  f"scoring={row['validation']['scoring_seconds']:.3f}s", flush=True)
        trajectory.append(row)
        progress(row)
        # Exclude scoring, bookkeeping and result persistence from the next epoch.
        epoch_start = time.perf_counter()

    model, _, observations = core.train_prepared_data_with_metrics(
        cfg, training, device, epoch_observer=observe)
    synchronize(device)
    if observations.epochs_completed != cfg.epochs or observations.early_stopped:
        raise RuntimeError("Frozen epoch budget was not completed")
    if [r["epoch"] for r in trajectory if "validation" in r] != list(checkpoints):
        raise RuntimeError("Checkpoint trajectory mismatch")
    if any(not math.isfinite(e.loss) for e in observations.history):
        raise RuntimeError("Nonfinite training loss")
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("Nonfinite final model parameters")
    return {"epochs_completed": observations.epochs_completed,
            "returned_state_epoch": cfg.epochs, "internal_best_epoch": observations.best_epoch,
            "training_seconds": cumulative_training,
            "evaluation_seconds": sum(r.get("validation", {}).get("scoring_seconds", 0.) for r in trajectory),
            "cuda_memory": cuda_memory(device), "process_memory": process_memory(),
            "device": str(next(model.parameters()).device), "trajectory": trajectory,
            "sys_executable": sys.executable, "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda, "finite_loss_and_parameters": True,
            "test_performance_computed": False, "publication_executed": False,
            "timing_scope": "Training excludes observer; epoch 1 includes model/Adam initialization"}


def worker(connection, payload, training, cfg, temporal):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback refused")
        result = train_trajectory(TemporalSnapshot(**payload), training, cfg, temporal,
                                  torch.device("cuda"), CHECKPOINTS,
                                  lambda row: connection.send({"epoch": row}))
        connection.send({"result": result})
    except Exception as exc:
        # Safe failure metadata only: canonical records may contain personal data.
        connection.send({"error_type": type(exc).__name__})
    finally:
        connection.close()


def isolated_run(snapshot, training, cfg, temporal, report):
    validate_run(snapshot, training, cfg, temporal, CHECKPOINTS)
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(sender, snapshot_payload(snapshot), training, cfg, temporal))
    clock = time.perf_counter()
    try:
        process.start()
        sender.close()
        while True:
            if not receiver.poll(30):
                if not process.is_alive():
                    raise RuntimeError("Convergence worker exited without results")
                print("Continuous CONV-01A worker active", flush=True)
                continue
            response = receiver.recv()
            if "epoch" in response:
                report["trajectory"].append(response["epoch"])
                write_json(OUTPUT_DIR / "results.json", report)
                continue
            process.join()
            if process.exitcode != 0 or "result" not in response:
                raise RuntimeError("Convergence worker failed: " + response.get("error_type", "ProcessExit"))
            return response["result"], time.perf_counter() - clock
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        receiver.close()
        sender.close()


def main():
    if (OUTPUT_DIR / "results.json").exists():
        raise RuntimeError("CONV-01A output already exists; automatic repeat refused")
    tests = json.loads((OUTPUT_DIR / "tests_before_run.json").read_text(encoding="utf-8"))
    if (tests["status"] != "passed" or tests["sys_executable"] != sys.executable
            or tests["torch_version"] != torch.__version__):
        raise RuntimeError("Required tests/environment verification has not passed")
    for name, sha in tests["source_sha256"].items():
        if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError("Code changed after verification")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback refused")
    reference = json.loads((REFERENCE_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    old = json.loads((REFERENCE_DIR / "results.json").read_text(encoding="utf-8"))
    cfg, temporal = baseline_config(reference), benchmark_config()
    report = {"experiment": "CONV-01A", "status": "preparing",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "hyperparameters": asdict(cfg), "optimizer": "torch.optim.Adam",
              "only_changed_hyperparameter": "epochs: 200 -> 30",
              "seeds": [42], "planned_training_runs": 1, "checkpoint_epochs": list(CHECKPOINTS),
              "primary_metric": "overall.validation.NDCG@10",
              "secondary_metrics": ["Recall@10", "NDCG@5", "Recall@5", "NDCG@20", "Recall@20"],
              "temporal_config": temporal_dict(temporal), "provenance": git_provenance(PROJECT_ROOT),
              "sys_executable": sys.executable, "torch_version": torch.__version__,
              "torch_cuda_build": torch.version.cuda, "device": "cuda",
              "gpu": torch.cuda.get_device_name(0), "trajectory": [],
              "test_performance_computed": False, "publication_executed": False,
              "automatic_follow_up_runs": False, "weight_sweep_executed": False}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "frozen_config.json", report)
    write_json(OUTPUT_DIR / "results.json", report)
    try:
        clock = time.perf_counter()
        events, source = load_canonical_events(DEFAULT_RAW_ROOT, DEFAULT_CATALOG_PATH,
                                              lambda message: print(message, flush=True))
        report["canonical_ingestion_seconds"] = time.perf_counter() - clock
        if source["canonical_revision"] != reference["canonical_source"]["canonical_revision"]:
            raise ValueError("Canonical revision differs from historical baseline")
        clock = time.perf_counter()
        # Reuse the unchanged protocol builder; its test object is discarded here.
        # Only validation is ever prepared, serialized to the worker or scored.
        protocol = build_temporal_protocol(events, temporal)
        snapshot = protocol.validation
        del events, protocol
        require_validation(snapshot, temporal)
        if len(snapshot.cases) != 4976:
            raise ValueError("Historical validation case count mismatch")
        report["snapshot_seconds"] = time.perf_counter() - clock
        clock = time.perf_counter()
        data = prepare_bpr_snapshot(snapshot, BprWeightConfig())
        training = data.training
        report["preparation_seconds"] = time.perf_counter() - clock
        report.update(training_users=len(training.mappings.idx2user), training_items=len(training.mappings.idx2item),
                      training_pairs=len(training.splits.train_pairs), mapping_sha256=mapping_fingerprint(training))
        for key in ("training_users", "training_items", "training_pairs", "mapping_sha256"):
            if report[key] != old[key]:
                raise ValueError("Historical mapping/pair snapshot mismatch")
        counts = [len(snapshot.item_universe) - len(case.seen_items) for case in snapshot.cases]
        report.update(canonical_source=source, dataset_sha256=snapshot_fingerprint(snapshot, training),
                      validation_cases=len(snapshot.cases), training_events=len(snapshot.history),
                      candidate_counts={"min": min(counts), "max": max(counts), "mean": sum(counts) / len(counts)},
                      validation_slices={k.value: sum(c.interaction_type is k for c in snapshot.cases)
                                         for k in InteractionType}, parent_memory_before_worker=process_memory())
        frozen = {k: v for k, v in report.items() if k not in ("status", "trajectory")}
        write_json(OUTPUT_DIR / "frozen_config.json", frozen)
        report["status"] = "training"
        write_json(OUTPUT_DIR / "results.json", report)
        print("One continuous 30-epoch GPU baseline; no follow-up configurations.", flush=True)
        report["run"], report["worker_wall_seconds"] = isolated_run(snapshot, training, cfg, temporal, report)
        report.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        write_json(OUTPUT_DIR / "results.json", report)
        write_json(OUTPUT_DIR / "trajectory.json", report["trajectory"])
        print("CONV-01A complete. Stopped after epoch 30; no publication or test scoring.", flush=True)
        return 0
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error_type=type(exc).__name__)
        write_json(OUTPUT_DIR / "results.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
