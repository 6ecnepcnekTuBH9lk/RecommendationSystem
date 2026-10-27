"""GPU-COST-01: one historical-baseline epoch, no ranking or publication."""
from dataclasses import asdict, replace
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
from Application.evaluation.bpr import prepare_bpr_snapshot
from Application.evaluation.experiments.bpr_weights import (
    benchmark_config, git_provenance, mapping_fingerprint, process_memory, temporal_dict, write_json,
)
from Application.evaluation.temporal import build_temporal_protocol
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
from Application.model import BPRMF as core
from Application.model.bpr_preparation import BprWeightConfig
from Application.product_resolution import DEFAULT_CATALOG_PATH

OUTPUT_DIR = PROJECT_ROOT / "diagnostics/gpu_cost01"
ENV_DIR = PROJECT_ROOT / "diagnostics/gpu_env02"
CPU_DIR = PROJECT_ROOT / "diagnostics/bpr_exp01_baseline_cost"


def one_epoch_config(plan):
    original = core.TrainConfig(**plan["hyperparameters"])
    if original.epochs != 200:
        raise ValueError("Historical baseline must have the original 200-epoch budget")
    cfg = replace(original, epochs=1)
    validate_config(cfg)
    if temporal_dict(benchmark_config()) != plan["temporal_config"]:
        raise ValueError("Historical temporal configuration mismatch")
    return cfg


def validate_config(cfg):
    actual = (cfg.epochs, cfg.seed, cfg.w_view_item, cfg.w_favorite, cfg.w_purchase,
              cfg.embedding_dim, cfg.batch_size, cfg.n_neg, cfg.early_stop, cfg.use_item_features)
    if actual != (1, 42, .1, 2., 10., 128, 256, 10, False, False):
        raise ValueError("GPU-COST-01 requires exactly one frozen baseline epoch")


def measure_epoch(training, cfg):
    validate_config(cfg)
    if len(training.splits.eval_users) or len(training.splits.eval_items):
        raise ValueError("Cost measurement requires empty internal evaluation")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA unavailable; CPU fallback is not a GPU cost measurement")
    counters = {"optimizer_steps": 0, "examples_processed": 0}
    original_sample, original_step = core._sample_batch, torch.optim.Adam.step

    def sample(*args, **kwargs):
        batch = original_sample(*args, **kwargs)
        counters["examples_processed"] += len(batch[0])
        return batch

    def step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        counters["optimizer_steps"] += 1
        return result

    # Count delegated operations only in this fresh worker; all arithmetic stays
    # in the original sampler and Adam implementation.
    core._sample_batch, torch.optim.Adam.step = sample, step
    try:
        core._set_seed(cfg.seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        clock = time.perf_counter()
        cpu_clock = time.process_time()
        model, _, observations = core.train_prepared_data_with_metrics(
            cfg, training, torch.device(device))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - clock
        cpu_seconds = time.process_time() - cpu_clock
        cuda_memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scope": "training only; sampled after CUDA synchronization",
        }
        memory = process_memory()
        expected_steps = max(1, math.ceil(len(training.splits.train_pairs) / cfg.batch_size))
        expected_examples = expected_steps * min(cfg.batch_size, len(training.splits.train_pairs))
        assert counters == {"optimizer_steps": expected_steps, "examples_processed": expected_examples}
        assert observations.epochs_completed == 1 and not observations.early_stopped
        assert all(math.isfinite(epoch.loss) for epoch in observations.history)
        assert all(parameter.device.type == "cuda" and torch.isfinite(parameter).all().item()
                   for parameter in model.parameters())
        return {
            **counters, "training_seconds": seconds, "seconds_per_epoch": seconds,
            "worker_cpu_seconds_during_training": cpu_seconds,
            "pairs_per_second": counters["examples_processed"] / seconds,
            "device": str(next(model.parameters()).device),
            "gpu_name": torch.cuda.get_device_name(0), "cuda_memory": cuda_memory,
            "process_memory": memory, "epochs_completed": observations.epochs_completed,
            "loss": observations.history[-1].loss, "finite_loss_and_parameters": True,
            "quality_metrics_computed": False, "publication_executed": False,
        }
    finally:
        core._sample_batch, torch.optim.Adam.step = original_sample, original_step


def worker(connection, training, cfg):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        result = measure_epoch(training, cfg)
        result.update(sys_executable=sys.executable, torch_version=torch.__version__,
                      torch_cuda_build=torch.version.cuda)
        connection.send({"result": result})
    except Exception as exc:
        connection.send({"error_type": type(exc).__name__})
    finally:
        connection.close()


def isolated_epoch(training, cfg):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(sender, training, cfg))
    started = last_progress = time.perf_counter()
    process.start()
    sender.close()
    try:
        while not receiver.poll(1):
            if not process.is_alive():
                raise RuntimeError("GPU cost worker exited without a result")
            if time.perf_counter() - last_progress >= 30:
                print(f"One GPU epoch still running: {time.perf_counter() - started:.1f} seconds including worker startup.", flush=True)
                last_progress = time.perf_counter()
        response = receiver.recv()
        process.join()
        if "result" not in response or process.exitcode != 0:
            raise RuntimeError("GPU cost worker failed: " + response.get("error_type", "ProcessExit"))
        return response["result"], time.perf_counter() - started
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        receiver.close()


def main():
    if (OUTPUT_DIR / "results.json").exists():
        raise RuntimeError("GPU-COST-01 results already exist; automatic repeat refused")
    cuda_check = json.loads((ENV_DIR / "cuda_environment.json").read_text(encoding="utf-8"))
    tests_check = json.loads((ENV_DIR / "tests_after.json").read_text(encoding="utf-8"))
    if (cuda_check["status"] != "passed" or tests_check["status"] != "passed"
            or tests_check["torch_version"] != torch.__version__
            or tests_check["sys_executable"] != sys.executable):
        raise RuntimeError("CUDA/environment/test prerequisites have not passed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA unavailable; no data preparation or training started")
    old_plan = json.loads((CPU_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    old_result = json.loads((CPU_DIR / "results.json").read_text(encoding="utf-8"))
    cfg = one_epoch_config(old_plan)
    temporal = benchmark_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "GPU-COST-01", "status": "preparing",
        "hyperparameters": asdict(cfg), "only_changed_hyperparameter": "epochs: 200 -> 1",
        "temporal_config": temporal_dict(temporal), "device": device,
        "seeds": [42], "planned_training_runs": 1, "quality_run": False,
        "validation_metrics_computed": False, "test_performance_computed": False,
        "publication_executed": False, "weight_sweep_executed": False,
        "automatic_follow_up_runs": False, "provenance": git_provenance(PROJECT_ROOT),
    }
    write_json(OUTPUT_DIR / "frozen_config.json", report)
    write_json(OUTPUT_DIR / "results.json", report)
    try:
        ingestion_clock = time.perf_counter()
        events, source = load_canonical_events(DEFAULT_RAW_ROOT, DEFAULT_CATALOG_PATH,
                                              lambda message: print(message, flush=True))
        report["canonical_ingestion_seconds"] = time.perf_counter() - ingestion_clock
        if source["canonical_revision"] != old_plan["canonical_source"]["canonical_revision"]:
            raise ValueError("Canonical revision differs from the historical cost pilot")
        snapshot_clock = time.perf_counter()
        protocol = build_temporal_protocol(events, temporal)
        snapshot = protocol.validation
        del events, protocol
        report["temporal_snapshot_seconds"] = time.perf_counter() - snapshot_clock
        assert all(event.interaction.event_datetime_utc < temporal.validation_start
                   for event in snapshot.history)
        preparation_clock = time.perf_counter()
        data = prepare_bpr_snapshot(snapshot, BprWeightConfig(
            view_weight=cfg.w_view_item, favorite_weight=cfg.w_favorite, purchase_weight=cfg.w_purchase))
        report["preparation_seconds"] = time.perf_counter() - preparation_clock
        training = data.training
        report.update(training_users=len(training.mappings.idx2user),
                      training_items=len(training.mappings.idx2item),
                      training_pairs=len(training.splits.train_pairs),
                      mapping_sha256=mapping_fingerprint(training),
                      canonical_revision=source["canonical_revision"])
        for key in ("training_users", "training_items", "training_pairs", "mapping_sha256"):
            if report[key] != old_result[key]:
                raise ValueError("Historical training snapshot mismatch")
        del snapshot, data
        report["parent_memory_before_worker"] = process_memory()
        report["status"] = "training_one_epoch"
        write_json(OUTPUT_DIR / "results.json", report)
        print(f"Exactly one GPU epoch: {report['training_users']} users, "
              f"{report['training_items']} items, {report['training_pairs']} pairs.", flush=True)
        report["run"], report["worker_wall_seconds"] = isolated_epoch(training, cfg)
        seconds = report["run"]["training_seconds"]
        report.update(status="complete", linear_reference_seconds={
            str(epochs): epochs * seconds for epochs in (20, 50, 200)},
            cpu_reference={"epoch_seconds": [370, 472, 505], "representative": "median",
                           "representative_seconds": 472, "approximate_speedup": 472 / seconds,
                           "scope": "first three epochs of an interrupted CPU pilot, not a full benchmark"})
        write_json(OUTPUT_DIR / "results.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error_type=type(exc).__name__)
        write_json(OUTPUT_DIR / "results.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

