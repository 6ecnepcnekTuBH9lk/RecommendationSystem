"""Bounded, paired feature ablation on the existing temporal/BPR implementation.

No ingestion or publication. Each arm runs sequentially in a fresh CPU process.
STRICT trusts only explicit reviewed catalog provenance, never filesystem mtime.
"""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from Application.evaluation.bpr import BprTemporalData, evaluate_bpr_snapshot, prepare_bpr_snapshot
from Application.evaluation.temporal import TemporalProtocolError, TemporalSnapshot
from Application.model import BPRMF as core
from Application.model.bpr_preparation import BprWeightConfig
from .bpr_weights import mapping_fingerprint, snapshot_payload

SEEDS = (42, 43, 44)
MODES = ("STRICT", "EXPLORATORY")


class FeatureExperimentError(ValueError):
    """Aggregate-only error: never include source IDs or catalog values."""


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class CatalogEvidence:
    sha256: str
    available_at: datetime
    basis: str
    reference: str
    reviewed_by: str


@dataclass(frozen=True)
class CatalogSource:
    path: Path
    evidence: CatalogEvidence | None = None

    def freeze(self, cutoff, mode, dataset_kind):
        if mode not in MODES:
            raise FeatureExperimentError("Expected STRICT or EXPLORATORY")
        content = Path(self.path).read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        evidence = self.evidence
        confirmed = False
        if evidence is not None:
            if evidence.sha256 != actual_hash:
                raise FeatureExperimentError("Catalog evidence checksum mismatch")
            stamp = evidence.available_at
            if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
                raise FeatureExperimentError("Catalog availability must be timezone-aware")
            allowed = ("CONTROLLED_SYNTHETIC",) if dataset_kind == "SYNTHETIC" else ("REVIEWED_HISTORICAL_SNAPSHOT",)
            confirmed = (stamp.astimezone(timezone.utc) <= cutoff and evidence.basis in allowed
                         and bool(evidence.reference.strip()) and bool(evidence.reviewed_by.strip()))
        if mode == "STRICT" and not confirmed:
            raise FeatureExperimentError("STRICT blocked: confirmed pre-cutoff catalog evidence required")
        return content, {"sha256": actual_hash, "as_of_confirmed": confirmed,
                         "availability": evidence.available_at.isoformat() if evidence else None,
                         "evidence_basis": evidence.basis if evidence else None,
                         "metadata_leakage_risk": not confirmed}


def common_initial_weights(users, items, dimension, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(users, dimension, generator=generator) * .1,
            torch.randn(items, dimension, generator=generator) * .1)


def fingerprints(snapshot, training):
    cases = [(case.customer_id, case.target_item, case.target_timestamp.isoformat(),
              case.interaction_type.value, sorted(case.seen_items), snapshot.candidates_for(case))
             for case in snapshot.cases]
    return {"training": digest([mapping_fingerprint(training),
                                 hashlib.sha256(training.splits.train_weights.tobytes()).hexdigest(),
                                 [(e.interaction.customer_id, e.item_id, e.interaction.event_datetime_utc.isoformat(),
                                   e.interaction.interaction_type.value, str(e.interaction.quantity))
                                  for e in snapshot.history]]),
            "cases_candidates_seen": digest(cases)}


def _run_arm(snapshot, training, cfg, catalog_bytes, arm, seed):
    """In a fresh process: no set_num_interop_threads workaround or global patch."""
    before = fingerprints(snapshot, training)
    initial = common_initial_weights(len(training.mappings.idx2user), len(training.mappings.idx2item),
                                     cfg.embedding_dim, seed)
    initialized = {}
    sampler = hashlib.sha256()
    batches = 0

    def initialize(model):
        model.user_emb.weight.copy_(initial[0])
        model.item_emb.weight.copy_(initial[1])
        initialized["users"] = tensor_hash(model.user_emb.weight)
        initialized["items"] = tensor_hash(model.item_emb.weight)
        if model.use_item_features:
            generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
            model.feat_emb.weight.copy_(torch.randn(model.feat_emb.weight.shape, generator=generator) * .1)
            initialized["features"] = tensor_hash(model.feat_emb.weight)
        # Only feature dropout consumes this stream after initialization.
        torch.manual_seed(seed + 2_000_003)

    def observe_batch(users, positives, negatives):
        nonlocal batches
        for array in (users, positives, negatives):
            sampler.update(str(array.shape).encode())
            sampler.update(array.tobytes())
        batches += 1

    with tempfile.TemporaryDirectory(prefix="bpr-feature-arm-") as temporary:
        Path(temporary, "nomenclature.csv").write_bytes(catalog_bytes)
        current = replace(cfg, data_dir=temporary, seed=seed, use_item_features=arm == "B")
        torch.manual_seed(seed)
        started = time.perf_counter()
        # Do not retain epoch logs or catalog details in worker IPC/results.
        with redirect_stdout(io.StringIO()):
            model, _, metrics = core.train_prepared_data_with_metrics(
                current, training, torch.device("cpu"), model_initializer=initialize, batch_observer=observe_batch)
        seconds = time.perf_counter() - started
    if metrics.epochs_completed != cfg.epochs or metrics.early_stopped:
        raise FeatureExperimentError("Frozen training budget not completed")
    if fingerprints(snapshot, training) != before:
        raise FeatureExperimentError("Training snapshot mutated")
    model.eval()
    feature_diagnostics = model._item_feature_snapshot.diagnostics()
    participates = bool(model.use_item_features and not torch.equal(model.item_vec_all(), model.item_emb.weight))
    learned = model.use_item_features and initialized["features"] != tensor_hash(model.feat_emb.weight)
    if (arm == "A" and (model.use_item_features or feature_diagnostics["unique_tokens"])) or (
            arm == "B" and not (participates and learned)):
        raise FeatureExperimentError("Feature ablation was not actually applied")
    contributions = []
    score = evaluate_bpr_snapshot(model, BprTemporalData(snapshot, training), 10,
                                  case_observer=lambda case, values: contributions.append(values))
    candidate_union = set().union(*(set(snapshot.candidates_for(case)) for case in snapshot.cases))
    eligible = snapshot.diagnostics.eligible_warm_user_targets
    return {"arm": arm, "seed": seed, "recall10": score.recall, "ndcg10": score.ndcg,
            "hitrate10": score.recall, "evaluated_users": score.evaluated_users,
            "targets": len(snapshot.cases), "candidate_items": len(snapshot.item_universe),
            "candidate_item_coverage": len(candidate_union) / len(snapshot.item_universe),
            "warm_target_coverage": len(snapshot.cases) / eligible if eligible else 0.,
            "training_events": len(snapshot.history), "training_pairs": len(training.splits.train_pairs),
            "epochs": metrics.epochs_completed, "training_seconds": seconds,
            "initial_hashes": initialized, "sampler_sha256": sampler.hexdigest(), "batches": batches,
            "fingerprints": before, "features_participate": participates,
            "feature_weights_changed": bool(learned), "feature_diagnostics": feature_diagnostics,
            "contributions": contributions}


def _worker(connection, payload, training, cfg, content, arm, seed):
    try:
        connection.send({"result": _run_arm(TemporalSnapshot(**payload), training, cfg, content, arm, seed)})
    except Exception as exc:
        # Exception messages from pandas/torch may contain source values.
        connection.send({"error": type(exc).__name__})
    finally:
        connection.close()


def isolated_arm(snapshot, training, cfg, content, arm, seed):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, snapshot_payload(snapshot), training, cfg, content, arm, seed))
    try:
        process.start()
        sender.close()
        while not receiver.poll(1):
            if not process.is_alive():
                raise FeatureExperimentError("Research worker exited without a result")
        message = receiver.recv()
        process.join()
        if process.exitcode or "error" in message:
            raise FeatureExperimentError("Research worker failed: " + message.get("error", "PROCESS_EXIT"))
        return message["result"]
    finally:
        sender.close()
        receiver.close()
        if process.is_alive():
            process.terminate()  # Only this isolated experimental worker, on error/cancel.
            process.join()


def check_pair(first, second):
    for key in ("fingerprints", "sampler_sha256", "batches", "evaluated_users", "targets", "candidate_items"):
        if first[key] != second[key]:
            raise FeatureExperimentError("Paired arms differ: " + key)
    for name in ("users", "items"):
        if first["initial_hashes"][name] != second["initial_hashes"][name]:
            raise FeatureExperimentError("Paired base initialization differs")


def paired_summary(runs, *, bootstrap_samples=1000):
    by_arm = {}
    for arm in ("A", "B"):
        selected = [run for run in runs if run["arm"] == arm]
        by_arm[arm] = {name: {"mean": float(np.mean([r[name] for r in selected])),
                              "sample_std": float(np.std([r[name] for r in selected], ddof=1)) if len(selected) > 1 else None}
                       for name in ("recall10", "ndcg10", "hitrate10", "training_seconds")}
    first = [r for r in runs if r["arm"] == "A"]
    second = [r for r in runs if r["arm"] == "B"]
    if not first or len(first) != len(second) or any(r["fingerprints"] != first[0]["fingerprints"] for r in runs):
        raise FeatureExperimentError("Bootstrap requires the same cohort across seeds")
    for a, b in zip(first, second):
        if a["seed"] != b["seed"]:
            raise FeatureExperimentError("Unpaired seeds")
        check_pair(a, b)
    differences = np.array([np.array(b["contributions"]) - np.array(a["contributions"])
                            for a, b in zip(first, second)])
    per_user = differences.mean(axis=0)
    uncertainty = {"method": "paired user bootstrap after averaging seed deltas", "users": len(per_user),
                   "seed_variability_is_separate": True, "interval95": None}
    if len(per_user) >= 30 and bootstrap_samples > 0:
        generator = np.random.default_rng(20261026)
        draws = np.array([per_user[generator.integers(len(per_user), size=len(per_user))].mean(axis=0)
                          for _ in range(bootstrap_samples)])
        uncertainty["interval95"] = {name: list(map(float, np.quantile(draws[:, index], [.025, .975])))
                                     for index, name in enumerate(("recall10", "ndcg10"))}
    return {"arms": by_arm, "delta_B_minus_A": {name: by_arm["B"][name]["mean"] - by_arm["A"][name]["mean"]
                                                for name in ("recall10", "ndcg10", "hitrate10")},
            "uncertainty": uncertainty}


def run_experiment(protocol, cfg, validation_catalog, test_catalog, *, mode="STRICT",
                   dataset_kind="REAL_BOUNDED", seeds=SEEDS, on_plan=None, on_phase=None):
    """Freeze the pair before validation; test never selects a configuration.

    Catalog provenance is an explicit caller attestation requiring independent
    review. A checksum and timestamp alone do not prove historical availability.
    The caller supplies bounded canonical snapshots; no raw source reader here.
    """
    if mode not in MODES or dataset_kind not in ("SYNTHETIC", "REAL_BOUNDED"):
        raise FeatureExperimentError("Invalid experiment mode/dataset kind")
    seeds = tuple(seeds)
    if not seeds or len(seeds) > 5 or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds):
        raise FeatureExperimentError("Expected one to five distinct predeclared seeds")
    cfg = deepcopy(replace(cfg, early_stop=False))
    core._validate_feature_config(cfg)
    if (not 1 <= cfg.epochs <= 50 or not 1 <= cfg.embedding_dim <= 128
            or not 1 <= cfg.batch_size <= 4096 or not 1 <= cfg.n_neg <= 20
            or cfg.feature_scale <= 0 or cfg.feature_dropout >= 1):
        raise FeatureExperimentError("Invalid bounded training budget/feature configuration")
    weights = BprWeightConfig(cfg.w_view_item, cfg.w_favorite, cfg.w_purchase)
    snapshots = (("validation", protocol.validation, validation_catalog), ("test", protocol.test, test_catalog))
    if (protocol.validation.cutoff != protocol.config.validation_start
            or protocol.validation.future_end != protocol.config.test_start
            or protocol.test.cutoff != protocol.config.test_start
            or protocol.test.future_end != protocol.config.test_end
            or protocol.config.test_end is None):
        raise TemporalProtocolError("Experiment snapshots must match disjoint configured windows")
    prepared = []
    # Both source gates pass before any model is trained.
    for phase, snapshot, source in snapshots:
        if not snapshot.cases or len(snapshot.history) > 100_000 or len(snapshot.seen_at_cutoff) > 5_000 or len(snapshot.item_universe) > 10_000:
            raise FeatureExperimentError("Empty evaluation cohort or bounded data limit exceeded")
        content, provenance = source.freeze(snapshot.cutoff, mode, dataset_kind)
        data = prepare_bpr_snapshot(snapshot, weights).training
        if any(len(items) >= len(data.mappings.idx2item) for items in data.splits.user_pos_train):
            raise FeatureExperimentError("No unseen negative candidates for a training user")
        if len(data.splits.train_pairs) * cfg.epochs * max(1, cfg.n_neg) * cfg.embedding_dim > 100_000_000:
            raise FeatureExperimentError("Bounded training operation limit exceeded")
        prepared.append((phase, snapshot, data, content, provenance))
    parameters = asdict(cfg)
    parameters.pop("data_dir")
    parameters.pop("use_item_features")
    plan = {"mode": mode, "dataset_kind": dataset_kind, "seeds": seeds, "parameters": parameters,
            "k": 10, "test_tuning": False, "selection": "one predeclared pair; no search or seed selection",
            "cutoffs": [s.cutoff.isoformat() for _, s, _, _, _ in prepared],
            "test_end": protocol.config.test_end.isoformat(),
            "catalogs": {phase: provenance for phase, _, _, _, provenance in prepared},
            "fingerprints": {phase: fingerprints(snapshot, data) for phase, snapshot, data, _, _ in prepared},
            "publication": False}
    plan_hash = digest(plan)
    if on_plan is not None:
        on_plan({"sha256": plan_hash, **plan})
    all_runs, summaries = [], {}
    for phase, snapshot, data, content, provenance in prepared:
        phase_runs = []
        for seed in seeds:
            a = isolated_arm(snapshot, data, cfg, content, "A", seed)
            b = isolated_arm(snapshot, data, cfg, content, "B", seed)
            check_pair(a, b)
            for run in (a, b):
                run.update(phase=phase, mode=mode, dataset_kind=dataset_kind, plan_sha256=plan_hash,
                           metadata_leakage_risk=provenance["metadata_leakage_risk"])
                phase_runs.append(run)
                all_runs.append(run)
        summaries[phase] = paired_summary(phase_runs)
        if on_phase is not None:
            on_phase(phase, phase_runs, summaries[phase])
    return {"plan": plan, "plan_sha256": plan_hash, "runs": all_runs, "summaries": summaries}
