"""Bounded synthetic validation of the paired BPR feature research mechanism.

Real data are intentionally not ingested here. The research API accepts already
resolved bounded TemporalEvaluationProtocol objects and reviewed catalog evidence.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from Application.evaluation.experiments.item_features import (
    CatalogEvidence, CatalogSource, FeatureExperimentError, run_experiment,
)
from Application.evaluation.temporal import TemporalConfig, build_temporal_protocol
from Application.interactions import InteractionRecord, InteractionSource, InteractionType
from Application.mindbox.records import ProductKey
from Application.model.BPRMF import TrainConfig
from Application.model.item_feature_snapshot import FEATURE_FIELDS
from Application.product_resolution import ResolvedInteraction


def synthetic_protocol(directory):
    """Fixed generator, not selected by evaluation quality; all identities artificial."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    codes = [f"{i + 1:06d}" for i in range(64)]
    rows = []
    for index, code in enumerate(codes):
        row = {"КодНоменклатуры": code}
        row.update({name: f"GROUP_{index // 8}" for name in FEATURE_FIELDS})
        if index % 5 == 0:
            row["СезонНоски"] = ""
        if index % 7 == 0:
            row["СтилеваяГруппа"] = ""
        rows.append(row)
    catalog = directory / "nomenclature.csv"
    pd.DataFrame(rows).to_csv(catalog, sep="|", index=False, encoding="utf-8-sig")
    generator = np.random.default_rng(20261026)
    validation = datetime(2020, 1, 1, tzinfo=timezone.utc)
    test = datetime(2020, 2, 1, tzinfo=timezone.utc)
    end = datetime(2020, 3, 1, tzinfo=timezone.utc)
    events = []

    def event(user, item, stamp, number):
        kind = tuple(InteractionType)[number % 3]
        source = InteractionSource.ORDER if kind is InteractionType.PURCHASE else InteractionSource.ACTION
        record = InteractionRecord(user, user, ProductKey("offline1C", item), kind, stamp,
                                   source, f"synthetic-event-{len(events)}", Decimal(1))
        return ResolvedInteraction(record, item)

    for user_index in range(64):
        user = f"synthetic-user-{user_index:03d}"
        group = list(range((user_index % 8) * 8, (user_index % 8 + 1) * 8))
        mandatory = group[user_index // 8]  # Guarantees every item exists before validation.
        other = [index for index in group if index != mandatory]
        preferred = [mandatory, *generator.choice(other, size=3, replace=False).tolist()]
        outside = [index for index in range(64) if index not in group]
        history = preferred + generator.choice(outside, size=8, replace=False).tolist()
        future = generator.choice([index for index in group if index not in preferred], size=2, replace=False)
        for offset, item_index in enumerate(history):
            stamp = datetime(2019, 12, 1, tzinfo=timezone.utc) + timedelta(minutes=user_index * 20 + offset)
            events.append(event(user, codes[item_index], stamp, user_index + offset))
        events.append(event(user, codes[future[0]], validation + timedelta(minutes=user_index), user_index))
        events.append(event(user, codes[future[1]], test + timedelta(minutes=user_index), user_index + 1))
    evidence = CatalogEvidence(hashlib.sha256(catalog.read_bytes()).hexdigest(),
                               datetime(2019, 11, 1, tzinfo=timezone.utc), "CONTROLLED_SYNTHETIC",
                               "fixed generator v1; artificial availability, not real historical evidence", "synthetic fixture")
    protocol = build_temporal_protocol(events, TemporalConfig(validation, test, end, min_history_events=10))
    return protocol, CatalogSource(catalog, evidence)


def production_signature():
    manifest = ROOT / "model/current.json"
    if not manifest.is_file():
        return {"manifest_exists": False}
    content = manifest.read_bytes()
    generation = json.loads(content)["generation"]
    files = {}
    for name in ("bprmf.pt", "mappings.json"):
        stat = (ROOT / "model/runs" / generation / name).stat()
        files[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"manifest_sha256": hashlib.sha256(content).hexdigest(), "files": files}


def write_results(output, result):
    fields = ("dataset_kind", "mode", "phase", "arm", "seed", "recall10", "ndcg10", "hitrate10",
              "evaluated_users", "targets", "candidate_items", "candidate_item_coverage", "warm_target_coverage",
              "feature_item_coverage", "feature_tokens", "training_events", "training_pairs", "epochs", "training_seconds",
              "metadata_leakage_risk", "initial_users_sha256", "initial_items_sha256", "sampler_sha256", "plan_sha256")
    with (output / "RESULTS.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in result["runs"]:
            row = {name: run[name] for name in fields if name in run}
            diag = run["feature_diagnostics"]
            row.update(feature_item_coverage=1 - diag["items_without_features"] / diag["items"],
                       feature_tokens=diag["unique_tokens"], initial_users_sha256=run["initial_hashes"]["users"],
                       initial_items_sha256=run["initial_hashes"]["items"])
            writer.writerow(row)
    # Per-user contributions stay in memory for paired bootstrap; no identifier table.
    aggregate = {**result, "runs": [{k: v for k, v in run.items() if k != "contributions"} for run in result["runs"]]}
    (output / "RESULTS.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Сопоставление метрик — только контролируемая синтетика", "",
             "Это проверка исследовательского механизма, не оценка качества на реальных покупателях.", "",
             "| Срез | Вариант | Recall@10 mean ± sample std | NDCG@10 mean ± sample std | Train seconds mean |",
             "|---|---|---:|---:|---:|"]
    notes = []
    for phase, summary in result["summaries"].items():
        for arm, metrics in summary["arms"].items():
            recall, ndcg = metrics["recall10"], metrics["ndcg10"]
            lines.append(f"| {phase} | {arm} | {recall['mean']:.6f} ± {recall['sample_std'] or 0:.6f} | "
                         f"{ndcg['mean']:.6f} ± {ndcg['sample_std'] or 0:.6f} | {metrics['training_seconds']['mean']:.3f} |")
        notes += ["", f"{phase}: Δ B−A = {json.dumps(summary['delta_B_minus_A'], ensure_ascii=False)}.",
                  f"Парный bootstrap: {json.dumps(summary['uncertainty'], ensure_ascii=False)}.", ""]
    lines += notes
    lines += ["Recall@10 = HitRate@10: один первый новый target на пользователя.",
              "Оценка следующего нового взаимодействия включает VIEW/FAVORITE/PURCHASE, а не только покупки.",
              "Интервалы условны на фиксированных seeds и синтетической когорте; доказательств бизнес-улучшения нет."]
    (output / "METRICS_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", required=True)
    parser.add_argument("--mode", choices=("STRICT", "EXPLORATORY"), default="STRICT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    protected = [ROOT / name for name in ("model", "input_data", "user_settings", ".git")]
    if output == ROOT or output in ROOT.parents or any(output == path or path in output.parents or output in path.parents for path in protected):
        parser.error("Research output must be separate from production/data/settings")
    if (output / "RESULTS.csv").exists() or (output / "FROZEN_PLAN.json").exists():
        parser.error("Use a fresh output directory; an existing plan/results must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    before = production_signature()
    protocol, catalog = synthetic_protocol(output / "synthetic_input")
    cfg = TrainConfig(epochs=20, embedding_dim=16, batch_size=128, n_neg=3, lr=.003, early_stop=False)

    def freeze(plan):
        (output / "FROZEN_PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    def progress(phase, runs, summary):
        print(f"{phase}: paired seeds completed; users={runs[0]['evaluated_users']}", flush=True)

    result = run_experiment(protocol, cfg, catalog, catalog, mode=args.mode, dataset_kind="SYNTHETIC",
                             on_plan=freeze, on_phase=progress)
    after = production_signature()
    if before != after:
        raise FeatureExperimentError("Production model changed during research; review external activity")
    (output / "MODEL_PROTECTION.json").write_text(json.dumps({"before": before, "after": after, "unchanged": True}, indent=2), encoding="utf-8")
    write_results(output, result)
    print("BPR FEATURE AB READY WITH LIMITATIONS — synthetic only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
