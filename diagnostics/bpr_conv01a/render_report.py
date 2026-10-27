"""Render completed CONV-01A aggregates with the separate bundled report runtime."""
import csv
import json
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Group, Rect, String
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent


def plot(name, title, series, labels, ylabel, decimals=4):
    drawing = Drawing(760, 330)
    drawing.add(Rect(0, 0, 760, 330, fillColor=colors.white, strokeColor=None))
    drawing.add(String(80, 308, title, fontSize=15))
    chart = LinePlot()
    chart.x, chart.y, chart.width, chart.height = 80, 55, 620, 220
    chart.data = series
    chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 1, 30
    chart.xValueAxis.valueSteps = [1, 5, 10, 15, 20, 25, 30]
    values = [y for line in series for x, y in line]
    lo, hi = min(values), max(values)
    padding = max((hi - lo) * .08, .0001)
    chart.yValueAxis.valueMin, chart.yValueAxis.valueMax = max(0., lo - padding), hi + padding
    chart.yValueAxis.labelTextFormat = f"%0.{decimals}f"
    chart.xValueAxis.labels.fontSize = chart.yValueAxis.labels.fontSize = 10
    palette = [colors.HexColor(c) for c in ("#2563eb", "#db5a27", "#178557")]
    for i, label in enumerate(labels):
        chart.lines[i].strokeColor = palette[i]
        chart.lines[i].strokeWidth = 2
        drawing.add(String(80 + i * 205, 288, label, fontSize=11, fillColor=palette[i]))
    drawing.add(chart)
    drawing.add(String(375, 23, "Epoch", fontSize=11))
    axis_title = Group(String(0, 0, ylabel, fontSize=10))
    axis_title.translate(18, 135)
    axis_title.rotate(90)
    drawing.add(axis_title)
    renderSVG.drawToFile(drawing, str(ROOT / name))


def main():
    report = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    if report["status"] != "complete":
        raise RuntimeError("Only completed research runs can be rendered")
    run, rows = report["run"], report["trajectory"]
    checkpoints = [r for r in rows if "validation" in r]
    fields = ["epoch", "loss", "training_seconds", "cumulative_training_seconds", "scoring_seconds",
              "ndcg10_delta", "ndcg10_relative_delta"]
    fields += [f"overall_{metric}{k}" for k in (5, 10, 20) for metric in ("ndcg", "recall")]
    fields += [f"{s}_{m}10" for s in ("VIEW", "PURCHASE", "FAVORITE") for m in ("ndcg", "recall")]
    with (ROOT / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {k: row[k] for k in fields if k in row}
            if "validation" in row:
                validation = row["validation"]
                record["scoring_seconds"] = validation["scoring_seconds"]
                for slice_name, ks in validation["metrics"].items():
                    for k, cell in ks.items():
                        for metric in ("ndcg", "recall"):
                            record[f"{slice_name}_{metric}{k}"] = cell[metric]
            writer.writerow(record)
    plot("loss.svg", "CONV-01A: historical baseline training loss", [[(r["epoch"], r["loss"]) for r in rows]],
         ["Training loss; all 30 epochs"], "Weighted BPR loss")
    for metric in ("ndcg", "recall"):
        series = [[(r["epoch"], r["validation"]["metrics"][s]["10"][metric]) for r in checkpoints]
                  for s in ("overall", "VIEW", "PURCHASE")]
        plot(f"{metric}10.svg", f"CONV-01A: temporal validation {metric.upper()}@10",
             series, ["Overall (4976)", "VIEW targets (3257)", "PURCHASE targets (1718)"], f"{metric.upper()}@10")
    plot("memory.svg", "CONV-01A: CUDA memory after validation", [
        [(r["epoch"], r["validation"]["cuda_memory_after_scoring"][k] / 2**20) for r in checkpoints]
        for k in ("allocated_bytes", "reserved_bytes")], ["Allocated", "Reserved"], "CUDA memory (MiB)", 1)
    cfg = report["hyperparameters"]
    lines = ["# CONV-01A — завершённый validation-only baseline", "",
             "Ровно один непрерывный trainer/Adam lifecycle, seed 42, 30 эпох. После эпохи 30 run остановлен.",
             "Production publication и test scoring не выполнялись. Лучший checkpoint не восстанавливался: возвращено final epoch-30 state.", "",
             "Primary до запуска: overall validation NDCG@10. Один seed позволяет оценить диапазон, но не universal optimum.", "",
             "## Frozen config", "", "```json", json.dumps(cfg, ensure_ascii=False, indent=2), "```", "",
             "Optimizer: torch.optim.Adam. Только budget изменён: 200 → 30; остальные trainer parameters взяты из прежнего frozen config.",
             "VIEW/FAVORITE/PURCHASE = 0.1/2/10; batch size 256. Features и early stopping disabled.", "",
             "Training history: timestamp < 2025-11-01T00:00:00Z; validation future: [2025-11-01T00:00:00Z, 2025-12-01T00:00:00Z). min_history_events=10.",
             "Существующий protocol builder не изменён. Создаваемый им test object сразу отбрасывается; он не подготавливается, не передаётся worker/scoring API и не оценивается.",
             "Runner и scoring callback проверяют validation cutoff/future_end до вызова evaluator. Targets остаются external; internal evaluation пустая.", "",
             f"Users/items/aggregated train pairs: {report['training_users']}/{report['training_items']}/{report['training_pairs']}. Training events: {report['training_events']}.",
             f"Validation cases: {report['validation_cases']}; slices: {json.dumps(report['validation_slices'])}.",
             f"Candidates per case: {json.dumps(report['candidate_counts'])}; идентичны на всех checkpoints.",
             "FAVORITE имеет ровно один case: цифры приведены только для диагностики, выводы по этому slice не делаются.", "",
             "## Environment и provenance", "",
             f"Python executable: `{report['sys_executable']}`. Torch: {report['torch_version']}; CUDA build: {report['torch_cuda_build']}; GPU: {report['gpu']}; training/scoring device: {run['device']}.",
             f"Branch: {report['provenance']['git_branch']}; HEAD: {report['provenance']['git_head']}; dirty: {report['provenance']['working_tree_dirty']}.",
             f"Canonical revision: {report['canonical_source']['canonical_revision']}.",
             f"Dataset fingerprint SHA-256: `{report['dataset_sha256']}`.",
             f"Mappings/positive-pair fingerprint SHA-256: `{report['mapping_sha256']}` (совпадает с historical CPU/GPU cost pilot).",
             "Source SHA-256, temporal config и all hyperparameters: frozen_config.json. Fingerprints не содержат открытых customer/item identifiers.", "",
             "## Overall temporal validation", "",
             "| Epoch | NDCG@5 | Recall@5 | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 | Δ NDCG@10 | Relative Δ | Scoring sec |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in checkpoints:
        m = r["validation"]["metrics"]["overall"]
        cells = [f"{m[str(k)][metric]:.10f}" for k in (5, 10, 20) for metric in ("ndcg", "recall")]
        delta = "—" if r["ndcg10_delta"] is None else f"{r['ndcg10_delta']:+.10f}"
        rel = "—" if r["ndcg10_relative_delta"] is None else f"{100*r['ndcg10_relative_delta']:+.3f}%"
        lines.append(f"| {r['epoch']} | " + " | ".join(cells) + f" | {delta} | {rel} | {r['validation']['scoring_seconds']:.3f} |")
    lines += ["", "## Target slices", "",
              "| Epoch | VIEW NDCG@10 | VIEW Recall@10 | PURCHASE NDCG@10 | PURCHASE Recall@10 | FAVORITE NDCG@10 | FAVORITE Recall@10 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in checkpoints:
        m = r["validation"]["metrics"]
        cells = [f"{m[s]['10'][metric]:.10f}" for s in ("VIEW", "PURCHASE", "FAVORITE") for metric in ("ndcg", "recall")]
        lines.append(f"| {r['epoch']} | " + " | ".join(cells) + " |")
    lines += ["", "## Loss и training time каждой эпохи", "",
              "| Epoch | Loss | Training sec | Cumulative training sec |", "|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['epoch']} | {r['loss']:.10f} | {r['training_seconds']:.3f} | {r['cumulative_training_seconds']:.3f} |")
    lines += ["", f"Training-only: {run['training_seconds']:.3f} sec ({run['training_seconds']/60:.3f} min). External validation: {run['evaluation_seconds']:.3f} sec. Worker wall: {report['worker_wall_seconds']:.3f} sec.",
              f"Ingestion: {report['canonical_ingestion_seconds']:.3f} sec; snapshot: {report['snapshot_seconds']:.3f} sec; preparation: {report['preparation_seconds']:.3f} sec.",
              "CUDA synchronization ограничивает timing intervals. Epoch 1 включает model/Adam initialization. Callback scoring/bookkeeping исключены из training time следующей эпохи.",
              "Loss берётся без изменений из existing TrainingEpochMetrics; ranking/metric mathematics выполняет existing temporal evaluator.", "",
              "## CUDA memory", "",
              f"Run peak allocated: {run['cuda_memory']['peak_allocated_bytes']/2**20:.3f} MiB; peak reserved: {run['cuda_memory']['peak_reserved_bytes']/2**20:.3f} MiB. Scope: training + repeated external scoring.",
              "| Epoch | Before allocated MiB | Before reserved MiB | After allocated MiB | After reserved MiB |",
              "|---:|---:|---:|---:|---:|"]
    for r in checkpoints:
        before, after = r["cuda_memory_before_scoring"], r["validation"]["cuda_memory_after_scoring"]
        cells = [f"{d[k]/2**20:.3f}" for d in (before, after) for k in ("allocated_bytes", "reserved_bytes")]
        lines.append(f"| {r['epoch']} | " + " | ".join(cells) + " |")
    lines += ["", "Reserved отражает caching allocator, allocated — живые CUDA allocations. Рост reserved сам по себе не доказывает утечку; важна последовательность after-scoring counters.",
              f"Worker OS process memory: `{json.dumps(run['process_memory'])}`.", "",
              "## Проверки и scope", "",
              "Targeted tests: 39 passed, 2 warnings. Evaluation/model: 690 passed, 106 warnings. Full pytest: 2061 passed, 112 warnings. Ruff и git diff --check passed.",
              "CPU no-observer/noop/read-only observation дают bit-exact одинаковый final state. Synthetic checkpoints 1/3/5 оценены только в этом порядке; Adam instance один; returned model совпадает с последней эпохой. Actual CUDA synthetic final-state test passed.",
              "Sampler, model class, feature builder, inner training loop и Adam initializer AST-identical предыдущему состоянию: math_unchanged.json. TRAIN-02F semantics сохранены.",
              "Ни второго seed, ни batch/weight variants, ни CONV-01B, ни Stage B, ни test scoring, ни публикации, ни commit/push не выполнялось.",
              "Защита canonical data/model/settings: protected_files_before.json и protected_files_after.json. Scope code changes: task_changes.json; git status: git_status.txt.", "",
              "Артефакты: frozen_config.json, results.json, trajectory.json/csv, loss.svg, ndcg10.svg, recall10.svg, memory.svg; full-precision данные находятся в JSON/CSV."]
    analysis_path = ROOT / "analysis.json"
    if analysis_path.exists():
        lines += ["", "## Интерпретация и provisional budget", "", *json.loads(
            analysis_path.read_text(encoding="utf-8"))["summary_lines"]]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
