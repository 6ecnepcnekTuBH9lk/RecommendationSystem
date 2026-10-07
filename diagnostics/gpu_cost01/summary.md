# GPU-COST-01: completed

Выполнен ровно один real-data training run, одна эпоха, seed 42, на `cuda:0` / NVIDIA GeForce RTX 3050. Это cost/sanity measurement; validation/test NDCG и Recall не вычислялись. Модель не публиковалась и checkpoint не сохранялся.

Training history: timestamp строго меньше `2025-11-01T00:00:00Z`. Canonical revision `af24df53c47349d0ae3170ab396df726`; counts и mapping/pairs fingerprint совпали с историческим CPU pilot.

## Exact execution configuration

Из historical `diagnostics/bpr_exp01_baseline_cost/frozen_config.json` скопированы все hyperparameters, изменён только `epochs: 200 → 1`. Полный config и source provenance находятся в `frozen_config.json` и `results.json` этой директории.

| Parameter | Value |
|---|---|
| VIEW / FAVORITE / PURCHASE | 0.1 / 2.0 / 10.0 |
| seed / epochs | 42 / 1 |
| embedding_dim / batch_size / n_neg | 128 / 256 / 10 |
| optimizer / lr | torch.optim.Adam / 0.0003 |
| bpr_reg / weight_decay | 0.0005 / 0.0 |
| early_stop / item features | False / disabled |
| early_stop metric / patience / min_delta / min_epochs | ndcg / 8 / 0.0005 / 30; inactive |
| topk / min_user_interactions_for_eval | 10 / 10; external quality evaluation not called |
| quantity / repeat aggregation / date mode | Existing unchanged contract / FULL_TIMESTAMP |

## Measurements

| Measurement | Value |
|---|---|
| Training users | 155669 |
| Training items | 5201 |
| Aggregated training pairs | 738640 |
| Canonical ingestion | 435.513858 seconds |
| Temporal snapshot construction | 10.551960 seconds |
| BPR preparation | 171.834691 seconds |
| One-epoch trainer wall-clock | 84.876358 seconds |
| Worker wall-clock including startup/exit | 87.695486 seconds |
| Optimizer steps, counted | 2886 |
| Sampled positive examples processed, counted | 738816 |
| Throughput | 8704.614766 sampled positive pairs/sec |
| Worker CPU time during training | 83.125 seconds |
| Peak CUDA allocated | 413211136 bytes / 394.068848 MiB |
| Peak CUDA reserved | 432013312 bytes / 412 MiB |
| Worker peak process RAM working set | 1132412928 bytes / 1079.953125 MiB |
| Parent peak process RAM working set | 1908490240 bytes / 1820.078125 MiB |
| Finite loss and model parameters | True |
| Completed epochs | 1 |

The sampled-example count is not a unique-pair count. The existing trainer samples 256 pairs per step for 2886 steps; pairs can repeat across batches. Sampler and Adam counters delegate to the original implementations without changing arithmetic or RNG.

Wall-clock uses high-resolution `time.perf_counter()` with CUDA synchronization before and after training. It includes trainer/model/optimizer setup. CUDA context initialization and worker startup are outside the trainer interval. PyTorch peak counters are reset in a fresh worker and read before the subsequent finite-parameter sanity scan. RAM counters refer separately to parent and worker, not a sampled combined-process peak.

## CPU comparison and linear references

CPU pilot first three approximate epochs: 370 / 472 / 505 seconds (6:10 / 7:52 / 8:25). Representative median: 472 seconds. Approximate speedup: **5.561030×**; ratios against the three CPU observations range from 4.359282× to 5.949831×. These are early epochs of an interrupted CPU pilot, not a complete comparative benchmark. CPU preparation was approximately 174.7 seconds; current preparation is 171.835 seconds.

| Epochs | Linear training-only reference |
|---|---|
| 20 | 1697.527162 seconds / 28.292119 minutes |
| 50 | 4243.817905 seconds / 70.730298 minutes |
| 200 | 16975.271620 seconds / 4.715353 hours |

These references multiply one measured epoch by the requested count. They exclude ingestion, snapshot construction and preparation and do not predict convergence, quality or actual sustained throughput. None of these epoch budgets was run.

CPU time remains substantial. The unchanged implementation performs NumPy sampling on CPU, creates/transfers batch tensors and synchronizes through per-step `loss.detach().cpu()`. Dense Adam still updates the complete embedding tables. These are concrete remaining costs in the code; their separate time shares were not profiled. No optimization or deterministic-setting changes were made.

`gpu_after_training.csv` was captured after completion and cannot describe utilization during the epoch. `run.log` preserves the single epoch's console output. `results.json` contains the complete measurement, environment and provenance.

Validation/test quality scoring, weight sweep, additional seeds, Stage A, Stage B and production publication were not executed. Protected files matched SHA256 baselines (23/23), and no training workers remained. Commit/push were not executed. Stop here for user review; CONV-01 is not started.
