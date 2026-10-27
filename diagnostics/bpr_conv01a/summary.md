# CONV-01A — завершённый validation-only baseline

Ровно один непрерывный trainer/Adam lifecycle, seed 42, 30 эпох. После эпохи 30 run остановлен.
Production publication и test scoring не выполнялись. Лучший checkpoint не восстанавливался: возвращено final epoch-30 state.

Primary до запуска: overall validation NDCG@10. Один seed позволяет оценить диапазон, но не universal optimum.

## Frozen config

```json
{
  "data_dir": "ВходныеДанные",
  "w_view_item": 0.1,
  "w_favorite": 2.0,
  "w_purchase": 10.0,
  "embedding_dim": 128,
  "epochs": 30,
  "batch_size": 256,
  "lr": 0.0003,
  "weight_decay": 0.0,
  "bpr_reg": 0.0005,
  "n_neg": 10,
  "seed": 42,
  "topk": 10,
  "min_user_interactions_for_eval": 10,
  "early_stop": false,
  "early_stop_metric": "ndcg",
  "early_stop_patience": 8,
  "early_stop_min_delta": 0.0005,
  "early_stop_min_epochs": 30,
  "use_item_features": false,
  "item_feature_cols": [
    "ВидНоменклатуры",
    "ВидАссортимента",
    "Марка",
    "Коллекция",
    "СезонНоски",
    "ПолНоменклатуры",
    "ГруппаСоставов",
    "КатегорияНаСайте",
    "СтилеваяГруппа"
  ],
  "max_item_features": 32,
  "feature_dropout": 0.1,
  "feature_scale": 0.2,
  "feature_norm": "mean",
  "feat_reg_mult": 1.0
}
```

Optimizer: torch.optim.Adam. Только budget изменён: 200 → 30; остальные trainer parameters взяты из прежнего frozen config.
VIEW/FAVORITE/PURCHASE = 0.1/2/10; batch size 256. Features и early stopping disabled.

Training history: timestamp < 2025-11-01T00:00:00Z; validation future: [2025-11-01T00:00:00Z, 2025-12-01T00:00:00Z). min_history_events=10.
Существующий protocol builder не изменён. Создаваемый им test object сразу отбрасывается; он не подготавливается, не передаётся worker/scoring API и не оценивается.
Runner и scoring callback проверяют validation cutoff/future_end до вызова evaluator. Targets остаются external; internal evaluation пустая.

Users/items/aggregated train pairs: 155669/5201/738640. Training events: 990054.
Validation cases: 4976; slices: {"VIEW": 3257, "FAVORITE": 1, "PURCHASE": 1718}.
Candidates per case: {"min": 4460, "max": 5199, "mean": 5171.96402733119}; идентичны на всех checkpoints.
FAVORITE имеет ровно один case: цифры приведены только для диагностики, выводы по этому slice не делаются.

## Environment и provenance

Python executable: `C:\Users\Ermolenko.i\PycharmProjects\РекомендательнаяСистема\.venv312\Scripts\python.exe`. Torch: 2.5.1+cu124; CUDA build: 12.4; GPU: NVIDIA GeForce RTX 3050; training/scoring device: cuda:0.
Branch: feature/training-settings; HEAD: cc1b6fa4bdda06f359e6a581e8a6723158d74492; dirty: True.
Canonical revision: af24df53c47349d0ae3170ab396df726.
Dataset fingerprint SHA-256: `bd69ed9299c51409cc9e27da6cdb2e6324a6b42b1287033de15e84b4db1f5b58`.
Mappings/positive-pair fingerprint SHA-256: `aeaccd129eab39c1b3b4fbb3558717afe3c45e46fac133da58130c146a694976` (совпадает с historical CPU/GPU cost pilot).
Source SHA-256, temporal config и all hyperparameters: frozen_config.json. Fingerprints не содержат открытых customer/item identifiers.

## Overall temporal validation

| Epoch | NDCG@5 | Recall@5 | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 | Δ NDCG@10 | Relative Δ | Scoring sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0004318152 | 0.0008038585 | 0.0007400989 | 0.0018086817 | 0.0013834311 | 0.0044212219 | — | — | 46.205 |
| 2 | 0.0013756787 | 0.0026125402 | 0.0016962561 | 0.0036173633 | 0.0029065929 | 0.0084405145 | +0.0009561572 | +129.193% | 47.319 |
| 3 | 0.0031935305 | 0.0050241158 | 0.0040827193 | 0.0078376206 | 0.0058962163 | 0.0150723473 | +0.0023864632 | +140.690% | 49.245 |
| 5 | 0.0054703397 | 0.0080385852 | 0.0068547556 | 0.0122588424 | 0.0091688988 | 0.0215032154 | +0.0027720363 | +67.897% | 49.644 |
| 8 | 0.0067702922 | 0.0100482315 | 0.0081764208 | 0.0144694534 | 0.0103084527 | 0.0229099678 | +0.0013216651 | +19.281% | 48.783 |
| 12 | 0.0065868328 | 0.0098472669 | 0.0082822025 | 0.0150723473 | 0.0108346428 | 0.0253215434 | +0.0001057817 | +1.294% | 51.384 |
| 20 | 0.0076458313 | 0.0114549839 | 0.0093057109 | 0.0166800643 | 0.0119759516 | 0.0273311897 | +0.0010235084 | +12.358% | 45.045 |
| 30 | 0.0074210492 | 0.0112540193 | 0.0098401566 | 0.0188906752 | 0.0133080592 | 0.0325562701 | +0.0005344458 | +5.743% | 47.780 |

## Target slices

| Epoch | VIEW NDCG@10 | VIEW Recall@10 | PURCHASE NDCG@10 | PURCHASE Recall@10 | FAVORITE NDCG@10 | FAVORITE Recall@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0008182221 | 0.0021492171 | 0.0005924229 | 0.0011641444 | 0.0000000000 | 0.0000000000 |
| 2 | 0.0010021259 | 0.0018421861 | 0.0030131817 | 0.0069848661 | 0.0000000000 | 0.0000000000 |
| 3 | 0.0012078005 | 0.0030703101 | 0.0095353930 | 0.0168800931 | 0.0000000000 | 0.0000000000 |
| 5 | 0.0011860228 | 0.0021492171 | 0.0176055809 | 0.0314318976 | 0.0000000000 | 0.0000000000 |
| 8 | 0.0013332817 | 0.0024562481 | 0.0211544653 | 0.0372526193 | 0.0000000000 | 0.0000000000 |
| 12 | 0.0016564138 | 0.0027632791 | 0.0208482537 | 0.0384167637 | 0.0000000000 | 0.0000000000 |
| 20 | 0.0023294733 | 0.0046054652 | 0.0225367420 | 0.0395809080 | 0.0000000000 | 0.0000000000 |
| 30 | 0.0030120759 | 0.0067546822 | 0.0227906218 | 0.0419091967 | 0.0000000000 | 0.0000000000 |

## Loss и training time каждой эпохи

| Epoch | Loss | Training sec | Cumulative training sec |
|---:|---:|---:|---:|
| 1 | 0.6827613862 | 95.483 | 95.483 |
| 2 | 0.6398189523 | 90.654 | 186.137 |
| 3 | 0.5693881372 | 90.303 | 276.440 |
| 4 | 0.4597331788 | 91.980 | 368.419 |
| 5 | 0.3322829719 | 91.128 | 459.547 |
| 6 | 0.2280044731 | 92.499 | 552.046 |
| 7 | 0.1605257423 | 93.899 | 645.945 |
| 8 | 0.1200273611 | 91.306 | 737.251 |
| 9 | 0.0937038830 | 89.401 | 826.652 |
| 10 | 0.0770167937 | 90.238 | 916.890 |
| 11 | 0.0650271665 | 92.301 | 1009.191 |
| 12 | 0.0562956116 | 86.552 | 1095.743 |
| 13 | 0.0499446057 | 90.281 | 1186.024 |
| 14 | 0.0452477891 | 93.140 | 1279.164 |
| 15 | 0.0413849503 | 92.231 | 1371.395 |
| 16 | 0.0382321314 | 91.700 | 1463.094 |
| 17 | 0.0356950211 | 91.283 | 1554.377 |
| 18 | 0.0335203974 | 91.923 | 1646.301 |
| 19 | 0.0316745354 | 91.577 | 1737.878 |
| 20 | 0.0302552038 | 85.912 | 1823.790 |
| 21 | 0.0289448822 | 83.588 | 1907.377 |
| 22 | 0.0276999240 | 84.645 | 1992.023 |
| 23 | 0.0266354587 | 86.336 | 2078.359 |
| 24 | 0.0257016575 | 87.036 | 2165.395 |
| 25 | 0.0249247245 | 92.878 | 2258.273 |
| 26 | 0.0242271284 | 90.029 | 2348.303 |
| 27 | 0.0235611569 | 90.410 | 2438.713 |
| 28 | 0.0229892516 | 93.427 | 2532.139 |
| 29 | 0.0224556673 | 92.464 | 2624.603 |
| 30 | 0.0220096176 | 91.139 | 2715.742 |

Training-only: 2715.742 sec (45.262 min). External validation: 385.405 sec. Worker wall: 3118.818 sec.
Ingestion: 475.498 sec; snapshot: 11.381 sec; preparation: 191.314 sec.
CUDA synchronization ограничивает timing intervals. Epoch 1 включает model/Adam initialization. Callback scoring/bookkeeping исключены из training time следующей эпохи.
Loss берётся без изменений из existing TrainingEpochMetrics; ranking/metric mathematics выполняет existing temporal evaluator.

## CUDA memory

Run peak allocated: 394.069 MiB; peak reserved: 412.000 MiB. Scope: training + repeated external scoring.
| Epoch | Before allocated MiB | Before reserved MiB | After allocated MiB | After reserved MiB |
|---:|---:|---:|---:|---:|
| 1 | 315.519 | 412.000 | 315.519 | 412.000 |
| 2 | 315.519 | 412.000 | 315.519 | 412.000 |
| 3 | 315.519 | 412.000 | 315.519 | 412.000 |
| 5 | 315.519 | 412.000 | 315.519 | 412.000 |
| 8 | 315.519 | 412.000 | 315.519 | 412.000 |
| 12 | 315.519 | 412.000 | 315.519 | 412.000 |
| 20 | 315.519 | 412.000 | 315.519 | 412.000 |
| 30 | 315.519 | 412.000 | 315.519 | 412.000 |

Reserved отражает caching allocator, allocated — живые CUDA allocations. Рост reserved сам по себе не доказывает утечку; важна последовательность after-scoring counters.
Worker OS process memory: `{"scope": "this process only", "current_working_set_bytes": 2164932608, "peak_working_set_bytes": 2792931328, "peak_commit_bytes": 3687235584}`.

## Проверки и scope

Targeted tests: 39 passed, 2 warnings. Evaluation/model: 690 passed, 106 warnings. Full pytest: 2061 passed, 112 warnings. Ruff и git diff --check passed.
CPU no-observer/noop/read-only observation дают bit-exact одинаковый final state. Synthetic checkpoints 1/3/5 оценены только в этом порядке; Adam instance один; returned model совпадает с последней эпохой. Actual CUDA synthetic final-state test passed.
Sampler, model class, feature builder, inner training loop и Adam initializer AST-identical предыдущему состоянию: math_unchanged.json. TRAIN-02F semantics сохранены.
Ни второго seed, ни batch/weight variants, ни CONV-01B, ни Stage B, ни test scoring, ни публикации, ни commit/push не выполнялось.
Защита canonical data/model/settings: protected_files_before.json и protected_files_after.json. Scope code changes: task_changes.json; git status: git_status.txt.

Артефакты: frozen_config.json, results.json, trajectory.json/csv, loss.svg, ndcg10.svg, recall10.svg, memory.svg; full-precision данные находятся в JSON/CSV.

## Интерпретация и provisional budget

Среди восьми проверенных checkpoints epoch 30 имеет максимальный заранее выбранный primary NDCG@10: 0.009840156621554253. Provisional candidate — 30 epochs; разумный текущий исследовательский диапазон — 20–30. Никакие production/research defaults автоматически не изменены.
20 → 30: NDCG@10 0.009305710852445196 → 0.009840156621554253; абсолютная Δ +0.0005344457691090574, relative Δ +5.7432%. Recall@10 вырос с 83/4976 до 94/4976 (+11 попаданий). Дополнительные десять эпох стоили 891.9523 sec training (~14.866 min).
VIEW NDCG@10 вырос с 0.002329473275495246 до 0.003012075854587713 (~+29.30%); VIEW Recall@10 — с 15/3257 до 22/3257. PURCHASE NDCG@10 вырос с 0.022536741992712026 до 0.02279062182215471 (~+1.13%); PURCHASE Recall@10 — с 68/1718 до 72/1718.
Loss 20 → 30: 0.030255203787818274 → 0.022009617568988272. Кривая primary metric замедляется, но продолжает расти; plateau к epoch 30 не подтверждён. FAVORITE slice содержит один case и не используется для этих выводов.
Не все secondary metrics растут: NDCG@5 20 → 30 снизился 0.00764583132568965 → 0.007421049166765873; Recall@5 — с 57/4976 до 56/4976. Provisional выбор 30 соответствует заранее закреплённому NDCG@10, а не универсальному улучшению всех K.
Рекомендация для отдельного review — CONV-01B с ограниченным исследованием до 40/50 epochs, затем при необходимости подтверждение несколькими seeds. Это описательное наблюдение одного seed, без заявления статистической значимости или universal optimum. Evidence необходимости 200 epochs сейчас нет; достаточность 30 как финального budget также не доказана.
В этом run финальный model/optimizer checkpoint на диск не сохранялся: artifacts — траектория и aggregates. Продолжение с точным сохранением optimizer/RNG state не подготовлено; следующий continuous run до 40/50 потребует отдельного решения и исполнения, без скрытого автоматического resume.
Training-only 2715.7420008 sec = 45 min 15.742 sec; average 90.5247 sec/epoch. Это на 6.6548% дольше линейного ориентира GPU-COST-01 (2546.290743 sec для 30 эпох), без резкого изменения порядка стоимости. Scoring total 385.4054933 sec = 6 min 25.405 sec; worker wall 3118.8177392 sec = 51 min 58.818 sec.
На всех восьми checkpoints after-scoring allocated = 330845696 bytes, reserved = 432013312 bytes (412 MiB). Ни current allocated, ни reserved не растут; наблюдаемого удержания CUDA tensors между repeated validation calls нет. Run peak allocated = 413211136 bytes (394.068848 MiB); peak reserved = 412 MiB.
Training worker, parent и launcher завершены: processes_after.json содержит 0 tracked processes; дополнительная CIM-проверка experiment/spawn processes пуста. SHA-256 23 canonical/model/settings files полностью совпадает с исходным baseline. 104 prior Application/scripts source files сверены с GPU-COST-01 frozen provenance: единственный изменённый prior source — BPRMF.py; его before-CONV копия точно совпадает с reference hash.
CONV-01A scope: Application/model/BPRMF.py — optional epoch observer, default None, torch.no_grad и restoration of mode; scripts/run_bpr_convergence.py — guarded single continuous run и validation-only timing/trajectory; tests/evaluation/test_bpr_convergence.py — 18 synthetic observer/runner regression tests; diagnostics/bpr_conv01a/render_report.py — report/plots renderer в отдельном bundled runtime.
Artifacts сохранены в diagnostics/bpr_conv01a/: frozen_config.json, results.json, trajectory.json/csv, summary.md, analysis.json, environment.json, tests_before_run.json, targeted/evaluation_model/full_pytest logs, run.log, math_unchanged.json, code_before.py.txt, observer.diff.txt, protected_files_before/after.json, processes_after.json, task_changes.json, git_status.txt и loss/ndcg10/recall10/memory SVG/PNG. Все эти artifacts завершены; старые interrupted CPU pilot artifacts не изменены и не удалены.
Остановились после epoch 30. Test scoring, production publication, другой seed, weight/batch variants, Stage B, commit и push не выполнялись. Следующий experiment ожидает отдельного решения пользователя.
