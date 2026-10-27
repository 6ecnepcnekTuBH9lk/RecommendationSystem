# BPR-EXP-01 controlled research runner

The full real-data sweep was stopped before training and produced no experiment outputs. The current plan is exactly one baseline run (0.1/2/10, seed 42, 200 epochs), followed by a user decision. Run from the repository root using the existing environment:

```powershell
.\.venv312\Scripts\python.exe -u scripts/run_bpr_weight_experiment.py --mode baseline
```

The default CLI mode is `baseline`: exactly one run, no model selection and no automatic follow-up. Outputs go to `diagnostics/bpr_exp01_baseline_cost/`. The preserved full sweep requires explicit `--mode sweep` and is not authorized for the current execution. OS process counters provide worker peak/current working set and commit usage; these counters do not measure the combined parent/worker peak. Wall-clock training excludes canonical loading, preparation and validation. Linear estimates report 3, 18, 30 unique and 33 logical runs; throughput and overhead may vary.

The runner reads canonical Mindbox exports using the TRAIN-03A loader, resolves customers/products with existing adapters and supplies only `TemporalProtocol.validation` to TRAIN-02E BPR preparation and evaluation. The test snapshot is discarded without scoring. Existing product-less Actions remain excluded. No new data semantics, dependencies or production integration are introduced.

Fixed UTC cutoffs: validation 2025-11-01, test start 2025-12-01, test end 2025-12-31; minimum historical event count 10. Validation targets are first novel warm items from November. Every historical event before November remains in training; internal evaluation arrays are empty. Item features and early stopping are explicitly off.

Existing `user_settings/train_config.json` values fill `TrainConfig`, with defaults for absent fields. Epochs are the existing 200; no budget tuning. Seeds are 42, 43 and 44 for every configuration. Device follows existing CUDA availability logic (this environment is CPU-only). Initialization calls the existing `_set_seed` before training. Optimizer, sampling, normalized weighted loss, quantity clip and repeat sum all remain in existing code.

Stage A sweeps VIEW 0.05/0.1/0.25/0.5/1/2 with FAVORITE=2, PURCHASE=10. Stage B begins only after all Stage A seeds finish, sweeps FAVORITE 0.5/1/2/5/10 with the Stage A representative VIEW and PURCHASE=10. The identical FAVORITE=2 configuration reuses its three exact Stage A runs: 30 unique training runs, 33 logical runs.

Primary selection uses full-precision mean overall validation NDCG@10. Sample standard deviation uses ddof=1. A configuration belongs to the descriptive tie set when its gap to the maximum is no larger than the larger of the two seed standard deviations. Among this set choose the swept weight nearest the historical value in absolute log ratio, then the smaller weight. This conservative rule is frozen before seeing metrics; it is not a statistical significance test. Recall@10, @5/@20 and VIEW/PURCHASE/FAVORITE slices are diagnostics. The real FAVORITE slice has one case and cannot support tuning conclusions.

Outputs are isolated in `diagnostics/bpr_exp01_weights/`: `frozen_config.json` before training, incremental `results.json`, and `summary.md`. Source digests supplement dirty Git HEAD; reports contain aggregates, hashes and configuration, never raw customer/product identifiers. Exact existing purchase confidence units and weight masses are computed before training, and aggregated pair quantiles before each configuration. Shares are diagnostic and never renormalize training weights.

Each seed trains in a fresh spawned process. This avoids the existing repeated `torch.set_num_interop_threads` limitation without modifying trainer behavior. Models live only in memory. Spawn serializes validation data internally to the child, never test targets. No model checkpoint, `model/current.json`, publication or settings write is performed. Existing output directories with `results.json` are rejected to preserve evidence. Interrupted output is explicitly incomplete; never infer results from missing runs.

TRAIN-02F state selection: `early_stop=False` returns the final executed epoch, irrespective of internal evaluation. With early stopping enabled and nonempty internal evaluation, the trainer monitors the configured metric and restores an independent copy of the best state; patience, minimum epochs and minimum delta are unchanged. Empty internal evaluation disables monitoring and stopping and returns the final epoch. `best_epoch`, `best_recall` and `best_ndcg` are then `-1` (no observation); per-epoch zero metric fields remain compatibility placeholders, not quality measurements. External temporal validation evaluates the returned final model. Synthetic regression tests cover both CPU storage aliases and emulated CUDA-to-CPU copy behavior, without requiring GPU.

Dense Adam's full embedding updates can make the prescribed CPU budget expensive; this experiment does not replace Adam or reduce epochs. Historical as-of catalog semantics remain unresolved and features stay disabled.
