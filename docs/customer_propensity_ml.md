# Independent order-placement baseline (CUSTOMER-PROPENSITY-02B)

Opt-in research only. `Application/model/propensity_ml/` does not integrate with
GUI, BPR-MF, hybrid recommendations or dataset builders. Target is existence of a
logical order in `[t,t+30d)`, regardless of payment/status/cancellation/return.
`RETROSPECTIVE_EVENT_TIME` and `REQUIRES_BUSINESS_VALIDATION` remain mandatory.

## Environment

Python 3.12, existing NumPy 2.2.6; optional research dependencies:

```powershell
.venv312\Scripts\python.exe -m pip install -r requirements-propensity-ml.txt
.venv312\Scripts\python.exe -m pip check
```

The four pinned packages do not replace the application dependency file. Do not
upgrade NumPy/PyTorch/PyQt for this experiment. An installation report with wheel
hashes and before/after package metadata belongs to the research run diagnostics.

## Preflight and explicit launch

```powershell
.venv312\Scripts\python.exe scripts\train_customer_propensity.py --dataset <published-version-directory>
.venv312\Scripts\python.exe scripts\train_customer_propensity.py --train --allow-retrospective --allow-test --dataset <published-version-directory> --run-dir <new-owned-run-directory> --models-root filtered_data/customer_propensity_models --cancel-file <manual-cancel-marker>
```

Without `--train` this only inspects metadata/resources. There is no implicit
build, key generation, training, repeated experiment or mass inference. Existing
run paths are rejected. The canonical datasets and their CURRENT pointers are
never written. Read failures do not trigger regeneration. Never run two trainers:
the model root has an OS byte lock, released on every normal/error exit.

Cancellation is cooperative: SIGINT/SIGBREAK or creation of the supplied cancel
marker. Checks occur while reading and between batches/optimizer iterations.
No unrequested fixed wall-clock timeout; max 10 epochs and selection patience2
bound optimization. Progress and resources go to anonymous JSON lines. A scoring
checkpoint is not a full averaged-SGD optimizer resume state; no automatic restart.

## Frozen protocol

* Train: first33 dates, 2022-01-31 through2024-09-17; only fit source.
* Selection: next5 dates, 2024-10-17 through2025-02-14; epoch choice only.
* Calibration: next3 dates, 2025-03-16 through2025-05-15; frozen score sigmoid fit.
* Holdout: next3 dates, 2025-06-14 through2025-08-13; independent diagnostics.
* Test: last11 dates, 2025-09-12 through2026-07-09; payload opened after pipeline freeze only.

Half-open 30-day target intervals touch but do not overlap at these boundaries.
The manifest's embargo rules are retained, with zero embargo rows for this grid.
No refit on validation after calibration, no test-driven tuning. Holdout problems
are reported, not repaired by selecting a new method. Final test is consumed even
if the result is disappointing; a later experiment needs a new independent holdout.

## Features and numerical representation

Lexicographically ordered22 features from schema v2, plus22 unavailable flags,
plus4 semantic-null flags =48. Counts/active days: log1p; recency: observed null
maps to its window length, unavailable maps to0, divide by window. Mean interval:
exact source Decimal string is checked then approximated at the ML boundary;
null maps to0 plus a distinct flag, followed by log1p. Binary has fields remain0/1.
19 continuous fields use train-only StandardScaler (float64 statistics); batch
output float32. No clipping, class balancing or negative undersampling.
Order/type/schema errors fail explicitly. Unknown coverage never silently means
known absence. Known absence is within a lookback, not proof of lifetime absence.

Dataset reader verifies physical SHA, schema and counts/CRC upon exhaustion. It
feeds bounded batches (default50k) to owned versioned numerical `.npy` caches,
never a raw JSON copy or a whole DataFrame. Cache manifests bind dataset/schema/
preprocessing/role/dates/dtypes/shapes/checksums. Reuse must pass load_cache gates;
the first CLI experiment deliberately does not auto-reuse a previous cache.
Cached features are locally protected derivative personal data. New run/model
directories restrict ACLs to the user, SYSTEM and Administrators on Windows.

SGD: log_loss, L2 alpha1e-4, optimal learning rate, average=True, seed20261020,
natural class frequencies. Each epoch uses a seeded permutation of all train
indices; the final partial batch is retained. Best actual selection log loss
checkpoint is preserved; min_delta1e-5/patience2 control stopping. No hidden search.

Sigmoid calibration is explicit two-parameter Platt-smoothed log loss, on raw
logits only, with positive slope and finite/convergence checks. Its score
normalization is internal to calibration and exported back to `expit(a*z+b)`.
No random CV/base-model retraining. Metrics: noninterpolated AP, ROC-AUC, log loss,
Brier, fixed-bin reliability/ECE, top1/5/10%, pooled and per-date values, against
one constant fitted from actual verified train counts. Empty bins have null
statistics. Correlated repeated customers preclude iid confidence claims;
customer/bootstrap confidence intervals are not implemented in this baseline.

## Safety and persistence

Windows GetProcessMemoryInfo reads PrivateUsage (private commit), WorkingSetSize
and peaks separately. GlobalMemoryStatusEx reads physical/commit reserves. These
are periodic cooperative checks, not a hard OS allocation cap: a native operation
can allocate between polls. Defaults: private commit<=3GiB, available physical
>=4GiB, available commit>=3GiB, actual-volume free>=20GiB. Preflight additionally
budgets numerical cache growth. Volume identity follows mounted volumes/junctions.
Only the current UUID staging can be cleaned on failure; complete caches and older
models are preserved. Resource failure stops the experiment without retry.

Research models use strict numeric JSON, shape/type/finite checks, checksums and
immutable version directories. No pickle/joblib, HMAC secret, raw ID, raw events
or post-row predictions in the model. Private derived caches contain no raw IDs.
Checkpoints contain scoring coefficients, not the internal averaged optimizer.
Float64 per-row reduction keeps inference independent of batch size; export
parity against the fitted sklearn estimator is covered on synthetic data.
Loading verifies model target/schema/columns/preprocessing/calibrator and hashes.
Checksums detect corruption, not the authenticity of an untrusted author.

New-process synthetic-only validation (no real buyer inference):

```powershell
.venv312\Scripts\python.exe scripts\train_customer_propensity.py --validate-artifact <model-version-directory>
```

No production model pointer or GUI integration exists. Inference snapshot dated
2026-09-01 still means forecast interval[2026-09-01,2026-10-01), not today's forecast.
