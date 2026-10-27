# TRAIN-03A — canonical temporal dataset audit

Resolved events: 1206864; customers: 184976; items: 5545.
Observed UTC range: 2022-10-15T17:29:20.663000+00:00 — 2025-12-31T23:57:16.107000+00:00.
Declared canonical coverage: 2025-01-01T00:00:00+00:00 — 2026-01-01T00:00:00+00:00 (exclusive end).
Interaction types: {'VIEW': 610086, 'FAVORITE': 5111, 'PURCHASE': 591667}.
Resolved events outside declared range: {'PURCHASE': 1087} (retained, never silently deleted).
Tail policy: exclude last observed calendar day; completeness not independently proven.

Cold-user / sparse / no-novel rates use future users as denominator.
Cold-item rate uses history-eligible first-novel targets; PURCHASE rate uses final cases.

| Horizon | Min history | Validation window | Test window | Val future / cases | Test future / cases | Cold users V/T % | Cold items V/T % | No novel V/T % | PURCHASE V/T % | Median delay V/T days |
|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 10 | 2025-12-03 → 2025-12-17 | 2025-12-17 → 2025-12-31 | 14780 / 2909 | 20334 / 3530 | 45.78/49.45 | 2.68/0.34 | 2.56/1.78 | 37.16/44.99 | 5.52/5.57 |
| 30 | 10 | 2025-11-01 → 2025-12-01 | 2025-12-01 → 2025-12-31 | 24951 / 4976 | 33661 / 5341 | 45.15/51.61 | 1.56/3.21 | 1.45/1.24 | 34.53/42.16 | 10.57/12.39 |
| 60 | 10 | 2025-09-02 → 2025-11-01 | 2025-11-01 → 2025-12-31 | 41875 / 6263 | 53029 / 7468 | 50.81/54.00 | 4.83/3.25 | 0.61/0.70 | 32.14/40.71 | 18.46/18.62 |
| 30 | 5 | 2025-11-01 → 2025-12-01 | 2025-12-01 → 2025-12-31 | 24951 / 7656 | 33661 / 8445 | 45.15/51.61 | 1.53/3.75 | 2.06/1.60 | 45.28/53.19 | 11.45/13.30 |
| 30 | 20 | 2025-11-01 → 2025-12-01 | 2025-12-01 → 2025-12-31 | 24951 / 2919 | 33661 / 2958 | 45.15/51.61 | 1.55/3.24 | 0.94/0.80 | 23.19/28.50 | 9.85/9.93 |

## 14 days / min history 10
User overlap: {'validation_only': 1895, 'test_only': 2516, 'both': 1014}.

### validation
Cutoff: 2025-12-03T00:00:00+00:00; window end: 2025-12-17T00:00:00+00:00.

| Diagnostic | Count |
|---|---|
| training_events | 1090697 |
| training_users | 167484 |
| training_items | 5339 |
| future_events | 49708 |
| future_users | 14780 |
| users_passing_history_threshold | 3367 |
| eligible_warm_user_targets | 2989 |
| excluded_cold_users | 6767 |
| excluded_sparse_users | 4646 |
| excluded_cold_item_targets | 80 |
| no_novel_target_users | 378 |
| evaluation_targets | 2909 |
| view_targets | 1824 |
| favorite_targets | 4 |
| purchase_targets | 1081 |

| Rate | Numerator | Denominator | % |
|---|---|---|---|
| cold_users_per_future_user | 6767 | 14780 | 45.78 |
| sparse_users_per_future_user | 4646 | 14780 | 31.43 |
| no_novel_per_future_user | 378 | 14780 | 2.56 |
| no_novel_per_history_eligible_user | 378 | 3367 | 11.23 |
| cold_items_per_eligible_novel_target | 80 | 2989 | 2.68 |
| cold_item_cases_per_future_user | 80 | 14780 | 0.54 |
| history_eligible_per_future_user | 3367 | 14780 | 22.78 |
| final_cases_per_future_user | 2909 | 14780 | 19.68 |

| Target type | Cases | % of final cases |
|---|---|---|
| VIEW | 1824 | 62.70 |
| FAVORITE | 4 | 0.14 |
| PURCHASE | 1081 | 37.16 |

| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |
|---|---|---|---|---|---|---|---|
| target_delay_days | 0.01 | 2.34 | 5.52 | 9.75 | 11.67 | 13.96 | 6.09 |
| benchmark_history_events | 10.00 | 15.00 | 25.00 | 55.00 | 120.00 | 2807.00 | 59.03 |
| benchmark_unique_history_items | 1.00 | 12.00 | 19.00 | 35.00 | 67.00 | 783.00 | 32.21 |
| candidate_count | 4556.00 | 5304.00 | 5320.00 | 5327.00 | 5330.00 | 5338.00 | 5306.79 |
| target_training_popularity_events | 1.00 | 322.00 | 598.00 | 1042.00 | 1447.60 | 7389.00 | 780.74 |

| Target type delay (days) | N | P25 | Median | P75 | P90 |
|---|---|---|---|---|---|
| VIEW | 1824 | 1.75 | 5.43 | 9.25 | 11.63 |
| FAVORITE | 4 | 2.15 | 3.88 | 7.21 | 10.15 |
| PURCHASE | 1081 | 3.49 | 6.42 | 10.52 | 11.68 |

Unique target items: 908.
top10 concentration: 274 / 2909 = 9.42%.
top50 concentration: 888 / 2909 = 30.53%.
No-leakage invariant violations: {'history_at_or_after_cutoff': 0, 'target_outside_window': 0, 'target_seen': 0, 'target_not_warm': 0, 'history_threshold_violation': 0}.

### test
Cutoff: 2025-12-17T00:00:00+00:00; window end: 2025-12-31T00:00:00+00:00.

| Diagnostic | Count |
|---|---|
| training_events | 1140405 |
| training_users | 174251 |
| training_items | 5447 |
| future_events | 63015 |
| future_users | 20334 |
| users_passing_history_threshold | 3903 |
| eligible_warm_user_targets | 3542 |
| excluded_cold_users | 10056 |
| excluded_sparse_users | 6375 |
| excluded_cold_item_targets | 12 |
| no_novel_target_users | 361 |
| evaluation_targets | 3530 |
| view_targets | 1940 |
| favorite_targets | 2 |
| purchase_targets | 1588 |

| Rate | Numerator | Denominator | % |
|---|---|---|---|
| cold_users_per_future_user | 10056 | 20334 | 49.45 |
| sparse_users_per_future_user | 6375 | 20334 | 31.35 |
| no_novel_per_future_user | 361 | 20334 | 1.78 |
| no_novel_per_history_eligible_user | 361 | 3903 | 9.25 |
| cold_items_per_eligible_novel_target | 12 | 3542 | 0.34 |
| cold_item_cases_per_future_user | 12 | 20334 | 0.06 |
| history_eligible_per_future_user | 3903 | 20334 | 19.19 |
| final_cases_per_future_user | 3530 | 20334 | 17.36 |

| Target type | Cases | % of final cases |
|---|---|---|
| VIEW | 1940 | 54.96 |
| FAVORITE | 2 | 0.06 |
| PURCHASE | 1588 | 44.99 |

| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |
|---|---|---|---|---|---|---|---|
| target_delay_days | 0.01 | 2.68 | 5.57 | 10.23 | 11.75 | 13.91 | 6.23 |
| benchmark_history_events | 10.00 | 14.00 | 23.00 | 49.00 | 109.10 | 3091.00 | 54.97 |
| benchmark_unique_history_items | 1.00 | 11.00 | 18.00 | 33.00 | 61.10 | 809.00 | 31.08 |
| candidate_count | 4638.00 | 5414.00 | 5429.00 | 5436.00 | 5438.00 | 5446.00 | 5415.92 |
| target_training_popularity_events | 1.00 | 274.25 | 607.00 | 1141.00 | 1708.00 | 7821.00 | 870.46 |

| Target type delay (days) | N | P25 | Median | P75 | P90 |
|---|---|---|---|---|---|
| VIEW | 1940 | 2.00 | 4.78 | 7.80 | 10.82 |
| FAVORITE | 2 | 6.96 | 7.78 | 8.60 | 9.09 |
| PURCHASE | 1588 | 3.57 | 7.41 | 10.71 | 12.61 |

Unique target items: 1003.
top10 concentration: 444 / 3530 = 12.58%.
top50 concentration: 1152 / 3530 = 32.63%.
No-leakage invariant violations: {'history_at_or_after_cutoff': 0, 'target_outside_window': 0, 'target_seen': 0, 'target_not_warm': 0, 'history_threshold_violation': 0}.

## 30 days / min history 10
User overlap: {'validation_only': 2781, 'test_only': 3146, 'both': 2195}.

### validation
Cutoff: 2025-11-01T00:00:00+00:00; window end: 2025-12-01T00:00:00+00:00.

| Diagnostic | Count |
|---|---|
| training_events | 990054 |
| training_users | 155669 |
| training_items | 5201 |
| future_events | 94706 |
| future_users | 24951 |
| users_passing_history_threshold | 5418 |
| eligible_warm_user_targets | 5055 |
| excluded_cold_users | 11266 |
| excluded_sparse_users | 8267 |
| excluded_cold_item_targets | 79 |
| no_novel_target_users | 363 |
| evaluation_targets | 4976 |
| view_targets | 3257 |
| favorite_targets | 1 |
| purchase_targets | 1718 |

| Rate | Numerator | Denominator | % |
|---|---|---|---|
| cold_users_per_future_user | 11266 | 24951 | 45.15 |
| sparse_users_per_future_user | 8267 | 24951 | 33.13 |
| no_novel_per_future_user | 363 | 24951 | 1.45 |
| no_novel_per_history_eligible_user | 363 | 5418 | 6.70 |
| cold_items_per_eligible_novel_target | 79 | 5055 | 1.56 |
| cold_item_cases_per_future_user | 79 | 24951 | 0.32 |
| history_eligible_per_future_user | 5418 | 24951 | 21.71 |
| final_cases_per_future_user | 4976 | 24951 | 19.94 |

| Target type | Cases | % of final cases |
|---|---|---|
| VIEW | 3257 | 65.45 |
| FAVORITE | 1 | 0.02 |
| PURCHASE | 1718 | 34.53 |

| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |
|---|---|---|---|---|---|---|---|
| target_delay_days | 0.01 | 5.29 | 10.57 | 18.60 | 25.49 | 29.85 | 12.33 |
| benchmark_history_events | 10.00 | 14.00 | 24.00 | 47.00 | 103.00 | 2188.00 | 50.33 |
| benchmark_unique_history_items | 2.00 | 11.00 | 17.00 | 31.00 | 58.00 | 741.00 | 29.04 |
| candidate_count | 4460.00 | 5170.00 | 5184.00 | 5190.00 | 5192.00 | 5199.00 | 5171.96 |
| target_training_popularity_events | 1.00 | 254.75 | 491.00 | 899.00 | 1406.00 | 6671.00 | 686.07 |

| Target type delay (days) | N | P25 | Median | P75 | P90 |
|---|---|---|---|---|---|
| VIEW | 3257 | 5.14 | 9.79 | 16.74 | 23.75 |
| FAVORITE | 1 | 4.70 | 4.70 | 4.70 | 4.70 |
| PURCHASE | 1718 | 5.63 | 13.55 | 21.65 | 27.60 |

Unique target items: 1039.
top10 concentration: 643 / 4976 = 12.92%.
top50 concentration: 1829 / 4976 = 36.76%.
No-leakage invariant violations: {'history_at_or_after_cutoff': 0, 'target_outside_window': 0, 'target_seen': 0, 'target_not_warm': 0, 'history_threshold_violation': 0}.

### test
Cutoff: 2025-12-01T00:00:00+00:00; window end: 2025-12-31T00:00:00+00:00.

| Diagnostic | Count |
|---|---|
| training_events | 1084760 |
| training_users | 166935 |
| training_items | 5338 |
| future_events | 118660 |
| future_users | 33661 |
| users_passing_history_threshold | 5934 |
| eligible_warm_user_targets | 5518 |
| excluded_cold_users | 17372 |
| excluded_sparse_users | 10355 |
| excluded_cold_item_targets | 177 |
| no_novel_target_users | 416 |
| evaluation_targets | 5341 |
| view_targets | 3083 |
| favorite_targets | 6 |
| purchase_targets | 2252 |

| Rate | Numerator | Denominator | % |
|---|---|---|---|
| cold_users_per_future_user | 17372 | 33661 | 51.61 |
| sparse_users_per_future_user | 10355 | 33661 | 30.76 |
| no_novel_per_future_user | 416 | 33661 | 1.24 |
| no_novel_per_history_eligible_user | 416 | 5934 | 7.01 |
| cold_items_per_eligible_novel_target | 177 | 5518 | 3.21 |
| cold_item_cases_per_future_user | 177 | 33661 | 0.53 |
| history_eligible_per_future_user | 5934 | 33661 | 17.63 |
| final_cases_per_future_user | 5341 | 33661 | 15.87 |

| Target type | Cases | % of final cases |
|---|---|---|
| VIEW | 3083 | 57.72 |
| FAVORITE | 6 | 0.11 |
| PURCHASE | 2252 | 42.16 |

| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |
|---|---|---|---|---|---|---|---|
| target_delay_days | 0.08 | 4.51 | 12.39 | 20.44 | 26.38 | 29.91 | 12.90 |
| benchmark_history_events | 10.00 | 14.00 | 22.00 | 46.00 | 101.00 | 2730.00 | 49.33 |
| benchmark_unique_history_items | 1.00 | 11.00 | 17.00 | 31.00 | 58.00 | 781.00 | 28.50 |
| candidate_count | 4557.00 | 5307.00 | 5321.00 | 5327.00 | 5329.00 | 5337.00 | 5309.50 |
| target_training_popularity_events | 1.00 | 306.00 | 586.00 | 1040.00 | 1573.00 | 7360.00 | 812.70 |

| Target type delay (days) | N | P25 | Median | P75 | P90 |
|---|---|---|---|---|---|
| VIEW | 3083 | 3.52 | 9.41 | 17.91 | 23.47 |
| FAVORITE | 6 | 4.95 | 10.84 | 20.14 | 23.78 |
| PURCHASE | 2252 | 6.71 | 15.47 | 23.54 | 27.48 |

Unique target items: 1150.
top10 concentration: 497 / 5341 = 9.31%.
top50 concentration: 1521 / 5341 = 28.48%.
No-leakage invariant violations: {'history_at_or_after_cutoff': 0, 'target_outside_window': 0, 'target_seen': 0, 'target_not_warm': 0, 'history_threshold_violation': 0}.

## 60 days / min history 10
User overlap: {'validation_only': 2743, 'test_only': 3948, 'both': 3520}.

### validation
Cutoff: 2025-09-02T00:00:00+00:00; window end: 2025-11-01T00:00:00+00:00.

| Diagnostic | Count |
|---|---|
| training_events | 809458 |
| training_users | 134393 |
| training_items | 4945 |
| future_events | 180596 |
| future_users | 41875 |
| users_passing_history_threshold | 6837 |
| eligible_warm_user_targets | 6581 |
| excluded_cold_users | 21276 |
| excluded_sparse_users | 13762 |
| excluded_cold_item_targets | 318 |
| no_novel_target_users | 256 |
| evaluation_targets | 6263 |
| view_targets | 4243 |
| favorite_targets | 7 |
| purchase_targets | 2013 |

| Rate | Numerator | Denominator | % |
|---|---|---|---|
| cold_users_per_future_user | 21276 | 41875 | 50.81 |
| sparse_users_per_future_user | 13762 | 41875 | 32.86 |
| no_novel_per_future_user | 256 | 41875 | 0.61 |
| no_novel_per_history_eligible_user | 256 | 6837 | 3.74 |
| cold_items_per_eligible_novel_target | 318 | 6581 | 4.83 |
| cold_item_cases_per_future_user | 318 | 41875 | 0.76 |
| history_eligible_per_future_user | 6837 | 41875 | 16.33 |
| final_cases_per_future_user | 6263 | 41875 | 14.96 |

| Target type | Cases | % of final cases |
|---|---|---|
| VIEW | 4243 | 67.75 |
| FAVORITE | 7 | 0.11 |
| PURCHASE | 2013 | 32.14 |

| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |
|---|---|---|---|---|---|---|---|
| target_delay_days | 0.05 | 6.43 | 18.46 | 33.46 | 47.88 | 59.78 | 21.58 |
| benchmark_history_events | 10.00 | 13.00 | 21.00 | 40.00 | 81.00 | 1867.00 | 41.10 |
| benchmark_unique_history_items | 1.00 | 11.00 | 16.00 | 26.00 | 48.00 | 654.00 | 24.54 |
| candidate_count | 4291.00 | 4919.00 | 4929.00 | 4934.00 | 4937.00 | 4944.00 | 4920.46 |
| target_training_popularity_events | 1.00 | 70.00 | 304.00 | 852.50 | 1230.00 | 5527.00 | 527.66 |

| Target type delay (days) | N | P25 | Median | P75 | P90 |
|---|---|---|---|---|---|
| VIEW | 4243 | 4.64 | 15.86 | 28.92 | 46.36 |
| FAVORITE | 7 | 6.64 | 21.41 | 41.07 | 55.25 |
| PURCHASE | 2013 | 11.68 | 25.59 | 39.67 | 51.60 |

Unique target items: 1135.
top10 concentration: 860 / 6263 = 13.73%.
top50 concentration: 2151 / 6263 = 34.34%.
No-leakage invariant violations: {'history_at_or_after_cutoff': 0, 'target_outside_window': 0, 'target_seen': 0, 'target_not_warm': 0, 'history_threshold_violation': 0}.

### test
Cutoff: 2025-11-01T00:00:00+00:00; window end: 2025-12-31T00:00:00+00:00.

| Diagnostic | Count |
|---|---|
| training_events | 990054 |
| training_users | 155669 |
| training_items | 5201 |
| future_events | 213366 |
| future_users | 53029 |
| users_passing_history_threshold | 8092 |
| eligible_warm_user_targets | 7719 |
| excluded_cold_users | 28638 |
| excluded_sparse_users | 16299 |
| excluded_cold_item_targets | 251 |
| no_novel_target_users | 373 |
| evaluation_targets | 7468 |
| view_targets | 4424 |
| favorite_targets | 4 |
| purchase_targets | 3040 |

| Rate | Numerator | Denominator | % |
|---|---|---|---|
| cold_users_per_future_user | 28638 | 53029 | 54.00 |
| sparse_users_per_future_user | 16299 | 53029 | 30.74 |
| no_novel_per_future_user | 373 | 53029 | 0.70 |
| no_novel_per_history_eligible_user | 373 | 8092 | 4.61 |
| cold_items_per_eligible_novel_target | 251 | 7719 | 3.25 |
| cold_item_cases_per_future_user | 251 | 53029 | 0.47 |
| history_eligible_per_future_user | 8092 | 53029 | 15.26 |
| final_cases_per_future_user | 7468 | 53029 | 14.08 |

| Target type | Cases | % of final cases |
|---|---|---|
| VIEW | 4424 | 59.24 |
| FAVORITE | 4 | 0.05 |
| PURCHASE | 3040 | 40.71 |

| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |
|---|---|---|---|---|---|---|---|
| target_delay_days | 0.01 | 7.96 | 18.62 | 36.35 | 50.61 | 59.83 | 23.04 |
| benchmark_history_events | 10.00 | 13.00 | 21.00 | 41.00 | 85.00 | 2188.00 | 43.36 |
| benchmark_unique_history_items | 1.00 | 11.00 | 16.00 | 28.00 | 52.00 | 741.00 | 25.85 |
| candidate_count | 4460.00 | 5173.00 | 5185.00 | 5190.00 | 5192.00 | 5200.00 | 5175.15 |
| target_training_popularity_events | 1.00 | 228.00 | 478.00 | 862.00 | 1460.00 | 6671.00 | 682.07 |

| Target type delay (days) | N | P25 | Median | P75 | P90 |
|---|---|---|---|---|---|
| VIEW | 4424 | 6.64 | 13.74 | 30.48 | 47.28 |
| FAVORITE | 4 | 26.73 | 43.11 | 52.96 | 54.44 |
| PURCHASE | 3040 | 11.45 | 26.69 | 43.51 | 54.46 |

Unique target items: 1254.
top10 concentration: 770 / 7468 = 10.31%.
top50 concentration: 2265 / 7468 = 30.33%.
No-leakage invariant violations: {'history_at_or_after_cutoff': 0, 'target_outside_window': 0, 'target_seen': 0, 'target_not_warm': 0, 'history_threshold_violation': 0}.

## 30 days / min history 5
User overlap: {'validation_only': 4846, 'test_only': 5635, 'both': 2810}.

### validation
Cutoff: 2025-11-01T00:00:00+00:00; window end: 2025-12-01T00:00:00+00:00.

| Diagnostic | Count |
|---|---|
| training_events | 990054 |
| training_users | 155669 |
| training_items | 5201 |
| future_events | 94706 |
| future_users | 24951 |
| users_passing_history_threshold | 8288 |
| eligible_warm_user_targets | 7775 |
| excluded_cold_users | 11266 |
| excluded_sparse_users | 5397 |
| excluded_cold_item_targets | 119 |
| no_novel_target_users | 513 |
| evaluation_targets | 7656 |
| view_targets | 4185 |
| favorite_targets | 4 |
| purchase_targets | 3467 |

| Rate | Numerator | Denominator | % |
|---|---|---|---|
| cold_users_per_future_user | 11266 | 24951 | 45.15 |
| sparse_users_per_future_user | 5397 | 24951 | 21.63 |
| no_novel_per_future_user | 513 | 24951 | 2.06 |
| no_novel_per_history_eligible_user | 513 | 8288 | 6.19 |
| cold_items_per_eligible_novel_target | 119 | 7775 | 1.53 |
| cold_item_cases_per_future_user | 119 | 24951 | 0.48 |
| history_eligible_per_future_user | 8288 | 24951 | 33.22 |
| final_cases_per_future_user | 7656 | 24951 | 30.68 |

| Target type | Cases | % of final cases |
|---|---|---|
| VIEW | 4185 | 54.66 |
| FAVORITE | 4 | 0.05 |
| PURCHASE | 3467 | 45.28 |

| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |
|---|---|---|---|---|---|---|---|
| target_delay_days | 0.01 | 5.61 | 11.45 | 20.45 | 26.41 | 29.85 | 12.99 |
| benchmark_history_events | 5.00 | 8.00 | 14.00 | 31.00 | 74.00 | 2188.00 | 35.04 |
| benchmark_unique_history_items | 1.00 | 7.00 | 11.00 | 22.00 | 45.00 | 741.00 | 20.94 |
| candidate_count | 4460.00 | 5179.00 | 5190.00 | 5194.00 | 5196.00 | 5200.00 | 5180.06 |
| target_training_popularity_events | 1.00 | 246.00 | 490.00 | 904.00 | 1479.00 | 6671.00 | 712.08 |

| Target type delay (days) | N | P25 | Median | P75 | P90 |
|---|---|---|---|---|---|
| VIEW | 4185 | 5.33 | 10.20 | 17.34 | 24.39 |
| FAVORITE | 4 | 5.03 | 5.88 | 8.07 | 10.66 |
| PURCHASE | 3467 | 6.54 | 14.45 | 22.44 | 28.33 |

Unique target items: 1250.
top10 concentration: 886 / 7656 = 11.57%.
top50 concentration: 2520 / 7656 = 32.92%.
No-leakage invariant violations: {'history_at_or_after_cutoff': 0, 'target_outside_window': 0, 'target_seen': 0, 'target_not_warm': 0, 'history_threshold_violation': 0}.

### test
Cutoff: 2025-12-01T00:00:00+00:00; window end: 2025-12-31T00:00:00+00:00.

| Diagnostic | Count |
|---|---|
| training_events | 1084760 |
| training_users | 166935 |
| training_items | 5338 |
| future_events | 118660 |
| future_users | 33661 |
| users_passing_history_threshold | 9314 |
| eligible_warm_user_targets | 8774 |
| excluded_cold_users | 17372 |
| excluded_sparse_users | 6975 |
| excluded_cold_item_targets | 329 |
| no_novel_target_users | 540 |
| evaluation_targets | 8445 |
| view_targets | 3947 |
| favorite_targets | 6 |
| purchase_targets | 4492 |

| Rate | Numerator | Denominator | % |
|---|---|---|---|
| cold_users_per_future_user | 17372 | 33661 | 51.61 |
| sparse_users_per_future_user | 6975 | 33661 | 20.72 |
| no_novel_per_future_user | 540 | 33661 | 1.60 |
| no_novel_per_history_eligible_user | 540 | 9314 | 5.80 |
| cold_items_per_eligible_novel_target | 329 | 8774 | 3.75 |
| cold_item_cases_per_future_user | 329 | 33661 | 0.98 |
| history_eligible_per_future_user | 9314 | 33661 | 27.67 |
| final_cases_per_future_user | 8445 | 33661 | 25.09 |

| Target type | Cases | % of final cases |
|---|---|---|
| VIEW | 3947 | 46.74 |
| FAVORITE | 6 | 0.07 |
| PURCHASE | 4492 | 53.19 |

| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |
|---|---|---|---|---|---|---|---|
| target_delay_days | 0.08 | 5.55 | 13.30 | 21.48 | 26.53 | 29.91 | 13.76 |
| benchmark_history_events | 5.00 | 7.00 | 13.00 | 29.00 | 71.00 | 2730.00 | 33.63 |
| benchmark_unique_history_items | 1.00 | 6.00 | 10.00 | 21.00 | 42.00 | 781.00 | 20.18 |
| candidate_count | 4557.00 | 5317.00 | 5328.00 | 5332.00 | 5333.00 | 5337.00 | 5317.82 |
| target_training_popularity_events | 1.00 | 294.00 | 582.00 | 1045.00 | 1670.00 | 7360.00 | 836.98 |

| Target type delay (days) | N | P25 | Median | P75 | P90 |
|---|---|---|---|---|---|
| VIEW | 3947 | 3.56 | 9.59 | 18.36 | 23.86 |
| FAVORITE | 6 | 4.95 | 10.84 | 20.14 | 23.78 |
| PURCHASE | 4492 | 7.60 | 15.80 | 23.62 | 27.47 |

Unique target items: 1344.
top10 concentration: 698 / 8445 = 8.27%.
top50 concentration: 2158 / 8445 = 25.55%.
No-leakage invariant violations: {'history_at_or_after_cutoff': 0, 'target_outside_window': 0, 'target_seen': 0, 'target_not_warm': 0, 'history_threshold_violation': 0}.

## 30 days / min history 20
User overlap: {'validation_only': 1401, 'test_only': 1440, 'both': 1518}.

### validation
Cutoff: 2025-11-01T00:00:00+00:00; window end: 2025-12-01T00:00:00+00:00.

| Diagnostic | Count |
|---|---|
| training_events | 990054 |
| training_users | 155669 |
| training_items | 5201 |
| future_events | 94706 |
| future_users | 24951 |
| users_passing_history_threshold | 3200 |
| eligible_warm_user_targets | 2965 |
| excluded_cold_users | 11266 |
| excluded_sparse_users | 10485 |
| excluded_cold_item_targets | 46 |
| no_novel_target_users | 235 |
| evaluation_targets | 2919 |
| view_targets | 2241 |
| favorite_targets | 1 |
| purchase_targets | 677 |

| Rate | Numerator | Denominator | % |
|---|---|---|---|
| cold_users_per_future_user | 11266 | 24951 | 45.15 |
| sparse_users_per_future_user | 10485 | 24951 | 42.02 |
| no_novel_per_future_user | 235 | 24951 | 0.94 |
| no_novel_per_history_eligible_user | 235 | 3200 | 7.34 |
| cold_items_per_eligible_novel_target | 46 | 2965 | 1.55 |
| cold_item_cases_per_future_user | 46 | 24951 | 0.18 |
| history_eligible_per_future_user | 3200 | 24951 | 12.83 |
| final_cases_per_future_user | 2919 | 24951 | 11.70 |

| Target type | Cases | % of final cases |
|---|---|---|
| VIEW | 2241 | 76.77 |
| FAVORITE | 1 | 0.03 |
| PURCHASE | 677 | 23.19 |

| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |
|---|---|---|---|---|---|---|---|
| target_delay_days | 0.01 | 4.70 | 9.85 | 17.44 | 24.71 | 29.80 | 11.57 |
| benchmark_history_events | 20.00 | 27.00 | 41.00 | 77.00 | 145.00 | 2188.00 | 76.14 |
| benchmark_unique_history_items | 3.00 | 20.00 | 28.00 | 46.00 | 78.00 | 741.00 | 41.71 |
| candidate_count | 4460.00 | 5155.00 | 5173.00 | 5181.00 | 5185.00 | 5198.00 | 5159.29 |
| target_training_popularity_events | 1.00 | 268.00 | 502.00 | 902.00 | 1370.00 | 6671.00 | 673.88 |

| Target type delay (days) | N | P25 | Median | P75 | P90 |
|---|---|---|---|---|---|
| VIEW | 2241 | 4.72 | 9.65 | 16.47 | 23.53 |
| FAVORITE | 1 | 4.70 | 4.70 | 4.70 | 4.70 |
| PURCHASE | 677 | 4.67 | 11.72 | 21.53 | 27.01 |

Unique target items: 804.
top10 concentration: 420 / 2919 = 14.39%.
top50 concentration: 1198 / 2919 = 41.04%.
No-leakage invariant violations: {'history_at_or_after_cutoff': 0, 'target_outside_window': 0, 'target_seen': 0, 'target_not_warm': 0, 'history_threshold_violation': 0}.

### test
Cutoff: 2025-12-01T00:00:00+00:00; window end: 2025-12-31T00:00:00+00:00.

| Diagnostic | Count |
|---|---|
| training_events | 1084760 |
| training_users | 166935 |
| training_items | 5338 |
| future_events | 118660 |
| future_users | 33661 |
| users_passing_history_threshold | 3326 |
| eligible_warm_user_targets | 3057 |
| excluded_cold_users | 17372 |
| excluded_sparse_users | 12963 |
| excluded_cold_item_targets | 99 |
| no_novel_target_users | 269 |
| evaluation_targets | 2958 |
| view_targets | 2111 |
| favorite_targets | 4 |
| purchase_targets | 843 |

| Rate | Numerator | Denominator | % |
|---|---|---|---|
| cold_users_per_future_user | 17372 | 33661 | 51.61 |
| sparse_users_per_future_user | 12963 | 33661 | 38.51 |
| no_novel_per_future_user | 269 | 33661 | 0.80 |
| no_novel_per_history_eligible_user | 269 | 3326 | 8.09 |
| cold_items_per_eligible_novel_target | 99 | 3057 | 3.24 |
| cold_item_cases_per_future_user | 99 | 33661 | 0.29 |
| history_eligible_per_future_user | 3326 | 33661 | 9.88 |
| final_cases_per_future_user | 2958 | 33661 | 8.79 |

| Target type | Cases | % of final cases |
|---|---|---|
| VIEW | 2111 | 71.37 |
| FAVORITE | 4 | 0.14 |
| PURCHASE | 843 | 28.50 |

| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |
|---|---|---|---|---|---|---|---|
| target_delay_days | 0.16 | 3.60 | 9.93 | 18.71 | 24.52 | 29.77 | 11.60 |
| benchmark_history_events | 20.00 | 28.00 | 42.00 | 79.00 | 148.30 | 2730.00 | 78.10 |
| benchmark_unique_history_items | 1.00 | 20.00 | 29.00 | 47.00 | 79.30 | 781.00 | 42.54 |
| candidate_count | 4557.00 | 5291.00 | 5309.00 | 5318.00 | 5322.00 | 5337.00 | 5295.46 |
| target_training_popularity_events | 1.00 | 321.00 | 589.50 | 1030.00 | 1498.70 | 7360.00 | 775.01 |

| Target type delay (days) | N | P25 | Median | P75 | P90 |
|---|---|---|---|---|---|
| VIEW | 2111 | 3.25 | 8.48 | 17.62 | 23.36 |
| FAVORITE | 4 | 10.99 | 18.13 | 22.96 | 24.44 |
| PURCHASE | 843 | 6.44 | 13.55 | 21.64 | 26.67 |

Unique target items: 887.
top10 concentration: 303 / 2958 = 10.24%.
top50 concentration: 943 / 2958 = 31.88%.
No-leakage invariant violations: {'history_at_or_after_cutoff': 0, 'target_outside_window': 0, 'target_seen': 0, 'target_not_warm': 0, 'history_threshold_violation': 0}.

Unavailable scenarios: [].
Zero-event days: [].

## Ingestion diagnostics

| Diagnostic | Count |
|---|---|
| customer_merges | 41412 |
| actions_raw | 3347664 |
| unmapped_actions | 0 |
| malformed_mapped_actions | 2732459 |
| order_lines | 666576 |
| orders_raw | 309044 |
| orders_unique | 309044 |
| orders_duplicate_identical | 0 |
| orders_duplicate_conflicting | 0 |
| classified_interactions | 1206884 |
| resolved_interactions | 1206864 |
| unresolved_interactions | 20 |
| unsupported_products | 0 |

## Last 14 declared days

| UTC day | Resolved events |
|---|---|
| 2025-12-18 | 4702 |
| 2025-12-19 | 3574 |
| 2025-12-20 | 5602 |
| 2025-12-21 | 4970 |
| 2025-12-22 | 3950 |
| 2025-12-23 | 3449 |
| 2025-12-24 | 4426 |
| 2025-12-25 | 3822 |
| 2025-12-26 | 3825 |
| 2025-12-27 | 6552 |
| 2025-12-28 | 5957 |
| 2025-12-29 | 4171 |
| 2025-12-30 | 4964 |
| 2025-12-31 | 3444 |

## Weekly density

| Week start | Events | Active customers | Active items | VIEW / FAVORITE / PURCHASE |
|---|---|---|---|---|
| 2022-10-10 | 2 | 1 | 1 | 0 / 0 / 2 |
| 2023-05-08 | 4 | 1 | 4 | 0 / 0 / 4 |
| 2023-07-10 | 1 | 1 | 1 | 0 / 0 / 1 |
| 2023-08-14 | 1 | 1 | 1 | 0 / 0 / 1 |
| 2023-11-27 | 1 | 1 | 1 | 0 / 0 / 1 |
| 2023-12-04 | 2 | 1 | 2 | 0 / 0 / 2 |
| 2023-12-18 | 1 | 1 | 1 | 0 / 0 / 1 |
| 2024-01-15 | 1 | 1 | 1 | 0 / 0 / 1 |
| 2024-03-18 | 1 | 1 | 1 | 0 / 0 / 1 |
| 2024-05-06 | 1 | 1 | 1 | 0 / 0 / 1 |
| 2024-05-27 | 8 | 2 | 7 | 0 / 0 / 8 |
| 2024-06-10 | 1 | 1 | 1 | 0 / 0 / 1 |
| 2024-07-01 | 3 | 1 | 3 | 0 / 0 / 3 |
| 2024-07-08 | 1 | 1 | 1 | 0 / 0 / 1 |
| 2024-07-15 | 5 | 1 | 5 | 0 / 0 / 5 |
| 2024-07-22 | 7 | 2 | 7 | 0 / 0 / 7 |
| 2024-07-29 | 3 | 1 | 3 | 0 / 0 / 3 |
| 2024-08-05 | 1 | 1 | 1 | 0 / 0 / 1 |
| 2024-08-12 | 5 | 3 | 5 | 0 / 0 / 5 |
| 2024-08-19 | 5 | 3 | 5 | 0 / 0 / 5 |
| 2024-08-26 | 9 | 3 | 8 | 0 / 0 / 9 |
| 2024-09-16 | 2 | 2 | 2 | 0 / 0 / 2 |
| 2024-09-23 | 8 | 3 | 7 | 0 / 0 / 8 |
| 2024-09-30 | 3 | 2 | 3 | 0 / 0 / 3 |
| 2024-10-07 | 7 | 3 | 7 | 0 / 0 / 7 |
| 2024-10-14 | 4 | 2 | 4 | 0 / 0 / 4 |
| 2024-10-21 | 4 | 4 | 4 | 0 / 0 / 4 |
| 2024-10-28 | 10 | 1 | 7 | 0 / 0 / 10 |
| 2024-11-04 | 9 | 3 | 8 | 0 / 0 / 9 |
| 2024-11-11 | 4 | 3 | 4 | 0 / 0 / 4 |
| 2024-11-18 | 18 | 6 | 18 | 0 / 0 / 18 |
| 2024-11-25 | 15 | 9 | 13 | 0 / 0 / 15 |
| 2024-12-02 | 14 | 5 | 13 | 0 / 0 / 14 |
| 2024-12-09 | 58 | 22 | 54 | 0 / 0 / 58 |
| 2024-12-16 | 205 | 108 | 171 | 0 / 0 / 205 |
| 2024-12-23 | 489 | 257 | 356 | 0 / 0 / 489 |
| 2024-12-30 | 18576 | 5957 | 1803 | 7878 / 0 / 10698 |
| 2025-01-06 | 27343 | 8029 | 1945 | 13530 / 0 / 13813 |
| 2025-01-13 | 30160 | 8132 | 1969 | 18322 / 0 / 11838 |
| 2025-01-20 | 31897 | 8351 | 1992 | 19980 / 22 / 11895 |
| 2025-01-27 | 26459 | 7291 | 1971 | 15437 / 134 / 10888 |
| 2025-02-03 | 24277 | 6957 | 1959 | 13635 / 212 / 10430 |
| 2025-02-10 | 30347 | 8287 | 2005 | 16908 / 140 / 13299 |
| 2025-02-17 | 29751 | 9526 | 2072 | 12418 / 50 / 17283 |
| 2025-02-24 | 26015 | 7381 | 1921 | 15972 / 113 / 9930 |
| 2025-03-03 | 21693 | 6610 | 1891 | 12753 / 95 / 8845 |
| 2025-03-10 | 21967 | 6694 | 1890 | 12077 / 150 / 9740 |
| 2025-03-17 | 21516 | 6232 | 1986 | 12217 / 95 / 9204 |
| 2025-03-24 | 20750 | 6354 | 1918 | 10903 / 100 / 9747 |
| 2025-03-31 | 19932 | 6360 | 1880 | 10912 / 69 / 8951 |
| 2025-04-07 | 23282 | 6817 | 1982 | 14114 / 159 / 9009 |
| 2025-04-14 | 22142 | 6581 | 1986 | 12543 / 94 / 9505 |
| 2025-04-21 | 20505 | 6232 | 1910 | 9991 / 141 / 10373 |
| 2025-04-28 | 21376 | 7024 | 1944 | 10588 / 63 / 10725 |
| 2025-05-05 | 19291 | 6678 | 1850 | 8717 / 57 / 10517 |
| 2025-05-12 | 20557 | 6606 | 1993 | 10094 / 139 / 10324 |
| 2025-05-19 | 20079 | 6550 | 1973 | 8767 / 55 / 11257 |
| 2025-05-26 | 22213 | 6963 | 1941 | 9722 / 109 / 12382 |
| 2025-06-02 | 19100 | 6494 | 1847 | 7518 / 76 / 11506 |
| 2025-06-09 | 22341 | 7656 | 1727 | 9358 / 36 / 12947 |
| 2025-06-16 | 17576 | 6242 | 1732 | 7227 / 44 / 10305 |
| 2025-06-23 | 20739 | 6848 | 1716 | 9171 / 41 / 11527 |
| 2025-06-30 | 21016 | 6912 | 1743 | 9255 / 70 / 11691 |
| 2025-07-07 | 22092 | 7269 | 1664 | 9722 / 116 / 12254 |
| 2025-07-14 | 22786 | 7375 | 1636 | 10329 / 47 / 12410 |
| 2025-07-21 | 22227 | 6905 | 1554 | 10764 / 136 / 11327 |
| 2025-07-28 | 23132 | 7347 | 1685 | 9977 / 104 / 13051 |
| 2025-08-04 | 24858 | 7817 | 1642 | 12009 / 101 / 12748 |
| 2025-08-11 | 23439 | 7626 | 1671 | 11172 / 50 / 12217 |
| 2025-08-18 | 22489 | 7247 | 1689 | 10376 / 56 / 12057 |
| 2025-08-25 | 23728 | 7786 | 1705 | 10613 / 81 / 13034 |
| 2025-09-01 | 20764 | 6779 | 1617 | 10939 / 36 / 9789 |
| 2025-09-08 | 19958 | 6274 | 1654 | 10199 / 43 / 9716 |
| 2025-09-15 | 20327 | 6374 | 1639 | 10621 / 59 / 9647 |
| 2025-09-22 | 22459 | 7152 | 1634 | 13219 / 140 / 9100 |
| 2025-09-29 | 20832 | 6509 | 1646 | 11604 / 86 / 9142 |
| 2025-10-06 | 19444 | 6075 | 1625 | 10414 / 132 / 8898 |
| 2025-10-13 | 22729 | 7387 | 1690 | 13214 / 123 / 9392 |
| 2025-10-20 | 22536 | 7240 | 1688 | 12448 / 173 / 9915 |
| 2025-10-27 | 20057 | 6159 | 1660 | 11455 / 180 / 8422 |
| 2025-11-03 | 20254 | 6769 | 1649 | 10219 / 72 / 9963 |
| 2025-11-10 | 24564 | 7467 | 1675 | 13899 / 227 / 10438 |
| 2025-11-17 | 21404 | 6702 | 1591 | 11605 / 78 / 9721 |
| 2025-11-24 | 22868 | 6998 | 1705 | 11652 / 91 / 11125 |
| 2025-12-01 | 24487 | 7657 | 1762 | 13198 / 301 / 10988 |
| 2025-12-08 | 24904 | 8033 | 1745 | 11578 / 136 / 13190 |
| 2025-12-15 | 28153 | 9510 | 1833 | 12497 / 139 / 15517 |
| 2025-12-22 | 31981 | 10877 | 1861 | 13028 / 98 / 18855 |
| 2025-12-29 | 12579 | 4871 | 1521 | 3328 / 42 / 9209 |

Source coverage is declarative for MANUAL imports; no independent completeness proof.
No model training, publication, item features or model-quality-based cutoff selection.
Current mutable catalog/identity metadata do not establish historical as-of provenance.
Full daily density and exact aggregates are retained in report.json.
