# Проверка после остановки BPR-EXP-01

Полный sweep отменён пользователем. Его experiment/training processes завершены. Оставшийся pytest завершился: 2016 passed, 110 warnings.

Прерванный real-data loader не дошёл до обучения и не создал `diagnostics/bpr_exp01_weights/`. Частичных результатов, metrics или checkpoints этого запуска нет. Ничего не удалено.

Существующие завершённые артефакты TRAIN-03A сохранены:

- `diagnostics/temporal_train03a/findings.md` — 18610 bytes.
- `diagnostics/temporal_train03a/report.json` — 370736 bytes.
- `diagnostics/temporal_train03a/report.md` — 31655 bytes.

До нового запуска проверены SHA-256 всех 201 существовавших до BPR-EXP-01 файлов: изменений нет, включая 23 файла canonical inputs / model / user_settings. Production model и current.json не изменялись.

Для BPR-EXP-01 добавлены только:

- `Application/evaluation/experiments/__init__.py` — isolated research package.
- `Application/evaluation/experiments/bpr_weights.py` — preparation/training/evaluation, fixed seeds, provenance, validation guard, aggregation and selection infrastructure. После изменения плана добавлен отдельный one-run path и штатные process memory counters; sweep сохранён.
- `scripts/run_bpr_weight_experiment.py` — CLI. Default теперь baseline; sweep требует explicit `--mode sweep` и в текущей задаче не запускается.
- `tests/evaluation/test_bpr_weight_experiment.py` — synthetic tests; 15 passed после добавления one-run mode.
- `docs/bpr_weight_experiment.md` — execution contract.

Новый разрешённый план: ровно одно обучение VIEW=0.1, FAVORITE=2, PURCHASE=10, seed=42, epochs=200, embedding_dim=128, batch_size=256, n_neg=10. Early stopping и item features выключены. Остальные existing hyperparameters сохранены. Только temporal validation, никакой test performance. После одного run — отчёт о времени и остановка до решения пользователя.

Этот файл — завершённая проверка состояния, не training result. `results.json` нового запуска следует считать incomplete, пока `status` не станет `complete` и `run` не будет содержать результат всех 200 эпох.

Git status после остановки (строки M — предыдущие TRAIN этапы):

```text
 M Application/model/BPRMF.py
 M Application/model/bpr_preparation.py
 M Application/model/mindbox_production_training.py
 M Application/model/mindbox_training_preparation.py
 M Application/model/training_data.py
 M Application/model/training_metrics.py
 M Application/tabs/train_model_tab.py
 M README.md
 M scripts/mindbox_production_train.py
 M tests/model/test_bpr_preparation.py
 M tests/model/test_mindbox_production_training.py
 M tests/model/test_mindbox_training_preparation.py
 M tests/model/test_seen_items.py
 M tests/model/test_training_data.py
 M tests/tabs/test_train_model_start.py
?? Application/evaluation/
?? diagnostics/
?? docs/
?? scripts/audit_temporal_evaluation.py
?? scripts/run_bpr_weight_experiment.py
?? tests/evaluation/
```

Tracked `git diff --stat` относится к прежним изменениям: 15 files changed, 1274 insertions(+), 330 deletions(-). Новая EXP infrastructure пока untracked и в этом stat не учитывается. Commit/push/staging не выполнялись.
