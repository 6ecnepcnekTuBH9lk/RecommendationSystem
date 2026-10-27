# TRAIN-UI-01: результат реализации

Ветка: `feature/training-settings`. Изменения не закоммичены и не отправлены.
Source of truth — текущие файлы проекта; предыдущая работа BPR/temporal/experiment сохранена.

До задачи вкладка содержала больше двадцати низкоуровневых training inputs, reset button,
два readonly информационных поля, start и журнал. Работал только canonical production CLI;
отдельной кнопки cancel, research mode, progress/result card и истории не было.

Теперь сверху layout 40/60: слева режим, четыре input, readiness и start/cancel;
справа epoch/progress/loss/elapsed/device, журнал и результат. Снизу на всю ширину — history.
Тема приложения и существующая вкладка сохранены. Preview PNG содержат синтетические записи,
а не результаты новых real-data экспериментов.

## Поля и режимы

| Поле | Начальное значение | Проверка |
|---|---:|---|
| Вес покупки | 10 | finite, > 0; GUI upper limit 1 000 000 |
| Вес избранного | 2 | finite, >= 0; GUI upper limit 1 000 000 |
| Вес просмотра | 0.5 | finite, >= 0; GUI upper limit 1 000 000 |
| Количество эпох | 50 | integer, 1–10000 |

Это provisional defaults. Оптимальность этих weights и epoch budget не утверждается.
GUI inputs между сессиями ранее не сохранялись; новый settings framework не добавлен.

Удалены controls для top-k, lr, n_neg, seed, BPR/L2 regularization, min eval history,
batch size, embedding dimension, early-stop metric/patience/min delta/min epochs,
feature limit/dropout/scale/regularization/normalization; reset button и строки
«Признаки номенклатуры…» / «Источник обучения…». Их параметры остаются в TrainConfig.
В MainWindow удалены только ссылка и static style для удалённой reset button.
Legacy CSV/weather/filter helpers в train_model_tab.py сохранены без изменения.

**Эксперимент** выбран по умолчанию. GUI передаёт только четыре значения во временном
JSON в `scripts/run_gui_research.py`, тот же `sys.executable`, отдельный QProcess.
Используются existing canonical loader, benchmark_config, temporal protocol,
prepare_bpr_snapshot и evaluate_bpr_snapshot. Одна configuration → одно обучение
→ один final validation checkpoint → aggregate result artifact. Следующих runs нет.

Research history cutoff: timestamp < 2025-11-01 UTC; validation:
[2025-11-01, 2025-12-01) UTC, min_history_events=10. Protocol создаётся существующим
builder; в runner передаётся только `.validation`, остальной protocol удаляется.
require_validation проверяет cutoff/end перед training/scoring. Test snapshot не
выбирается и не оценивается; test metrics не попадают в GUI/history.

Общий lightweight research contract фиксирует seed=42, embedding_dim=128,
batch_size=256, n_neg=10, lr=0.0003, bpr_reg=0.0005, weight_decay=0,
early_stop=False, use_item_features=False. Остальные defaults предоставляет TrainConfig.
Trainer должен выполнить ровно requested epochs; нет checkpoint winner selection.
Publication API в research orchestration не вызывается.

**Рабочая модель**: GUI → существующая CLI command `gui` → canonical preflight →
production preparation → trainer → существующая publication. Четыре inputs
перекрывают соответствующие поля existing train_config.json; скрытые параметры
проверяются existing typed loader, отсутствующие получают TrainConfig defaults.
Early stopping, item features и прочие production settings сохраняются.
Existing settings file не перезаписывается; analysis filters/legacy CSV не используются.
Перед start подтверждаются weights, epoch budget и новая generation при успехе.
PASS разрешает training; WARN требует дополнительного согласия (default No);
BLOCK не допускает trainer. Production runs не записываются в research history.

## Readiness, lifecycle и отмена

Открытие вкладки проверяет nomenclature.csv, training.json, current_batch/catalog
и имена canonical parts. Raw payload/canonical ingestion не выполняются.
Для research дополнительно проверяется metadata coverage history/validation.
Полная preparation/quality gate выполняется вне GUI, в QProcess.

Start disabled при отсутствии/повреждении metadata, missing catalog/batch/parts,
недоступном benchmark, invalid/incomplete input, активном run и известном BLOCK.
Причина показана рядом с start. Production BLOCK сохраняется для того же dataset,
catalog и configuration; изменение inputs/settings/data позволяет новый preflight.
Недоступный research benchmark остаётся blocked при изменении weights и требует
обновления данных. Форму и mode нельзя менять во время run.

Cancel создаёт отдельный per-run file; проверки между stages/epochs вызывают
KeyboardInterrupt и штатную cleanup. При задержке single QProcess завершается
через 2 секунды. Эти GUI runners не создают multiprocessing training workers.
Принятую GUI cancellation нельзя превратить в publication: непосредственно перед
записью CLI ждёт `PUBLISH` handshake. GUI сначала запрещает cancel, затем отправляет
разрешение; при уже запрошенной отмене отвечает отказом. Cancel недоступен в короткой
publication critical section. Model lock/staging/atomic current/readback/rollback
не переписаны. Check cancellation также встроен в existing precommit guard.
Existing CLI без cancel-file сохраняет прежний protocol/behavior.

Закрытие окна во время active run сначала запрашивает cancel и отклоняет close;
после завершения окно можно закрыть. Normal/error/failed-to-start/cancel завершения
разблокируют inputs; temporary config/cancel-file удаляются.

Progress использует `TRAINING_EVENT` JSON, а не разбор русского trainer stdout:
loading, preparation, training, epoch, validation, publication и finished events.
Сохраняются byte buffering/UTF-8 split protection. Qt timer показывает elapsed;
worker events — epoch X/N, latest loss и CPU/CUDA. Журнал очищается при старте,
остаётся readonly, ограничен 5000 blocks. Raw trainer diagnostics подавлены,
структурированные события обходят suppression через captured terminal stream.

## Результат и история

Research result card: NDCG@10, Recall@10, VIEW/PURCHASE NDCG@10, completed epochs,
total duration и device; precision 6 decimals, duration HH:MM:SS. FAVORITE не
главный показатель. Artifact хранит overall и type NDCG/Recall @5/@10/@20.
Production card: epochs, duration, device, published generation и publication/readback
status, без research scores. Смена mode сбрасывает старую card; выбор history row
показывает её research результат и не меняет weights автоматически.

History path относительно корня проекта:
`user_settings/research_experiments/history.json`.
Artifacts: `user_settings/research_experiments/runs/<run_id>/result.json`.

Schema version 1: `runs[]`, уникальный UUID hex `run_id`, started_at/finished_at,
status, hyperparameters (weights/epochs/seed/frozen parameters), epochs_requested/
epochs_completed, benchmark identity/dates/min_history_events, training_users/items/
pairs, validation metrics, total_seconds/training_seconds, device, torch_version,
cuda_build, git_head/working_tree_dirty provenance, relative artifact и safe error code.
Неизвестные runtime/count fields — null. Allowlist исключает customer_id/email/phone,
raw payload, arbitrary exception text и test metrics.

OS lock защищает read-modify-write; existing atomic_json создаёт unique temporary,
выполняет flush/fsync, close и atomic replace. Upsert исключает duplicate append;
newest-first ordering сохраняется между instances. Running и выполненные эпохи
тоже записываются. Completed/failed/cancelled остаются в history. После restart
finished artifact восстанавливает финальный статус; dead running worker становится
interrupted. Alive worker не объявляется завершённым. Corrupt history не
перезаписывается; вкладка показывает предупреждение, новые artifacts сохраняются
отдельно, production остаётся доступен. Автоматического выбора best/winner нет.

## Файлы текущей задачи

| Файл | Изменение |
|---|---|
| Application/tabs/train_model_tab.py | layout, modes, QProcess, progress, readiness, cancel, result/history |
| Application/tabs/training_workflow.py (new) | lightweight inputs/metadata/production config |
| Application/evaluation/experiments/gui_history.py (new) | versioned allowlisted atomic persistence/recovery, shared frozen contract |
| Application/evaluation/experiments/gui_run.py (new) | existing trainer/adapter/evaluator orchestration, validation guard |
| scripts/run_gui_research.py (new) | one-run research QProcess entry point and artifacts |
| Application/model/mindbox_production_training.py | optional observer/cancel/prepublication hooks; default behavior preserved |
| scripts/mindbox_production_train.py | managed GUI progress/cancel/handshake, existing CLI preserved |
| main.py | removed dependency on deleted reset control |
| tests/tabs/test_train_model_start.py | four-input/config-layer semantics and existing canonical QProcess protections |
| tests/tabs/test_train_model_completion.py | active-run/readiness fixture; failed exits still cannot claim success |
| tests/tabs/test_training_research_ui.py (new) | Qt form/modes/lifecycle/history/progress/readiness states |
| tests/evaluation/test_gui_history.py (new) | persistence/dedup/recovery/corruption/atomic failure/no PII |
| tests/evaluation/test_gui_run.py (new) | fixed validation/config, one training, no publication/test, short synthetic QProcess |
| tests/model/test_mindbox_production_training.py | epoch/precommit cancellation and publication-handshake regression tests |
| README.md | compact modes/config/history/lifecycle documentation |
| diagnostics/train_ui01/ | logs, baseline/final SHA, integrity, git review/status/stat, previews and this report |

## Проверки и ограничения

- Expanded targeted set: `tests/tabs tests/evaluation tests/model/test_mindbox_production_training.py`:
  **564 passed, 45 warnings**. Последние three additional benchmark/progress cases вошли в final full run.
- Final new GUI/history/research files отдельно: **47 passed** (`new_tests.log`).
- Full pytest: **2113 passed, 115 warnings**, baseline 2061 passed/112 warnings.
  Все прежние тесты сохранены; warning types прежние torch.load FutureWarning,
  дополнительные warnings возникают в добавленных synthetic publication checks.
- Ruff modified/new Python files: **All checks passed** (`ruff.log`).
- `git diff --check`: passed, `diff_check.log` пустой.
- `git diff --stat`: **18 files changed, 1906 insertions(+), 816 deletions(-)**.
  Это общий diff относительно HEAD, включая прежнюю незакоммиченную работу;
  новые untracked files в этот stat не включены. Полный вывод в `diff_stat.txt`.
- `git status --short`: сохранён полностью в `git_status.txt`.
- SHA verification: **23/23 protected files unchanged**, production/canonical/settings
  не изменены; existing BPRMF.py и пользовательская start_training.png совпадают
  с состоянием до TRAIN-UI-01. Legacy helper block unchanged. Подробности: `integrity.json`.
- Project training processes отсутствуют. Найденный generic multiprocessing PID 5692
  принадлежит unrelated uvicorn worker; он не завершался (`processes.json`).

BPR objective/sampling/aggregation/optimizer/architecture, temporal targets/candidates
и evaluation formulas не изменены. Production publication serializer/current/rollback
semantics сохранены; изменения только в optional orchestration hooks и GUI workflow.
Audit/GPU-cost/convergence/weight CLI и старые diagnostic artifacts не переписаны.
Real 50-epoch training, weight sweep, второй seed и Stage B не запускались.
Real test performance не считалась. Synthetic publication tests использовали
temporary model roots; пользовательская production generation не заменялась.
Dependencies/environment не менялись. Commit/push/merge/rebase/reset/branch checkout не выполнялись.
