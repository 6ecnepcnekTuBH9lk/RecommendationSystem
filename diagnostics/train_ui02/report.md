# TRAIN-UI-02 — итоговый отчёт

Выполнено на ветке `feature/training-settings`. Все предыдущие незакоммиченные
изменения сохранены. Commit, push, merge, rebase и reset не выполнялись.
Сравнение именно с исходным состоянием TRAIN-UI-01 сохранено в `task.diff`;
общий Git diff включает работу предыдущих задач.

## Вкладка и отображение данных

1. **Компоновка.** Слева сверху — «Параметры» и «Данные для обучения»;
   справа — «Процесс обучения», progress bar и большой журнал;
   снизу на всю ширину — «История экспериментов» и таблица.
2. **Вертикальный разделитель.** Один `QFrame#vSeparator`, ширина 1 px,
   между верхними областями. Использует существующий shared QSS.
3. **Горизонтальный разделитель.** Один `QFrame#hSeparator`, высота 1 px,
   перед областью истории. Использует тот же существующий стиль.
4. **Заголовки.** Все четыре плашки центрированы внутри своих секций,
   `sectionHeader` переиспользуется; ширина плашек определяется содержимым.
5. **Параметры.** Порядок: «Режим обучения:» → «Количество эпох:» →
   «Вес покупки:» → «Вес избранного:» → «Вес просмотра:».
   Defaults сохранены: 50 / 10.0 / 2.0 / 0.5; целочисленная validation эпох прежняя.
6. **Веса.** Только GUI spin boxes используют два десятичных знака и русскую
   локаль: `10,00`, `2,00`, `0,50`. Таблица также форматирует значения только
   при отображении. Backend float, прежние config/history и точность метрик
   не округляются при сохранении; regression test проверяет JSON с весами
   `.123456789` и NDCG `.0123456789`.
7. **Подсказки.** Постоянный `mode_description` удалён в обоих режимах.
   Причины блокировки запуска остаются рядом с кнопками.
8. **До запуска.** Блок показывает «Данные готовы» и три счётчика.
   Без подходящих агрегатов: «Взаимодействий: —», «Пользователей: —»,
   «Товаров: —». Нет данных — «Данные для обучения отсутствуют»;
   повреждённые metadata — «Ошибка проверки данных». Start blocking сохранён.
9. **Источники до запуска.** Readiness использует прежние manifest, canonical
   catalog metadata, имена part files и stat каталога товаров. Счётчики могут
   использовать уже полученные aggregates обычного preflight/preparation;
   cache разделён по режиму и revision данных. Исторические статистические
   агрегаты с analysis filters не подставляются как training counts:
   они не доказывают соответствие текущему snapshot/подготовке.
   На холодном старте при отсутствии подходящих агрегатов остаётся «—».
10. **Нет ingestion при открытии.** Новый тест открывает вкладку с настоящей
    структурой synthetic canonical metadata, запрещает чтение файлов
    `canonical/objects` и строк nomenclature.csv и запрещает ingestion.
    Research корректно блокируется без November coverage, Production доступен.
11. **Во время обучения.** После штатной подготовки отображаются
    взаимодействия, пользователи, товары и обучающие пары; при начале обучения
    статус «Данные используются в обучении». Research: `len(snapshot.history)`
    и уже подготовленные mappings/train_pairs; Production: существующие
    `bpr_events`, `users`, `items`, `train_pairs`. Для research добавлено только
    одно aggregate-поле `training_events` в событие `training` — без новой
    загрузки, изменения результирующего record или schema истории.
12. **После успеха.** «Данные использованы», фактические counts сохраняются
    при refresh и возврате на вкладку/в режим. Новый запуск сбрасывает старые
    counts, смена revision не показывает предыдущие агрегаты для новых данных.
13. **Validation cases.** Не отображаются в data block, progress, журнале,
    итогах или таблице. Dataset/полный JSON результата больше не печатается
    в GUI; технические counters сохраняются в backend/artifacts. Проверено
    для обоих режимов, включая preflight и final metrics.
14. **Секция «Результат».** Заголовок, карточка и зависимости от
    `training_result` удалены. Журнал получил освободившуюся высоту.
15. **Итог в журнале.** Research: итоговый статус, overall NDCG/Recall@10,
    NDCG@10 для просмотров/покупок, эпохи, время, устройство.
    Production: эпохи, время, устройство, результат публикации, поколение;
    research metrics не добавляются. Сводка выводится один раз после окончания
    QProcess; выбор строки истории журнал не меняет.
16. **Локализация.** Загрузка данных, подготовка данных, обучение, эпоха,
    ошибка, валидация/оценка модели, публикация, устройство, поколение,
    проверка данных, отмена и итоговые статусы переведены в presentation layer.
    CPU/CUDA/NDCG/Recall сохранены. Коды диагностических issues и имена
    технических quality metrics остаются в подробной диагностике журнала.
17. **Progress text.** Центрирован. Пример:
    `Эпоха 14 из 50 · Ошибка: 0,045248 · Время: 00:21:32 · CUDA`.
    Итог: `Завершено · 50 из 50 эпох · Время: 01:12:34 · CUDA`.
    Существующий progress/lifecycle, lock полей и отмена сохранены.
18. **История.** Заголовки: Дата, Просмотр, Избранное, Покупка, Эпохи,
    NDCG@10, Recall@10, NDCG@10 (просмотры), NDCG@10 (покупки), Время,
    Устройство, Статус. Между title и таблицей spacing 4 px;
    пустой history_notice скрыт и больше не занимает строку.
19. **Ручной resize.** Все 12 колонок `Interactive`, `stretchLastSection=False`;
    заданы только начальные ширины, горизонтальная прокрутка AsNeeded.
    Refresh таблицы не переустанавливает ширины; regression test проверяет
    сохранение вручную изменённых первой и последней колонок.
20. **Статусы и schema.** `completed` → Завершено, `running` → Выполняется,
    `cancelled` → Отменено, `failed` → Ошибка, `interrupted` → Прервано.
    Persisted codes прежние, `schema_version=1`, migration отсутствует.
    Чтение истории и отображение не меняют исходный JSON; это проверено тестом.

## Preview и изменённые файлы

21. **Screenshots.** Оба снимка получены offscreen Qt с синтетическими событиями,
    без subprocess/trainer. Числа на них иллюстративные.

![Research — синтетический preview](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui02/research_preview.png)

![Production — синтетический preview](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui02/production_preview.png)

22. **Именно TRAIN-UI-02 изменяет пять исходных файлов:**

- [Application/tabs/train_model_tab.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/tabs/train_model_tab.py) — layout, data counts, localization, progress, journal summary, history presentation.
- [Application/evaluation/experiments/gui_run.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/evaluation/experiments/gui_run.py) — одно additive aggregate-поле `training_events` для GUI; training/evaluation logic прежняя.
- [tests/tabs/test_training_research_ui.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/tests/tabs/test_training_research_ui.py) — обновлены ожидания нового UI и добавлена 21 проверка TRAIN-UI-02.
- [tests/evaluation/test_gui_run.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/tests/evaluation/test_gui_run.py) — итог в журнале, точные interactions history без November/December targets, record schema прежняя.
- [README.md](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/README.md) — актуальная компоновка и правила отображения.

Новые диагностические материалы только в `diagnostics/train_ui02/`:
`preview.py`, `check_integrity.py`, две PNG, `report.md`, source snapshots,
SHA-256 baselines/`integrity.json`, `task.diff`/`task_stat.txt`, логи pytest/Ruff,
`diff_check.txt`, `git_diff_stat.txt`, `git_status.txt`.
Артефакты предыдущих экспериментов и диагностик не удалялись.

## Проверки

23. **Targeted:** `pytest tests/tabs tests/evaluation tests/model/test_mindbox_production_training.py -q --tb=short`
    → **588 passed, 45 warnings, 87.73 s**. Включены новые layout/data/log/history
    checks, существующие train tab, GUI history/research/production, relevant
    evaluation и publication tests. Первичная узкая проверка: 94 passed,
    2 warnings; последующая расширенная включает дополнительный metadata test.
24. **Полный pytest:** `pytest -q --tb=short`
    → **2134 passed, 115 warnings, 146.35 s**.
    Baseline 2113 passed, 115 warnings; добавлена 21 проверка,
    новых падений и увеличения числа warnings нет.
25. **Ruff:** изменённые Python-файлы и два диагностических scripts:
    **All checks passed!** Existing `.venv310aboba`, без установки пакетов.
26. **`git diff --check`:** exit 0, вывод пустой.
27. **`git diff --stat`:** общий snapshot приведён ниже; включает предыдущие
    незакоммиченные задачи. Отдельный TRAIN-UI-02 snapshot diff:
    **5 files, +415 / -107** (сравнение с сохранённым TRAIN-UI-01).
28. **`git status --short`:** приведён ниже полностью.

## Сохранение поведения и защищённых файлов

Research contract сохранён: seed 42, history cutoff 2025-11-01 UTC,
November validation, test blind, features off, early_stop=False,
одна config → training → final validation → history, без публикации.
Production использует прежний canonical CLI и PASS/WARN/BLOCK, publication lock,
staging, readback, rollback, current switch и cancellation handshake.

BPR samplers/loss/weights/aggregation/quantity/optimizer/embeddings,
temporal protocol, metrics и final/best state selection не менялись.
`BPRMF.py`, production core/CLI, `training_workflow.py`, GUI history schema,
иконка и legacy helpers имеют тот же SHA-256/содержимое, что перед TRAIN-UI-02.
Все 23 защищённых canonical/catalog/settings файла совпали по SHA-256
до и после проверок. В корневом `model/` файлов нет, `model/current.json`
не появился. Production model и пользовательские данные не изменялись.

Реальное обучение, production run, sweep и оценка реального test benchmark
не запускались. Запускались только existing tests с короткими синтетическими
данными и temporary model roots; synthetic protection tests сохранены.
PyQt, PyTorch, CUDA, Python и dependencies не устанавливались/не обновлялись.

Полные логи: `targeted.txt`, `full.txt`, `ruff.txt`, `integrity.json`.

## Общий Git snapshot

`git diff --stat`:

```text
 Application/model/BPRMF.py                        |  116 ++-
 Application/model/bpr_preparation.py              |   35 +-
 Application/model/mindbox_production_training.py  |   20 +-
 Application/model/mindbox_training_preparation.py |    3 +-
 Application/model/training_data.py                |    3 +
 Application/model/training_metrics.py             |    6 +
 Application/tabs/train_model_tab.py               | 1160 ++++++++++++---------
 README.md                                         |   81 ++
 assets/icons/start_training.png                   |  Bin 15123 -> 24703 bytes
 main.py                                           |    2 -
 scripts/mindbox_production_train.py               |  111 +-
 tests/model/test_bpr_preparation.py               |  109 +-
 tests/model/test_mindbox_production_training.py   |  206 ++++
 tests/model/test_mindbox_training_preparation.py  |  120 ++-
 tests/model/test_seen_items.py                    |    3 +-
 tests/model/test_training_data.py                 |  295 ++++++
 tests/tabs/test_train_model_completion.py         |   11 +-
 tests/tabs/test_train_model_start.py              |  537 +++++-----
 18 files changed, 2009 insertions(+), 809 deletions(-)
```

`git status --short`:

```text
 M Application/model/BPRMF.py
 M Application/model/bpr_preparation.py
 M Application/model/mindbox_production_training.py
 M Application/model/mindbox_training_preparation.py
 M Application/model/training_data.py
 M Application/model/training_metrics.py
 M Application/tabs/train_model_tab.py
 M README.md
 M assets/icons/start_training.png
 M main.py
 M scripts/mindbox_production_train.py
 M tests/model/test_bpr_preparation.py
 M tests/model/test_mindbox_production_training.py
 M tests/model/test_mindbox_training_preparation.py
 M tests/model/test_seen_items.py
 M tests/model/test_training_data.py
 M tests/tabs/test_train_model_completion.py
 M tests/tabs/test_train_model_start.py
?? Application/evaluation/
?? Application/tabs/training_workflow.py
?? diagnostics/
?? docs/
?? scripts/audit_temporal_evaluation.py
?? scripts/run_bpr_convergence.py
?? scripts/run_bpr_gpu_cost.py
?? scripts/run_bpr_weight_experiment.py
?? scripts/run_gui_research.py
?? tests/evaluation/
?? tests/tabs/test_training_research_ui.py
```
