# TRAIN-UI-03 — итоговый отчёт

Выполнено на ветке `feature/training-settings`. Сохранены предыдущие
незакоммиченные изменения. Ниже описаны изменения относительно состояния после
TRAIN-UI-02; общий Git diff включает также предыдущие задачи.

1. **Layout hierarchy.** Корневой `QVBoxLayout`: верхний `QHBoxLayout`,
   горизонтальный разделитель, история на всю ширину. Верхняя строка:
   `trainingLeftPanel` → вертикальный разделитель → `trainingChartsPanel`.
   Слева находятся «Параметры», форма, «Процесс обучения», progress text/bar,
   журнал и общая строка кнопок. Справа — «Графики обучения» и два line charts.
   Внизу — «История экспериментов» и таблица.

2. **Стык разделителей.** У root и верхнего layout все margins и spacing равны
   нулю. Вертикальный `QFrame#vSeparator` занимает всю высоту верхней строки;
   следующий элемент root — `QFrame#hSeparator`. Между ними нет layout gap.
   Padding перенесён внутрь панелей: `(8, 0, 8, 8)`; история имеет собственные
   margins `(8, 4, 8, 0)`. Сохранены shared QSS, одна вертикальная линия шириной
   1 px и одна горизонтальная высотой 1 px. Дополнительных нарисованных линий нет.

3. **Stretch.** Левая и правая панели имеют коэффициенты **4:7** — примерно
   36:64% доступной ширины. Фиксированная ширина панелей не задана. Верх/история
   сохраняют stretch **3:2**. Правый chart использует масштабируемую size policy.

4. **Блок данных удалён.** Нет заголовка «Данные для обучения»,
   `training_readiness` и `training_data_counts`. Readiness, проверка параметров
   и причины блокировки запуска сохранены. Порядок полей и defaults прежние:
   режим «Эксперимент», 50 эпох, покупка `10,00`, избранное `2,00`, просмотр
   `0,50`. Низкоуровневые параметры не возвращались; карточки «Результат» нет.

5. **Available counts в журнале.** При старте выводятся название запуска,
   «Проверка данных...» и «Данные готовы». Если уже есть подходящий cache
   агрегатов того же режима/revision, выводятся взаимодействия, пользователи
   и товары. Агрегаты штатного production preflight также могут попасть в
   этот журнал. При отсутствии metadata counts выводится «Точные объёмы будут
   определены при подготовке». Нули не подставляются. Старое число обучающих
   пар не переносится в новый запуск с другими весами.

6. **Exact counts после подготовки.** Existing structured preparation/training
   events дают четыре значения текущего run. Research: `len(snapshot.history)`,
   размеры подготовленных user/item mappings и `train_pairs`. Production:
   existing `bpr_events`, `users`, `items`, `train_pairs`. Они выводятся под
   «Подготовка данных завершена. Для обучения подготовлено:».
   Повторное событие с теми же значениями не дублирует блок. Числа из задания
   не зашиты в production-код. Validation cases остаются внутренними counters
   и не попадают в журнал, progress, статусы, историю или графики.

7. **Дополнительного ingestion нет.** Новый код читает только уже полученные
   скалярные агрегаты. Existing metadata readiness сохранена. Tests запрещают
   ingestion/preflight preparation в GUI и чтение canonical objects/строк
   каталога при открытии вкладки. Смена режима/revision не подменяет counts.

8. **Процесс слева.** Компактный центрированный заголовок, центрированный
   статус, progress bar, read-only журнал со stretch 1 и кнопки теперь внутри
   левой панели. Журнал — одна хронология загрузки, подготовки, эпох, sparse
   оценки и итогов. Пользовательские строки русские; CPU/CUDA/NDCG/Recall
   сохранены. Причина блокировки видна при невозможности старта; во время run
   состояние отражают progress и общий статус.

9. **Правая область.** Два `TrainingChart` на уже имеющемся native QtCharts:
   loss и validation NDCG/Recall@10. Общая палитра, fonts, axes/grid и series
   styling извлечены из существующего StatisticsChart в `style_chart`.
   Статистические расчёты/данные не изменены. Регрессии статистических графиков
   прошли. В `.venv312` уже были PyQt6/PyQt6-Charts/PyQt6-Charts-Qt6 **6.10.0**
   и qt-material **2.17**; requirements уже содержали QtCharts.

10. **Loss chart.** Каждое structured epoch event добавляет `(epoch, loss)`.
    До первого события — пустые series и компактный placeholder. Русская
    локаль, целые подписи эпох, автоматический диапазон Y. Повторная точка той
    же эпохи заменяется, не дублируется; неверные/нечисловые значения отсекаются.
    Точки сохраняются после успеха, отмены и ошибки; новый run очищает series.
    Анимация выключена; обновления происходят по событиям, без model/GPU polling.

11. **Validation chart.** Только overall NDCG@10 и Recall@10 на existing
    внешнем November validation snapshot. Existing CONV-01A `epoch_observer`
    используется внутри одного непрерывного trainer/optimizer. Scoring
    выполняется в worker под `torch.no_grad()` с guard validation-only и
    проверкой сохранения parameter versions/model mode. GUI получает только
    epoch/ndcg/recall/device. Последняя точка переиспользует штатную final
    validation; дополнительного final @10 scan нет. Схема возвращаемого result
    не менялась. Intermediate trajectory сейчас живёт в structured event
    stream и chart, отдельно в history/result JSON не сохраняется.

12. **Sparse schedule.** Budget 1–10: каждая эпоха. Для budget >10:
    `{1, round(10%), round(20%), round(40%), round(60%), round(80%), budget}`,
    округление положительных значений half-up, sorted unique. Для 50:
    **1, 5, 10, 20, 30, 40, 50**. Для 200:
    **1, 20, 40, 80, 120, 160, 200**. Schedule зависит только от budget и
    одинаков для разных весов при одинаковом количестве эпох. Дополнительной
    настройки GUI нет. Intermediate scoring time исключается из
    `training_seconds`; существующее wall-clock total включает все этапы.
    Дополнительная стоимость validation на реальных данных здесь не измерялась.

13. **Нет early stopping от графиков.** `early_stop=False` сохранён. Observer
    не выбирает best epoch, не восстанавливает checkpoint, не меняет веса,
    optimizer или budget и не запускает следующую конфигурацию. Выполняются
    все запрошенные эпохи, если нет Cancel/error. Final metrics относятся
    к final epoch.

14. **Final-state parity.** Новый synthetic regression сравнивает три эпохи
    с observer и без него при одинаковых seed/config: каждый tensor final
    state совпал с `rtol=0, atol=0`; совпали loss/observations. Проверены один
    trainer, один Adam, отсутствие перезапуска optimizer, неизменность state
    и mode во время scoring, `best_epoch=-1`, отсутствие publication.
    Final overall @10 не оценивается повторно. Проверена отмена во время
    checkpoint scoring до следующей эпохи. Existing convergence/observer
    coverage тоже прошла. Проверка parity синтетическая, на CPU.

15. **Production.** Metric chart скрыт, loss получает всю оставшуюся высоту.
    Research evaluator не добавлен в production-путь, fake zero metrics нет,
    research metrics не выводятся. Existing canonical CLI, preflight
    PASS/WARN/BLOCK и WARN confirmation, publication lock, staging, readback,
    rollback, atomic current switch и cancellation handshake сохранены.

16. **Общий status_label.** Structured loading/preparation/training/epoch/
    validation/publication обновляют existing global status соответствующим
    русским текстом. При начале/смене активной фазы прежний status-reset timer
    останавливается, чтобы timeout предыдущей операции не сбросил active run
    в idle. После промежуточной оценки статус возвращается к обучению;
    финальная оценка остаётся оценкой до завершения процесса.

17. **Lifecycle success/cancel/failure.** Общий `_release_training` для обоих
    режимов снимает active state, останавливает progress timer, освобождает
    временные файлы запуска, обновляет readiness/inputs и планирует reset
    статуса через existing общий timer (**5 секунд**). Start включается при
    readiness, Cancel выключается. После краткого итога — «Готов к работе».
    Проверены success/cancel/failure/failed-to-start в обоих режимах и stop
    старого timeout при новом запуске. Новая система таймеров не добавлялась.

18. **Start/Cancel.** Одна строка `QHBoxLayout`, margins 0, одинаковая minimum
    height **36 px**, vertical policy Fixed, AlignVCenter; ширины stretch 3:1.
    На всех четырёх preview верх/низ кнопок совпадают.

19. **Иконка отмены.** Используется существующая
    [failure.png](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/assets/icons/failure.png)
    через тот же QIcon/assets helper; icon size 17×17. Asset не изменён.

20. **История заполняет ширину.** `stretchLastSection(True)` отдаёт остаток
    viewport последней колонке «Статус». Большой пустой участок справа убран;
    содержимое таблицы и её рамка используют всю нижнюю область.

21. **Ручной resize.** Все sections по-прежнему Interactive; обычные колонки
    можно изменять мышью, refresh их не переустанавливает. Последняя гибкая
    колонка заполняет остаток; при суммарной ширине больше viewport работает
    horizontal ScrollBarAsNeeded. Initial widths сохранены. Проверены
    сохранение ручной ширины и горизонтальная прокрутка при увеличении колонки.

22. **History JSON v1.** `gui_history.py` не изменён (SHA-256 совпадает).
    Storage enums/schema, значения и точность backend metrics прежние;
    final metrics остаются метриками final epoch. Чтение/выбор строки не
    переписывают JSON и не заменяют журнал текущего запуска. Migration нет.

23. **Screenshots.** Четыре offscreen Qt preview размером 1520×1020; FakeProcess
    не запускает child/trainer. Все числа и надпись CUDA на снимках синтетические,
    это иллюстрация состояний UI, а не результат измеренного обучения.
    Снимки визуально проверены: соединённые линии, 4:7 панели, пустые charts
    до run, live points, заполненная история, выровненные кнопки, failure icon,
    отсутствующие Data/Result blocks и production без metric chart.

24. **Изменённые файлы.** TRAIN-UI-03 меняет шесть существующих исходных файлов
    и добавляет один widget; перечень и назначение приведены ниже. Созданы
    материалы только в `diagnostics/train_ui03/`. Предыдущие experiment artifacts
    и диагностики сохранены. Сравнение 189 исходных baseline files подтвердило,
    что остальные не изменились.

25. **Targeted tests.** Узкая проверка train UI/history/research, convergence и
    StatisticsChart: **176 passed, 2 warnings, 27.95 s**. Расширенная:
    `pytest tests/tabs tests/evaluation tests/model tests/test_statistics_charts.py -q --tb=short`
    → **1165 passed, 111 warnings, 125.62 s**. Включены production GUI,
    publication/cancel coverage и relevant model/evaluation tests. Все прошли.

26. **Full pytest.** `pytest -q --tb=short` в `.venv312`, Qt offscreen:
    **2163 passed, 115 warnings, 157.46 s**. Baseline **2134 passed,
    115 warnings**: добавлено 29 проверок, падений и новых warnings нет.

27. **Ruff.** Изменённые Python-файлы, `preview.py` и `check_integrity.py`
    проверены существующим `.venv310aboba`: **All checks passed!**
    Пакеты не устанавливались.

28. **git diff --check.** Exit **0**, вывод пустой. Task diff просмотрен.

29. **git diff --stat.** Общий snapshot ниже включает предыдущие
    незакоммиченные изменения; untracked files Git stat не считает.
    Изолированный diff TRAIN-UI-03 относительно сохранённого UI-02:
    **7 files, +570 / -137**, включая новый chart и изменения уже untracked
    research/test files. Полный изолированный diff сохранён в `task.diff`.

30. **git status --short.** Полный snapshot приведён ниже. Ветка осталась
    `feature/training-settings`. Commit/push/reset/checkout/merge/rebase
    не выполнялись.

Изменённые исходные файлы:

| Файл | Назначение |
|---|---|
| [Application/tabs/train_model_tab.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/tabs/train_model_tab.py) | Компоновка, data log, chart events, общий status lifecycle, кнопки, flexible history. |
| [Application/training_charts.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/training_charts.py) — новый | Два простых native Qt line charts с placeholder, event updates, reset и общей темой. |
| [Application/statistics_charts.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/statistics_charts.py) | Извлечение общего style_chart для переиспользования существующей палитры/оформления. |
| [Application/evaluation/experiments/gui_run.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/evaluation/experiments/gui_run.py) | Sparse read-only validation через existing epoch observer; final point reuse; accounting scoring time. |
| [tests/tabs/test_training_research_ui.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/tests/tabs/test_training_research_ui.py) | Layout/buttons/log/graphs/history, production presentation, общий lifecycle и timer regression tests. |
| [tests/evaluation/test_gui_run.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/tests/evaluation/test_gui_run.py) | Schedule, bit-exact parity, no-grad/read-only observer, один optimizer, no duplicate final scan, cancellation. |
| [README.md](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/README.md) | Краткая документация новой UI semantics и sparse observation policy. |

Диагностические материалы в `diagnostics/train_ui03/`:

| Материалы | Назначение |
|---|---|
| `preview.py`, `research_idle.png`, `research_running.png`, `research_completed.png`, `production.png` | Воспроизводимые синтетические preview без обучения/публикации. |
| `check_integrity.py`, `before_hashes.json`, `protected_baseline.json`, `integrity.json`, `integrity_check.txt` | SHA-256, неизменность защищённых файлов/contracts, проверка состава task changes. |
| `before_sources/Application/tabs/train_model_tab.py`, `before_sources/Application/statistics_charts.py`, `before_sources/Application/evaluation/experiments/gui_run.py`, `before_sources/tests/tabs/test_training_research_ui.py`, `before_sources/tests/evaluation/test_gui_run.py`, `before_sources/README.md` | Сохранённое состояние до задачи для точного diff. |
| `task.diff`, `task_stat.txt`, `global_review.diff`, `git_diff_stat.txt`, `git_status.txt`, `diff_check.txt` | Изолированный task diff и общий Git snapshot. |
| `initial_tests.txt`, `targeted_initial.txt`, `targeted.txt`, `full.txt`, `ruff.txt` | Логи первичной, targeted/расширенной/полной проверки и Ruff. |
| `report.md` | Этот подробный отчёт по 30 пунктам. |

Подтверждения сохранения:

Все **23** защищённых canonical/catalog/settings файла совпадают по SHA-256.
`model/` остаётся без файлов; `model/current.json` отсутствовал и не появился.
Production model не изменена. `BPRMF.py`, `training_workflow.py`, production
core/CLI, history schema, start/failure icons, requirements и legacy helpers
совпадают с baseline. BPR mathematics — architecture, sampling, weighted loss,
aggregation/quantity, optimizer/LR/regularization/embeddings/state selection —
не менялась. Temporal protocol и test benchmark не менялись.

Research сохраняет seed 42, historical cutoff, November validation, test blind,
features disabled, early_stop=False, одну вручную выбранную configuration,
requested epoch budget и отсутствие публикации. Единственное дополнение —
read-only intermediate validation observations. Production semantics сохранены.
Training/evaluation остаются вне GUI thread; GUI получает structured events.

Реальное длительное training, production run, weight sweep и scoring реального
test benchmark не запускались. Выполнялись только короткие synthetic tests и
preview с временными config/model roots. PyQt/PyTorch/CUDA/Python и dependencies
не устанавливались, не обновлялись и не изменялись.

Синтетические preview:

[Research idle](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui03/research_idle.png) ·
[Research running](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui03/research_running.png) ·
[Research completed](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui03/research_completed.png) ·
[Production](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui03/production.png).

![Research completed — синтетический preview](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui03/research_completed.png)

![Production — синтетический preview](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui03/production.png)

`git diff --stat` (общий, включая предыдущие задачи):

```text
 Application/model/BPRMF.py                        |  116 +-
 Application/model/bpr_preparation.py              |   35 +-
 Application/model/mindbox_production_training.py  |   20 +-
 Application/model/mindbox_training_preparation.py |    3 +-
 Application/model/training_data.py                |    3 +
 Application/model/training_metrics.py             |    6 +
 Application/statistics_charts.py                  |   73 +-
 Application/tabs/train_model_tab.py               | 1196 ++++++++++++---------
 README.md                                         |   91 ++
 assets/icons/start_training.png                   |  Bin 15123 -> 24703 bytes
 main.py                                           |    2 -
 scripts/mindbox_production_train.py               |  111 +-
 tests/model/test_bpr_preparation.py               |  109 +-
 tests/model/test_mindbox_production_training.py   |  206 ++++
 tests/model/test_mindbox_training_preparation.py  |  120 ++-
 tests/model/test_seen_items.py                    |    3 +-
 tests/model/test_training_data.py                 |  295 +++++
 tests/tabs/test_train_model_completion.py         |   11 +-
 tests/tabs/test_train_model_start.py              |  537 ++++-----
 19 files changed, 2100 insertions(+), 837 deletions(-)
```

`git status --short`:

```text
 M Application/model/BPRMF.py
 M Application/model/bpr_preparation.py
 M Application/model/mindbox_production_training.py
 M Application/model/mindbox_training_preparation.py
 M Application/model/training_data.py
 M Application/model/training_metrics.py
 M Application/statistics_charts.py
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
?? Application/training_charts.py
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
