# TRAIN-UI-04 — итоговый отчёт

Точечные presentation fixes поверх TRAIN-UI-03 на ветке
`feature/training-settings`. Предыдущие незакоммиченные изменения сохранены.
Task diff сравнивается с сохранённым состоянием UI-03; общий Git diff также
содержит изменения предыдущих задач.

1. **Правый блок.** Compact centered header переименован в
   **«Визуализация обучения»**. Layout, соотношение 4:7, соединённые
   разделители, параметры/процесс слева и история снизу сохранены.

2. **Центрирование placeholder.** Оба сообщения теперь занимают фактический
   `QChart.plotArea()`, преобразованный в координаты viewport, и используют
   одинаковый AlignCenter. Пересчёт происходит при plotAreaChanged/resize/show
   через очередь Qt; скрытые charts откладывают layout до показа.
   учитываются native title, оси и легенда. До исправления центр сообщения
   metrics был примерно на 29 px ниже центра plot area в synthetic preview,
   поскольку общий viewport включает место под легенду. Фиксированного
   корректирующего сдвига нет. До первого event series по-прежнему пустые.

3. **Заголовки графиков.** «ОШИБКА ОБУЧЕНИЯ» и «МЕТРИКИ ВАЛИДАЦИИ» — верхний
   регистр, **12 pt, italic**, как `statisticsSection` в существующем shared
   QSS вкладок статистики второго уровня. Общий `style_chart` получил
   необязательный `section_title=True`; training включает его. Цвета, family,
   сетка и палитра остаются общими. Default оформления существующих
   StatisticsChart не менялся. Тест сравнивает font со styled statistics QLabel.

4. **Ось эпох.** Для обычного 50-epoch chart достаточной ширины диапазон
   **1–50**, tick anchor/interval **1**, подписи всех **1, 2, …, 50**, font **8 pt**.
   Отдельные подписи используют chart-owned `QGraphicsTextItem`, по тому же
   принципу, что bar/value labels у StatisticsChart. Это необходимо потому,
   что native QtCharts автоматически заменял двузначные dense labels на `…`,
   в том числе при повороте. Native ось сохраняет scale/tick/grid/layout;
   её дублирующие labels прозрачные, custom labels не elide и не перекрываются.
   При узкой области шаг подписей адаптируется по ширине текста; для budgets
   до 100 minor ticks/grid сохраняют отметку каждой эпохи. Первая и final epoch
   подписаны. Для больших budgets количество подписей ограничено доступной
   шириной, примерно до 100; тысячи minor ticks не создаются. Это только
   presentation оси: loss points и частота validation не менялись.

5. **Выравнивание кнопок.** Общая строка, Fixed vertical policy и AlignVCenter
   сохранены. Теперь обе кнопки имеют **fixed height 36 px**, одинаковые
   contents margins 0, stylesheet margin 0 и padding `2px 14px`. Нельзя получить
   разные высоты из разных sizeHint при enabled/disabled или смене текста
   Research/Production. Проверены реальные top/bottom рамки и native
   `SE_PushButtonContents`, font/icon sizes в обеих темах и состояниях
   idle/running/completed. В исходном offscreen sample при текущей теме геометрия
   уже совпадала: обе рамки y=537, h=36, content y=3, h=30. Новая настройка
   явно закрепляет этот invariant и исключает различия inherited margin/padding.

6. **Failure icon сохранена.** Existing
   [failure.png](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/assets/icons/failure.png),
   QIcon, размер 17×17. В измеренном sample иконка не увеличивала высоту Cancel.
   Никакие image assets не изменялись; тест сравнивает icon pixmap с исходным.

7. **Файлы.** Изменены только четыре исходных файла; новых production modules
   нет. Перечень с назначением приведён ниже. Добавлены только диагностические
   материалы в `diagnostics/train_ui04/`. README и workflow/evaluator/history
   для этой presentation-задачи не менялись.

8. **Tests.** Обновлено ожидание compact header. Добавлено **9** проверок:
   центрирование обоих placeholder при двух размерах и изменении легенды;
   font/style заголовков и неперекрывающиеся labels 1–50; narrow/large-budget
   axis policy; четыре комбинации темы/режима для геометрии кнопок в трёх
   состояниях и сохранения failure icon; deferred initialization скрытой вкладки
   при смене темы. Screenshot pixel equality не используется.
   В первой попытке новых тестов было три падения: renderer оставался в скрытом
   parent fixture. Тесты исправлены так, чтобы показывать реальное родительское
   окно/QTabWidget. После этого проверка прошла; финальная версия отдельно
   подтверждает фактические custom label strings без многоточий. При переходе
   на queued layout ещё одна промежуточная проверка выявила слишком строгий
   запас ширины подписи; плотность скорректирована с проверкой реального
   отсутствия перекрытия. Все первоначальные логи сохранены.

9. **Targeted.** `pytest tests/tabs tests/evaluation tests/test_statistics_charts.py -q --tb=short`
   → **606 passed, 4 warnings, 144.27 s**. Включены train tab, history,
   Research/Production GUI, existing observer/parity/validation-only evaluation
   coverage и StatisticsChart. После последней точечной правки подписи final
   epoch выполнена финальная узкая проверка:
   `pytest tests/tabs/test_training_research_ui.py tests/test_statistics_charts.py -q --tb=short`
   → **103 passed, 12.68 s**. После Qt lifecycle fix финальная проверка
   `pytest tests/tabs/test_training_research_ui.py tests/tabs/test_reference_loading_section.py tests/test_statistics_charts.py -q --tb=short`
   → **112 passed, 12.91 s** (финальный повтор). Включён existing main-window theme smoke и новый
   hidden-chart regression. Падений в финальной проверке нет.

10. **Full pytest.** Финальная версия: **2172 passed, 115 warnings, 200.77 s**,
    exit **0**. Baseline **2163 passed, 115 warnings**: добавлено 9 проверок,
    новых warnings нет. Два первых полных запуска остановились на 69% с
    native Windows access violation (`0xC0000005`) в `QApplication.setStyleSheet`
    при existing main-window theme test. Логи сохранены в `full_initial_crash.txt`
    и `full_retry.txt`.
    Изолированный main-window test прошёл: **1 passed, 3.66 s**. Synthetic
    stress probe — 20 циклов create/resize/deferred-delete, 40 графиков и 60
    смен темы — тоже прошёл. Исправлен потенциально опасный reentrant callback:
    `_update_plot_layout` теперь QObject slot с queued connection/invocation;
    он меняет axes/text items после native stylesheet/layout traversal, только
    для видимого chart. Точная native C++ причина по Python stack не определяется;
    Полный запуск после исправления прошёл без native crash:
    **2170 passed, 2 failed, 115 warnings**; оба падения — новые placeholder
    assertions, сравнивавшие rectangle до завершения queued layout (разница
    2 px ширины). Тесты теперь обрабатывают события native/queued layout и
    проверяют центр с допуском 1 px на округление, без screenshot equality.
    Финальный full run прошёл полностью, без native crash и assertion failures.
    Production UI code после устранения native crash не менялся.

    **High — `Application/training_charts.py`, `TrainingChart._update_plot_layout`.**
    Исходный callback синхронно менял axis/text layout из plotAreaChanged и
    style/resize handling, что создаёт риск reentrancy внутри native Qt layout.
    Это влияет на стабильность; точная связь с C++ access violation не доказана
    Python stack, но два запуска до исправления падали, два после него завершились
    без native crash. Реализованы queued QObject slot и deferred hidden-chart
    initialization. Риск изменения низкий: визуальный layout обновляется на
    следующем проходе event loop, расчёты/данные/lifecycle worker сохраняются.
    Добавлен hidden-chart regression; финальный полный pytest прошёл.

11. **Ruff.** Три production UI/chart файла, test file и пять diagnostic
    scripts: **All checks passed!** Использован existing `.venv310aboba`.

12. **git diff --check.** Exit **0**, вывод пустой. Изолированный task diff
    просмотрен; workflow body от `def _research` до конца train tab совпадает
    с baseline побуквенно.

13. **git diff --stat.** Ниже общий snapshot с предыдущими задачами. Именно
    TRAIN-UI-04 относительно UI-03: **4 files, +241 / -15**. Общий Git stat не
    включает already-untracked training chart/test files, поэтому для scope
    задачи сохранены отдельные `task.diff` и `task_stat.txt`.

14. **git status --short.** Полный snapshot ниже. Ветка
    `feature/training-settings`; commit/push/merge/rebase/reset не выполнялись.

15. **Screenshots.** Новые offscreen Qt synthetic preview:
    [Research idle](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui04/research_idle.png),
    [Research completed](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui04/research_completed.png).
    Дополнительно сохранены
    [Research running](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui04/research_running.png)
    и [Production](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui04/production.png).
    Размер 1520×1020. Idle/completed визуально проверены: новый header, одинаковое
    центрирование внутри plot area, italic uppercase titles, все подписи 1–50
    и выровненные кнопки. Все числа/CUDA на снимках синтетические; FakeProcess
    не запускает child/trainer и не публикует модель.

Изменённые исходные файлы:

| Файл | Изменение |
|---|---|
| [Application/tabs/train_model_tab.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/tabs/train_model_tab.py) | Новый header и одинаковые фиксированные geometry/margins/padding кнопок. |
| [Application/training_charts.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/training_charts.py) | Plot-area placeholder, title presentation, adaptive epoch ticks, читабельные chart-owned labels и безопасный queued layout lifecycle. |
| [Application/statistics_charts.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/statistics_charts.py) | Optional statistics-section title font в shared style helper, прежний default сохранён. |
| [tests/tabs/test_training_research_ui.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/tests/tabs/test_training_research_ui.py) | Обновлён header, добавлены 9 regressions для placeholder/title/axis/buttons и hidden chart lifecycle. |

Новые материалы `diagnostics/train_ui04/`:

| Материалы | Назначение |
|---|---|
| `capture_baseline.py`, `before_hashes.json`, `protected_baseline.json`, четыре файла `before_sources/` | Baseline 190 исходных/contract файлов и исходники четырёх task files до изменений. |
| `before_preview.py`, `before_preview.txt`, четыре PNG и четыре geometry JSON в `before/` | Synthetic измерение исходного положения placeholder и geometry/content rect кнопок. |
| `preview.py`, `preview.txt`, четыре PNG и четыре `.png.json` | Обновлённые synthetic previews и численные geometry/tick measurements. |
| `check_integrity.py`, `integrity.json`, `integrity_check.txt`, `task.diff`, `task_stat.txt` | Проверка scope, защищённых файлов и изолированный diff UI-04. |
| `initial_tests.txt`, `initial_tests_repaired.txt`, `final_ui_tests.txt`, `queued_ui_tests.txt`, `queued_ui_tests_final.txt`, `final_targeted.txt`, `targeted.txt`, `full.txt`, `full_initial_crash.txt`, `full_retry.txt`, `full_after_queue.txt`, `full_final.txt`, `ruff.txt` | Полные test/lint логи, включая первичные и финальные проверки. |
| `theme_reproduction.txt`, `theme_stress.py`, `theme_stress.txt` | Изолированный main-window test и synthetic probe смены темы/resize/deferred deletion после native crash. |
| `diff_check.txt`, `git_diff_stat.txt`, `git_status.txt`, `global_review.diff`, `report.md` | Git evidence и этот отчёт. |

Сохранение поведения подтверждено SHA-256 baseline: изменились только четыре
указанных файла из 190. `BPRMF.py`, `training_workflow.py`, `gui_run.py`,
temporal/evaluation modules, history v1, production core/CLI, requirements,
README, start/failure assets совпадают с состоянием UI-03. Тело всех workflow
функций train tab также совпадает с baseline. Sparse schedule **1, 5, 10, 20,
30, 40, 50** не менялся. Seed/config, early_stop=False, test-blind protection,
observer semantics, production publication/cancel semantics, data log, status
lifecycle и история сохранены.

Все 23 защищённых canonical/catalog/settings файла совпадают по SHA-256.
В корневом `model/` файлов нет; `model/current.json` не появился. Real long
training, production training/publication, sweep и scoring реального test
benchmark не запускались. PyQt/PyTorch/CUDA/Python/dependencies не менялись,
не устанавливались и не обновлялись. Предыдущие artifacts не удалялись.

![Research idle — синтетический preview](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui04/research_idle.png)

![Research completed — синтетический preview](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/train_ui04/research_completed.png)

`git diff --stat` (включая предыдущие задачи):

```text
 Application/model/BPRMF.py                        |  116 +-
 Application/model/bpr_preparation.py              |   35 +-
 Application/model/mindbox_production_training.py  |   20 +-
 Application/model/mindbox_training_preparation.py |    3 +-
 Application/model/training_data.py                |    3 +
 Application/model/training_metrics.py             |    6 +
 Application/statistics_charts.py                  |   78 +-
 Application/tabs/train_model_tab.py               | 1198 ++++++++++++---------
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
 19 files changed, 2107 insertions(+), 837 deletions(-)
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
