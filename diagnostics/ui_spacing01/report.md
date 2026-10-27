# UI-SPACING-01 — итоговый отчёт

Ветка `feature/training-settings`, checkpoint
`a6c124851e9d65c21c96c4a08d37a167ad4dfa8f` — Refactor training pipeline and research UI.
До работы локально был изменён только `main.py`: пользователь уже удалил legacy
start_train margin. Эта правка сохранена. Commit/push/reset/checkout/merge/rebase
не выполнялись. Сравнение именно с рабочим состоянием до задачи — `task.diff`.

1. **Shared module.** Создан
   [layout_metrics.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/theme/layout_metrics.py).
   Пять простых presentation constants и короткий developer docstring,
   без config-framework или дополнительных зависимостей.

2. **Финальная шкала.** COMPACT=**4**, CONTROL=**8**, BLOCK=**12**,
   SECTION=**18**, PAGE_MARGIN=**8** px. Compact — text/status icon и связанный
   progress text; Control — соседние controls/action buttons; Block — форма,
   helper/status, progress/log и внутреннее содержимое секции; Section —
   самостоятельные секции; Page — внутренние границы панелей. Отдельная
   COLUMN_SPACING не понадобилась. Это не механическая замена всех чисел на 8.

3. **Мигрированы все пять основных вкладок.** Получение данных вместе со
   справочниками, настройка анализа, статистика с пятью вторичными страницами,
   обучение модели, результаты. Проверен main window и reusable MultiSelect.
   Chart geometry, column widths, icon sizes, specialized/adaptive bounds
   оставлены отдельными от шкалы.

4. **Qt defaults заменены явными значениями.** Panel margins; API/customer
   period rows, selection/state/manual grids и action rows; analysis popup,
   heading rows, groups/date/product forms, Apply/Reset; statistics cards и
   toolbar; training parameters form, Parameters/Process sections, progress
   text/bar; result control rows, wrappers и client forms; main layout/status bar.

5. **Структурные нули сохранены.** Root/top layouts Data и Training, root
   Analysis, root/separator area Results; technical wrappers, внутренние
   section containers; statistics progress overlay остаётся с margins/spacing 0,
   text/bar занимают одну grid cell. Results client grid имеет одну строку:
   vertical spacing 0 сохранён намеренно. Separator width/height 1 px прежние.

6. **Получение данных.** Panel margins 8; root sections 18; внутренние API,
   operation state и manual blocks 12. Period/selection/state/manual field rows
   и соседние buttons — 8. Resume использует 12. Reference selectors — 8,
   button/status block — 12; text/icon — 4, между status groups — 8.
   Status groups получили естественную maximum size policy и trailing stretch,
   чтобы свободная ширина оставалась справа, не раздвигала indicators.
   Убраны presentation padding префикса и хвостовые пробелы отображаемых
   status captions; ключи файлов, lookup и import behavior прежние.
   Operation log имеет panel margins 8. Scrollable left panel сохранён.

7. **Настройка анализа.** Обе панели используют Page 8 / Block 12;
   Apply/Reset и popup buttons — Control 8, формы — 8 по обеим осям.
   Перед store mapping добавлен semantic delta SECTION−BLOCK, доводящий
   межсекционный gap до 18. Сохранена исходная прямая hierarchy левой панели:
   дополнительная вложенность при первоначальной миграции нарушала equal-column
   invariant при width 1200/light; она устранена. `_size_inputs` сохраняет
   исходные численные adaptive thresholds 18 через SECTION_SPACING, а не новый
   BLOCK=12. Narrow/wide wrapping, dynamic field height по sizeHint, maximum
   widths, popup и mapping behavior сохранены; AST-нормализация подтвердила
   идентичность расчётов `_size_inputs` с заменой имени прежней константы.
   Dependency analysis → statistics ради spacing удалена.

8. **Статистика.** Main content — Block 12 / Page 8, cards grid — 12,
   внутренний card text/number — Compact 4. Toolbar соседних кнопок — 8.
   Вторичные страницы используют Section 18 между самостоятельными группами,
   header/chart/table внутри группы — Block 12. Порядок, captions, число
   charts/tables, данные, immutable display и размеры графиков сохранены.
   Локальный BLOCK_SPACING=18 удалён; imports идут из общего theme module.

9. **Обучение.** Layout LEFT Parameters+Process / RIGHT Visualization /
   BOTTOM History, stretch 4:7, прежние defaults/validation сохранены.
   Panel margins 8; Parameters→Process — Section 18; внутренние блоки — 12,
   form horizontal/vertical spacing — 8; progress text/bar — Compact 4;
   Start/Cancel gap — Control 8. History сохраняет Compact 4 и использует
   Page margins 8. Нет индивидуального stylesheet или height setters на
   обычных кнопках. Failure icon 17×17 сохранена. Data log, loss/metric charts,
   sparse checkpoints, production presentation и status lifecycle прежние.

10. **Результаты.** Две control rows — Control 8, верхние panel margins —
    Page 8 с Block 12 перед separator. Client info forms — 8, межгрупповой
    horizontal gap — Block 12. Нижние panels имеют Page 8 / Block 12;
    separator wrapper spacing 0. Удалён отдельный addSpacing(10).
    Колонки, header resize modes, row preparation, export/inference/photo
    functions не менялись. Table placement полностью принадлежит layouts.

11. **Legacy start margin отсутствует.** В рабочем baseline пользователь уже
    удалил start_train margin 5 px. Правка сохранена; migration не возвращает
    правило и не компенсирует его перемещением Cancel.

12. **apply_static_widget_styles удалён.** Вызов из constructor и сам helper
    удалены после устранения prefix/table cross-tab geometry styling. Поиск
    подтвердил отсутствие production callers. Existing main-window smoke test
    теперь проверяет отсутствие метода вместо его вызова.

13. **Table QSS margins удалены.** Обе копии в Results и MainWindow устранены.
    purchases_table/recs_table имеют пустой individual stylesheet; outer gaps
    задаются panel contentsMargins и Block spacing. QSS отвечает за control
    appearance. Global QPushButton block с height 30px, padding 2px 14px,
    border/radius/typography полностью совпадает с baseline. SectionHeader
    internal padding 8px 32px сохранён, внешний margin теперь 0: все usages
    имеют явный panel/section layout contract, без случайного 8+18 удвоения.

14. **Start/Cancel geometry совпадает.** Новый regression использует полностью
    initialized/shown MainWindow, обе темы и processed Qt events. Проверяются
    actual height, top/bottom, одна QHBox row, отсутствие local stylesheet и
    fixed bounds. Высота определяется QSS/sizeHint. Геометрия также визуально
    совпадает на after screenshots.

15. **Enabled/disabled.** Проверены idle → synthetic running → idle без
    запуска QProcess. Start enabled/Cancel disabled и обратное состояние
    сохраняют одинаковые рамки. Новый общий theme regression проверяет
    неизменность geometry action buttons и section headers на всех пяти tabs
    при light→dark→light. Action-row invariants покрывают API/manual, Analysis,
    Statistics, Training, Results и MultiSelect popup.

16. **Separator intersections сохранены.** Data/Training top/root zero layouts
    соединяют вертикальный и горизонтальный QFrame без gap. Results lower
    separator теперь расположен непосредственно под horizontal frame;
    прежний horizontal wrapper gap 10 удалён, внутренний padding принадлежит
    panels. Проверены layout properties и визуальные стыки, без screenshot
    pixel assertions.

17. **Theme-switch queue fix сохранён.** `Application/training_charts.py`
    совпадает по SHA-256 с baseline. Queued QObject slot / hidden-chart
    deferred initialization не менялись. MainWindow.apply_theme, status timer
    и ThemeSwitch bounds также не менялись; bottom status text/icon gap стал
    Compact 4, соседство с ThemeSwitch — Control 8.

18. **Before screenshots.** `diagnostics/ui_spacing01/before/`:
    `01_data_loading.png`, `02_analysis_filter.png`, `03_statistics.png`,
    `04_training.png`, `05_results.png`; dark counterparts с суффиксом `_dark`.

19. **After screenshots.** `diagnostics/ui_spacing01/after/` — те же пять
    имён и пять dark counterparts. Всего **20 PNG**, до/после при одинаковых
    1920×1080, 96 logical DPI, font/theme, synthetic state и fixed demo dates.
    Фактические size/DPI и button geometry сохранены в `geometry.json` обеих папок.

20. **Visual comparison.** Все пять экранов до/после просмотрены в обеих
    темах. Matching section headers Data/Analysis/Training теперь начинаются
    согласованно с Page 8; controls не склеены; neighboring action gaps — 8;
    убраны лишние внешние header margins и разреженная reference status line.
    Forms имеют сопоставимую density; table frames используют одинаковые
    panel margins; chart/table группы статистики остаются читаемыми, размеры
    chart widgets прежние. Большие пустые области лишь соответствуют
    исходной структуре (например placeholder «В разработке»). Новый color/theme
    design не вводился. Снимки — synthetic UI, не реальные данные/результаты.

21. **Files changed/added.** Восемь presentation source files изменены,
    три существующих test files обновлены; добавлены shared metrics module
    и `test_ui_spacing.py`. Список с назначением ниже. Diagnostic artifacts
    находятся только в `diagnostics/ui_spacing01/`. README не менялся:
    пользовательская документация о spacing constants не добавлялась.

22. **Targeted.** `pytest tests/tabs tests/test_statistics_charts.py -q --tb=short`
    → **459 passed, 2 warnings, 290.37 s**. Включены data/reference/manual,
    analysis adaptivity/popup/mapping, statistics/calculation/export lifecycle,
    training Research/Production UI/history, results/photo and main/theme tests.
    Финальная focused проверка — **127 passed, 31.10 s**. Добавлено **16**
    новых проверок shared scale/layout/geometry. До изменений отдельная
    baseline-проверка выявила **5** устаревших fixed-height assertions; они
    обновлены на согласованный global-QSS contract. Ранее существовавшая
    leading-space подпись Production нормализована в test через strip.
    Первичная новая проверка RowWrapPolicy уточнена с учётом существующей
    dynamic DontWrap/WrapAll behavior. Промежуточный layout regression анализа
    исправлен в реализации, а assertion equal columns сохранён. Все логи
    неудачных и успешных проверок сохранены.

23. **Full pytest.** **2188 passed, 115 warnings, 366.73 s**, exit **0**. Baseline 2172/115: добавлено 16 проверок, новых warnings нет. Native Qt crash не возник.

24. **Ruff.** Scope Python files и diagnostic helpers: **All checks passed!**
    Existing `.venv310aboba`, без установки dependencies.

25. **git diff --check.** Exit **0**. Git сообщает только line-ending notice
    о будущем LF→CRLF при касании test_training_research_ui.py; whitespace
    errors отсутствуют. Task diff просмотрен. Оставшиеся raw zero values
    структурные; numerical layout gaps 10/12/18 в audited files заменены
    semantic constants. Specialized chart/table/icon/adaptive sizes сохранены.

26. **git diff --stat.** Общий Git snapshot приведён ниже/в `git_diff_stat.txt`.
    Изолированный task diff от сохранённого рабочего baseline:
    **13 files, +456 / -173**, включая два новых source/test files.
    Git tracked stat не включает untracked files и учитывает прежнюю локальную
    правку main.py; это отделено от изменений UI-SPACING-01.

27. **git status --short.** Полный snapshot ниже/в `git_status.txt`.
    Ветка и HEAD сохранены; commit/push не выполнялись.

Перечень исходных файлов:

| Файл | Назначение |
|---|---|
| [Application/theme/layout_metrics.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/theme/layout_metrics.py) — новый | Общая semantic spacing scale. |
| [Application/theme/custom.qss](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/theme/custom.qss) | Только внешний margin sectionHeader; internal padding и control sizes сохранены. |
| [Application/tabs/data_loading_tab.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/tabs/data_loading_tab.py) | API/operation/manual section groups, rows/grids/panels/log spacing. |
| [Application/tabs/reference_loading_section.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/tabs/reference_loading_section.py) | Reference rows, compact indicators, natural status-group widths без prefix QSS padding. |
| [Application/tabs/analysis_filter_tab.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/tabs/analysis_filter_tab.py) | Shared metrics вместо cross-tab import, unified panels/forms/buttons/popup; прежние adaptive calculations. |
| [Application/tabs/dataset_statistics_tab.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/tabs/dataset_statistics_tab.py) | Shared card/control/page metrics; explicit major sections и internal block layouts без изменения data rendering. |
| [Application/tabs/train_model_tab.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/tabs/train_model_tab.py) | Explicit Parameters/Process blocks, form/progress/buttons/history spacing. |
| [Application/tabs/create_results_tab.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/Application/tabs/create_results_tab.py) | Control/client forms/panels/section spacing, удаление table QSS margins. |
| [main.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/main.py) | Main/bottom spacing, удаление legacy cross-tab geometry helper и вызова. |
| [tests/tabs/test_ui_spacing.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/tests/tabs/test_ui_spacing.py) — новый | 16 regressions: scale, rows, main-window buttons, panels, compact indicators, separators, dependency/style bounds, theme geometry. |
| [tests/tabs/test_dataset_statistics_tab.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/tests/tabs/test_dataset_statistics_tab.py) | Иерархия секций при сохранении прежних order/content/immutable-render assertions. |
| [tests/tabs/test_reference_loading_section.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/tests/tabs/test_reference_loading_section.py) | Main-window smoke проверяет отсутствие legacy helper. |
| [tests/tabs/test_training_research_ui.py](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/tests/tabs/test_training_research_ui.py) | Existing button tests проверяют global-QSS size policy/bounds и semantic text. |

Диагностические материалы: `capture_baseline.py`, `preview.py`, `check_integrity.py`, `finish_report.py`,
`before_sources/`, source/data hash baselines, `integrity.json`, `integrity_check.txt`,
`task.diff`, `task_stat.txt`, before/after sets PNG+geometry JSON, preview/test/lint
logs, Git snapshot и этот `report.md`. Материалы предыдущих задач не удалялись.

Аудит выявил две **Medium** presentation/maintainability проблемы: геометрия
нескольких вкладок принадлежала MainWindow.apply_static_widget_styles, а analysis
зависел от statistics ради constant с двумя смыслами (spacing/adaptive width).
Реализованы local layouts/shared theme metrics; adaptive value 18 сохранён.
Ожидаемый результат — одинаковые gaps без скрытых cross-tab overrides; риск
ограничен layout geometry. Regression tests добавлены; functional behavior
сохранён. Во время миграции найденный narrow-width regression исправлен с
сохранением исходной hierarchy и прежнего equal-column test.

Подтверждения: **191** baseline source/contract files, изменены только указанные
11 existing files; добавлены ровно два source/test files. **190** проверок AST
методов вне presentation, включая нормализованный `_size_inputs`, совпали.
Все **23** защищённых canonical/catalog/settings файла совпали по SHA-256.
BPR/evaluation/temporal/Research/Production/workflow/history/math/dependencies,
statistics calculation/export, loading/controllers, filters/mapping and
recommendation/photo processing не менялись. Production model не изменена.
Queue fix TrainingChart и global QPushButton QSS совпадают с baseline.

Fixed/min/max heights обычным QPushButton не добавлялись. Existing specialized
DateEdit/MultiSelect sizeHint-based heights, chart/table bounds и ThemeSwitch
dimensions сохранены. Common action icon 17×17 сохранён. Real Mindbox export,
full-dataset statistics, real BPR training/temporal benchmark/production inference
не запускались. Synthetic tests и pure UI previews использовали временные
settings/config/model roots; preview loaders отключены. Commit/push отсутствуют.

Screenshots и Git snapshot добавлены ниже.

| Вкладка | Before light | After light | Before dark | After dark |
|---|---|---|---|---|
| Получение данных | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/before/01_data_loading.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/after/01_data_loading.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/before/01_data_loading_dark.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/after/01_data_loading_dark.png) |
| Настройка анализа | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/before/02_analysis_filter.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/after/02_analysis_filter.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/before/02_analysis_filter_dark.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/after/02_analysis_filter_dark.png) |
| Статистика | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/before/03_statistics.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/after/03_statistics.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/before/03_statistics_dark.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/after/03_statistics_dark.png) |
| Обучение | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/before/04_training.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/after/04_training.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/before/04_training_dark.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/after/04_training_dark.png) |
| Результаты | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/before/05_results.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/after/05_results.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/before/05_results_dark.png) | [PNG](C:/Users/Ermolenko.i/PycharmProjects/РекомендательнаяСистема/diagnostics/ui_spacing01/after/05_results_dark.png) |

git diff --stat:

```text
 Application/tabs/analysis_filter_tab.py       |  24 ++++-
 Application/tabs/create_results_tab.py        |  36 ++++---
 Application/tabs/data_loading_tab.py          |  67 ++++++++-----
 Application/tabs/dataset_statistics_tab.py    | 136 ++++++++++++++------------
 Application/tabs/reference_loading_section.py |  21 ++--
 Application/tabs/train_model_tab.py           |  44 ++++++---
 Application/theme/custom.qss                  |   2 +-
 main.py                                       |  18 +---
 tests/tabs/test_dataset_statistics_tab.py     |  66 ++++++++-----
 tests/tabs/test_reference_loading_section.py  |   2 +-
 tests/tabs/test_training_research_ui.py       |   9 +-
 11 files changed, 254 insertions(+), 171 deletions(-)
```

git status --short:

```text
 M Application/tabs/analysis_filter_tab.py
 M Application/tabs/create_results_tab.py
 M Application/tabs/data_loading_tab.py
 M Application/tabs/dataset_statistics_tab.py
 M Application/tabs/reference_loading_section.py
 M Application/tabs/train_model_tab.py
 M Application/theme/custom.qss
 M main.py
 M tests/tabs/test_dataset_statistics_tab.py
 M tests/tabs/test_reference_loading_section.py
 M tests/tabs/test_training_research_ui.py
?? Application/theme/layout_metrics.py
?? diagnostics/ui_spacing01/
?? tests/tabs/test_ui_spacing.py
```
