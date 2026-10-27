# TRAIN-03A: read-only temporal dataset audit

Запуск из корня проекта в существующем окружении:

```powershell
.venv312\Scripts\python.exe scripts/audit_temporal_evaluation.py
```

Параметры: `--raw-root`, `--catalog`, `--output-dir`, `--horizons 14 30 60`,
`--thresholds 5 10 20`, `--sensitivity-horizon 30`, optional aware UTC `--test-end`.
Defaults сохраняют primary minimum-history baseline 10. Сначала сравниваются
горизонты при 10 events, затем thresholds для sensitivity horizon; лишние
дублирующие комбинации не создаются. Default output — отдельный каталог
`diagnostics/temporal_train03a`, с aggregate-only `report.json` и `report.md`.

Источник — canonical `current_batch`: самая длинная общая continuous range
Actions/Orders, с existing manual precedence и CustomerMerges coverage.
`iter_export`, adapters, `InteractionBuilder`, `ProductResolver`, `OrderSnapshots`
переиспользуются без нового JSON parser или правил дедупликации. Полная source
coverage также сохраняется в JSON; участки вне выбранной continuous range не
подменяют production-compatible источник. Сохраняются все resolved occurrences.
Unmapped actions не адаптируются, malformed mapped actions/unresolved products
учитываются диагностически. Orders используют первый снимок внутри выбранного
периода в порядке компонентов, числовых part-файлов и записей; одинаковые дубли
и конфликты сохраняются в диагностике, позиции снимков не объединяются.
Некорректные записи по-прежнему отклоняются. Production quality gate не меняется.

Фильтр `[batch.window.interaction_since, batch.window.interaction_until)` использует
`Actions.dateTimeUtc` и `Orders.firstAction.dateTimeUtc` с UTC-нормализацией.
Orders фильтруются до дедупликации, как в статистике; вспомогательные timestamps
не заменяют дату PURCHASE. Годы не зашиты в loader.

Существующий storage lock координирует сканирование с writers. Coordination file
может открываться, но source JSON/catalog/production model/settings не изменяются.
Revision и размеры/mtime consumed files проверяются после сканирования. Script
не импортирует BPR trainer, не строит tensors/features и не публикует artifacts.
Output внутри source/model/settings деревьев запрещён.

Отдельный fixed-benchmark preflight без обучения/оценки модели:

```powershell
.venv312\Scripts\python.exe scripts/preflight_research.py
```

Он вызывает `load_canonical_events(read_only=True)`: coordination file также
не открывается на запись, дрейф входов приводит к отказу. Следует запускать
без параллельного импорта/GUI; это режим проверки стабильного локального источника.
Используются существующие November/December 2025 cutoffs и cold-start правила.
Результат — aggregate-only `diagnostics/data_research_preflight01/preflight.json`.
Положительные train pairs считаются по distinct history user/item keys, без
материализации BPR arrays; соответствие существующей агрегации проверено тестом.
NDCG/Recall не вычисляются. Текущий `test_end=2025-12-31T00:00Z` не меняется.

Default test_end — минимум конца declared continuous coverage и начала последнего
наблюдаемого календарного дня UTC. Последний день не удаляется: остаётся в coverage,
но не используется как future. Для MANUAL import полнота source не доказана
независимыми export logs; exclusion — осторожная граница, а не доказательство
полноты предыдущих дней. Explicit test_end не может выходить за эту границу.
Cutoffs: end − 2×horizon, end − horizon, end; history — всё доступное прошлое
соответствующего prediction момента. Сам temporal protocol не меняется.

Rates содержат numerator, denominator, percent; denominator=0 даёт null.
Cold/sparse/no-novel/final-case rates используют future users. No-novel counter
относится к history-eligible users и не смешивается с cold/sparse exclusions.
Cold-item rate использует first-novel targets history-eligible users;
target type/concentration shares — final cases. Additional rates дают долю
no-novel среди history-eligible users и cold-item exclusions среди future users.

Distributions: linear NumPy quantiles; target delay в днях, history occurrences,
unique historical items, candidates = historical universe − seen count.
Target popularity считается только по соответствующей training history.
Report сохраняет daily/weekly density, active customers/items, target types,
overlap, concentration и проверенные no-leakage invariants. При нарушении
invariants интерпретация benchmark блокируется.

Конкретная benchmark configuration выбирается человеком после аудита.
Script не подбирает даты по Recall/NDCG и не меняет protocol default.
Mutable current catalog и canonical customer-merge metadata не доказывают
historical as-of доступность; это отдельная граница будущих model experiments.
