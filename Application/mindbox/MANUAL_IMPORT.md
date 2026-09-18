# Ручной JSON fallback

Обновление 18.09.2026: ручной Customers через GUI/CLI теперь атомарно заменяет
`canonical/customers.sqlite`, используя тот же store, что monthly API update.
История Customers snapshots больше не создаётся этим маршрутом. Подробности:
[canonical storage](CANONICAL_STORAGE.md). Описание profile schema v2 ниже относится
к legacy manifests, которые остаются читаемыми. По уточнению пользователя ручной
Actions/Orders пока сохраняет прежнюю модель; его перенос в daily storage отложен.

Основной путь — API. Ручной импорт использует сохранённую историю CustomerMerges
из завершённого API batch, поэтому полностью автономным режимом не является.

## Работа в GUI

На вкладке «Получение данных» один набор selection controls используется для API
и ручных Actions + Orders. Для ручной загрузки выберите оба JSON и задайте общий
период в верхних полях. Границы UTC, верхняя граница исключительная. В интерфейсе
Mindbox UTC+03 границы нужно сдвинуть на +3 часа: 00:00 UTC соответствует 03:00.
Actions следует выгружать широко; точные VIEW/FAVORITE mappings применяет приложение.
Период Orders не выводится из firstAction: старые даты создания заказов допустимы.

Customers импортируется отдельной кнопкой: полный snapshot без interaction dates.
Обе операции выполняются через QProcess. Ход копирования в bytes и проверки в
records отображается в журнале, общий progress остаётся indeterminate до публикации.
Кнопка «Отменить» останавливает активную операцию. Manual import не имеет resume;
повторный запуск читает выбранные исходные файлы заново.

CSV-раздел содержит только Номенклатуру, Категории сайта и Координаты городов.
Это полная замена справочника; append и interaction CSV routes удалены из UI.
Парсинг, нормализация и запись CSV выполняются в отдельном процессе. GUI получает
готовые списки видов/коллекций/городов, обновляет связанные controls и кэши, без
пересчёта всей статистики взаимодействий. Внутренние legacy interaction processors
пока сохранены для совместимости старых тестов; пользовательских маршрутов к ним нет.

## Общий pipeline

API или manual transport → raw → strict streaming reader → selection → адаптеры
→ CustomerIdResolver → interactions → ProductResolver → PreparedBprData → QC.
Отдельных manual-адаптеров, selection rules и подготовки BPR нет. ProductResolver
сохраняет prefix-6 и обязательное сопоставление с Номенклатура.csv.

Customers использует общий потоковый reader и адаптер контактов. После импорта
обычный CustomerProfileSnapshot читается через load_customer_contact_index.
JSON не материализуется целиком: буфер копирования 1 MiB, парсера 64 KiB и один
текущий raw record. Файл копируется, затем проверяется; требуется место для одной
полной raw-копии. Память resolver/contact index зависит от числа идентификаторов.

## Метаданные и публикация

Новые training manifests — schema v4. Загрузчик продолжает читать v2 (legacy
selection defaults) и v3 (сохранённый selection). TrainingBatchExport содержит
source_kind; API сохраняет настоящий export_id, MANUAL использует null.
Manual batch содержит три компонента: ссылку на API CustomerMerges и Actions/Orders
за явно заданный общий период, а также merge_source_training_batch_id.

Новые manual profile snapshots — schema v2; API snapshots v1 и старые v1 manifests
без transport fields читаются. У manual Customers originating_training_batch_id
равен null: отдельное поле merge_source_training_batch_id указывает только источник
истории объединений. Время импорта хранится в created_at/created_at_utc.
Абсолютный путь к выбранному исходному файлу в manifest не сохраняется.

Manual raw находится внутри собственного training_batches/<uuid> либо
customer_profile_snapshots/<uuid>. Raw и metadata сначала записываются в скрытый
временный каталог; один rename публикует весь набор. Ошибка до rename не оставляет
валидного published manifest. После принудительного завершения процесса возможен
скрытый .manual-* каталог: selectors его игнорируют. Исходный файл не изменяется.
Публикация уже завершена, если cancel пришёл после commit rename.

CustomerMerges выбирается по максимальному created_at среди совместимых завершённых
API batches. Для interactions merge_since источника должен быть не позже выбранного
начала истории, а interaction_until — не раньше конца периода. Customers использует
последний завершённый API source без требования interaction window. Повреждённый API
final manifest блокирует выбор; скрытой подмены старым источником нет. Raw merges
проверяются через существующий адаптер и resolver до копирования файлов.
GUI показывает время создания source batch и coverage; произвольный TTL не вводится.

## Orders dedup

Общий preparation хранит order_id → SHA-256 семантического raw snapshot.
Порядок ключей словаря не влияет на fingerprint; массивы сохраняют порядок,
числа нормализуются без округления Decimal. Первое occurrence принимается,
идентичное повторение пропускается. Отличающийся snapshot вызывает безопасную
ошибку в strict mode, а в diagnose увеличивает conflict counter, устанавливает
complete=False и даёт QC BLOCK (включая production preflight).

Диагностика: orders_raw, orders_unique, orders_duplicate_identical,
orders_duplicate_conflicting. Order IDs и payload не выводятся.

## CLI без сети

```text
python scripts/mindbox_manual_import.py interactions --actions Actions.json --orders Orders.json --since 2026-08-01 --until 2026-08-04 --merge-since "2025-01-01 00:00"
python scripts/mindbox_manual_import.py customers --customers Customers.json
```

Для interactions доступны те же repeated selection arguments, что у export-daily.
Оба режима поддерживают --raw-root. CLI не читает credentials и не обращается к API.
