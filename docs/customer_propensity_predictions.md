# Исследовательские прогнозы Customer Propensity

Этот слой не обучает модель и не подключается к GUI/BPR/hybrid. Он читает опубликованный
inference JSONL.gz существующим валидирующим reader и применяет только сохранённый
calibrated pipeline: numeric_batch → frozen transform → float64 decision → saved sigmoid.

## Запуск

Используется отдельный CLI `scripts/infer_customer_propensity.py`. Обязательны `--model`,
`--dataset`, `--report-dir` (новый собственный каталог отчётов). Режимы взаимоисключающие:

- `--preflight`: полный read-only проход, SHA256, gzip CRC/schema/count, уникальность ключей;
- `--smoke`: не более 128 первых записей, проверка равенства разных batch sizes, без публикации;
- `--run`: одна полная попытка. `--batch-size` от 1 до 50 000, по умолчанию 50 000;
  `--output-root` по умолчанию `filtered_data/customer_propensity_predictions`.

Сначала синтетические тесты, preflight и smoke, затем один `--run`. Автоматического retry,
обучения, wall-clock timeout, обращения к target, истории SQLite или HMAC-секрету нет.
Ручная отмена: Ctrl+C/SIGINT, SIGTERM или создание файла `--cancel-file`.
ResourceGuard контролирует private commit ≤3 GiB, disk reserve ≥20 GiB, available physical
≥4 GiB и available system commit ≥3 GiB. Это кооперативная проверка, не квота ОС.

Вход должен совпадать с CURRENT_INFERENCE.json, моделью, feature schema v2,
source/generation/key_id и одним UTC-срезом. Reader не допускает target-полей в inference.
Контрольные суммы входных manifests/model JSON повторно сверяются перед публикацией.
Один writer_lock в отдельном output root предотвращает конкурирующие публикации.
Обнаруженный staging от прежней аварии блокирует новую попытку до отдельного разбора.

## Хранение и публикация

Одна версия содержит `predictions.sqlite` и `manifest.json`. Version — SHA256 canonical
manifest body без version. Manifest хранит SHA256/размер SQLite, model/dataset versions,
schema, source/generation/key_id, UTC-срез/горизонт, calibrated, RESEARCH_ONLY,
диагностические агрегаты и ограничения качества. CURRENT/production model pointer не меняется.

SQLite schema 1:

- `metadata(key,value)` содержит общий contract;
- `predictions(customer_pseudonym PRIMARY KEY, probability REAL, warning_mask INTEGER)`
  хранит одну запись на псевдоним, `WITHOUT ROWID`.

Дата, горизонт и версия модели нормализованы в общем contract: read-only API возвращает их
в каждой записи. Отдельной JSON-выгрузки прогнозов и открытой таблицы реальных ID нет.
Точные 48-векторы используются только в RAM для счётчиков повторов, в SQLite не записываются.

Новый UUID staging и при создании output root защищаются существующим ACL helper
(пользователь/SYSTEM/Administrators). Родительские пользовательские каталоги не меняются.
После полного исчерпания reader выполняются commit/close, SQLite integrity_check,
row count, fsync, SHA256, manifest validation и последний guard. Только затем staging
атомарно переименовывается в готовую версию на том же томе. Это commit point.
Ошибка/отмена до него закрывает ресурсы и удаляет только собственный UUID staging;
предыдущие версии не затрагиваются. Ошибка записи отчёта после commit point не отменяет
готовый артефакт: его можно независимо проверить и загрузить, новый inference не нужен.
При принудительном завершении ОС возможен непубликованный staging, требующий разбора.

## Read-only lookup

```python
from Application.model.propensity_ml.predictions import PredictionStore, PredictionStatus

# Создать один раз в worker, с явно выбранной версией прогнозов.
store = PredictionStore(prediction_version_directory)
result = store.get_prediction(customer_pseudonym, model_version, prediction_date)
if result.status is PredictionStatus.READY:
    probability = result.prediction.probability
    warnings = result.prediction.warnings
```

Конструктор проверяет SHA256, metadata и count один раз. Не создавать его на каждый
lookup и не запускать в GUI-потоке. Optional `verify_integrity=True` предназначен для
явной проверки опубликованной небольшой prediction DB, не для открытия карточки.
Это никогда не проверка 25-GiB history index.

Каждый lookup открывает `mode=ro`, `query_only=ON`, `trusted_schema=OFF`, делает один
primary-key SELECT и закрывает соединение. Перед ним проверяется size/mtime immutable DB.
Полного SHA256/scan на каждом lookup нет. Одновременного открытого write handle нет.
Статусы: READY, NOT_FOUND, INVALID_INPUT, INCOMPATIBLE_VERSION, ARTIFACT_UNAVAILABLE,
ARTIFACT_INCONSISTENT. Ошибка загрузки версии отдельно обрабатывается вызывающим worker
(FileNotFoundError/OSError/MLDataError/ошибка SQLite), без блокировки профиля/рекомендаций.

## Диагностика и качество

- `NEAR_ZERO_WITH_RECENT_FAVORITE`: calibrated p≤10⁻⁶ и доступный has_favorite_90d=1.
  Это исследовательское предупреждение, не бизнес-правило и не исправление вероятности.
- `LOW_INDIVIDUALIZATION`: все 22 поля доступны, count/has=0, recency/average interval=null.
  Нет событий в доступных окнах, а не за всю жизнь покупателя.
- `HISTORICAL_FEATURES_UNAVAILABLE`: хотя бы один available=false; не смешивается с
  наблюдаемым отсутствием событий.

Вероятности сохраняются без ручной коррекции, flooring и округления. В отчётах только
агрегаты; диапазоны 0,1/1/5/10% описательные, не сегменты. Inference не имеет target,
поэтому конверсию, AP/AUC, Brier и калибровку нового среза здесь оценивать нельзя.

## Будущий GUI и identity

Нельзя передавать в HMAC произвольный ID из поля ввода или BPR index. Карточка получает
canonical_customer_id из согласованных profile/contact/identity; при конфликте показ
должен блокироваться независимо от рекомендаций. Dataset builder использует
HMAC-SHA256(secret, b"kanzler-propensity-customer-v1\0" + canonical_id.encode("utf-8"));
key_id — отдельный HMAC domain. Новый способ псевдонимизации не вводится.

Для 02E нужен worker-side helper, который подтверждает совпадение canonical source/generation,
загружает существующий стабильный ключ из защищённого локального файла, сверяет key_id,
получает один псевдоним в RAM и делает lookup. Не логировать secret/ID/key, не сохранять
таблицу соответствия. В 02D secret не требуется и не читается; end-to-end identity binding
с действующим секретом остаётся проверкой 02E, схема сопоставления установлена по коду.

Показ будущего блока: экспериментальная вероятность оформления **заказа**, исторический
as-of 01.09.2026, [01.09,01.10) UTC, RESEARCH_ONLY. Не называть оплатой или текущим прогнозом.
Near-zero warning позволяет показать «Оценка ненадежна», inactive warning — «Слабо
индивидуализирован». Решение о визуализации принимается отдельно. Автоматические
маркетинговые воздействия и бизнес-пороги не разрешены результатами этого baseline.
