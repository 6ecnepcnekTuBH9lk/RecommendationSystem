# RecommendationSystem

Десктопное приложение для построения и применения рекомендательной системы на основе модели BPR-MF.

Приложение позволяет подготовить входные данные, обучить рекомендательную модель и получить персональные рекомендации для клиентов через графический интерфейс.

## Основные возможности

- загрузка и обработка входных данных;
- настройка параметров обработки и фильтрации;
- обучение рекомендательной модели BPR-MF;
- настройка параметров обучения модели;
- получение персональных рекомендаций для выбранного клиента;
- просмотр истории покупок клиента;
- выбор количества рекомендаций: Топ-1, Топ-3, Топ-5 или Топ-10;
- выгрузка рекомендаций в Excel;
- отображение фотографий товаров;
- переключение темы интерфейса.

## Модель

В качестве основы рекомендательной системы используется модель **BPR-MF**  
(Bayesian Personalized Ranking Matrix Factorization).

Модель работает с неявной обратной связью пользователей и учитывает различные типы взаимодействий с товарами, например:

- просмотры;
- добавления в избранное;
- покупки.

Для разных типов взаимодействий используются различные веса.

Обучение модели реализовано с использованием библиотеки PyTorch.

## Структура проекта

```text
RecommendationSystem/
├── Application/
│   ├── files/
│   │   └── files_processing.py
│   ├── model/
│   │   └── BPRMF.py
│   ├── photo/
│   │   └── photo_processing.py
│   ├── settings/
│   │   ├── set_status.py
│   │   └── settings_and_filter.py
│   ├── tabs/
│   │   ├── create_results_tab.py
│   │   ├── reference_loading_section.py
│   │   └── train_model_tab.py
│   └── theme/
│       ├── SwitchTheme.py
│       └── apply_theme.py
├── assets/icons/
├── main.py
└── README.md
```

## Требования

Для запуска проекта требуется:

- Python 3.10 или новее;
- pip.

Основные используемые библиотеки:

- PyQt6;
- pandas;
- numpy;
- PyTorch;
- openpyxl;
- requests;
- chardet.

## Установка

Клонировать репозиторий:

```bash
git clone https://github.com/6ecnepcnekTuBH9lk/RecommendationSystem.git
```

Перейти в директорию проекта:

```bash
cd RecommendationSystem
```

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать виртуальное окружение в Windows:

```bash
.venv\Scripts\activate
```

Установить необходимые библиотеки:

```bash
pip install -r requirements.txt
```

## Запуск

Запуск приложения выполняется из корневой директории проекта:

```bash
python main.py
```

## Обучение из GUI (TRAIN-UI-01 / TRAIN-UI-02)

Вкладка «Обучение модели» содержит режим, четыре параметра (веса покупки,
избранного, просмотра и количество эпох), состояние данных, запуск/отмену,
progress, журнал с итоговой сводкой и историю экспериментов. Начальные значения:
10 / 2 / 0.5 / 50; это provisional research defaults, а не доказанный optimum.
Веса конечные, покупка > 0, остальные >= 0, эпохи — целое число 1–10000.
Пустые/ошибочные поля блокируют запуск.

**Эксперимент** выбран по умолчанию: один запуск seed=42, одно обучение до
заданного бюджета, затем external validation. Используются существующие
canonical loader, temporal protocol, BPR adapter и evaluator: история до
2025-11-01 UTC, validation [2025-11-01, 2025-12-01) UTC, min_history_events=10.
Blind test не оценивается, не доступен в GUI и не записывается в историю.
Early stopping и item features отключены. Frozen параметры заданы общим
research contract: dimension=128, batch=256, n_neg=10, lr=0.0003,
bpr_reg=0.0005, weight_decay=0. Автоматических sweep, следующих seeds,
выбора победителя и публикации нет.

**Рабочая модель** использует существующий production CLI `gui`:
canonical storage → preflight → preparation → trainer → publication.
Четыре значения формы перекрывают соответствующие поля существующего
`user_settings/train_config.json`; скрытые параметры валидирует штатный
TrainConfig loader, отсутствующие значения берутся из TrainConfig defaults.
Существующие early stopping и item-feature settings сохраняются.
Перед стартом требуется подтверждение weights/epochs и новой generation;
WARN отдельно требует согласия, PASS разрешает обучение, BLOCK запрещает.
Analysis filters и legacy interaction CSV в этот маршрут не входят.
Publication lock, staging, disk validation, атомарное переключение
`model/current.json`, readback и rollback сохранены. Production не использует
research temporal split и не добавляется в историю экспериментов.

Оба режима работают в отдельном QProcess, тем же Python, что приложение.
Открытие вкладки проверяет только metadata и имена canonical part files,
не читает raw payload. Отсутствующий/повреждённый batch, недоступный benchmark,
невалидные параметры и активный процесс блокируют start с объяснением.
После обнаруженного preflight BLOCK запуск закрыт до изменения revision данных,
каталога товаров или параметров; обновлённая configuration проходит новый preflight.
Полная проверка остаётся в дочернем процессе. Во время run поля заблокированы;
структурированные события показывают epoch/loss/device, таймер — elapsed time.
Слева расположены параметры в порядке режим → эпохи → покупка → избранное →
просмотр и блок данных; справа — процесс и журнал. Секции разделены линиями,
история занимает нижнюю область. Веса в GUI отображаются с двумя знаками.
До подготовки неизвестные counts отображаются как «—»; отдельная загрузка
canonical events ради счётчиков не выполняется. Уже полученные агрегаты обычной
подготовки показывают взаимодействия, пользователей, товары и обучающие пары.
После успешного запуска они сохраняются до смены данных или нового запуска.
Research взаимодействия — события именно training history выбранного snapshot,
production взаимодействия — уже рассчитанный bpr_events. Validation cases в GUI
не отображаются. Отдельная карточка результата и постоянная подсказка удалены.

«Отменить» создаёт cancel signal: worker проверяет его между этапами/эпохами.
При задержке до публикации QProcess принудительно завершается через 2 секунды;
training не создаёт отдельного multiprocessing worker. Перед publication
CLI ждёт handshake от GUI. GUI сначала отключает cancel, затем разрешает
публикацию; в короткой критической секции отмена недоступна. Это не позволяет
принятой отмене активировать новую модель. Закрытие окна во время run сначала
отменяет его и ждёт завершения; после завершения окно можно закрыть.

История: `user_settings/research_experiments/history.json`, schema_version=1,
массив runs с уникальным run_id. Aggregate-only result artifacts:
`user_settings/research_experiments/runs/<run_id>/result.json`.
Хранятся timestamps, status, weights/frozen config, epochs, benchmark identity,
users/items/pairs, validation overall и type metrics @5/@10/@20, duration,
device, torch/CUDA build, git HEAD/dirty, artifact reference и safe error code.
PII, raw payload, exception text и test metrics не сохраняются.
Состояние running и выполненные эпохи тоже сохраняются; неизвестные runtime
агрегаты при ошибке запуска записываются как null. JSON обновляется под OS lock через unique temporary file → flush/fsync →
atomic replace. Повторная запись run_id обновляет одну запись; порядок newest-first.
Completed, failed и cancelled остаются в истории. При повторном открытии
завершённый worker artifact восстанавливает результат, умерший running worker
получает interrupted. Повреждённая история не перезаписывается: предупреждение
в GUI, новые artifacts сохраняются отдельно; production остаётся доступен.
Итог текущего запуска выводится в журнал; выбор строки истории его не заменяет.
Подписи и статусы таблицы локализованы, storage codes и JSON v1 сохранены.
Колонки можно изменять вручную, ширина сохраняется при обновлении таблицы;
доступна горизонтальная прокрутка. NDCG/Recall отображаются с шестью знаками,
время — HH:MM:SS. Inputs между сессиями не сохраняются;
существующий train_config.json не переписывается. Временный config/cancel signal
удаляются после завершения. Audit, GPU-cost, convergence и weight CLI сохранены.


## Локальные файлы

Иконки находятся в assets/icons/. Общие пути определены в Application/paths.py.
Пользовательские данные, raw Mindbox и отчёты размещаются в input_data/,
настройки — в user_settings/; оба каталога целиком игнорируются Git.
Legacy CSV используют имена orders.csv, views.csv, favorites.csv,
nomenclature.csv, site_categories.csv, city_coordinates.csv, weather.csv и stores.csv.
Отфильтрованные датасеты создаются в filtered_data/, файлы рекомендаций —
в model/ с ASCII-именами. Оба generated-каталога также игнорируются Git.

Переименование каталогов не переводит интерфейс, бизнес-данные или сообщения.
Содержимое перенесённых пользовательских файлов сохраняется. При чтении старого
train_config.json загрузчик автоматически заменяет legacy-компонент пути
ВходныеДанные на input_data в data_dir, в том числе для прямого CLI.
Остальные параметры, разделители и компоненты пути сохраняются; сам JSON не перезаписывается.
