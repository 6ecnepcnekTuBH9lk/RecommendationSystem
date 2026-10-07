# TRAIN-03A: 30 дней / history ≥ 10 — предварительный benchmark с ограничениями источника

Рекомендуемая конфигурация для рассмотрения Иваном:

```json
{
  "validation_start": "2025-11-01T00:00:00Z",
  "test_start": "2025-12-01T00:00:00Z",
  "test_end": "2025-12-31T00:00:00Z",
  "min_history_events": 10
}
```

Оба future windows имеют ровно 30 суток. Final cases: validation **4 976**, test
**5 341**. Это предложение, а не изменение default или production configuration.
По объёму наблюдаемого resolved subset данных достаточно для controlled warm-start
experiments. Перед выводами о качестве модели необходимо разобраться с крупным
исключением mapped Actions без products и подтвердить историческую допустимость
используемых metadata. Protocol под эти результаты менять не требуется.

## Источник и фактическое покрытие

Canonical current batch: самая длинная общая continuous range Actions/Orders,
с existing manual precedence, CustomerMerges и selection из canonical storage.
Revision: `af24df53c47349d0ae3170ab396df726`; дата публикации source metadata:
`2026-09-23T11:32:46.254830+00:00`. Active Actions/Orders — MANUAL; версия данных
описывает события 2025 года, а не текущую активность 2026 года.

Declared common coverage: **[2025-01-01, 2026-01-01) UTC**, 365 дней.
CustomerMerges имеют то же declared coverage. В metadata нет gaps, а в resolved
dataset нет ни одного дня 2025 года с нулевым числом событий. Это не доказывает
полноту каждой исходной выгрузки: MANUAL period задаётся декларативно.

Observed timestamps: **2022-10-15T17:29:20.663Z — 2025-12-31T23:57:16.107Z**.
Это не непрерывное покрытие с 2022 года: до 2025 года есть лишь **1 087 PURCHASE
events**, 0,0901% всех resolved events. VIEW начинаются
`2025-01-01T00:53:58.967Z`, FAVORITE — `2025-01-24T11:11:29.720Z`.
Старые PURCHASE timestamps берутся из existing `firstAction.dateTimeUtc` semantics.
Все occurrences сохранены, новая дедупликация или фильтрация дат не введена.

| Population | Count |
|---|---:|
| Resolved interactions | 1 206 864 |
| Canonical customers | 184 976 |
| Resolved items | 5 545 |
| VIEW | 610 086 (50,55%) |
| FAVORITE | 5 111 (0,42%) |
| PURCHASE | 591 667 (49,03%) |

Raw и resolved counts различаются. Прочитано 3 347 664 Actions, 309 044 Orders,
41 412 CustomerMerges. Order lines: 666 576; classified interactions до product
resolution: 1 206 884. Unresolved products: 20 (8 VIEW, 12 PURCHASE), unsupported:
0. Orders duplicate identical/conflicting: 0/0. Нового duplicate policy нет.

**High, подтверждённый count / неизвестна upstream причина:** existing builder
исключил **2 732 459 mapped Actions**, **81,6229% от 3 347 664 mapped Actions**.
Unmapped Actions: 0. Эти события имеют пустой `action.products` в existing adapter
contract; `InteractionBuilder.from_action()` помечает их malformed и не создаёт
item interaction. Аудит использует production-compatible diagnose semantics.
Нельзя по одному этому count доказать повреждение выгрузки: нужно проверить,
отсутствует ли product identity в источнике или находится в другом поле, и являются
ли эти записи по бизнес-смыслу ожидаемыми. Они не могут молча трактоваться как
наблюдаемые item interactions. Benchmark измеряет **resolved accepted subset**.
Даты, сезонный состав и user eligibility могут зависеть от этого исключения.

**Medium:** 1 087 старых PURCHASE events выходят за declared 2025 range. Вероятное
объяснение — первоначальное время заказа в более позднем экспортированном snapshot;
это интерпретация, а не доказанная причина. Полную history до 2025 года заявлять
нельзя. Existing timestamps/Orders semantics не исправлялись.

## Плотность и конец dataset

В 2025 году daily events: min **1 205**, P25 **2 734**, median **3 209**, P75
**3 780**, P90 **4 423,2**, max **6 998**, mean **3 303,5**. Минимум — 1 января;
максимум — 16 января. Daily rows покрывают declared 2025 range и в сумме дают
**1 205 777 events**; остальные 1 087 — старые PURCHASE. Weekly counts и active
customers/items для всех наблюдаемых недель находятся в `report.md`, точная daily
density — в `report.json`.

Последние даты: 27 декабря — 6 552, 28 декабря — 5 957, 29 декабря — 4 171,
30 декабря — 4 964, 31 декабря — 3 444. Последний день ниже предыдущего, но выше
годовой median; последняя VIEW запись — в 23:57 UTC. Резкого исчезновения активности
на хвосте не установлено. Низкое число за неделю 29 декабря нельзя сравнивать с
полной предыдущей неделей: в dataset у неё только три календарных дня.

**Medium:** независимых export logs для доказательства полноты последнего дня
нет. Консервативный test_end — **2025-12-31T00:00:00Z**, то есть evaluation
использует даты по 30 декабря включительно. События 31 декабря не удалены и входят
в coverage aggregates. Полнота 30 декабря также не доказана независимо;
граница является осторожным предположением на declared continuous range.

## Проверенные варианты

Все горизонты доступны. Этап A: 14/30/60 суток при history ≥ 10. Этап B:
5/10/20 для 30 суток; повторный вариант 30/10 не пересчитывался как отдельная строка.
Каждый snapshot использует всё доступное прошлое до собственного cutoff.

| Horizon / history | Validation window UTC | Test window UTC | Val future users / cases | Test future users / cases | Median delay V/T, days |
|---|---|---|---:|---:|---:|
| 14 / 10 | [2025-12-03, 2025-12-17) | [2025-12-17, 2025-12-31) | 14 780 / 2 909 | 20 334 / 3 530 | 5,52 / 5,57 |
| 30 / 10 | [2025-11-01, 2025-12-01) | [2025-12-01, 2025-12-31) | 24 951 / 4 976 | 33 661 / 5 341 | 10,57 / 12,39 |
| 60 / 10 | [2025-09-02, 2025-11-01) | [2025-11-01, 2025-12-31) | 41 875 / 6 263 | 53 029 / 7 468 | 18,46 / 18,62 |
| 30 / 5 | [2025-11-01, 2025-12-01) | [2025-12-01, 2025-12-31) | 24 951 / 7 656 | 33 661 / 8 445 | 11,45 / 13,30 |
| 30 / 20 | [2025-11-01, 2025-12-01) | [2025-12-01, 2025-12-31) | 24 951 / 2 919 | 33 661 / 2 958 | 9,85 / 9,93 |

Знаменатели: cold/sparse/no-novel % — **future users**. Cold-item % —
**first-novel targets пользователей, прошедших history threshold**. PURCHASE % —
**final cases**. No-novel здесь относится к history-eligible population;
пользователи, уже исключённые как cold/sparse, не смешиваются с этой категорией.
Дополнительные rates и все numerator/denominator сохранены в JSON.

| Horizon / history | Cold users V/T % | Sparse V/T % | Cold items V/T % | No novel V/T % | PURCHASE V/T % |
|---|---:|---:|---:|---:|---:|
| 14 / 10 | 45,78 / 49,45 | 31,43 / 31,35 | 2,68 / 0,34 | 2,56 / 1,78 | 37,16 / 44,99 |
| 30 / 10 | 45,15 / 51,61 | 33,13 / 30,76 | 1,56 / 3,21 | 1,45 / 1,24 | 34,53 / 42,16 |
| 60 / 10 | 50,81 / 54,00 | 32,86 / 30,74 | 4,83 / 3,25 | 0,61 / 0,70 | 32,14 / 40,71 |
| 30 / 5 | 45,15 / 51,61 | 21,63 / 20,72 | 1,53 / 3,75 | 2,06 / 1,60 | 45,28 / 53,19 |
| 30 / 20 | 45,15 / 51,61 | 42,02 / 38,51 | 1,55 / 3,24 | 0,94 / 0,80 | 23,19 / 28,50 |

## Рекомендуемый 30/10: population

| Count | Validation | Test |
|---|---:|---:|
| Training events | 990 054 | 1 084 760 |
| Training customers | 155 669 | 166 935 |
| Training items | 5 201 | 5 338 |
| Future events | 94 706 | 118 660 |
| Future users | 24 951 | 33 661 |
| History eligible | 5 418 | 5 934 |
| Cold users | 11 266 | 17 372 |
| Sparse users | 8 267 | 10 355 |
| Eligible no-novel users | 363 | 416 |
| Eligible first-novel targets | 5 055 | 5 518 |
| First-novel globally cold | 79 | 177 |
| First-novel globally warm / final cases | 4 976 | 5 341 |

Temporal count reconciliation: **990 054 history + 94 706 validation + 118 660
test + 3 444 excluded tail = 1 206 864 resolved events**. Test history равна
990 054 + 94 706 = 1 084 760; весь validation period включён в неё.

Причины исключения mutually exclusive и в сумме с final cases дают future users.
Cold item не заменялся поздним warm target. No-novel среди history eligible:
363/5 418 = **6,70%**, 416/5 934 = **7,01%**. Warm item среди eligible first-novel:
4 976/5 055 = **98,44%**, 5 341/5 518 = **96,79%**. Но history eligible составляют
лишь **21,71% / 17,63%** future users; final cases — **19,94% / 15,87%**.
Этот primary benchmark не описывает cold/sparse majority.
Cold здесь означает отсутствие **наблюдаемой resolved history** до cutoff,
а не доказательство, что человек впервые стал клиентом магазина.

| Target type | Validation count / share | Test count / share |
|---|---:|---:|
| VIEW | 3 257 / 65,45% | 3 083 / 57,72% |
| FAVORITE | 1 / 0,02% | 6 / 0,11% |
| PURCHASE | 1 718 / 34,53% | 2 252 / 42,16% |

Общая метрика будет преимущественно измерять VIEW, но PURCHASE составляют
существенную часть. FAVORITE observations недостаточно для устойчивых отдельных
выводов. По этим proportions веса BPR не выбирались и не изменялись.

## Delay, history, candidates

| Distribution | Min | P25 | Median | P75 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Target delay V, days | 0,013 | 5,29 | 10,57 | 18,60 | 25,49 | 29,85 |
| Target delay T, days | 0,081 | 4,51 | 12,39 | 20,44 | 26,38 | 29,91 |
| History events V | 10 | 14 | 24 | 47 | 103 | 2 188 |
| History events T | 10 | 14 | 22 | 46 | 101 | 2 730 |
| Unique seen items V | 2 | 11 | 17 | 31 | 58 | 741 |
| Unique seen items T | 1 | 11 | 17 | 31 | 58 | 781 |
| Candidates V | 4 460 | 5 170 | 5 184 | 5 190 | 5 192 | 5 199 |
| Candidates T | 4 557 | 5 307 | 5 321 | 5 327 | 5 329 | 5 337 |

Mean delay: **12,33 / 12,90 days**. Mean history: **50,33 / 49,33 events**:
выше median из-за длинного хвоста, но типичный benchmark user имеет 22–24 events.
Ранжирование почти по всему historical universe; small candidate-set shortcut нет.

| Target type delay | N | P25 | Median | P75 | P90, days |
|---|---:|---:|---:|---:|---:|
| VIEW validation | 3 257 | 5,14 | 9,79 | 16,74 | 23,75 |
| VIEW test | 3 083 | 3,52 | 9,41 | 17,91 | 23,47 |
| PURCHASE validation | 1 718 | 5,63 | 13,55 | 21,65 | 27,60 |
| PURCHASE test | 2 252 | 6,71 | 15,47 | 23,54 | 27,48 |

Quantiles FAVORITE доступны в JSON, но N=1/6 не позволяет интерпретировать их как
устойчивые distributions. Все quantiles — NumPy linear; delay измеряется в днях.

## Overlap и concentration

Evaluation users: только validation **2 781**, только test **3 146**, оба snapshots
**2 195**. Повторный customer означает новый prediction moment; test history
включает весь validation period, а не только validation cases.

| Target-item statistic | Validation | Test |
|---|---:|---:|
| Unique target items | 1 039 | 1 150 |
| Top-10 case concentration | 643/4 976 = 12,92% | 497/5 341 = 9,31% |
| Top-50 case concentration | 1 829/4 976 = 36,76% | 1 521/5 341 = 28,48% |
| Target train popularity P25 | 254,75 | 306 |
| Target train popularity median | 491 | 586 |
| Target train popularity P75 | 899 | 1 040 |
| Target train popularity P90 | 1 406 | 1 573 |

Popularity — occurrences товара только в history собственного snapshot, не
whole-dataset frequency и не количество будущих покупок. Concentration показывает
существенный вес частых target items, особенно в validation; evaluation не
перевзвешивалась по popularity.

## Сходство периодов и обоснование рекомендации

При 30/10 case count test больше на **7,34%**, тогда как future users больше на
**34,91%**. History distributions близки; candidate universe немного растёт.
Но populations не идентичны: cold-user rate +**6,46 процентного пункта**,
PURCHASE target share +**7,64 п.п.**, median delay +**1,82 дня**. Общую будущую
Recall/NDCG нужно читать с учётом этого состава, а не как чистое изменение модели.

30 дней дают около 5k cases на каждом prediction моменте, достаточно PURCHASE
targets и месячный business horizon. 14 дней тоже доступны, но cases меньше
(2 909/3 530) и их количество меняется сильнее; median PURCHASE delay для 30-day
protocol достигает 13,55/15,47 дня. 60 дней дают больше cases, но ставят более
длинную задачу и соединяют сентябрь–октябрь с ноябрём–декабрём, усиливая возможный
season shift. Декабрьский рост покупок и пользователей без прежней наблюдаемой
resolved history установлен; его
причина (seasonality, campaigns, assortment) данным аудитом не установлена.
Cutoffs не переставлялись ради равных target distributions.

Threshold 5 расширяет cases до 7 656/8 445 и меняет PURCHASE shares до
45,28/53,19%; median history снижается до 14/13. Threshold 20 даёт 2 919/2 958
cases с median history 41/42 и PURCHASE shares 23,19/28,50%. Это изменение
population и бизнес-смысла benchmark, а не автоматически лучший threshold.
Предлагается сохранить исходный **10** как компромисс и зафиксировать его до
model-quality experiments. Recall/NDCG не вычислялись, модели не обучались.

## Корректность и готовность к experiments

Проверены **52 465 case occurrences** в пяти сценариях и десяти snapshots
(это не 52 465 уникальных customers). Все нарушения равны нулю:
history timestamp ≥ cutoff, target вне window, target seen, target globally cold,
history ниже выбранного threshold. При любом нарушении script блокирует audit.
Первый novel target, fixed seen-set, warm item universe и boundary semantics
TRAIN-02E сохранены.

Контролируемые experiments на **наблюдаемом resolved warm subset** технически
возможны по объёму. Для доверенного вывода о магазине в целом сначала нужно
уточнить исключение 81,62% mapped Actions. Правки protocol для увеличения cases
не рекомендуются. Current mutable catalog и full-year customer merge/order
snapshots не доказывают historical as-of metadata; timestamp invariants этого
ограничения не устраняют. Item features в аудите не использовались.

## Воспроизводимость и проверки

```powershell
.venv312\Scripts\python.exe scripts/audit_temporal_evaluation.py --horizons 14 30 60 --thresholds 5 10 20 --sensitivity-horizon 30 --test-end 2025-12-31T00:00:00Z
```

`report.json` — точные агрегаты и provenance; `report.md` — все scenarios,
distributions, denominators и weekly tables; этот файл — интерпретация для решения.
Runtime dependency environment не менялся. Audit uses existing parsers/adapters,
identity/product resolution и order policy, без parallel raw parser.

Tests: 10 new targeted passed; temporal + relevant canonical/preparation tests:
312 passed; full suite: **2005 passed, 110 warnings**. Ruff и diff-check прошли.
Сравнение byte SHA-256 всех 23 existing input/model/settings files до/после:
изменений нет. Все 216 pre-existing code/doc files также сохранены byte-for-byte.

Реальное BPR training не запускалось, production model/current.json и canonical
data не менялись, publication не выполнялась, PII не выводился. Production path,
математика BPR и temporal protocol semantics не изменены. Commit/push не выполнялись.
