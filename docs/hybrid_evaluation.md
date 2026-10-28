# HYBRID-EVAL-CONTRACT-01

Независимый in-memory evaluator: `Application/evaluation/hybrid.py` и
`hybrid_provenance.py`. Он не загружает источники, не обучает модель, не вызывает
inference и не формирует рекомендации. Production `compose()` и прежние
`TemporalSnapshot` / `evaluate_snapshot()` не меняются.

Переиспользуются `training_metrics.single_target_metrics` и согласованный набор
статусов `DEFAULT_PURCHASE_STATUSES`. Старый next-novel-event protocol,
scorer-per-candidate API и reranking фиксированного Top-10 не подходят как
готовые evaluator для этой задачи: у них другие targets/маски/пространства.

## Протокол first_qualified_purchase_v1

`HybridEvaluationCohort` заранее фиксирует клиентов, aware UTC cutoff, конец
окна и `UniverseMode`. Один case на клиента; все cases имеют одинаковые окно и
режим universe. Cohort нельзя подбирать по успешной выдаче алгоритма. Метод
выбора клиентов до запуска и полнота наблюдения требуют внешнего обоснования.

`HybridEvaluationCase` хранит тестовый customer ID, поколение модели,
`CandidateUniverse`, исторические и будущие `Purchase`, uncertain-коды,
свидетельства входов и типизированные причины исключения. Коллекции копируются
в tuple/frozenset; структуры frozen. Часы нормализуются в UTC без усечения даты.

Историческая покупка должна произойти строго до cutoff и быть известна не позже
cutoff. Поздно опубликованный статус старого события не становится историческим
по одному лишь `occurred_at`. Нарушение — ошибка контракта, не пустая выдача.

Qualified purchase: статус **CP / delivering / F**, конечное положительное
Decimal-количество, время события в **[cutoff, future_end)**, товар не входит в
подтверждённые покупки до cutoff. Это операционное определение приложения, не
самостоятельное доказательство оплаты. Просмотры и избранное не исключают товар.
При одинаковом времени выбирается минимальный строковый item code. Порядок
входных записей не меняет target. Повторные покупки ранее купленного товара не
являются targets этого протокола. Несколько релевантных targets пока не поддерживаются.

Первый qualified target сохраняется даже вне candidates: следующая удобная
покупка не подставляется. Future-события ровно на future_end и позже не участвуют.
Статусы других типов и неположительные количества не квалифицируются.
Подготовка/дедупликация согласованных order lines и разрешение идентичности —
ответственность поставщика входов, evaluator не выбирает победивший order snapshot.

`candidates = eligible_items - purchased_items - uncertain_items`.
Target из uncertain может оказаться недостижимым; он не скрывается из метрик.
Маска не расширяется событиями future window.

## Два пространства сравнения

* **SHARED**: `eligible_items` должны входить в `bpr_items`. Обе системы получают
  этот общий universe и маску **до** формирования рекомендаций. Полный список
  исторических источников Hybrid и база R/B-нормализации сохраняются.
* **FULL_ELIGIBLE**: весь доступный на cutoff ассортимент, включая отсутствующие
  в BPR товары. Raw baseline вправе возвращать короткую/пустую выдачу, не создавая
  score для неизвестных товаров. `bpr_target_reachability` — доля targets в
  `candidates ∩ bpr_items`, верхняя граница товарного покрытия BPR; наличие
  model-user этим не подтверждается.

В обоих случаях reachability измеряется относительно universe, а не только
фактически найденных greedy пар. Полнота eligible assortment и корректность его
получения должны подтверждаться provenance. Сам evaluator не доказывает, что
список не получен фильтрацией уже готового Top-10: это обязанность будущего
adapter и его интеграционных тестов. Несопоставимые universes нельзя выдавать за
чистое сравнение ranking.

## Свидетельства STRICT / PROXY / UNAVAILABLE

Обязательны десять `InputRole`: training history, model, historical attributes,
catalog, categories, availability, rules, identity, purchase history, future labels.

`InputEvidence` содержит неизменяемые bytes, проверяемый SHA-256 этих bytes,
case-context SHA-256, поколение модели, as_of, available_at, верхнюю границу
исторических event timestamps и, для labels, границы покрытия окна.
Контекст связывает окно, клиента, universe, маски, поколение, события, определение
qualifying statuses и исключения. Его digest не публикуется в агрегатах.

`EvidenceReview` содержит тип основания, ссылку на свидетельство, проверившего и
digest всего statement (роль, checksum, контекст, поколение и временные поля).
Изменение bytes, statement или использование свидетельства другого case
отклоняется. Название файла и один boolean не заменяют эти условия.

**STRICT** требует всех ролей, совпадающего поколения и контекста, подтверждённого
review, исторических входов, доступных не позже cutoff, и event bounds строго до
cutoff. Purchase evidence не может иметь границу раньше переданных исторических
покупок/их наблюдения. Future labels — отдельный последующий outcome: as_of равен
future_end, покрытие включает всё окно, labels доступны не раньше конца окна и
не раньше наблюдения переданных labels. Их поздняя доступность не считается
утечкой в признаки.

Для синтетических fixtures принимается только `CONTROLLED_SYNTHETIC`, для иных
наборов — `REVIEWED_POINT_IN_TIME`. Переключение `synthetic` само по себе ничего
не подтверждает; receipts другого основания не дают STRICT.

**PROXY** допускает имеющиеся, но исторически неподтверждённые входы: например,
неизвестное/позднее available_at, отсутствие review или неподтверждённую верхнюю
границу истории. Все ограничения выводятся фиксированными числовыми счётчиками.
При запросе PROXY итоговый отчёт явно остаётся PROXY даже при подтверждённых cases.

**UNAVAILABLE**: отсутствует обязательный вход, не совпадает поколение, в истории
есть future, не покрыто future window, события противоречат границам evidence
или есть явное исключение качества данных. Это нельзя исправить выбором PROXY.

Malformed данные, duplicate roles, повреждённые hashes, разные cutoff/context,
наивные timestamps и неверные рекомендации дают `HybridEvaluationError` со
статическим кодом. Такие нарушения не превращаются в обычный отказ алгоритма.

Граница доверия: receipt — проверяемая **структура внешнего свидетельства**, не
цифровая подпись и не независимая сертификация его истинности. Слой проверяет
связность и время заявлений, но не умеет доказать происхождение произвольных
внешних bytes, полноту выгрузки или честность reviewer. Для реальных данных
понадобится отдельно проверенный adapter/source manifest. В этой задаче только
synthetic fixtures; реальные источники STRICT не объявляются. Training snapshot
не подтверждает as-of автоматически, текущие identity/purchase services сюда
не подключаются.

## Списки, preflight и парное сравнение

`RecommendationList` принимает tuple/list кодов в готовом порядке, копирует его,
проверяет уникальность. Допустимо 0..K элементов, K — положительное целое.
Нет сортировки, truncation, padding и числовых scores. Production Hybrid сам
ограничен десятью товарами; evaluator может использовать и другие K.

Каждый выданный код должен входить в `case.candidates`. На непустом списке
нельзя указывать refusal. Пустой список может иметь типизированную причину:
EMPTY, NO_CANDIDATES, MODEL_USER_MISSING, BACKEND_UNAVAILABLE или CANCELLED.
Refusal остаётся в знаменателях, а не исключает клиента.

До вызова обоих алгоритмов выполнить `assess_case(case, synthetic=cohort.synthetic)`:

```python
from Application.evaluation.hybrid import assess_case, compare_recommendations
from Application.evaluation.hybrid_provenance import Validity

mode = Validity.STRICT
eligible_cases = tuple(
    case for case in cohort.cases
    if (assessment := assess_case(case, synthetic=cohort.synthetic)).status
    in ((Validity.STRICT,) if mode is Validity.STRICT else (Validity.STRICT, Validity.PROXY))
)
# Сформировать baseline/challenger по одним eligible_cases и их candidates.
# Оба mapping содержат ровно эти customer IDs и RecommendationList для каждого,
# включая явные пустые списки при ошибке/отказе алгоритма.
report = compare_recommendations(cohort, baseline, challenger, k=10, mode=mode)
```

Mappings должны содержать **ровно** data-eligible клиентов для выбранного режима;
пропущенный результат не трактуется как empty. Все исходные cases сохраняются
в cohort и счётчиках исключений. В STRICT cases с PROXY evidence исключаются
явно с `STRICT_EVIDENCE_REQUIRED`. Нельзя сравнивать только успешную подвыборку.
`compare_recommendations` принимает одну cohort и одну политику для обеих сторон,
возвращает два агрегата и challenger-minus-baseline дельты HitRate/NDCG, без
статистических заявлений об uplift.

## Метрики и знаменатели

Пусть C — исходная cohort, E — cases, допустимые по данным/режиму, Q — cases из E
с qualified target, L(u) — выданный список. Причины исключений не зависят от
алгоритма. Отсутствие target не означает ошибку данных при подтверждённой полноте
labels: такие cases входят в E, но не Q.

| Метрика | Определение |
|---|---|
| HitRate@K = Recall@K | Среднее `1[target ∈ L(u)]` по Q; один target, характеристики не независимы |
| NDCG@K | Среднее `1/log2(rank+1)` при попадании, иначе 0, по Q |
| Recommendation coverage | Доля `len(L)>0` по E; это customer coverage, не catalog coverage |
| Full Top-K coverage | Доля `len(L)=K` по E |
| Mean recommendations | Средняя длина списка по E |
| Target reachability | Доля `target ∈ candidates` по Q |
| Empty rate | Доля `len(L)=0` по E |
| Incomplete rate | Доля `0<len(L)<K` по E, пустые отдельно |

Недостижимые targets остаются в Q и дают нулевые HitRate/NDCG. Пустые и refused
списки также дают ноль, если target есть. Никакой условной оценки только по
успешным выдачам или достижимым targets нет. При пустом знаменателе метрика None,
а не фиктивный ноль. При E=0 общий status UNAVAILABLE, включая пустую cohort.
При Q=0 качество неопределено, но coverage по непустому E остаётся определённым.

`cohort_cases`, `evaluated_cases`, `excluded_cases`, `proxy_cases`, `quality_cases`,
`no_target_cases`, `reachable_targets`, `unreachable_targets`, empty/incomplete counts
сохраняются явно. `exclusion_counts` может содержать несколько причин одного
case; сумма причин не обязана равняться `excluded_cases`. Аналогично отдельные
`evidence_issue_counts` считают ограничения по ролям, включая excluded cases.
`refusal_counts` считает причины пустой выдачи по E. Все reason keys фиксированы.

Агрегаты frozen, не содержат ID клиентов/товаров, payload, reviewer, paths или
per-customer hashes. Case/evidence/list repr скрывают поля; ошибки не включают
входные значения. Нельзя публиковать `asdict(case)` как агрегированный отчёт.

## Проверки baseline и ограничения

Синтетические интеграционные тесты вызывают настоящий `compose()` с небольшими
in-memory CatalogItem/PurchaseEvidence/RawBprResult. Они сохраняют полный ranking,
его нормализацию, фильтрацию покупок/uncertain, уникальность targets и production
порядок. Формулы matching не копируются в evaluator.

Зафиксированы оба ограничения baseline: greedy может дать одну пару вместо двух;
остановка после десяти успехов может пропустить позднюю пару с большим score.
Это тесты текущего поведения, не утверждение его оптимальности и не доказательство
бизнес-пользы альтернативы. Синтетические метрики не оценивают реальных клиентов.

Реальные as-of каталог/категории/остатки, доказанная история модели и identity,
независимые purchase labels и их полнота остаются зависимостями следующего этапа.
SQLite index STALE не восстанавливается этим модулем. Нет CLI реального benchmark,
новых зависимостей, автозапуска экспериментов, обучения или публикации моделей.
