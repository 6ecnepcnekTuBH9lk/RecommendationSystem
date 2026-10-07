# Global temporal evaluation (TRAIN-02E)

Исследовательский API находится в `Application/evaluation`; production GUI/CLI
его автоматически не вызывают. Вход — поток `ResolvedInteraction`, уже прошедший
существующие `CustomerIdResolver`, `InteractionBuilder` и `ProductResolver`.
Raw технические ID и готовый `PreparedBprData` не заменяют этот вход: последний
уже потерял timestamps и отдельные occurrences. Загрузка исходных данных остаётся
внешней ответственностью; protocol не читает файлы и не разрешает identity повторно.

## Контракт

`TemporalConfig(validation_start, test_start, test_end=None, min_history_events=10)`:
aware datetime, нормализация в UTC, строго возрастающие границы.
`build_temporal_protocol(events, config)` возвращает два immutable `TemporalSnapshot`:

| Snapshot | History | Future |
| --- | --- | --- |
| validation | timestamp < validation_start | validation_start <= timestamp < test_start |
| test | timestamp < test_start | test_start <= timestamp < test_end, либо без верхней границы |

Событие ровно на cutoff относится к future. Событие ровно на test_end исключено.
Test history включает **все** события validation-периода, включая повторы и targets.
Cutoffs не усекутся до календарной даты. Exact ties: PURCHASE → FAVORITE → VIEW,
затем стабильный входной source order внутри типа, как в canonical BPR preparation.

Snapshot содержит cutoff/window end, полный tuple history, canonical item universe,
read-only `seen_at_cutoff`, tuple cases и diagnostics. Case содержит canonical
customer/item, timestamp, interaction type и frozen seen-set пользователя.

`seen(u,t) = {item(e): customer(e)=u, timestamp(e)<t}`. Target — первое по
указанному порядку future event, item которого отсутствует в **фиксированном**
seen(u,t). History внутри future window не расширяется; поздние события не меняют
target. Нет novel target — нет case, увеличивается `no_novel_target_users`.

Primary benchmark: минимум 10 history **events** до cutoff (параметр настраивается,
повторы считаются), novel для пользователя, известен глобально в history других
или того же пользователя. Item universe/mappings/seen строятся только из history.
Если первый novel item глобально cold, case исключён; более поздний warm item
**не подставляется**. Cold/sparse пользователи также исключены с отдельными счётчиками.

Diagnostics: events/users/items истории, events/users future, пользователи future,
прошедшие history threshold, их novel targets до warm-item фильтра, cold users,
sparse users, cold-item targets, no-novel users, финальные targets по типам,
распределение количества history events по пользователям и средняя задержка target
от cutoff в секундах (None при пустом benchmark). Причины исключения взаимоисключающие.

## Scoring и BPR

`evaluate_snapshot(snapshot, scorer, k=10)` использует `score(customer_id,
candidate_items) -> sequence[finite float]`: один score на каждый candidate в
переданном порядке. Candidates — historical universe минус seen пользователя.
При одинаковых scores порядок canonical item ID. Recall@K = HitRate@K для одного
target; NDCG@K использует общую с production математическую функцию.
Пустой benchmark: count=0, обе метрики=0; count обязателен для интерпретации.
BPR, будущие Profile и Hybrid должны получать **те же cases/candidates**, без
собственной повторной генерации targets или изменения primary population.

```python
from Application.evaluation.temporal import TemporalConfig, build_temporal_protocol
from Application.evaluation.bpr import prepare_bpr_snapshot, evaluate_bpr_snapshot
from Application.model.BPRMF import train_prepared_data
from dataclasses import replace

protocol = build_temporal_protocol(resolved_events, TemporalConfig(validation_start, test_start))
data = prepare_bpr_snapshot(protocol.validation)
research_cfg = replace(train_cfg, early_stop=False, use_item_features=False)
# Заранее выбранное число epochs; features включать только с допустимым as-of каталогом.
model, _ = train_prepared_data(research_cfg, data.training, device)
validation_metrics = evaluate_bpr_snapshot(model, data, k=research_cfg.topk, device=device)
# После выбора гиперпараметров: новый data/model из protocol.test.history,
# затем одна финальная оценка protocol.test, без подбора параметров по test.
```

BPR adapter использует `prepare_bpr_training_history`: все supplied events в train,
eval arrays пусты, FULL_TIMESTAMP и существующие веса/агрегация. Обычный
`prepare_bpr()` сохраняет TRAIN-02D internal holdout. Double holdout отсутствует.
Existing trainer с пустыми eval arrays не имеет meaningful early-stop метрики:
для research явно выключать early_stop и выбирать epochs до final test.
External early stopping/grid search и production refit здесь не реализованы.

Backend обязан обучаться/строить profile/нормализацию только по snapshot.history;
никаких current published SeenItemsIndex, future interactions или статистик future.
Catalog features и identity metadata, если используются, должны быть допустимы на
prediction момент; их исторические версии этот слой не восстанавливает.
BPR adapter проверяет mappings, positives и размеры embeddings, но не может доказать
происхождение произвольной переданной модели. Metadata provenance, экспериментальные
run IDs, автоматический tuning и отдельный cold-start benchmark — дальнейшие задачи.
Validation используется для выбора параметров; test — один раз после их фиксации.
Protocol не создаёт production artifacts, не запускает training сам и не публикует модель.
