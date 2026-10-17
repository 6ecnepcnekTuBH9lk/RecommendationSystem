"""Deterministic analog matching over unchanged, independently available Raw BPR.

Similarity uses fixed weights (missing evidence earns zero, never a match).
Final score = .30 rank percentile + .30 min/max Raw BPR + .40 similarity.
An explicit primary collection receives a .05 bonus, scaled back to [0,1].
Sources claim one best unused target each, in original BPR rank order.
The first ten successful pairs are then sorted by their unchanged final scores.
Scores are ranking utilities, never purchase probabilities.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
import logging
import re

from Application.paths import INPUT_DATA_DIR
from Application.model.raw_bpr_inference import RawBprRecommendation, RawBprResult, RawBprStatus
from Application.purchased_products import PurchaseEvidence, PurchasedProductsService
from Application.recommendation_catalog import RecommendationCatalog, item_from_metadata, signature
from Application.recommendation_rules import RULES_PATH, load_rules


logger = logging.getLogger(__name__)
WEIGHTS = {"ВидНоменклатуры": .12, "ПолНоменклатуры": .10, "КатегорияНаСайте": .20,
           "СтилеваяГруппа": .12, "Марка": .10, "ГруппаСоставов": .10,
           "ВидАссортимента": .08, "НазваниеНаСайте": .08, "СезонНоски": .05, "ТипКоллекции": .05}
MIN_SIMILARITY = .55
MATCH_METHOD = "weighted_attributes_v1"


def _collection_type(value):
    # A soft feature only. Exact collection values in rules remain distinct;
    # this does not infer wearing season or merge carry-over collections.
    if value in ("NOS", "Всесезонный"):
        return value
    match = re.match(r"^(Весна-Лето|Осень-Зима) \d{4}(?:$| )", value)
    return match.group(1) if match else ""


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    feature: str
    historical_value: str
    target_value: str
    weight: float
    match: float
    contribution: float


def _feature_values(source, target, catalog):
    source_gender, target_gender = source.get("ПолНоменклатуры"), target.get("ПолНоменклатуры")
    compatible = (bool(source.get("ВидНоменклатуры")) and source.get("ВидНоменклатуры") == target.get("ВидНоменклатуры")
                  and bool(source_gender) and bool(target_gender) and source_gender == target_gender)
    for key, weight in WEIGHTS.items():
        left, right = source.get(key), target.get(key)
        if key == "НазваниеНаСайте":
            left, right = source.name, target.name
        elif key == "ТипКоллекции":
            left, right = _collection_type(source.get("Коллекция")), _collection_type(target.get("Коллекция"))
        if not compatible or not left or not right:
            match = 0.0
        elif key == "КатегорияНаСайте":
            a, b = catalog.category_paths.get(left), catalog.category_paths.get(right)
            match = len(a & b) / len(a | b) if a and b else 0.0
        elif key == "НазваниеНаСайте":
            a, b = set(re.findall(r"\w+", left.casefold())), set(re.findall(r"\w+", right.casefold()))
            match = len(a & b) / len(a | b) if a and b else 0.0
        else:
            match = float(left == right)
        yield key, left, right, weight, match, weight * match


def similarity_breakdown(source, target, catalog):
    return tuple(FeatureContribution(*values) for values in _feature_values(source, target, catalog))


def similarity(source, target, catalog):
    return min(1.0, sum(values[-1] for values in _feature_values(source, target, catalog)))


def _normalized_scores(position, count, score, low, high):
    return (1 - position / max(1, count - 1), (score - low) / (high - low) if high > low else 1.)


def _rating(rank_score, raw_score, quality, primary_enabled, primary_match, *, explain=False):
    rank_term, bpr_term, match_term = .30 * rank_score, .30 * raw_score, .40 * quality
    base = rank_term + bpr_term + match_term
    bonus = .05 * primary_match if primary_enabled else 0.
    divisor = 1.05 if primary_enabled else 1.
    final = (base + bonus) / divisor if primary_enabled else base
    return (rank_term, bpr_term, match_term, base, bonus, divisor, final) if explain else final


@dataclass(frozen=True, slots=True)
class ScoreExplanation:
    position: int
    source_count: int
    minimum_bpr_score: float
    maximum_bpr_score: float
    rank_score: float
    normalized_bpr_score: float
    rank_contribution: float
    bpr_contribution: float
    similarity_contribution: float
    before_bonus: float
    bonus: float
    divisor: float
    final_score: float
    primary_collection: str | None
    features: tuple[FeatureContribution, ...]


@dataclass(frozen=True, slots=True)
class HistoricalItem:
    code: str
    name: str
    collection: str


@dataclass(frozen=True, slots=True)
class MatchSource:
    source_model_item_code: str
    source_name: str
    source_raw_score: float
    source_raw_rank: int
    similarity_score: float
    match_method: str
    evidence_score: float
    source_collection: str = ""
    calculation: ScoreExplanation | None = None


@dataclass(frozen=True, slots=True)
class HybridItem:
    target_item_code: str
    target_name: str
    target_kind: str
    target_collection: str
    target_stock: Decimal
    photo_url: str
    final_rank: int
    final_score: float
    sources: tuple[MatchSource, ...]


@dataclass(frozen=True, slots=True, repr=False)
class HybridResult:
    status: str
    message: str
    items: tuple[HybridItem, ...] = ()
    raw_bpr: RawBprResult | None = None
    purchases: PurchaseEvidence | None = None
    catalog_key: tuple | None = None
    rules_key: tuple | None = None
    diagnostics: dict | None = None
    historical_items: tuple[HistoricalItem, ...] = ()
    displayed_raw: tuple[RawBprRecommendation, ...] = ()
    raw_status: str = "UNAVAILABLE"


def _historical_evidence(raw, purchases):
    checked = purchases.historical if purchases is not None else None
    codes = frozenset(rec.model_item_code for rec in raw.recommendations)
    if checked is None or checked.status != "READY" or not codes <= checked.checked_codes:
        return None
    return checked


def _ranked_sources(raw):
    unique = {}
    for rec in raw.recommendations:
        unique.setdefault(rec.model_item_code, rec)
    return tuple(sorted(unique.values(), key=lambda rec: rec.rank))


def _historical_source(code, metadata, rules):
    """Use only snapshot metadata and exact allowed kinds, never current CSV."""
    values = metadata.get(code) if isinstance(metadata, dict) else None
    if not isinstance(values, dict) or rules is None:
        return None
    kind = values.get("ВидНоменклатуры")
    if not isinstance(kind, str) or kind not in rules.kinds:
        return None
    source = item_from_metadata(code, values)
    return source if source.get("ВидНоменклатуры") else None


def select_displayed_raw(raw, purchases, *, metadata=None, rules=None):
    """Up to ten checked historical sources of permitted, known kinds."""
    checked = _historical_evidence(raw, purchases)
    if checked is None:
        return ()
    excluded = checked.purchased | checked.uncertain
    selected = []
    for rec in _ranked_sources(raw):
        if rec.model_item_code not in excluded and _historical_source(rec.model_item_code, metadata, rules) is not None:
            selected.append(rec)
            if len(selected) == 10:
                break
    return tuple(selected)


def compose(raw, metadata, catalog, rules, purchases, *, k=10, per_source=10, cancelled=lambda: False):
    """Greedy matching: one checked historical source claims one free target.

    per_source is retained for call compatibility only; it no longer caps the
    candidates examined for a source. Normalization uses the full ranking.
    """
    historical = _historical_evidence(raw, purchases)
    limit = min(10, k)
    if historical is None or purchases.status != "READY" or limit <= 0 or cancelled():
        return ()
    rules.validate_catalog(catalog)
    targets = {}
    for item in catalog.items.values():
        if (item.get("ВидНоменклатуры") in rules.kinds and item.get("Коллекция") in rules.collections
                and item.stock is not None and item.stock > 0 and item.stock >= rules.minimum_stock
                and item.code not in purchases.purchased and item.code not in purchases.uncertain):
            targets.setdefault(item.get("ВидНоменклатуры"), []).append(item)
    ranked_sources = _ranked_sources(raw)
    scores = [r.raw_score for r in ranked_sources]
    low, high = (min(scores), max(scores)) if scores else (0., 0.)
    excluded_sources = historical.purchased | historical.uncertain
    selected, used_targets = [], set()
    for position, rec in enumerate(ranked_sources):
        if cancelled():
            return ()
        if rec.model_item_code in excluded_sources:
            continue
        source = _historical_source(rec.model_item_code, metadata, rules)
        if source is None:
            continue
        rank_score, raw_score = _normalized_scores(position, len(ranked_sources), rec.raw_score, low, high)
        best = None
        for item in targets.get(source.get("ВидНоменклатуры"), ()):
            if cancelled():
                return ()
            if item.code in used_targets:
                continue
            quality = similarity(source, item, catalog)
            if quality >= MIN_SIMILARITY:
                final = _rating(rank_score, raw_score, quality, rules.primary_collection is not None,
                                item.get("Коллекция") == rules.primary_collection)
                if best is None or (-final, item.code) < (-best[0], best[2].code):
                    best = (final, quality, item)
        if best is None:
            continue
        selected.append((*best, rec, position, source))
        used_targets.add(best[2].code)
        if len(selected) == limit:
            break  # Later sources cannot displace an already assigned pair.
    selected.sort(key=lambda pair: (-pair[0], pair[2].code))
    result = []
    for rank, (final, quality, item, rec, position, source) in enumerate(selected, 1):
        if cancelled():
            return ()
        r, b = _normalized_scores(position, len(ranked_sources), rec.raw_score, low, high)
        terms = _rating(r, b, quality, rules.primary_collection is not None,
                        item.get("Коллекция") == rules.primary_collection, explain=True)
        features = similarity_breakdown(source, item, catalog)
        explanation = ScoreExplanation(position, len(ranked_sources), low, high, r, b, *terms, rules.primary_collection, features)
        match = MatchSource(rec.model_item_code, source.name, rec.raw_score, rec.rank, quality, MATCH_METHOD,
                            final, source.get("Коллекция"), explanation)
        result.append(HybridItem(item.code, item.name, item.get("ВидНоменклатуры"), item.get("Коллекция"),
                                 item.stock, item.get("ТитульнаяФотография"), rank, final, (match,)))
    return tuple(result)


def _purchase_failure_message(evidence, *, scope):
    reason = {
        "INDEX_NOT_READY": "индекс истории отсутствует или требует обновления",
        "SOURCE_UNAVAILABLE": "источники истории недоступны",
        "SOURCE_INCONSISTENT": "источники истории несогласованы",
        "INCOMPLETE_COVERAGE": "покрытие истории заказов неполное",
        "UNRESOLVED_PRODUCTS": "часть товарных идентификаторов не удалось достоверно проверить",
        "CANCELLED": "проверка отменена",
    }.get(evidence.status if evidence is not None else None, "полнота проверки не подтверждена")
    if evidence is not None and evidence.diagnostics.get("failure_reason") == "sources_changed":
        reason = "данные истории изменились во время поиска; повторите поиск"
    return f"Проверка покупок {scope}: {reason}. Рекомендации не сформированы."


def _log_purchase_checks(checked, raw):
    historical = checked.historical
    for scope, evidence in (("current", checked), ("historical", historical)):
        # Explicit numeric/boolean whitelist; no IDs, raw statuses or object dumps.
        metrics = {name: evidence.diagnostics.get(name) if evidence is not None else None for name in (
            "coverage_complete", "unresolved_non_cancelled_products", "unsafe_unresolved_identifier_lines",
            "conflicting_orders", "unknown_status_lines", "uncertain_codes", "purchased_codes",
            "historical_out_of_scope_codes", "historical_out_of_scope_lines", "normalized_suffix_space_lines",
        )}
        reasons = evidence.diagnostics.get("unresolved_identifier_reasons", {}) if evidence is not None else {}
        reasons = {name: reasons.get(name, 0) for name in (
            "missing_reference", "unsupported_namespace", "invalid_identifier", "incomplete_identifier",
            "damaged_identity_prefix", "nonprintable_identifier", "ambiguous_prefix",
        )}
        failure = evidence.diagnostics.get("failure_reason") if evidence is not None else None
        failure = failure if failure in ("missing_index", "sources_changed", "source_unavailable", "stale_index", "index_error") else "none"
        logger.info("purchase_check scope=%s status=%s metrics=%s reasons=%s failure=%s", scope,
                    evidence.status if evidence is not None else "ABSENT", metrics, reasons, failure)
    logger.info("purchase_check stage=scopes historical_present=%s historical_codes_complete=%s generation_equal=%s source_version_equal=%s",
                historical is not None,
                historical is not None and all(r.model_item_code in historical.checked_codes for r in raw.recommendations),
                historical is not None and checked.generation == historical.generation,
                historical is not None and checked.source_version == historical.source_version)


class HybridRecommendationService:
    def __init__(self, *, catalog_path=INPUT_DATA_DIR / "nomenclature.csv",
                 categories_path=INPUT_DATA_DIR / "site_categories.csv", rules_path=RULES_PATH, purchases=None):
        self.catalog = RecommendationCatalog(catalog_path, categories_path)
        self.rules_path = rules_path
        self.purchases = purchases if purchases is not None else PurchasedProductsService()

    def get(self, identity, inference, *, model_dir, raw_root, cancelled=lambda: False):
        raw = None
        historical = ()
        displayed = ()
        checked = None
        rules_key = None
        raw_status = "UNAVAILABLE"
        def make_result(*args, **kwargs):
            result = HybridResult(*args, raw_bpr=raw, purchases=checked, historical_items=historical,
                                  displayed_raw=displayed, raw_status=raw_status, rules_key=rules_key, **kwargs)
            if result.status != "CANCELLED" and raw_status in ("READY", "EMPTY") and not self._rules_current(rules_key):
                return replace(result, status="STALE", raw_status="STALE", items=(), displayed_raw=(), historical_items=(),
                               message="Правила рекомендаций изменились. Повторите поиск.")
            return result
        try:
            # Rank the complete small model item universe; never stop at Raw Top-10.
            raw, metadata = inference.get_candidate_sources(identity.model_user_id, model_dir=model_dir)
            if cancelled():
                return make_result("CANCELLED", "Поиск отменён.")
            if raw.model_generation != identity.model_generation or raw.model_user_id != identity.model_user_id:
                return make_result("STALE", "Модель изменилась. Повторите поиск.")
            if raw.status is not RawBprStatus.OK:
                return make_result("UNAVAILABLE", "Клиент отсутствует в активной модели.")
            # Current CSV/settings failures must not bypass historical verification.
            # Both identity scopes share one SQLite read, but NEVER one resolver.
            catalog = None
            try:
                catalog = self.catalog.load()
            except (OSError, ValueError, TypeError, KeyError) as exc:
                # Historical verification does not require today's catalog.
                logger.warning("hybrid_catalog error=%s", type(exc).__name__)
            checked = self.purchases.get(identity.canonical_customer_id, catalog, raw_root=raw_root, cancelled=cancelled,
                                         historical_codes=frozenset(rec.model_item_code for rec in raw.recommendations))
            _log_purchase_checks(checked, raw)
            if cancelled():
                return make_result("CANCELLED", "Поиск отменён.")
            try:
                before_rules = signature(self.rules_path)
                rules = load_rules(self.rules_path)
                if before_rules != signature(self.rules_path):
                    raw_status = "STALE"
                    return make_result("STALE", "Правила рекомендаций изменились. Повторите поиск.")
                rules_key = before_rules
            except (OSError, ValueError, TypeError, KeyError):
                raw_status = "RULES_NOT_READY"
                return make_result("RULES_NOT_READY", "Настройте правила формирования рекомендаций в пользовательских настройках.")
            if not isinstance(metadata, dict) or not metadata:
                raw_status = "METADATA_NOT_READY"
                return make_result("METADATA_NOT_READY", "Исторические метаданные модели недоступны.")
            if _historical_evidence(raw, checked) is None:
                raw_status = "PURCHASES_NOT_READY"
                return make_result("PURCHASES_NOT_READY", _purchase_failure_message(
                    checked.historical if checked.historical is not None else checked, scope="исторических товаров BPR-MF"))
            displayed = select_displayed_raw(raw, checked, metadata=metadata, rules=rules)
            historical = tuple(HistoricalItem(rec.model_item_code, item.name, item.get("Коллекция"))
                               for rec in displayed for item in (_historical_source(rec.model_item_code, metadata, rules),))
            raw_status = "READY" if displayed else "EMPTY"
            if catalog is None:
                return make_result("UNAVAILABLE", "Актуальный каталог недоступен. Повторите поиск.")
            try:
                rules.validate_catalog(catalog)
            except ValueError:
                return make_result("RULES_NOT_READY", "Виды или сезоны изменились в каталоге. Обновите правила рекомендаций.")
            if checked.status != "READY":
                return make_result("PURCHASES_NOT_READY", _purchase_failure_message(checked, scope="актуальных товаров"))
            items = compose(raw, metadata, catalog, rules, checked, cancelled=cancelled)
            if cancelled():
                return make_result("CANCELLED", "Поиск отменён.")
            message = ("Рекомендации из актуального ассортимента" if items else
                       "После проверки покупок и разрешенных видов нет допустимых исторических источников." if not displayed else
                       "Подходящих товаров актуального ассортимента не найдено.")
            result = make_result("READY" if items else "EMPTY", message,
                                 items, catalog_key=catalog._key, diagnostics=dict(catalog.diagnostics))
            return result if self.is_current(result, raw_root=raw_root) else replace(
                result, status="STALE", items=(), message="Источники или настройки изменились. Повторите поиск.")
        except Exception as exc:
            logger.warning("hybrid_recommendations error=%s", type(exc).__name__)
            return make_result("UNAVAILABLE", "Не удалось проверить условия рекомендаций. Повторите поиск.")

    def _rules_current(self, rules_key):
        try:
            return rules_key is not None and rules_key == signature(self.rules_path)
        except OSError:
            return False

    def raw_is_current(self, result, *, raw_root):
        checked = _historical_evidence(result.raw_bpr, result.purchases) if result.raw_bpr is not None else None
        return (result.raw_status in ("READY", "EMPTY") and self._rules_current(result.rules_key) and checked is not None
                and self.purchases.is_current(checked, raw_root=raw_root))

    def is_current(self, result, *, raw_root):
        try:
            return (result.catalog_key == (signature(self.catalog.path), signature(self.catalog.categories_path))
                    and result.rules_key == signature(self.rules_path)
                    and self.purchases.is_current(result.purchases, raw_root=raw_root))
        except (OSError, ValueError):
            return False
