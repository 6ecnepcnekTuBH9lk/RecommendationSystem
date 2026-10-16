"""Deterministic analog matching over unchanged, independently available Raw BPR.

Similarity uses fixed weights (missing evidence earns zero, never a match).
Final score = .30 rank percentile + .30 min/max Raw BPR + .40 similarity.
An explicit primary collection receives a .05 bonus, scaled back to [0,1].
Multiple sources use MAX evidence, not a sum; every retained source is recorded.
Scores are ranking utilities, never purchase probabilities.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
import logging
import re

from Application.paths import INPUT_DATA_DIR
from Application.model.raw_bpr_inference import RawBprResult, RawBprStatus
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


def compose(raw, metadata, catalog, rules, purchases, *, k=10, per_source=10, cancelled=lambda: False):
    """Pure composition; only reliably checked targets can enter the result."""
    if purchases.status != "READY":
        return ()
    rules.validate_catalog(catalog)
    targets = {}
    for item in catalog.items.values():
        if (item.get("ВидНоменклатуры") in rules.kinds and item.get("Коллекция") in rules.collections
                and item.stock is not None and item.stock > 0 and item.stock >= rules.minimum_stock
                and item.code not in purchases.purchased and item.code not in purchases.uncertain):
            targets.setdefault(item.get("ВидНоменклатуры"), []).append(item)
    unique = {}
    for rec in raw.recommendations:
        unique.setdefault(rec.model_item_code, rec)
    ranked_sources = tuple(unique.values())
    scores = [r.raw_score for r in ranked_sources]
    low, high = (min(scores), max(scores)) if scores else (0., 0.)
    evidence = {}
    for position, rec in enumerate(ranked_sources):
        if cancelled():
            return ()
        values = metadata.get(rec.model_item_code)
        if not isinstance(values, dict):
            continue  # Never substitute current attributes for missing training data.
        source = item_from_metadata(rec.model_item_code, values)
        rank_score, raw_score = _normalized_scores(position, len(ranked_sources), rec.raw_score, low, high)
        candidates = []
        for item in targets.get(source.get("ВидНоменклатуры"), ()):
            if cancelled():
                return ()
            quality = similarity(source, item, catalog)
            if quality >= MIN_SIMILARITY:
                final = _rating(rank_score, raw_score, quality, rules.primary_collection is not None,
                                item.get("Коллекция") == rules.primary_collection)
                candidates.append((final, quality, item.code, item))
        candidates.sort(key=lambda x: (-x[0], x[2]))
        for final, quality, _, target in candidates[:per_source]:
            source_match = MatchSource(rec.model_item_code, source.name, rec.raw_score, rec.rank,
                                       quality, MATCH_METHOD, final, source.get("Коллекция"))
            evidence.setdefault(target.code, []).append(source_match)
    ranked = sorted(evidence, key=lambda code: (-max(s.evidence_score for s in evidence[code]), code))
    result = []
    positions = {r.model_item_code: i for i, r in enumerate(ranked_sources)}
    for rank, code in enumerate(ranked[:k], 1):
        item = catalog.items[code]
        sources = tuple(sorted(evidence[code], key=lambda s: (-s.evidence_score, s.source_raw_rank, s.source_model_item_code)))
        best = sources[0]
        position = positions[best.source_model_item_code]
        r, b = _normalized_scores(position, len(ranked_sources), best.source_raw_score, low, high)
        terms = _rating(r, b, best.similarity_score, rules.primary_collection is not None,
                        item.get("Коллекция") == rules.primary_collection, explain=True)
        features = similarity_breakdown(item_from_metadata(best.source_model_item_code, metadata[best.source_model_item_code]), item, catalog)
        explanation = ScoreExplanation(position, len(ranked_sources), low, high, r, b, *terms, rules.primary_collection, features)
        sources = (replace(best, calculation=explanation), *sources[1:])
        result.append(HybridItem(code, item.name, item.get("ВидНоменклатуры"), item.get("Коллекция"),
                                 item.stock, item.get("ТитульнаяФотография"), rank, sources[0].evidence_score, sources))
    return tuple(result)


class HybridRecommendationService:
    def __init__(self, *, catalog_path=INPUT_DATA_DIR / "nomenclature.csv",
                 categories_path=INPUT_DATA_DIR / "site_categories.csv", rules_path=RULES_PATH, purchases=None):
        self.catalog = RecommendationCatalog(catalog_path, categories_path)
        self.rules_path = rules_path
        self.purchases = purchases if purchases is not None else PurchasedProductsService()

    def get(self, identity, inference, *, model_dir, raw_root, cancelled=lambda: False):
        raw = None
        historical = ()
        def make_result(*args, **kwargs):
            return HybridResult(*args, historical_items=historical, **kwargs)
        try:
            # Rank the complete small model item universe; never stop at Raw Top-10.
            raw, metadata = inference.get_candidate_sources(identity.model_user_id, model_dir=model_dir)
            if isinstance(metadata, dict):
                historical = tuple(HistoricalItem(rec.model_item_code, item.name, item.get("Коллекция"))
                                   for rec in raw.recommendations[:10]
                                   for item in (item_from_metadata(rec.model_item_code, metadata.get(rec.model_item_code)
                                                if isinstance(metadata.get(rec.model_item_code), dict) else {}),))
            if cancelled():
                return make_result("CANCELLED", "Поиск отменён.")
            if raw.model_generation != identity.model_generation:
                return make_result("STALE", "Модель изменилась. Повторите поиск.")
            if raw.status is not RawBprStatus.OK:
                return make_result("UNAVAILABLE", "Клиент отсутствует в активной модели.", raw_bpr=raw)
            try:
                rules_key = signature(self.rules_path)
                rules = load_rules(self.rules_path)
            except (OSError, ValueError, TypeError, KeyError):
                return make_result("RULES_NOT_READY", "Настройте правила формирования рекомендаций в пользовательских настройках.", raw_bpr=raw)
            catalog = self.catalog.load()
            try:
                rules.validate_catalog(catalog)
            except ValueError:
                return make_result("RULES_NOT_READY", "Виды или сезоны изменились в каталоге. Обновите правила рекомендаций.", raw_bpr=raw)
            if not isinstance(metadata, dict) or not metadata:
                return make_result("METADATA_NOT_READY", "Исторические метаданные модели недоступны.", raw_bpr=raw)
            checked = self.purchases.get(identity.canonical_customer_id, catalog, raw_root=raw_root, cancelled=cancelled)
            if checked.status != "READY":
                return make_result("PURCHASES_NOT_READY", "Проверка покупок недоступна или неполна. Рекомендации не сформированы.", raw_bpr=raw, purchases=checked)
            items = compose(raw, metadata, catalog, rules, checked, cancelled=cancelled)
            if cancelled():
                return make_result("CANCELLED", "Поиск отменён.")
            result = make_result("READY" if items else "EMPTY", "Рекомендации из актуального ассортимента",
                                  items, raw, checked, catalog._key, rules_key, dict(catalog.diagnostics))
            return result if self.is_current(result, raw_root=raw_root) else replace(
                result, status="STALE", items=(), message="Источники или настройки изменились. Повторите поиск.")
        except Exception as exc:
            logger.warning("hybrid_recommendations error=%s", type(exc).__name__)
            return make_result("UNAVAILABLE", "Не удалось проверить условия рекомендаций. Повторите поиск.", raw_bpr=raw)

    def is_current(self, result, *, raw_root):
        try:
            return (result.catalog_key == (signature(self.catalog.path), signature(self.catalog.categories_path))
                    and result.rules_key == signature(self.rules_path)
                    and self.purchases.is_current(result.purchases, raw_root=raw_root))
        except (OSError, ValueError):
            return False
