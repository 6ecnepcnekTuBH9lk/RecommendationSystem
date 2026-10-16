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


def similarity(source, target, catalog):
    if not source.get("ВидНоменклатуры") or source.get("ВидНоменклатуры") != target.get("ВидНоменклатуры"):
        return 0.0
    source_gender, target_gender = source.get("ПолНоменклатуры"), target.get("ПолНоменклатуры")
    if not source_gender or not target_gender or source_gender != target_gender:
        return 0.0
    score = 0.0
    for key, weight in WEIGHTS.items():
        left, right = source.get(key), target.get(key)
        if key == "НазваниеНаСайте":
            left, right = source.name, target.name
        elif key == "ТипКоллекции":
            left, right = _collection_type(source.get("Коллекция")), _collection_type(target.get("Коллекция"))
        if not left or not right:
            continue
        if key == "КатегорияНаСайте":
            a, b = catalog.category_paths.get(left), catalog.category_paths.get(right)
            match = len(a & b) / len(a | b) if a and b else 0.0
        elif key == "НазваниеНаСайте":
            a, b = set(re.findall(r"\w+", left.casefold())), set(re.findall(r"\w+", right.casefold()))
            match = len(a & b) / len(a | b) if a and b else 0.0
        else:
            match = float(left == right)
        score += weight * match
    return min(1.0, score)


@dataclass(frozen=True, slots=True)
class MatchSource:
    source_model_item_code: str
    source_name: str
    source_raw_score: float
    source_raw_rank: int
    similarity_score: float
    match_method: str
    evidence_score: float


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


def compose(raw, metadata, catalog, rules, purchases, *, k=10, per_source=10, cancelled=lambda: False):
    """Pure composition; only reliably checked targets can enter the result."""
    if purchases.status != "READY":
        return ()
    rules.validate_catalog(catalog)
    targets = {}
    for item in catalog.items.values():
        if (item.get("ВидНоменклатуры") in rules.kinds and item.get("Коллекция") in rules.collections
                and item.stock is not None and item.stock >= rules.minimum_stock
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
        rank_score = 1 - position / max(1, len(ranked_sources) - 1)
        raw_score = (rec.raw_score - low) / (high - low) if high > low else 1.
        candidates = []
        for item in targets.get(source.get("ВидНоменклатуры"), ()):
            if cancelled():
                return ()
            quality = similarity(source, item, catalog)
            if quality >= MIN_SIMILARITY:
                final = .30 * rank_score + .30 * raw_score + .40 * quality
                if rules.primary_collection is not None:
                    final = (final + .05 * (item.get("Коллекция") == rules.primary_collection)) / 1.05
                candidates.append((final, quality, item.code, item))
        candidates.sort(key=lambda x: (-x[0], x[2]))
        for final, quality, _, target in candidates[:per_source]:
            source_match = MatchSource(rec.model_item_code, source.name, rec.raw_score, rec.rank,
                                       quality, MATCH_METHOD, final)
            evidence.setdefault(target.code, []).append(source_match)
    ranked = sorted(evidence, key=lambda code: (-max(s.evidence_score for s in evidence[code]), code))
    result = []
    for rank, code in enumerate(ranked[:k], 1):
        item = catalog.items[code]
        sources = tuple(sorted(evidence[code], key=lambda s: (-s.evidence_score, s.source_raw_rank, s.source_model_item_code)))
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
        try:
            # Rank the complete small model item universe; never stop at Raw Top-10.
            raw, metadata = inference.get_candidate_sources(identity.model_user_id, model_dir=model_dir)
            if cancelled():
                return HybridResult("CANCELLED", "Поиск отменён.")
            if raw.model_generation != identity.model_generation:
                return HybridResult("STALE", "Модель изменилась. Повторите поиск.")
            if raw.status is not RawBprStatus.OK:
                return HybridResult("UNAVAILABLE", "Клиент отсутствует в активной модели.", raw_bpr=raw)
            try:
                rules_key = signature(self.rules_path)
                rules = load_rules(self.rules_path)
            except (OSError, ValueError, TypeError, KeyError):
                return HybridResult("RULES_NOT_READY", "Настройте правила формирования рекомендаций в пользовательских настройках.", raw_bpr=raw)
            catalog = self.catalog.load()
            try:
                rules.validate_catalog(catalog)
            except ValueError:
                return HybridResult("RULES_NOT_READY", "Виды или сезоны изменились в каталоге. Обновите правила рекомендаций.", raw_bpr=raw)
            if not isinstance(metadata, dict) or not metadata:
                return HybridResult("METADATA_NOT_READY", "Исторические метаданные модели недоступны.", raw_bpr=raw)
            checked = self.purchases.get(identity.canonical_customer_id, catalog, raw_root=raw_root, cancelled=cancelled)
            if checked.status != "READY":
                return HybridResult("PURCHASES_NOT_READY", "Проверка покупок недоступна или неполна. Рекомендации не сформированы.", raw_bpr=raw, purchases=checked)
            items = compose(raw, metadata, catalog, rules, checked, cancelled=cancelled)
            if cancelled():
                return HybridResult("CANCELLED", "Поиск отменён.")
            result = HybridResult("READY" if items else "EMPTY", "Рекомендации из актуального ассортимента",
                                  items, raw, checked, catalog._key, rules_key, dict(catalog.diagnostics))
            return result if self.is_current(result, raw_root=raw_root) else replace(
                result, status="STALE", items=(), message="Источники или настройки изменились. Повторите поиск.")
        except Exception as exc:
            logger.warning("hybrid_recommendations error=%s", type(exc).__name__)
            return HybridResult("UNAVAILABLE", "Не удалось проверить условия рекомендаций. Повторите поиск.", raw_bpr=raw)

    def is_current(self, result, *, raw_root):
        try:
            return (result.catalog_key == (signature(self.catalog.path), signature(self.catalog.categories_path))
                    and result.rules_key == signature(self.rules_path)
                    and self.purchases.is_current(result.purchases, raw_root=raw_root))
        except (OSError, ValueError):
            return False
