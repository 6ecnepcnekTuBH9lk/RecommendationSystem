"""Display contracts and one-customer lookup; no widgets, history or exports."""

import csv
import logging
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Application.hybrid_recommendations import HybridItem, HybridResult


logger = logging.getLogger(__name__)

from Application import customer_contact_search as contact_backend
from Application.customer_profile_service import CustomerProfileResult, CustomerProfileService, ProfileStatus

from Application.model.customer_identity_bridge import (
    CustomerIdentityBridge, CustomerModelIdentityResult, CustomerModelIdentityStatus,
)
from Application.model.raw_bpr_inference import RawBprInferenceService, RawBprRecommendation, RawBprStatus
from Application.paths import INPUT_DATA_DIR, PROJECT_ROOT


class ViewState(str, Enum):
    INITIAL = "initial"
    LOADING = "loading"
    READY = "ready"
    EMPTY = "empty"
    NOT_CONNECTED = "not_connected"
    ERROR = "error"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class DisplayField:
    title: str
    value: str | None = None
    state: ViewState = ViewState.NOT_CONNECTED
    message: str | None = None
    note: str | None = None

    def __post_init__(self):
        if not isinstance(self.title, str) or not self.title or not isinstance(self.state, ViewState):
            raise TypeError("A display field requires a title and typed state")
        if self.state is ViewState.READY:
            if not isinstance(self.value, str):
                raise TypeError("Ready display values must be explicitly formatted strings")
        elif self.value is not None:
            raise ValueError("Unavailable display fields cannot contain a value")
        if any(v is not None and not isinstance(v, str) for v in (self.message, self.note)):
            raise TypeError("Display messages must be strings")


@dataclass(frozen=True, slots=True)
class SectionContent:
    state: ViewState
    message: str
    fields: tuple[DisplayField, ...] = ()

    def __post_init__(self):
        if (not isinstance(self.state, ViewState) or not isinstance(self.message, str)
                or not isinstance(self.fields, tuple) or any(not isinstance(f, DisplayField) for f in self.fields)):
            raise TypeError("Invalid immutable section content")


@dataclass(frozen=True, slots=True)
class ProductContent:
    recommendation: RawBprRecommendation
    name: str = ""
    photo_url: str = ""
    reference_message: str = ""
    collection: str = ""
    hybrid_item: "HybridItem | None" = None
    display_rank: int | None = None


@dataclass(frozen=True, slots=True, repr=False)
class CustomerCardResult:
    state: ViewState
    message: str
    identity: CustomerModelIdentityResult | None = None
    model_generation: str | None = None
    products: tuple[ProductContent, ...] = ()
    profile: CustomerProfileResult | None = None
    contact_search: contact_backend.ContactSearchResult | None = None
    hybrid: "HybridResult | None" = None
    raw_products: tuple[ProductContent, ...] = ()
    raw_state: ViewState | None = None
    raw_message: str = ""


def contact_search_failure(result):
    """No profile/model identity is confirmed by a failed contact lookup."""
    status = contact_backend.ContactSearchStatus
    messages = {
        status.AMBIGUOUS: (ViewState.UNAVAILABLE,
                          f"Найдено несколько клиентов ({result.candidate_count}). Уточните поиск или используйте MindboxID."),
        status.NOT_FOUND: (ViewState.EMPTY, "Клиенты с указанным контактом не найдены."),
        status.INVALID_INPUT: (ViewState.ERROR, "Некорректный формат. Проверьте значение и подсказку для выбранного типа поиска."),
        status.INDEX_NOT_READY: (ViewState.UNAVAILABLE, "Поисковый индекс отсутствует или устарел. Необходимо построить или обновить индекс контактов."),
        status.SOURCE_UNAVAILABLE: (ViewState.UNAVAILABLE, "Источник поиска клиентов недоступен. Повторите поиск позднее."),
        status.SOURCE_INCONSISTENT: (ViewState.ERROR, "Источники поиска клиентов несогласованы. Карточка недоступна."),
    }
    state, message = messages[result.status]
    return CustomerCardResult(state, message, contact_search=result)


def validate_contact_match(result, *, raw_root):
    """Reuse backend freshness gates at handoff, without a contact query or scan.

    The backend currently exposes search + source_version, but no public
    is_current(). Reuse its own metadata readers, keeping that compatibility
    boundary here. Both lookups and this SQLite freshness check run in the worker.
    """
    status = contact_backend.ContactSearchStatus
    stage = "lock"
    try:
        root = Path(raw_root).resolve()
        with contact_backend._reader_lock(root):
            stage = "sources"
            sources = contact_backend._sources(root)
            if result.source_version != sources.version:
                logger.debug("contact_validation stage=source_version status=INDEX_NOT_READY")
                return contact_backend.ContactSearchResult(status.INDEX_NOT_READY)
            path = contact_backend.CustomerContactSearchService().index_path(raw_root=root)
            if not path.exists():
                return contact_backend.ContactSearchResult(status.INDEX_NOT_READY)
            stage = "sqlite_open"
            with contact_backend._read_database(path) as connection:
                stage = "metadata"
                contact_backend._metadata(connection, sources)
        logger.debug("contact_validation stage=complete status=MATCHED")
        return result
    except contact_backend._IndexNotReady:
        logger.debug("contact_validation stage=%s status=INDEX_NOT_READY", stage)
        return contact_backend.ContactSearchResult(status.INDEX_NOT_READY)
    except Exception as exc:
        logger.warning("contact_validation stage=%s error=%s", stage, type(exc).__name__)
        return contact_backend.ContactSearchResult(status.SOURCE_UNAVAILABLE if contact_backend._unavailable(exc)
                                                 else status.SOURCE_INCONSISTENT)


IDENTITY_MESSAGES = {
    CustomerModelIdentityStatus.MATCHED_EXACT: "Найден единственный профиль в нейросетевой модели",
    CustomerModelIdentityStatus.MATCHED_VIA_MERGE: "Найден единственный исторический профиль после объединения клиентов",
    CustomerModelIdentityStatus.AMBIGUOUS: "У клиента несколько исторических профилей в модели. Рекомендации недоступны",
    CustomerModelIdentityStatus.NOT_IN_MODEL: "Клиент отсутствует в данных нейросетевой модели. Персональные рекомендации недоступны",
    CustomerModelIdentityStatus.INVALID_ID: "Введите корректный MindboxID без пробелов по краям",
    CustomerModelIdentityStatus.SOURCE_UNAVAILABLE: "Источники сопоставления клиентов недоступны. Повторите поиск позднее",
    CustomerModelIdentityStatus.SOURCE_INCONSISTENT: "Источники сопоставления клиентов несовместимы. Рекомендации недоступны",
}


def _version(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


class CustomerCardService:
    """Owned by a single worker; backend caches and catalog are reused there."""

    def __init__(self, *, bridge=None, inference=None, profile_service=None, model_dir=PROJECT_ROOT / "model",
                 raw_root=INPUT_DATA_DIR / "MindboxRaw", catalog_path=INPUT_DATA_DIR / "nomenclature.csv", hybrid=None):
        self.bridge = bridge if bridge is not None else CustomerIdentityBridge()
        self.inference = inference if inference is not None else RawBprInferenceService()
        self.profile_service = profile_service if profile_service is not None else CustomerProfileService()
        self.profile_observer = None
        self.hybrid = hybrid
        self.model_dir, self.raw_root, self.catalog_path = Path(model_dir), Path(raw_root), Path(catalog_path)
        self._catalog_key = None
        self._catalog = {}
        self._catalog_ambiguous = set()
        self._catalog_history_fields = {}
        self._history_resolver = None

    def _references(self):
        key = _version(self.catalog_path)
        if self._catalog_key == key:
            return self._catalog
        values, ambiguous, history_fields = {}, set(), {}
        with self.catalog_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="|")
            if not reader.fieldnames or "КодНоменклатуры" not in reader.fieldnames:
                raise ValueError("Invalid display catalog")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Malformed display catalog row")
                code = row["КодНоменклатуры"].strip()
                if code:
                    info = ((row.get("НазваниеНаСайте") or "").strip() or (row.get("Номенклатура") or "").strip(),
                            (row.get("ТитульнаяФотография") or "").strip())
                    fields = ((row.get("ВидНоменклатуры") or "").strip(), (row.get("Коллекция") or "").strip())
                    if code in values:
                        if values[code] != info:
                            ambiguous.add(code)
                        fields = tuple(value if value == previous else "" for value, previous in zip(fields, history_fields[code]))
                    values[code] = info
                    history_fields[code] = fields
        if _version(self.catalog_path) != key:
            raise ValueError("Display catalog changed during reading")
        self._catalog_key, self._catalog = key, values
        self._catalog_ambiguous = ambiguous
        self._catalog_history_fields = history_fields
        self._history_resolver = None
        return values

    def history_product_references(self, page):
        """Reuse the display catalog and existing product identity rules in worker."""
        from Application.product_resolution import CatalogDiagnostics, ProductCatalog, ProductResolver, ResolutionStatus
        from Application.customer_history_records import OrderSnapshot
        from Application.photo.photo_processing import _photo_to_url
        metadata = self._references()
        if self._history_resolver is None:
            self._history_resolver = ProductResolver(ProductCatalog(frozenset(metadata), CatalogDiagnostics()))
        resolver = self._history_resolver
        output = {}
        for event in page.events:
            for variant in event.variants:
                detail = variant.event.details
                references = (line.reference for line in detail.lines) if isinstance(detail, OrderSnapshot) else (
                    occurrence.reference for occurrence in detail.products)
                for reference in references:
                    key = reference.product
                    if key is None or key in output:
                        continue
                    match = resolver.resolve(key, strict=False)
                    if match.status is ResolutionStatus.RESOLVED and match.item_id not in self._catalog_ambiguous:
                        name, url = metadata[match.item_id]
                        output[key] = (name, _photo_to_url(url), *self._catalog_history_fields[match.item_id])
        return output

    def is_current(self, result: CustomerCardResult) -> bool:
        return (result.identity is not None and result.identity.is_matched
                and result.model_generation == result.identity.model_generation
                and self.bridge.is_current(result.identity, model_dir=self.model_dir, raw_root=self.raw_root))

    def validate_result(self, result: CustomerCardResult) -> CustomerCardResult:
        """Validate model and purchase evidence independently of profile/history UI."""
        match = result.contact_search
        if match is not None:
            if match.status is not contact_backend.ContactSearchStatus.MATCHED:
                return contact_search_failure(match)
            match = validate_contact_match(match, raw_root=self.raw_root)
            if match.status is not contact_backend.ContactSearchStatus.MATCHED:
                return contact_search_failure(match)
            # A merge/source race must not replace the contact-selected customer.
            confirmed = [result.identity.canonical_customer_id] if result.identity is not None else []
            if result.profile is not None:
                confirmed.append(result.profile.canonical_customer_id)
            if any(value is not None and value != match.canonical_customer_id for value in confirmed):
                return contact_search_failure(contact_backend.ContactSearchResult(contact_backend.ContactSearchStatus.INDEX_NOT_READY))
        if result.state in (ViewState.READY, ViewState.EMPTY) or result.raw_state in (ViewState.READY, ViewState.EMPTY):
            try:
                current = self.is_current(result)
            except Exception:
                current = False
            if not current:
                result = replace(result, state=ViewState.STALE, products=(), raw_products=(), raw_state=ViewState.STALE,
                                 message="Источники рекомендаций изменились. Повторите поиск.")
            elif result.hybrid is not None:
                if result.raw_state in (ViewState.READY, ViewState.EMPTY):
                    try:
                        raw_current = self.hybrid is not None and self.hybrid.raw_is_current(result.hybrid, raw_root=self.raw_root)
                    except Exception:
                        raw_current = False
                    if not raw_current:
                        result = replace(result, raw_products=(), raw_state=ViewState.STALE,
                                         raw_message="История покупок изменилась или недоступна. Повторите поиск.")
                try:
                    hybrid_current = (result.state not in (ViewState.READY, ViewState.EMPTY)
                                      or self.hybrid is not None and self.hybrid.is_current(result.hybrid, raw_root=self.raw_root))
                except Exception:
                    hybrid_current = False
                if not hybrid_current:
                    result = replace(result, state=ViewState.STALE, products=(),
                                     message="Источники или правила рекомендаций изменились. Повторите поиск.")
        profile = result.profile
        if profile is not None and profile.status in (ProfileStatus.FOUND, ProfileStatus.NOT_FOUND):
            profile = self.validate_profile(profile)
            result = replace(result, profile=profile)
            if (profile.status is ProfileStatus.FOUND and result.identity is not None
                  and result.identity.canonical_customer_id is not None
                  and profile.canonical_customer_id != result.identity.canonical_customer_id):
                result = replace(result, state=ViewState.STALE, products=(), raw_products=(), raw_state=ViewState.STALE,
                                 message="Идентификация источников изменилась. Повторите поиск.")
        return result

    def validate_profile(self, profile):
        if profile.status in (ProfileStatus.FOUND, ProfileStatus.NOT_FOUND):
            try:
                current = self.profile_service.is_current(profile, raw_root=self.raw_root)
            except Exception:
                current = False
            if not current:
                return CustomerProfileResult(ProfileStatus.STALE)
        return profile

    def _historical_products(self, hybrid):
        """Historical labels only; optional exact-code current photos in worker."""
        historical = {item.code: item for item in hybrid.historical_items}
        try:
            photos = self._references()
        except Exception:
            photos = {}
        return tuple(ProductContent(
            rec, historical[rec.model_item_code].name if rec.model_item_code in historical else "",
            photos.get(rec.model_item_code, ("", ""))[1] if rec.model_item_code not in self._catalog_ambiguous else "",
            collection=historical[rec.model_item_code].collection if rec.model_item_code in historical else "",
            display_rank=display_rank,
        ) for display_rank, rec in enumerate(hybrid.displayed_raw, 1))

    def load(self, customer_id: str, cancelled=lambda: False) -> CustomerCardResult:
        try:
            profile = self.profile_service.get_profile(customer_id, raw_root=self.raw_root, cancelled=cancelled)
        except Exception:
            profile = CustomerProfileResult(ProfileStatus.SOURCE_INCONSISTENT)
        if cancelled():
            return CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        if self.profile_observer is not None:
            self.profile_observer(profile)
        try:
            result = self._load_recommendations(customer_id, cancelled)
        except Exception:
            result = CustomerCardResult(ViewState.ERROR, "Не удалось получить рекомендации. Повторите поиск.")
        return self.validate_result(replace(result, profile=profile))

    def _load_recommendations(self, customer_id: str, cancelled=lambda: False) -> CustomerCardResult:
        if cancelled():
            return CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        identity = self.bridge.resolve(customer_id, model_dir=self.model_dir, raw_root=self.raw_root)
        if cancelled():
            return CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        if not identity.is_matched:
            state = (ViewState.ERROR if identity.status in (
                CustomerModelIdentityStatus.INVALID_ID, CustomerModelIdentityStatus.SOURCE_UNAVAILABLE,
                CustomerModelIdentityStatus.SOURCE_INCONSISTENT) else ViewState.UNAVAILABLE)
            return CustomerCardResult(state, IDENTITY_MESSAGES[identity.status], identity)
        if self.hybrid is not None:
            hybrid = self.hybrid.get(identity, self.inference, model_dir=self.model_dir,
                                     raw_root=self.raw_root, cancelled=cancelled)
            state = {"READY": ViewState.READY, "EMPTY": ViewState.EMPTY, "STALE": ViewState.STALE,
                     "CANCELLED": ViewState.CANCELLED}.get(hybrid.status, ViewState.UNAVAILABLE)
            products = tuple(ProductContent(
                RawBprRecommendation(item.sources[0].source_raw_rank, item.sources[0].source_model_item_code,
                                     item.sources[0].source_raw_score),
                item.target_name, item.photo_url, collection=item.target_collection, hybrid_item=item,
            ) for item in hybrid.items)
            raw_state, raw_products = ViewState.UNAVAILABLE, ()
            raw = hybrid.raw_bpr
            if raw is not None and raw.status is RawBprStatus.OK and state is not ViewState.CANCELLED:
                if raw.model_generation == identity.model_generation and raw.model_user_id == identity.model_user_id:
                    evidence = hybrid.purchases.historical if hybrid.purchases is not None else None
                    if (evidence is not None and evidence.status == "READY"
                            and all(rec.model_item_code in evidence.checked_codes for rec in raw.recommendations)):
                        raw_products = self._historical_products(hybrid)
                        raw_state = ViewState.READY if raw_products else ViewState.EMPTY
                else:
                    raw_state = ViewState.STALE
            if raw_state is ViewState.STALE:
                state, products = ViewState.STALE, ()
            raw_message = ""
            if raw_state is ViewState.EMPTY:
                raw_message = "После исключения покупок и неопределённых соответствий допустимых рекомендаций нет."
            elif raw_state is ViewState.STALE:
                raw_message = "Модель изменилась. Повторите поиск."
            elif raw_state is ViewState.UNAVAILABLE:
                raw_message = ("Рекомендации BPR-MF недоступны: не удалось достоверно проверить историю покупок."
                               if raw is not None and raw.status is RawBprStatus.OK else "Исходные рекомендации BPR-MF недоступны.")
            return CustomerCardResult(state, hybrid.message, identity, identity.model_generation,
                                      products, hybrid=hybrid, raw_products=raw_products, raw_state=raw_state,
                                      raw_message=raw_message)
        try:
            raw = self.inference.get_recommendations(identity.model_user_id, k=10, model_dir=self.model_dir)
        except Exception:
            return CustomerCardResult(ViewState.ERROR, "Не удалось загрузить рекомендации из модели BPR-MF.", identity)
        if cancelled():
            return CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        if raw.model_generation != identity.model_generation:
            return CustomerCardResult(ViewState.STALE, "Модель изменилась во время поиска. Повторите запрос.", identity)
        if raw.status is not RawBprStatus.OK:
            return CustomerCardResult(ViewState.UNAVAILABLE, "Клиент больше не найден в активной модели.", identity)
        try:
            metadata = self._references()
            reference_error = False
        except Exception:
            metadata, reference_error = {}, True
        products = tuple(ProductContent(
            rec, *metadata.get(rec.model_item_code, ("", "")),
            "Справочник товаров недоступен" if reference_error else (
                "Нет в текущем каталоге" if rec.model_item_code not in metadata else ""),
            collection=(self._catalog_history_fields.get(rec.model_item_code, ("", ""))[1]
                        if not reference_error and rec.model_item_code not in self._catalog_ambiguous else ""),
        ) for rec in raw.recommendations)
        result = CustomerCardResult(ViewState.READY if products else ViewState.EMPTY,
                                    "Оригинальные рекомендации нейросетевой модели без дополнительной фильтрации",
                                    identity, raw.model_generation, products)
        if cancelled():
            return CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        if not self.is_current(result):
            return CustomerCardResult(ViewState.STALE, "Модель или объединения клиентов изменились. Повторите поиск.", identity)
        return result
