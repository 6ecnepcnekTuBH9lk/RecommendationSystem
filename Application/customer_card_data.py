"""Display contracts and one-customer lookup; no widgets, history or exports."""

import csv
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

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


@dataclass(frozen=True, slots=True, repr=False)
class CustomerCardResult:
    state: ViewState
    message: str
    identity: CustomerModelIdentityResult | None = None
    model_generation: str | None = None
    products: tuple[ProductContent, ...] = ()
    profile: CustomerProfileResult | None = None
    contact_search: contact_backend.ContactSearchResult | None = None


def contact_search_failure(result):
    """No profile/model identity is confirmed by a failed contact lookup."""
    status = contact_backend.ContactSearchStatus
    messages = {
        status.AMBIGUOUS: (ViewState.UNAVAILABLE,
                          f"Найдено несколько покупателей ({result.candidate_count}). Уточните поиск или используйте MindboxID."),
        status.NOT_FOUND: (ViewState.EMPTY, "Покупатели с указанным контактом не найдены."),
        status.INVALID_INPUT: (ViewState.ERROR, "Некорректный формат. Проверьте значение и подсказку для выбранного типа поиска."),
        status.INDEX_NOT_READY: (ViewState.UNAVAILABLE, "Поисковый индекс отсутствует или устарел. Необходимо построить или обновить индекс контактов."),
        status.SOURCE_UNAVAILABLE: (ViewState.UNAVAILABLE, "Источник поиска покупателей недоступен. Повторите поиск позднее."),
        status.SOURCE_INCONSISTENT: (ViewState.ERROR, "Источники поиска покупателей несогласованы. Карточка недоступна."),
    }
    state, message = messages[result.status]
    return CustomerCardResult(state, message, contact_search=result)


def validate_contact_match(result, *, raw_root):
    """Reuse backend freshness gates at handoff, without a contact query or scan.

    The backend currently exposes search + source_version, but no public
    is_current(). Reuse its own metadata readers, keeping that compatibility
    boundary here. Contact lookups themselves execute only in the Qt worker.
    """
    status = contact_backend.ContactSearchStatus
    try:
        root = Path(raw_root).resolve()
        with contact_backend._reader_lock(root):
            sources = contact_backend._sources(root)
            if result.source_version != sources.version:
                return contact_backend.ContactSearchResult(status.INDEX_NOT_READY)
            path = contact_backend.CustomerContactSearchService().index_path(raw_root=root)
            if not path.exists():
                return contact_backend.ContactSearchResult(status.INDEX_NOT_READY)
            with contact_backend._read_database(path) as connection:
                contact_backend._metadata(connection, sources)
        return result
    except contact_backend._IndexNotReady:
        return contact_backend.ContactSearchResult(status.INDEX_NOT_READY)
    except Exception as exc:
        return contact_backend.ContactSearchResult(status.SOURCE_UNAVAILABLE if contact_backend._unavailable(exc)
                                                 else status.SOURCE_INCONSISTENT)


IDENTITY_MESSAGES = {
    CustomerModelIdentityStatus.MATCHED_EXACT: "Найден единственный профиль в модели BPR-MF.",
    CustomerModelIdentityStatus.MATCHED_VIA_MERGE: "Найден единственный исторический профиль после объединения клиентов.",
    CustomerModelIdentityStatus.AMBIGUOUS: "У покупателя несколько исторических профилей в модели. Рекомендации недоступны.",
    CustomerModelIdentityStatus.NOT_IN_MODEL: "Покупатель отсутствует в BPR-MF. Персональные рекомендации недоступны.",
    CustomerModelIdentityStatus.INVALID_ID: "Введите корректный MindboxID без пробелов по краям.",
    CustomerModelIdentityStatus.SOURCE_UNAVAILABLE: "Источники сопоставления покупателей недоступны. Повторите поиск позднее.",
    CustomerModelIdentityStatus.SOURCE_INCONSISTENT: "Источники сопоставления покупателей несовместимы. Рекомендации недоступны.",
}


def _version(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


class CustomerCardService:
    """Owned by a single worker; backend caches and catalog are reused there."""

    def __init__(self, *, bridge=None, inference=None, profile_service=None, model_dir=PROJECT_ROOT / "model",
                 raw_root=INPUT_DATA_DIR / "MindboxRaw", catalog_path=INPUT_DATA_DIR / "nomenclature.csv"):
        self.bridge = bridge if bridge is not None else CustomerIdentityBridge()
        self.inference = inference if inference is not None else RawBprInferenceService()
        self.profile_service = profile_service if profile_service is not None else CustomerProfileService()
        self.profile_observer = None
        self.model_dir, self.raw_root, self.catalog_path = Path(model_dir), Path(raw_root), Path(catalog_path)
        self._catalog_key = None
        self._catalog = {}

    def _references(self):
        key = _version(self.catalog_path)
        if self._catalog_key == key:
            return self._catalog
        values = {}
        with self.catalog_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="|")
            if not reader.fieldnames or "КодНоменклатуры" not in reader.fieldnames:
                raise ValueError("Invalid display catalog")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Malformed display catalog row")
                code = row["КодНоменклатуры"].strip()
                if code:
                    values[code] = ((row.get("НазваниеНаСайте") or "").strip() or (row.get("Номенклатура") or "").strip(),
                                    (row.get("ТитульнаяФотография") or "").strip())
        if _version(self.catalog_path) != key:
            raise ValueError("Display catalog changed during reading")
        self._catalog_key, self._catalog = key, values
        return values

    def is_current(self, result: CustomerCardResult) -> bool:
        return (result.identity is not None and result.identity.is_matched
                and result.model_generation == result.identity.model_generation
                and self.bridge.is_current(result.identity, model_dir=self.model_dir, raw_root=self.raw_root))

    def validate_result(self, result: CustomerCardResult) -> CustomerCardResult:
        """Validate each source independently; an error never erases the other block."""
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
        if result.state in (ViewState.READY, ViewState.EMPTY):
            try:
                current = self.is_current(result)
            except Exception:
                current = False
            if not current:
                result = replace(result, state=ViewState.STALE, products=(),
                                 message="Источники рекомендаций изменились. Повторите поиск.")
        profile = result.profile
        if profile is not None and profile.status in (ProfileStatus.FOUND, ProfileStatus.NOT_FOUND):
            profile = self.validate_profile(profile)
            result = replace(result, profile=profile)
            if (profile.status is ProfileStatus.FOUND and result.identity is not None
                  and result.identity.canonical_customer_id is not None
                  and profile.canonical_customer_id != result.identity.canonical_customer_id):
                result = replace(result, state=ViewState.STALE, products=(),
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
        try:
            raw = self.inference.get_recommendations(identity.model_user_id, k=10, model_dir=self.model_dir)
        except Exception:
            return CustomerCardResult(ViewState.ERROR, "Не удалось загрузить рекомендации из модели BPR-MF.", identity)
        if cancelled():
            return CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        if raw.model_generation != identity.model_generation:
            return CustomerCardResult(ViewState.STALE, "Модель изменилась во время поиска. Повторите запрос.", identity)
        if raw.status is not RawBprStatus.OK:
            return CustomerCardResult(ViewState.UNAVAILABLE, "Покупатель больше не найден в активной модели.", identity)
        try:
            metadata = self._references()
            reference_error = False
        except Exception:
            metadata, reference_error = {}, True
        products = tuple(ProductContent(
            rec, *metadata.get(rec.model_item_code, ("", "")),
            "Справочник товаров недоступен" if reference_error else (
                "Нет в текущем каталоге" if rec.model_item_code not in metadata else ""),
        ) for rec in raw.recommendations)
        result = CustomerCardResult(ViewState.READY if products else ViewState.EMPTY,
                                    "Оригинальные рекомендации BPR-MF без дополнительной фильтрации.",
                                    identity, raw.model_generation, products)
        if cancelled():
            return CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        if not self.is_current(result):
            return CustomerCardResult(ViewState.STALE, "Модель или объединения покупателей изменились. Повторите поиск.", identity)
        return result
