"""One serial Qt worker; cancellation is cooperative, never terminate()."""

from dataclasses import dataclass, field, replace
from threading import Event
import logging

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from Application.customer_card_data import CustomerCardResult, ViewState, contact_search_failure
from Application.customer_contact_search import ContactSearchResult, ContactSearchStatus, ContactType, search_customers_by_contact
from Application.customer_profile_service import CustomerProfileResult, ProfileStatus
from Application.customer_activity_service import ActivityStatus, CustomerActivity
from Application.customer_history_service import HistoryPage, HistoryStatus, get_customer_history
from Application.customer_history_records import HistoryKind
from Application.customer_history_presentation import present_history


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, repr=False)
class CustomerCardRequest:
    request_id: int
    customer_id: str
    cancelled: Event = field(default_factory=Event)
    contact_type: ContactType | None = None


@dataclass(frozen=True, slots=True, repr=False)
class CustomerHistoryRequest:
    request_id: int
    customer_id: str
    kind: HistoryKind
    cursor: str | None = None
    cancelled: Event = field(default_factory=Event)


class CustomerCardWorker(QObject):
    finished = pyqtSignal(int, object)
    validated = pyqtSignal(int, object)
    profile_ready = pyqtSignal(int, object)
    history_ready = pyqtSignal(object, object)

    def __init__(self, service, *, history_lookup=None):
        super().__init__()
        self.service = service
        self.history_lookup = history_lookup or get_customer_history

    @pyqtSlot(object, object)
    def validate_delivery(self, request, result):
        """Recheck after the GUI queue handoff, serially with all other reads."""
        if request.cancelled.is_set():
            result = CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        else:
            try:
                result = self.service.validate_result(result)
                logger.debug("customer_delivery stage=validated request=%d state=%s", request.request_id, result.state.value)
            except Exception as exc:
                logger.warning("customer_delivery stage=validation error=%s", type(exc).__name__)
                if result.contact_search is not None:
                    result = contact_search_failure(ContactSearchResult(ContactSearchStatus.SOURCE_INCONSISTENT))
                else:
                    result = replace(result, state=ViewState.STALE, products=(), raw_products=(), raw_state=ViewState.STALE,
                                     profile=CustomerProfileResult(ProfileStatus.STALE),
                                     activity=CustomerActivity(ActivityStatus.STALE) if result.activity is not None else None,
                                     message="Источники изменились до показа данных. Повторите поиск.")
        self.validated.emit(request.request_id, result)

    @pyqtSlot(object)
    def load_history(self, request):
        # One bounded page, on the existing card thread. Never builds or verifies
        # the index. Cancellation discards delivery; an in-flight read may finish.
        page = None
        if not request.cancelled.is_set():
            try:
                page = self.history_lookup(request.customer_id, kinds=(request.kind,), limit=50,
                                           cursor=request.cursor, raw_root=self.service.raw_root)
                if request.cancelled.is_set():
                    self.history_ready.emit(request, None)
                    return
                catalog = {}
                if page.events and not request.cancelled.is_set():
                    try:
                        catalog = self.service.history_product_references(page)
                        if not isinstance(catalog, dict):
                            catalog = {}
                    except Exception:
                        pass  # Optional catalog must never hide source history.
                if not request.cancelled.is_set():
                    page = present_history(page, request.kind, catalog)
            except Exception:
                page = HistoryPage(HistoryStatus.SOURCE_INCONSISTENT)
        self.history_ready.emit(request, None if request.cancelled.is_set() else page)

    @pyqtSlot(object)
    def run(self, request):
        profile = None
        match = None
        def received(value):
            nonlocal profile
            profile = value
            if not request.cancelled.is_set():
                # Validate here before continuing BPR; the GUI must never open
                # SQLite or acquire competing canonical locks on this signal.
                delivery = value if request.contact_type is None else CustomerCardResult(
                    ViewState.LOADING, "Загрузка карточки…", profile=value, contact_search=match)
                try:
                    if isinstance(delivery, CustomerCardResult):
                        delivery = self.service.validate_result(delivery)
                    else:
                        delivery = self.service.validate_profile(delivery)
                except Exception as exc:
                    logger.warning("customer_delivery stage=profile_validation error=%s", type(exc).__name__)
                    delivery = CustomerProfileResult(ProfileStatus.STALE)
                logger.debug("customer_delivery stage=profile request=%d", request.request_id)
                self.profile_ready.emit(request.request_id, delivery)
        self.service.profile_observer = received
        try:
            if request.cancelled.is_set():
                result = CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
            elif request.contact_type is not None:
                logger.debug("customer_contact stage=lookup request=%d", request.request_id)
                match = search_customers_by_contact(request.contact_type, request.customer_id, raw_root=self.service.raw_root)
                logger.debug("customer_contact stage=lookup_result request=%d status=%s", request.request_id, match.status.value)
                if request.cancelled.is_set():
                    result = CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
                elif match.status is not ContactSearchStatus.MATCHED:
                    result = contact_search_failure(match)
                else:
                    result = self.service.load(match.canonical_customer_id, request.cancelled.is_set)
                    result = replace(result, contact_search=match)
            else:
                result = self.service.load(request.customer_id, request.cancelled.is_set)
            if not request.cancelled.is_set():
                # Legacy injected card services need not implement analytics.
                # Failure of this optional stage must not replace other sections.
                load_activity = getattr(self.service, "load_activity", None)
                if callable(load_activity):
                    try:
                        activity_result = load_activity(result, request.cancelled.is_set)
                        if not isinstance(activity_result, CustomerCardResult):
                            raise TypeError("Invalid activity delivery")
                        result = activity_result
                    except Exception as exc:
                        logger.warning("customer_activity stage=worker error=%s", type(exc).__name__)
                        result = replace(result, activity=CustomerActivity(ActivityStatus.ERROR))
            result = self.service.validate_result(result)
        except Exception as exc:
            logger.warning("customer_contact stage=worker error=%s", type(exc).__name__)
            result = CustomerCardResult(ViewState.ERROR, "Не удалось получить данные. Повторите поиск.",
                                        profile=profile, contact_search=match)
        finally:
            self.service.profile_observer = None
        if request.cancelled.is_set():
            result = CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        self.finished.emit(request.request_id, result)
