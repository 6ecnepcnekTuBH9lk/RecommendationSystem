"""One serial Qt worker; cancellation is cooperative, never terminate()."""

from dataclasses import dataclass, field, replace
from threading import Event

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from Application.customer_card_data import CustomerCardResult, ViewState, contact_search_failure
from Application.customer_contact_search import ContactSearchStatus, ContactType, search_customers_by_contact


@dataclass(frozen=True, slots=True, repr=False)
class CustomerCardRequest:
    request_id: int
    customer_id: str
    cancelled: Event = field(default_factory=Event)
    contact_type: ContactType | None = None


class CustomerCardWorker(QObject):
    finished = pyqtSignal(int, object)
    profile_ready = pyqtSignal(int, object)

    def __init__(self, service):
        super().__init__()
        self.service = service

    @pyqtSlot(object)
    def run(self, request):
        profile = None
        match = None
        def received(value):
            nonlocal profile
            profile = value
            if not request.cancelled.is_set():
                # Carry the contact snapshot with an early profile so the GUI
                # can reject a stale selection without waiting for Raw BPR.
                delivery = value if request.contact_type is None else CustomerCardResult(
                    ViewState.LOADING, "Загрузка карточки…", profile=value, contact_search=match)
                self.profile_ready.emit(request.request_id, delivery)
        self.service.profile_observer = received
        try:
            if request.cancelled.is_set():
                result = CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
            elif request.contact_type is not None:
                match = search_customers_by_contact(request.contact_type, request.customer_id, raw_root=self.service.raw_root)
                if request.cancelled.is_set():
                    result = CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
                elif match.status is not ContactSearchStatus.MATCHED:
                    result = contact_search_failure(match)
                else:
                    result = self.service.load(match.canonical_customer_id, request.cancelled.is_set)
                    result = replace(result, contact_search=match)
            else:
                result = self.service.load(request.customer_id, request.cancelled.is_set)
            result = self.service.validate_result(result)
        except Exception:
            result = CustomerCardResult(ViewState.ERROR, "Не удалось получить данные. Повторите поиск.",
                                        profile=profile, contact_search=match)
        finally:
            self.service.profile_observer = None
        if request.cancelled.is_set():
            result = CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        self.finished.emit(request.request_id, result)
