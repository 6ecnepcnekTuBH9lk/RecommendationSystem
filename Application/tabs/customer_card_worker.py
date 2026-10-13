"""One serial Qt worker; cancellation is cooperative, never terminate()."""

from dataclasses import dataclass, field
from threading import Event

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from Application.customer_card_data import CustomerCardResult, ViewState


@dataclass(frozen=True, slots=True, repr=False)
class CustomerCardRequest:
    request_id: int
    customer_id: str
    cancelled: Event = field(default_factory=Event)


class CustomerCardWorker(QObject):
    finished = pyqtSignal(int, object)
    profile_ready = pyqtSignal(int, object)

    def __init__(self, service):
        super().__init__()
        self.service = service

    @pyqtSlot(object)
    def run(self, request):
        profile = None
        def received(value):
            nonlocal profile
            profile = value
            if not request.cancelled.is_set():
                self.profile_ready.emit(request.request_id, value)
        self.service.profile_observer = received
        try:
            result = self.service.load(request.customer_id, request.cancelled.is_set)
            result = self.service.validate_result(result)
        except Exception:
            result = CustomerCardResult(ViewState.ERROR, "Не удалось получить данные. Повторите поиск.", profile=profile)
        finally:
            self.service.profile_observer = None
        if request.cancelled.is_set():
            result = CustomerCardResult(ViewState.CANCELLED, "Поиск отменён.")
        self.finished.emit(request.request_id, result)
