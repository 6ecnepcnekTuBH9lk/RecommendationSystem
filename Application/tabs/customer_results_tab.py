"""Vertical customer card. Widgets stay on GUI thread; one worker owns services."""

from collections import defaultdict, deque
from dataclasses import replace
import logging

from PyQt6.QtCore import QEvent, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtNetwork import QNetworkAccessManager
from PyQt6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget

from Application.customer_card_data import (
    CustomerCardResult, CustomerCardService, DisplayField, SectionContent, ViewState, contact_search_failure,
)
from Application.paths import ICONS_DIR
from Application.customer_profile_service import CustomerProfileResult, ProfileStatus
from Application.customer_contact_search import ContactSearchStatus, ContactType
from Application.customer_profile_presentation import PROFILE_FIELDS, profile_content
from Application.photo.photo_processing import _invalidate_photo_targets, _set_photo_label
from Application.settings.set_status import schedule_status_reset, set_status_error, set_status_ok, set_status_processing
from Application.theme.layout_metrics import BLOCK_SPACING, CONTROL_SPACING, PAGE_MARGIN, SECTION_SPACING
from .customer_card_widgets import AdaptiveCardGrid, CustomerBusyProgressBar, FieldCard, ProductCard, ProfileFieldGrid, StateSection, card_frame, text_label
from .customer_card_worker import CustomerCardRequest, CustomerCardWorker
from .customer_history_widgets import CustomerHistorySection
from .hybrid_calculation_widgets import HybridCalculationSection


logger = logging.getLogger(__name__)


class CustomerResultsTab(QWidget):
    dispatch = pyqtSignal(object)
    history_dispatch = pyqtSignal(object)
    validation_dispatch = pyqtSignal(object, object)
    result_applied = pyqtSignal(int)

    def __init__(self, window, *, service=None, history_lookup=None):
        super().__init__(window)
        self.setObjectName("customerResultsTab")
        self.window = window
        if service is None:
            from Application.hybrid_recommendations import HybridRecommendationService
            service = CustomerCardService(hybrid=HybridRecommendationService())
        self.service = service
        self._history_lookup = history_lookup
        self._deferred_history = None
        self._thread = self._worker = self._active = self._pending = None
        self._request_id = 0
        self._closing = False
        self.result = None
        self._profile_notice = ""
        self.net = getattr(window, "net", None) or QNetworkAccessManager(self)
        self._img_gen = 0
        self._img_queue = deque()
        self._img_inflight = set()
        self._img_targets, self._img_cache, self._img_replies = {}, {}, {}
        self._img_retry_count = defaultdict(int)
        self._img_retry_max, self._img_max_inflight, self._img_cache_limit = 2, 3, 40

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(self.scroll)
        self.page = QWidget()
        self.page.setMinimumWidth(0)
        self.page_layout = QVBoxLayout(self.page)
        self.page_layout.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        self.page_layout.setSpacing(SECTION_SPACING)
        self.page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.page)
        self._build_search()
        self._search_message = self.search_status.text()
        self._build_profile()
        self._build_kpis()
        self._build_propensity()
        self._build_recommendations()
        self._build_history()
        self.page_layout.addWidget(self.calculations)
        QApplication.instance().installEventFilter(self)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def _build_search(self):
        self.search_card, box = card_frame()
        self.search_card.setObjectName("customerSearch")
        self.search_card.setProperty("state", ViewState.INITIAL.value)
        box.setSpacing(BLOCK_SPACING)
        box.addWidget(text_label("Профиль клиента", style="resultsTitle"))
        self.search_row = QHBoxLayout()
        self.search_row.setContentsMargins(0, 0, 0, 0)
        self.search_row.setSpacing(CONTROL_SPACING)
        self.search_type = QComboBox()
        self.search_type.setAccessibleName("Тип поиска клиента")
        self.search_type.setMinimumWidth(0)
        self.search_type.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.search_type.setMinimumContentsLength(12)
        for title, kind in (("MindboxID", None), ("Телефон", ContactType.PHONE),
                            ("Email", ContactType.EMAIL), ("Дисконтная карта", ContactType.DISCOUNT_CARD)):
            self.search_type.addItem(title, kind)
        self.search_row.addWidget(self.search_type)
        self.customer_input = QLineEdit()
        self.customer_input.setMinimumWidth(0)
        self.customer_input.setPlaceholderText("Введите MindboxID клиента...")
        self.customer_input.returnPressed.connect(self.search)
        self.search_row.addWidget(self.customer_input, 1)
        self.search_button = QPushButton(QIcon(str(ICONS_DIR / "search.png")), " Найти")
        self.search_button.clicked.connect(self.search)
        self.cancel_button = QPushButton(QIcon(str(ICONS_DIR / "failure.png")), " Отменить поиск")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_search)
        self.export_button = QPushButton(QIcon(str(ICONS_DIR / "export.png")), " Экспорт рекомендаций")
        self.export_button.setEnabled(False)
        self.export_button.setToolTip("Единый Excel-файл будет доступен после согласования его структуры.")
        for button in (self.search_button, self.cancel_button, self.export_button):
            self.search_row.addWidget(button)
        box.addLayout(self.search_row)
        self.search_hint = text_label("Вводить значение без пробелов по краям", style="statisticsInfo")
        box.addWidget(self.search_hint)
        self.search_status = text_label("Введите значение для выбранного типа поиска, чтобы посмотреть оригинальные рекомендации нейросетевой модели")
        box.addWidget(self.search_status)
        self.progress = CustomerBusyProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        box.addWidget(self.progress)
        self.page_layout.addWidget(self.search_card)
        self.search_type.currentIndexChanged.connect(self._search_type_changed)

    def _search_type_changed(self):
        if self._closing:
            return
        self._request_id += 1
        if self._active is not None:
            self._active.cancelled.set()
        self._pending = None
        self._clear_result()
        kind = self.search_type.currentData()
        placeholder, hint = {
            None: ("Введите MindboxID клиента...", "Вводить значение без пробелов по краям"),
            ContactType.PHONE: ("Введите телефон клиента: +7...", "Не поддерживаются десятизначные номера, и начинающиеся с 8..."),
            ContactType.EMAIL: ("Введите email клиента...", "Введите полный email. При поиске регистр не учитывается"),
            ContactType.DISCOUNT_CARD: ("Введите номер дисконтной карты клиента...", "Введите точный номер карты, включая ведущие нули"),
        }[kind]
        self.customer_input.setPlaceholderText(placeholder)
        self.search_hint.setText(hint)
        self._display_state(ViewState.INITIAL, "Введите значение для выбранного типа поиска, чтобы посмотреть оригинальные рекомендации нейросетевой модели")

    def _build_profile(self):
        self.profile = StateSection("Профиль клиента", SectionContent(
            ViewState.INITIAL, "Введите MindboxID клиента."),
            name="customerProfile")
        card, box = card_frame()
        box.setSpacing(BLOCK_SPACING)
        self.profile.heading.hide()
        self.profile.message.hide()
        self.profile_fields = ProfileFieldGrid()
        self.profile_fields.set_cards([FieldCard(DisplayField(title, state=ViewState.INITIAL), framed=False)
                                      for title in PROFILE_FIELDS])
        box.addWidget(self.profile_fields)
        self.profile.box.addWidget(card)
        self.page_layout.addWidget(self.profile)

    def _build_kpis(self):
        self.kpis = StateSection("Показатели активности", SectionContent(
            ViewState.NOT_CONNECTED, ""),
            name="customerKpis")
        self.kpis.message.hide()
        self.kpi_grid = AdaptiveCardGrid()
        self.kpi_grid.set_cards([FieldCard(DisplayField(title), numeric=True) for title in (
            "Заказы", "Приобретённые единицы", "Сумма покупок", "Средний чек", "Просмотры", "Избранное",
            "Дней с последней покупки", "Уникальные просмотренные товары")])
        self.kpis.box.addWidget(self.kpi_grid)
        self.page_layout.addWidget(self.kpis)

    def _build_recommendations(self):
        self.recommendations = StateSection("Рекомендации нейросетевой модели BPR-MF", SectionContent(
            ViewState.INITIAL, "После поиска здесь появятся исходные рекомендации обученной модели."), name="customerRecommendations")
        self.recommendations.message.hide()
        self.raw_grid = AdaptiveCardGrid(minimum_card_width=250, maximum_columns=5)
        self.recommendations.box.addWidget(self.raw_grid)
        self.page_layout.addWidget(self.recommendations)
        self.calculations = HybridCalculationSection()
        self.hybrid_recommendations = StateSection("Рекомендации гибридной модели", SectionContent(
            ViewState.INITIAL, "После поиска появятся актуальные товары."), name="customerHybridRecommendations")
        self.product_grid = AdaptiveCardGrid(minimum_card_width=250, maximum_columns=5)
        self.hybrid_recommendations.box.addWidget(self.product_grid)
        self.page_layout.addWidget(self.hybrid_recommendations)

    def _build_history(self):
        self.history = CustomerHistorySection(photo_owner=self)
        self.history_filter = self.history.category
        self.history.dispatch.connect(self._load_history)
        self.page_layout.addWidget(self.history)

    def _load_history(self, request):
        # Keep history behind the complete card handoff, including its final
        # background freshness check, so all source reads remain serial.
        if self._active is not None:
            self._deferred_history = request
            return
        self._ensure_worker()
        self.history_dispatch.emit(request)

    def _flush_history(self):
        if self._active is not None or self._closing or self._deferred_history is None:
            return
        request, self._deferred_history = self._deferred_history, None
        if request.cancelled.is_set():
            # Acknowledge cancellation so the section can start its latest pending request.
            self.history.receive(request, None)
        else:
            self._load_history(request)

    def _build_propensity(self):
        self.propensity = StateSection("Вероятность покупки в следующие 30 дней", SectionContent(
            ViewState.UNAVAILABLE, "Прогноз пока недоступен. Для расчёта нужна отдельная модель вероятности покупки."),
            name="customerPropensity")
        self.propensity.message.hide()
        fields = AdaptiveCardGrid()
        fields.set_cards([FieldCard(field) for field in (
            DisplayField("Горизонт прогноза", "30 дней", ViewState.READY),
            DisplayField("Дата расчёта", state=ViewState.UNAVAILABLE),
            DisplayField("Сегмент клиента", state=ViewState.UNAVAILABLE),
            DisplayField("Версия прогнозной модели", state=ViewState.UNAVAILABLE))])
        self.propensity.box.addWidget(fields)
        self.page_layout.addWidget(self.propensity)

    def _ensure_worker(self):
        if self._thread is not None:
            return
        # Application ownership prevents QThread destruction with an active
        # worker even if a window is deleted directly instead of being closed.
        thread = QThread(QApplication.instance())
        worker = CustomerCardWorker(self.service, history_lookup=self._history_lookup)
        worker.moveToThread(thread)
        self.dispatch.connect(worker.run, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._queue_validation)
        self.validation_dispatch.connect(worker.validate_delivery, Qt.ConnectionType.QueuedConnection)
        worker.validated.connect(self._finished)
        worker.profile_ready.connect(self._profile_received)
        self.history_dispatch.connect(worker.load_history, Qt.ConnectionType.QueuedConnection)
        worker.history_ready.connect(self.history.receive)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        self.destroyed.connect(thread.quit)
        QApplication.instance().aboutToQuit.connect(thread.quit)
        QApplication.instance().aboutToQuit.connect(thread.wait)
        self._thread, self._worker = thread, worker
        thread.start()

    def _global_status(self, state, message):
        if not hasattr(self.window, "status_label") or not hasattr(self.window, "status_icon"):
            return
        timer = getattr(self.window, "_status_reset_timer", None)
        if timer is not None:
            timer.stop()
        setter = set_status_processing if state is ViewState.LOADING else (
            set_status_error if state in (ViewState.ERROR, ViewState.STALE, ViewState.UNAVAILABLE) else set_status_ok)
        setter(self.window, message)
        if state is not ViewState.LOADING:
            schedule_status_reset(self.window, 5)

    def _clear_result(self, *, clear_history=True):
        if clear_history:
            self.history.set_customer(None)
        self.result = None
        _invalidate_photo_targets(self)
        self.raw_grid.set_cards([])
        self.product_grid.set_cards([])
        self.calculations.clear(ViewState.INITIAL, "После поиска появятся расчёты.")
        self._profile_notice = ""
        self.profile.set_content(SectionContent(ViewState.INITIAL, "Введите MindboxID клиента."))
        self.profile_fields.set_cards([FieldCard(DisplayField(title, state=ViewState.INITIAL), framed=False)
                                      for title in PROFILE_FIELDS])

    def search(self):
        if self._closing:
            return
        self._request_id += 1
        if self._active is not None:
            self._active.cancelled.set()
        self._pending = None
        self._clear_result()
        value = self.customer_input.text()
        kind = self.search_type.currentData()
        if kind is None and (not value or value != value.strip()):
            message = "Введите MindboxID." if not value else "Введите MindboxID без пробелов по краям."
            self._render_profile(CustomerProfileResult(ProfileStatus.INVALID_ID))
            self._display_state(ViewState.ERROR, message)
            return
        request = CustomerCardRequest(self._request_id, value, contact_type=kind)
        self.profile.set_content(SectionContent(ViewState.LOADING, "Загрузка профиля клиента…"))
        self.profile_fields.set_cards([FieldCard(DisplayField(title, state=ViewState.LOADING), framed=False)
                                      for title in PROFILE_FIELDS])
        self._display_state(ViewState.LOADING, "Ищем клиента и загружаем исходные рекомендации…")
        if self._active is not None:
            self._pending = request
        else:
            self._start(request)

    def _start(self, request):
        self._ensure_worker()
        self._active = request
        self.dispatch.emit(request)

    def _display_state(self, state, message):
        self.search_card.setProperty("state", state.value)
        self._search_message = message
        self._update_search_message()
        self.recommendations.set_content(SectionContent(state, message))
        self.hybrid_recommendations.set_content(SectionContent(state, message))
        self.calculations.clear(state, message)
        self.recommendations.message.setVisible(state in (
            ViewState.ERROR, ViewState.UNAVAILABLE, ViewState.STALE, ViewState.EMPTY, ViewState.CANCELLED))
        if state is ViewState.EMPTY:
            self.recommendations.message.setText("Рекомендаций нет.")
        loading = state is ViewState.LOADING
        self.progress.setVisible(loading)
        self.cancel_button.setEnabled(loading)
        self._global_status(state, self.search_status.text())

    def _update_search_message(self):
        self.search_status.setText(" ".join(filter(None, (self._search_message, self._profile_notice))))

    def _set_profile_content(self, content):
        self.profile.set_content(content)
        placeholders = {
            ViewState.EMPTY: "Нет данных профиля", ViewState.ERROR: "Профиль недоступен",
            ViewState.UNAVAILABLE: "Профиль недоступен", ViewState.STALE: "Данные устарели",
            ViewState.CANCELLED: "Загрузка отменена",
        }
        fields = content.fields or tuple(DisplayField(title, state=content.state,
                                                      message=placeholders.get(content.state)) for title in PROFILE_FIELDS)
        self.profile_fields.set_cards([FieldCard(field, framed=False) for field in fields])

    def _set_profile_notice(self, message):
        self._profile_notice = message
        self._update_search_message()

    def _profile_received(self, request_id, profile):
        if self._closing or request_id != self._request_id:
            return
        if isinstance(profile, CustomerCardResult):
            if profile.profile is None:
                self._set_profile_content(SectionContent(profile.state, profile.message))
                self._set_profile_notice(profile.message)
                return
            profile = profile.profile
        self._render_profile(profile)
        if profile.status in (ProfileStatus.FOUND, ProfileStatus.NOT_FOUND):
            self.history.set_customer(profile.canonical_customer_id)

    def _render_profile(self, profile):
        content = profile_content(profile)
        self._set_profile_content(content)
        self._set_profile_notice(content.message if profile.status is not ProfileStatus.FOUND else "")

    def _queue_validation(self, request_id, result):
        if (self._active is not None and request_id == self._active.request_id
                and request_id == self._request_id and self._pending is None and not self._closing):
            logger.debug("customer_delivery stage=queued request=%d", request_id)
            self.validation_dispatch.emit(self._active, result)
        else:
            self._finished(request_id, result)

    def _finished(self, request_id, result):
        if self._active is None or request_id != self._active.request_id:
            return
        self._active = None
        if self._closing:
            return
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self._start(pending)
            return
        if request_id != self._request_id:
            self._flush_history()
            return
        # Freshness has been rechecked by validate_delivery after the first GUI
        # handoff. This slot only applies the current request; no source I/O.
        logger.debug("customer_delivery stage=applied request=%d state=%s", request_id, result.state.value)
        self.apply_result(result)
        self._flush_history()
        self.result_applied.emit(request_id)

    def apply_result(self, result: CustomerCardResult):
        """Presentation only; worker has checked snapshot freshness before emission."""
        if result.contact_search is not None and result.contact_search.status is not ContactSearchStatus.MATCHED:
            result = contact_search_failure(result.contact_search)
        elif (result.state in (ViewState.READY, ViewState.EMPTY) or result.raw_state in (ViewState.READY, ViewState.EMPTY)) and (
                result.identity is None or not result.identity.is_matched
                or result.model_generation != result.identity.model_generation):
            result = replace(result, state=ViewState.STALE, products=(), raw_products=(), raw_state=ViewState.STALE,
                             message="Источники рекомендаций изменились. Повторите поиск.")
        self._clear_result(clear_history=False)
        self.result = result
        identity = result.identity
        if result.profile is not None:
            self._render_profile(result.profile)
        elif result.contact_search is not None and result.contact_search.status is not ContactSearchStatus.MATCHED:
            self._set_profile_content(SectionContent(result.state, result.message))
        self._display_state(result.state, result.message)
        customer_id = None
        if result.profile is not None and result.profile.status in (ProfileStatus.FOUND, ProfileStatus.NOT_FOUND):
            customer_id = result.profile.canonical_customer_id
        elif result.contact_search is not None and result.contact_search.status is ContactSearchStatus.MATCHED:
            customer_id = result.contact_search.canonical_customer_id
        elif identity is not None and result.state is not ViewState.STALE:
            customer_id = identity.canonical_customer_id
        self.history.set_customer(customer_id)
        raw_state = result.raw_state if result.raw_state is not None else result.state
        raw_products = result.raw_products if result.raw_state is not None else (
            result.products if result.hybrid is None else ())
        raw_message = result.raw_message or ("Исходных рекомендаций нет." if raw_state is ViewState.EMPTY else result.message)
        self.recommendations.set_content(SectionContent(raw_state, raw_message))
        self.recommendations.message.setVisible(raw_state is not ViewState.READY)
        if raw_state is ViewState.READY:
            self.raw_grid.set_cards([ProductCard(p) for p in raw_products[:10]])
        if result.hybrid is not None:
            state, message = result.state, result.message
        else:
            state, message = (ViewState.UNAVAILABLE, "Гибридные рекомендации не рассчитаны.") if result.state in (
                ViewState.READY, ViewState.EMPTY) else (result.state, result.message)
        self.hybrid_recommendations.set_content(SectionContent(state, message))
        self.hybrid_recommendations.message.setVisible(state is not ViewState.READY)
        if state is ViewState.READY:
            self.product_grid.set_cards([ProductCard(p) for p in result.products[:10]])
            self.calculations.set_items(result.hybrid.items[:10])
        else:
            self.calculations.clear(state, "Подходящих актуальных товаров нет." if state is ViewState.EMPTY else message)
        for card in (*self.raw_grid.cards, *self.product_grid.cards):
            _set_photo_label(self, card.photo, card.content.photo_url, self._img_gen)

    def cancel_search(self):
        self._request_id += 1
        if self._active is not None:
            self._active.cancelled.set()
        self._pending = None
        self._clear_result()
        self._render_profile(CustomerProfileResult(ProfileStatus.CANCELLED))
        self._display_state(ViewState.CANCELLED, "Поиск отменён. Можно начать новый запрос.")

    def shutdown(self):
        self._closing = True
        self._deferred_history = None
        self.history.shutdown()
        if self._active is not None:
            self._active.cancelled.set()
        self._pending = None
        _invalidate_photo_targets(self)
        if self._thread is not None:
            self._thread.quit()

    def _thread_finished(self):
        thread = self._thread
        self._thread = self._worker = None
        if thread is not None:
            thread.deleteLater()
        if self._closing:
            self.window.close()

    def eventFilter(self, watched, event):
        if event.type() is QEvent.Type.ToolTip and isinstance(watched, QWidget) and (watched is self or self.isAncestorOf(watched)):
            return True
        if watched is self.window and event.type() is QEvent.Type.Close:
            self.shutdown()
            if self._thread is not None and self._thread.isRunning():
                event.ignore()
                return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        direction = QHBoxLayout.Direction.TopToBottom if self.width() < 760 else QHBoxLayout.Direction.LeftToRight
        self.search_row.setDirection(direction)
