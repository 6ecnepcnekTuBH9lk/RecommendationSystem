"""Vertical customer card. Widgets stay on GUI thread; one worker owns services."""

from collections import defaultdict, deque
from dataclasses import replace

from PyQt6.QtCore import QEvent, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtNetwork import QNetworkAccessManager
from PyQt6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLineEdit, QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget

from Application.customer_card_data import (
    CustomerCardResult, CustomerCardService, DisplayField, IDENTITY_MESSAGES, SectionContent, ViewState, contact_search_failure,
)
from Application.paths import ICONS_DIR
from Application.customer_profile_service import CustomerProfileResult, ProfileStatus
from Application.customer_contact_search import ContactSearchResult, ContactSearchStatus, ContactType
from Application.customer_profile_presentation import PROFILE_FIELDS, profile_content
from Application.photo.photo_processing import _invalidate_photo_targets, _set_photo_label
from Application.settings.set_status import schedule_status_reset, set_status_error, set_status_ok, set_status_processing
from Application.theme.layout_metrics import BLOCK_SPACING, CONTROL_SPACING, PAGE_MARGIN, SECTION_SPACING
from .customer_card_widgets import AdaptiveCardGrid, FieldCard, ProductCard, StateSection, card_frame, text_label
from .customer_card_worker import CustomerCardRequest, CustomerCardWorker


class CustomerResultsTab(QWidget):
    dispatch = pyqtSignal(object)
    result_applied = pyqtSignal(int)

    def __init__(self, window, *, service=None):
        super().__init__(window)
        self.setObjectName("customerResultsTab")
        self.window = window
        self.service = service if service is not None else CustomerCardService()
        self._thread = self._worker = self._active = self._pending = None
        self._request_id = 0
        self._closing = False
        self.result = None
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
        self._build_profile()
        self._build_kpis()
        self._build_recommendations()
        self._build_history()
        self._build_propensity()
        window.installEventFilter(self)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def _build_search(self):
        self.search_card, box = card_frame()
        self.search_card.setObjectName("customerSearch")
        self.search_card.setProperty("state", ViewState.INITIAL.value)
        box.setSpacing(BLOCK_SPACING)
        box.addWidget(text_label("Карточка покупателя", style="resultsTitle"))
        self.search_row = QHBoxLayout()
        self.search_row.setContentsMargins(0, 0, 0, 0)
        self.search_row.setSpacing(CONTROL_SPACING)
        self.search_type = QComboBox()
        self.search_type.setAccessibleName("Тип поиска покупателя")
        self.search_type.setMinimumWidth(0)
        self.search_type.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.search_type.setMinimumContentsLength(12)
        for title, kind in (("MindboxID", None), ("Телефон", ContactType.PHONE),
                            ("Email", ContactType.EMAIL), ("Дисконтная карта", ContactType.DISCOUNT_CARD)):
            self.search_type.addItem(title, kind)
        self.search_row.addWidget(self.search_type)
        self.customer_input = QLineEdit()
        self.customer_input.setMinimumWidth(0)
        self.customer_input.setPlaceholderText("Введите MindboxID покупателя")
        self.customer_input.returnPressed.connect(self.search)
        self.search_row.addWidget(self.customer_input, 1)
        self.search_button = QPushButton(QIcon(str(ICONS_DIR / "search.png")), "Найти")
        self.search_button.clicked.connect(self.search)
        self.cancel_button = QPushButton("Отменить поиск")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_search)
        self.export_button = QPushButton(QIcon(str(ICONS_DIR / "excel.png")), "Экспорт в Excel")
        self.export_button.setEnabled(False)
        self.export_button.setToolTip("Единый Excel-файл будет доступен после согласования его структуры.")
        for button in (self.search_button, self.cancel_button, self.export_button):
            self.search_row.addWidget(button)
        box.addLayout(self.search_row)
        self.search_hint = text_label("MindboxID без пробелов по краям.", style="statisticsInfo")
        box.addWidget(self.search_hint)
        self.search_status = text_label("Введите MindboxID, чтобы посмотреть оригинальные рекомендации BPR-MF.")
        box.addWidget(self.search_status)
        self.progress = QProgressBar()
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
            None: ("Введите MindboxID покупателя", "MindboxID без пробелов по краям."),
            ContactType.PHONE: ("Введите телефон: +7…", "Международный формат +… или 00…; также полный 7… (11 цифр). Номера с 8… и десятизначные не поддерживаются."),
            ContactType.EMAIL: ("Введите email покупателя", "Полный email. Регистр не учитывается; точки и +suffix сохраняются."),
            ContactType.DISCOUNT_CARD: ("Введите номер дисконтной карты", "Точный номер карты, включая ведущие нули."),
        }[kind]
        self.customer_input.setPlaceholderText(placeholder)
        self.search_hint.setText(hint)
        self.identity_status.setText("Покупатель не выбран.")
        self._display_state(ViewState.INITIAL, "Введите значение для выбранного типа поиска.")

    def _build_profile(self):
        self.profile = StateSection("Профиль покупателя", SectionContent(
            ViewState.INITIAL, "Введите MindboxID покупателя."),
            name="customerProfile")
        card, box = card_frame()
        box.setSpacing(BLOCK_SPACING)
        self.profile_title = text_label("Покупатель не выбран", style="resultsPlaceholderValue")
        self.canonical_id = text_label("Canonical MindboxID ещё не определён")
        self.model_id = text_label("ID в модели BPR-MF ещё не определён")
        for label in (self.canonical_id, self.model_id):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.identity_status = text_label("Введите MindboxID для проверки доступности рекомендаций.", style="statisticsInfo")
        for label in (self.profile_title, self.canonical_id, self.model_id, self.identity_status):
            box.addWidget(label)
        self.profile_fields = AdaptiveCardGrid(minimum_card_width=200, maximum_columns=4)
        self.profile_fields.set_cards([FieldCard(DisplayField(title, state=ViewState.INITIAL), framed=False)
                                      for title in PROFILE_FIELDS])
        box.addWidget(self.profile_fields)
        self.profile.box.addWidget(card)
        self.page_layout.addWidget(self.profile)

    def _build_kpis(self):
        self.kpis = StateSection("Показатели активности", SectionContent(
            ViewState.NOT_CONNECTED, "Показатели появятся после подключения истории и согласования финансовых правил."),
            name="customerKpis")
        self.kpi_grid = AdaptiveCardGrid()
        self.kpi_grid.set_cards([FieldCard(DisplayField(title), numeric=True) for title in (
            "Заказы", "Приобретённые единицы", "Сумма покупок", "Средний чек", "Просмотры", "Избранное",
            "Дней с последней покупки", "Уникальные просмотренные товары")])
        self.kpis.box.addWidget(self.kpi_grid)
        self.page_layout.addWidget(self.kpis)

    def _build_recommendations(self):
        self.recommendations = StateSection("Рекомендации BPR-MF", SectionContent(
            ViewState.INITIAL, "После поиска здесь появятся исходные рекомендации обученной модели."), name="customerRecommendations")
        self.topk_row = QHBoxLayout()
        self.topk_row.setContentsMargins(0, 0, 0, 0)
        self.topk_row.setSpacing(CONTROL_SPACING)
        self.topk_row.addStretch(1)
        from PyQt6.QtWidgets import QLabel
        self.topk_row.addWidget(QLabel("Показывать"))
        self.topk = QComboBox()
        for count in (1, 3, 5, 10):
            self.topk.addItem(f"Топ-{count}", count)
        self.topk.setCurrentIndex(3)
        self.topk.currentIndexChanged.connect(self._show_topk)
        self.topk_row.addWidget(self.topk)
        self.recommendations.box.insertLayout(1, self.topk_row)
        self.product_grid = AdaptiveCardGrid(minimum_card_width=250, maximum_columns=5)
        self.recommendations.box.addWidget(self.product_grid)
        self.page_layout.addWidget(self.recommendations)

    def _build_history(self):
        self.history = StateSection("История взаимодействий", SectionContent(
            ViewState.NOT_CONNECTED, "История взаимодействий будет доступна после подключения канонических данных Mindbox."),
            name="customerHistory")
        card, box = card_frame()
        self.history_row = QHBoxLayout()
        self.history_row.setContentsMargins(0, 0, 0, 0)
        self.history_row.setSpacing(CONTROL_SPACING)
        self.history_filter = QComboBox()
        self.history_filter.addItems(["Все события", "Покупки", "Просмотры", "Избранное"])
        self.history_view = QComboBox()
        self.history_view.addItems(["Хронологическая лента", "Таблица"])
        for control in (self.history_filter, self.history_view):
            control.setEnabled(False)
            control.setToolTip("Источник истории ещё не подключён.")
            self.history_row.addWidget(control)
        box.addLayout(self.history_row)
        box.addWidget(text_label("События не загружены. Здесь будет хронологическая лента покупок, просмотров и избранного."))
        self.history.box.addWidget(card)
        self.page_layout.addWidget(self.history)

    def _build_propensity(self):
        self.propensity = StateSection("Вероятность покупки в следующие 30 дней", SectionContent(
            ViewState.UNAVAILABLE, "Прогноз пока недоступен. Для расчёта нужна отдельная модель вероятности покупки."),
            name="customerPropensity")
        fields = AdaptiveCardGrid()
        fields.set_cards([FieldCard(field) for field in (
            DisplayField("Горизонт прогноза", "30 дней", ViewState.READY),
            DisplayField("Дата расчёта", state=ViewState.UNAVAILABLE),
            DisplayField("Сегмент покупателя", state=ViewState.UNAVAILABLE),
            DisplayField("Версия прогнозной модели", state=ViewState.UNAVAILABLE))])
        self.propensity.box.addWidget(fields)
        self.page_layout.addWidget(self.propensity)

    def _ensure_worker(self):
        if self._thread is not None:
            return
        # Application ownership prevents QThread destruction with an active
        # worker even if a window is deleted directly instead of being closed.
        thread = QThread(QApplication.instance())
        worker = CustomerCardWorker(self.service)
        worker.moveToThread(thread)
        self.dispatch.connect(worker.run, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._finished)
        worker.profile_ready.connect(self._profile_received)
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

    def _clear_result(self):
        self.result = None
        _invalidate_photo_targets(self)
        self.product_grid.set_cards([])
        self.profile_title.setText("Покупатель не выбран")
        self.canonical_id.setText("Canonical MindboxID ещё не определён")
        self.model_id.setText("ID в модели BPR-MF ещё не определён")
        self.profile.set_content(SectionContent(ViewState.INITIAL, "Введите MindboxID покупателя."))
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
            self.identity_status.setText(message)
            self._render_profile(CustomerProfileResult(ProfileStatus.INVALID_ID))
            self._display_state(ViewState.ERROR, message)
            return
        request = CustomerCardRequest(self._request_id, value, contact_type=kind)
        self.identity_status.setText("Проверяем доступность рекомендаций для покупателя…")
        self.profile.set_content(SectionContent(ViewState.LOADING, "Загрузка профиля покупателя…"))
        self.profile_title.setText("Загрузка профиля…")
        self.profile_fields.set_cards([FieldCard(DisplayField(title, state=ViewState.LOADING), framed=False)
                                      for title in PROFILE_FIELDS])
        self._display_state(ViewState.LOADING, "Ищем покупателя и загружаем исходные рекомендации…")
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
        self.search_status.setText(message)
        self.recommendations.set_content(SectionContent(state, message))
        loading = state is ViewState.LOADING
        self.progress.setVisible(loading)
        self.cancel_button.setEnabled(loading)
        self._global_status(state, message)

    def _profile_received(self, request_id, profile):
        if self._closing or request_id != self._request_id:
            return
        try:
            if isinstance(profile, CustomerCardResult):
                delivery = self.service.validate_result(profile)
                if delivery.profile is None:
                    self.profile.set_content(SectionContent(delivery.state, delivery.message))
                    self.profile_fields.set_cards([])
                    self.profile_title.setText("Покупатель не выбран")
                    self.canonical_id.setText("Canonical MindboxID ещё не определён")
                    self.model_id.setText("ID в модели BPR-MF ещё не определён")
                    return
                profile = delivery.profile
            profile = self.service.validate_profile(profile)
        except Exception:
            profile = CustomerProfileResult(ProfileStatus.STALE)
        self._render_profile(profile)

    def _render_profile(self, profile):
        content = profile_content(profile)
        self.profile.set_content(content)
        self.profile_fields.set_cards([FieldCard(field, framed=False) for field in content.fields])
        if profile.status is ProfileStatus.FOUND:
            self.profile_title.setText(profile.profile.full_name or "Не указано")
            self.canonical_id.setText(f"Canonical MindboxID: {profile.canonical_customer_id}")
        else:
            self.profile_title.setText(content.message)

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
            return
        # Metadata-only validation of independent sources at the GUI handoff.
        try:
            result = self.service.validate_result(result)
        except Exception:
            if result.contact_search is not None:
                result = contact_search_failure(ContactSearchResult(ContactSearchStatus.SOURCE_INCONSISTENT))
            else:
                result = replace(result, state=ViewState.STALE, products=(), profile=CustomerProfileResult(ProfileStatus.STALE),
                                 message="Источники изменились до показа данных. Повторите поиск.")
        self.apply_result(result)
        self.result_applied.emit(request_id)

    def apply_result(self, result: CustomerCardResult):
        """Presentation only; worker has checked snapshot freshness before emission."""
        if result.contact_search is not None and result.contact_search.status is not ContactSearchStatus.MATCHED:
            result = contact_search_failure(result.contact_search)
        elif result.state in (ViewState.READY, ViewState.EMPTY) and (
                result.identity is None or not result.identity.is_matched
                or result.model_generation != result.identity.model_generation):
            result = replace(result, state=ViewState.STALE, products=(), message="Источники рекомендаций изменились. Повторите поиск.")
        self._clear_result()
        self.result = result
        identity = result.identity
        if identity is not None:
            self.identity_status.setText(IDENTITY_MESSAGES[identity.status])
            if identity.canonical_customer_id is not None:
                self.canonical_id.setText(f"Canonical MindboxID: {identity.canonical_customer_id}")
            if identity.model_user_id is not None:
                self.model_id.setText(f"ID в модели BPR-MF: {identity.model_user_id}")
        else:
            self.identity_status.setText(result.message)
        if result.profile is not None:
            self._render_profile(result.profile)
        elif result.contact_search is not None and result.contact_search.status is not ContactSearchStatus.MATCHED:
            self.profile.set_content(SectionContent(result.state, result.message))
            self.profile_fields.set_cards([])
            self.profile_title.setText("Покупатель не выбран")
        self._display_state(result.state, result.message)
        if result.state is ViewState.READY:
            cards = [ProductCard(product) for product in result.products]
            self.product_grid.set_cards(cards)
            self._show_topk()
            for card in cards:
                _set_photo_label(self, card.photo, card.content.photo_url, self._img_gen)

    def _show_topk(self):
        self.product_grid.show_prefix(self.topk.currentData())

    def cancel_search(self):
        self._request_id += 1
        if self._active is not None:
            self._active.cancelled.set()
        self._pending = None
        self._clear_result()
        self.identity_status.setText("Поиск отменён.")
        self._render_profile(CustomerProfileResult(ProfileStatus.CANCELLED))
        self._display_state(ViewState.CANCELLED, "Поиск отменён. Можно начать новый запрос.")

    def shutdown(self):
        self._closing = True
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
        self.history_row.setDirection(QHBoxLayout.Direction.TopToBottom if self.width() < 550
                                      else QHBoxLayout.Direction.LeftToRight)
