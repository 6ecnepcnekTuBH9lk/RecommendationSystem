from Application.paths import ICONS_DIR
import sys
from Application.theme.SwitchTheme import ThemeSwitch
from collections import deque
from PyQt6.QtCore import Qt, QTimer
from collections import defaultdict
from PyQt6.QtNetwork import QNetworkAccessManager
from PyQt6.QtGui import QIcon, QPixmap, QGuiApplication, QCursor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
                             QSizePolicy, QLayout)

from Application.tabs.data_processing_tab import create_input_data_widgets_tab
from Application.tabs.data_loading_tab import create_data_loading_widgets_tab
from Application.tabs.train_model_tab import create_train_model_widgets_tab
from Application.tabs.create_results_tab import create_result_widgets_tab
from Application.settings.set_status import set_ready_status
from Application.theme.apply_theme import apply_app_theme


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Заголовок и иконка
        self.setWindowTitle("Рекомендательная система")
        self.setWindowIcon(QIcon(str(ICONS_DIR / "app_icon.png")))

        # Центральный виджет и основной layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        # Long tab forms must not impose their full sizeHint on the window.
        # Their deeper adaptation to small windows is a separate UI task.
        main_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        central_widget.setLayout(main_layout)

        # Таймер для обновления статуса
        self._status_reset_timer = QTimer(self)
        self._status_reset_timer.setSingleShot(True)
        self._status_reset_timer.timeout.connect(self.my_set_ready_status)

        # Менеджер сетевых запросов
        self.net = QNetworkAccessManager(self)

        # Объявление атрибутов
        self.heading_load_data = None
        self.combo_box_add_or_not = None
        self.btn_load = None
        self.heading_filters = None
        self.recs_table = None
        self.purchases_table = None
        self.label_recs = None
        self.label_123 = None
        self.train_log = None
        self.start_train = None
        self.btn_settings = None
        self.label_69 = None
        self.heading_enter_parameter = None
        self.prefix = None
        self.heading_analysis = None
        self.filter_summary = None
        self.btn_reset = None
        self.btn_apply = None
        self.btn_weather = None
        self._cities = []
        self._store_city_map = {}
        self._img_queue = deque()
        self._img_max_inflight = 3
        self._img_retry_max = 2
        self._img_retry_count = defaultdict(int)
        self._name_by_code = None
        self._photo_by_code = None
        self._img_cache = {}
        self._img_targets = {}
        self._img_inflight = set()
        self._img_gen = 0

        # Текст статуса
        self.status_label = QLabel("Готов к работе")
        self.status_label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)

        # Иконка статуса
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(30, 30)
        self.status_icon.setPixmap(QPixmap(str(ICONS_DIR / "success.png")).scaled(17, 17, Qt.AspectRatioMode.KeepAspectRatio,
                                                                        Qt.TransformationMode.SmoothTransformation))
        self.status_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Создаем вкладки для разных групп виджетов
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Получение согласованных наборов Mindbox
        create_data_loading_widgets_tab(self)
        # Legacy CSV: обработка входных данных
        create_input_data_widgets_tab(self)
        # Вкладка с обучением модели
        create_train_model_widgets_tab(self)
        # Вкладка с выгрузкой результатов
        create_result_widgets_tab(self)

        # Отступы на форме
        self.apply_static_widget_styles()

        # --- Кастомный нижний бар ---
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        bottom_bar.setSpacing(10)

        # Контейнер статуса
        status_wrap = QWidget()
        status_layout = QHBoxLayout(status_wrap)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(6)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_icon)
        status_layout.addStretch(1)

        # Свитч справа
        self.theme_switch = ThemeSwitch()
        self.theme_switch.themeChanged.connect(self.apply_theme)

        status_wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        bottom_bar.addWidget(status_wrap, 1)
        bottom_bar.addWidget(self.theme_switch, 0, Qt.AlignmentFlag.AlignRight)

        main_layout.addLayout(bottom_bar)

        self._restore_window_placement()

    def _restore_window_placement(self):
        screen = (
                QGuiApplication.screenAt(QCursor.pos())
                or QGuiApplication.primaryScreen()
        )

        available = (
            screen.availableGeometry()
            if screen is not None
            else None
        )

        # Reserve room for native window decorations on small screens.
        width = (
            max(1, available.width() - 32)
            if available is not None
            else 1920
        )
        height = (
            max(1, available.height() - 48)
            if available is not None
            else 1080
        )

        self.setMinimumSize(
            min(1280, width),
            min(900, height),
        )

        self.resize(
            max(
                self.minimumWidth(),
                min(1920, int(width * 0.9)),
            ),
            max(
                self.minimumHeight(),
                min(1080, int(height * 0.9)),
            ),
        )

        QTimer.singleShot(
            0,
            self.center_on_cursor_screen,
        )

    # ///////////////////////////////////////////ПОМОГАТОРЫ/////////////////////////////////////////////////////////////
    def my_set_ready_status(self):
        set_ready_status(self)

    # -------------------------------------------ЦЕНТРИРОВАТЬ ОКНО------------------------------------------------------
    def center_on_cursor_screen(self):

        try:
            # Экран под курсором
            screen = QGuiApplication.screenAt(QCursor.pos())
            if screen is None:
                screen = QGuiApplication.primaryScreen()
            if screen is None:
                return

            screen_geo = screen.availableGeometry()  # рабочая область (без панели задач)
            frame_geo = self.frameGeometry()  # геометрия окна с рамкой

            frame_geo.moveCenter(screen_geo.center())
            self.move(frame_geo.topLeft())
        except Exception:
            pass

    # -------------------------------------------ПЕРЕКЛЮЧАТЕЛЬ ТЕМЫ-----------------------------------------------------
    def apply_theme(self, is_dark: bool):

        app = QApplication.instance()
        if app is None:
            return

        # Если тема уже применена, ничего не делаем
        if getattr(self, "_current_is_dark", None) == is_dark:
            return

        self.setUpdatesEnabled(False)

        try:
            apply_app_theme(app, is_dark)
            self._current_is_dark = is_dark
        finally:
            self.setUpdatesEnabled(True)
            self.update()

    # -------------------------------------------СТАТИЧЕСКИЕ ПРАВКИ РАЗМЕРОВ--------------------------------------------
    def apply_static_widget_styles(self):

        self.btn_apply.setStyleSheet("""QPushButton { margin: 5px 0px 0px 0px; }""")
        self.btn_weather.setStyleSheet("""QPushButton { margin: 5px 0px 0px 0px; }""")
        self.btn_reset.setStyleSheet("""QPushButton { margin: 5px 0px 0px 0px; }""")
        self.filter_summary.setStyleSheet("""QLineEdit { margin: 5px 0px 0px 0px; }""")
        self.prefix.setStyleSheet("""padding: 0px 3px 0px 0px;""")
        self.btn_settings.setStyleSheet("""QPushButton { margin: 5px 0px 0px 0px; }""")
        self.start_train.setStyleSheet("""QPushButton { margin: 5px 0px 0px 0px; }""")
        self.purchases_table.setStyleSheet("""QTableWidget { margin: 0px 0px 10px 10px; }""")
        self.recs_table.setStyleSheet("""QTableWidget { margin: 0px 10px 10px 0px; }""")


# -----------------------------------------------MAIN-------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.apply_theme(True)
    window.show()

    QTimer.singleShot(0, lambda: print(f"[WINDOW SIZE] {window.width()} x {window.height()}"))

    sys.exit(app.exec())
