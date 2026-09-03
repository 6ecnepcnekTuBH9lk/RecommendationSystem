import os
import json
import sys
import tempfile
import chardet
import pandas as pd
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QListWidget, QFileDialog, QFrame,
    QAbstractItemView, QStackedWidget, QButtonGroup,
    QTableWidget, QHeaderView, QTableWidgetItem, QApplication,
    QSpinBox)

from Application.settings.settings_and_filter import (save_order_filter_settings, order_filters_settings_path,
                                                      dataset_paths, update_filter_summary, get_selected_list_values,
                                                      clear_layout)

from Application.settings.set_status import (set_status_processing, schedule_status_reset, show_custom_message,
                                             set_status_error, set_status_ok)

from Application.files.files_processing import (process_orders_file, process_views_file, process_favorites_file,
                                                process_categories_file, process_nomenclature_file,
                                                process_coordinates_file, generate_weather_for_saved_coordinates)


def _refresh_filter_references_on_startup(aboba):
    refresh_functions = (
        refresh_kind_values_from_loaded_files,
        refresh_season_values_from_nomenclature_file,
        refresh_export_kind_values_from_nomenclature_file,
    )

    for refresh_function in refresh_functions:
        try:
            refresh_function(aboba)
        except Exception as error:
            set_status_error(
                aboba,
                f"Не удалось обновить справочник фильтров: {error}",
            )


# -------------------------------------------ВКЛАДКА ОБРАБОТКА ДАТАСЕТА-------------------------------------------------
def create_input_data_widgets_tab(aboba):
    tab = QWidget()

    # Главный горизонтальный сплит: левая и правая части
    root = QHBoxLayout(tab)
    root.setContentsMargins(0, 0, 0, 0)  # Слева и справа по 10 от всего
    root.setSpacing(0)  # Расстояние между частями и разделителем

    # Левая панель
    left_wrap = QWidget()
    left_layout = QVBoxLayout(left_wrap)
    left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Все прижимаем к верху

    # Правая панель
    right_wrap = QWidget()
    right_layout = QVBoxLayout(right_wrap)
    right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Все прижимаем к верху

    # Вертикальный разделитель
    separator = QFrame()
    separator.setObjectName("vSeparator")
    separator.setFixedWidth(1)  # Толщина
    separator.setFrameShape(QFrame.Shape.NoFrame)  # Без рамки

    # Добавляем на форму, 1 - распределение ширины
    root.addWidget(left_wrap, 1)
    root.addWidget(separator)
    root.addWidget(right_wrap, 1)

    row_layout = QHBoxLayout()

    # ----- Левая часть -----
    # Заголовок "Загрузка данных"
    aboba.heading_load_data = QLabel("Загрузка данных")
    aboba.heading_load_data.setSizePolicy(aboba.heading_load_data.sizePolicy().Policy.Fixed,  # Фиксируем размер
                                          aboba.heading_load_data.sizePolicy().Policy.Fixed)  # по ширине и высоте
    aboba.heading_load_data.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aboba.heading_load_data.setStyleSheet("""
        QLabel {
                    background-color: #FAFAFA;
                    padding: 7px 65px; 
                    border-radius: 10px;
                    border: 1px solid #C8C8C8;
                    margin: 10px 0px;
                }
    """)
    left_layout.addWidget(aboba.heading_load_data, alignment=Qt.AlignmentFlag.AlignHCenter)

    # Выпадающий список с типом данных
    aboba.combo_box_types = QComboBox()
    aboba.combo_box_types.addItems(["Заказы клиентов из Mindbox",
                                    "Просмотры товаров и категорий из Mindbox",
                                    "Добавление товаров в избранное из Mindbox",
                                    "Номенклатура из 1С", "Категории сайта из 1С", "Координаты городов и погода"
                                    ])
    aboba.combo_box_types.setStyleSheet("""QComboBox { margin: 0px 0px 5px 0px; }""")

    # Выпадающий список с вариантом загрузки
    aboba.combo_box_add_or_not = QComboBox()
    aboba.combo_box_add_or_not.addItems(["Добавить новый / Обновить существующий",
                                         "Добавить данные к существующему"])

    # Левая колонка: два выпадающих списка
    left_col = QVBoxLayout()
    left_col.addWidget(aboba.combo_box_types)
    left_col.addWidget(aboba.combo_box_add_or_not)

    # Кнопка "Загрузить файл"
    aboba.btn_load = QPushButton(QIcon("Картинки/ЗагрузитьФайл.png"), " Загрузить файл")
    aboba.btn_load.setIconSize(QSize(17, 17))
    aboba.btn_load.clicked.connect(lambda: load_csv_file(aboba))

    # Правая колонка: кнопка
    right_col = QVBoxLayout()
    right_col.addWidget(aboba.btn_load)

    # Собираем строку: слева списки, справа кнопка
    row_layout.addLayout(left_col, stretch=5)
    row_layout.addLayout(right_col, stretch=3)

    left_layout.addLayout(row_layout)

    # Статус загрузки файлов
    aboba.status_files_layout = QHBoxLayout()
    aboba.status_files_layout.setContentsMargins(0, 0, 0, 0)
    aboba.status_files_layout.setSpacing(0)

    aboba.status_files_container = QWidget()
    aboba.status_files_container.setLayout(aboba.status_files_layout)

    left_layout.addWidget(aboba.status_files_container)

    # Заголовок "Установка отбора"
    aboba.heading_filters = QLabel("Настройки и установка отбора")
    aboba.heading_filters.setSizePolicy(aboba.heading_load_data.sizePolicy().Policy.Fixed,  # Фиксируем размер
                                        aboba.heading_load_data.sizePolicy().Policy.Fixed)  # по ширине и высоте
    aboba.heading_filters.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aboba.heading_filters.setStyleSheet("""
        QLabel {
            background-color: #FAFAFA;
            padding: 7px 65px;
            border-radius: 10px;
            border: 1px solid #C8C8C8;
            margin: 0px 0px 5px 0px;
        }
    """)
    left_layout.addWidget(aboba.heading_filters, alignment=Qt.AlignmentFlag.AlignHCenter)

    # Панель фильтров
    filters_wrap = QWidget()

    filters_layout = QVBoxLayout(filters_wrap)
    filters_layout.setContentsMargins(0, 0, 0, 0)
    filters_layout.setSpacing(10)

    # -------------------- ПЕРИОД --------------------
    aboba.filter_date_from = QLineEdit()
    aboba.filter_date_to = QLineEdit()
    aboba.filter_date_from.setInputMask("99.99.9999; ")
    aboba.filter_date_to.setInputMask("99.99.9999; ")

    period_row = QHBoxLayout()
    period_row.setContentsMargins(0, 0, 0, 0)
    period_row.setSpacing(10)

    lbl_period = QLabel("Период (дд.мм.гггг):")
    period_row.addWidget(lbl_period, 0, Qt.AlignmentFlag.AlignHCenter)
    period_row.addWidget(aboba.filter_date_from, 1)
    period_row.addWidget(aboba.filter_date_to, 1)

    filters_layout.addLayout(period_row)

    # -------------------- ВИД НОМЕНКЛАТУРЫ --------------------
    aboba.kind_mode = QComboBox()
    aboba.kind_mode.addItems(["В группе", "Не в группе"])

    aboba.filter_kind = QListWidget()
    aboba.filter_kind.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

    kind_wrap = QWidget()
    kind_wrap_l = QHBoxLayout(kind_wrap)
    kind_wrap_l.setContentsMargins(0, 0, 0, 0)
    kind_wrap_l.setSpacing(10)
    kind_wrap_l.addWidget(aboba.kind_mode)
    kind_wrap_l.addWidget(aboba.filter_kind, 1)

    kind_row = QHBoxLayout()
    kind_row.setContentsMargins(0, 0, 0, 0)
    kind_row.setSpacing(10)

    lbl_kind = QLabel("Вид номенклатуры:")
    kind_row.addWidget(lbl_kind, 0, Qt.AlignmentFlag.AlignHCenter)
    kind_row.addWidget(kind_wrap, 1)

    filters_layout.addLayout(kind_row)

    # -------------------- ВИДЫ НОМЕНКЛАТУРЫ В РЕКОМЕНДАЦИЯХ --------------------
    aboba.export_kind_filter = QListWidget()
    aboba.export_kind_filter.setSelectionMode(
        QAbstractItemView.SelectionMode.MultiSelection
    )
    aboba.export_kind_filter.setMaximumHeight(100)
    aboba.export_kind_filter.setToolTip(
        "Выберите виды номенклатуры, которые должны попасть в итоговые "
        "рекомендации.\n"
        "Можно выбрать несколько значений.\n"
        "Если ничего не выбрано, ограничение не применяется."
    )

    export_kind_row = QHBoxLayout()
    export_kind_row.setContentsMargins(0, 0, 0, 0)
    export_kind_row.setSpacing(10)

    lbl_export_kind = QLabel("Виды в рекомендациях:")
    export_kind_row.addWidget(
        lbl_export_kind,
        0,
        Qt.AlignmentFlag.AlignVCenter
    )
    export_kind_row.addWidget(aboba.export_kind_filter, 1)

    filters_layout.addLayout(export_kind_row)

    # -------------------- МАГАЗИН (СКЛАД) --------------------
    aboba.store_mode = QComboBox()
    aboba.store_mode.addItems(["В группе", "Не в группе"])

    aboba.filter_store = QListWidget()
    aboba.filter_store.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

    store_wrap = QWidget()
    store_wrap_l = QHBoxLayout(store_wrap)
    store_wrap_l.setContentsMargins(0, 0, 0, 0)
    store_wrap_l.setSpacing(10)
    store_wrap_l.addWidget(aboba.store_mode)
    store_wrap_l.addWidget(aboba.filter_store, 1)

    store_row = QHBoxLayout()
    store_row.setContentsMargins(0, 0, 0, 0)
    store_row.setSpacing(10)

    lbl_store = QLabel("Магазин (склад):    ")
    store_row.addWidget(lbl_store, 0, Qt.AlignmentFlag.AlignHCenter)
    store_row.addWidget(store_wrap, 1)

    filters_layout.addLayout(store_row)

    # -------------------- АКТУАЛЬНЫЙ СЕЗОН НОСКИ --------------------
    aboba.filter_season = QListWidget()
    aboba.filter_season.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

    season_row = QHBoxLayout()
    season_row.setContentsMargins(0, 0, 0, 0)
    season_row.setSpacing(10)

    lbl_season = QLabel("Актуальные сезоны:")
    season_row.addWidget(lbl_season, 0, Qt.AlignmentFlag.AlignHCenter)
    season_row.addWidget(aboba.filter_season, 1)

    filters_layout.addLayout(season_row)

    # -------------------- КОЛИЧЕСТВО КЛИЕНТОВ В ВЫГРУЗКЕ --------------------
    aboba.max_export_users_input = QSpinBox()

    # 0 означает выгрузку всех подходящих клиентов
    aboba.max_export_users_input.setRange(0, 1_000_000)
    aboba.max_export_users_input.setSingleStep(100)
    aboba.max_export_users_input.setValue(1000)
    aboba.max_export_users_input.setSpecialValueText("Все клиенты")
    aboba.max_export_users_input.setSuffix(" клиентов")
    aboba.max_export_users_input.setToolTip(
        "Количество наиболее лояльных клиентов, для которых будут "
        "сформированы рекомендации.\n"
        "Значение 0 означает выгрузку всех подходящих клиентов."
    )

    export_users_row = QHBoxLayout()
    export_users_row.setContentsMargins(0, 0, 0, 0)
    export_users_row.setSpacing(10)

    lbl_export_users = QLabel("Клиентов в выгрузке:")
    export_users_row.addWidget(
        lbl_export_users,
        0,
        Qt.AlignmentFlag.AlignVCenter
    )
    export_users_row.addWidget(aboba.max_export_users_input, 1)

    filters_layout.addLayout(export_users_row)

    # -------------------- ТАБЛИЦА СКЛАД -> ГОРОД --------------------
    aboba.store_city_table = QTableWidget()
    aboba.store_city_table.setColumnCount(2)
    aboba.store_city_table.setHorizontalHeaderLabels(["Магазин (склад)", "Город"])
    aboba.store_city_table.verticalHeader().setVisible(False)
    aboba.store_city_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    aboba.store_city_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    hdr = aboba.store_city_table.horizontalHeader()
    hdr.setStretchLastSection(False)
    hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

    filters_layout.addWidget(aboba.store_city_table)

    btns = QHBoxLayout()
    btns.setSpacing(10)

    # Кнопка "Применить фильтр"
    aboba.btn_apply = QPushButton(QIcon("Картинки/Фильтр.png"), " Применить")
    aboba.btn_apply.setIconSize(QSize(17, 17))
    aboba.btn_apply.setStyleSheet("""QPushButton { margin: 5px 0px 0px 0px; }""")
    aboba.btn_apply.clicked.connect(lambda: save_and_apply_filters(aboba))

    # Кнопка "Сбросить фильтр"
    aboba.btn_reset = QPushButton(QIcon("Картинки/Корзина.png"), " Сбросить")
    aboba.btn_reset.setIconSize(QSize(17, 17))
    aboba.btn_reset.setStyleSheet("""QPushButton { margin: 5px 0px 0px 0px; }""")
    aboba.btn_reset.clicked.connect(lambda: reset_order_filters(aboba))

    aboba.btn_weather = QPushButton(QIcon("Картинки/Солнце.png"), " Загрузить погоду")
    aboba.btn_weather.setIconSize(QSize(17, 17))
    aboba.btn_weather.setStyleSheet("""QPushButton { margin: 5px 0px 0px 0px; }""")
    aboba.btn_weather.clicked.connect(lambda: _maybe_update_weather(aboba))

    btns.addWidget(aboba.btn_apply)
    btns.addWidget(aboba.btn_reset)
    btns.addWidget(aboba.btn_weather)

    # Текущий отбор строкой
    aboba.filter_summary = QLineEdit()
    aboba.filter_summary.setReadOnly(True)
    aboba.filter_summary.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    aboba.filter_summary.setCursor(Qt.CursorShape.IBeamCursor)
    aboba.filter_summary.setPlaceholderText("Отбор не установлен")
    aboba.filter_summary.setStyleSheet("""QLineEdit { margin: 5px 0px 0px 0px; }""")

    left_layout.addWidget(filters_wrap)
    left_layout.addWidget(aboba.filter_summary)
    left_layout.addLayout(btns)

    # ----- Правая часть -----
    # Заголовок "Статистика и анализ"
    aboba.heading_analysis = QLabel("Статистика и анализ")
    aboba.heading_analysis.setSizePolicy(aboba.heading_load_data.sizePolicy().Policy.Fixed,  # Фиксируем размер
                                         aboba.heading_load_data.sizePolicy().Policy.Fixed)  # по ширине и высоте
    aboba.heading_analysis.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aboba.heading_analysis.setStyleSheet("""
                QLabel {
                    background-color: #FAFAFA;
                    padding: 7px 65px; 
                    border-radius: 10px;
                    border: 1px solid #C8C8C8;
                    margin: 10px 0px 10px 0px;
                }
            """)
    right_layout.addWidget(aboba.heading_analysis, alignment=Qt.AlignmentFlag.AlignHCenter)

    # Переключатели (Заказы / Просмотры / Избранное)
    switch_row = QHBoxLayout()
    switch_row.setSpacing(10)

    aboba.btn_show_orders = QPushButton(QIcon("Картинки/Заказы.png"), " Заказы")
    aboba.btn_show_orders.setIconSize(QSize(17, 17))

    aboba.btn_show_views = QPushButton(QIcon("Картинки/Просмотры.png"), " Просмотры")
    aboba.btn_show_views.setIconSize(QSize(17, 17))

    aboba.btn_show_favs = QPushButton(QIcon("Картинки/Избранное.png"), " Избранное")
    aboba.btn_show_favs.setIconSize(QSize(17, 17))

    # Делаем кнопки переключателями
    for b in (aboba.btn_show_orders, aboba.btn_show_views, aboba.btn_show_favs):
        b.setCheckable(True)

    # Заказы активные
    aboba.btn_show_orders.setChecked(True)

    # Формируем группу переключателей
    aboba.stats_btn_group = QButtonGroup(aboba)
    aboba.stats_btn_group.setExclusive(True)
    aboba.stats_btn_group.addButton(aboba.btn_show_orders, 0)
    aboba.stats_btn_group.addButton(aboba.btn_show_views, 1)
    aboba.stats_btn_group.addButton(aboba.btn_show_favs, 2)

    switch_row.addStretch(1)
    switch_row.addWidget(aboba.btn_show_orders)
    switch_row.addWidget(aboba.btn_show_views)
    switch_row.addWidget(aboba.btn_show_favs)
    switch_row.addStretch(1)

    right_layout.addLayout(switch_row)

    # Страницы статистики
    aboba.stats_stack = QStackedWidget()
    right_layout.addWidget(aboba.stats_stack, 1)

    # Страница "Заказы"
    orders_page = QWidget()
    orders_l = QVBoxLayout(orders_page)
    orders_l.setContentsMargins(0, 0, 0, 0)

    aboba.order_full_output_layout = QVBoxLayout()
    orders_l.addLayout(aboba.order_full_output_layout)

    aboba.order_full_stats_label = QLabel("")

    aboba.order_full_stats_label.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse |
        Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    aboba.order_full_stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aboba.order_full_output_layout.addWidget(aboba.order_full_stats_label)

    aboba.stats_stack.addWidget(orders_page)

    # Страница "Просмотры"
    views_page = QWidget()
    views_l = QVBoxLayout(views_page)
    views_l.setContentsMargins(0, 0, 0, 0)

    aboba.views_full_output_layout = QVBoxLayout()
    views_l.addLayout(aboba.views_full_output_layout)

    aboba.views_full_stats_label = QLabel("")
    aboba.views_full_stats_label.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse |
        Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    aboba.views_full_stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aboba.views_full_output_layout.addWidget(aboba.views_full_stats_label)

    aboba.stats_stack.addWidget(views_page)

    # Страница "Избранное"
    favs_page = QWidget()
    favs_l = QVBoxLayout(favs_page)
    favs_l.setContentsMargins(0, 0, 0, 0)

    aboba.favorites_full_output_layout = QVBoxLayout()
    favs_l.addLayout(aboba.favorites_full_output_layout)

    aboba.favorites_full_stats_label = QLabel("")
    aboba.favorites_full_stats_label.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse |
        Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    aboba.favorites_full_stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aboba.favorites_full_output_layout.addWidget(aboba.favorites_full_stats_label)

    aboba.stats_stack.addWidget(favs_page)

    # Переключение страниц
    aboba.btn_show_orders.clicked.connect(lambda: aboba.stats_stack.setCurrentIndex(0))
    aboba.btn_show_views.clicked.connect(lambda: aboba.stats_stack.setCurrentIndex(1))
    aboba.btn_show_favs.clicked.connect(lambda: aboba.stats_stack.setCurrentIndex(2))

    # Обновляем статус загрузки
    update_file_status(aboba)

    # Восстанавливаем настройки
    load_order_filter_settings(aboba)

    # Настройка доступа к полям отбора
    update_filter_controls_availability(aboba)

    # Обновляем справочники фильтров
    _refresh_filter_references_on_startup(aboba)

    # Формируем текстовую строку с настройками
    update_filter_summary(aboba)

    # Формируем статистику заказов
    analyze_orders_full_dataset(aboba)

    # Формируем статистику просмотров
    analyze_views_full_dataset(aboba)

    # Формируем статистику добавлений
    analyze_favorites_full_dataset(aboba)

    # Название вкладки
    aboba.tabs.addTab(tab, "Обработка датасета")


# ///////////////////////////////////////////НАСТРОЙКИ//////////////////////////////////////////////////////////////////
# -------------------------------------------ВЫВОД ЗАГЛУШЕК---------------------------------------------------------
def vyvod_zaglyschek(text, icon, main_layout, stats_label):
    # ---- Горизонтальный контейнер для текста + иконки ----
    h_layout = QHBoxLayout()
    h_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # Текст
    stats_label.setText(text)
    stats_label.setStyleSheet("font-size: 16px;")

    # Иконка (пример)
    icon_label = QLabel()
    icon_label.setPixmap(QPixmap(icon).scaled(35, 35,
                                              Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))

    # Добавляем элементы
    h_layout.addWidget(stats_label)
    h_layout.addSpacing(0)  # расстояние между текстом и иконкой
    h_layout.addWidget(icon_label)

    main_layout.addLayout(h_layout)


# -------------------------------------------ОБНОВЛЕНИЕ СТАТУСА ЗАГРУЗКИ------------------------------------------------
def update_file_status(aboba):
    input_dir = os.path.join(os.getcwd(), "ВходныеДанные")

    files = {
        "Заказы": "Заказы.csv",
        "Просмотры": "Просмотры.csv",
        "Избранное": "Избранное.csv",
        "Номенклатура": "Номенклатура.csv",
        "Категории": "КатегорииСайта.csv",
        "Координаты": "КоординатыГородов.csv"
    }

    # Очистка старых виджетов
    while aboba.status_files_layout.count():
        item = aboba.status_files_layout.takeAt(0)
        w = item.widget()
        if w:
            w.deleteLater()

    # Префикс
    aboba.prefix = QLabel("Статус загрузки:")
    aboba.prefix.setStyleSheet("""padding: 0px 3px 0px 0px;""")
    aboba.status_files_layout.addWidget(aboba.prefix, 0, Qt.AlignmentFlag.AlignLeft)

    # Основная часть
    right_widget = QWidget()
    right_layout = QHBoxLayout()
    right_widget.setLayout(right_layout)

    ok_path = "Картинки/Успех.png"
    fail_path = "Картинки/Неудача.png"

    items = list(files.items())

    for i, (title, filename) in enumerate(items):
        exists = os.path.exists(os.path.join(input_dir, filename))

        block = QWidget()
        block_l = QHBoxLayout()
        block_l.setSpacing(4)  # расстояние между словом и иконкой
        block.setLayout(block_l)

        text_lbl = QLabel(title)

        icon_lbl = QLabel()
        pix = QPixmap(ok_path if exists else fail_path)
        icon_lbl.setPixmap(pix.scaled(
            17, 17,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignBottom)

        block_l.addWidget(text_lbl)
        block_l.addWidget(icon_lbl)

        # Добавляем блок в правую часть
        right_layout.addWidget(block, 0, Qt.AlignmentFlag.AlignVCenter)

        # Stretch между блоками, чтобы растягивались по ширине
        if i != len(items) - 1:
            right_layout.addStretch(1)

    # добавляем правую часть с растягивающим коэффициентом
    aboba.status_files_layout.addWidget(right_widget, 1)


# -------------------------------------------ВОССТАНОВЛЕНИЕ НАСТРОЕК----------------------------------------------------
def load_order_filter_settings(aboba):
    path = order_filters_settings_path()

    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # PREPARE: сначала преобразуем все значения, не изменяя UI/state.
        date_from = (
            str(data.get("date_from", ""))
            if hasattr(aboba, "filter_date_from")
            else None
        )
        date_to = (
            str(data.get("date_to", ""))
            if hasattr(aboba, "filter_date_to")
            else None
        )

        max_export_users = None
        if hasattr(aboba, "max_export_users_input"):
            try:
                max_export_users = int(data.get("max_export_users", 1000))
            except (TypeError, ValueError):
                max_export_users = 1000

        store_mode_index = None
        if hasattr(aboba, "store_mode"):
            store_mode_text = str(data.get("store_mode", "В группе"))
            index = aboba.store_mode.findText(store_mode_text)
            if index >= 0:
                store_mode_index = index

        kind_mode_index = None
        if hasattr(aboba, "kind_mode"):
            kind_mode_text = str(data.get("kind_mode", "В группе"))
            index = aboba.kind_mode.findText(kind_mode_text)
            if index >= 0:
                kind_mode_index = index

        pending_store_selection = list(data.get("stores_selected", []))
        pending_kind_selection = list(data.get("kinds_selected", []))
        pending_season_selection = list(data.get("seasons_selected", []))
        pending_export_kind_selection = list(
            data.get("export_kinds_selected", [])
        )
        pending_store_city_map = dict(data.get("store_city_map", {}) or {})
        store_city_map = dict(pending_store_city_map)

        # COMMIT: все settings уже успешно прочитаны и преобразованы.
        if hasattr(aboba, "filter_date_from"):
            aboba.filter_date_from.setText(date_from)
        if hasattr(aboba, "filter_date_to"):
            aboba.filter_date_to.setText(date_to)

        # Восстанавливаем количество клиентов в выгрузке
        if hasattr(aboba, "max_export_users_input"):
            aboba.max_export_users_input.setValue(max_export_users)

        if store_mode_index is not None:
            aboba.store_mode.setCurrentIndex(store_mode_index)

        if kind_mode_index is not None:
            aboba.kind_mode.setCurrentIndex(kind_mode_index)

        aboba._pending_store_selection = pending_store_selection

        aboba._pending_kind_selection = pending_kind_selection

        aboba._pending_season_selection = pending_season_selection

        # Отдельный отбор видов номенклатуры для итоговой выгрузки
        aboba._pending_export_kind_selection = pending_export_kind_selection

        aboba._pending_store_city_map = pending_store_city_map
        aboba._store_city_map = store_city_map

        if hasattr(aboba, "store_city_table"):
            load_cities_from_coordinates_file(aboba)
            refresh_store_city_table(aboba)

    except Exception:
        set_status_error(aboba, "Настройки фильтров повреждены")
        schedule_status_reset(aboba, 5)


# -------------------------------------------КНОПКА "ПРИМЕНИТЬ НАСТРОЙКИ"-----------------------------------------------
def apply_filters_all_stats(aboba) -> bool:

    # пересчитываем все страницы статистики по текущим фильтрам
    results = (
        analyze_orders_full_dataset(aboba),
        analyze_views_full_dataset(aboba),
        analyze_favorites_full_dataset(aboba),
    )
    return all(results)


def save_and_apply_filters(aboba):

    set_status_processing(aboba, "Сохранение настроек...")
    QApplication.processEvents()
    ok = save_order_filter_settings(aboba)

    if ok is False:
        return

    set_status_processing(aboba, "Применение фильтров...")
    QApplication.processEvents()

    stats_ok = apply_filters_all_stats(aboba)

    if not stats_ok:
        set_status_error(aboba, "Не удалось применить фильтры")
        QApplication.processEvents()
        schedule_status_reset(aboba, 5)
        return

    set_status_ok(aboba, "Фильтры применены")
    QApplication.processEvents()
    schedule_status_reset(aboba, 5)


def _masked_date_is_empty(le) -> bool:
    if le is None:
        return True
    t = le.text()
    if t is None:
        return True
    return t.replace(" ", "").replace(".", "") == ""


def _maybe_update_weather(aboba):
    coords_path = os.path.join(os.getcwd(), "ВходныеДанные", "КоординатыГородов.csv")
    if not os.path.isfile(coords_path):
        return  # координаты не загружены — погоду не трогаем

    # период не задан — просто не обновляем погоду
    if _masked_date_is_empty(getattr(aboba, "filter_date_from", None)) or _masked_date_is_empty(getattr(aboba, "filter_date_to", None)):
        return

    df_text = aboba.filter_date_from.text().strip()
    dt_text = aboba.filter_date_to.text().strip()

    d_from = pd.to_datetime(df_text, errors="coerce", dayfirst=True)
    d_to = pd.to_datetime(dt_text, errors="coerce", dayfirst=True)

    if pd.isna(d_from) or pd.isna(d_to) or d_from > d_to:
        show_custom_message(
            aboba,
            title="Ошибка",
            text="Период заполнен некорректно. Погода не обновлена.",
            image_path="Картинки/Неудача.png",
        )
        return

    start_date = d_from.strftime("%Y-%m-%d")
    end_date = d_to.strftime("%Y-%m-%d")

    try:
        generate_weather_for_saved_coordinates(aboba, start_date=start_date, end_date=end_date)
        set_status_ok(aboba, "Погода успешно загружена")
        QApplication.processEvents()

    except Exception as e:
        show_custom_message(
            aboba,
            title="Ошибка",
            text=f"Не удалось обновить файл Погода.csv:\n{e}",
            image_path="Картинки/Неудача.png",
        )


# -------------------------------------------КНОПКА "СБРОСИТЬ НАСТРОЙКИ"------------------------------------------------
def reset_order_filters(aboba):

    set_status_processing(aboba, "Сброс фильтров...")
    QApplication.processEvents()

    aboba.filter_date_from.clear()
    aboba.filter_date_to.clear()

    if hasattr(aboba, "max_export_users_input"):
        aboba.max_export_users_input.setValue(1000)

    if hasattr(aboba, "store_mode"):
        aboba.store_mode.setCurrentIndex(0)
    if hasattr(aboba, "kind_mode"):
        aboba.kind_mode.setCurrentIndex(0)

    if hasattr(aboba, "filter_season"):
        aboba.filter_season.clearSelection()

    if hasattr(aboba, "filter_store"):
        aboba.filter_store.clearSelection()
    if hasattr(aboba, "filter_kind"):
        aboba.filter_kind.clearSelection()
    if hasattr(aboba, "export_kind_filter"):
        aboba.export_kind_filter.clearSelection()

    apply_filters_all_stats(aboba)

    set_status_ok(aboba, "Фильтры сброшены")
    QApplication.processEvents()
    schedule_status_reset(aboba, 5)


# -------------------------------------------ФОРМИРУЕМ СПИСОК ГОРОДОВ ДЛЯ ВЫБОРА----------------------------------------
def load_cities_from_coordinates_file(aboba) -> None:
    path = os.path.join(os.getcwd(), "ВходныеДанные", "КоординатыГородов.csv")

    if not os.path.isfile(path):
        aboba._cities = []
        return

    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    except Exception:
        # если вдруг файл в другой кодировке
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)

    # нормализуем заголовки
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if "Город" not in df.columns:
        aboba._cities = []
        return

    cities = (
        df["Город"]
        .astype("string")
        .fillna("")
        .str.strip()
    )
    cities = [c for c in cities.tolist() if c and c.lower() not in ("nan", "none", "<na>")]

    # уникальные + сортировка
    aboba._cities = sorted(set(cities))


# -------------------------------------------ФОРМИРУЕМ СПИСОК МАГАЗИНОВ В ТАБЛИЦЕ---------------------------------------
def refresh_store_city_table(aboba) -> None:
    # Читаем магазины из файла
    stores_path = os.path.join(os.getcwd(), "ВходныеДанные", "СписокМагазинов.csv")

    stores: list[str] = []
    if os.path.isfile(stores_path):
        df = pd.read_csv(stores_path, sep="|", encoding="utf-8-sig", dtype=str)
        if not df.empty:
            col = "Магазин" if "Магазин" in df.columns else df.columns[0]
            s = df[col].dropna().astype(str).str.strip()
            s = s[s != ""]
            # Уникальные магазины
            stores = pd.unique(s).tolist()

    # Если файл пуст/не найден — таблицу всё равно покажем, но без строк
    aboba.store_city_table.setRowCount(len(stores))

    have_cities = bool(aboba._cities)

    for r, store in enumerate(stores):
        it = QTableWidgetItem(store)
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        aboba.store_city_table.setItem(r, 0, it)

        cb = QComboBox()
        cb.setEditable(False)

        if have_cities:
            cb.addItem("")  # пустое значение "не задано"
            cb.addItems(aboba._cities)

            # восстановим сохранённое значение
            cur_city = aboba._store_city_map.get(store, "")
            if cur_city:
                idx = cb.findText(cur_city)
                if idx >= 0:
                    cb.setCurrentIndex(idx)
        else:
            cb.addItem("Отсутствует файл КоординатыГородов.csv")
            cb.setEnabled(False)

        aboba.store_city_table.setCellWidget(r, 1, cb)


# -------------------------------------------ДОСТУПНОСТЬ ПОЛЕЙ ОТБОРА---------------------------------------------------
def update_filter_controls_availability(aboba):

    base = os.path.join(os.getcwd(), "ВходныеДанные")

    required = {
        "Заказы": "Заказы.csv",
        "Просмотры": "Просмотры.csv",
        "Избранное": "Избранное.csv",
        "Номенклатура": "Номенклатура.csv",
        "Категории": "КатегорииСайта.csv",
        "Координаты": "КоординатыГородов.csv",
        "Список магазинов": "СписокМагазинов.csv",
    }

    missing = [name for name, fn in required.items()
               if not os.path.isfile(os.path.join(base, fn))]

    ready = (len(missing) == 0)

    # Блокируем ВЕСЬ отбор + кнопки
    for attr in (
            "filter_date_from", "filter_date_to",
            "kind_mode", "filter_kind",
            "export_kind_filter",
            "store_mode", "filter_store",
            "store_city_table", "filter_season",
            "btn_apply", "btn_weather", "btn_reset",
            "filter_summary",
    ):
        w = getattr(aboba, attr, None)
        if w is not None:
            w.setEnabled(ready)

    # Подсказка пользователю (по желанию)
    if not ready:
        w = getattr(aboba, "filter_summary", None)
        if w is not None:
            w.setText("Для доступа к отбору загрузите файлы: " + ", ".join(missing))
            w.setCursorPosition(0)


# -------------------------------------------ОБНОВЛЯЕМ ВИДЫ НОМЕНКЛАТУРЫ В ОТБОРЕ---------------------------------------
def refresh_kind_values_from_loaded_files(aboba):
    lw = getattr(aboba, "filter_kind", None)
    if lw is None:
        return

    paths = dataset_paths()
    files = [paths["orders"], paths["views"], paths["favs"]]

    kinds_set = set()

    for fp in files:
        if not os.path.isfile(fp):
            continue
        # читаем только колонку
        df = pd.read_csv(fp, sep="|", dtype=str, usecols=["ВидНоменклатуры"])
        for v in df["ВидНоменклатуры"].dropna().astype(str).tolist():
            v = v.strip()
            if v:
                kinds_set.add(v)

    kinds = sorted(kinds_set)

    pending = getattr(aboba, "_pending_kind_selection", None)
    cur_sel = pending if pending is not None else get_selected_list_values(lw)

    set_list_widget_items(aboba, lw, kinds, cur_sel)

    if pending is not None:
        aboba._pending_kind_selection = None


def refresh_export_kind_values_from_nomenclature_file(aboba):
    """
    Заполняет список видов номенклатуры для фильтра итоговой выгрузки.

    Источник — текущий файл ВходныеДанные/Номенклатура.csv,
    потому что рекомендации формируются по актуальному каталогу.
    """
    lw = getattr(aboba, "export_kind_filter", None)
    if lw is None:
        return

    nom_path = os.path.join(
        os.getcwd(),
        "ВходныеДанные",
        "Номенклатура.csv"
    )

    if not os.path.isfile(nom_path):
        set_list_widget_items(aboba, lw, [], [])
        return

    kinds_set = set()

    df = pd.read_csv(
        nom_path,
        sep="|",
        dtype=str,
        encoding="utf-8-sig",
        usecols=lambda column: column == "ВидНоменклатуры",
    )

    if "ВидНоменклатуры" in df.columns:
        values = (
            df["ВидНоменклатуры"]
            .astype("string")
            .fillna("")
            .str.strip()
        )

        for value in values.tolist():
            if (
                value
                and value.lower()
                not in ("nan", "none", "null", "<na>", "-")
            ):
                kinds_set.add(value)

    kinds = sorted(kinds_set)

    pending = getattr(
        aboba,
        "_pending_export_kind_selection",
        None
    )

    if pending is not None:
        current_selection = pending
    else:
        current_selection = get_selected_list_values(lw)

    set_list_widget_items(
        aboba,
        lw,
        kinds,
        current_selection
    )

    if pending is not None:
        aboba._pending_export_kind_selection = None


def refresh_season_values_from_nomenclature_file(aboba):
    lw = getattr(aboba, "filter_season", None)
    if lw is None:
        return

    nom_path = os.path.join(os.getcwd(), "ВходныеДанные", "Номенклатура.csv")
    if not os.path.isfile(nom_path):
        set_list_widget_items(aboba, lw, [], [])
        return

    seasons_set = set()
    df = pd.read_csv(nom_path, sep="|", dtype=str, encoding="utf-8-sig", usecols=["Коллекция"])
    for v in df["Коллекция"].dropna().astype(str).tolist():
        v = v.strip()
        if v and v.lower() not in ("nan", "none", "null", "-"):
            seasons_set.add(v)

    seasons = sorted(seasons_set)

    pending = getattr(aboba, "_pending_season_selection", None)
    cur_sel = pending if pending is not None else get_selected_list_values(lw)

    set_list_widget_items(aboba, lw, seasons, cur_sel)

    if pending is not None:
        aboba._pending_season_selection = None


# -------------------------------------------ПЕРЕСОЗДАЕТ СПИСОК СОДЕРЖИМОГО И ВЫДЕЛЕННОГО В ОТБОРЕ----------------------
def set_list_widget_items(aboba, lw: QListWidget, values: list[str], selected_texts: list[str] | None = None):
    if lw is None:
        return
    selected_texts = set(selected_texts or [])
    lw.blockSignals(True)
    lw.clear()
    for v in values:
        lw.addItem(str(v))
    for i in range(lw.count()):
        item = lw.item(i)
        if item.text() in selected_texts:
            item.setSelected(True)
    lw.blockSignals(False)

    if lw is getattr(aboba, "filter_store", None):
        on_stores_list_updated(aboba)


# -------------------------------------------ОБНОВЛЯЕТ ТАБЛИЦУ СООТВЕТСТВИЯ СКЛАДОВ И ГОРОДОВ---------------------------
def on_stores_list_updated(aboba) -> None:
    # применяем карту из настроек, если она была загружена раньше
    pending_map = getattr(aboba, "_pending_store_city_map", None)
    if pending_map is not None:
        aboba._store_city_map = dict(pending_map or {})
        aboba._pending_store_city_map = None

    # подгружаем список городов из файла координат
    load_cities_from_coordinates_file(aboba)

    # перестраиваем таблицу "Склад -> Город"
    if hasattr(aboba, "store_city_table"):
        refresh_store_city_table(aboba)


# -------------------------------------------ЕЩЕ ОДНА ДОСТУПНОСТЬ ПОЛЕЙ ОТБОРА------------------------------------------
def set_order_filters_enabled(aboba, enabled: bool):
    for attr in (
            "filter_date_from",
            "filter_date_to",
            "store_mode",
            "filter_store",
            "kind_mode",
            "filter_kind",
            "export_kind_filter",
    ):
        w = getattr(aboba, attr, None)
        if w is not None:
            w.setEnabled(enabled)

    btn_apply = getattr(aboba, "btn_apply", None)
    if btn_apply is not None:
        btn_apply.setEnabled(enabled)

    btn_reset = getattr(aboba, "btn_reset", None)
    if btn_reset is not None:
        btn_reset.setEnabled(enabled)

    btn_weather = getattr(aboba, "btn_weather", None)
    if btn_weather is not None:
        btn_weather.setEnabled(enabled)


# ///////////////////////////////////////////АНАЛИЗ ФАЙЛОВ//////////////////////////////////////////////////////////
# -------------------------------------------АНАЛИЗ ЗАКЗАОВ---------------------------------------------------------
def analyze_orders_full_dataset(aboba) -> bool:
    try:
        file_path = "ВходныеДанные/Заказы.csv"

        # --- helpers for masked date fields ---
        def _masked_date_is_empty(le: QLineEdit) -> bool:
            if le is None:
                return True
            # inputMask оставляет в тексте пробелы и точки даже если пользователь ничего не ввёл
            t = le.text().replace(" ", "").replace(".", "")
            return t == ""

        def _get_date_or_none(le: QLineEdit):

            if le is None or _masked_date_is_empty(le):
                return None, None

            txt = le.text().strip()
            dt = pd.to_datetime(txt, errors="coerce", dayfirst=True)

            # сюда попадём, если введено неполностью или мусор
            if pd.isna(dt):
                return None, "Некорректная дата (ожидается дд.мм.гггг)"
            return dt, None

        # Очищаем ТОЛЬКО блок результата, фильтры оставляем на месте
        clear_layout(aboba, aboba.order_full_output_layout)
        aboba.order_full_stats_label = QLabel("")
        aboba.order_full_stats_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        aboba.order_full_stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        aboba.order_full_output_layout.addWidget(aboba.order_full_stats_label)

        # Проверяем, существует ли файл
        if not os.path.isfile(file_path):
            update_filter_controls_availability(aboba)
            refresh_kind_values_from_loaded_files(aboba)
            refresh_season_values_from_nomenclature_file(aboba)
            refresh_export_kind_values_from_nomenclature_file(aboba)
            update_filter_summary(aboba)
            vyvod_zaglyschek(
                text="Файл ещё не загружен",
                icon="Картинки/Внимание.png",
                main_layout=aboba.order_full_output_layout,
                stats_label=aboba.order_full_stats_label
            )
            return True

        # Загружаем CSV
        df = pd.read_csv(file_path, sep="|", dtype=str)
        update_filter_controls_availability(aboba)
        refresh_kind_values_from_loaded_files(aboba)
        refresh_season_values_from_nomenclature_file(aboba)
        refresh_export_kind_values_from_nomenclature_file(aboba)

        # ----------------- Обновляем списки фильтров из данных -----------------
        # (сохраняем текущие выделения)
        # Магазин
        if hasattr(aboba, "filter_store") and "Магазин" in df.columns:
            pending = getattr(aboba, "_pending_store_selection", None)
            cur_sel = pending if pending is not None else get_selected_list_values(aboba.filter_store)

            stores = sorted(df["Магазин"].dropna().unique().tolist())
            set_list_widget_items(aboba, aboba.filter_store, stores, cur_sel)

            if pending is not None:
                aboba._pending_store_selection = None  # применили один раз

        # ----------------- Дата в datetime (нужно для фильтра по периоду) -----------------
        if "Дата" in df.columns:
            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", format="%Y-%m-%d")

        # ----------------- Применяем фильтры -----------------
        errors = []

        # Дата "с" (учитываем inputMask: пустое поле != пустая строка)
        date_from, err_from = _get_date_or_none(getattr(aboba, "filter_date_from", None))
        if err_from:
            errors.append("Некорректная дата 'с' (ожидается дд.мм.гггг)")

        # Дата "по"
        date_to, err_to = _get_date_or_none(getattr(aboba, "filter_date_to", None))
        if err_to:
            errors.append("Некорректная дата 'по' (ожидается дд.мм.гггг)")

        if errors:
            vyvod_zaglyschek(
                text=";\n".join(errors),
                icon="Картинки/Внимание.png",
                main_layout=aboba.order_full_output_layout,
                stats_label=aboba.order_full_stats_label
            )
            return True

        # Фильтр по периоду
        if (date_from is not None or date_to is not None):
            if "Дата" not in df.columns:
                vyvod_zaglyschek(
                    text="В файле нет колонки 'Дата' — фильтр периода недоступен.",
                    icon="Картинки/Внимание.png",
                    main_layout=aboba.order_full_output_layout,
                    stats_label=aboba.order_full_stats_label
                )
                return True
            if date_from is not None:
                df = df[df["Дата"] >= date_from]
            if date_to is not None:
                df = df[df["Дата"] <= date_to]

        # -------- Магазин: мультивыбор + в группе / не в группе --------
        selected_stores = (
            get_selected_list_values(aboba.filter_store)
            if hasattr(aboba, "filter_store") else []
        )

        if selected_stores:
            if "Магазин" not in df.columns:
                vyvod_zaglyschek(
                    text="В файле нет колонки 'Магазин' — фильтр магазина недоступен.",
                    icon="Картинки/Внимание.png",
                    main_layout=aboba.order_full_output_layout,
                    stats_label=aboba.order_full_stats_label
                )
                return True

            mode = aboba.store_mode.currentText() if hasattr(aboba, "store_mode") else "В группе"

            if mode == "Не в группе":
                df = df[~df["Магазин"].isin(selected_stores)]
            else:  # "В группе"
                df = df[df["Магазин"].isin(selected_stores)]

        # -------- ВидНоменклатуры: мультивыбор + в группе / не в группе --------
        selected_kinds = (
            get_selected_list_values(aboba.filter_kind)
            if hasattr(aboba, "filter_kind") else []
        )

        if selected_kinds:
            if "ВидНоменклатуры" not in df.columns:
                vyvod_zaglyschek(
                    text="В файле нет колонки 'ВидНоменклатуры' — фильтр недоступен.",
                    icon="Картинки/Внимание.png",
                    main_layout=aboba.order_full_output_layout,
                    stats_label=aboba.order_full_stats_label
                )
                return True

            mode = aboba.kind_mode.currentText() if hasattr(aboba, "kind_mode") else "В группе"

            if mode == "Не в группе":
                df = df[~df["ВидНоменклатуры"].isin(selected_kinds)]
            else:  # "В группе"
                df = df[df["ВидНоменклатуры"].isin(selected_kinds)]

        # Если после фильтров данных нет — показываем заглушку
        if df.empty:
            vyvod_zaglyschek(
                text="Нет данных по выбранным фильтрам",
                icon="Картинки/Внимание.png",
                main_layout=aboba.order_full_output_layout,
                stats_label=aboba.order_full_stats_label
            )
            return True

        # Числовые поля
        df["Количество"] = pd.to_numeric(df["Количество"], errors="coerce").fillna(0).astype(int)
        df["КонечнаяСтоимость"] = pd.to_numeric(df["КонечнаяСтоимость"], errors="coerce").fillna(0).astype(
            float)
        df["НачальнаяСтоимость"] = pd.to_numeric(df["НачальнаяСтоимость"], errors="coerce").fillna(0).astype(
            float)
        df["ПроцентСкидки"] = pd.to_numeric(df["ПроцентСкидки"], errors="coerce").fillna(0).astype(float)
        df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce").fillna(0).astype(int)

        # Дата (ещё раз, чтобы гарантировать корректный dtype после фильтров/преобразований)
        if "Дата" in df.columns:
            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", format="%Y-%m-%d")

        # Количество заказов
        total_pokypok = df["Количество"].sum()
        total_orders = df["НомерЗаказа"].nunique()
        total_clients = df["MindboxID"].nunique()
        total_products = df["КодНоменклатуры"].nunique()

        # Покупки клиента
        purchases_per_client = df.groupby("MindboxID")["Количество"].sum()
        avg_purchases = round(purchases_per_client.mean(), 1)
        median_purchases = purchases_per_client.median()

        # Распределение по полу
        gender_series = df["ПолКлиента"].fillna("Не указан")
        gender_counts = df.groupby(gender_series)["Количество"].sum()
        total_gender = gender_counts.sum()
        gender_percent = (gender_counts / total_gender * 100).round(1).sort_values(ascending=False)
        gender_text = "\n".join([f"{gender}: {percent}%" for gender, percent in gender_percent.items()])

        # Возраст
        age_series = pd.to_numeric(df["Возраст"], errors="coerce")
        valid_age_series = age_series[(age_series >= 18) & (age_series <= 80)].dropna()
        avg_age = round(valid_age_series.mean(), 1) if not valid_age_series.empty else 0
        median_age = valid_age_series.median() if not valid_age_series.empty else 0

        valid_age_groups = df.loc[(age_series >= 18) & (age_series <= 80), "ВозрастнаяГруппа"].dropna()
        top_age_group = valid_age_groups.value_counts().idxmax() if not valid_age_groups.empty else "Не указано"

        # Разреженность
        num_clients = total_clients
        num_items = total_products
        interactions = total_orders
        sparsity = 1 - interactions / (num_clients * num_items) if (num_clients * num_items) else 0
        sparsity = round(sparsity * 100, 2)

        # Период
        if "Дата" in df.columns and df["Дата"].notna().any():
            period_start = df["Дата"].min().date()
            period_end = df["Дата"].max().date()
            period_str = f"{period_start.strftime('%d.%m.%Y')} — {period_end.strftime('%d.%m.%Y')}"
        else:
            period_str = "Не определён (нет корректных дат)"

        # Месяц с наибольшим количеством продаж
        months_ru = {
            1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
            5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
            9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
        }

        if "Дата" in df.columns:
            df["Месяц"] = df["Дата"].dt.month.astype("Int64")
            df["Год"] = df["Дата"].dt.year.astype("Int64")
            month_sales = df.groupby(["Год", "Месяц"])["Количество"].sum()
            if not month_sales.empty:
                top_year, top_month_num = month_sales.idxmax()
                top_month_str = f"{months_ru.get(top_month_num, top_month_num)} {int(top_year)}"
            else:
                top_month_str = "Не определён"
        else:
            top_month_str = "Не определён"

        # Популярные товары
        top_codes = (
            df.groupby("КодНоменклатуры")["Количество"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )

        top_products_pretty = []
        for code, count in top_codes.items():
            name = (
                df.loc[df["КодНоменклатуры"] == code, "НазваниеНаСайте"]
                .dropna()
                .astype(str)
                .iloc[0]
                if (df["КодНоменклатуры"] == code).any()
                else f"Название не найдено ({code})"
            )
            top_products_pretty.append(f"{name} ({code}) — {int(count)}")

        top_products_text = "\n".join(top_products_pretty)

        # Финансовые показатели
        grouped = df.groupby("Валюта")

        total_sales_dict = {}
        avg_check_dict = {}
        avg_discount_dict = {}

        for currency, df_cur in grouped:
            total_sales = int(df_cur["КонечнаяСтоимость"].sum())
            total_sales_str = f"{total_sales:,}".replace(",", ".")

            order_totals = df_cur.groupby("НомерЗаказа")["КонечнаяСтоимость"].sum()
            avg_check = int(order_totals.mean()) if not order_totals.empty else 0
            avg_check_str = f"{avg_check:,}".replace(",", ".")

            avg_discount = int(round(df_cur["ПроцентСкидки"].mean(), 0)) if not df_cur.empty else 0

            total_sales_dict[currency] = total_sales_str
            avg_check_dict[currency] = avg_check_str
            avg_discount_dict[currency] = avg_discount

        rub_sales = total_sales_dict.get("RUB", "0")
        kzt_sales = total_sales_dict.get("KZT", "0")

        rub_check = avg_check_dict.get("RUB", "0")
        kzt_check = avg_check_dict.get("KZT", "0")

        rub_disc = avg_discount_dict.get("RUB", 0)
        kzt_disc = avg_discount_dict.get("KZT", 0)

        output = (
            f"Финансовые показатели (RUB / KZT):\n"
            f"Общая сумма продаж — {rub_sales} / {kzt_sales}\n"
            f"Средняя сумма чека — {rub_check} / {kzt_check}\n"
            f"Средняя скидка — {rub_disc}% / {kzt_disc}%"
        )

        # Топ магазинов
        top_stores = df.groupby("Магазин")["Количество"].sum().sort_values(ascending=False).head(
            5) if "Магазин" in df.columns else pd.Series(dtype=float)
        top_stores_text = "\n".join([f"{store} — {int(count)}" for store, count in
                                     top_stores.items()]) if not top_stores.empty else "Нет данных"

        result = (
            f"Количество продаж: {total_pokypok}\n"
            f"Количество заказов: {total_orders}\n"
            f"Количество клиентов: {total_clients}\n"
            f"Количество товаров: {total_products}\n\n"
            f"Распределение продаж по полу:\n{gender_text}\n\n"
            f"Возраст клиента (средний/медианный): {avg_age} / {median_age}\n"
            f"Преобладающая возрастная категория: {top_age_group}\n\n"
            f"Количество продаж на клиента (среднее/медианное): {avg_purchases} / {median_purchases}\n\n"
            f"Разреженность матрицы клиент-товар: {sparsity}%\n\n"
            f"Период: {period_str}\n"
            f"Месяц с наибольшим количеством продаж: {top_month_str}\n\n"
            f"{output}\n\n"
            f"Топ-5 товаров по количеству продаж:\n{top_products_text}\n\n"
            f"Топ-5 магазинов по количеству продаж:\n{top_stores_text}"
        )

        aboba.order_full_stats_label.setText(result)
        aboba.order_full_stats_label.setStyleSheet("""QLabel {font-size: 16px;}""")

        update_filter_summary(aboba)
        return True

    except Exception as e:
        set_order_filters_enabled(aboba, False)
        vyvod_zaglyschek(
            text=f"Ошибка при анализе файла: {e}",
            icon="Картинки/Неудача.png",
            main_layout=aboba.order_full_output_layout,
            stats_label=aboba.order_full_stats_label)
        return False


# -------------------------------------------АНАЛИЗ ПРОСМОТРОВ------------------------------------------------------
def analyze_views_full_dataset(aboba) -> bool:
    try:
        file_path = "ВходныеДанные/Просмотры.csv"

        # Очищаем ТОЛЬКО блок результата на странице "Просмотры"
        clear_layout(aboba, aboba.views_full_output_layout)

        aboba.views_full_stats_label = QLabel("")
        aboba.views_full_stats_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        aboba.views_full_stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        aboba.views_full_output_layout.addWidget(aboba.views_full_stats_label)

        # Проверяем файл
        if not os.path.isfile(file_path):
            vyvod_zaglyschek(
                text="Файл ещё не загружен",
                icon="Картинки/Внимание.png",
                main_layout=aboba.views_full_output_layout,
                stats_label=aboba.views_full_stats_label
            )
            return True

        # Загружаем CSV
        df = pd.read_csv(file_path, sep="|", dtype=str)

        # Числовые поля
        if "Возраст" in df.columns:
            df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce").fillna(0).astype(int)

        # Дата (ВАЖНО: dayfirst=True, если у вас дд.мм.гггг)
        if "Дата" in df.columns:
            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", format="%Y-%m-%d")

        # -------- Период: отбор --------
        def _masked_date_is_empty(qle) -> bool:
            if qle is None:
                return True
            t = qle.text()
            if t is None:
                return True
            return t.replace(" ", "").replace(".", "") == ""

        date_from = ""
        if hasattr(aboba, "filter_date_from") and not _masked_date_is_empty(aboba.filter_date_from):
            date_from = aboba.filter_date_from.text().strip()

        date_to = ""
        if hasattr(aboba, "filter_date_to") and not _masked_date_is_empty(aboba.filter_date_to):
            date_to = aboba.filter_date_to.text().strip()

        if date_from or date_to:
            if "Дата" not in df.columns:
                vyvod_zaglyschek(
                    text="В файле нет колонки 'Дата' — отбор по периоду недоступен.",
                    icon="Картинки/Внимание.png",
                    main_layout=aboba.views_full_output_layout,
                    stats_label=aboba.views_full_stats_label
                )
                return True

            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", format="%Y-%m-%d")

            d_from = pd.to_datetime(date_from, errors="coerce", dayfirst=True) if date_from else None
            d_to = pd.to_datetime(date_to, errors="coerce", dayfirst=True) if date_to else None

            if d_from is not None and pd.notna(d_from):
                df = df[df["Дата"] >= d_from]
            if d_to is not None and pd.notna(d_to):
                df = df[df["Дата"] <= d_to]

        # -------- ВидНоменклатуры: мультивыбор + в группе / не в группе --------
        selected_kinds = (
            get_selected_list_values(aboba.filter_kind)
            if hasattr(aboba, "filter_kind") else []
        )

        if selected_kinds:
            if "ВидНоменклатуры" not in df.columns:
                vyvod_zaglyschek(
                    text="В файле нет колонки 'ВидНоменклатуры' — отбор по виду номенклатуры недоступен.",
                    icon="Картинки/Внимание.png",
                    main_layout=aboba.views_full_output_layout,
                    stats_label=aboba.views_full_stats_label
                )
                return True

            mode = aboba.kind_mode.currentText() if hasattr(aboba, "kind_mode") else "В группе"
            if mode == "Не в группе":
                df = df[~df["ВидНоменклатуры"].isin(selected_kinds)]
            else:
                df = df[df["ВидНоменклатуры"].isin(selected_kinds)]

        # Если после фильтров данных нет — показываем заглушку
        if df.empty:
            vyvod_zaglyschek(
                text="Нет данных по выбранным фильтрам",
                icon="Картинки/Внимание.png",
                main_layout=aboba.views_full_output_layout,
                stats_label=aboba.views_full_stats_label
            )
            return True

        # Если данных нет
        if df.empty:
            vyvod_zaglyschek(
                text="Файл пустой — нет данных для анализа.",
                icon="Картинки/Внимание.png",
                main_layout=aboba.views_full_output_layout,
                stats_label=aboba.views_full_stats_label
            )
            return True

        # Количество просмотров
        total_views = len(df)
        total_clients = df["MindboxID"].nunique() if "MindboxID" in df.columns else 0

        total_products = (
            df.loc[df.get("ТипТовара", "") == "Номенклатура", "КодНоменклатуры"].nunique()
            if "КодНоменклатуры" in df.columns else 0
        )
        total_categories = (
            df.loc[df.get("ТипТовара", "") == "Категория", "КодНоменклатуры"].nunique()
            if "КодНоменклатуры" in df.columns else 0
        )

        # Просмотры клиента
        if "MindboxID" in df.columns:
            views_per_client = df.groupby("MindboxID").size()
            avg_views = round(views_per_client.mean(), 1) if not views_per_client.empty else 0
            median_views = views_per_client.median() if not views_per_client.empty else 0
        else:
            avg_views, median_views = 0, 0

        # Распределение по полу
        if "ПолКлиента" in df.columns:
            gender_series = df["ПолКлиента"].fillna("Не указан")
            gender_counts = gender_series.value_counts()
            total_gender = gender_counts.sum()
            gender_percent = (gender_counts / total_gender * 100).round(1).sort_values(ascending=False)
            gender_text = "\n".join([f"{g}: {p}%" for g, p in gender_percent.items()])
        else:
            gender_text = "Нет данных"

        # Возраст
        if "Возраст" in df.columns:
            age_series = pd.to_numeric(df["Возраст"], errors="coerce")
            valid_age_mask = (age_series >= 18) & (age_series <= 80)
            valid_age_series = age_series[valid_age_mask].dropna()
            avg_age = round(valid_age_series.mean(), 1) if not valid_age_series.empty else 0
            median_age = valid_age_series.median() if not valid_age_series.empty else 0

            if "ВозрастнаяГруппа" in df.columns:
                valid_age_groups = df.loc[valid_age_mask, "ВозрастнаяГруппа"].dropna()
                top_age_group = valid_age_groups.value_counts().idxmax() if not valid_age_groups.empty else "Не указано"
            else:
                top_age_group = "Не указано"
        else:
            avg_age, median_age, top_age_group = 0, 0, "Не указано"

        # Период
        if "Дата" in df.columns and df["Дата"].notna().any():
            period_start = df["Дата"].min().date()
            period_end = df["Дата"].max().date()
            period_str = f"{period_start:%d.%m.%Y} — {period_end:%d.%m.%Y}"
        else:
            period_str = "Не определён (нет корректных дат)"

        # Месяц с наибольшим количеством просмотров
        months_ru = {
            1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
            5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
            9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
        }

        if "Дата" in df.columns and df["Дата"].notna().any():
            df["Месяц"] = df["Дата"].dt.month.astype("Int64")
            df["Год"] = df["Дата"].dt.year.astype("Int64")
            month_views = df.groupby(["Год", "Месяц"]).size()
            if not month_views.empty:
                top_year, top_month_num = month_views.idxmax()
                top_month_str = f"{months_ru.get(top_month_num, top_month_num)} {int(top_year)}"
            else:
                top_month_str = "Не определён"
        else:
            top_month_str = "Не определён"

        # Топ-5 товаров
        top_products_text = "Нет данных"
        if "ТипТовара" in df.columns:
            df_nom = df[df["ТипТовара"] == "Номенклатура"]
            if not df_nom.empty and "КодНоменклатуры" in df_nom.columns:
                if "НазваниеНаСайте" in df_nom.columns:
                    top_nom = (
                        df_nom.groupby(["КодНоменклатуры", "НазваниеНаСайте"])
                        .size()
                        .sort_values(ascending=False)
                        .head(5)
                    )
                    top_products_text = "\n".join(
                        f"{name} ({code}) — {count}"
                        for (code, name), count in top_nom.items()
                    )
                else:
                    top_nom = (
                        df_nom["КодНоменклатуры"]
                        .value_counts()
                        .head(5)
                    )
                    top_products_text = "\n".join(
                        f"{code} — {count}"
                        for code, count in top_nom.items()
                    )

        # Топ-5 категорий
        top_cat_text = "Нет данных"
        if "ТипТовара" in df.columns:
            df_cat = df[df["ТипТовара"] == "Категория"]
            if not df_cat.empty and "КодНоменклатуры" in df_cat.columns:
                if "НазваниеКатегории" in df_cat.columns:
                    top_cat = (
                        df_cat.groupby(["КодНоменклатуры", "НазваниеКатегории"])
                        .size()
                        .sort_values(ascending=False)
                        .head(5)
                    )
                    top_cat_text = "\n".join(
                        f"{name} ({code}) — {count}"
                        for (code, name), count in top_cat.items()
                    )
                else:
                    top_cat = df_cat["КодНоменклатуры"].value_counts().head(5)
                    top_cat_text = "\n".join(
                        f"{code} — {count}"
                        for code, count in top_cat.items()
                    )

        # Формируем текст
        result = (
            f"Количество просмотров: {total_views}\n"
            f"Количество клиентов: {total_clients}\n"
            f"Количество товаров: {total_products}\n"
            f"Количество категорий: {total_categories}\n\n"
            f"Распределение просмотров по полу:\n{gender_text}\n\n"
            f"Возраст клиента (средний/медианный): {avg_age} / {median_age}\n"
            f"Преобладающая возрастная категория: {top_age_group}\n\n"
            f"Количество просмотров на клиента (среднее/медианное): {avg_views} / {median_views}\n\n"
            f"Период: {period_str}\n"
            f"Месяц с наибольшим количеством просмотров: {top_month_str}\n\n"
            f"Топ-5 товаров по количеству просмотров:\n{top_products_text}\n\n"
            f"Топ-5 категорий по количеству просмотров:\n{top_cat_text}"
        )

        aboba.views_full_stats_label.setText(result)
        aboba.views_full_stats_label.setStyleSheet("""QLabel {font-size: 16px;}""")
        return True

    except Exception as e:
        vyvod_zaglyschek(
            text=f"Ошибка при анализе файла: {e}",
            icon="Картинки/Неудача.png",
            main_layout=aboba.views_full_output_layout,
            stats_label=aboba.views_full_stats_label
        )
        return False


# -------------------------------------------АНАЛИЗ ИЗБРАННОГО------------------------------------------------------
def analyze_favorites_full_dataset(aboba) -> bool:
    try:
        file_path = "ВходныеДанные/Избранное.csv"

        # Очищаем ТОЛЬКО блок результата на странице "Избранное"
        clear_layout(aboba, aboba.favorites_full_output_layout)

        aboba.favorites_full_stats_label = QLabel("")
        aboba.favorites_full_stats_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        aboba.favorites_full_stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        aboba.favorites_full_output_layout.addWidget(aboba.favorites_full_stats_label)

        if not os.path.isfile(file_path):
            vyvod_zaglyschek(
                text="Файл ещё не загружен",
                icon="Картинки/Внимание.png",
                main_layout=aboba.favorites_full_output_layout,
                stats_label=aboba.favorites_full_stats_label
            )
            return True

        df = pd.read_csv(file_path, sep="|", dtype=str)

        # Числовые поля
        df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce").fillna(0).astype(int)

        # Дата
        if "Дата" in df.columns:
            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")

        # -------- Период: отбор --------
        def _masked_date_is_empty(qle) -> bool:
            if qle is None:
                return True
            t = qle.text()
            if t is None:
                return True
            return t.replace(" ", "").replace(".", "") == ""

        date_from = ""
        if hasattr(aboba, "filter_date_from") and not _masked_date_is_empty(aboba.filter_date_from):
            date_from = aboba.filter_date_from.text().strip()

        date_to = ""
        if hasattr(aboba, "filter_date_to") and not _masked_date_is_empty(aboba.filter_date_to):
            date_to = aboba.filter_date_to.text().strip()

        if date_from or date_to:
            if "Дата" not in df.columns:
                vyvod_zaglyschek(
                    text="В файле нет колонки 'Дата' — отбор по периоду недоступен.",
                    icon="Картинки/Внимание.png",
                    main_layout=aboba.favorites_full_output_layout,
                    stats_label=aboba.favorites_full_stats_label
                )
                return True

            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", format="%Y-%m-%d")

            d_from = pd.to_datetime(date_from, errors="coerce", dayfirst=True) if date_from else None
            d_to = pd.to_datetime(date_to, errors="coerce", dayfirst=True) if date_to else None

            if d_from is not None and pd.notna(d_from):
                df = df[df["Дата"] >= d_from]
            if d_to is not None and pd.notna(d_to):
                df = df[df["Дата"] <= d_to]

        # -------- ВидНоменклатуры: мультивыбор + в группе / не в группе --------
        selected_kinds = (
            get_selected_list_values(aboba.filter_kind)
            if hasattr(aboba, "filter_kind") else []
        )

        if selected_kinds:
            if "ВидНоменклатуры" not in df.columns:
                vyvod_zaglyschek(
                    text="В файле нет колонки 'ВидНоменклатуры' — отбор по виду номенклатуры недоступен.",
                    icon="Картинки/Внимание.png",
                    main_layout=aboba.favorites_full_output_layout,
                    stats_label=aboba.favorites_full_stats_label
                )
                return True

            mode = aboba.kind_mode.currentText() if hasattr(aboba, "kind_mode") else "В группе"
            if mode == "Не в группе":
                df = df[~df["ВидНоменклатуры"].isin(selected_kinds)]
            else:
                df = df[df["ВидНоменклатуры"].isin(selected_kinds)]

        # Если после фильтров данных нет — показываем заглушку
        if df.empty:
            vyvod_zaglyschek(
                text="Нет данных по выбранным фильтрам",
                icon="Картинки/Внимание.png",
                main_layout=aboba.favorites_full_output_layout,
                stats_label=aboba.favorites_full_stats_label
            )
            return True

        # Количество добавлений
        total_fav = len(df)
        total_clients = df["MindboxID"].nunique()
        total_products = df["КодНоменклатуры"].nunique()

        # Добавления клиента
        fav_per_client = df.groupby("MindboxID").size()
        avg_fav = round(fav_per_client.mean(), 1)
        median_fav = fav_per_client.median()

        # Распределение по полу
        gender_series = df["ПолКлиента"].fillna("Не указан")
        gender_counts = gender_series.value_counts()
        total_gender = gender_counts.sum()
        gender_percent = (gender_counts / total_gender * 100).round(1)
        gender_percent = gender_percent.sort_values(ascending=False)

        # Формируем красивый текст
        gender_text = "\n".join([f"{gender}: {percent}%" for gender, percent in gender_percent.items()])

        # Возраст
        age_series = pd.to_numeric(df["Возраст"], errors="coerce").dropna()
        valid_age_series = age_series[(age_series >= 18) & (age_series <= 80)]
        avg_age = round(valid_age_series.mean(), 1) if not valid_age_series.empty else 0
        median_age = valid_age_series.median() if not valid_age_series.empty else 0

        valid_age_groups = df.loc[(age_series >= 18) & (age_series <= 80), "ВозрастнаяГруппа"].dropna()
        top_age_group = (
            valid_age_groups.value_counts().idxmax()
            if not valid_age_groups.empty
            else "Не указано"
        )

        # Период
        if "Дата" in df.columns and df["Дата"].notna().any():
            period_start = df["Дата"].min().date()
            period_end = df["Дата"].max().date()
            period_str = f"{period_start:%d.%m.%Y} — {period_end:%d.%m.%Y}"
        else:
            period_str = "Не определён (нет корректных дат)"

        # Месяц с наибольшим количеством добавлений
        months_ru = {
            1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
            5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
            9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
        }

        if "Дата" in df.columns and df["Дата"].notna().any():
            df["Месяц"] = df["Дата"].dt.month.astype("Int64")
            df["Год"] = df["Дата"].dt.year.astype("Int64")

            # Суммируем добавления по месяцу и году
            month_sales = df.groupby(["Год", "Месяц"]).size()
            top_year, top_month_num = month_sales.idxmax()
            top_month_str = f"{months_ru[top_month_num]} {int(top_year)}"
        else:
            top_month_str = "Не определён"

        # Притягиваем номенклатуру
        top_fav = (
            df.groupby(["КодНоменклатуры", "НазваниеНаСайте"])
            .size()
            .sort_values(ascending=False)
            .head(5)
        )

        top_products_text = "\n".join(
            f"{name} ({code}) — {count}"
            for (code, name), count in top_fav.items()
        )

        # Формируем текст
        result = (
            f"Количество добавлений: {total_fav}\n"
            f"Количество клиентов: {total_clients}\n"
            f"Количество товаров: {total_products}\n\n"
            f"Распределение добавлений по полу: \n{gender_text}\n\n"
            f"Возраст клиента (средний/медианный): {avg_age} / {median_age}\n"
            f"Преобладающая возрастная категория: {top_age_group}\n\n"
            f"Количество добавлений на клиента (среднее/медианное): {avg_fav} / {median_fav}\n\n"
            f"Период: {period_str}\n"
            f"Месяц с наибольшим количеством добавлений: {top_month_str}\n\n"
            f"Топ-5 товаров по количеству добавлений: \n{top_products_text}"
        )

        aboba.favorites_full_stats_label.setText(result)
        aboba.favorites_full_stats_label.setStyleSheet("""QLabel {font-size: 16px;}""")
        return True

    except Exception as e:
        vyvod_zaglyschek(text=f"Ошибка при анализе файла: {e}", icon="Картинки/Неудача.png",
                         main_layout=aboba.favorites_full_output_layout,
                         stats_label=aboba.favorites_full_stats_label)
        return False


# ///////////////////////////////////////////ЗАГРУЗКА ФАЙЛОВ////////////////////////////////////////////////////////////
def load_csv_file(aboba):
    # Статус бар
    set_status_processing(aboba, "Обработка данных...")

    selected_type = aboba.combo_box_types.currentText()
    mode = aboba.combo_box_add_or_not.currentText()

    file_path, _ = QFileDialog.getOpenFileName(
        aboba,
        "Выберите CSV файл",
        "",
        "CSV files (*.csv);;All files (*)",
    )

    if not file_path:
        set_status_ok(aboba, "Не хочешь, как хочешь...")
        schedule_status_reset(aboba, 5)
        return

    try:
        input_dir = os.path.join(os.getcwd(), "ВходныеДанные")
        os.makedirs(input_dir, exist_ok=True)

        # --- helpers (локально, чтобы не засорять класс лишними методами) ---
        def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()

            if "Дата" in df.columns:
                df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce").dt.normalize()

            if "ДатаРождения" in df.columns:
                df["ДатаРождения"] = pd.to_datetime(df["ДатаРождения"], errors="coerce")

            for col in ("Телефон", "ДисконтнаяКарта", "Возраст", "MindboxID"):
                if col in df.columns:
                    df[col] = (
                        df[col]
                        .astype("string")
                        .str.replace(r"\.0$", "", regex=True)
                        .str.strip()
                    )

            return df

        def _save(df: pd.DataFrame, save_path: str) -> None:
            save_dir = os.path.dirname(os.path.abspath(save_path))
            temp_fd = None
            temp_path = None

            try:
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=save_dir,
                    prefix=f".{os.path.basename(save_path)}.",
                    suffix=".tmp",
                )
                try:
                    os.close(temp_fd)
                finally:
                    # Не закрываем дескриптор повторно, если os.close() вызвал исключение.
                    temp_fd = None

                df.to_csv(temp_path, index=False, sep="|", encoding="utf-8-sig")
                os.replace(temp_path, save_path)
                temp_path = None
            except BaseException:
                cleanup_errors = []

                if temp_fd is not None:
                    try:
                        os.close(temp_fd)
                    except OSError as cleanup_error:
                        cleanup_errors.append(("close", cleanup_error))

                if temp_path is not None:
                    try:
                        os.remove(temp_path)
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        cleanup_errors.append(("remove", cleanup_error))

                for cleanup_action, cleanup_error in cleanup_errors:
                    try:
                        print(
                            "Temporary CSV cleanup failed "
                            f"({cleanup_action}, path={temp_path!a}): "
                            f"{cleanup_error!a}",
                            file=sys.stderr,
                        )
                    except Exception:
                        # Диагностика cleanup не должна скрывать исходную ошибку.
                        pass

                raise

        def _append_or_overwrite(df: pd.DataFrame, save_path: str) -> pd.DataFrame:

            if mode == "Добавить новый / Обновить существующий":
                _save(df, save_path)
                return df

            # mode == "Добавить данные к существующему"
            if os.path.exists(save_path):
                df_old = pd.read_csv(save_path, sep="|", encoding="utf-8-sig", dtype=str)
                df_old = _sanitize_df(df_old)
                df = pd.concat([df_old, df], ignore_index=True)

            _save(df, save_path)
            return df

        def _process_pair(reader_sep: str, processor_fn, filename: str) -> tuple[pd.DataFrame | None, str]:
            df_src = read_csv_auto_encoding(aboba, file_path=file_path, sep=reader_sep)
            if df_src is None:
                return None, ""

            df = processor_fn(aboba, df_src)
            if df is None:
                return None, ""

            save_path = os.path.join(input_dir, filename)
            df_final = _append_or_overwrite(df, save_path)
            return df_final, save_path

        def _process_single(reader_sep: str, processor_fn, filename: str) -> bool:
            df_src = read_csv_auto_encoding(aboba, file_path=file_path, sep=reader_sep)
            if df_src is None:
                return False

            df_res = processor_fn(aboba, df_src)
            if df_res is None:
                return False

            save_path_full = os.path.join(input_dir, filename)
            _save(df_res, save_path_full)

            return True

        def _save_store_list(df_orders: pd.DataFrame, save_dir: str) -> None:
            if "Магазин" not in df_orders.columns:
                stores_df = pd.DataFrame({"Магазин": []})
            else:
                s = (
                    df_orders["Магазин"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                s = s[s != ""]
                stores = sorted(pd.unique(s))  # можно убрать sorted(), если хочешь сохранить исходный порядок
                stores_df = pd.DataFrame({"Магазин": stores})

            stores_path = os.path.join(save_dir, "СписокМагазинов.csv")
            _save(stores_df, stores_path)

        # --- routing по типу ---
        if selected_type == "Заказы клиентов из Mindbox":
            df_final, _ = _process_pair(
                reader_sep=";",
                processor_fn=process_orders_file,
                filename="Заказы.csv",
            )
            if df_final is None:
                return

            _save_store_list(df_final, input_dir)

        elif selected_type == "Просмотры товаров и категорий из Mindbox":
            df_final, _ = _process_pair(
                reader_sep=";",
                processor_fn=process_views_file,
                filename="Просмотры.csv",
            )
            if df_final is None:
                return

        elif selected_type == "Добавление товаров в избранное из Mindbox":
            df_final, _ = _process_pair(
                reader_sep=";",
                processor_fn=process_favorites_file,
                filename="Избранное.csv",
            )
            if df_final is None:
                return

        elif selected_type == "Номенклатура из 1С":
            ok = _process_single(
                reader_sep="|",
                processor_fn=process_nomenclature_file,
                filename="Номенклатура.csv",
            )

            if not ok:
                return

            aboba._name_by_code = None
            aboba._collection_by_code = None
            aboba._stock_by_code = None

        elif selected_type == "Категории сайта из 1С":
            ok = _process_single(
                reader_sep="|",
                processor_fn=process_categories_file,
                filename="КатегорииСайта.csv",
            )

            if not ok:
                return

        elif selected_type == "Координаты городов и погода":
            ok = _process_single(
                reader_sep=",",
                processor_fn=process_coordinates_file,
                filename="КоординатыГородов.csv",
            )

            if not ok:
                return

            load_cities_from_coordinates_file(aboba)
            refresh_store_city_table(aboba)

        else:
            show_custom_message(aboba,
                                title="Ошибка",
                                text="Неизвестный тип данных",
                                image_path="Картинки/Неудача.png",
                                )
            set_status_error(aboba, "Неизвестный тип данных")
            schedule_status_reset(aboba, 5)
            return

        # Обновляем статус + вкладки статистики
        update_file_status(aboba)

        update_filter_controls_availability(aboba)
        refresh_kind_values_from_loaded_files(aboba)
        refresh_season_values_from_nomenclature_file(aboba)
        refresh_export_kind_values_from_nomenclature_file(aboba)
        update_filter_summary(aboba)

        # Пересчитываем статистику
        analyze_orders_full_dataset(aboba)
        analyze_views_full_dataset(aboba)
        analyze_favorites_full_dataset(aboba)

    except Exception as e:
        show_custom_message(aboba,
                            title="Ошибка",
                            text=f"Не удалось загрузить файл:\n{str(e)}",
                            image_path="Картинки/Неудача.png",
                            )
        set_status_error(aboba, "Ошибка обработки")
        schedule_status_reset(aboba, 5)


# -------------------------------------------АВТОМАТИЧЕСКАЯ КОДИРОВКА-----------------------------------------------
def read_csv_auto_encoding(aboba, file_path: str, sep: str):
    try:
        # Определяем кодировку
        with open(file_path, "rb") as f:
            raw = f.read(200_000)
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "utf-8")

        # Пробуем прочитать файл
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
        return df

    except Exception as e:
        show_custom_message(aboba,
                            title="Ошибка",
                            text=f"Не удалось прочитать файл:\n{file_path}\n\nПричина:\n{str(e)}",
                            image_path="Картинки/Неудача.png"
                            )
        set_status_error(aboba, "Ошибка чтения файла")
        schedule_status_reset(aboba, 5)
        return None
