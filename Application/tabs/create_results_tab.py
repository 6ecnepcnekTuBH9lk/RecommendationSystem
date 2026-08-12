import os
import pandas as pd
import re
import requests
from Application.model import BPRMF as BPRMF_module
from PyQt6.QtCore import Qt, QSize
from typing import Iterable, Optional
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QLineEdit, QComboBox, QFrame, QFormLayout, QGridLayout, QSizePolicy,
                             QAbstractItemView, QTableWidget, QHeaderView,
                             QTableWidgetItem)

from Application.photo.photo_processing import (_ensure_photo_map, _set_photo_cell)

from Application.settings.settings_and_filter import (
    get_selected_list_values
)

from Application.settings.set_status import (set_status_processing, schedule_status_reset, show_custom_message,
                                             set_status_error, set_status_ok)


# -------------------------------------------ВКЛАДКА ВЫГРУЗКА РЕЗУЛЬТАТОВ-------------------------------------------
def create_result_widgets_tab(aboba):
    tab = QWidget()

    # === ВЕРХНИЙ УРОВЕНЬ: вертикально ===
    outer = QVBoxLayout(tab)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    # ================== 1) ВЕРХНЯЯ ПАНЕЛЬ УПРАВЛЕНИЯ (2 строки) ==================
    top_controls = QVBoxLayout()
    top_controls.setContentsMargins(10, 10, 10, 0)
    top_controls.setSpacing(10)

    # ---- фильтр клиента ----
    aboba.client_filter_field = QComboBox()
    aboba.client_filter_field.addItems(["MindboxID", "Дисконтная карта", "Почта", "Телефон"])
    aboba.client_filter_field.setCurrentText("MindboxID")

    aboba.mb_input = QLineEdit()
    aboba.mb_input.setPlaceholderText("Введите значение...")

    # ---- рекомендации ----
    aboba.recs_topk = QComboBox()
    aboba.recs_topk.addItems(["Топ-1", "Топ-3", "Топ-5", "Топ-10"])
    aboba.recs_topk.setCurrentText("Топ-10")

    # ---- кнопки ----
    aboba.btn_show_history = QPushButton(QIcon("Картинки/Поиск.png"), " Получить данные")
    aboba.btn_show_history.setIconSize(QSize(17, 17))
    aboba.btn_show_history.clicked.connect(lambda: show_purchase_history_clicked(aboba))

    aboba.btn_excel = QPushButton(QIcon("Картинки/Эксель.png"), " Выгрузить рекомендации")
    aboba.btn_excel.setIconSize(QSize(17, 17))
    aboba.btn_excel.clicked.connect(lambda: export_recommendations_to_excel(aboba))

    # ---- 1 строка: Идентификатор клиента: ----
    row1 = QHBoxLayout()
    row1.setSpacing(10)

    lbl_client = QLabel("Идентификатор клиента:")

    row1.addWidget(lbl_client, 0)
    row1.addWidget(aboba.client_filter_field, 0)
    row1.addWidget(aboba.mb_input, 1)

    # ---- 2 строка: Количество рекомендаций ----
    row2 = QHBoxLayout()
    row2.setSpacing(10)

    lbl_recs = QLabel("Количество рекомендаций:")
    lbl_recs.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    row2.addWidget(lbl_recs, 0)
    row2.addWidget(aboba.recs_topk, 0)

    # кнопки поровну делят всё оставшееся пространство
    aboba.btn_show_history.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    aboba.btn_excel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    row2.addWidget(aboba.btn_excel, 1)
    row2.addWidget(aboba.btn_show_history, 1)  # stretch=1

    # складываем строки
    top_controls.addLayout(row1)
    top_controls.addLayout(row2)

    # ====== ОБЁРТКА: верхняя панель = 2 колонки (левая 50%, правая 50%) ======
    top_wrap = QHBoxLayout()

    top_left = QWidget()
    top_left.setLayout(top_controls)

    aboba.top_right = QWidget()

    right_box = QVBoxLayout(aboba.top_right)
    right_box.setContentsMargins(10, 10, 10, 0)
    right_box.setAlignment(Qt.AlignmentFlag.AlignTop)

    grid = QGridLayout()
    grid.setHorizontalSpacing(15)
    grid.setVerticalSpacing(0)

    def _mk_readonly_line() -> QLineEdit:
        le = QLineEdit()
        le.setReadOnly(True)
        le.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return le

    def _mk_col(pairs: list[tuple[str, str]]) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(5)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        for title, attr in pairs:
            lab = QLabel(title)
            le = _mk_readonly_line()
            setattr(aboba, attr, le)
            form.addRow(lab, le)

        return w

    col1 = _mk_col([
        ("MindboxID:", "le_mb"),
        ("Дисконтная карта:", "le_card"),
    ])

    col2 = _mk_col([
        ("Почта:", "le_email"),
        ("Телефон:", "le_phone"),
    ])

    col3 = _mk_col([
        ("Пол:", "le_gender"),
        ("ФИО:", "le_fio"),
    ])

    col4 = _mk_col([
        ("Возраст:", "le_age"),
        ("Возрастная группа:", "le_agegrp"),
    ])

    grid.addWidget(col1, 0, 0)
    grid.addWidget(col2, 0, 1)
    grid.addWidget(col3, 0, 2)
    grid.addWidget(col4, 0, 3)

    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    grid.setColumnStretch(2, 1)
    grid.setColumnStretch(3, 1)

    right_box.addLayout(grid)

    top_wrap.addWidget(top_left, 1)
    top_wrap.addWidget(aboba.top_right, 1)

    outer.addLayout(top_wrap)
    outer.addSpacing(10)

    # ================== 2) ГОРИЗОНТАЛЬНАЯ ЛИНИЯ ==================
    hline = QFrame()
    hline.setObjectName("hSeparator")
    hline.setFixedHeight(1)
    hline.setFrameShape(QFrame.Shape.NoFrame)
    outer.addWidget(hline)

    # ================== 3) НИЖНЯЯ ЧАСТЬ: ДВЕ КОЛОНКИ ==================
    root = QHBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(10)
    outer.addLayout(root, 1)

    # Левая панель
    left_wrap = QWidget()
    left_layout = QVBoxLayout(left_wrap)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(0)

    # Правая панель
    right_wrap = QWidget()
    right_layout = QVBoxLayout(right_wrap)
    right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(0)

    # Вертикальный разделитель
    separator = QFrame()
    separator.setObjectName("vSeparator")
    separator.setFixedWidth(1)
    separator.setFrameShape(QFrame.Shape.NoFrame)

    root.addWidget(left_wrap, 1)
    root.addWidget(separator)
    root.addWidget(right_wrap, 1)

    # -------------------- ЛЕВАЯ ЧАСТЬ --------------------
    aboba.label_123 = QLabel("История взаимодействий клиента")
    aboba.label_123.setSizePolicy(aboba.label_123.sizePolicy().Policy.Fixed, aboba.label_123.sizePolicy().Policy.Fixed)
    aboba.label_123.setContentsMargins(0, 0, 0, 0)
    aboba.label_123.setStyleSheet("""
        QLabel {
            background-color: #FAFAFA;
            padding: 7px 65px;
            border-radius: 10px;
            border: 1px solid #C8C8C8;
            margin: 10px 0px 10px 0px;
        }
    """)
    left_layout.addWidget(aboba.label_123, alignment=Qt.AlignmentFlag.AlignHCenter)

    aboba.purchases_table = QTableWidget(0, 6)
    aboba.purchases_table.setHorizontalHeaderLabels([
        "Фото", "Код", "Название", "Коллекция", "Взаимодействие", "Дата"
    ])
    aboba.purchases_table.verticalHeader().setVisible(False)
    aboba.purchases_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    aboba.purchases_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    aboba.purchases_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
    aboba.purchases_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    aboba.purchases_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
    aboba.purchases_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
    aboba.purchases_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
    aboba.purchases_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
    aboba.purchases_table.setGridStyle(Qt.PenStyle.SolidLine)
    aboba.purchases_table.setStyleSheet("""QTableWidget { margin: 0px 0px 10px 10px; }""")

    left_layout.addWidget(aboba.purchases_table, 1)

    # -------------------- ПРАВАЯ ЧАСТЬ --------------------
    aboba.label_recs = QLabel("Рекомендации")
    aboba.label_recs.setSizePolicy(aboba.label_recs.sizePolicy().Policy.Fixed,
                                   aboba.label_recs.sizePolicy().Policy.Fixed)
    aboba.label_recs.setContentsMargins(0, 0, 0, 0)
    aboba.label_recs.setStyleSheet("""
        QLabel {
            background-color: #FAFAFA;
            padding: 7px 65px;
            border-radius: 10px;
            border: 1px solid #C8C8C8;
            margin: 10px 0px 10px 0px;
        }
    """)
    right_layout.addWidget(aboba.label_recs, alignment=Qt.AlignmentFlag.AlignHCenter)

    aboba.recs_table = QTableWidget(0, 7)
    aboba.recs_table.setHorizontalHeaderLabels([
        "Фото", "Код", "Название", "Коллекция", "Коэффициент", "Конверсия", "Остаток"
    ])
    aboba.recs_table.verticalHeader().setVisible(False)
    aboba.recs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    aboba.recs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    aboba.recs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
    aboba.recs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    aboba.recs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
    aboba.recs_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
    aboba.recs_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
    aboba.recs_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
    aboba.recs_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
    aboba.recs_table.setGridStyle(Qt.PenStyle.SolidLine)
    aboba.recs_table.setStyleSheet("""QTableWidget { margin: 0px 10px 10px 0px; }""")

    right_layout.addWidget(aboba.recs_table, 1)

    # Название вкладки
    aboba.tabs.addTab(tab, "Выгрузка результатов")


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# -------------------------------------------ОСНОВНАЯ ФУНКЦИЯ ПОЛУЧЕНИЯ ДАННЫХ------------------------------------------
def show_purchase_history_clicked(aboba):
    field_ui = aboba.client_filter_field.currentText().strip()
    value = aboba.mb_input.text().strip()

    if not value:
        set_status_error(aboba, "Не введён идентификатор клиента")
        aboba.status_label.repaint()
        aboba.status_icon.repaint()
        QApplication.processEvents()
        schedule_status_reset(aboba, 5)

        show_custom_message(aboba, "Ошибка", "Необходимо ввести идентификатор клиента для поиска",
                            "Картинки/Неудача.png")
        aboba.purchases_table.setRowCount(0)
        aboba.recs_table.setRowCount(0)
        _clear_client_info_panel(aboba)
        return

    set_status_processing(aboba, "Идёт поиск данных клиента...")

    aboba.status_label.repaint()
    aboba.status_icon.repaint()
    QApplication.processEvents()

    try:

        mindbox_ids = _resolve_mindbox_ids(field_ui, value)
        if not mindbox_ids:
            set_status_error(aboba, "Клиент не найден")
            aboba.status_label.repaint()
            aboba.status_icon.repaint()
            QApplication.processEvents()
            schedule_status_reset(aboba, 5)

            show_custom_message(aboba, "Ошибка", "По заданному идентификатору клиент не найден",
                                "Картинки/Неудача.png")
            aboba.purchases_table.setRowCount(0)
            aboba.recs_table.setRowCount(0)
            _clear_client_info_panel(aboba)
            return

        mindbox_id = mindbox_ids[0]

        # заполняем правую панель данными клиента
        info = _load_client_info(mindbox_id)
        _fill_client_info_panel(aboba, info)

        # фото-мапа (код -> ссылка/путь)
        _ensure_photo_map(aboba)

        # --- новый цикл заполнения таблиц: новое "поколение" картинок ---
        aboba._img_gen += 1
        gen = aboba._img_gen

        aboba._img_queue.clear()
        aboba._img_targets.clear()
        aboba._img_inflight.clear()

        if hasattr(aboba, "_img_retry_count"):
            aboba._img_retry_count.clear()

        # ===================== ИСТОРИЯ ВЗАИМОДЕЙСТВИЙ =====================
        df = _load_client_interactions(aboba, mindbox_id)
        if df.empty:
            aboba.purchases_table.setRowCount(0)
        else:
            aboba.purchases_table.setRowCount(0)
            aboba.purchases_table.clearContents()
            aboba.purchases_table.setRowCount(len(df))

            for r, row in enumerate(df.itertuples(index=False)):
                code = str(row.КодНоменклатуры)
                name = str(row.НазваниеНоменклатуры)
                collection = str(getattr(row, "Коллекция", "") or "").strip()
                interaction = str(row.Взаимодействие)
                dt_text = str(row.ДатаВзаимодействия)

                if collection.lower() == "nan":
                    collection = ""

                it = QTableWidgetItem(code)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.purchases_table.setItem(r, 1, it)

                it = QTableWidgetItem(name)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.purchases_table.setItem(r, 2, it)

                it = QTableWidgetItem(collection)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.purchases_table.setItem(r, 3, it)

                it = QTableWidgetItem(interaction)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.purchases_table.setItem(r, 4, it)

                it = QTableWidgetItem(dt_text)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.purchases_table.setItem(r, 5, it)

                _set_photo_cell(aboba, aboba.purchases_table, r, code, gen)

        # ===================== РЕКОМЕНДАЦИИ ИЗ EXCEL =====================
        topk = int(aboba.recs_topk.currentText().replace("Топ-", "").strip())

        recs_df = _load_recommendations_from_excel(aboba, mindbox_id, topk)
        if recs_df.empty:
            aboba.recs_table.setRowCount(0)
        else:
            aboba.recs_table.setRowCount(0)
            aboba.recs_table.clearContents()
            aboba.recs_table.setRowCount(len(recs_df))

            for r, row in enumerate(recs_df.itertuples(index=False)):
                code = str(row.КодНоменклатуры)
                name = str(row.НазваниеНоменклатуры)
                collection = str(getattr(row, "Коллекция", "") or "").strip()
                coef = str(getattr(row, "Коэффициент", "") or "").strip()
                conversion = str(getattr(row, "Конверсия", "") or "").strip()
                stock = str(getattr(row, "Остаток", "") or "").strip()

                if collection.lower() == "nan":
                    collection = ""

                if coef.lower() == "nan":
                    coef = ""

                if conversion.lower() == "nan":
                    conversion = ""

                if stock.lower() == "nan":
                    stock = ""

                it = QTableWidgetItem(code)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.recs_table.setItem(r, 1, it)

                it = QTableWidgetItem(name)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.recs_table.setItem(r, 2, it)

                it = QTableWidgetItem(collection)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.recs_table.setItem(r, 3, it)

                it = QTableWidgetItem(coef)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.recs_table.setItem(r, 4, it)

                it = QTableWidgetItem(conversion)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.recs_table.setItem(r, 5, it)

                it = QTableWidgetItem(stock)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                aboba.recs_table.setItem(r, 6, it)

                _set_photo_cell(aboba, aboba.recs_table, r, code, gen)

        # сообщение про пустую историю — оставим как раньше (если хочешь)
        if df.empty:
            set_status_error(aboba, "У выбранного клиента взаимодействий не найдено")
            aboba.status_label.repaint()
            aboba.status_icon.repaint()
            QApplication.processEvents()
            schedule_status_reset(aboba, 5)

            show_custom_message(aboba, "Ошибка", "У выбранного клиента взаимодействий не найдено",
                                "Картинки/Неудача.png")

        set_status_ok(aboba, "Данные успешно получены")
        schedule_status_reset(aboba, 5)

    except Exception as e:

        set_status_error(aboba, "Не удалось загрузить данные")
        aboba.status_label.repaint()
        aboba.status_icon.repaint()
        QApplication.processEvents()
        schedule_status_reset(aboba, 5)

        show_custom_message(aboba, "Ошибка", f"Не удалось загрузить данные:\n{e}",
                            "Картинки/Неудача.png")


# -------------------------------------------ОЧИСТКА ДАННЫХ КЛИЕНТА НА ФОРМЕ----------------------------------------
def _clear_client_info_panel(aboba) -> None:
    # если вдруг панель ещё не создана
    for name in ["le_card", "le_mb", "le_email", "le_phone", "le_fio", "le_gender", "le_age", "le_agegrp"]:
        w = getattr(aboba, name, None)
        if isinstance(w, QLineEdit):
            w.setText("")


# -------------------------------------------ПОЛУЧЕНИЕ MINDBOXID ПО ДРУГОМУ ИДЕНТИФИКАТОРУ--------------------------
def _resolve_mindbox_ids(field_ui: str, value: str):
    field_map = {
        "MindboxID": "MindboxID",
        "Дисконтная карта": "ДисконтнаяКарта",
        "Почта": "Почта",
        "Телефон": "Телефон",
    }

    col = field_map.get(field_ui)
    if not col:
        return []

    # Если ищем по MindboxID — просто возвращаем его (после чистки)
    def clean(s: str) -> str:
        s = str(s).strip()
        s = re.sub(r"\.0$", "", s)  # убрать хвост ".0"
        return s

    value_clean = clean(value)

    if col == "MindboxID":
        return [value_clean]

    # нормализация телефона/почты
    if col == "Телефон":
        value_clean = re.sub(r"\D+", "", value_clean)  # только цифры
    if col == "Почта":
        value_clean = value_clean.lower()

    orders_path = os.path.join(os.getcwd(), "ВходныеДанные", "Заказы.csv")
    if not os.path.isfile(orders_path):
        return []

    # читаем только нужные колонки
    usecols = ["MindboxID", col]
    df = pd.read_csv(orders_path, sep="|", encoding="utf-8-sig", dtype=str, usecols=lambda c: c in usecols)

    if "MindboxID" not in df.columns or col not in df.columns:
        return []

    df["MindboxID"] = df["MindboxID"].map(clean)
    s = df[col].fillna("").map(clean)

    if col == "Телефон":
        s = s.map(lambda x: re.sub(r"\D+", "", x))
    if col == "Почта":
        s = s.str.lower()

    hits = df.loc[s == value_clean, "MindboxID"].dropna().unique().tolist()
    return hits


# -------------------------------------------ПОЛУЧЕНИЕ ДАННЫХ КЛИЕНТА НА ФОРМЕ--------------------------------------
def _load_client_info(mindbox_id: str) -> dict:
    orders_path = os.path.join(os.getcwd(), "ВходныеДанные", "Заказы.csv")
    if not os.path.isfile(orders_path):
        return {}

    need = [
        "MindboxID", "ДисконтнаяКарта", "Почта", "Телефон",
        "ФИО", "ПолКлиента", "Возраст", "ВозрастнаяГруппа",
    ]

    df = pd.read_csv(
        orders_path, sep="|", encoding="utf-8-sig", dtype=str,
        usecols=lambda c: c in set(need)
    )
    if df.empty or "MindboxID" not in df.columns:
        return {}

    def clean(s: str) -> str:
        s = str(s).strip()
        s = re.sub(r"\.0$", "", s)
        return s

    mb = clean(mindbox_id)
    df["MindboxID"] = df["MindboxID"].map(clean)
    df = df[df["MindboxID"] == mb]
    if df.empty:
        return {}

    out = {}
    for col in need:
        if col not in df.columns:
            out[col] = ""
            continue

        s = df[col].fillna("").map(clean)

        if col == "Почта":
            s = s.str.lower()
        if col == "Телефон":
            s = s.map(lambda x: re.sub(r"\D+", "", x))

        val = ""
        for x in s.tolist():
            if x:
                val = x
                break
        out[col] = val

    return out


def _fill_client_info_panel(aboba, info: dict) -> None:
    def g(k: str) -> str:
        return str(info.get(k, "") or "").strip()

    if not info:
        _clear_client_info_panel(aboba)
        return

    # 1 колонка
    aboba.le_card.setText(g("ДисконтнаяКарта"))
    aboba.le_mb.setText(g("MindboxID"))

    # 2 колонка
    aboba.le_email.setText(g("Почта"))
    aboba.le_phone.setText(g("Телефон"))

    # 3 колонка
    aboba.le_fio.setText(g("ФИО"))
    aboba.le_gender.setText(g("ПолКлиента"))

    # 4 колонка
    aboba.le_age.setText(g("Возраст"))
    aboba.le_agegrp.setText(g("ВозрастнаяГруппа"))

    # курсор в начало
    for le in (
            aboba.le_card, aboba.le_mb,
            aboba.le_email, aboba.le_phone,
            aboba.le_fio, aboba.le_gender,
            aboba.le_age, aboba.le_agegrp
    ):
        le.setCursorPosition(0)


# -------------------------------------------ЧТЕНИЕ И СБОР ВСЕХ ВЗАИМОДЕЙСТВИЙ КЛИЕНТА------------------------------
def _load_client_interactions(aboba, mindbox_id: str):
    data_dir = os.path.join(os.getcwd(), "ВходныеДанные")
    paths = {
        "Покупка": os.path.join(data_dir, "Заказы.csv"),
        "Просмотр": os.path.join(data_dir, "Просмотры.csv"),
        "Избранное": os.path.join(data_dir, "Избранное.csv"),
    }

    result_cols = [
        "КодНоменклатуры",
        "НазваниеНоменклатуры",
        "Коллекция",
        "Взаимодействие",
        "ДатаВзаимодействия",
    ]

    frames = []

    for label, path in paths.items():
        if not os.path.isfile(path):
            continue

        base_cols = ["MindboxID", "КодНоменклатуры", "Дата"]
        extra = ["ТипТовара"] if label == "Просмотр" else []
        need = set(base_cols + extra)

        df = pd.read_csv(
            path,
            sep="|",
            encoding="utf-8-sig",
            dtype=str,
            usecols=lambda c: c in need
        )

        if df.empty or "MindboxID" not in df.columns or "КодНоменклатуры" not in df.columns:
            continue

        df["MindboxID"] = (
            df["MindboxID"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        df = df[df["MindboxID"] == str(mindbox_id).strip()]
        if df.empty:
            continue

        if label == "Просмотр" and "ТипТовара" in df.columns:
            df = df[df["ТипТовара"] == "Номенклатура"]
            if df.empty:
                continue

        cols = ["КодНоменклатуры"]
        if "Дата" in df.columns:
            cols.append("Дата")

        part = df[cols].copy()

        part["КодНоменклатуры"] = (
            part["КодНоменклатуры"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        part["Взаимодействие"] = label
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=result_cols)

    out = pd.concat(frames, ignore_index=True)

    # Дата: парсим устойчиво, даже если в файлах смешанные форматы дат
    if "Дата" in out.columns:
        raw_dates = (
            out["Дата"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        try:
            parsed_dates = pd.to_datetime(
                raw_dates,
                errors="coerce",
                dayfirst=True,
                format="mixed"
            )
        except TypeError:
            parsed_dates = pd.to_datetime(
                raw_dates,
                errors="coerce",
                dayfirst=True
            )

        for fmt in [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%d.%m.%Y",
            "%d.%m.%Y %H:%M:%S",
        ]:
            mask = parsed_dates.isna() & raw_dates.ne("")
            if mask.any():
                parsed_dates.loc[mask] = pd.to_datetime(
                    raw_dates.loc[mask],
                    errors="coerce",
                    format=fmt
                )

        out["_dt"] = parsed_dates.dt.normalize()
        out = out.sort_values("_dt", ascending=False, na_position="last")
    else:
        out["_dt"] = pd.NaT

    # Название номенклатуры
    try:
        _ensure_item_name_map(aboba)
        name_map = getattr(aboba, "_name_by_code", {}) or {}
    except Exception:
        name_map = {}

    out["НазваниеНоменклатуры"] = (
        out["КодНоменклатуры"]
        .map(name_map)
        .fillna("")
    )

    # Коллекция номенклатуры
    try:
        _ensure_item_collection_map(aboba)
        collection_map = getattr(aboba, "_collection_by_code", {}) or {}
    except Exception:
        collection_map = {}

    out["Коллекция"] = (
        out["КодНоменклатуры"]
        .map(collection_map)
        .fillna("")
    )

    def _fmt_dt(x):
        if pd.isna(x):
            return ""
        if x.hour == 0 and x.minute == 0 and x.second == 0:
            return x.strftime("%d.%m.%Y")
        return x.strftime("%d.%m.%Y %H:%M:%S")

    out["ДатаВзаимодействия"] = out["_dt"].map(_fmt_dt)

    # Удаляем дубли взаимодействий
    # В истории одна и та же покупка/просмотр/избранное не должны отображаться несколько раз,
    # если совпадают товар, тип взаимодействия и дата.
    dedup_cols = [
        "КодНоменклатуры",
        "Взаимодействие",
        "ДатаВзаимодействия",
    ]

    out = out.drop_duplicates(subset=dedup_cols, keep="first").copy()

    # Гарантируем наличие всех колонок перед return
    for col in result_cols:
        if col not in out.columns:
            out[col] = ""

    return out[result_cols]


# -------------------------------------------ПОЛУЧАЕМ НАЗВАНИЕ НОМЕНКЛАТУРЫ ПО КОДУ-------------------------------------
def _ensure_item_name_map(aboba):
    if aboba._name_by_code is not None:
        return

    aboba._name_by_code = {}
    nom_path = os.path.join(os.getcwd(), "ВходныеДанные", "Номенклатура.csv")
    if not os.path.isfile(nom_path):
        return

    df = pd.read_csv(nom_path, sep="|", encoding="utf-8-sig", dtype=str)
    if "КодНоменклатуры" not in df.columns:
        return

    primary_col = "НазваниеНаСайте"
    fallback_col = "Номенклатура"

    # если нет ни одной колонки с названием — выходим
    if primary_col not in df.columns and fallback_col not in df.columns:
        return

    df["КодНоменклатуры"] = (
        df["КодНоменклатуры"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    # аккуратно чистим обе колонки (если есть)
    if primary_col in df.columns:
        df[primary_col] = df[primary_col].astype("string").fillna("").str.strip()
    else:
        df[primary_col] = ""

    if fallback_col in df.columns:
        df[fallback_col] = df[fallback_col].astype("string").fillna("").str.strip()
    else:
        df[fallback_col] = ""

    # итоговое имя: если НазваниеНаСайте пустое -> берём Номенклатура
    df["_name_final"] = df[primary_col]
    df.loc[df["_name_final"].eq(""), "_name_final"] = df.loc[df["_name_final"].eq(""), fallback_col]

    aboba._name_by_code = (
        df.dropna(subset=["КодНоменклатуры"])
        .drop_duplicates(subset=["КодНоменклатуры"], keep="first")
        .set_index("КодНоменклатуры")["_name_final"]
        .to_dict()
    )


def _format_stock_value_ui(v) -> str:
    if v is None:
        return "0"

    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null", "<na>", "-"):
        return "0"

    s = s.replace(",", ".").replace(" ", "")

    try:
        num = pd.to_numeric(s, errors="coerce")
        if pd.isna(num):
            return "0"
        return str(int(round(float(num))))
    except Exception:
        return "0"


def _ensure_item_stock_map(aboba):
    if getattr(aboba, "_stock_by_code", None) is not None:
        return

    aboba._stock_by_code = {}

    nom_path = os.path.join(os.getcwd(), "ВходныеДанные", "Номенклатура.csv")
    if not os.path.isfile(nom_path):
        return

    df = pd.read_csv(nom_path, sep="|", encoding="utf-8-sig", dtype=str)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if "КодНоменклатуры" not in df.columns or "Остаток" not in df.columns:
        return

    df["КодНоменклатуры"] = (
        df["КодНоменклатуры"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["Остаток"] = df["Остаток"].map(_format_stock_value_ui)

    df = df[df["КодНоменклатуры"] != ""]
    df = df.drop_duplicates("КодНоменклатуры", keep="last")

    aboba._stock_by_code = df.set_index("КодНоменклатуры")["Остаток"].to_dict()


def _ensure_item_collection_map(aboba):
    if getattr(aboba, "_collection_by_code", None) is not None:
        return

    aboba._collection_by_code = {}

    nom_path = os.path.join(os.getcwd(), "ВходныеДанные", "Номенклатура.csv")
    if not os.path.isfile(nom_path):
        return

    df = pd.read_csv(nom_path, sep="|", encoding="utf-8-sig", dtype=str)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if "КодНоменклатуры" not in df.columns or "Коллекция" not in df.columns:
        return

    df["КодНоменклатуры"] = (
        df["КодНоменклатуры"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["Коллекция"] = (
        df["Коллекция"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    df = df[df["КодНоменклатуры"] != ""]
    df = df.drop_duplicates("КодНоменклатуры", keep="last")

    aboba._collection_by_code = df.set_index("КодНоменклатуры")["Коллекция"].to_dict()


def _format_conversion_value_ui(v) -> str:
    """Форматирует значение конверсии из Excel как процент для таблицы."""
    if v is None:
        return ""

    s = str(v).strip().replace("%", "").replace(",", ".")
    if not s or s.lower() in ("nan", "none", "null", "<na>", "-"):
        return ""

    try:
        num = float(s)
    except Exception:
        return ""

    text = f"{num:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") + "%"


# -------------------------------------------РЕКОМЕНДАЦИИ ИЗ EXCEL------------------------------------------------------
def _load_recommendations_from_excel(aboba, mindbox_id: str, topk: int) -> pd.DataFrame:
    path = os.path.join(os.getcwd(), "Модель", "Рекомендации.xlsx")
    empty_cols = ["КодНоменклатуры", "НазваниеНоменклатуры", "Коллекция", "Коэффициент", "Конверсия", "Остаток"]

    if not os.path.isfile(path):
        return pd.DataFrame(columns=empty_cols)

    df = _get_recommendations_excel_cache(aboba)
    if df is None or df.empty:
        return pd.DataFrame(columns=empty_cols)

    # чистим
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        df[c] = (
            df[c]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    mb = str(mindbox_id).strip()

    # 1) ищем клиента по MindboxID
    client_rows = pd.DataFrame()
    if "MindboxID" in df.columns:
        client_rows = df[df["MindboxID"] == mb].copy()

    # 2) fallback по дисконтной карте
    if client_rows.empty and "ДисконтнаяКарта" in df.columns:
        cards = _get_discount_cards_for_mindbox(mb)
        if cards:
            client_rows = df[df["ДисконтнаяКарта"].isin(cards)].copy()

    if client_rows.empty:
        return pd.DataFrame(columns=empty_cols)

    row = client_rows.iloc[0]
    recs = []

    # A) длинный формат
    if "КодНоменклатуры" in client_rows.columns:
        long_df = client_rows.copy()

        coef_col = None
        for cand in ["Коэффициент", "Score", "score"]:
            if cand in long_df.columns:
                coef_col = cand
                break
        if coef_col is None:
            coef_col = "Коэффициент"
            long_df[coef_col] = ""

        conversion_col = None
        for cand in ["Конверсия", "Conversion", "conversion"]:
            if cand in long_df.columns:
                conversion_col = cand
                break
        if conversion_col is None:
            conversion_col = "Конверсия"
            long_df[conversion_col] = ""

        stock_col = None
        for cand in ["Остаток", "ОстатокТовара", "Stock", "stock"]:
            if cand in long_df.columns:
                stock_col = cand
                break
        if stock_col is None:
            stock_col = "Остаток"
            long_df[stock_col] = ""

        collection_col = None
        for cand in ["Коллекция", "Collection", "collection"]:
            if cand in long_df.columns:
                collection_col = cand
                break
        if collection_col is None:
            collection_col = "Коллекция"
            long_df[collection_col] = ""

        long_df = long_df[["КодНоменклатуры", collection_col, coef_col, conversion_col, stock_col]].rename(
            columns={
                collection_col: "Коллекция",
                coef_col: "Коэффициент",
                conversion_col: "Конверсия",
                stock_col: "Остаток",
            }
        )

        long_df = long_df[long_df["КодНоменклатуры"].astype(str).str.len() > 0]
        long_df = long_df.head(topk)

        recs = list(
            zip(
                long_df["КодНоменклатуры"].tolist(),
                long_df["Коллекция"].tolist(),
                long_df["Коэффициент"].tolist(),
                long_df["Конверсия"].tolist(),
                long_df["Остаток"].tolist(),
            )
        )

    # B) широкий формат: КодНоменклатуры_1, Коллекция_1, Коэффициент_1, Остаток_1, ...
    else:
        import re as _re

        code_cols = {}
        collection_cols = {}
        coef_cols = {}
        conversion_cols = {}
        stock_cols = {}

        for c in df.columns:
            c_str = str(c).strip()

            m = _re.search(r"(\d+)$", c_str.replace(" ", ""))
            if not m:
                continue

            idx = int(m.group(1))
            base = _re.sub(r"[_\s-]*\d+$", "", c_str.lower()).replace(" ", "")

            if (
                    "кодноменклатуры" in base
                    or ("код" in base and "номенклат" in base)
                    or "item" in base
                    or "recommend" in base
                    or "рекомендац" in base
            ):
                code_cols[idx] = c

            if "коллекция" in base or "collection" in base:
                collection_cols[idx] = c

            if "коэффициент" in base or "коэфф" in base or "score" in base or "вес" in base:
                coef_cols[idx] = c

            if "конверсия" in base or "conversion" in base:
                conversion_cols[idx] = c

            if "остаток" in base or "stock" in base:
                stock_cols[idx] = c

        if not code_cols:
            for c in df.columns:
                m = _re.search(r"(\d+)$", str(c).replace(" ", ""))
                if not m:
                    continue
                idx = int(m.group(1))
                if "рекоменда" in str(c).lower():
                    code_cols[idx] = c

        for i in sorted(code_cols.keys()):
            if len(recs) >= topk:
                break

            code = str(row.get(code_cols[i], "")).strip()
            if not code:
                continue

            collection = str(row.get(collection_cols[i], "")).strip() if i in collection_cols else ""
            coef = str(row.get(coef_cols[i], "")).strip() if i in coef_cols else ""
            conversion = str(row.get(conversion_cols[i], "")).strip() if i in conversion_cols else ""
            stock = str(row.get(stock_cols[i], "")).strip() if i in stock_cols else ""

            if collection.lower() == "nan":
                collection = ""

            if coef.lower() == "nan":
                coef = ""

            if conversion.lower() == "nan":
                conversion = ""

            if stock.lower() == "nan":
                stock = ""

            recs.append((code, collection, coef, conversion, stock))

    if not recs:
        return pd.DataFrame(columns=empty_cols)

    out = pd.DataFrame(
        recs,
        columns=["КодНоменклатуры", "Коллекция", "Коэффициент", "Конверсия", "Остаток"]
    )

    out["КодНоменклатуры"] = (
        out["КодНоменклатуры"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    out["Коллекция"] = (
        out["Коллекция"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    out["Конверсия"] = out["Конверсия"].map(_format_conversion_value_ui)
    out["Остаток"] = out["Остаток"].map(_format_stock_value_ui)

    # подтягиваем названия
    _ensure_item_name_map(aboba)
    name_map = aboba._name_by_code or {}
    out["НазваниеНоменклатуры"] = out["КодНоменклатуры"].map(name_map).fillna("")

    # fallback по коллекциям из Номенклатура.csv, если в Excel коллекция пустая
    _ensure_item_collection_map(aboba)
    collection_map = getattr(aboba, "_collection_by_code", {}) or {}

    mask_empty_collection = out["Коллекция"].astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
    out.loc[mask_empty_collection, "Коллекция"] = (
        out.loc[mask_empty_collection, "КодНоменклатуры"]
        .map(collection_map)
        .fillna("")
    )

    # fallback по остаткам из Номенклатура.csv, если в Excel остаток пустой
    _ensure_item_stock_map(aboba)
    stock_map = getattr(aboba, "_stock_by_code", {}) or {}

    mask_empty_stock = out["Остаток"].astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
    out.loc[mask_empty_stock, "Остаток"] = (
        out.loc[mask_empty_stock, "КодНоменклатуры"]
        .map(stock_map)
        .fillna("0")
        .map(_format_stock_value_ui)
    )

    return out[["КодНоменклатуры", "НазваниеНоменклатуры", "Коллекция", "Коэффициент", "Конверсия", "Остаток"]]


# -------------------------------------------КЭШИРУЕМ EXCEL С РЕКОМЕНДАЦИЯМИ--------------------------------------------
def _get_recommendations_excel_cache(aboba) -> pd.DataFrame:
    path = os.path.join(os.getcwd(), "Модель", "Рекомендации.xlsx")

    if not os.path.isfile(path):
        return pd.DataFrame()

    mtime = os.path.getmtime(path)

    if (
        getattr(aboba, "_recs_excel_cache", None) is not None
        and getattr(aboba, "_recs_excel_mtime", None) == mtime
    ):
        return aboba._recs_excel_cache

    df = pd.read_excel(path, dtype=str)

    aboba._recs_excel_cache = df
    aboba._recs_excel_mtime = mtime

    return df


# -------------------------------------------ДИСКОНТНАЯ КАРТА ПО MINDBOX ID---------------------------------------------
def _get_discount_cards_for_mindbox(mindbox_id: str) -> list[str]:
    orders_path = os.path.join(os.getcwd(), "ВходныеДанные", "Заказы.csv")
    if not os.path.isfile(orders_path):
        return []

    usecols = ["MindboxID", "ДисконтнаяКарта"]
    df = pd.read_csv(orders_path, sep="|", encoding="utf-8-sig", dtype=str, usecols=lambda c: c in usecols)
    if df.empty or "MindboxID" not in df.columns or "ДисконтнаяКарта" not in df.columns:
        return []

    def clean(s: str) -> str:
        s = str(s).strip()
        s = re.sub(r"\.0$", "", s)
        return s

    df["MindboxID"] = df["MindboxID"].map(clean)
    df["ДисконтнаяКарта"] = df["ДисконтнаяКарта"].map(clean)

    cards = df.loc[df["MindboxID"] == clean(mindbox_id), "ДисконтнаяКарта"].dropna().unique().tolist()
    cards = [c for c in cards if c]
    return cards


# -------------------------------------------ВЫГРУЗИТЬ РЕКОМЕНДАЦИИ В EXCEL-----------------------------------------
def export_recommendations_to_excel(aboba):

    # Количество клиентов берём с вкладки «Обработка датасета»
    max_export_users_widget = getattr(
        aboba,
        "max_export_users_input",
        None
    )

    if max_export_users_widget is not None:
        max_export_users = int(max_export_users_widget.value())
    else:
        max_export_users = 1000

    # Виды номенклатуры, которые разрешены в итоговой выгрузке
    export_kind_widget = getattr(
        aboba,
        "export_kind_filter",
        None
    )

    if export_kind_widget is not None:
        export_item_kinds = get_selected_list_values(
            export_kind_widget
        )
    else:
        export_item_kinds = []

    if max_export_users == 0:
        status_text = "Идёт формирование рекомендаций для всех клиентов..."
    else:
        status_text = (
            f"Идёт формирование рекомендаций для "
            f"{max_export_users:,} клиентов..."
        ).replace(",", " ")

    set_status_processing(aboba, status_text)

    # чтобы статус и иконка успели отрисоваться до тяжёлой операции
    aboba.status_label.repaint()
    aboba.status_icon.repaint()
    QApplication.processEvents()

    try:
        BPRMF_module.export_recommendations_excel(
            out_xlsx="Модель/Рекомендации.xlsx",
            k=10,
            include_item_names=True,
            include_scores=True,
            filter_seen=True,
            device_str="cuda",
            include_discount_card=True,
            out_csv_format1="Модель/InternetMagazin.csv",
            out_csv_kanzler_ml="Модель/Mindbox.csv",
            max_export_users=max_export_users,
            export_item_kinds=export_item_kinds,
        )

        aboba._recs_excel_cache = None
        aboba._recs_excel_mtime = None

        set_status_ok(aboba, "Формирование рекомендаций завершено")
        schedule_status_reset(aboba, 5)

    except Exception as e:
        set_status_error(aboba, "Ошибка формирования рекомендаций")
        schedule_status_reset(aboba, 5)

        show_custom_message(
            aboba,
            "Ошибка",
            f"Не удалось выгрузить рекомендации:\n{e}",
            "Картинки/Неудача.png"
        )

        QApplication.processEvents()