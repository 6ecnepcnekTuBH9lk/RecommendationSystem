import os
import sys
import json
import time

import pandas as pd
import shutil
from Application.model import BPRMF as BPRMF_module
from functools import partial
from PyQt6.QtCore import Qt, QSize, QProcess
from PyQt6.QtGui import QIcon, QTextCursor
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QLineEdit, QComboBox, QSpinBox,
                             QDoubleSpinBox, QTextEdit, QFrame, QAbstractSpinBox, QSizePolicy)

from Application.settings.settings_and_filter import (update_filter_summary, get_selected_list_values)

from Application.settings.set_status import (set_status_processing, schedule_status_reset,
                                             set_status_error, set_status_ok)


# -------------------------------------------ВКЛАДКА ОБУЧЕНИЕ МОДЕЛИ------------------------------------------------
def create_train_model_widgets_tab(aboba):
    tab = QWidget()

    # Главный горизонтальный сплит
    root = QHBoxLayout(tab)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    # Левая панель
    left_wrap = QWidget()
    left_layout = QVBoxLayout(left_wrap)

    # Правая панель
    right_wrap = QWidget()
    right_layout = QVBoxLayout(right_wrap)
    right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    # Вертикальный разделитель
    separator = QFrame()
    separator.setObjectName("vSeparator")
    separator.setFixedWidth(1)
    separator.setFrameShape(QFrame.Shape.NoFrame)

    root.addWidget(left_wrap, 1)
    root.addWidget(separator)
    root.addWidget(right_wrap, 1)

    # -------------------- ЛЕВАЯ ЧАСТЬ --------------------

    # Заголовок
    aboba.heading_enter_parameter = QLabel("Входные параметры")
    aboba.heading_enter_parameter.setSizePolicy(aboba.heading_load_data.sizePolicy().Policy.Fixed,  # Фиксируем размер
                                                aboba.heading_load_data.sizePolicy().Policy.Fixed)  # по ширине и высоте
    aboba.heading_enter_parameter.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aboba.heading_enter_parameter.setStyleSheet("""
        QLabel {
            background-color: #FAFAFA;
            padding: 7px 65px;
            border-radius: 10px;
            border: 1px solid #C8C8C8;
            margin: 10px 0px;
        }
    """)
    left_layout.addWidget(aboba.heading_enter_parameter, alignment=Qt.AlignmentFlag.AlignHCenter)

    # -------------------- ПАРАМЕТРЫ --------------------
    form_wrap = QWidget()
    form_wrap.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    form_layout = QVBoxLayout(form_wrap)
    form_layout.setContentsMargins(0, 0, 0, 0)
    form_layout.setSpacing(7)

    LABEL_W = 400  # можно подобрать (чтобы все поля начинались по одной вертикали)

    def add_param(label_text: str, widget, stretch_after: bool = True):
        # строка: [Label][Widget]
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(0)

        lbl = QLabel(label_text)
        lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        lbl.setFixedWidth(LABEL_W)

        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        row_l.addWidget(lbl)
        row_l.addWidget(widget, 1)

        form_layout.addWidget(row_w)

        # аналог "пустой строки" из grid (делает немного воздуха между реквизитами)
        if stretch_after:
            form_layout.addSpacing(0)

    # -------------------- Создание виджетов --------------------

    # Параметры BPR-MF
    aboba.embedding_dim_input = QComboBox()
    aboba.embedding_dim_input.addItems(["16", "32", "64", "128", "256", "512",
                                       "1024", "2048", "4096", "8192", "16384"])
    aboba.embedding_dim_input.setCurrentText("128")

    aboba.epochs_input = QSpinBox()
    aboba.epochs_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.epochs_input.setRange(1, 10000)
    aboba.epochs_input.setSingleStep(10)
    aboba.epochs_input.setValue(200)

    aboba.batch_size_input = QComboBox()
    aboba.batch_size_input.addItems(["16", "32", "64", "128", "256", "512",
                                    "1024", "2048", "4096", "8192", "16384"])
    aboba.batch_size_input.setCurrentText("256")

    aboba.lr_input = QDoubleSpinBox()
    aboba.lr_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.lr_input.setRange(0, 1)
    aboba.lr_input.setDecimals(4)
    aboba.lr_input.setSingleStep(0.0001)
    aboba.lr_input.setValue(0.0003)

    aboba.weight_decay_input = QDoubleSpinBox()
    aboba.weight_decay_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.weight_decay_input.setRange(0, 1)
    aboba.weight_decay_input.setDecimals(4)
    aboba.weight_decay_input.setSingleStep(0.0001)
    aboba.weight_decay_input.setValue(0)

    aboba.bpr_reg_input = QDoubleSpinBox()
    aboba.bpr_reg_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.bpr_reg_input.setRange(0, 1)
    aboba.bpr_reg_input.setDecimals(4)
    aboba.bpr_reg_input.setSingleStep(0.0001)
    aboba.bpr_reg_input.setValue(0.0005)

    aboba.seed_input = QSpinBox()
    aboba.seed_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.seed_input.setRange(0, 10000)
    aboba.seed_input.setSingleStep(1)
    aboba.seed_input.setValue(42)

    # Остальные параметры
    aboba.w_purchase = QDoubleSpinBox()
    aboba.w_purchase.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.w_purchase.setRange(0, 10)
    aboba.w_purchase.setDecimals(1)
    aboba.w_purchase.setSingleStep(0.1)
    aboba.w_purchase.setValue(10.0)

    aboba.w_favorite = QDoubleSpinBox()
    aboba.w_favorite.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.w_favorite.setRange(0, 10)
    aboba.w_favorite.setDecimals(1)
    aboba.w_favorite.setSingleStep(0.1)
    aboba.w_favorite.setValue(2.0)

    aboba.w_view_item = QDoubleSpinBox()
    aboba.w_view_item.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.w_view_item.setRange(0, 10)
    aboba.w_view_item.setDecimals(1)
    aboba.w_view_item.setSingleStep(0.1)
    aboba.w_view_item.setValue(0.1)

    aboba.top_rec = QSpinBox()
    aboba.top_rec.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.top_rec.setRange(1, 100)
    aboba.top_rec.setSingleStep(1)
    aboba.top_rec.setValue(10)

    aboba.min_user_interactions_for_eval = QSpinBox()
    aboba.min_user_interactions_for_eval.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.min_user_interactions_for_eval.setRange(1, 100)
    aboba.min_user_interactions_for_eval.setSingleStep(1)
    aboba.min_user_interactions_for_eval.setValue(10)

    aboba.n_neg = QSpinBox()
    aboba.n_neg.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.n_neg.setRange(1, 100)
    aboba.n_neg.setSingleStep(1)
    aboba.n_neg.setValue(10)

    # -------------------- Early stopping --------------------
    aboba.early_stop_metric = QComboBox()
    aboba.early_stop_metric.addItems(["NDCG", "RECALL"])
    aboba.early_stop_metric.setCurrentText("NDCG")

    aboba.early_stop_patience = QSpinBox()
    aboba.early_stop_patience.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.early_stop_patience.setRange(0, 1000)
    aboba.early_stop_patience.setSingleStep(1)
    aboba.early_stop_patience.setValue(8)

    aboba.early_stop_min_delta = QDoubleSpinBox()
    aboba.early_stop_min_delta.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.early_stop_min_delta.setRange(0, 1)
    aboba.early_stop_min_delta.setDecimals(4)
    aboba.early_stop_min_delta.setSingleStep(0.0001)
    aboba.early_stop_min_delta.setValue(0.0005)

    aboba.early_stop_min_epochs = QSpinBox()
    aboba.early_stop_min_epochs.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.early_stop_min_epochs.setRange(0, 1000)
    aboba.early_stop_min_epochs.setSingleStep(5)
    aboba.early_stop_min_epochs.setValue(30)

    # -------------------- Признаки номенклатуры --------------------
    aboba.max_item_features_input = QSpinBox()
    aboba.max_item_features_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.max_item_features_input.setRange(1, 100)
    aboba.max_item_features_input.setSingleStep(1)
    aboba.max_item_features_input.setValue(32)

    aboba.feature_dropout_input = QDoubleSpinBox()
    aboba.feature_dropout_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.feature_dropout_input.setRange(0, 0.50)
    aboba.feature_dropout_input.setDecimals(2)
    aboba.feature_dropout_input.setSingleStep(0.05)
    aboba.feature_dropout_input.setValue(0.10)

    aboba.feature_scale_input = QDoubleSpinBox()
    aboba.feature_scale_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.feature_scale_input.setRange(0.0, 1.0)
    aboba.feature_scale_input.setDecimals(2)
    aboba.feature_scale_input.setSingleStep(0.05)
    aboba.feature_scale_input.setValue(0.20)

    aboba.feature_norm_input = QComboBox()
    aboba.feature_norm_input.addItems(["SUM", "MEAN", "SQRT"])
    aboba.feature_norm_input.setCurrentText("MEAN")

    aboba.feat_reg_mult_input = QDoubleSpinBox()
    aboba.feat_reg_mult_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    aboba.feat_reg_mult_input.setRange(0, 10.00)
    aboba.feat_reg_mult_input.setDecimals(2)
    aboba.feat_reg_mult_input.setSingleStep(0.10)
    aboba.feat_reg_mult_input.setValue(1.00)

    # -------------------- Добавляем строки --------------------
    add_param("Вес покупки:", aboba.w_purchase)
    add_param("Вес избранного:", aboba.w_favorite)
    add_param("Вес просмотра:", aboba.w_view_item)
    add_param("Количество рекомендаций:", aboba.top_rec)

    add_param("Количество эпох:", aboba.epochs_input)
    add_param("Скорость обучения:", aboba.lr_input)

    add_param("Количество отрицательных примеров:", aboba.n_neg)

    add_param("Инициализатор случайных чисел:", aboba.seed_input)
    add_param("Регуляризация BPR:", aboba.bpr_reg_input)
    add_param("Регуляризация L2:", aboba.weight_decay_input)

    add_param("Минимум действий для оценки:", aboba.min_user_interactions_for_eval)
    add_param("Размер обучающего пакета:", aboba.batch_size_input)
    add_param("Размерность векторов:", aboba.embedding_dim_input)

    add_param("Количество эпох без улучшения:", aboba.early_stop_patience)
    add_param("Минимальный прирост метрики:", aboba.early_stop_min_delta)
    add_param("Минимум эпох до остановки:", aboba.early_stop_min_epochs)
    add_param("Метрика для ранней остановки:", aboba.early_stop_metric)

    add_param("Максимальное число признаков номенклатуры:", aboba.max_item_features_input)
    add_param("Случайное отключение признаков номенклатуры:", aboba.feature_dropout_input)
    add_param("Вес признаков номенклатуры в модели:", aboba.feature_scale_input)
    add_param("Сила регуляризации признаков номенклатуры:", aboba.feat_reg_mult_input)
    add_param("Способ объединения признаков номенклатуры:", aboba.feature_norm_input, stretch_after=False)

    # чтобы лишняя высота уходила вниз (а не растягивала поля)
    form_layout.addStretch(1)

    # Добавляем растягиваемую форму в левый layout (она съедает свободную высоту)
    left_layout.addWidget(form_wrap, 1)

    # -------------------- Поля под формой (фиксированные по высоте) --------------------
    aboba.item_feature_cols_input = QLineEdit()
    aboba.item_feature_cols_input.setText(
        "Признаки номенклатуры: вид номенклатуры, вид ассортимента, марка, "
        "коллекция, сезон носки, пол, группа составов, категория на сайте, "
        "стилевая группа"
    )
    aboba.item_feature_cols_input.setCursorPosition(0)
    aboba.item_feature_cols_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    aboba.item_feature_cols_input.setCursor(Qt.CursorShape.IBeamCursor)
    aboba.item_feature_cols_input.setReadOnly(True)
    aboba.item_feature_cols_input.setStyleSheet("""QLineEdit { margin: 3px 0px 0px 0px; }""")
    left_layout.addWidget(aboba.item_feature_cols_input, 0)

    # Копия поля со вкладки "Обработка датасета" (обновляется из update_filter_summary)
    aboba.train_filter_summary = QLineEdit()
    aboba.train_filter_summary.setReadOnly(True)
    aboba.train_filter_summary.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    aboba.train_filter_summary.setCursor(Qt.CursorShape.IBeamCursor)
    aboba.train_filter_summary.setPlaceholderText("Отбор не установлен")
    aboba.train_filter_summary.setStyleSheet("""QLineEdit { margin: 5px 0px 0px 0px; }""")
    left_layout.addWidget(aboba.train_filter_summary, 0)  # стоит перед кнопками
    update_filter_summary(aboba)

    # --- Кнопки (две штуки снизу) ---
    btns = QHBoxLayout()
    btns.setSpacing(10)

    aboba.btn_settings = QPushButton(QIcon("Картинки/СтандартныеНастройки.png"), " Стандартные настройки")
    aboba.btn_settings.setIconSize(QSize(17, 17))
    aboba.btn_settings.clicked.connect(lambda: standart_settigs(aboba))
    aboba.btn_settings.setStyleSheet("""QPushButton { margin: 5px 0px 0px 0px; }""")

    aboba.start_train = QPushButton(QIcon("Картинки/НачатьОбучение.png"), " Начать обучение")
    aboba.start_train.setIconSize(QSize(17, 17))
    aboba.start_train.clicked.connect(lambda: start_training_process(aboba))
    aboba.start_train.setStyleSheet("""QPushButton { margin: 5px 0px 0px 0px; }""")

    btns.addWidget(aboba.btn_settings)
    btns.addWidget(aboba.start_train)

    left_layout.addLayout(btns, 0)

    # -------------------- ПРАВАЯ ЧАСТЬ --------------------
    # Заголовок (новый текст)
    aboba.label_69 = QLabel("Процесс обучения")
    aboba.label_69.setSizePolicy(aboba.heading_load_data.sizePolicy().Policy.Fixed,  # Фиксируем размер
                                 aboba.heading_load_data.sizePolicy().Policy.Fixed)  # по ширине и высоте
    aboba.label_69.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aboba.label_69.setStyleSheet("""
                QLabel {
                    background-color: #FAFAFA;
                    padding: 7px 65px;
                    border-radius: 10px;
                    border: 1px solid #C8C8C8;
                    margin: 10px 0px 10px 0px;
                }
            """)
    right_layout.addWidget(aboba.label_69, alignment=Qt.AlignmentFlag.AlignHCenter)

    # Поле для логов обучения (пока просто вывод)
    aboba.train_log = QTextEdit()
    aboba.train_log.setReadOnly(True)
    aboba.train_log.setPlaceholderText("Логи обучения будут отображаться здесь...")
    aboba.train_log.setStyleSheet("""QTextEdit {margin: 0px 0px 0px 0px;}""")
    right_layout.addWidget(aboba.train_log, 1)

    # Название вкладки
    aboba.tabs.addTab(tab, "Обучение модели")
    
    
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# -------------------------------------------СТАНДАРТНЫЕ НАСТРОЙКИ------------------------------------------------------
def standart_settigs(aboba):
    # чтобы не дергались лишние обработчики, если они есть
    widgets = [
        aboba.embedding_dim_input,
        aboba.epochs_input,
        aboba.batch_size_input,
        aboba.lr_input,
        aboba.n_neg,
        aboba.weight_decay_input,
        aboba.bpr_reg_input,
        aboba.seed_input,
        aboba.w_view_item,
        aboba.w_favorite,
        aboba.w_purchase,
        aboba.top_rec,
        aboba.min_user_interactions_for_eval,
        aboba.early_stop_metric,
        aboba.early_stop_patience,
        aboba.early_stop_min_delta,
        aboba.early_stop_min_epochs,
        aboba.max_item_features_input,
        aboba.feature_dropout_input,
        aboba.feature_scale_input,
        aboba.feat_reg_mult_input,
        aboba.feature_norm_input,
    ]

    for w in widgets:
        try:
            w.blockSignals(True)
        except Exception:
            pass

    aboba.embedding_dim_input.setCurrentText("128")
    aboba.epochs_input.setValue(200)
    aboba.batch_size_input.setCurrentText("256")
    aboba.lr_input.setValue(0.0003)
    aboba.n_neg.setValue(10)
    aboba.weight_decay_input.setValue(0.0)
    aboba.bpr_reg_input.setValue(0.0005)
    aboba.seed_input.setValue(42)
    aboba.w_view_item.setValue(0.1)
    aboba.w_favorite.setValue(2.0)
    aboba.w_purchase.setValue(10.0)
    aboba.top_rec.setValue(10)
    aboba.min_user_interactions_for_eval.setValue(10)
    aboba.early_stop_metric.setCurrentText("NDCG")
    aboba.early_stop_patience.setValue(8)
    aboba.early_stop_min_delta.setValue(0.0005)
    aboba.early_stop_min_epochs.setValue(30)
    aboba.max_item_features_input.setValue(32)
    aboba.feature_dropout_input.setValue(0.10)
    aboba.feature_scale_input.setValue(0.20)
    aboba.feat_reg_mult_input.setValue(1.00)
    aboba.feature_norm_input.setCurrentText("MEAN")

    for w in widgets:
        try:
            w.blockSignals(False)
        except Exception:
            pass

    set_status_ok(aboba, "Применены стандартные настройки параметров")
    schedule_status_reset(aboba, 5)
    

# -------------------------------------------ОБУЧЕНИЕ МОДЕЛИ------------------------------------------------------------
def start_training_process(aboba):

    # защита от повторного запуска
    if hasattr(aboba, "train_proc") and aboba.train_proc is not None:
        if aboba.train_proc.state() != QProcess.ProcessState.NotRunning:
            set_status_error(aboba, "Обучение уже запущено")
            schedule_status_reset(aboba, 5)
            return

    set_status_processing(aboba, "Идёт обучение модели...")
    aboba.start_train.setEnabled(False)

    aboba.status_label.repaint()
    aboba.status_icon.repaint()
    QApplication.processEvents()

    # очищаем лог
    if hasattr(aboba, "train_log"):
        aboba.train_log.clear()
        aboba.train_log.append("Запуск обучения нейросетевой модели BPR-MF...\n")

        filt_le = getattr(aboba, "train_filter_summary", None)
        if filt_le is not None:
            aboba.train_log.append(filt_le.text() + "\n\n")
        else:
            # запасной вариант, если поле на вкладке обучения ещё не создано
            le = getattr(aboba, "filter_summary", None)
            if le is not None:
                aboba.train_log.append(le.text() + "\n\n")

    py = sys.executable
    script = os.path.abspath(BPRMF_module.__file__)

    aboba.train_proc = QProcess(aboba)
    aboba.train_proc.setProgram(py)
    # --- build train config from UI and pass to trainer ---
    try:

        # актуализируем строку отбора (чтобы поле на вкладке обучения тоже обновилось)
        update_filter_summary(aboba)

        # готовим папку данных для обучения с учётом отбора
        train_data_dir = _prepare_training_data_dir(aboba)

        cfg_data = {
            "data_dir": train_data_dir,
            "filter_summary": getattr(aboba, "filter_summary", None).text()
            if hasattr(aboba, "filter_summary") else "",

            # веса implicit
            "w_view_item": float(aboba.w_view_item.value()),
            "w_favorite": float(aboba.w_favorite.value()),
            "w_purchase": float(aboba.w_purchase.value()),

            # BPR-MF
            "embedding_dim": int(aboba.embedding_dim_input.currentText()),
            "epochs": int(aboba.epochs_input.value()),
            "batch_size": int(aboba.batch_size_input.currentText()),
            "lr": float(aboba.lr_input.value()),
            "n_neg": int(aboba.n_neg.value()),
            "weight_decay": float(aboba.weight_decay_input.value()),
            "bpr_reg": float(aboba.bpr_reg_input.value()),
            "seed": int(aboba.seed_input.value()),

            # eval
            "topk": int(aboba.top_rec.value()),
            "min_user_interactions_for_eval": int(aboba.min_user_interactions_for_eval.value()),

            # early stopping
            "early_stop_metric": str(aboba.early_stop_metric.currentText()).strip().lower(),
            "early_stop_patience": int(aboba.early_stop_patience.value()),
            "early_stop_min_delta": float(aboba.early_stop_min_delta.value()),
            "early_stop_min_epochs": int(aboba.early_stop_min_epochs.value()),

            # -------- признаки номенклатуры --------
            "max_item_features": int(aboba.max_item_features_input.value()),
            "feature_dropout": float(aboba.feature_dropout_input.value()),
            "feature_scale": float(aboba.feature_scale_input.value()),
            "feature_norm": str(aboba.feature_norm_input.currentText()).strip().lower(),
            "feat_reg_mult": float(aboba.feat_reg_mult_input.value()),
        }

        cfg_dir = os.path.join(os.getcwd(), "Настройки")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, "train_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        if hasattr(aboba, "train_log"):
            aboba.train_log.append(f"Не удалось сформировать перечень входных параметров обучения из формы: {e}\n")
        set_status_error(aboba, "Не удалось подготовить параметры обучения")
        schedule_status_reset(aboba, 5)
        aboba.start_train.setEnabled(True)
        return

    args = ["-u", script, "--train"]
    args += ["--config", cfg_path]

    aboba.train_proc.setArguments(args)

    # чтобы stdout+stderr шли одним потоком
    aboba.train_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

    aboba.train_proc.readyReadStandardOutput.connect(partial(_on_train_output, aboba))
    aboba.train_proc.finished.connect(partial(_on_train_finished, aboba))

    aboba.train_proc.setWorkingDirectory(os.getcwd())
    aboba.train_proc.start()

    if not aboba.train_proc.waitForStarted(2000):
        set_status_error(aboba, "Не удалось запустить обучение")
        aboba.start_train.setEnabled(True)
        if hasattr(aboba, "train_log"):
            aboba.train_log.append("Не удалось запустить обучение модели\n")


def _get_store_city_map(aboba) -> dict:
    m = getattr(aboba, "_store_city_map", None)
    if isinstance(m, dict) and m:
        return m

    # запасной вариант: из JSON настроек
    path = os.path.join(os.getcwd(), "Настройки", "filter_settings.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(data.get("store_city_map", {}) or {})

    return {}


def _enrich_orders_with_city_and_weather(aboba, orders_df: pd.DataFrame, weather_path: str) -> pd.DataFrame:
    df = orders_df.copy()

    # --- Магазин -> Город ---
    store_city = _get_store_city_map(aboba)
    if "Магазин" in df.columns:
        df["Магазин"] = df["Магазин"].astype(str).str.strip()
        df["Город"] = df["Магазин"].map(lambda s: store_city.get(s, pd.NA))
    else:
        df["Город"] = pd.NA

    # --- Город + Дата -> Погода ---
    for c in ("ПогодныеУсловия", "СредняяТемпература", "КоличествоОсадков"):
        if c not in df.columns:
            df[c] = pd.NA

    if not os.path.isfile(weather_path):
        return df

    w = pd.read_csv(weather_path, sep="|", encoding="utf-8-sig", dtype=str)
    w.columns = [str(c).replace("\ufeff", "").strip() for c in w.columns]

    if "Дата" in df.columns:
        df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce").dt.normalize()
    if "Дата" in w.columns:
        w["Дата"] = pd.to_datetime(w["Дата"], errors="coerce").dt.normalize()

    if not {"Дата", "Город"}.issubset(df.columns) or not {"Дата", "Город"}.issubset(w.columns):
        return df

    # оставляем только нужное из погоды
    keep = [c for c in ["Дата", "Город", "ПогодныеУсловия", "СредняяТемпература", "КоличествоОсадков"] if c in w.columns]
    w = w[keep].copy()
    w["Город"] = w["Город"].astype(str).str.strip()

    df["Город"] = df["Город"].astype("string").str.strip()
    df = df.merge(w, on=["Дата", "Город"], how="left", suffixes=("", "_wx"))

    return df


# -------------------------------------------ФОРМИРУЕМ ИТОГОВЫЙ ДАТАСЕТ ДЛЯ ОБУЧЕНИЯ------------------------------------
def _prepare_training_data_dir(aboba) -> str:

    def _ts() -> str:
        return time.strftime("%d-%m-%Y %H:%M:%S")

    def _n(x: int) -> str:
        return f"{int(x):,}".replace(",", ".")

    def _dbg(tag: str, df: pd.DataFrame, extra: str = "") -> None:
        msg = f"[{_ts()}] [FILTER DEBUG] {tag}: { _n(len(df)) } строк"
        if extra:
            msg += f" | {extra}"

    base_dir = os.path.join(os.getcwd(), "ВходныеДанные")

    if not _any_order_filters_set(aboba):
        return "ВходныеДанные"

    out_rel = "ФильтрованныеДанные"
    out_dir = os.path.join(os.getcwd(), out_rel)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    f = _get_current_order_filters(aboba)

    def _parse_date(s: str):
        if not s:
            return None
        d = pd.to_datetime(s, errors="coerce", dayfirst=True)
        return d if pd.notna(d) else None

    d_from = _parse_date(f["date_from"])
    d_to = _parse_date(f["date_to"])

    # делаем date_to "включительно на весь день", если в данных есть время
    if d_to is not None:
        d_to = d_to.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if d_from is not None:
        d_from = d_from.normalize()

    def _apply_date(df: pd.DataFrame, tag: str = "") -> pd.DataFrame:
        if "Дата" not in df.columns:
            if d_from is not None or d_to is not None:
                return df.iloc[0:0].copy()
            return df

        df = df.copy()
        src = df["Дата"].astype("string")

        # Маски форматов
        m_iso = src.str.match(r"^\d{4}-\d{2}-\d{2}")  # 2025-03-04 или 2025-03-04 00:00:00
        m_dot = src.str.contains(r"\.", regex=True)  # 04.03.2025 или 04.03.2025 00:00:00

        dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

        # 1) ISO -> dayfirst=False (иначе месяц/день меняются местами)
        if m_iso.any():
            dt.loc[m_iso] = pd.to_datetime(src.loc[m_iso], errors="coerce", dayfirst=False)

        # 2) dd.mm.yyyy -> dayfirst=True
        if m_dot.any():
            dt.loc[m_dot] = pd.to_datetime(src.loc[m_dot], errors="coerce", dayfirst=True)

        # 3) Остальное (если осталось) -> пробуем обычный парсинг без dayfirst
        rest = dt.isna() & src.notna() & (src.str.strip() != "")
        if rest.any():
            dt.loc[rest] = pd.to_datetime(src.loc[rest], errors="coerce", dayfirst=False)

        # применяем фильтр периода
        if d_from is not None or d_to is not None:
            ok = dt.notna()

            if d_from is not None:
                ok &= (dt >= d_from)
            if d_to is not None:
                ok &= (dt <= d_to)

            df = df[ok]

        # дату оставляем как в исходнике (чтобы не портить формат в ФильтрованныеДанные)
        df["Дата"] = src.loc[df.index]
        return df

    def _apply_kind(df: pd.DataFrame) -> pd.DataFrame:
        kinds = f.get("kinds") or []
        if not kinds or "ВидНоменклатуры" not in df.columns:
            return df
        df = df.copy()
        if f.get("kind_mode") == "Не в группе":
            return df[~df["ВидНоменклатуры"].isin(kinds)]
        return df[df["ВидНоменклатуры"].isin(kinds)]

    def _apply_store(df: pd.DataFrame) -> pd.DataFrame:
        stores = f.get("stores") or []
        if not stores or "Магазин" not in df.columns:
            return df
        df = df.copy()
        if f.get("store_mode") == "Не в группе":
            return df[~df["Магазин"].isin(stores)]
        return df[df["Магазин"].isin(stores)]

    # --- Заказы ---
    p_orders = os.path.join(base_dir, "Заказы.csv")
    if os.path.isfile(p_orders):
        df = pd.read_csv(p_orders, sep="|", dtype=str)
        _dbg("orders loaded", df)

        df = _apply_date(df, tag="orders")
        _dbg("orders after _apply_date", df)

        df = _apply_kind(df)
        _dbg("orders after _apply_kind", df)

        df = _apply_store(df)
        _dbg("orders after _apply_store", df)

        # полезные доп. метрики (чтобы сравнивать со статистикой)
        try:
            qty_sum = pd.to_numeric(df.get("Количество"), errors="coerce").fillna(0).sum()
            uniq_orders = df.get("НомерЗаказа", pd.Series(dtype=str)).nunique()
            _dbg("orders sanity", df, extra=f"sum(Количество)={_n(qty_sum)}; uniq(НомерЗаказа)={_n(uniq_orders)}")
        except Exception:
            pass

        weather_path = os.path.join(base_dir, "Погода.csv")
        before_enrich = len(df)
        df = _enrich_orders_with_city_and_weather(aboba, df, weather_path)
        _dbg("orders after _enrich_orders_with_city_and_weather", df, extra=f"delta={_n(len(df) - before_enrich)}")

        df.to_csv(os.path.join(out_dir, "Заказы.csv"), sep="|", index=False)

    # --- Просмотры (дата + вид номенклатуры) ---
    p_views = os.path.join(base_dir, "Просмотры.csv")
    if os.path.isfile(p_views):
        df = pd.read_csv(p_views, sep="|", dtype=str)
        _dbg("views loaded", df)

        df = _apply_date(df, tag="views")
        _dbg("views after _apply_date", df)

        df = _apply_kind(df)
        _dbg("views after _apply_kind", df)

        df.to_csv(os.path.join(out_dir, "Просмотры.csv"), sep="|", index=False)

    # --- Избранное (дата + вид номенклатуры) ---
    p_favs = os.path.join(base_dir, "Избранное.csv")
    if os.path.isfile(p_favs):
        df = pd.read_csv(p_favs, sep="|", dtype=str)
        _dbg("favs loaded", df)

        df = _apply_date(df, tag="favs")
        _dbg("favs after _apply_date", df)

        df = _apply_kind(df)
        _dbg("favs after _apply_kind", df)

        # полезное для сверки
        try:
            uniq_users = df.get("MindboxID", pd.Series(dtype=str)).nunique()
            uniq_items = df.get("КодНоменклатуры", pd.Series(dtype=str)).nunique()
            _dbg("favs sanity", df, extra=f"uniq(MindboxID)={_n(uniq_users)}; uniq(КодНоменклатуры)={_n(uniq_items)}")
        except Exception:
            pass

        df.to_csv(os.path.join(out_dir, "Избранное.csv"), sep="|", index=False)

    # --- Справочники (копируем как есть, чтобы тренер не сломался) ---
    for fn in ("Номенклатура.csv", "КатегорииСайта.csv"):
        src = os.path.join(base_dir, fn)
        dst = os.path.join(out_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    return out_rel


# -------------------------------------------ПОЛУЧАЕМ ТЕКУЩИЕ ЗНАЧЕНИЯ ФИЛЬТРОВ-----------------------------------------
def _any_order_filters_set(aboba) -> bool:
    f = _get_current_order_filters(aboba)
    return bool(f["date_from"] or f["date_to"] or f["kinds"] or f["stores"])


# -------------------------------------------ПОЛУЧАЕМ ТЕКУЩИЕ ЗНАЧЕНИЯ ФИЛЬТРОВ-----------------------------------------
def _get_current_order_filters(aboba) -> dict:
    # даты (учитываем inputMask)
    def _masked_date_is_empty(qle) -> bool:
        if qle is None:
            return True
        t = qle.text()
        if t is None:
            return True
        t = t.replace(" ", "").replace(".", "")
        return t == ""

    date_from = ""
    if hasattr(aboba, "filter_date_from") and not _masked_date_is_empty(aboba.filter_date_from):
        date_from = aboba.filter_date_from.text().strip()

    date_to = ""
    if hasattr(aboba, "filter_date_to") and not _masked_date_is_empty(aboba.filter_date_to):
        date_to = aboba.filter_date_to.text().strip()

    kind_mode = aboba.kind_mode.currentText() if hasattr(aboba, "kind_mode") else "В группе"
    store_mode = aboba.store_mode.currentText() if hasattr(aboba, "store_mode") else "В группе"

    kinds = get_selected_list_values(aboba.filter_kind) if hasattr(aboba, "filter_kind") else []
    stores = get_selected_list_values(aboba.filter_store) if hasattr(aboba, "filter_store") else []

    return {
        "date_from": date_from,
        "date_to": date_to,
        "kind_mode": kind_mode,
        "store_mode": store_mode,
        "kinds": kinds,
        "stores": stores,
    }


# -------------------------------------------ОБРАБОТКА ВЫВОДА ПРОЦЕССА ОБУЧЕНИЯ-----------------------------------------
def _on_train_output(aboba):
    if not hasattr(aboba, "train_log"):
        return

    data = aboba.train_proc.readAllStandardOutput()
    text = bytes(data).decode("utf-8", errors="replace")

    # пишем без лишних "append" переносов
    aboba.train_log.moveCursor(QTextCursor.MoveOperation.End)
    aboba.train_log.insertPlainText(text)
    aboba.train_log.moveCursor(QTextCursor.MoveOperation.End)


# -------------------------------------------ЗАВЕРШЕНИЕ ОБУЧЕНИЯ--------------------------------------------------------
def _on_train_finished(aboba, exit_code: int, exit_status: QProcess.ExitStatus):
    aboba.start_train.setEnabled(True)

    if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
        set_status_ok(aboba, "Обучение завершено")
        schedule_status_reset(aboba, 5)
        if hasattr(aboba, "train_log"):
            aboba.train_log.append("Обучение успешно завершено.")
    else:
        set_status_error(aboba, f"Обучение завершилось с ошибкой (код {exit_code})")
        if hasattr(aboba, "train_log"):
            aboba.train_log.append(f"Обучение завершилось с ошибкой (код {exit_code}).")





