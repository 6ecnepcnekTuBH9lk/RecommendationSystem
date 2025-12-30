import os
import sys
import chardet
import numpy as np
import pandas as pd
from SwitchTheme import ThemeSwitch
from LightFM import train_recommender
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QIcon, QPixmap, QPalette, QColor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QLineEdit, QComboBox, QCheckBox, QRadioButton, QSlider, QProgressBar, QSpinBox,
                             QDoubleSpinBox, QTextEdit, QListWidget, QTabWidget, QMessageBox, QInputDialog,
                             QFileDialog, QFrame, QFormLayout, QAbstractSpinBox, QGridLayout, QSizePolicy)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Заголовок и иконка
        self.setWindowTitle("Рекомендательная система")
        self.setWindowIcon(QIcon("Картинки/ChatGPT.png"))

        # Размер окна
        screen = app.primaryScreen().availableGeometry()
        x = (screen.width() - 846) // 2
        y = (screen.height() - 1000) // 2
        self.setGeometry(x, y, 846, 1000)

        # Центральный виджет и основной layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Создаем вкладки для разных групп виджетов
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""QTabWidget::pane { border: 1px solid #DCDCDC; }
                                   
                                   QTabBar::tab { 
                                                  font-family: "Verdana"; 
                                                  font-weight: 500;
                                                  font-size: 14px;
                                                  letter-spacing: 0.1px;
                                                  color: black;
                                                  background: #EBEBEB;
                                                  padding: 7px 10px; 
                                                  border-radius: 10px; }
                                                                
                                   QTabBar::tab:selected { background: #DCDCDC; }""")
        main_layout.addWidget(self.tabs)

        # Вкладка с загрузкой входных данных
        self.create_input_data_widgets_tab()
        # Вкладка с обучением модели
        self.create_train_model_widgets_tab()

        # Вкладка с базовыми виджетами
        self.create_basic_widgets_tab()
        # Вкладка с виджетами ввода
        self.create_input_widgets_tab()
        # Вкладка с индикаторами
        self.create_indicator_widgets_tab()
        # Вкладка с контейнерами
        self.create_container_widgets_tab()

        # --- Кастомный нижний бар ---
        bottom_bar = QHBoxLayout()

        # Лейбл слева
        self.status_label = QLabel("Сделай уже что-нибудь...")
        self.status_label.setStyleSheet("""
            font-size: 14px;
        """)
        bottom_bar.addWidget(self.status_label)

        bottom_bar.addStretch()  # растягиваем пространство между лейблом и свитчем

        # Свитч справа
        self.theme_switch = ThemeSwitch()
        self.theme_switch.themeChanged.connect(self.apply_theme)
        bottom_bar.addWidget(self.theme_switch)

        # Добавляем этот блок в main_layout
        main_layout.addLayout(bottom_bar)

    # -------------------------------------------ПЕРЕКЛЮЧАТЕЛЬ ТЕМЫ-----------------------------------------------------
    def apply_theme(self, is_dark: bool):
        if is_dark:
            # === ТЁМНАЯ ТЕМА ===
            dark_palette_tymbler = QPalette()
            dark_palette_tymbler.setColor(QPalette.ColorRole.Window, QColor(60, 60, 60))  # общий фон (да)
            dark_palette_tymbler.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)  # текст на фоне (да)
            dark_palette_tymbler.setColor(QPalette.ColorRole.Base, QColor(110, 110, 110))  # поля ввода (да, галочки)
            dark_palette_tymbler.setColor(QPalette.ColorRole.AlternateBase,
                                          QColor(100, 100, 100))  # чередующиеся строки в тч (нет)
            dark_palette_tymbler.setColor(QPalette.ColorRole.ToolTipBase,
                                          QColor(100, 100, 100))  # всплывающие подсказки (нет)
            dark_palette_tymbler.setColor(QPalette.ColorRole.ToolTipText,
                                          Qt.GlobalColor.white)  # текст всплывающих подсказок (нет)
            dark_palette_tymbler.setColor(QPalette.ColorRole.Text,
                                          Qt.GlobalColor.white)  # основной текст внутри полей (да)
            dark_palette_tymbler.setColor(QPalette.ColorRole.Button, QColor(100, 100, 100))  # фон кнопки (нет)
            dark_palette_tymbler.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)  # текст на кнопках (нет)
            dark_palette_tymbler.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)  # текст ошибки (да)
            dark_palette_tymbler.setColor(QPalette.ColorRole.Highlight, QColor(100, 100, 100))  # выделение (да)
            dark_palette_tymbler.setColor(QPalette.ColorRole.HighlightedText,
                                          Qt.GlobalColor.white)  # текст при выделении (да)
            dark_palette_tymbler.setColor(QPalette.ColorRole.PlaceholderText,
                                          QColor(100, 100, 100))  # цвет текста в пустом поле (нет)
            app.setPalette(dark_palette_tymbler)

            # Стиль виджетов
            app.setStyleSheet("""

                        * {
                            font-size: 14px;
                            font-family: "Roboto";
                        }

                        QLabel {
                            padding: 5px 0px;
                            color: white;
                        }
                        
                        QTabWidget::pane { border: 1px solid #555; }
                        QTabBar::tab { 
                            background: #4B4B4B; 
                            color: white; 
                            padding: 6px 10px; 
                            border-radius: 7px;
                            font-family: "Roboto";
                            }
                        QTabBar::tab:selected { background: #5A5A5A; }

                        QPushButton {
                            background-color: #4B4B4B;
                            border: 1px solid #9B4DFF;
                            padding: 6px 10px;
                            border-radius: 10px;
                            color: white;
                        }
                        QPushButton:hover { background-color: #5F5F5F; }
                        QPushButton:pressed { background-color: #6E6E6E; }
                        QPushButton:disabled { background-color: #373737; color: #888; }

                        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                            background-color: #4B4B4B;
                            color: white;
                            border: 1px solid #555;
                            border-radius: 7px;
                            padding: 4px;
                        }
                        
                        QComboBox::drop-down {
                            width: 24px;
                            border: none;
                            background: transparent;
                        }

                        QComboBox::down-arrow {
                            image: url("Картинки/Вниз.png");
                            width: 18px;
                            height: 18px;
                        }

                        QListWidget {
                            background-color: #464646;
                            color: white;
                            border: 1px solid #555;
                        }

                        QProgressBar {
                            border: 1px solid #555;
                            text-align: center;
                            color: white;
                            border-radius: 7px;
                            background-color: #464646;
                        }
                        QProgressBar::chunk {
                            background-color: #A0A0A0;
                            border-radius: 7px;
                        }
                        QMessageBox {
                            margin: 0;
                            padding: 0;
                        }
                        QMessageBox QLabel {
                            padding: 0;
                            margin: 0;
                        }
                        
                        QFrame[frameRole="separator"] {
                            background-color: #555;
                            border: none;
                            min-height: 1px;
                            max-height: 1px;
                        }
                    """)

        else:
            # === СВЕТЛАЯ ТЕМА ===
            bright_palette_tymbler = QPalette()
            bright_palette_tymbler.setColor(QPalette.ColorRole.Window, QColor(250, 250, 250))  # общий фон (да)
            bright_palette_tymbler.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)  # текст на фоне (да)
            bright_palette_tymbler.setColor(QPalette.ColorRole.Base, QColor(240, 240, 240))  # поля ввода (да, галочки)
            bright_palette_tymbler.setColor(QPalette.ColorRole.AlternateBase,
                                            QColor(100, 100, 100))  # чередующиеся строки в тч (нет)
            bright_palette_tymbler.setColor(QPalette.ColorRole.ToolTipBase,
                                            QColor(100, 100, 100))  # всплывающие подсказки (нет)
            bright_palette_tymbler.setColor(QPalette.ColorRole.ToolTipText,
                                            Qt.GlobalColor.white)  # текст всплывающих подсказок (нет)
            bright_palette_tymbler.setColor(QPalette.ColorRole.Text,
                                            Qt.GlobalColor.black)  # основной текст внутри полей (да)
            bright_palette_tymbler.setColor(QPalette.ColorRole.Button, QColor(100, 100, 100))  # фон кнопки (нет)
            bright_palette_tymbler.setColor(QPalette.ColorRole.ButtonText,
                                            Qt.GlobalColor.white)  # текст на кнопках (нет)
            bright_palette_tymbler.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)  # текст ошибки (да)
            bright_palette_tymbler.setColor(QPalette.ColorRole.Highlight, QColor(210, 210, 210))  # выделение (да)
            bright_palette_tymbler.setColor(QPalette.ColorRole.HighlightedText,
                                            Qt.GlobalColor.black)  # текст при выделении (да)
            bright_palette_tymbler.setColor(QPalette.ColorRole.PlaceholderText,
                                            QColor(100, 100, 100))  # цвет текста в пустом поле (нет)
            app.setPalette(bright_palette_tymbler)

            # Стиль виджетов
            app.setStyleSheet("""

                                                    * {
                                                        font-size: 14px;
                                                        font-family: "Roboto";
                                                    }

                                                    QTabWidget::pane { border: 1px solid #EBEBEB; }
                                                    QTabBar::tab { 
                                                        background: #EBEBEB; 
                                                        color: black; 
                                                        padding: 6px 10px;
                                                        border-radius: 7px;
                                                        font-family: "Roboto";
                                                    }
                                                    QTabBar::tab:selected { background: #DCDCDC; }

                                                    QLabel {
                                                        padding: 5px 0px;
                                                        color: black;
                                                    }

                                                    QPushButton {
                                                        background-color: #EBEBEB;
                                                        border: 1px solid #9B4DFF;
                                                        padding: 6px 10px;
                                                        border-radius: 10px;
                                                        color: black;
                                                    }
                                                    QPushButton:hover { background-color: #D7D7D7; }
                                                    QPushButton:pressed { background-color: #C8C8C8; }
                                                    QPushButton:disabled { background-color: #373737; color: #888; }

                                                    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                                                        background-color: #EBEBEB;
                                                        color: black;
                                                        border: 1px solid #D7D7D7;
                                                        border-radius: 7px;
                                                        padding: 4px;
                                                    }

                                                    QComboBox::drop-down {
                                                        width: 24px;
                                                        border: none;
                                                        background: transparent;
                                                    }

                                                    QComboBox::down-arrow {
                                                        image: url("Картинки/Вниз.png");
                                                        width: 18px;
                                                        height: 18px;
                                                    }

                                                    QListWidget {
                                                        background-color: #F0F0F0;
                                                        color: black;
                                                        border: 1px solid #D7D7D7;
                                                    }

                                                    QProgressBar {
                                                        border: 1px solid #D7D7D7;
                                                        text-align: center;
                                                        color: black;
                                                        border-radius: 7px;
                                                        background-color: #F0F0F0;
                                                    }
                                                    QProgressBar::chunk {
                                                        background-color: #969696;
                                                        border-radius: 7px;
                                                    }
                                                    QMessageBox {
                                                        margin: 0;
                                                        padding: 0;
                                                    }
                                                    QMessageBox QLabel {
                                                        padding: 0;
                                                        margin: 0;
                                                    }

                                                    QFrame[frameRole="separator"] {
                                                        background-color: #EBEBEB;
                                                        border: none;
                                                        min-height: 1px;
                                                        max-height: 1px;
                                                    }
                                                """)

    # -------------------------------------------ВКЛАДКА ОБРАБОТКА ДАТАСЕТА---------------------------------------------
    def create_input_data_widgets_tab(self):

        tab = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        tab.setLayout(layout)

        row_layout = QHBoxLayout()

        # Заголовок
        self.label_1 = QLabel("Загрузка данных")
        self.label_1.setSizePolicy(self.label_1.sizePolicy().Policy.Fixed, self.label_1.sizePolicy().Policy.Fixed)
        self.label_1.setContentsMargins(0, 0, 0, 0)
        self.label_1.setStyleSheet("""
            QLabel {
                        font-family: "Verdana"; 
                        font-weight: 500;
                        font-size: 14px;
                        letter-spacing: 0.1px;
                        color: black;
                        background-color: #FAFAFA;
                        padding: 7px 65px; 
                        border-radius: 10px;
                        border: 1px solid #C8C8C8;
                        margin: 8px 0px 8px 0px;
                    }
        """)
        layout.addWidget(self.label_1, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Вываливающийся список
        self.combo_box_types = QComboBox()
        self.combo_box_types.addItems(["Заказы клиентов из Mindbox",
                                       "Просмотры товаров и категорий из Mindbox",
                                       "Добавление товаров в избранное из Mindbox",
                                       "Номенклатура из 1С", "Категории сайта из 1С",
                                       ])
        self.combo_box_types.setStyleSheet("""
        QComboBox {
        font-family: "Verdana"; 
        font-weight: 500;
        font-size: 14px;
        letter-spacing: 0.1px;
        color: black;
        background-color: #EBEBEB;
        padding: 4px 5px; 
        border-radius: 10px;
        border: 1px solid #DCDCDC;
        margin: 0px 2px 2px 0px;
        }
                                        
        QComboBox::drop-down {
        width: 19px;
        border: none;
        background: transparent;
        }

        QComboBox::down-arrow {
        image: url("Картинки/Вниз.png");
        width: 17px;
        height: 17px;
        }""")

        # Вываливающийся список
        self.combo_box_add_or_not = QComboBox()
        self.combo_box_add_or_not.addItems(["Добавить новый / Обновить существующий",
                                            "Добавить данные к существующему"])
        self.combo_box_add_or_not.setStyleSheet("""
                QComboBox {
                font-family: "Verdana"; 
                font-weight: 500;
                font-size: 14px;
                letter-spacing: 0.1px;
                color: black;
                background-color: #EBEBEB;
                padding: 4px 5px; 
                border-radius: 10px;
                border: 1px solid #DCDCDC;
                margin: 2px 2px 8px 0px;
                }

                QComboBox::drop-down {
                width: 19px;
                border: none;
                background: transparent;
                }

                QComboBox::down-arrow {
                image: url("Картинки/Вниз.png");
                width: 17px;
                height: 17px;
                }""")

        # Левая колонка: два выпадающих списка (вертикально)
        left_col = QVBoxLayout()
        left_col.addWidget(self.combo_box_types)
        left_col.addWidget(self.combo_box_add_or_not)

        # Кнопки
        self.btn_analyse = QPushButton(QIcon("Картинки/ПровестиАнализ.png"), " Провести анализ")
        self.btn_analyse.setIconSize(QSize(17, 17))
        self.btn_analyse.clicked.connect(self.run_analysis)
        self.btn_analyse.setStyleSheet("""
                                        QPushButton {
                                        font-family: "Verdana"; 
                                        font-weight: 500;
                                        font-size: 14px;
                                        letter-spacing: 0.1px;
                                        color: black;
                                        padding: 5px 0px; 
                                        border-radius: 10px;
                                        border: 1px solid #9B4DFF;
                                        margin: 0px 0px 2px 2px;
                                        background: qlineargradient(
                                            x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #FFFFFF,
                                            stop:1 #EBEBEB
                                            );
                                        }

                                        QPushButton:hover { background-color: #D7D7D7; }
                                        QPushButton:pressed { background-color: #C8C8C8; }
                                        QPushButton:disabled { background-color: #373737; color: #888; }
                                        }""")

        self.btn_load = QPushButton(QIcon("Картинки/ЗагрузитьФайл.png"), " Загрузить файл")
        self.btn_load.setIconSize(QSize(17, 17))
        self.btn_load.clicked.connect(self.load_csv_file)
        self.btn_load.setStyleSheet("""
                        QPushButton {
                        font-family: "Verdana"; 
                        font-weight: 500;
                        font-size: 14px;
                        letter-spacing: 0.1px;
                        color: black;
                        padding: 5px 0px; 
                        border-radius: 10px;
                        border: 1px solid #9B4DFF;
                        margin: 2px 0px 8px 2px;
                        background: qlineargradient(
                            x1:0, y1:0, x2:0, y2:1,
                            stop:0 #FFFFFF,
                            stop:1 #EBEBEB
                            );
                        }
                                        
                        QPushButton:hover { background-color: #D7D7D7; }
                        QPushButton:pressed { background-color: #C8C8C8; }
                        QPushButton:disabled { background-color: #373737; color: #888; }
                        }""")

        # Правая колонка: две кнопки (вертикально)
        right_col = QVBoxLayout()
        right_col.addWidget(self.btn_analyse)
        right_col.addWidget(self.btn_load)

        # Собираем строку: слева списки, справа кнопки
        row_layout.addLayout(left_col, stretch=5)
        row_layout.addLayout(right_col, stretch=3)

        layout.addLayout(row_layout)

        # Статус загрузки файлов
        self.status_files_label = QLabel("")
        self.status_files_label.setStyleSheet("""
                    font-family: "Verdana"; 
                    font-weight: 500;
                    font-size: 14px;
                    letter-spacing: 0.1px;
                    color: black;
                """)
        layout.addWidget(self.status_files_label)

        # Обновляем статус
        self.update_file_status()

        # Заголовок
        label_2 = QLabel("Статистика и анализ")
        label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_2.setStyleSheet("""
                    QLabel {
                        font-family: "Verdana"; 
                        font-weight: 500;
                        font-size: 14px;
                        letter-spacing: 0.1px;
                        color: black;
                        background-color: #FAFAFA;
                        padding: 7px 65px; 
                        border-radius: 10px;
                        border: 1px solid #C8C8C8;
                        margin: 8px 0px 8px 0px;
                    }
                """)
        layout.addWidget(label_2, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Создаем вкладки для анализа датасетов
        self.static_tabs = QTabWidget()
        self.static_tabs.setStyleSheet("""QTabBar::tab { font-size: 14px; }""")
        layout.addWidget(self.static_tabs)

        # Вкладка со всеми заказами
        self.create_static_order_full_tab()

        # Вкладка со всеми просмотрами
        self.create_static_views_full_tab()

        # Вкладка со всем избранным
        self.create_static_favorites_full_tab()

        # Формируем статистику заказов
        self.analyze_orders_full_dataset()

        # Формируем статистику просмотров
        self.analyze_views_full_dataset()

        # Формируем статистику добавлений
        self.analyze_favorites_full_dataset()

        # Название вкладки
        self.tabs.addTab(tab, "Обработка датасета")

    # -------------------------------------------ВКЛАДКА ОБУЧЕНИЕ МОДЕЛИ------------------------------------------------
    def create_train_model_widgets_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        tab.setLayout(layout)

        # Заголовок
        label_1 = QLabel("Параметры для обучения модели")
        label_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_1.setStyleSheet("font-size: 16px; padding: 8px 0px; font-style: italic;")
        layout.addWidget(label_1)

        # Основной layout для параметров (два столбца)
        grid_layout = QGridLayout()

        # Параметры BPR-MF
        self.embedding_dim_input = QSpinBox()
        self.embedding_dim_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.embedding_dim_input.setRange(0, 10000)
        self.embedding_dim_input.setValue(64)

        self.epochs_input = QSpinBox()
        self.epochs_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.epochs_input.setRange(0, 10000)
        self.epochs_input.setValue(20)

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.batch_size_input.setRange(0, 10000)
        self.batch_size_input.setValue(4096)

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.lr_input.setRange(0, 1)
        self.lr_input.setDecimals(3)
        self.lr_input.setValue(2e-3)

        self.weight_decay_input = QDoubleSpinBox()
        self.weight_decay_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.weight_decay_input.setRange(0, 1)
        self.weight_decay_input.setDecimals(6)
        self.weight_decay_input.setValue(1e-6)

        self.bpr_reg_input = QDoubleSpinBox()
        self.bpr_reg_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.bpr_reg_input.setRange(0, 1)
        self.bpr_reg_input.setDecimals(4)
        self.bpr_reg_input.setValue(1e-4)

        self.seed_input = QSpinBox()
        self.seed_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.seed_input.setRange(0, 10000)
        self.seed_input.setValue(42)

        # Параметры EASE^R
        self.ease_lambda_input = QSpinBox()
        self.ease_lambda_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.ease_lambda_input.setRange(0, 10000)
        self.ease_lambda_input.setValue(200)

        self.max_items_for_ease_input = QSpinBox()
        self.max_items_for_ease_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.max_items_for_ease_input.setRange(0, 100000)
        self.max_items_for_ease_input.setValue(15000)

        # Остальные параметры
        self.w_view_item = QDoubleSpinBox()
        self.w_view_item.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.w_view_item.setRange(0, 10)
        self.w_view_item.setDecimals(1)
        self.w_view_item.setValue(1.0)

        self.w_favorite = QDoubleSpinBox()
        self.w_favorite.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.w_favorite.setRange(0, 10)
        self.w_favorite.setDecimals(1)
        self.w_favorite.setValue(3.0)

        self.w_purchase = QDoubleSpinBox()
        self.w_purchase.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.w_purchase.setRange(0, 10)
        self.w_purchase.setDecimals(1)
        self.w_purchase.setValue(5.0)

        self.top_rec = QSpinBox()
        self.top_rec.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.top_rec.setRange(0, 100)
        self.top_rec.setValue(10)

        self.min_user_interactions_for_eval = QSpinBox()
        self.min_user_interactions_for_eval.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.min_user_interactions_for_eval.setRange(0, 100)
        self.min_user_interactions_for_eval.setValue(2)

        # Заполнение первой колонки
        grid_layout.addWidget(QLabel("Вес покупки:"), 0, 0)
        grid_layout.addWidget(self.w_purchase, 0, 1)

        grid_layout.addWidget(QLabel("Вес избранного:"), 1, 0)
        grid_layout.addWidget(self.w_favorite, 1, 1)

        grid_layout.addWidget(QLabel("Вес просмотра:"), 2, 0)
        grid_layout.addWidget(self.w_view_item, 2, 1)

        grid_layout.addWidget(QLabel("Количество рекомендаций:"), 3, 0)
        grid_layout.addWidget(self.top_rec, 3, 1)

        grid_layout.addWidget(QLabel("Количество эпох:"), 4, 0)
        grid_layout.addWidget(self.epochs_input, 4, 1)

        grid_layout.addWidget(QLabel("Скорость обучения:"), 5, 0)
        grid_layout.addWidget(self.lr_input, 5, 1)

        grid_layout.addWidget(QLabel("Инициализатор случайных чисел:"), 6, 0)
        grid_layout.addWidget(self.seed_input, 6, 1)

        # Заполнение второй колонки
        grid_layout.addWidget(QLabel("Регуляризация BPR:"), 0, 2)
        grid_layout.addWidget(self.bpr_reg_input, 0, 3)

        grid_layout.addWidget(QLabel("Регуляризация L2:"), 1, 2)
        grid_layout.addWidget(self.weight_decay_input, 1, 3)

        grid_layout.addWidget(QLabel("Регуляризация EASE^R:"), 2, 2)
        grid_layout.addWidget(self.ease_lambda_input, 2, 3)

        grid_layout.addWidget(QLabel("Лимит товаров EASE^R:"), 3, 2)
        grid_layout.addWidget(self.max_items_for_ease_input, 3, 3)

        grid_layout.addWidget(QLabel("Размер обучающего пакета:"), 4, 2)
        grid_layout.addWidget(self.batch_size_input, 4, 3)

        grid_layout.addWidget(QLabel("Размерность векторов:"), 5, 2)
        grid_layout.addWidget(self.embedding_dim_input, 5, 3)

        grid_layout.addWidget(QLabel("Минимум действий для оценки:"), 6, 2)
        grid_layout.addWidget(self.min_user_interactions_for_eval, 6, 3)

        # Горизонтальная линия
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setProperty("frameRole", "separator")

        # Кнопки: добавляем кнопку слева
        label_btn_layout = QHBoxLayout()

        # Кнопка для остановки обучения (дополнительная кнопка)
        btn_settings = QPushButton(QIcon("Картинки/СтандартныеНастройки.png"), " Стандартные настройки")
        btn_settings.setIconSize(QSize(17, 17))
        btn_settings.clicked.connect(self.standart_settigs)
        label_btn_layout.addWidget(btn_settings)

        # Кнопка для начала обучения
        btn_load = QPushButton(QIcon("Картинки/НачатьОбучение.png"), " Начать обучение")
        btn_load.setIconSize(QSize(17, 17))
        btn_load.clicked.connect(train_recommender)
        label_btn_layout.addWidget(btn_load)

        layout.addLayout(grid_layout)
        layout.addWidget(line)
        layout.addLayout(label_btn_layout)  # Помещаем обе кнопки внизу

        # Название вкладки
        self.tabs.addTab(tab, "Обучение модели")

    # -------------------------------------------ЗАГРУЗКА ФАЙЛОВ-----------------------------------------------------
    def load_csv_file(self):
        """Загрузка исходных CSV/1C файлов, обработка и сохранение в папку `ВходныеДанные`.

        ВАЖНО: интерфейс и названия файлов/типов не меняем — только уменьшаем дублирование кода.
        """

        # Статус бар
        self.status_label.setText("Обработка данных...")

        selected_type = self.combo_box_types.currentText()
        mode = self.combo_box_add_or_not.currentText()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите CSV файл",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if not file_path:
            self.status_label.setText(" Не хочешь, как хочешь...")
            return

        try:
            input_dir = os.path.join(os.getcwd(), "ВходныеДанные")
            os.makedirs(input_dir, exist_ok=True)

            # --- helpers (локально, чтобы не засорять класс лишними методами) ---
            def _save(df: pd.DataFrame, save_path: str) -> None:
                df.to_csv(save_path, index=False, sep="|", encoding="utf-8-sig")

            def _append_or_overwrite(df: pd.DataFrame, save_path: str) -> None:
                if mode == "Добавить новый / Обновить существующий":
                    _save(df, save_path)
                    return

                # mode == "Добавить данные к существующему"
                if os.path.exists(save_path):
                    try:
                        df_old = pd.read_csv(save_path, sep="|", encoding="utf-8-sig")
                        df = pd.concat([df_old, df], ignore_index=True)
                    except Exception:
                        # Если файл поврежден/кодировка/структура — просто перезапишем
                        pass
                _save(df, save_path)

            def _process_pair(
                reader_sep: str,
                processor_fn,
                filename_full: str,
                filename_selection: str,
            ) -> None:
                df_src = self.read_csv_auto_encoding(file_path=file_path, sep=reader_sep)
                df_full, df_selection = processor_fn(df_src)

                if df_full is None or df_selection is None:
                    return

                save_path_full = os.path.join(input_dir, filename_full)
                save_path_selection = os.path.join(input_dir, filename_selection)

                _append_or_overwrite(df_full, save_path_full)
                _append_or_overwrite(df_selection, save_path_selection)

            def _process_single(reader_sep: str, processor_fn, filename: str) -> None:
                df_src = self.read_csv_auto_encoding(file_path=file_path, sep=reader_sep)
                df_res = processor_fn(df_src)
                if df_res is None:
                    return
                save_path_full = os.path.join(input_dir, filename)
                _save(df_res, save_path_full)

            # --- routing по типу ---
            if selected_type == "Заказы клиентов из Mindbox":
                _process_pair(
                    reader_sep=";",
                    processor_fn=self.process_orders_file,
                    filename_full="ЗаказыОригинал.csv",
                    filename_selection="ЗаказыОтбор.csv",
                )

            elif selected_type == "Просмотры товаров и категорий из Mindbox":
                _process_pair(
                    reader_sep=";",
                    processor_fn=self.process_views_file,
                    filename_full="ПросмотрыОригинал.csv",
                    filename_selection="ПросмотрыОтбор.csv",
                )

            elif selected_type == "Добавление товаров в избранное из Mindbox":
                _process_pair(
                    reader_sep=";",
                    processor_fn=self.process_favorites_file,
                    filename_full="ИзбранноеОригинал.csv",
                    filename_selection="ИзбранноеОтбор.csv",
                )

            elif selected_type == "Номенклатура из 1С":
                _process_single(
                    reader_sep="|",
                    processor_fn=self.process_nomenclature_file,
                    filename="Номенклатура.csv",
                )

            elif selected_type == "Категории сайта из 1С":
                _process_single(
                    reader_sep="|",
                    processor_fn=self.process_categories_file,
                    filename="КатегорииСайта.csv",
                )

            else:
                self.show_custom_message(
                    title="Ошибка",
                    text="Неизвестный тип данных",
                    image_path="Картинки/Неудача.png",
                )
                self.status_label.setText(" Неизвестный тип данных")
                return

            # Обновляем статус + вкладки статистики
            self.update_file_status()

            # Пересчитываем статистику (как было)
            self.analyze_orders_full_dataset()
            self.analyze_orders_selection_dataset()
            self.analyze_views_full_dataset()
            self.analyze_views_selection_dataset()
            self.analyze_favorites_full_dataset()
            self.analyze_favorites_selection_dataset()

            self.status_label.setText(" Обработка завершена")

        except Exception as e:
            self.show_custom_message(
                title="Ошибка",
                text=f"Не удалось загрузить файл:\n{str(e)}",
                image_path="Картинки/Неудача.png",
            )
            self.status_label.setText(" Ошибка обработки")

    # -------------------------------------------АВТОМАТИЧЕСКАЯ КОДИРОВКА-----------------------------------------------
    def read_csv_auto_encoding(self, file_path: str, sep: str):

        try:
            # Определяем кодировку
            with open(file_path, "rb") as f:
                raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding", "utf-8")

            # Пробуем прочитать файл
            df = pd.read_csv(file_path, sep=sep, encoding=encoding)
            return df

        except Exception as e:
            self.show_custom_message(
                title="Ошибка",
                text=f"Не удалось прочитать файл:\n{file_path}\n\nПричина:\n{str(e)}",
                image_path="Картинки/Неудача.png"
            )
            self.status_label.setText(" Ошибка чтения файла")
            return None

    # -------------------------------------------ОБРАБОТКА ЗАКАЗОВ------------------------------------------------------
    def process_orders_file(self, df):

        # Список нужных колонок и новые имена
        columns_map = {
            "OrderIdsMindboxId": "НомерЗаказа",
            "OrderFirstActionDateTimeUtc": "Дата",
            "OrderFirstActionChannelName": "Магазин",
            "OrderLineProductIdsOffline1C": "КодНоменклатурыРФ",
            "OrderLineProductIdsKanzlerKz": "КодНоменклатурыКЗ",
            "OrderLineQuantity": "Количество",
            "OrderLineBasePricePerItem": "НачальнаяЦена",
            "OrderLinePriceOfLine": "КонечнаяСтоимость",
            "OrderCustomerLastActivatedCardIdsNumber": "ДисконтнаяКарта",
            "OrderCustomerIdsMindboxId": "MindboxID",
            "OrderCustomerFirstName": "Имя",
            "OrderCustomerLastName": "Фамилия",
            "OrderCustomerMiddleName": "Отчество",
            "OrderCustomerBirthDate": "ДатаРождения",
            "OrderCustomerSex": "ПолКлиента",
            "OrderCustomerEmail": "Почта",
            "OrderCustomerMobilePhone": "ТелефонОсновной",
            "OrderCustomerPendingMobilePhone": "ЗапаснойТелефон",
            "OrderCustomerCustomFieldsMostViewedCategory": "СамаяПросматриваемаяКатегория",
            "OrderCustomerCustomFieldsMostViewedRootCategory": "СамаяПросматриваемаяРодительскаяКатегория",
            "OrderCustomerCustomFieldsMostViewedSubsidiaryCategory": "СамаяПросматриваемаяДочерняяКатегория"
        }

        # Проверка наличия всех колонок
        missing = [col for col in columns_map.keys() if col not in df.columns]

        if missing:
            self.show_custom_message(title="Ошибка",
                                     text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
                                     image_path="Картинки/Неудача.png")
            self.status_label.setText(" Ошибка обработки")

            return None, None

        # Оставляем только нужные колонки
        df = df[list(columns_map.keys())]

        # Переименовываем
        df = df.rename(columns=columns_map)

        # ---------- Обработка данных ----------

        # Удаляем строки, где не заполнено НИ одно из полей "КодНоменклатурыРФ" или "КодНоменклатурыКЗ"
        df = df.dropna(subset=["КодНоменклатурыРФ", "КодНоменклатурыКЗ"], how="all")

        # Объединяем номера телефонов
        df["Телефон"] = (df["ТелефонОсновной"].combine_first(df["ЗапаснойТелефон"]).apply(
            lambda x: str(x)[:-2] if pd.notnull(x) and str(x).endswith(".0") else str(x) if pd.notnull(x) else np.nan))
        df = df.drop(columns=["ТелефонОсновной", "ЗапаснойТелефон"])

        # Преобразуем цены и количество в числовой формат
        df["НачальнаяЦена"] = pd.to_numeric(df["НачальнаяЦена"], errors='coerce').fillna(0).astype(int)
        df["КонечнаяСтоимость"] = pd.to_numeric(df["КонечнаяСтоимость"], errors='coerce').fillna(0).astype(int)
        df["Количество"] = pd.to_numeric(df["Количество"], errors='coerce').fillna(0).astype(int)

        df["НачальнаяСтоимость"] = df["НачальнаяЦена"] * df["Количество"]

        df["ПроцентСкидки"] = ((df["НачальнаяСтоимость"] - df["КонечнаяСтоимость"]) / df["НачальнаяСтоимость"] * 100)

        df["ПроцентСкидки"] = (
            df["ПроцентСкидки"]
            .replace([float('inf'), -float('inf')], 0)
            .fillna(0)
            .round()
            .astype(int)
        )

        # Определяем валюту по коду номенклатуры
        df["Валюта"] = np.where(df["КодНоменклатурыРФ"].notna(), "RUB",
                                np.where(df["КодНоменклатурыКЗ"].notna(), "KZT", None)).astype(object)

        # Объединяем коды номенклатуры
        df["КодНоменклатуры"] = df["КодНоменклатурыРФ"].combine_first(df["КодНоменклатурыКЗ"]).astype(str).str[:6]
        df = df.drop(columns=["КодНоменклатурыРФ", "КодНоменклатурыКЗ"])

        # В колонке Дата оставляем только дату (убираем время)
        df["Дата"] = pd.to_datetime(df["Дата"], errors='coerce').dt.date

        df["ДатаРождения"] = pd.to_datetime(df["ДатаРождения"], errors='coerce')

        # Рассчитываем возраст на момент заказа
        df["Возраст"] = df.apply(
            lambda row: row["Дата"].year - row["ДатаРождения"].year -
                        ((row["Дата"].month, row["Дата"].day) < (row["ДатаРождения"].month, row["ДатаРождения"].day))
            if pd.notnull(row["ДатаРождения"]) and pd.notnull(row["Дата"]) else np.nan,
            axis=1
        ).astype('Int64')

        # Добавляем колонку с возрастной группой
        df["ВозрастнаяГруппа"] = df["Возраст"].apply(self.get_age_group)

        # В колонке Магазин заменяем значение
        df["Магазин"] = df["Магазин"].replace({"kanzler-style.ru": "ИНТЕРНЕТ-МАГАЗИН"})

        # Объединяем Имя, Фамилия, Отчество в ФИО
        df["ФИО"] = df[["Фамилия", "Имя", "Отчество"]].fillna("").agg(" ".join, axis=1).str.strip()
        df = df.drop(columns=["Имя", "Фамилия", "Отчество"])

        # В колонке Пол заменяем значения
        df["ПолКлиента"] = df["ПолКлиента"].replace({"male": "Мужской", "female": "Женский"})

        # Дисконтную карту приводим к строке, убираем ".0", если он был числом
        df["ДисконтнаяКарта"] = df["ДисконтнаяКарта"].apply(
            lambda x: str(int(x)) if pd.notnull(x) and str(x).endswith(".0") else str(x))

        # Дисконтную карту приводим к строке, убираем ".0", если он был числом
        df["НомерЗаказа"] = df["НомерЗаказа"].apply(
            lambda x: str(int(x)) if pd.notnull(x) and str(x).endswith(".0") else str(x))

        # Очистка категорий от точек и запятых
        for col in [
            "СамаяПросматриваемаяКатегория",
            "СамаяПросматриваемаяРодительскаяКатегория",
            "СамаяПросматриваемаяДочерняяКатегория"
        ]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)  # убираем только .0 в конце
                .str.replace(r"[.,]", "", regex=True)  # убираем лишние знаки, если вдруг есть
                .str.strip()
            )

        # Объединяем любимые категории в одну колонку через "_"
        df["ЛюбимаяКатегория"] = (df[["СамаяПросматриваемаяКатегория",
                                      "СамаяПросматриваемаяРодительскаяКатегория",
                                      "СамаяПросматриваемаяДочерняяКатегория"]].apply(
            lambda row: "_".join([str(x) for x in row if pd.notnull(x) and x != "nan"]),
            axis=1))

        # Удалим отдельные колонки категорий
        df = df.drop(columns=["СамаяПросматриваемаяКатегория",
                              "СамаяПросматриваемаяРодительскаяКатегория",
                              "СамаяПросматриваемаяДочерняяКатегория"])

        # --- Подтягиваем данные из Номенклатуры.csv ---
        nom_path = "ВходныеДанные/Номенклатура.csv"

        if not os.path.isfile(nom_path):
            self.show_custom_message(
                title="Ошибка",
                text="Для корректной загрузки необходимо сначала загрузить файл Номенклатура.csv",
                image_path="Картинки/Неудача.png"
            )
            return None, None

        # Читаем файл номенклатуры
        nom_df = pd.read_csv(nom_path, sep="|", dtype=str)

        # Проверяем обязательные колонки
        required_nom_cols = ["КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте"]
        missing_nom = [col for col in required_nom_cols if col not in nom_df.columns]

        if missing_nom:
            self.show_custom_message(
                title="Ошибка",
                text="В файле Номенклатура.csv отсутствуют колонки:\n" + "\n".join(missing_nom),
                image_path="Картинки/Неудача.png"
            )
            return None, None

        # Оставляем только нужные колонки
        nom_df = nom_df[required_nom_cols]

        # Объединяем основной df с номенклатурой
        df = df.merge(nom_df, on="КодНоменклатуры", how="left")

        # Заменяем все виды пустых значений на np.nan
        df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "nan"], np.nan)

        # Упорядочиваем колонки
        column_order = [
            "Дата", "НомерЗаказа", "Магазин", "КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте",
            "Количество", "НачальнаяЦена", "НачальнаяСтоимость", "КонечнаяСтоимость", "ПроцентСкидки", "Валюта",
            "ДисконтнаяКарта", "MindboxID", "ФИО", "ДатаРождения", "Возраст", "ВозрастнаяГруппа", "ПолКлиента",
            "Почта", "Телефон", "ЛюбимаяКатегория"
        ]
        df = df[column_order]

        df = df.sort_values(by="Дата", ascending=True)

        # Разбиваем файл
        df_full = df.copy()
        df_selection = df_full[~df_full["ВидНоменклатуры"].isin(["Наборы трусов", "Трусы NOS (нижнее белье)",
                                                                 "Трусы (нижнее белье)", "Носки", "Носки NOS"])].copy()

        # возвращаем обработанный DataFrame
        return df_full, df_selection

    # -------------------------------------------ОБРАБОТКА ПРОСМОТРОВ---------------------------------------------------
    def process_views_file(self, df):
        # Список нужных колонок и новые имена
        columns_map = {
            "CustomerActionDateTimeUtc": "Дата",
            "CustomerActionProductsIdsOffline1C": "КодНоменклатурыРФ",
            "CustomerActionProductsIdsKanzlerKz": "КодНоменклатурыКЗ",
            "CustomerActionProductCategoriesIdsOffline1C": "КодКатегории",
            "CustomerActionCustomerLastActivatedCardIdsNumber": "ДисконтнаяКарта",
            "CustomerActionCustomerIdsMindboxId": "MindboxID",
            "CustomerActionCustomerFirstName": "Имя",
            "CustomerActionCustomerLastName": "Фамилия",
            "CustomerActionCustomerMiddleName": "Отчество",
            "CustomerActionCustomerBirthDate": "ДатаРождения",
            "CustomerActionCustomerSex": "ПолКлиента",
            "CustomerActionCustomerEmail": "Почта",
            "CustomerActionCustomerMobilePhone": "ТелефонОсновной",
            "CustomerActionCustomerPendingMobilePhone": "ЗапаснойТелефон",
            "CustomerActionCustomerCustomFieldsMostViewedCategory": "СамаяПросматриваемаяКатегория",
            "CustomerActionCustomerCustomFieldsMostViewedRootCategory": "СамаяПросматриваемаяРодительскаяКатегория",
            "CustomerActionCustomerCustomFieldsMostViewedSubsidiaryCategory": "СамаяПросматриваемаяДочерняяКатегория"
        }

        # Проверка наличия всех колонок
        missing = [col for col in columns_map.keys() if col not in df.columns]

        if missing:
            self.show_custom_message(title="Ошибка",
                                     text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
                                     image_path="Картинки/Неудача.png")
            self.status_label.setText(" Ошибка обработки")

            return None, None

        # Оставляем только нужные колонки
        df = df[list(columns_map.keys())]

        # Переименовываем
        df = df.rename(columns=columns_map)

        # ---------- Обработка данных ----------

        # Удаляем строки, где не заполнено НИ одно из полей
        # "КодНоменклатурыРФ", "КодНоменклатурыКЗ" или "ПросмотреннаяКатегория"
        df = df.dropna(subset=["КодНоменклатурыРФ", "КодНоменклатурыКЗ", "КодКатегории"], how="all")

        # Объединяем коды номенклатуры
        df["КодНоменклатурыПервый"] = df["КодНоменклатурыРФ"].combine_first(df["КодНоменклатурыКЗ"]).astype(str).str[:6]
        df = df.drop(columns=["КодНоменклатурыРФ", "КодНоменклатурыКЗ"])

        # Объединяем номера телефонов
        df["Телефон"] = (df["ТелефонОсновной"].combine_first(df["ЗапаснойТелефон"])
        .apply(
            lambda x: str(x)[:-2] if pd.notnull(x) and str(x).endswith(".0") else str(x) if pd.notnull(x) else np.nan))
        df = df.drop(columns=["ТелефонОсновной", "ЗапаснойТелефон"])

        # В колонке Дата оставляем только дату (убираем время)
        df["Дата"] = pd.to_datetime(df["Дата"], errors='coerce').dt.date

        df["ДатаРождения"] = pd.to_datetime(df["ДатаРождения"], errors='coerce')

        # Рассчитываем возраст на момент заказа
        df["Возраст"] = df.apply(
            lambda row: row["Дата"].year - row["ДатаРождения"].year -
                        ((row["Дата"].month, row["Дата"].day) < (row["ДатаРождения"].month, row["ДатаРождения"].day))
            if pd.notnull(row["ДатаРождения"]) and pd.notnull(row["Дата"]) else np.nan,
            axis=1
        ).astype('Int64')

        # Добавляем колонку с возрастной группой
        df["ВозрастнаяГруппа"] = df["Возраст"].apply(self.get_age_group)

        # Объединяем Имя, Фамилия, Отчество в ФИО
        df["ФИО"] = df[["Фамилия", "Имя", "Отчество"]].fillna("").agg(" ".join, axis=1).str.strip()
        df = df.drop(columns=["Имя", "Фамилия", "Отчество"])

        # В колонке Пол заменяем значения
        df["ПолКлиента"] = df["ПолКлиента"].replace({"male": "Мужской", "female": "Женский"})

        # Дисконтную карту приводим к строке, убираем ".0", если он был числом
        df["ДисконтнаяКарта"] = df["ДисконтнаяКарта"].apply(
            lambda x: str(int(x)) if pd.notnull(x) and str(x).endswith(".0") else str(x))

        # Очистка категорий от точек и запятых
        for col in [
            "КодКатегории",
            "СамаяПросматриваемаяКатегория",
            "СамаяПросматриваемаяРодительскаяКатегория",
            "СамаяПросматриваемаяДочерняяКатегория"
        ]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)  # убираем только .0 в конце
                .str.replace(r"[.,]", "", regex=True)  # убираем лишние знаки, если вдруг есть
                .str.strip()
            )

        # Объединяем любимые категории в одну колонку через "_"
        df["ЛюбимаяКатегория"] = df[
            ["СамаяПросматриваемаяКатегория", "СамаяПросматриваемаяРодительскаяКатегория",
             "СамаяПросматриваемаяДочерняяКатегория"]
        ].apply(lambda row: "_".join([str(x) for x in row if pd.notnull(x) and x != "nan"]), axis=1)

        # Удалим отдельные колонки категорий
        df = df.drop(columns=["СамаяПросматриваемаяКатегория",
                              "СамаяПросматриваемаяРодительскаяКатегория",
                              "СамаяПросматриваемаяДочерняяКатегория"])

        # Заменяем все виды пустых значений на np.nan
        df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "nan"], np.nan)

        # Определяем тип значения до объединения
        df["ТипТовара"] = df["КодКатегории"].notna().map({
            True: "Категория",
            False: "Номенклатура"
        })

        # Объединяем номенклатуру и категории
        df["КодНоменклатуры"] = df["КодКатегории"].combine_first(
            df["КодНоменклатурыПервый"]).astype(str)
        df = df.drop(columns=["КодКатегории", "КодНоменклатурыПервый"])

        # --- Подтягиваем данные из Номенклатуры.csv ---
        nom_path = "ВходныеДанные/Номенклатура.csv"

        if not os.path.isfile(nom_path):
            self.show_custom_message(
                title="Ошибка",
                text="Для корректной загрузки необходимо сначала загрузить файл Номенклатура.csv",
                image_path="Картинки/Неудача.png"
            )
            return None, None

        # Читаем файл номенклатуры
        nom_df = pd.read_csv(nom_path, sep="|", dtype=str)

        # Проверяем обязательные колонки
        required_nom_cols = ["КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте"]
        missing_nom = [col for col in required_nom_cols if col not in nom_df.columns]

        if missing_nom:
            self.show_custom_message(
                title="Ошибка",
                text="В файле Номенклатура.csv отсутствуют колонки:\n" + "\n".join(missing_nom),
                image_path="Картинки/Неудача.png"
            )
            return None, None

        # Оставляем только нужные колонки
        nom_df = nom_df[required_nom_cols]

        # Объединяем основной df с номенклатурой
        df = df.merge(nom_df, on="КодНоменклатуры", how="left")

        # --- Подтягиваем данные из КатегорииСайта.csv ---
        cat_path = "ВходныеДанные/КатегорииСайта.csv"

        if not os.path.isfile(cat_path):
            self.show_custom_message(
                title="Ошибка",
                text="Для корректной загрузки необходимо сначала загрузить файл КатегорииСайта.csv",
                image_path="Картинки/Неудача.png"
            )
            return None, None

        # Читаем файл категорий
        cat_df = pd.read_csv(cat_path, sep="|", dtype=str)

        # Проверяем обязательные колонки
        required_cat_cols = ["КодКатегории", "НазваниеКатегории"]
        missing_cat = [col for col in required_cat_cols if col not in cat_df.columns]

        if missing_cat:
            self.show_custom_message(
                title="Ошибка",
                text="В файле КатегорииСайта.csv отсутствуют колонки:\n" + "\n".join(missing_cat),
                image_path="Картинки/Неудача.png"
            )
            return None, None

        # Оставляем только нужные колонки
        cat_df = cat_df[required_cat_cols]

        # Объединяем основной df с номенклатурой
        df = df.merge(cat_df, left_on="КодНоменклатуры", right_on="КодКатегории", how="left")

        # Упорядочиваем колонки
        column_order = [
            "Дата", "КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте",
            "НазваниеКатегории", "ТипТовара", "ДисконтнаяКарта", "MindboxID", "ФИО", "ДатаРождения",
            "Возраст", "ВозрастнаяГруппа", "ПолКлиента", "Почта", "Телефон", "ЛюбимаяКатегория"
        ]
        df = df[column_order]

        df = df.sort_values(by="Дата", ascending=True)

        # Разбиваем файл
        df_full = df.copy()
        df_selection = df_full[~df_full["ВидНоменклатуры"].isin(["Наборы трусов", "Трусы NOS (нижнее белье)",
                                                                 "Трусы (нижнее белье)", "Носки", "Носки NOS"])].copy()

        # возвращаем обработанный DataFrame
        return df_full, df_selection

    # -------------------------------------------ОБРАБОТКА ИЗБРАННОГО---------------------------------------------------
    def process_favorites_file(self, df):
        # Список нужных колонок и новые имена
        columns_map = {
            "CustomerActionDateTimeUtc": "Дата",
            "CustomerActionProductsIdsOffline1C": "КодНоменклатурыРФ",
            "CustomerActionProductsIdsKanzlerKz": "КодНоменклатурыКЗ",
            "CustomerActionCustomerLastActivatedCardIdsNumber": "ДисконтнаяКарта",
            "CustomerActionCustomerIdsMindboxId": "MindboxID",
            "CustomerActionCustomerFirstName": "Имя",
            "CustomerActionCustomerLastName": "Фамилия",
            "CustomerActionCustomerMiddleName": "Отчество",
            "CustomerActionCustomerBirthDate": "ДатаРождения",
            "CustomerActionCustomerSex": "ПолКлиента",
            "CustomerActionCustomerEmail": "Почта",
            "CustomerActionCustomerMobilePhone": "ТелефонОсновной",
            "CustomerActionCustomerPendingMobilePhone": "ЗапаснойТелефон",
            "CustomerActionCustomerCustomFieldsMostViewedCategory": "СамаяПросматриваемаяКатегория",
            "CustomerActionCustomerCustomFieldsMostViewedRootCategory": "СамаяПросматриваемаяРодительскаяКатегория",
            "CustomerActionCustomerCustomFieldsMostViewedSubsidiaryCategory": "СамаяПросматриваемаяДочерняяКатегория",
            "CustomerActionActionTemplateIdsSystemName": "ТипОперации"
        }

        # Проверка наличия всех колонок
        missing = [col for col in columns_map.keys() if col not in df.columns]

        if missing:
            self.show_custom_message(title="Ошибка",
                                     text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
                                     image_path="Картинки/Неудача.png")
            self.status_label.setText(" Ошибка обработки")

            return None, None

        # Оставляем только нужные колонки
        df = df[list(columns_map.keys())]

        # Переименовываем
        df = df.rename(columns=columns_map)

        # ---------- Обработка данных ----------

        # Удаляем строки, где ТипОперации = DobavlenieProduktaVSpisokVOperaciiUstanovka
        df = df[df["ТипОперации"] != "DobavlenieProduktaVSpisokVOperaciiUstanovka"]

        # Удаляем строки, где не заполнено НИ одно из полей
        # "КодНоменклатурыРФ", "КодНоменклатурыКЗ" или "ПросмотреннаяКатегория"
        df = df.dropna(subset=["КодНоменклатурыРФ", "КодНоменклатурыКЗ"], how="all")

        # Объединяем коды номенклатуры
        df["КодНоменклатуры"] = df["КодНоменклатурыРФ"].combine_first(df["КодНоменклатурыКЗ"]).astype(str).str[:6]
        df = df.drop(columns=["КодНоменклатурыРФ", "КодНоменклатурыКЗ"])

        # Объединяем номера телефонов
        df["Телефон"] = (df["ТелефонОсновной"].combine_first(df["ЗапаснойТелефон"])
                         .apply(
            lambda x: str(x)[:-2] if pd.notnull(x) and str(x).endswith(".0") else str(x) if pd.notnull(x) else np.nan))
        df = df.drop(columns=["ТелефонОсновной", "ЗапаснойТелефон"])

        # В колонке Дата оставляем только дату (убираем время)
        df["Дата"] = pd.to_datetime(df["Дата"], errors='coerce').dt.date

        df["ДатаРождения"] = pd.to_datetime(df["ДатаРождения"], errors='coerce')

        # Рассчитываем возраст на момент заказа
        df["Возраст"] = df.apply(
            lambda row: row["Дата"].year - row["ДатаРождения"].year -
                        ((row["Дата"].month, row["Дата"].day) < (row["ДатаРождения"].month, row["ДатаРождения"].day))
            if pd.notnull(row["ДатаРождения"]) and pd.notnull(row["Дата"]) else np.nan,
            axis=1
        ).astype('Int64')

        # Добавляем колонку с возрастной группой
        df["ВозрастнаяГруппа"] = df["Возраст"].apply(self.get_age_group)

        # Объединяем Имя, Фамилия, Отчество в ФИО
        df["ФИО"] = df[["Фамилия", "Имя", "Отчество"]].fillna("").agg(" ".join, axis=1).str.strip()
        df = df.drop(columns=["Имя", "Фамилия", "Отчество"])

        # В колонке Пол заменяем значения
        df["ПолКлиента"] = df["ПолКлиента"].replace({"male": "Мужской", "female": "Женский"})

        # Дисконтную карту приводим к строке, убираем ".0", если он был числом
        df["ДисконтнаяКарта"] = df["ДисконтнаяКарта"].apply(
            lambda x: str(int(x)) if pd.notnull(x) and str(x).endswith(".0") else str(x))

        # Айди майндбокса приводим к строке, убираем ".0", если он был числом
        df["MindboxID"] = df["MindboxID"].apply(
            lambda x: str(int(x)) if pd.notnull(x) and str(x).endswith(".0") else str(x))

        # Очистка категорий от точек и запятых
        for col in [
            "СамаяПросматриваемаяКатегория",
            "СамаяПросматриваемаяРодительскаяКатегория",
            "СамаяПросматриваемаяДочерняяКатегория"
        ]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)  # убираем только .0 в конце
                .str.replace(r"[.,]", "", regex=True)  # убираем лишние знаки, если вдруг есть
                .str.strip()
            )

        # Объединяем любимые категории в одну колонку через "_"
        df["ЛюбимаяКатегория"] = (df[["СамаяПросматриваемаяКатегория",
                                      "СамаяПросматриваемаяРодительскаяКатегория",
                                      "СамаяПросматриваемаяДочерняяКатегория"]]
                                  .apply(lambda row: "_".join([str(x) for x in row if pd.notnull(x) and x != "nan"]),
                                         axis=1))

        # Удалим отдельные колонки категорий
        df = df.drop(columns=["СамаяПросматриваемаяКатегория",
                              "СамаяПросматриваемаяРодительскаяКатегория",
                              "СамаяПросматриваемаяДочерняяКатегория"])

        # --- Подтягиваем данные из Номенклатуры.csv ---
        fav_path = "ВходныеДанные/Номенклатура.csv"

        if not os.path.isfile(fav_path):
            self.show_custom_message(
                title="Ошибка",
                text="Для корректной загрузки необходимо сначала загрузить файл Номенклатура.csv",
                image_path="Картинки/Неудача.png"
            )
            return None, None

        # Читаем файл номенклатуры
        fav_df = pd.read_csv(fav_path, sep="|", dtype=str)

        # Проверяем обязательные колонки
        required_fav_cols = ["КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте"]
        missing_fav = [col for col in required_fav_cols if col not in fav_df.columns]

        if missing_fav:
            self.show_custom_message(
                title="Ошибка",
                text="В файле Номенклатура.csv отсутствуют колонки:\n" + "\n".join(missing_fav),
                image_path="Картинки/Неудача.png"
            )
            return None, None

        # Оставляем только нужные колонки
        fav_df = fav_df[required_fav_cols]

        # Объединяем основной df с номенклатурой
        df = df.merge(fav_df, on="КодНоменклатуры", how="left")

        # Заменяем все виды пустых значений на np.nan
        df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "nan"], np.nan)

        # Упорядочиваем колонки
        column_order = [
            "Дата", "КодНоменклатуры", "Номенклатура", "ВидНоменклатуры", "НазваниеНаСайте",
            "ДисконтнаяКарта", "MindboxID", "ФИО", "ДатаРождения", "Возраст", "ВозрастнаяГруппа", "ПолКлиента",
            "Почта", "Телефон", "ЛюбимаяКатегория"
        ]
        df = df[column_order]

        df = df.sort_values(by="Дата", ascending=True)

        # Разбиваем файл
        df_full = df.copy()
        df_selection = df_full[~df_full["ВидНоменклатуры"].isin(["Наборы трусов", "Трусы NOS (нижнее белье)",
                                                                 "Трусы (нижнее белье)", "Носки", "Носки NOS"])].copy()

        # возвращаем обработанный DataFrame
        return df_full, df_selection

    # -------------------------------------------ВОЗРАСТНАЯ ГРУППА------------------------------------------------------
    @staticmethod
    def get_age_group(age: int) -> str:
        if pd.isnull(age):
            return "Не указан"
        if age < 14:
            return "до 14"
        elif 14 <= age <= 25:
            return "14-25"
        elif 26 <= age <= 35:
            return "26-35"
        elif 36 <= age <= 45:
            return "36-45"
        elif 46 <= age <= 55:
            return "46-55"
        elif 56 <= age <= 65:
            return "56-65"
        else:
            return "65+"

    # -------------------------------------------ОБРАБОТКА НОМНЕКЛАТУРЫ-------------------------------------------------
    def process_nomenclature_file(self, df):

        # Список нужных колонок
        required_columns = [
            "КодНоменклатуры", "Номенклатура", "НазваниеНаСайте", "ВидНоменклатуры",
            "ВидАссортимента", "Марка", "Коллекция", "СезонНоски", "ПолНоменклатуры",
            "ГруппаСоставов", "КатегорияНаСайте", "СтилеваяГруппа", "ТитульнаяФотография"
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            self.show_custom_message(title="Ошибка",
                                     text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
                                     image_path="Картинки/Неудача.png")
            self.status_label.setText(" Ошибка обработки")

            return None

        # Удаляем строки, где КодНоменклатуры пустой, None, NaN или только пробелы
        df = df[df["КодНоменклатуры"].notna()]  # убираем NaN
        df = df[df["КодНоменклатуры"].astype(str).str.strip() != ""]  # убираем пустые и пробельные

        # Очистка категорий от точек и запятых
        for col in [
            "КатегорияНаСайте",
        ]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)  # убираем только .0 в конце
                .str.replace(r"[.,]", "", regex=True)  # убираем лишние знаки, если вдруг есть
                .str.strip()
            )

        # Заменяем все виды пустых значений на np.nan
        df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "nan"], np.nan)

        # Упорядочиваем колонки
        column_order = [
            "КодНоменклатуры", "Номенклатура", "НазваниеНаСайте", "ВидНоменклатуры",
            "ВидАссортимента", "Марка", "Коллекция", "СезонНоски", "ПолНоменклатуры",
            "ГруппаСоставов", "КатегорияНаСайте", "СтилеваяГруппа", "ТитульнаяФотография"
        ]
        df = df[column_order]

        df = df.sort_values(by="КодНоменклатуры", ascending=True)

        # возвращаем обработанный DataFrame
        return df

    # -------------------------------------------ОБРАБОТКА КАТЕГОРИЙ----------------------------------------------------
    def process_categories_file(self, df):

        # Список нужных колонок
        required_columns = [
            "КодКатегории", "НазваниеКатегории", "КодРодительскойКатегории"
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            self.show_custom_message(title="Ошибка",
                                     text="В загруженном файле отсутствуют необходимые колонки:\n" + "\n".join(missing),
                                     image_path="Картинки/Неудача.png")
            self.status_label.setText(" Ошибка обработки")

            return None

        # Очистка категорий от точек и запятых
        for col in [
            "КодКатегории",
            "КодРодительскойКатегории"
        ]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)  # убираем только .0 в конце
                .str.replace(r"[.,]", "", regex=True)  # убираем лишние знаки, если вдруг есть
                .str.strip()
            )

        # Заменяем все виды пустых значений на np.nan
        df = df.replace(["", " ", "  ", "None", "none", "NULL", "null", "-", "--", "nan"], np.nan)

        # Упорядочиваем колонки
        column_order = ["КодКатегории", "НазваниеКатегории", "КодРодительскойКатегории"]

        df = df[column_order]

        df = df.sort_values(by="КодКатегории", ascending=True)

        # возвращаем обработанный DataFrame
        return df

    # -------------------------------------------ОБНОВЛЕНИЕ СТАТУСА ЗАГРУЗКИ--------------------------------------------
    def update_file_status(self):

        input_dir = os.path.join(os.getcwd(), "ВходныеДанные")

        files = {
            "Заказы": "ЗаказыОригинал.csv",
            "Просмотры": "ПросмотрыОригинал.csv",
            "Избранное": "ИзбранноеОригинал.csv",
            "Номенклатура": "Номенклатура.csv",
            "Категории": "КатегорииСайта.csv"
        }

        result = {}
        for title, filename in files.items():
            path = os.path.join(input_dir, filename)
            result[title] = os.path.exists(path)

        text = "Статус загрузки: " + ", ".join(
            f"{k} {'✔️' if v else '✖️'}"
            for k, v in result.items())

        self.status_files_label.setText(text)

    # -------------------------------------------УВЕДОМЛЕНИЕ ДЛЯ ПОЛЬЗОВАТЕЛЯ-------------------------------------------
    def show_custom_message(self, title: str, text: str, image_path: str = None):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(QMessageBox.Icon.NoIcon)  # Без стандартной иконки

        if image_path:  # Если передан путь к картинке
            pix = QPixmap(image_path)
            msg.setIconPixmap(pix.scaled(45, 45, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))

        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    # -------------------------------------------ПОДВКЛАДКА ЗАКАЗЫ------------------------------------------------------
    def create_static_order_full_tab(self):

        tab = QWidget()
        self.main_order_full_layout = QVBoxLayout()
        tab.setLayout(self.main_order_full_layout)

        self.order_full_stats_label = QLabel("")
        self.main_order_full_layout.addWidget(self.order_full_stats_label)

        self.static_tabs.addTab(tab, "Заказы")

    def create_static_order_selection_tab(self):

        tab = QWidget()
        self.main_order_selection_layout = QVBoxLayout()
        tab.setLayout(self.main_order_selection_layout)

        self.order_selection_stats_label = QLabel("")
        self.main_order_selection_layout.addWidget(self.order_selection_stats_label)

        self.static_tabs.addTab(tab, "Заказы (без трусов и носков)")

    # -------------------------------------------ПОДВКЛАДКА ПРОСМОТРЫ---------------------------------------------------
    def create_static_views_full_tab(self):

        tab = QWidget()
        self.main_views_full_layout = QVBoxLayout()
        tab.setLayout(self.main_views_full_layout)

        self.views_full_stats_label = QLabel("")
        self.main_views_full_layout.addWidget(self.views_full_stats_label)

        self.static_tabs.addTab(tab, "Просмотры")

    def create_static_views_selection_tab(self):

        tab = QWidget()
        self.main_views_selection_layout = QVBoxLayout()
        tab.setLayout(self.main_views_selection_layout)

        self.views_selection_stats_label = QLabel("")
        self.main_views_selection_layout.addWidget(self.views_selection_stats_label)

        self.static_tabs.addTab(tab, "Просмотры (без трусов и носков)")

    # -------------------------------------------ПОДВКЛАДКА ИЗБРАННОЕ---------------------------------------------------
    def create_static_favorites_full_tab(self):

        tab = QWidget()
        self.main_favorites_full_layout = QVBoxLayout()
        tab.setLayout(self.main_favorites_full_layout)

        self.favorites_full_stats_label = QLabel("")
        self.main_favorites_full_layout.addWidget(self.favorites_full_stats_label)

        self.static_tabs.addTab(tab, "Избранное")

    def create_static_favorites_selection_tab(self):

        tab = QWidget()
        self.main_favorites_selection_layout = QVBoxLayout()
        tab.setLayout(self.main_favorites_selection_layout)

        self.favorites_selection_stats_label = QLabel("")
        self.main_favorites_selection_layout.addWidget(self.favorites_selection_stats_label)

        self.static_tabs.addTab(tab, "Избранное (без трусов и носков)")

    # -------------------------------------------АНАЛИЗ ФАЙЛОВ----------------------------------------------------------
    def run_analysis(self):

        # Статус бар
        self.status_label.setText("Обработка данных...")

        selected_type = self.combo_box_types.currentText()

        if selected_type == "Заказы клиентов из Mindbox":

            # Проверяем, существует ли файл
            if not os.path.isfile("ВходныеДанные/ЗаказыОригинал.csv"):
                self.show_custom_message(title="Ошибка",
                                         text="Необходимо загрузить файл Заказы.csv перед началом анализа",
                                         image_path="Картинки/Неудача.png")
                return

            self.analyze_orders_full_dataset()

            # Проверяем, существует ли файл
            if not os.path.isfile("ВходныеДанные/ЗаказыОтбор.csv"):
                self.show_custom_message(title="Ошибка",
                                         text="Необходимо загрузить файл Заказы.csv перед началом анализа",
                                         image_path="Картинки/Неудача.png")
                return

            self.analyze_orders_selection_dataset()

        elif selected_type == "Просмотры товаров и категорий из Mindbox":

            # Проверяем, существует ли файл
            if not os.path.isfile("ВходныеДанные/ПросмотрыОригинал.csv"):
                self.show_custom_message(title="Ошибка",
                                         text="Необходимо загрузить файл Просмотры.csv перед началом анализа",
                                         image_path="Картинки/Неудача.png")
                return

            self.analyze_views_full_dataset()

            # Проверяем, существует ли файл
            if not os.path.isfile("ВходныеДанные/ПросмотрыОтбор.csv"):
                self.show_custom_message(title="Ошибка",
                                         text="Необходимо загрузить файл Просмотры.csv перед началом анализа",
                                         image_path="Картинки/Неудача.png")
                return

            self.analyze_views_selection_dataset()

        elif selected_type == "Добавление товаров в избранное из Mindbox":

            # Проверяем, существует ли файл
            if not os.path.isfile("ВходныеДанные/ИзбранноеОригинал.csv"):
                self.show_custom_message(title="Ошибка",
                                         text="Необходимо загрузить файл Избранное.csv перед началом анализа",
                                         image_path="Картинки/Неудача.png")
                return

            self.analyze_favorites_full_dataset()

            # Проверяем, существует ли файл
            if not os.path.isfile("ВходныеДанные/ИзбранноеОтбор.csv"):
                self.show_custom_message(title="Ошибка",
                                         text="Необходимо загрузить файл Избранное.csv перед началом анализа",
                                         image_path="Картинки/Неудача.png")
                return

            self.analyze_favorites_selection_dataset()

        else:
            self.show_custom_message(title="Ошибка",
                                     text="Данный тип данных не подходит для анализа",
                                     image_path="Картинки/Неудача.png")

        self.status_label.setText("Обработка завершена")

    # -------------------------------------------ВЫВОД ЗАГЛУШЕК АНАЛИЗА-------------------------------------------------
    @staticmethod
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

    # -------------------------------------------АНАЛИЗ ЗАКЗАОВ---------------------------------------------------------
    def analyze_orders_full_dataset(self):
        try:
            file_path = "ВходныеДанные/ЗаказыОригинал.csv"

            # Проверяем, существует ли файл
            if not os.path.isfile(file_path):
                self.vyvod_zaglyschek(text="Файл ещё не загружен", icon="Картинки/Внимание.png",
                                      main_layout=self.main_order_full_layout, stats_label=self.order_full_stats_label)
                return

            self.order_full_stats_label = self.reset_layout_with_label(self.main_order_full_layout)

            # Загружаем CSV
            df = pd.read_csv(file_path, sep="|", dtype=str)

            # Числовые поля
            df["Количество"] = pd.to_numeric(df["Количество"], errors="coerce").fillna(0).astype(int)
            df["КонечнаяСтоимость"] = pd.to_numeric(df["КонечнаяСтоимость"], errors="coerce").fillna(0).astype(float)
            df["НачальнаяСтоимость"] = pd.to_numeric(df["НачальнаяСтоимость"], errors="coerce").fillna(0).astype(float)
            df["ПроцентСкидки"] = pd.to_numeric(df["ПроцентСкидки"], errors="coerce").fillna(0).astype(float)
            df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce").fillna(0).astype(int)

            # Дата
            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")

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

            # Разреженность
            num_clients = total_clients
            num_items = total_products
            interactions = total_orders

            sparsity = 1 - interactions / (num_clients * num_items)
            sparsity = round(sparsity * 100, 2)

            # Период
            period_start = df["Дата"].min().date()
            period_end = df["Дата"].max().date()

            period_start_str = period_start.strftime("%d.%m.%Y")
            period_end_str = period_end.strftime("%d.%m.%Y")

            period_str = f"{period_start_str} — {period_end_str}"

            # Месяц с наибольшим количеством продаж
            months_ru = {
                1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
                5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
                9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
            }

            df["Месяц"] = df["Дата"].dt.month
            df["Год"] = df["Дата"].dt.year

            # Суммируем продажи по месяцу и году
            month_sales = df.groupby(["Год", "Месяц"])["Количество"].sum()

            # Находим месяц с максимальными продажами
            top_month_index = month_sales.idxmax()
            top_year, top_month_num = top_month_index

            top_month_str = f"{months_ru[top_month_num]} {top_year}"

            # Популярные товары
            top_codes = (
                df.groupby("КодНоменклатуры")["Количество"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )

            # Формируем текст, используя уже доступную колонку НазваниеНаСайте
            top_products_pretty = []
            for code, count in top_codes.items():
                # Берём первое встречающееся название у этого кода
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
                # Общая сумма продаж (целое число)
                total_sales = int(df_cur["КонечнаяСтоимость"].sum())
                total_sales_str = f"{total_sales:,}".replace(",", ".")

                # Средний чек — считаем суммы заказов → берём среднее → целое число
                order_totals = df_cur.groupby("НомерЗаказа")["КонечнаяСтоимость"].sum()
                avg_check = int(order_totals.mean())
                avg_check_str = f"{avg_check:,}".replace(",", ".")

                # Средняя скидка — целое число
                avg_discount = int(round(df_cur["ПроцентСкидки"].mean(), 0))

                total_sales_dict[currency] = total_sales_str
                avg_check_dict[currency] = avg_check_str
                avg_discount_dict[currency] = avg_discount

            # Формирование строки вывода
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

            # Топ магазинов по количеству проданных товаров
            top_stores = df.groupby("Магазин")["Количество"].sum().sort_values(ascending=False).head(5)

            # Форматируем красиво
            top_stores_text = "\n".join(
                [f"{store} — {int(count)}" for store, count in top_stores.items()]
            )

            # Формируем текст
            result = (
                f"Количество продаж: {total_pokypok}\n"
                f"Количество заказов: {total_orders}\n"
                f"Количество клиентов: {total_clients}\n"
                f"Количество товаров: {total_products}\n\n"
                f"Распределение продаж по полу: \n{gender_text}\n\n"
                f"Возраст клиента (средний/медианный): {avg_age} / {median_age}\n"
                f"Преобладающая возрастная категория: {top_age_group}\n\n"
                f"Количество продаж на клиента (среднее/медианное): {avg_purchases} / {median_purchases}\n\n"
                f"Разреженность матрицы клиент-товар: {sparsity}%\n\n"
                f"Период: {period_str}\n"
                f"Месяц с наибольшим количеством продаж: {top_month_str}\n\n"
                f"{output}\n\n"
                f"Топ-5 товаров по количеству продаж: \n{top_products_text}\n\n"
                f"Топ-5 магазинов по количеству продаж: \n{top_stores_text}"
            )

            self.order_full_stats_label.setText(result)

        except Exception as e:
            self.vyvod_zaglyschek(text=f"Ошибка при анализе файла: {e}", icon="Картинки/Неудача.png",
                                  main_layout=self.main_order_full_layout, stats_label=self.order_full_stats_label)

    def analyze_orders_selection_dataset(self):
        try:
            file_path = "ВходныеДанные/ЗаказыОтбор.csv"

            # Проверяем, существует ли файл
            if not os.path.isfile(file_path):
                self.vyvod_zaglyschek(text="Файл ещё не загружен", icon="Картинки/Внимание.png",
                                      main_layout=self.main_order_selection_layout,
                                      stats_label=self.order_selection_stats_label)
                return

            self.order_selection_stats_label = self.reset_layout_with_label(self.main_order_selection_layout)

            # Загружаем CSV
            df = pd.read_csv(file_path, sep="|", dtype=str)

            # Числовые поля
            df["Количество"] = pd.to_numeric(df["Количество"], errors="coerce").fillna(0).astype(int)
            df["КонечнаяСтоимость"] = pd.to_numeric(df["КонечнаяСтоимость"], errors="coerce").fillna(0).astype(float)
            df["НачальнаяСтоимость"] = pd.to_numeric(df["НачальнаяСтоимость"], errors="coerce").fillna(0).astype(float)
            df["ПроцентСкидки"] = pd.to_numeric(df["ПроцентСкидки"], errors="coerce").fillna(0).astype(float)
            df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce").fillna(0).astype(int)

            # Дата
            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")

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

            # Разреженность
            num_clients = total_clients
            num_items = total_products
            interactions = total_orders

            sparsity = 1 - interactions / (num_clients * num_items)
            sparsity = round(sparsity * 100, 2)

            # Период
            period_start = df["Дата"].min().date()
            period_end = df["Дата"].max().date()

            period_start_str = period_start.strftime("%d.%m.%Y")
            period_end_str = period_end.strftime("%d.%m.%Y")

            period_str = f"{period_start_str} — {period_end_str}"

            # Месяц с наибольшим количеством продаж
            months_ru = {
                1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
                5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
                9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
            }

            df["Месяц"] = df["Дата"].dt.month
            df["Год"] = df["Дата"].dt.year

            # Суммируем продажи по месяцу и году
            month_sales = df.groupby(["Год", "Месяц"])["Количество"].sum()

            # Находим месяц с максимальными продажами
            top_month_index = month_sales.idxmax()
            top_year, top_month_num = top_month_index

            top_month_str = f"{months_ru[top_month_num]} {top_year}"

            # Популярные товары
            top_codes = (
                df.groupby("КодНоменклатуры")["Количество"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )

            # Формируем текст, используя уже доступную колонку НазваниеНаСайте
            top_products_pretty = []
            for code, count in top_codes.items():
                # Берём первое встречающееся название у этого кода
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
                # Общая сумма продаж (целое число)
                total_sales = int(df_cur["КонечнаяСтоимость"].sum())
                total_sales_str = f"{total_sales:,}".replace(",", ".")

                # Средний чек — считаем суммы заказов → берём среднее → целое число
                order_totals = df_cur.groupby("НомерЗаказа")["КонечнаяСтоимость"].sum()
                avg_check = int(order_totals.mean())
                avg_check_str = f"{avg_check:,}".replace(",", ".")

                # Средняя скидка — целое число
                avg_discount = int(round(df_cur["ПроцентСкидки"].mean(), 0))

                total_sales_dict[currency] = total_sales_str
                avg_check_dict[currency] = avg_check_str
                avg_discount_dict[currency] = avg_discount

            # Формирование строки вывода
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

            # Топ магазинов по количеству проданных товаров
            top_stores = df.groupby("Магазин")["Количество"].sum().sort_values(ascending=False).head(5)

            # Форматируем красиво
            top_stores_text = "\n".join(
                [f"{store} — {int(count)}" for store, count in top_stores.items()]
            )

            # Формируем текст
            result = (
                f"Количество продаж: {total_pokypok}\n"
                f"Количество заказов: {total_orders}\n"
                f"Количество клиентов: {total_clients}\n"
                f"Количество товаров: {total_products}\n\n"
                f"Распределение продаж по полу: \n{gender_text}\n\n"
                f"Возраст клиента (средний/медианный): {avg_age} / {median_age}\n"
                f"Преобладающая возрастная категория: {top_age_group}\n\n"
                f"Количество продаж на клиента (среднее/медианное): {avg_purchases} / {median_purchases}\n\n"
                f"Разреженность матрицы клиент-товар: {sparsity}%\n\n"
                f"Период: {period_str}\n"
                f"Месяц с наибольшим количеством продаж: {top_month_str}\n\n"
                f"{output}\n\n"
                f"Топ-5 товаров по количеству продаж: \n{top_products_text}\n\n"
                f"Топ-5 магазинов по количеству продаж: \n{top_stores_text}"
            )

            self.order_selection_stats_label.setText(result)

        except Exception as e:
            self.vyvod_zaglyschek(text=f"Ошибка при анализе файла: {e}", icon="Картинки/Неудача.png",
                                  main_layout=self.main_order_selection_layout,
                                  stats_label=self.order_selection_stats_label)

    # -------------------------------------------АНАЛИЗ ПРОСМОТРОВ------------------------------------------------------
    def analyze_views_full_dataset(self):
        try:
            file_path = "ВходныеДанные/ПросмотрыОригинал.csv"

            # Проверяем, существует ли файл
            if not os.path.isfile(file_path):
                self.vyvod_zaglyschek(text="Файл ещё не загружен", icon="Картинки/Внимание.png",
                                      main_layout=self.main_views_full_layout,
                                      stats_label=self.views_full_stats_label)
                return

            # Загружаем CSV
            df = pd.read_csv(file_path, sep="|", dtype=str)

            # Числовые поля
            df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce").fillna(0).astype(int)

            # Дата
            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")

            # Количество просмотров
            total_views = len(df)
            total_clients = df["MindboxID"].nunique()
            total_products = df.loc[df["ТипТовара"] == "Номенклатура", "КодНоменклатуры"].nunique()
            total_categories = df.loc[df["ТипТовара"] == "Категория", "КодНоменклатуры"].nunique()

            # Просмотры клиента
            views_per_client = df.groupby("MindboxID").size()
            avg_views = round(views_per_client.mean(), 1)
            median_views = views_per_client.median()

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
            period_start = df["Дата"].min().date()
            period_end = df["Дата"].max().date()

            period_start_str = period_start.strftime("%d.%m.%Y")
            period_end_str = period_end.strftime("%d.%m.%Y")

            period_str = f"{period_start_str} — {period_end_str}"

            # Месяц с наибольшим количеством просмотров
            months_ru = {
                1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
                5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
                9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
            }

            df["Месяц"] = df["Дата"].dt.month
            df["Год"] = df["Дата"].dt.year

            # Суммируем просмотры по месяцу и году
            month_sales = df.groupby(["Год", "Месяц"]).size()

            # Находим месяц с максимальными просмотрами
            top_month_index = month_sales.idxmax()
            top_year, top_month_num = top_month_index

            top_month_str = f"{months_ru[top_month_num]} {top_year}"

            # Притягиваем номенклатуру
            df_nom = df[df["ТипТовара"] == "Номенклатура"]

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

            # ---Подтягиваем категории---
            df_cat = df[df["ТипТовара"] == "Категория"]

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

            # Формируем текст
            result = (
                f"Количество просмотров: {total_views}\n"
                f"Количество клиентов: {total_clients}\n"
                f"Количество товаров: {total_products}\n"
                f"Количество категорий: {total_categories}\n\n"
                f"Распределение просмотров по полу: \n{gender_text}\n\n"
                f"Возраст клиента (средний/медианный): {avg_age} / {median_age}\n"
                f"Преобладающая возрастная категория: {top_age_group}\n\n"
                f"Количество просмотров на клиента (среднее/медианное): {avg_views} / {median_views}\n\n"
                f"Период: {period_str}\n"
                f"Месяц с наибольшим количеством просмотров: {top_month_str}\n\n"
                f"Топ-5 товаров по количеству просмотров: \n{top_products_text}\n\n"
                f"Топ-5 категорий по количеству просмотров: \n{top_cat_text}"
            )

            self.views_full_stats_label = self.reset_layout_with_label(self.main_views_full_layout)
            self.views_full_stats_label.setText(result)

        except Exception as e:
            self.vyvod_zaglyschek(text=f"Ошибка при анализе файла: {e}", icon="Картинки/Неудача.png",
                                  main_layout=self.main_views_full_layout, stats_label=self.views_full_stats_label)

    def analyze_views_selection_dataset(self):
        try:
            file_path = "ВходныеДанные/ПросмотрыОтбор.csv"

            # Проверяем, существует ли файл
            if not os.path.isfile(file_path):
                self.vyvod_zaglyschek(text="Файл ещё не загружен", icon="Картинки/Внимание.png",
                                      main_layout=self.main_views_selection_layout,
                                      stats_label=self.views_selection_stats_label)
                return

            # Загружаем CSV
            df = pd.read_csv(file_path, sep="|", dtype=str)

            # Числовые поля
            df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce").fillna(0).astype(int)

            # Дата
            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")

            # Количество просмотров
            total_views = len(df)
            total_clients = df["MindboxID"].nunique()
            total_products = df.loc[df["ТипТовара"] == "Номенклатура", "КодНоменклатуры"].nunique()
            total_categories = df.loc[df["ТипТовара"] == "Категория", "КодНоменклатуры"].nunique()

            # Просмотры клиента
            views_per_client = df.groupby("MindboxID").size()
            avg_views = round(views_per_client.mean(), 1)
            median_views = views_per_client.median()

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
            period_start = df["Дата"].min().date()
            period_end = df["Дата"].max().date()

            period_start_str = period_start.strftime("%d.%m.%Y")
            period_end_str = period_end.strftime("%d.%m.%Y")

            period_str = f"{period_start_str} — {period_end_str}"

            # Месяц с наибольшим количеством просмотров
            months_ru = {
                1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
                5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
                9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
            }

            df["Месяц"] = df["Дата"].dt.month
            df["Год"] = df["Дата"].dt.year

            # Суммируем просмотры по месяцу и году
            month_sales = df.groupby(["Год", "Месяц"]).size()

            # Находим месяц с максимальными просмотрами
            top_month_index = month_sales.idxmax()
            top_year, top_month_num = top_month_index

            top_month_str = f"{months_ru[top_month_num]} {top_year}"

            # Притягиваем номенклатуру
            df_nom = df[df["ТипТовара"] == "Номенклатура"]

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

            # ---Подтягиваем категории---
            df_cat = df[df["ТипТовара"] == "Категория"]

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

            # Формируем текст
            result = (
                f"Количество просмотров: {total_views}\n"
                f"Количество клиентов: {total_clients}\n"
                f"Количество товаров: {total_products}\n"
                f"Количество категорий: {total_categories}\n\n"
                f"Распределение просмотров по полу: \n{gender_text}\n\n"
                f"Возраст клиента (средний/медианный): {avg_age} / {median_age}\n"
                f"Преобладающая возрастная категория: {top_age_group}\n\n"
                f"Количество просмотров на клиента (среднее/медианное): {avg_views} / {median_views}\n\n"
                f"Период: {period_str}\n"
                f"Месяц с наибольшим количеством просмотров: {top_month_str}\n\n"
                f"Топ-5 товаров по количеству просмотров: \n{top_products_text}\n\n"
                f"Топ-5 категорий по количеству просмотров: \n{top_cat_text}"
            )

            self.views_selection_stats_label = self.reset_layout_with_label(self.main_views_selection_layout)
            self.views_selection_stats_label.setText(result)

        except Exception as e:
            self.vyvod_zaglyschek(text=f"Ошибка при анализе файла: {e}", icon="Картинки/Неудача.png",
                                  main_layout=self.main_views_selection_layout,
                                  stats_label=self.views_selection_stats_label)

    # -------------------------------------------АНАЛИЗ ИЗБРАННОГО------------------------------------------------------
    def analyze_favorites_full_dataset(self):
        try:
            file_path = "ВходныеДанные/ИзбранноеОригинал.csv"

            # Проверяем, существует ли файл
            if not os.path.isfile(file_path):
                self.vyvod_zaglyschek(text="Файл ещё не загружен", icon="Картинки/Внимание.png",
                                      main_layout=self.main_favorites_full_layout,
                                      stats_label=self.favorites_full_stats_label)
                return

            # Загружаем CSV
            df = pd.read_csv(file_path, sep="|", dtype=str)

            # Числовые поля
            df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce").fillna(0).astype(int)

            # Дата
            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")

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
            period_start = df["Дата"].min().date()
            period_end = df["Дата"].max().date()

            period_start_str = period_start.strftime("%d.%m.%Y")
            period_end_str = period_end.strftime("%d.%m.%Y")

            period_str = f"{period_start_str} — {period_end_str}"

            # Месяц с наибольшим количеством добавлений
            months_ru = {
                1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
                5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
                9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
            }

            df["Месяц"] = df["Дата"].dt.month
            df["Год"] = df["Дата"].dt.year

            # Суммируем добавления по месяцу и году
            month_sales = df.groupby(["Год", "Месяц"]).size()

            # Находим месяц с максимальными добавлениями
            top_month_index = month_sales.idxmax()
            top_year, top_month_num = top_month_index

            top_month_str = f"{months_ru[top_month_num]} {top_year}"

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

            self.favorites_full_stats_label = self.reset_layout_with_label(self.main_favorites_full_layout)
            self.favorites_full_stats_label.setText(result)

        except Exception as e:
            self.vyvod_zaglyschek(text=f"Ошибка при анализе файла: {e}", icon="Картинки/Неудача.png",
                                  main_layout=self.main_favorites_full_layout,
                                  stats_label=self.favorites_full_stats_label)

    def analyze_favorites_selection_dataset(self):
        try:
            file_path = "ВходныеДанные/ИзбранноеОтбор.csv"

            # Проверяем, существует ли файл
            if not os.path.isfile(file_path):
                self.vyvod_zaglyschek(text="Файл ещё не загружен", icon="Картинки/Внимание.png",
                                      main_layout=self.main_favorites_selection_layout,
                                      stats_label=self.favorites_selection_stats_label)
                return

            # Загружаем CSV
            df = pd.read_csv(file_path, sep="|", dtype=str)

            # Числовые поля
            df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce").fillna(0).astype(int)

            # Дата
            df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")

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
            period_start = df["Дата"].min().date()
            period_end = df["Дата"].max().date()

            period_start_str = period_start.strftime("%d.%m.%Y")
            period_end_str = period_end.strftime("%d.%m.%Y")

            period_str = f"{period_start_str} — {period_end_str}"

            # Месяц с наибольшим количеством добавлений
            months_ru = {
                1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
                5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
                9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
            }

            df["Месяц"] = df["Дата"].dt.month
            df["Год"] = df["Дата"].dt.year

            # Суммируем добавления по месяцу и году
            month_sales = df.groupby(["Год", "Месяц"]).size()

            # Находим месяц с максимальными добавлениями
            top_month_index = month_sales.idxmax()
            top_year, top_month_num = top_month_index

            top_month_str = f"{months_ru[top_month_num]} {top_year}"

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

            self.favorites_selection_stats_label = self.reset_layout_with_label(self.main_favorites_selection_layout)
            self.favorites_selection_stats_label.setText(result)

        except Exception as e:
            self.vyvod_zaglyschek(text=f"Ошибка при анализе файла: {e}", icon="Картинки/Неудача.png",
                                  main_layout=self.main_favorites_selection_layout,
                                  stats_label=self.favorites_selection_stats_label)

    # -------------------------------------------ОЧИСТИТЬ ФОРМУ---------------------------------------------------------
    def reset_layout_with_label(self, layout):

        # Удаляем все виджеты из layout
        while layout.count():
            item = layout.takeAt(0)
            # Если элемент — виджет
            if item.widget():
                item.widget().deleteLater()

            # Если элемент — layout
            elif item.layout():
                self.clear_layout(item.layout())
                item.layout().deleteLater()

        # Создаём новый QLabel
        new_label = QLabel()
        new_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )

        # Добавляем в layout
        layout.addWidget(new_label)

        return new_label

    @staticmethod
    def clear_layout(layout):

        while layout.count():
            item = layout.takeAt(0)

            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                clear_layout(item.layout())
                item.layout().deleteLater()

    # -------------------------------------------СТАНДАРТНЫЕ НАСТРОЙКИ--------------------------------------------------
    def standart_settigs(self):
        # Параметры BPR-MF
        self.embedding_dim_input = QSpinBox()
        self.embedding_dim_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.embedding_dim_input.setRange(0, 10000)
        self.embedding_dim_input.setValue(64)

        self.epochs_input = QSpinBox()
        self.epochs_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.epochs_input.setRange(0, 10000)
        self.epochs_input.setValue(20)

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.batch_size_input.setRange(0, 10000)
        self.batch_size_input.setValue(4096)

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.lr_input.setRange(0, 1)
        self.lr_input.setDecimals(3)
        self.lr_input.setValue(2e-3)

        self.weight_decay_input = QDoubleSpinBox()
        self.weight_decay_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.weight_decay_input.setRange(0, 1)
        self.weight_decay_input.setDecimals(6)
        self.weight_decay_input.setValue(1e-6)

        self.bpr_reg_input = QDoubleSpinBox()
        self.bpr_reg_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.bpr_reg_input.setRange(0, 1)
        self.bpr_reg_input.setDecimals(4)
        self.bpr_reg_input.setValue(1e-4)

        self.seed_input = QSpinBox()
        self.seed_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.seed_input.setRange(0, 10000)
        self.seed_input.setValue(42)

        # Параметры EASE^R
        self.ease_lambda_input = QSpinBox()
        self.ease_lambda_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.ease_lambda_input.setRange(0, 10000)
        self.ease_lambda_input.setValue(200)

        self.max_items_for_ease_input = QSpinBox()
        self.max_items_for_ease_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.max_items_for_ease_input.setRange(0, 100000)
        self.max_items_for_ease_input.setValue(15000)

        # Остальные параметры
        self.w_view_item = QDoubleSpinBox()
        self.w_view_item.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.w_view_item.setRange(0, 10)
        self.w_view_item.setDecimals(1)
        self.w_view_item.setValue(1.0)

        self.w_favorite = QDoubleSpinBox()
        self.w_favorite.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.w_favorite.setRange(0, 10)
        self.w_favorite.setDecimals(1)
        self.w_favorite.setValue(3.0)

        self.w_purchase = QDoubleSpinBox()
        self.w_purchase.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.w_purchase.setRange(0, 10)
        self.w_purchase.setDecimals(1)
        self.w_purchase.setValue(5.0)

        self.top_rec = QSpinBox()
        self.top_rec.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.top_rec.setRange(0, 100)
        self.top_rec.setValue(10)

        self.min_user_interactions_for_eval = QSpinBox()
        self.min_user_interactions_for_eval.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.min_user_interactions_for_eval.setRange(0, 100)
        self.min_user_interactions_for_eval.setValue(2)

    # -------------------------------------------ЛИШНЕЕ-----------------------------------------------------------------
    def create_basic_widgets_tab(self):
        """Создаем вкладку с базовыми виджетами"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Метка с изображением
        label = QLabel("Демонстрация PyQt6 Widgets")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = label.font()
        font.setPointSize(16)
        label.setFont(font)
        layout.addWidget(label)

        # Изображение
        image_label = QLabel()
        pixmap = QPixmap("Картинки/python-logo.png")  # Используем встроенный ресурс
        if not pixmap.isNull():
            image_label.setPixmap(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(image_label)

        # Кнопки
        btn_layout = QHBoxLayout()

        btn1 = QPushButton("Обычная кнопка")
        btn1.clicked.connect(self.on_button_click)
        btn_layout.addWidget(btn1)

        btn2 = QPushButton(QIcon("Картинки/python-logo.png"), "Кнопка с иконкой")
        btn2.clicked.connect(self.on_button_click)
        btn_layout.addWidget(btn2)

        btn3 = QPushButton("Отключенная кнопка")
        btn3.setEnabled(False)
        btn_layout.addWidget(btn3)

        layout.addLayout(btn_layout)

        # Горизонтальная линия
        layout.addWidget(QLabel("<hr>"), 1)

        # Текстовое поле
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Введите текст здесь...")
        self.line_edit.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.line_edit)

        self.tabs.addTab(tab, "Базовые виджеты")

    def create_input_widgets_tab(self):
        """Создаем вкладку с виджетами ввода"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Комбобокс (выпадающий список)
        combo_label = QLabel("Выберите вариант:")
        layout.addWidget(combo_label)

        self.combo_box = QComboBox()
        self.combo_box.addItems(["Вариант 1", "Вариант 2", "Вариант 3", "Вариант 4"])
        self.combo_box.currentIndexChanged.connect(self.on_combo_changed)
        layout.addWidget(self.combo_box)

        # Чекбоксы
        check_label = QLabel("Выберите опции:")
        layout.addWidget(check_label)

        self.check1 = QCheckBox("Опция 1")
        self.check1.stateChanged.connect(self.on_check_changed)
        layout.addWidget(self.check1)

        self.check2 = QCheckBox("Опция 2")
        self.check2.stateChanged.connect(self.on_check_changed)
        layout.addWidget(self.check2)

        # Радио кнопки
        radio_label = QLabel("Выберите один вариант:")
        layout.addWidget(radio_label)

        self.radio1 = QRadioButton("Вариант A")
        self.radio1.toggled.connect(self.on_radio_toggled)
        layout.addWidget(self.radio1)

        self.radio2 = QRadioButton("Вариант B")
        self.radio2.toggled.connect(self.on_radio_toggled)
        layout.addWidget(self.radio2)

        self.radio3 = QRadioButton("Вариант C")
        self.radio3.toggled.connect(self.on_radio_toggled)
        layout.addWidget(self.radio3)

        # Слайдер
        slider_label = QLabel("Регулировка значения:")
        layout.addWidget(slider_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)

        # Спинбоксы
        spin_layout = QHBoxLayout()

        self.spin_box = QSpinBox()
        self.spin_box.setRange(0, 100)
        self.spin_box.setValue(25)
        self.spin_box.valueChanged.connect(self.on_spin_changed)
        spin_layout.addWidget(self.spin_box)

        self.double_spin = QDoubleSpinBox()
        self.double_spin.setRange(0, 10)
        self.double_spin.setSingleStep(0.1)
        self.double_spin.setValue(2.5)
        self.double_spin.valueChanged.connect(self.on_double_spin_changed)
        spin_layout.addWidget(self.double_spin)

        layout.addLayout(spin_layout)

        self.tabs.addTab(tab, "Виджеты ввода")

    def create_indicator_widgets_tab(self):
        """Создаем вкладку с индикаторами"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Прогресс бар
        progress_label = QLabel("Прогресс бар:")
        layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Кнопка для запуска прогресса
        self.progress_btn = QPushButton("Запустить прогресс")
        self.progress_btn.clicked.connect(self.start_progress)
        layout.addWidget(self.progress_btn)

        # Таймер для прогресс бара
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_value = 0

        # Метка для отображения значений
        self.value_label = QLabel("Значения будут отображаться здесь")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)

        self.tabs.addTab(tab, "Индикаторы")

    def create_container_widgets_tab(self):
        """Создаем вкладку с контейнерными виджетами"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Текстовый редактор
        text_edit_label = QLabel("Текстовый редактор:")
        layout.addWidget(text_edit_label)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Введите текст здесь...")
        layout.addWidget(self.text_edit)

        # Кнопки для работы с текстом
        text_btn_layout = QHBoxLayout()

        clear_btn = QPushButton("Очистить")
        clear_btn.clicked.connect(self.text_edit.clear)
        text_btn_layout.addWidget(clear_btn)

        get_text_btn = QPushButton("Показать текст")
        get_text_btn.clicked.connect(self.show_text)
        text_btn_layout.addWidget(get_text_btn)

        layout.addLayout(text_btn_layout)

        # Список
        list_label = QLabel("Список элементов:")
        layout.addWidget(list_label)

        self.list_widget = QListWidget()
        self.list_widget.addItems(["Элемент 1", "Элемент 2", "Элемент 3"])
        self.list_widget.itemClicked.connect(self.on_list_item_clicked)
        layout.addWidget(self.list_widget)

        # Кнопки для работы со списком
        list_btn_layout = QHBoxLayout()

        add_item_btn = QPushButton("Добавить")
        add_item_btn.clicked.connect(self.add_list_item)
        list_btn_layout.addWidget(add_item_btn)

        remove_item_btn = QPushButton("Удалить")
        remove_item_btn.clicked.connect(self.remove_list_item)
        list_btn_layout.addWidget(remove_item_btn)

        layout.addLayout(list_btn_layout)

        self.tabs.addTab(tab, "Контейнеры")

    def on_button_click(self):
        sender = self.sender()
        QMessageBox.information(self, "Кнопка нажата", f"Нажата кнопка: {sender.text()}")

    def on_text_changed(self, text):
        self.statusBar().showMessage(f"Текст изменен: {text}", 2000)

    def on_combo_changed(self, index):
        self.value_label.setText(f"Выбран комбобокс: {self.combo_box.currentText()} (индекс {index})")

    def on_check_changed(self, state):
        sender = self.sender()
        checked = "включена" if state else "отключена"
        self.value_label.setText(f"Опция {sender.text()} {checked}")

    def on_radio_toggled(self, checked):
        if checked:
            sender = self.sender()
            self.value_label.setText(f"Выбран радио-вариант: {sender.text()}")

    def on_slider_changed(self, value):
        self.value_label.setText(f"Значение слайдера: {value}")

    def on_spin_changed(self, value):
        self.value_label.setText(f"Значение спинбокса: {value}")

    def on_double_spin_changed(self, value):
        self.value_label.setText(f"Значение double спинбокса: {value:.2f}")

    def start_progress(self):
        if not self.progress_timer.isActive():
            self.progress_value = 0
            self.progress_bar.setValue(0)
            self.progress_timer.start(100)
            self.progress_btn.setText("Остановить прогресс")
        else:
            self.progress_timer.stop()
            self.progress_btn.setText("Запустить прогресс")

    def update_progress(self):
        self.progress_value += 1
        self.progress_bar.setValue(self.progress_value)

        if self.progress_value >= 100:
            self.progress_timer.stop()
            self.progress_btn.setText("Запустить прогресс")
            QMessageBox.information(self, "Готово", "Прогресс завершен!")

    def show_text(self):
        text = self.text_edit.toPlainText()
        if text:
            QMessageBox.information(self, "Текст", text)
        else:
            QMessageBox.warning(self, "Ошибка", "Текст не введен!")

    def on_list_item_clicked(self, item):
        self.value_label.setText(f"Выбран элемент списка: {item.text()}")

    def add_list_item(self):
        text, ok = QInputDialog.getText(self, "Добавить элемент", "Введите текст элемента:")
        if ok and text:
            self.list_widget.addItem(text)

    def remove_list_item(self):
        current_item = self.list_widget.currentItem()
        if current_item:
            self.list_widget.takeItem(self.list_widget.row(current_item))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # === СВЕТЛАЯ ТЕМА ===
    bright_palette_tymbler = QPalette()
    bright_palette_tymbler.setColor(QPalette.ColorRole.Window, QColor(250, 250, 250))  # общий фон (да)
    bright_palette_tymbler.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)  # текст на фоне (да)
    bright_palette_tymbler.setColor(QPalette.ColorRole.Base, QColor(240, 240, 240))  # поля ввода (да, галочки)
    bright_palette_tymbler.setColor(QPalette.ColorRole.AlternateBase,
                                    QColor(100, 100, 100))  # чередующиеся строки в тч (нет)
    bright_palette_tymbler.setColor(QPalette.ColorRole.ToolTipBase,
                                    QColor(100, 100, 100))  # всплывающие подсказки (нет)
    bright_palette_tymbler.setColor(QPalette.ColorRole.ToolTipText,
                                    Qt.GlobalColor.white)  # текст всплывающих подсказок (нет)
    bright_palette_tymbler.setColor(QPalette.ColorRole.Text,
                                    Qt.GlobalColor.black)  # основной текст внутри полей (да)
    bright_palette_tymbler.setColor(QPalette.ColorRole.Button, QColor(100, 100, 100))  # фон кнопки (нет)
    bright_palette_tymbler.setColor(QPalette.ColorRole.ButtonText,
                                    Qt.GlobalColor.white)  # текст на кнопках (нет)
    bright_palette_tymbler.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)  # текст ошибки (да)
    bright_palette_tymbler.setColor(QPalette.ColorRole.Highlight, QColor(210, 210, 210))  # выделение (да)
    bright_palette_tymbler.setColor(QPalette.ColorRole.HighlightedText,
                                    Qt.GlobalColor.black)  # текст при выделении (да)
    bright_palette_tymbler.setColor(QPalette.ColorRole.PlaceholderText,
                                    QColor(100, 100, 100))  # цвет текста в пустом поле (нет)
    app.setPalette(bright_palette_tymbler)

    # Стиль виджетов
    app.setStyleSheet("""

                                        * {
                                            font-size: 14px;
                                            font-family: "Verdana";
                                            letter-spacing: 0.1px;
                                        }

                                        QTabWidget::pane { 
                                            border: 1px solid #EBEBEB;
                                        }
                                        
                                        QTabBar::tab { 
                                            background: #EBEBEB; 
                                            color: black; 
                                            padding: 6px 10px;
                                            border-radius: 10px;
                                            font-family: "Roboto";
                                        }
                                        QTabBar::tab:selected { background: #DCDCDC; }

                                        QLabel {
                                            padding: 5px 0px;
                                            color: black;
                                        }

                                        QPushButton {
                                            font-weight: 500;
                                            border: 1px solid #9B4DFF;
                                            padding: 6px 10px;
                                            border-radius: 10px;
                                            color: black;
                                            background: qlineargradient(
                                                x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #FFFFFF,
                                                stop:1 #EBEBEB
                                            );
                                        }
                                        
                                        QPushButton:hover { background-color: #D7D7D7; }
                                        QPushButton:pressed { background-color: #C8C8C8; }
                                        QPushButton:disabled { background-color: #373737; color: #888; }

                                        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                                            background-color: #EBEBEB;
                                            color: black;
                                            border: 1px solid #D7D7D7;
                                            border-radius: 10px;
                                            padding: 4px;
                                        }
                                        
                                        QComboBox::drop-down {
                                            width: 24px;
                                            border: none;
                                            background: transparent;
                                        }

                                        QComboBox::down-arrow {
                                            image: url("Картинки/Вниз.png");
                                            width: 18px;
                                            height: 18px;
                                        }

                                        QListWidget {
                                            background-color: #F0F0F0;
                                            color: black;
                                            border: 1px solid #D7D7D7;
                                        }

                                        QProgressBar {
                                            border: 1px solid #D7D7D7;
                                            text-align: center;
                                            color: black;
                                            border-radius: 7px;
                                            background-color: #F0F0F0;
                                        }
                                        QProgressBar::chunk {
                                            background-color: #969696;
                                            border-radius: 7px;
                                        }
                                        QMessageBox {
                                            margin: 0;
                                            padding: 0;
                                        }
                                        QMessageBox QLabel {
                                            padding: 0;
                                            margin: 0;
                                        }
                                        
                                        QFrame[frameRole="separator"] {
                                            background-color: #EBEBEB;
                                            border: none;
                                            min-height: 1px;
                                            max-height: 1px;
                                        }
                                    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
