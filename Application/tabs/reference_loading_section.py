"""Reference CSV selectors and statuses for the data acquisition tab."""

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QGridLayout, QSizePolicy,
)

from Application.paths import ICONS_DIR
from Application.files.reference_import import REFERENCE_TYPES
from Application.theme.layout_metrics import COMPACT_SPACING, CONTROL_SPACING, BLOCK_SPACING


def create_csv_loading_section(aboba):
    """Only reference snapshots are selectable; processing runs in a subprocess."""
    section = QWidget()
    left_layout = QVBoxLayout(section)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(BLOCK_SPACING)

    # Заголовок CSV-раздела
    aboba.heading_load_data = QLabel("Загрузка справочников")
    aboba.heading_load_data.setSizePolicy(
        QSizePolicy.Policy.Fixed,
        QSizePolicy.Policy.Fixed,
    )
    aboba.heading_load_data.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aboba.heading_load_data.setProperty("class", "sectionHeader")
    left_layout.addWidget(
        aboba.heading_load_data,
        alignment=Qt.AlignmentFlag.AlignHCenter,
    )

    aboba.reference_paths = {
        kind: None
        for kind in REFERENCE_TYPES
    }
    aboba.reference_fields = {}
    aboba.reference_buttons = {}
    aboba.reference_controls = []
    aboba.reference_status_overrides = {}

    fields = QGridLayout()
    fields.setContentsMargins(0, 0, 0, 0)
    fields.setSpacing(CONTROL_SPACING)

    button_specs = (
        (
            " Выбрать номенклатуру",
            "nomenclature.png",
        ),
        (
            " Выбрать категории",
            "categories.png",
        ),
        (
            " Выбрать координаты",
            "coordinates.png",
        ),
    )

    for row, (
        (kind, (filename, *_)),
        (button_text, icon_name),
    ) in enumerate(
        zip(
            REFERENCE_TYPES.items(),
            button_specs,
        )
    ):
        editor = QLineEdit()
        editor.setReadOnly(True)
        display_name = "".join(part.capitalize() for part in Path(filename).stem.split("_")) + Path(filename).suffix
        editor.setPlaceholderText(display_name)

        button = QPushButton(
            QIcon(
                str(
                    ICONS_DIR
                    / icon_name
                )
            ),
            button_text,
        )
        button.setIconSize(
            QSize(17, 17)
        )

        def choose(
            _checked=False,
            kind=kind,
        ):
            path, _ = QFileDialog.getOpenFileName(
                aboba,
                "Выберите CSV справочник",
                "",
                "CSV (*.csv)",
            )

            if path:
                selected = Path(path).resolve()
                aboba.reference_paths[kind] = selected

                aboba.reference_fields[kind].setText(
                    selected.name
                )
                aboba.reference_fields[kind].setToolTip(
                    str(selected)
                )

        button.clicked.connect(choose)

        aboba.reference_fields[kind] = editor
        aboba.reference_buttons[kind] = button
        aboba.reference_controls.extend(
            (editor, button)
        )

        fields.addWidget(
            editor,
            row,
            0,
        )
        fields.addWidget(
            button,
            row,
            1,
        )

    fields.setColumnStretch(0, 1)

    left_layout.addLayout(fields)

    # ----------------------------------------------------------
    # Общая кнопка + статус на одной строке
    # ----------------------------------------------------------

    aboba.btn_load = QPushButton(
        QIcon(
            str(
                ICONS_DIR
                / "load_file.png"
            )
        ),
        " Загрузить справочники",
    )
    aboba.btn_load.setIconSize(
        QSize(17, 17)
    )
    aboba.btn_load.clicked.connect(
        lambda: load_csv_file(aboba)
    )

    # Статус загрузки файлов
    aboba.status_files_layout = QHBoxLayout()
    aboba.status_files_layout.setContentsMargins(
        0,
        0,
        0,
        0,
    )
    aboba.status_files_layout.setSpacing(CONTROL_SPACING)

    aboba.status_files_container = QWidget()
    aboba.status_files_container.setLayout(
        aboba.status_files_layout
    )

    # Кнопка и статус — одна строка
    load_status_row = QHBoxLayout()
    load_status_row.setContentsMargins(
        0,
        0,
        0,
        0,
    )
    load_status_row.setSpacing(BLOCK_SPACING)

    load_status_row.addWidget(
        aboba.btn_load,
        0,
        Qt.AlignmentFlag.AlignVCenter,
    )

    load_status_row.addWidget(
        aboba.status_files_container,
        1,
        Qt.AlignmentFlag.AlignVCenter,
    )

    left_layout.addLayout(
        load_status_row
    )

    update_file_status(aboba)

    return section


def update_file_status(aboba):
    input_dir = os.path.join(os.getcwd(), "input_data")

    files = {
        "Номенклатура": "nomenclature.csv",
        "Категории сайта  ": "site_categories.csv",
        "Координаты городов  ": "city_coordinates.csv"
    }

    # Очистка старых виджетов
    while aboba.status_files_layout.count():
        item = aboba.status_files_layout.takeAt(0)
        w = item.widget()
        if w:
            w.deleteLater()

    # Префикс
    aboba.prefix = QLabel("Статус загрузки:")
    aboba.status_files_layout.addWidget(aboba.prefix, 0, Qt.AlignmentFlag.AlignLeft)

    # Основная часть
    right_widget = QWidget()
    right_layout = QHBoxLayout()
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(CONTROL_SPACING)
    right_widget.setLayout(right_layout)

    ok_path = str(ICONS_DIR / "success.png")
    fail_path = str(ICONS_DIR / "failure.png")

    items = list(files.items())

    for (title, filename), kind in zip(items, REFERENCE_TYPES):
        exists = os.path.exists(os.path.join(input_dir, filename))
        exists = getattr(aboba, "reference_status_overrides", {}).get(kind, exists)

        block = QWidget()
        block.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        block_l = QHBoxLayout()
        block_l.setContentsMargins(0, 0, 0, 0)
        block_l.setSpacing(COMPACT_SPACING)
        block.setLayout(block_l)

        text_lbl = QLabel(title.strip())

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

    right_layout.addStretch(1)


    # добавляем правую часть с растягивающим коэффициентом
    aboba.status_files_layout.addWidget(right_widget, 1)


def load_csv_file(aboba):
    aboba.mb_controller.start_references()


def apply_reference_result(aboba, result):
    """Invalidate result caches after the background importer publishes a reference."""
    if result["kind"] == "Номенклатура из 1С":
        for name in ("_name_by_code", "_collection_by_code", "_stock_by_code", "_photo_by_code"):
            setattr(aboba, name, None)
    update_file_status(aboba)
