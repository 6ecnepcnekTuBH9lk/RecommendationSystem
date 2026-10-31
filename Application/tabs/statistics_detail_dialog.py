"""Compact explicit settings; no Orders/Customers reading or backend execution."""
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QDate, Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDateEdit, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from Application.paths import PROJECT_ROOT
from Application.statistics_detail_contract import DetailPeriod, OrderMode
from Application.statistics_detail_job import ERRORS, default_temporary_root, validate_paths, validate_period
from Application.statistics_detail_storage import DetailExportError

PERIOD_NOTE = ('Выбранный период определяет экспортируемые заказы и позиции. Для расчёта истории покупателей, '
               'когорт и RFM используется необходимая предыдущая история до даты отсечения.')
PII_NOTE = ('Архив содержит персональные данные покупателей: реальные идентификаторы Mindbox, доступные имена, '
            'телефоны, email и историю заказов. Храните файл в защищённом расположении и передавайте только '
            'уполномоченным получателям. Наличие контакта не подтверждает согласие на рассылку.')


def _note(text):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


def _date_edit(value):
    result = QDateEdit(QDate(value.year, value.month, value.day))
    result.setCalendarPopup(True)
    result.setDisplayFormat('dd.MM.yyyy')
    return result


class DetailExportDialog(QDialog):
    def __init__(self, parent, source, selection, raw_root):
        super().__init__(parent)
        self.setWindowTitle('Экспорт данных покупателей')
        self.setMinimumWidth(590)
        self.raw_root = raw_root
        self.options = None
        self._confirmed_path = None
        start = source.first_date
        end = source.last_date
        has_period = selection.start_date is not None
        if has_period:
            from datetime import date
            start, end = map(date.fromisoformat, (selection.start_date, selection.end_date))
        self.all_history = QCheckBox('Вся доступная история')
        self.all_history.setChecked(not has_period)
        self.start_date = _date_edit(start)
        self.end_date = _date_edit(end)
        self.as_of = _date_edit(end)
        self.mode = QComboBox()
        self.mode.addItem('Все заказы выбранного периода', OrderMode.ALL_SELECTED)
        self.mode.addItem('Заказы с покупкой по STRICT или EXTENDED', OrderMode.PURCHASE_EITHER)
        self.destination = QLineEdit()
        self.temporary_root = QLineEdit(str(default_temporary_root()))
        self.temporary_root.setCursorPosition(0)
        self.error = _note('')
        self.error.setObjectName('detailExportError')
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow(self.all_history)
        form.addRow('Заказы с', self.start_date)
        form.addRow('Заказы по', self.end_date)
        form.addRow('Дата оценки покупателей (UTC)', self.as_of)
        form.addRow('Режим заказов', self.mode)
        for title, edit, callback in (
            ('Выходной ZIP', self.destination, self.choose_destination),
            ('Приватный временный каталог D:', self.temporary_root, self.choose_temporary),
        ):
            row = QHBoxLayout()
            row.addWidget(edit)
            button = QPushButton('Выбрать…')
            button.clicked.connect(callback)
            row.addWidget(button)
            form.addRow(title, row)
        layout.addLayout(form)
        layout.addWidget(_note('Дата оценки предложена по окончанию выбранного периода; без периода — по последней '
                              'доступной дате покрытия Orders, не позднее текущей UTC-даты. Дату можно изменить.'))
        layout.addWidget(_note('Все заказы включает возвраты и отмены. Режим покупок оставляет заказы, '
                              'признанные хотя бы одной из двух политик. Фильтр товаров статистики не применяется.'))
        layout.addWidget(_note(PERIOD_NOTE))
        layout.addWidget(_note(PII_NOTE))
        layout.addWidget(self.error)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.begin = QPushButton('Начать экспорт')
        self.dismiss = QPushButton('Отмена')
        buttons.addWidget(self.begin)
        buttons.addWidget(self.dismiss)
        layout.addLayout(buttons)
        self.begin.clicked.connect(self.accept)
        self.dismiss.clicked.connect(self.reject)
        self.all_history.toggled.connect(self._toggle_period)
        self._toggle_period(self.all_history.isChecked())

    def _toggle_period(self, checked):
        self.start_date.setEnabled(not checked)
        self.end_date.setEnabled(not checked)

    def choose_destination(self):
        directory = PROJECT_ROOT.parent
        filename = 'Данные_покупателей_'+datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')+'.zip'
        value, _ = QFileDialog.getSaveFileName(self, 'Экспорт данных покупателей', str(directory/filename), 'ZIP (*.zip)')
        if value:
            target = Path(value if value.lower().endswith('.zip') else value+'.zip')
            # Native dialog confirmed the exact existing path, when no suffix was added.
            self._confirmed_path = target if target == Path(value) else None
            self.destination.setText(str(target))

    def choose_temporary(self):
        value = QFileDialog.getExistingDirectory(self, 'Приватный временный каталог D:', self.temporary_root.text())
        if value:
            self.temporary_root.setText(value)

    def accept(self):
        try:
            period = (DetailPeriod() if self.all_history.isChecked() else
                      DetailPeriod(self.start_date.date().toPyDate(), self.end_date.date().toPyDate()))
            as_of, mode = self.as_of.date().toPyDate(), self.mode.currentData()
            validate_period(as_of, period, mode)
            if not self.destination.text().strip() or not self.temporary_root.text().strip():
                raise DetailExportError('INVALID_EXPORT_PATH')
            root, target, temporary = validate_paths(self.raw_root, self.destination.text().strip(),
                                                     self.temporary_root.text().strip())
            if target.exists() and target != self._confirmed_path:
                answer = QMessageBox.question(self, 'Заменить ZIP?', 'Заменить существующий ZIP только после успешного '
                    'экспорта?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    return
            self.options = dict(raw_root=root, destination=target, temporary_root=temporary, as_of=as_of,
                start=period.start_date, end=period.end_date, order_mode=mode.value)
        except Exception as error:
            code = getattr(error, 'code', 'INVALID_OPTIONS')
            self.error.setText(ERRORS.get(code, ERRORS['INVALID_OPTIONS'])+' Период должен завершаться не позже даты оценки.')
            return
        super().accept()
