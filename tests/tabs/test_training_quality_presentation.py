"""Synthetic production quality presentation; no preparation or training."""
from copy import deepcopy

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from Application.tabs import train_model_tab as tab
from Application.theme.apply_theme import apply_app_theme


def warning_report():
    return {'level': 'WARN', 'training_allowed': True, 'metrics': {
        'mapped_actions': 3347664, 'malformed_mapped_actions': 2732459,
        'unknown_candidate_max_rate': .0001, 'unknown_candidate_max_count': 100}, 'issues': [
        {'code': 'MAPPED_ACTION_WITHOUT_PRODUCT', 'count': 2732459, 'level': 'WARN',
         'message': 'Mapped actions without products were excluded', 'rate': .816,
         'breakdown': {'ProsmotrProduktaVApiMethod': 2702582}},
        {'code': 'UNRESOLVED_PRODUCT', 'count': 20, 'level': 'WARN', 'rate': 20 / 1206884,
         'message': 'Catalog candidates not found', 'breakdown': {}}]}


def test_warning_formatter_has_russian_summary_without_technical_fields_and_preserves_report():
    report = warning_report()
    original = deepcopy(report)
    status, lines, text = tab._quality_presentation(report)
    assert status == 'Проверка данных: предупреждение'
    assert '2 732 459 действий без товара исключены из обучения.' in lines[0]
    assert 'Для 20 событий (0,001657%) не удалось сопоставить товар со справочником номенклатуры.' in lines[1]
    assert 'Эти взаимодействия исключены из обучения.' in lines[1]
    assert 'Продолжить обучение?' in text and 'Обучение можно продолжить.' in text
    for technical in ('MAPPED_ACTION_WITHOUT_PRODUCT', 'UNKNOWN_CANDIDATE', 'UNRESOLVED_PRODUCT',
                      'mapped_actions', 'malformed_mapped_actions', 'ProsmotrProduktaVApiMethod',
                      'Показатели проверки:', '{', '}', '0.01%', '100 events', 'Catalog candidates'):
        assert technical not in '\n'.join([status, *lines, text])
    assert report == original


@pytest.mark.parametrize('level,word', [('PASS', 'успешно'), ('BLOCK', 'заблокировано')])
def test_quality_status_wording(level, word):
    assert tab._quality_presentation({'level': level, 'issues': []})[0] == 'Проверка данных: ' + word


@pytest.mark.parametrize('code', ['UNRESOLVED_PRODUCT', 'INVALID_PRODUCT_ID', 'UNSUPPORTED_PRODUCT_NAMESPACE',
                                 'INVALID_PREPARED_DATA', 'CONFLICTING_ORDER_SNAPSHOTS', 'FUTURE_ISSUE'])
def test_block_issue_messages_never_show_technical_english_or_recovery_permission(code):
    text = tab._quality_issue_text({'code': code, 'count': 1234, 'level': 'BLOCK', 'message': 'Technical backend message'})
    assert code not in text and 'Technical' not in text
    assert 'Обучение можно продолжить' not in text
    if code != 'INVALID_PREPARED_DATA':
        assert '1 234' in text


@pytest.mark.parametrize('dark', [False, True])
@pytest.mark.parametrize('action', ['yes', 'no', 'escape', 'close', 'reject'])
def test_warning_dialog_icons_russian_buttons_default_escape_and_rejection(monkeypatch, dark, action):
    app = QApplication.instance() or QApplication([])
    apply_app_theme(app, dark)
    parent = QWidget()
    text = tab._quality_presentation(warning_report())[2]
    original_exec = QMessageBox.exec
    errors = []

    def execute(box):
        assert box.windowTitle() == 'Предупреждение перед обучением'
        assert box.text() == text
        assert not box.iconPixmap().isNull() and not box.windowIcon().isNull()
        assert box.windowIcon().pixmap(24, 24).toImage() == QIcon(str(tab.ICONS_DIR / 'app_icon.png')).pixmap(24, 24).toImage()
        expected = QPixmap(str(tab.ICONS_DIR / 'question.png')).scaled(
            48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        assert box.iconPixmap().toImage() == expected.toImage()
        yes, no = (next(button for button in box.buttons() if button.text() == name) for name in ('Да', 'Нет'))
        assert len(box.buttons()) == 2 and box.standardButtons() == QMessageBox.StandardButton.NoButton
        assert box.buttonRole(yes) == QMessageBox.ButtonRole.AcceptRole
        assert box.buttonRole(no) == QMessageBox.ButtonRole.RejectRole
        assert box.defaultButton() is no and box.escapeButton() is no
        assert all(not button.styleSheet() for button in box.buttons())
        assert box.styleSheet() == '' and box.minimumSize() != box.maximumSize()
        def act():
            try:
                if action in {'yes', 'no'}:
                    (yes if action == 'yes' else no).click()
                elif action == 'escape':
                    QTest.keyClick(box, Qt.Key.Key_Escape)
                elif action == 'close':
                    box.close()
                else:
                    box.reject()
            except Exception as error:
                errors.append(error)
                box.reject()
        QTimer.singleShot(0, act)
        return original_exec(box)

    monkeypatch.setattr(QMessageBox, 'exec', execute)
    try:
        assert tab._confirm_quality_warning(parent, text) is (action == 'yes')
        assert not errors
    finally:
        parent.close()
        parent.deleteLater()
        app.processEvents()
        apply_app_theme(app, False)
