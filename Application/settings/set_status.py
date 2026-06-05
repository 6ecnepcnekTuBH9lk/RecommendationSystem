from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QMessageBox)


# ///////////////////////////////////////////РАБОТА СО СТАТУСОМ ЗАДАЧИ//////////////////////////////////////////////
def set_status_processing(aboba, text):
    aboba.status_label.setText(text)
    aboba.status_icon.setPixmap(QPixmap("Картинки/Часы.png").scaled(
        17, 17, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    ))


def set_status_ok(aboba, text):
    aboba.status_label.setText(text)
    aboba.status_icon.setPixmap(QPixmap("Картинки/Успех.png").scaled(
        17, 17, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    ))


def set_status_error(aboba, text):
    aboba.status_label.setText(text)
    aboba.status_icon.setPixmap(QPixmap("Картинки/Неудача.png").scaled(
        17, 17, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    ))


def set_ready_status(aboba):
    set_status_ok(aboba, "Готов к работе")


def schedule_status_reset(aboba, seconds: int):
    timer = getattr(aboba, "_status_reset_timer", None)
    if timer is None:
        return

    timer.stop()
    timer.start(seconds * 1000)


# -------------------------------------------УВЕДОМЛЕНИЕ ДЛЯ ПОЛЬЗОВАТЕЛЯ-----------------------------------------------
def show_custom_message(aboba, title: str, text: str, image_path: str = None):
    msg = QMessageBox(aboba)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)

    if image_path:
        pix = QPixmap(image_path)
        msg.setIconPixmap(pix.scaled(45, 45, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))

    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.setStyleSheet("""QPushButton { padding: 5px 10px; }""")
    msg.exec()
