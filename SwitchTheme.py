from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt, QPropertyAnimation, pyqtProperty, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap, QPainterPath


class ThemeSwitch(QWidget):

    themeChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(54, 32)

        # положение шарика
        self._x = 3
        self._y = 3
        self._checked = False

        # шарик-контейнер для PNG
        self.knob = QLabel(self)
        self.knob.setFixedSize(26, 26)
        self.knob.move(self._x, self._y)
        self.knob.setScaledContents(False)  # важно: не растягивать PNG

        # текущая иконка
        self.sun_icon = QPixmap("Картинки/Солнце.png")
        self.moon_icon = QPixmap("Картинки/Луна.png")
        self.knob.setPixmap(self._center_icon(self.sun_icon))

        # анимация
        self.anim = QPropertyAnimation(self, b"knob_pos", self)
        self.anim.setDuration(250)
        self.anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # === делаем PNG идеально центрированным ===
    def _center_icon(self, pix: QPixmap) -> QPixmap:

        size = 22
        result = QPixmap(26, 26)
        result.fill(Qt.GlobalColor.transparent)

        p = QPainter(result)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.drawPixmap(
            (26 - size) // 2,
            (26 - size) // 2,
            pix.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        p.end()
        return result

    # === свойство для анимации ===
    def get_knob_pos(self):
        return self._x

    def set_knob_pos(self, x):
        self._x = x
        self.knob_y = (self.height() - self.knob.height()) // 2
        self.knob.move(self._x, self.knob_y)

    knob_pos = pyqtProperty(int, fget=get_knob_pos, fset=set_knob_pos)

    # === рисуем фон свитча ===
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        color = QColor("#505050") if self._checked else QColor("#EBEBEB")

        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 16, 16)

        # рисуем шарик (круг под PNG)
        knob_rect = self.knob.geometry()
        p.setBrush(QColor("#FAFAFA") if not self._checked else QColor("#3C3C3C"))
        p.drawEllipse(knob_rect)

    # === клик ===
    def mousePressEvent(self, event):
        self.toggle()

    def toggle(self):
        self._checked = not self._checked

        if self._checked:
            icon = self._center_icon(self.moon_icon)
            target_x = self.width() - self.knob.width() - 3
        else:
            icon = self._center_icon(self.sun_icon)
            target_x = 3

        self.knob.setPixmap(icon)

        self.anim.stop()
        self.anim.setStartValue(self.knob_pos)
        self.anim.setEndValue(target_x)
        self.anim.start()

        self.update()

        # === отправляем сигнал наружу ===
        self.themeChanged.emit(self._checked)
