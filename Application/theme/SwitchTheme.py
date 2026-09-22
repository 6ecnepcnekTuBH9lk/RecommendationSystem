from __future__ import annotations

from Application.paths import ICONS_DIR

import os

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, pyqtProperty, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap, QPalette
from PyQt6.QtWidgets import QLabel, QWidget


# --- Visual constants (do not change external behavior) ---
SWITCH_W, SWITCH_H = 54, 32
KNOB_SIZE = 26
ICON_SIZE = 22
PADDING = 3

SUN_ICON_PATH = str(ICONS_DIR / "sun.png")
MOON_ICON_PATH = str(ICONS_DIR / "moon.png")

class ThemeSwitch(QWidget):

    themeChanged = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setFixedSize(SWITCH_W, SWITCH_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Knob position state
        self._x = SWITCH_W - KNOB_SIZE - PADDING
        self._y = PADDING
        self._checked = True

        # Knob container for PNG (do not stretch icon)
        self.knob = QLabel(self)
        self.knob.setFixedSize(KNOB_SIZE, KNOB_SIZE)
        self.knob.move(self._x, self._y)
        self.knob.setScaledContents(False)
        self.knob.setStyleSheet("background: transparent; border: none;")

        # Icons
        self.sun_icon = QPixmap(SUN_ICON_PATH)
        self.moon_icon = QPixmap(MOON_ICON_PATH)
        self.knob.setPixmap(self._center_icon(self.moon_icon))

        # Animation
        self.anim = QPropertyAnimation(self, b"knob_pos", self)
        self.anim.setDuration(250)
        self.anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

    # === PNG centering inside knob (keeps original look) ===
    def _center_icon(self, pix: QPixmap) -> QPixmap:
        result = QPixmap(KNOB_SIZE, KNOB_SIZE)
        result.fill(Qt.GlobalColor.transparent)

        p = QPainter(result)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.drawPixmap(
            (KNOB_SIZE - ICON_SIZE) // 2,
            (KNOB_SIZE - ICON_SIZE) // 2,
            pix.scaled(
                ICON_SIZE,
                ICON_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ),
        )
        p.end()
        return result

    # === Property for animation ===
    def get_knob_pos(self) -> int:
        return self._x

    def set_knob_pos(self, x: int) -> None:
        self._x = int(x)
        knob_y = (self.height() - self.knob.height()) // 2
        self.knob.move(self._x, knob_y)

    knob_pos = pyqtProperty(int, fget=get_knob_pos, fset=set_knob_pos)

    # === Background painting ===
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        palette = self.palette()
        track = QColor(os.environ.get("QTMATERIAL_SECONDARYCOLOR", palette.color(QPalette.ColorRole.Mid).name()))
        knob = QColor(os.environ.get("QTMATERIAL_SECONDARYDARKCOLOR", palette.color(QPalette.ColorRole.Base).name()))
        p.setBrush(track)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 16, 16)

        # Knob circle (under PNG)
        knob_rect = self.knob.geometry()
        p.setBrush(knob)
        p.drawEllipse(knob_rect)

    # === Click handling ===
    def mousePressEvent(self, event) -> None:
        # Keep original behavior: click toggles state
        self.toggle()
        event.accept()

    def toggle(self) -> None:
        self._checked = not self._checked

        if self._checked:
            icon = self._center_icon(self.moon_icon)
            target_x = self.width() - self.knob.width() - PADDING
        else:
            icon = self._center_icon(self.sun_icon)
            target_x = PADDING

        self.knob.setPixmap(icon)

        self.anim.stop()
        self.anim.setStartValue(self.knob_pos)
        self.anim.setEndValue(target_x)
        self.anim.start()

        self.update()
        self.themeChanged.emit(self._checked)
