"""Central Material theme for all application widgets."""

from pathlib import Path
import os

# Import Qt first: qt-material detects the active Qt binding at import time.
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor, QFont, QFontDatabase
from qt_material import apply_stylesheet


CUSTOM_QSS = Path(__file__).with_name("custom.qss")


def apply_app_theme(app: QApplication, is_dark: bool):
    family = "Segoe UI Variable" if "Segoe UI Variable" in QFontDatabase.families() else "Segoe UI"
    app.setFont(QFont(family, 10))
    apply_stylesheet(
        app,
        theme="dark_purple.xml" if is_dark else "light_purple.xml",
        invert_secondary=not is_dark,
        extra={"density_scale": "-1", "font_family": family},
    )
    # Material publishes the active (including inverted light) colors on apply.
    colors = dict(os.environ)
    colors["APP_FONT"] = family
    for name, source in (("APP_TEXT_RGB", "SECONDARYTEXTCOLOR"), ("APP_PRIMARY_RGB", "PRIMARYCOLOR")):
        rgb = QColor(colors[f"QTMATERIAL_{source}"]).getRgb()[:3]
        colors[name] = ", ".join(str(channel) for channel in rgb)
    app.setStyleSheet(app.styleSheet() + CUSTOM_QSS.read_text(encoding="utf-8").format(**colors))
