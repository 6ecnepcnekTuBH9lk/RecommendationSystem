"""Central Material theme for all application widgets."""

from pathlib import Path
import os

# Import Qt first: qt-material detects the active Qt binding at import time.
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor, QFont, QFontDatabase
from qt_material import apply_stylesheet


THEME_DIR = Path(__file__).resolve().parent
CUSTOM_QSS = THEME_DIR / "custom.qss"

LIGHT_THEME = THEME_DIR / "light_custom.xml"
DARK_THEME = THEME_DIR / "dark_custom.xml"


def apply_app_theme(app: QApplication, is_dark: bool):
    family = (
        "Segoe UI Variable"
        if "Segoe UI Variable" in QFontDatabase.families()
        else "Segoe UI"
    )

    app.setFont(
        QFont(family, 10)
    )

    theme_path = (
        DARK_THEME
        if is_dark
        else LIGHT_THEME
    )

    apply_stylesheet(
        app,
        theme=str(theme_path),
        invert_secondary=not is_dark,
        extra={
            "density_scale": "-1",
            "font_family": family,
        },
    )

    # qt-material publishes the colors of the currently applied theme
    # through QTMATERIAL_* environment variables.
    colors = dict(os.environ)

    colors["APP_FONT"] = family

    for name, source in (
        ("APP_TEXT_RGB", "SECONDARYTEXTCOLOR"),
        ("APP_PRIMARY_RGB", "PRIMARYCOLOR"),
    ):
        rgb = QColor(
            colors[f"QTMATERIAL_{source}"]
        ).getRgb()[:3]

        colors[name] = ", ".join(
            str(channel)
            for channel in rgb
        )

    custom_style = CUSTOM_QSS.read_text(
        encoding="utf-8"
    ).format(**colors)

    app.setStyleSheet(
        app.styleSheet()
        + custom_style
    )