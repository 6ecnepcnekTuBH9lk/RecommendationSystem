"""Central Material theme; build once, apply once per runtime switch."""

from pathlib import Path
import os
from weakref import WeakSet

# Import Qt first: qt-material detects the active Qt binding at import time.
from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from qt_material import build_stylesheet


THEME_DIR = Path(__file__).resolve().parent
CUSTOM_QSS = THEME_DIR / "custom.qss"
LIGHT_THEME = THEME_DIR / "light_custom.xml"
DARK_THEME = THEME_DIR / "dark_custom.xml"
_THEME_CACHE = {}
_FONT_FAMILY = None
_INITIALIZED_APPS = WeakSet()


def prepare_app_theme(app: QApplication):
    """Warm both variants before constructing widgets; Qt work stays on GUI thread."""
    global _FONT_FAMILY
    if app not in _INITIALIZED_APPS:
        # Qt drops registered application fonts when QApplication is destroyed.
        # A subsequent application (e.g. a test fixture) needs fresh resources.
        _THEME_CACHE.clear()
        _FONT_FAMILY = None
        app.setStyle("Fusion")
        _INITIALIZED_APPS.add(app)
    if _FONT_FAMILY is None:
        _FONT_FAMILY = "Segoe UI Variable" if "Segoe UI Variable" in QFontDatabase.families() else "Segoe UI"
    if app.font().family() != _FONT_FAMILY or app.font().pointSizeF() != 10:
        app.setFont(QFont(_FONT_FAMILY, 10))
    if _THEME_CACHE:
        return

    custom = CUSTOM_QSS.read_text(encoding="utf-8")
    variants = {}
    for dark, path in ((False, LIGHT_THEME), (True, DARK_THEME)):
        # Separate generated SVG directories: building dark must not overwrite light.
        resource_name = "recommendation_dark" if dark else "recommendation_light"
        material = build_stylesheet(
            theme=str(path), invert_secondary=not dark,
            extra={"density_scale": "-1", "font_family": _FONT_FAMILY},
            parent=resource_name,
        )
        environment = {key: value for key, value in os.environ.items() if key.startswith("QTMATERIAL_")}
        colors = {**environment, "APP_FONT": _FONT_FAMILY}
        for name, source in (("APP_TEXT_RGB", "SECONDARYTEXTCOLOR"), ("APP_PRIMARY_RGB", "PRIMARYCOLOR")):
            colors[name] = ", ".join(str(channel) for channel in QColor(colors[f"QTMATERIAL_{source}"]).getRgb()[:3])
        # build_stylesheet also sets the Text palette role and registers its icon path.
        variants[dark] = (material + custom.format(**colors), environment,
                          [p for p in QDir.searchPaths("icon") if Path(p).name == resource_name],
                          app.palette().color(QPalette.ColorRole.Text))
    _THEME_CACHE.update(variants)


def apply_app_theme(app: QApplication, is_dark: bool):
    prepare_app_theme(app)
    stylesheet, environment, icon_paths, text_color = _THEME_CACHE[bool(is_dark)]
    os.environ.update(environment)
    QDir.setSearchPaths("icon", icon_paths)
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Text, text_color)
    app.setPalette(palette)
    app.setStyleSheet(stylesheet)
