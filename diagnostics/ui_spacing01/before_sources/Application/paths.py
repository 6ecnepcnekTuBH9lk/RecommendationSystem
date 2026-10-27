"""Filesystem locations shared by the application."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA_DIR = PROJECT_ROOT / "input_data"
USER_SETTINGS_DIR = PROJECT_ROOT / "user_settings"
ICONS_DIR = PROJECT_ROOT / "assets" / "icons"
