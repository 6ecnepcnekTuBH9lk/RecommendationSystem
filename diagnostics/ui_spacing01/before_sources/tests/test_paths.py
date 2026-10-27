import ast
from pathlib import Path

from Application.paths import ICONS_DIR, INPUT_DATA_DIR, PROJECT_ROOT, USER_SETTINGS_DIR
from Application.settings.settings_and_filter import dataset_paths
from Application.tabs.data_loading_tab import RAW_ROOT


def test_ascii_runtime_paths_and_icons(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert INPUT_DATA_DIR == PROJECT_ROOT / "input_data"
    assert USER_SETTINGS_DIR == PROJECT_ROOT / "user_settings"
    assert RAW_ROOT == INPUT_DATA_DIR / "MindboxRaw"
    assert {key: Path(value) for key, value in dataset_paths().items()} == {
        "orders": tmp_path / "input_data/orders.csv",
        "views": tmp_path / "input_data/views.csv",
        "favs": tmp_path / "input_data/favorites.csv",
    }
    icons = list(ICONS_DIR.glob("*.png"))
    required = {name + ".png" for name in (
        "actions analyze app_icon cart clock continue customers default_settings down excel failure favorites "
        "filter get_data load_file moon order orders refresh save search start_training success sun "
        "triangle_down triangle_up views warning"
    ).split()}
    assert required <= {path.name for path in icons}
    assert all(path.name.isascii() for path in icons)
    assert (ICONS_DIR / "app_icon.png").is_file()


def test_no_legacy_runtime_path_literals():
    for path in [PROJECT_ROOT / "main.py", *PROJECT_ROOT.glob("Application/**/*.py"),
                 *PROJECT_ROOT.glob("scripts/*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for old in ("Картинки", "ВходныеДанные", "Настройки", "ФильтрованныеДанные", "Модель"):
                    assert node.value != old, (path, node.lineno)
                    assert not node.value.startswith((old + "/", old + "\\")), (path, node.lineno)
