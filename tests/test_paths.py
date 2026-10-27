import ast
from pathlib import Path

import pytest

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


LEGACY_DIRECTORIES = frozenset(("Картинки", "ВходныеДанные", "Настройки", "ФильтрованныеДанные", "Модель"))


def legacy_path_literals(tree):
    """Distinguish path components/operands from human-facing captions."""
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value.replace("\\", "/")
        if "/" in value and LEGACY_DIRECTORIES.intersection(value.split("/")):
            yield node
            continue
        if value not in LEGACY_DIRECTORIES:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div):
            yield node
        elif isinstance(parent, ast.Call) and ast.unparse(parent.func).split(".")[-1].lower() in {
            "path", "purepath", "purewindowspath", "pureposixpath", "qdir", "join", "joinpath",
            "open", "mkdir", "makedirs", "isdir", "isfile", "exists", "chdir", "listdir", "glob", "rglob",
        }:
            yield node
        elif isinstance(parent, (ast.Assign, ast.AnnAssign)):
            targets = parent.targets if isinstance(parent, ast.Assign) else (parent.target,)
            if any(part in ast.unparse(target).lower() for target in targets
                   for part in ("path", "dir", "root", "folder", "filename")):
                yield node
        elif isinstance(parent, ast.Dict):
            for key, item in zip(parent.keys, parent.values):
                if item is node and isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if any(part in key.value.lower() for part in ("path", "dir", "root", "folder", "filename")):
                        yield node


@pytest.mark.parametrize("source,forbidden", [
    ('QLabel("Модель")', False),
    ('FIELDS = ("Вероятность", "Модель")', False),
    ('caption = "Модель"', False),
    ('Path("model")', False),
    ('Path("Модель")', True),
    ('PROJECT_ROOT / "Модель"', True),
    ('os.path.join("Модель", "weights.pt")', True),
    ('open("Модель")', True),
    ('MODEL_DIR = "Модель"', True),
    ('config = {"output_dir": "Модель"}', True),
    ('path = "C:/project/Модель/weights.pt"', True),
    ('path = r"ВходныеДанные\\orders.csv"', True),
])
def test_legacy_path_detection_distinguishes_ui_text(source, forbidden):
    assert bool(tuple(legacy_path_literals(ast.parse(source)))) is forbidden


def test_no_legacy_runtime_path_literals():
    for path in [PROJECT_ROOT / "main.py", *PROJECT_ROOT.glob("Application/**/*.py"),
                 *PROJECT_ROOT.glob("scripts/*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not tuple(legacy_path_literals(tree)), path
