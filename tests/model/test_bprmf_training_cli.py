import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from Application.model import BPRMF

NORMAL_SHUTDOWN_MARKER = "synthetic-normal-python-shutdown"


@pytest.mark.parametrize("source, expected", [
    ("ВходныеДанные", "input_data"),
    ("./ВходныеДанные/processed", "./input_data/processed"),
    (r".\ВходныеДанные\processed", r".\input_data\processed"),
    (r"C:\project\ВходныеДанные", r"C:\project\input_data"),
    ("/project/ВходныеДанные/", "/project/input_data/"),
    ("input_data", "input_data"),
    ("C:\\project\\input_data\\", "C:\\project\\input_data\\"),
    ("custom_data", "custom_data"),
    ("archive_ВходныеДанные", "archive_ВходныеДанные"),
    ("ВходныеДанные_backup", "ВходныеДанные_backup"),
])
def test_training_config_normalizes_only_legacy_data_dir(tmp_path, source, expected):
    path = tmp_path / "train_config.json"
    values = {"data_dir": source, "epochs": 7, "lr": 0.002,
              "item_feature_cols": ["Марка"], "use_item_features": False}
    path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    original = path.read_bytes()
    config = BPRMF._load_train_config_from_json(str(path))
    assert vars(config) == {**vars(BPRMF.TrainConfig()), **values, "data_dir": expected}
    assert path.read_bytes() == original


def test_training_config_legacy_absolute_path_resolves_existing_ascii_directory(tmp_path):
    canonical = tmp_path / "input_data"
    canonical.mkdir()
    path = tmp_path / "train_config.json"
    path.write_text(json.dumps({"data_dir": str(tmp_path / "ВходныеДанные")}), encoding="utf-8")
    config = BPRMF._load_train_config_from_json(str(path))
    assert Path(config.data_dir) == canonical
    assert Path(config.data_dir).is_dir()


def _subprocess_env(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(
        "import atexit\n"
        f"atexit.register(lambda: print({NORMAL_SHUTDOWN_MARKER!r}))\n",
        encoding="utf-8",
    )
    python_path = os.pathsep.join(
        part
        for part in (str(tmp_path), os.environ.get("PYTHONPATH", ""))
        if part
    )
    return {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": python_path,
    }


def test_cli_returns_controlled_failure_when_required_files_are_missing(tmp_path):
    config_path = tmp_path / "train_config.json"
    config_path.write_text(
        json.dumps(
            {"data_dir": str(tmp_path / "missing_training_data")},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(BPRMF.__file__).resolve()),
            "--train",
            "--config",
            str(config_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        env=_subprocess_env(tmp_path),
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "Отсутствуют следующие необходимые файлы" in result.stdout
    assert NORMAL_SHUTDOWN_MARKER not in result.stdout


def test_training_cli_returns_controlled_failure_for_invalid_schema(tmp_path):
    data_dir = tmp_path / "training_data"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "MindboxID": "synthetic-user",
                "КодНоменклатуры": "order-item",
                "Количество": "1",
            }
        ]
    ).to_csv(
        data_dir / "orders.csv",
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [{"MindboxID": "synthetic-user", "ТипТовара": "Номенклатура"}]
    ).to_csv(
        data_dir / "views.csv",
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(columns=["MindboxID", "КодНоменклатуры"]).to_csv(
        data_dir / "favorites.csv",
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )
    config_path = tmp_path / "train_config.json"
    config_path.write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "epochs": 1,
                "batch_size": 1,
                "embedding_dim": 4,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(BPRMF.__file__).resolve()),
            "--train",
            "--config",
            str(config_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        env=_subprocess_env(tmp_path),
    )

    assert result.returncode == BPRMF.TRAIN_EXIT_NO_DATA, result.stdout + result.stderr
    assert "views.csv" in result.stdout
    assert "КодНоменклатуры" in result.stdout
    assert "Устройство для обучения" not in result.stdout
    assert not (tmp_path / "model" / "current.json").exists()


def test_cli_without_train_uses_normal_shutdown_and_flushes_stdout(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(BPRMF.__file__).resolve())],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        env=_subprocess_env(tmp_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert NORMAL_SHUTDOWN_MARKER in result.stdout
