import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from Application.model import BPRMF

NORMAL_SHUTDOWN_MARKER = "synthetic-normal-python-shutdown"


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
        data_dir / "Заказы.csv",
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [{"MindboxID": "synthetic-user", "ТипТовара": "Номенклатура"}]
    ).to_csv(
        data_dir / "Просмотры.csv",
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(columns=["MindboxID", "КодНоменклатуры"]).to_csv(
        data_dir / "Избранное.csv",
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
    assert "Просмотры.csv" in result.stdout
    assert "КодНоменклатуры" in result.stdout
    assert "Устройство для обучения" not in result.stdout
    assert not (tmp_path / "Модель" / "current.json").exists()


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
