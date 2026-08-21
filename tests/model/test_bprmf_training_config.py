import json
import subprocess
import sys
from pathlib import Path

import pytest

from Application.model import BPRMF


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _fail_if_called(name):
    def fail(*args, **kwargs):
        pytest.fail(f"{name} must not be called")

    return fail


def test_main_rejects_missing_explicit_config_before_training(tmp_path, monkeypatch):
    training_calls = []
    monkeypatch.setattr(
        BPRMF,
        "_train_in_this_process",
        lambda cfg: training_calls.append(cfg) or True,
    )

    missing_config = tmp_path / "missing_train_config.json"

    with pytest.raises(FileNotFoundError):
        BPRMF.main(["--train", "--config", str(missing_config)])

    assert training_calls == []


def test_loader_applies_partial_config_defaults_and_ignores_extra_fields(tmp_path):
    config_path = tmp_path / "train_config.json"
    _write_json(
        config_path,
        {
            "data_dir": "synthetic-data",
            "epochs": 7,
            "lr": 1,
            "early_stop": False,
            "filter_summary": "synthetic GUI metadata",
        },
    )

    cfg = BPRMF._load_train_config_from_json(str(config_path))

    defaults = BPRMF.TrainConfig()
    assert cfg.data_dir == "synthetic-data"
    assert cfg.epochs == 7
    assert cfg.lr == 1.0
    assert type(cfg.lr) is float
    assert cfg.early_stop is False
    assert cfg.batch_size == defaults.batch_size
    assert not hasattr(cfg, "filter_summary")


def test_main_rejects_corrupt_explicit_config_before_training(tmp_path, monkeypatch):
    config_path = tmp_path / "train_config.json"
    config_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(
        BPRMF,
        "_train_in_this_process",
        _fail_if_called("training"),
    )

    with pytest.raises(json.JSONDecodeError):
        BPRMF.main(["--train", "--config", str(config_path)])


def test_main_propagates_config_read_error_before_training(tmp_path, monkeypatch):
    config_path = tmp_path / "train_config.json"

    def fail_open(*args, **kwargs):
        raise PermissionError("synthetic config read error")

    monkeypatch.setattr(BPRMF, "open", fail_open, raising=False)
    monkeypatch.setattr(
        BPRMF,
        "_train_in_this_process",
        _fail_if_called("training"),
    )

    with pytest.raises(PermissionError, match="synthetic config read error"):
        BPRMF.main(["--train", "--config", str(config_path)])


@pytest.mark.parametrize("invalid_root", [[], 42])
def test_main_rejects_non_object_json_root_before_training(
    tmp_path, invalid_root, monkeypatch
):
    config_path = tmp_path / "train_config.json"
    _write_json(config_path, invalid_root)
    monkeypatch.setattr(
        BPRMF,
        "_train_in_this_process",
        _fail_if_called("training"),
    )

    with pytest.raises(TypeError, match="JSON object"):
        BPRMF.main(["--train", "--config", str(config_path)])


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("data_dir", 42),
        ("early_stop", "false"),
        ("epochs", "128"),
        ("item_feature_cols", "ВидНоменклатуры"),
        ("item_feature_cols", ["ВидНоменклатуры", 42]),
    ],
)
def test_main_rejects_invalid_known_field_types_before_training(
    tmp_path, field, invalid_value, monkeypatch
):
    config_path = tmp_path / "train_config.json"
    _write_json(config_path, {field: invalid_value})
    monkeypatch.setattr(
        BPRMF,
        "_train_in_this_process",
        _fail_if_called("training"),
    )

    with pytest.raises(TypeError, match=field):
        BPRMF.main(["--train", "--config", str(config_path)])


def test_main_rejects_config_flag_without_path_before_training(monkeypatch):
    monkeypatch.setattr(
        BPRMF,
        "_train_in_this_process",
        _fail_if_called("training"),
    )

    with pytest.raises(ValueError, match="--config"):
        BPRMF.main(["--train", "--config"])


def test_main_without_config_uses_intentional_defaults(monkeypatch):
    received_configs = []
    monkeypatch.setattr(
        BPRMF,
        "_load_train_config_from_json",
        _fail_if_called("config loader"),
    )
    monkeypatch.setattr(
        BPRMF,
        "_train_in_this_process",
        lambda cfg: received_configs.append(cfg) or True,
    )

    assert BPRMF.main(["--train"]) == BPRMF.TRAIN_EXIT_SUCCESS
    assert received_configs == [BPRMF.TrainConfig()]


def test_valid_config_without_required_data_preserves_no_data_exit_code(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "train_config.json"
    data_dir = tmp_path / "synthetic_missing_data"
    _write_json(config_path, {"data_dir": str(data_dir)})
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)

    assert (
        BPRMF.main(["--train", "--config", str(config_path)])
        == BPRMF.TRAIN_EXIT_NO_DATA
    )


def test_cli_bad_explicit_config_exits_nonzero_without_starting_training(tmp_path):
    missing_config = tmp_path / "missing_train_config.json"

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(BPRMF.__file__).resolve()),
            "--train",
            "--config",
            str(missing_config),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    assert result.returncode != BPRMF.TRAIN_EXIT_SUCCESS
    assert "Отсутствуют следующие необходимые файлы" not in result.stdout
    assert "FileNotFoundError" in result.stderr
    assert missing_config.name in result.stderr
