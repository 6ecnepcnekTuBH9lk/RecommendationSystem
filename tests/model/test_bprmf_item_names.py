from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from Application.model import BPRMF


_INVALID_INTERACTION_SCHEMA_CASES = [
    ("Заказы.csv", "MindboxID"),
    ("Заказы.csv", "КодНоменклатуры"),
    ("Просмотры.csv", "MindboxID"),
    ("Просмотры.csv", "КодНоменклатуры"),
    ("Просмотры.csv", "ТипТовара"),
    ("Избранное.csv", "MindboxID"),
    ("Избранное.csv", "КодНоменклатуры"),
]


def _write_nomenclature(data_dir: Path, rows, *, columns=None) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "Номенклатура.csv"
    pd.DataFrame(rows, columns=columns).to_csv(
        path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )
    return path


def test_load_item_names_returns_empty_when_source_is_missing(tmp_path):
    assert BPRMF._load_item_names(str(tmp_path / "missing-data")) == {}


def test_load_item_names_uses_site_name(tmp_path):
    data_dir = tmp_path / "site-name"
    _write_nomenclature(
        data_dir,
        [{"КодНоменклатуры": "A", "НазваниеНаСайте": "  Site name  "}],
    )

    assert BPRMF._load_item_names(str(data_dir)) == {"A": "Site name"}


def test_load_item_names_falls_back_to_nomenclature_name(tmp_path):
    data_dir = tmp_path / "fallback-name"
    _write_nomenclature(
        data_dir,
        [
            {
                "КодНоменклатуры": "A",
                "НазваниеНаСайте": "-",
                "Номенклатура": "  Fallback name  ",
            }
        ],
    )

    assert BPRMF._load_item_names(str(data_dir)) == {"A": "Fallback name"}


def test_load_item_names_prefers_site_name_when_both_are_present(tmp_path):
    data_dir = tmp_path / "preferred-name"
    _write_nomenclature(
        data_dir,
        [
            {
                "КодНоменклатуры": "A",
                "НазваниеНаСайте": "Site name",
                "Номенклатура": "Nomenclature name",
            }
        ],
    )

    assert BPRMF._load_item_names(str(data_dir)) == {"A": "Site name"}


def test_load_item_names_ignores_empty_and_sentinel_names(tmp_path):
    data_dir = tmp_path / "empty-names"
    _write_nomenclature(
        data_dir,
        [
            {"КодНоменклатуры": "A", "НазваниеНаСайте": ""},
            {"КодНоменклатуры": "B", "НазваниеНаСайте": " NaN "},
            {"КодНоменклатуры": "C", "НазваниеНаСайте": "-"},
        ],
    )

    assert BPRMF._load_item_names(str(data_dir)) == {}


def test_load_item_names_preserves_duplicate_keep_last_semantics(tmp_path):
    data_dir = tmp_path / "duplicates"
    _write_nomenclature(
        data_dir,
        [
            {"КодНоменклатуры": "A", "НазваниеНаСайте": "First A"},
            {"КодНоменклатуры": "A", "НазваниеНаСайте": "Last A"},
            {"КодНоменклатуры": "B", "НазваниеНаСайте": "Only non-empty B"},
            {"КодНоменклатуры": "B", "НазваниеНаСайте": "-"},
        ],
    )

    assert BPRMF._load_item_names(str(data_dir)) == {
        "A": "Last A",
        "B": "Only non-empty B",
    }


def test_load_item_names_returns_empty_for_valid_empty_source(tmp_path):
    data_dir = tmp_path / "valid-empty"
    _write_nomenclature(
        data_dir,
        [],
        columns=["КодНоменклатуры", "НазваниеНаСайте"],
    )

    assert BPRMF._load_item_names(str(data_dir)) == {}


def test_load_item_names_returns_empty_without_item_code_column(tmp_path):
    data_dir = tmp_path / "without-code"
    _write_nomenclature(data_dir, [{"НазваниеНаСайте": "Synthetic name"}])

    assert BPRMF._load_item_names(str(data_dir)) == {}


def test_load_item_names_returns_empty_without_name_columns(tmp_path):
    data_dir = tmp_path / "without-names"
    _write_nomenclature(data_dir, [{"КодНоменклатуры": "A"}])

    assert BPRMF._load_item_names(str(data_dir)) == {}


def test_load_item_names_propagates_permission_error_for_existing_source(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "unreadable"
    _write_nomenclature(
        data_dir,
        [{"КодНоменклатуры": "A", "НазваниеНаСайте": "Synthetic"}],
    )

    def fail_read(path):
        raise PermissionError("synthetic item names read failure")

    monkeypatch.setattr(BPRMF, "_read_csv_pipe", fail_read)

    with pytest.raises(PermissionError, match="synthetic item names read failure"):
        BPRMF._load_item_names(str(data_dir))


def test_load_item_names_propagates_parser_error_for_existing_source(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "malformed"
    _write_nomenclature(
        data_dir,
        [{"КодНоменклатуры": "A", "НазваниеНаСайте": "Synthetic"}],
    )

    def fail_read(path):
        raise pd.errors.ParserError("synthetic item names parser failure")

    monkeypatch.setattr(BPRMF, "_read_csv_pipe", fail_read)

    with pytest.raises(
        pd.errors.ParserError,
        match="synthetic item names parser failure",
    ):
        BPRMF._load_item_names(str(data_dir))


class _SyntheticCliModel:
    def __init__(self):
        self.user_emb = type(
            "SyntheticEmbedding",
            (),
            {"weight": torch.tensor([[1.0]], dtype=torch.float32)},
        )()

    def load_state_dict(self, state_dict):
        return None

    def eval(self):
        return None

    def item_vec_all(self):
        return torch.tensor([[2.0], [1.0]], dtype=torch.float32)


def _prepare_cli_recommendations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "ВходныеДанные"
    data_dir.mkdir()
    pd.DataFrame(
        columns=[
            "MindboxID",
            "КодНоменклатуры",
            "Количество",
            "Телефон",
            "ДисконтнаяКарта",
            "Почта",
        ]
    ).to_csv(
        data_dir / "Заказы.csv",
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        columns=["MindboxID", "КодНоменклатуры", "ТипТовара"]
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

    mappings = {
        "idx2user": ["user-1"],
        "idx2item": ["item-a", "item-b"],
    }
    checkpoint = {
        "num_users": 1,
        "num_items": 2,
        "config": {"embedding_dim": 1},
        "state_dict": {},
        "feat2idx": {},
        "item_feat_mat": None,
    }
    monkeypatch.setattr(
        BPRMF,
        "_load_artifacts",
        lambda model_dir="Модель": (mappings, checkpoint),
    )
    monkeypatch.setattr(
        BPRMF,
        "_user_seen_items_from_processed",
        lambda *args, **kwargs: np.array([], dtype=np.int64),
    )
    monkeypatch.setattr(BPRMF, "BPRMF", lambda *args, **kwargs: _SyntheticCliModel())


def test_cli_name_failure_preserves_codes_scores_order_and_success(
    tmp_path, monkeypatch, capsys
):
    _prepare_cli_recommendations(tmp_path, monkeypatch)
    monkeypatch.setattr(
        BPRMF,
        "_load_item_names",
        lambda data_dir: {"item-a": "Name A", "item-b": "Name B"},
    )

    assert BPRMF.main(["--recommend", "user-1", "--k", "2"]) == 0
    baseline = capsys.readouterr()

    def fail_names(data_dir):
        raise PermissionError("synthetic CLI item names failure")

    monkeypatch.setattr(BPRMF, "_load_item_names", fail_names)

    assert BPRMF.main(["--recommend", "user-1", "--k", "2"]) == 0
    degraded = capsys.readouterr()

    baseline_rows = [
        line for line in baseline.out.splitlines() if line.startswith(("01.", "02."))
    ]
    degraded_rows = [
        line for line in degraded.out.splitlines() if line.startswith(("01.", "02."))
    ]
    assert baseline_rows == [
        "01. item-a | Name A | score=2.0000",
        "02. item-b | Name B | score=1.0000",
    ]
    assert degraded_rows == [
        "01. item-a | score=2.0000",
        "02. item-b | score=1.0000",
    ]
    assert degraded.err.count("Не удалось загрузить названия товаров") == 1
    assert "synthetic CLI item names failure" in degraded.err


@pytest.mark.parametrize(
    "missing_filename",
    ["Заказы.csv", "Просмотры.csv", "Избранное.csv"],
)
def test_cli_recommendations_fail_when_required_interaction_source_is_missing(
    tmp_path,
    monkeypatch,
    capsys,
    missing_filename,
):
    _prepare_cli_recommendations(tmp_path, monkeypatch)
    (tmp_path / "ВходныеДанные" / missing_filename).unlink()

    exit_code = BPRMF.main(["--recommend", "user-1", "--k", "2"])
    captured = capsys.readouterr()

    assert exit_code == BPRMF.TRAIN_EXIT_NO_DATA
    assert missing_filename in captured.err
    assert "Recommendations (BPR-MF)" not in captured.out


@pytest.mark.parametrize(
    ("filename", "missing_column"),
    _INVALID_INTERACTION_SCHEMA_CASES,
)
def test_cli_recommendations_fail_when_interaction_schema_is_invalid(
    tmp_path,
    monkeypatch,
    capsys,
    filename,
    missing_column,
):
    _prepare_cli_recommendations(tmp_path, monkeypatch)
    source_path = tmp_path / "ВходныеДанные" / filename
    columns = list(pd.read_csv(source_path, sep="|", dtype=str).columns)
    columns.remove(missing_column)
    pd.DataFrame(columns=columns).to_csv(
        source_path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )

    exit_code = BPRMF.main(["--recommend", "user-1", "--k", "2"])
    captured = capsys.readouterr()

    assert exit_code == BPRMF.TRAIN_EXIT_NO_DATA
    assert filename in captured.err
    assert missing_column in captured.err
    assert "Recommendations (BPR-MF)" not in captured.out


@pytest.mark.parametrize(
    "read_error",
    [
        PermissionError("synthetic interaction permission error"),
        pd.errors.ParserError("synthetic interaction parser error"),
    ],
    ids=["permission", "parser"],
)
def test_cli_interaction_schema_read_error_remains_technical(
    tmp_path,
    monkeypatch,
    capsys,
    read_error,
):
    _prepare_cli_recommendations(tmp_path, monkeypatch)

    def fail_read(*args, **kwargs):
        raise read_error

    monkeypatch.setattr(BPRMF.pd, "read_csv", fail_read)

    with pytest.raises(type(read_error), match=str(read_error)):
        BPRMF.main(["--recommend", "user-1", "--k", "2"])

    captured = capsys.readouterr()
    assert "Recommendations (BPR-MF)" not in captured.out
