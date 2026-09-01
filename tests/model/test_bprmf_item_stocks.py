from pathlib import Path

import pandas as pd
import pytest

from Application.model import BPRMF


def _write_nomenclature(data_dir: Path, rows, *, columns=None) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "Номенклатура.csv"
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, sep="|", index=False, encoding="utf-8-sig")
    return path


def test_load_item_stocks_returns_empty_when_source_is_missing(tmp_path):
    assert BPRMF._load_item_stocks(str(tmp_path / "missing-data")) == {}


def test_load_item_stocks_preserves_valid_mapping(tmp_path):
    data_dir = tmp_path / "valid-data"
    _write_nomenclature(
        data_dir,
        [
            {"КодНоменклатуры": "A", "Остаток": "150"},
            {"КодНоменклатуры": "B", "Остаток": "20"},
        ],
    )

    assert BPRMF._load_item_stocks(str(data_dir)) == {
        "A": "150",
        "B": "20",
    }


def test_load_item_stocks_returns_empty_for_valid_empty_source(tmp_path):
    data_dir = tmp_path / "empty-data"
    _write_nomenclature(
        data_dir,
        [],
        columns=["КодНоменклатуры", "Остаток"],
    )

    assert BPRMF._load_item_stocks(str(data_dir)) == {}


def test_load_item_stocks_propagates_permission_error_for_existing_source(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "unreadable-data"
    _write_nomenclature(
        data_dir,
        [{"КодНоменклатуры": "A", "Остаток": "150"}],
    )

    def fail_read(path):
        raise PermissionError("synthetic stock read failure")

    monkeypatch.setattr(BPRMF, "_read_csv_pipe", fail_read)

    with pytest.raises(PermissionError, match="synthetic stock read failure"):
        BPRMF._load_item_stocks(str(data_dir))


def test_load_item_stocks_propagates_parser_error_for_existing_source(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "malformed-data"
    _write_nomenclature(
        data_dir,
        [{"КодНоменклатуры": "A", "Остаток": "150"}],
    )

    def fail_read(path):
        raise pd.errors.ParserError("synthetic stock parser failure")

    monkeypatch.setattr(BPRMF, "_read_csv_pipe", fail_read)

    with pytest.raises(pd.errors.ParserError, match="synthetic stock parser failure"):
        BPRMF._load_item_stocks(str(data_dir))


@pytest.mark.parametrize("missing_column", ["КодНоменклатуры", "Остаток"])
def test_load_item_stocks_rejects_existing_source_without_required_column(
    tmp_path, missing_column
):
    data_dir = tmp_path / f"missing-{missing_column}"
    row = {"КодНоменклатуры": "A", "Остаток": "150"}
    del row[missing_column]
    _write_nomenclature(data_dir, [row])

    with pytest.raises(ValueError, match=missing_column):
        BPRMF._load_item_stocks(str(data_dir))
