from pathlib import Path

import pandas as pd
import pytest

from Application.files import reference_import as api
from Application.files.files_processing import process_coordinates_file


@pytest.mark.parametrize("latitude,longitude", [(91, 0), (-91, 0), (0, 181), (0, -181), (float("inf"), 0)])
def test_coordinate_physical_bounds(latitude, longitude):
    with pytest.raises(ValueError, match="диапазона"):
        process_coordinates_file(None, pd.DataFrame([{"Город": "Synthetic", "Широта": latitude, "Долгота": longitude}]))


@pytest.mark.parametrize("latitude,longitude", [(90, 180), (-90, -180)])
def test_coordinate_endpoints_valid(latitude, longitude):
    result = process_coordinates_file(None, pd.DataFrame([{"Город": "Synthetic", "Широта": latitude, "Долгота": longitude}]))
    assert len(result) == 1


@pytest.mark.parametrize("failure", [None, "schema", "replace"])
def test_reference_snapshot_atomic_replace(tmp_path, monkeypatch, failure):
    source = tmp_path / "source.csv"
    source.write_text("Город,Широта,Долгота\nSynthetic,55.75,37.61\n", encoding="utf-8-sig")
    output = tmp_path / "out"
    output.mkdir()
    destination = output / "КоординатыГородов.csv"
    destination.write_bytes(b"previous snapshot")
    if failure == "schema":
        source.write_text("wrong\nvalue\n", encoding="utf-8")
    if failure == "replace":
        def fail(*args):
            raise OSError("synthetic")
        monkeypatch.setattr(api.os, "replace", fail)
    before = source.read_bytes()
    if failure:
        with pytest.raises((ValueError, OSError)):
            api.import_reference(source, "Координаты городов и погода", output_dir=output)
        assert destination.read_bytes() == b"previous snapshot"
    else:
        result = api.import_reference(source, "Координаты городов и погода", output_dir=output)
        assert result["cities"] == ["Synthetic"]
        assert len(pd.read_csv(destination, sep="|")) == 1
    assert source.read_bytes() == before
    assert list(output.iterdir()) == [destination]


def test_no_legacy_reference_dispatch(tmp_path):
    with pytest.raises(ValueError):
        api.import_reference(Path("unused"), "Заказы клиентов из Mindbox", output_dir=tmp_path)
