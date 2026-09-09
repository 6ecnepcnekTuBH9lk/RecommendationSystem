import json

import pytest

from Application.model import BPRMF


def _write_settings(tmp_path, data):
    settings_dir = tmp_path / "Настройки"
    settings_dir.mkdir(parents=True, exist_ok=True)
    path = settings_dir / "filter_settings.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_selected_collections_returns_empty_when_settings_are_missing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    assert BPRMF._load_selected_collections_from_settings() == []


def test_selected_collections_preserves_valid_empty_current_key(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_settings(tmp_path, {"collections_selected": []})

    assert BPRMF._load_selected_collections_from_settings() == []


def test_selected_collections_normalizes_values_and_preserves_order_and_duplicates(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_settings(
        tmp_path,
        {
            "collections_selected": [
                "  Весна–Лето   2026  ",
                "",
                None,
                "Осень—Зима\u00a02025",
                "  Весна–Лето   2026  ",
            ]
        },
    )

    assert BPRMF._load_selected_collections_from_settings() == [
        "Весна-Лето 2026",
        "Осень-Зима 2025",
        "Весна-Лето 2026",
    ]


def test_selected_collections_supports_legacy_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_settings(
        tmp_path,
        {"seasons_selected": ["Весна-Лето 2026"]},
    )

    assert BPRMF._load_selected_collections_from_settings() == [
        "Весна-Лето 2026"
    ]


def test_selected_collections_prefers_non_empty_current_key(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_settings(
        tmp_path,
        {
            "collections_selected": ["Весна-Лето 2027"],
            "seasons_selected": ["Осень-Зима 2026"],
        },
    )

    assert BPRMF._load_selected_collections_from_settings() == [
        "Весна-Лето 2027"
    ]


def test_selected_collections_uses_legacy_when_current_key_is_empty(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_settings(
        tmp_path,
        {
            "collections_selected": [],
            "seasons_selected": ["Осень-Зима 2026"],
        },
    )

    assert BPRMF._load_selected_collections_from_settings() == [
        "Осень-Зима 2026"
    ]


def test_selected_collections_propagates_permission_error_for_existing_file(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    settings_path = _write_settings(tmp_path, {"seasons_selected": []})

    def fail_settings_open(path, *args, **kwargs):
        if path == str(settings_path):
            raise PermissionError("synthetic unreadable filter settings")
        return open(path, *args, **kwargs)

    monkeypatch.setattr(BPRMF, "open", fail_settings_open, raising=False)

    with pytest.raises(
        PermissionError,
        match="synthetic unreadable filter settings",
    ):
        BPRMF._load_selected_collections_from_settings()


def test_selected_collections_propagates_real_json_decode_error(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    settings_dir = tmp_path / "Настройки"
    settings_dir.mkdir()
    (settings_dir / "filter_settings.json").write_text(
        '{"seasons_selected": [',
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        BPRMF._load_selected_collections_from_settings()


def test_selected_collections_keeps_none_as_valid_empty(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_settings(tmp_path, {"collections_selected": None})

    assert BPRMF._load_selected_collections_from_settings() == []


def test_selected_collections_propagates_natural_error_for_list_root(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_settings(tmp_path, [])

    with pytest.raises(AttributeError):
        BPRMF._load_selected_collections_from_settings()
