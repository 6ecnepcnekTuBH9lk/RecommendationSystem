import pandas as pd

from Application.settings.settings_and_filter import add_age_features, clean_id_series


def test_add_age_features_accounts_for_birthday_boundary():
    source = pd.DataFrame(
        {
            "Дата": ["2024-02-28", "2024-02-29"],
            "ДатаРождения": ["2000-02-29", "2000-02-29"],
        }
    )

    result = add_age_features(source)

    assert result["Возраст"].tolist() == [23, 24]
    assert result["ВозрастнаяГруппа"].tolist() == ["14-25", "14-25"]


def test_clean_id_series_removes_excel_suffix_and_normalizes_empty_values():
    source = pd.Series([" 123.0 ", "null", "", None], dtype="object")

    result = clean_id_series(source)

    assert result.iloc[0] == "123"
    assert result.iloc[1:].isna().all()
