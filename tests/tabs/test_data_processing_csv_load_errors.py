from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from Application.tabs import data_processing_tab


class _Choice:
    def __init__(self, value):
        self._value = value

    def currentText(self):
        return self._value


class _FileDialog:
    selected_path = ""

    @classmethod
    def getOpenFileName(cls, *args, **kwargs):
        return cls.selected_path, "CSV files (*.csv)"


DOWNSTREAM_FUNCTIONS = (
    "update_file_status",
    "update_filter_controls_availability",
    "refresh_kind_values_from_loaded_files",
    "refresh_season_values_from_nomenclature_file",
    "refresh_export_kind_values_from_nomenclature_file",
    "update_filter_summary",
    "analyze_orders_full_dataset",
    "analyze_views_full_dataset",
    "analyze_favorites_full_dataset",
)


def _prepare_window(monkeypatch, tmp_path, selected_type):
    source_path = tmp_path / "synthetic-source.csv"
    source_path.write_text("synthetic source", encoding="utf-8")
    _FileDialog.selected_path = str(source_path)

    window = SimpleNamespace(
        combo_box_types=_Choice(selected_type),
        combo_box_add_or_not=_Choice("Добавить новый / Обновить существующий"),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data_processing_tab, "QFileDialog", _FileDialog)

    messages = []
    processing_statuses = []
    error_statuses = []
    success_statuses = []
    resets = []
    monkeypatch.setattr(
        data_processing_tab,
        "show_custom_message",
        lambda current_window, **kwargs: messages.append(kwargs),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_processing",
        lambda current_window, text: processing_statuses.append(text),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_error",
        lambda current_window, text: error_statuses.append(text),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_ok",
        lambda current_window, text: success_statuses.append(text),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "schedule_status_reset",
        lambda current_window, seconds: resets.append(seconds),
    )

    downstream = {}
    for function_name in DOWNSTREAM_FUNCTIONS:
        downstream[function_name] = Mock(name=function_name)
        monkeypatch.setattr(
            data_processing_tab,
            function_name,
            downstream[function_name],
        )

    return {
        "window": window,
        "source_path": source_path,
        "messages": messages,
        "processing_statuses": processing_statuses,
        "error_statuses": error_statuses,
        "success_statuses": success_statuses,
        "resets": resets,
        "downstream": downstream,
    }


@pytest.mark.parametrize(
    ("selected_type", "processor_name", "target_filename"),
    [
        (
            "Заказы клиентов из Mindbox",
            "process_orders_file",
            "Заказы.csv",
        ),
        (
            "Просмотры товаров и категорий из Mindbox",
            "process_views_file",
            "Просмотры.csv",
        ),
        (
            "Добавление товаров в избранное из Mindbox",
            "process_favorites_file",
            "Избранное.csv",
        ),
    ],
)
def test_pair_route_read_failure_stops_without_secondary_processing(
    tmp_path,
    monkeypatch,
    selected_type,
    processor_name,
    target_filename,
):
    context = _prepare_window(monkeypatch, tmp_path, selected_type)
    target_dir = tmp_path / "ВходныеДанные"
    target_dir.mkdir()
    target_path = target_dir / target_filename
    old_bytes = b"synthetic existing target bytes"
    target_path.write_bytes(old_bytes)

    processor = Mock(
        side_effect=AttributeError("secondary processor call with None")
    )
    monkeypatch.setattr(data_processing_tab, processor_name, processor)
    monkeypatch.setattr(
        data_processing_tab.pd,
        "read_csv",
        Mock(side_effect=PermissionError("synthetic source read failure")),
    )
    publication = Mock(
        side_effect=AssertionError("publication must not start")
    )
    monkeypatch.setattr(data_processing_tab.tempfile, "mkstemp", publication)

    data_processing_tab.load_csv_file(context["window"])

    processor.assert_not_called()
    publication.assert_not_called()
    assert target_path.read_bytes() == old_bytes
    assert list(target_dir.glob(".*.tmp")) == []
    for downstream_call in context["downstream"].values():
        downstream_call.assert_not_called()
    assert context["processing_statuses"] == ["Обработка данных..."]
    assert context["error_statuses"] == ["Ошибка чтения файла"]
    assert context["success_statuses"] == []
    assert context["resets"] == [5]
    assert len(context["messages"]) == 1
    assert "Не удалось прочитать файл" in context["messages"][0]["text"]
    assert "synthetic source read failure" in context["messages"][0]["text"]
    assert "Ошибка обработки" not in context["error_statuses"]


def test_successful_pair_route_preserves_processing_publication_and_refresh(
    tmp_path, monkeypatch
):
    context = _prepare_window(
        monkeypatch,
        tmp_path,
        "Просмотры товаров и категорий из Mindbox",
    )
    source_frame = pd.DataFrame([{"source": "value"}])
    processed_frame = pd.DataFrame(
        [{"MindboxID": "synthetic-user", "КодНоменклатуры": "synthetic-item"}]
    )
    reader = Mock(return_value=source_frame)
    processor = Mock(return_value=processed_frame)
    monkeypatch.setattr(data_processing_tab, "read_csv_auto_encoding", reader)
    monkeypatch.setattr(data_processing_tab, "process_views_file", processor)
    real_mkstemp = data_processing_tab.tempfile.mkstemp
    publication = Mock(wraps=real_mkstemp)
    monkeypatch.setattr(data_processing_tab.tempfile, "mkstemp", publication)

    data_processing_tab.load_csv_file(context["window"])

    reader.assert_called_once_with(
        context["window"],
        file_path=str(context["source_path"]),
        sep=";",
    )
    assert processor.call_count == 1
    assert processor.call_args.args[0] is context["window"]
    assert processor.call_args.args[1] is source_frame
    assert publication.call_count == 1
    target_path = tmp_path / "ВходныеДанные" / "Просмотры.csv"
    result = pd.read_csv(target_path, sep="|", encoding="utf-8-sig", dtype=str)
    assert result.to_dict(orient="records") == processed_frame.to_dict(
        orient="records"
    )
    for downstream_call in context["downstream"].values():
        downstream_call.assert_called_once_with(context["window"])
    assert context["messages"] == []
    assert context["error_statuses"] == []
