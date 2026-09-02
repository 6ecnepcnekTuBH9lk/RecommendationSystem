from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from Application.photo import photo_processing
from Application.tabs import create_results_tab


CLIENT_FIELDS = (
    "le_mb",
    "le_card",
    "le_email",
    "le_phone",
    "le_gender",
    "le_fio",
    "le_age",
    "le_agegrp",
)


def test_item_name_map_failed_read_does_not_publish_and_second_call_retries(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "ВходныеДанные"
    data_dir.mkdir()
    (data_dir / "Номенклатура.csv").write_text(
        "synthetic source",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    read_results = iter([
        PermissionError("synthetic item name read failure"),
        pd.DataFrame([
            {
                "КодНоменклатуры": "item-1",
                "НазваниеНаСайте": "Synthetic item",
            }
        ]),
    ])
    read_calls = []

    def read_csv(path, **kwargs):
        read_calls.append((path, kwargs))
        result = next(read_results)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(create_results_tab.pd, "read_csv", read_csv)
    tab = SimpleNamespace(_name_by_code=None)

    with pytest.raises(
        PermissionError,
        match="synthetic item name read failure",
    ):
        create_results_tab._ensure_item_name_map(tab)

    assert tab._name_by_code is None

    create_results_tab._ensure_item_name_map(tab)

    assert len(read_calls) == 2
    assert tab._name_by_code == {"item-1": "Synthetic item"}


def test_persistent_item_name_read_error_keeps_client_search_in_degraded_mode(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "ВходныеДанные"
    data_dir.mkdir()
    pd.DataFrame([
        {
            "MindboxID": "client-1",
            "КодНоменклатуры": "history-item",
            "Дата": "01.01.2026",
        }
    ]).to_csv(
        data_dir / "Заказы.csv",
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )
    nomenclature_path = data_dir / "Номенклатура.csv"
    nomenclature_path.write_text("synthetic source", encoding="utf-8")

    model_dir = tmp_path / "Модель"
    model_dir.mkdir()
    (model_dir / "Рекомендации.xlsx").write_bytes(b"synthetic workbook")
    monkeypatch.chdir(tmp_path)

    recommendations = pd.DataFrame([
        {
            "MindboxID": "client-1",
            "КодНоменклатуры": "recommendation-item",
            "Коллекция": "Synthetic collection",
            "Коэффициент": "1.0",
            "Конверсия": "12.5",
            "Остаток": "150",
        }
    ])
    real_read_csv = create_results_tab.pd.read_csv
    item_name_reads = []

    def read_csv(path, *args, **kwargs):
        if Path(path).resolve() == nomenclature_path.resolve():
            item_name_reads.append(Path(path))
            raise PermissionError("synthetic persistent item name read failure")
        return real_read_csv(path, *args, **kwargs)

    tab = SimpleNamespace(
        client_filter_field=_Choice("MindboxID"),
        mb_input=_TextField("client-1"),
        recs_topk=_Choice("Топ-1"),
        status_label=_TextField(),
        status_icon=_TextField(),
        _name_by_code=None,
        _collection_by_code={},
        _stock_by_code={},
    )
    applied_results = []
    status_events = []
    messages = []

    monkeypatch.setattr(create_results_tab, "QApplication", _Application)
    monkeypatch.setattr(
        create_results_tab,
        "_resolve_mindbox_ids",
        lambda field, value: [value],
    )
    monkeypatch.setattr(
        create_results_tab,
        "_load_client_info",
        lambda mindbox_id: {"MindboxID": mindbox_id},
    )
    monkeypatch.setattr(create_results_tab, "_ensure_photo_map", lambda aboba: None)
    monkeypatch.setattr(
        create_results_tab,
        "_photo_url_for_code",
        lambda aboba, code: "",
    )
    monkeypatch.setattr(
        create_results_tab,
        "_ensure_item_collection_map",
        lambda aboba: None,
    )
    monkeypatch.setattr(
        create_results_tab,
        "_ensure_item_stock_map",
        lambda aboba: None,
    )
    monkeypatch.setattr(create_results_tab.pd, "read_csv", read_csv)
    monkeypatch.setattr(
        create_results_tab.pd,
        "read_excel",
        lambda path, dtype: recommendations.copy(),
    )
    monkeypatch.setattr(
        create_results_tab,
        "_apply_client_search_result",
        lambda aboba, info, purchase_rows, recommendation_rows: applied_results.append(
            (info, purchase_rows, recommendation_rows)
        ),
    )
    monkeypatch.setattr(
        create_results_tab,
        "set_status_processing",
        lambda aboba, text: status_events.append("processing"),
    )
    monkeypatch.setattr(
        create_results_tab,
        "set_status_ok",
        lambda aboba, text: status_events.append("success"),
    )
    monkeypatch.setattr(
        create_results_tab,
        "set_status_error",
        lambda aboba, text: status_events.append("error"),
    )
    monkeypatch.setattr(
        create_results_tab,
        "schedule_status_reset",
        lambda aboba, seconds: None,
    )
    monkeypatch.setattr(
        create_results_tab,
        "show_custom_message",
        lambda *args, **kwargs: messages.append((args, kwargs)),
    )

    create_results_tab.show_purchase_history_clicked(tab)

    assert len(item_name_reads) == 2
    assert tab._name_by_code is None
    assert len(applied_results) == 1
    _, purchase_rows, recommendation_rows = applied_results[0]
    assert purchase_rows[0][0:2] == ("history-item", "")
    assert recommendation_rows[0][0:2] == ("recommendation-item", "")
    assert status_events == ["processing", "success"]
    assert messages == []


class _TextField:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text

    def setCursorPosition(self, position):
        self.cursor_position = position

    def repaint(self):
        pass


class _Choice:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text

    def setCurrentText(self, text):
        self._text = text


class _TableItem:
    def __init__(self, text):
        self.text = text

    def setTextAlignment(self, alignment):
        pass


class _Table:
    def __init__(self):
        self._row_count = 0
        self.items = {}

    def rowCount(self):
        return self._row_count

    def setRowCount(self, count):
        self._row_count = count
        self.items = {
            (row, column): item
            for (row, column), item in self.items.items()
            if row < count
        }

    def clearContents(self):
        self.items.clear()

    def setItem(self, row, column, item):
        self.items[(row, column)] = item

    def values(self, columns):
        return [
            tuple(self.items[(row, column)].text for column in columns)
            for row in range(self._row_count)
        ]


class _Application:
    @staticmethod
    def processEvents():
        pass


def _client_info(client):
    return {
        "MindboxID": client,
        "ДисконтнаяКарта": f"card-{client}",
        "Почта": f"{client.lower()}@example.test",
        "Телефон": f"phone-{client}",
        "ФИО": f"Name {client}",
        "ПолКлиента": f"gender-{client}",
        "Возраст": f"age-{client}",
        "ВозрастнаяГруппа": f"group-{client}",
    }


def _interactions(client):
    return pd.DataFrame(
        [
            {
                "КодНоменклатуры": f"purchase-code-{client}",
                "НазваниеНоменклатуры": f"purchase-name-{client}",
                "Коллекция": f"purchase-collection-{client}",
                "Взаимодействие": "Покупка",
                "ДатаВзаимодействия": "01.01.2026",
            }
        ]
    )


def _recommendations(client):
    return pd.DataFrame(
        [
            {
                "КодНоменклатуры": f"rec-code-{client}",
                "НазваниеНоменклатуры": f"rec-name-{client}",
                "Коллекция": f"rec-collection-{client}",
                "Коэффициент": f"coef-{client}",
                "Конверсия": f"conversion-{client}",
                "Остаток": f"stock-{client}",
            }
        ]
    )


def _empty_interactions():
    return pd.DataFrame(
        columns=[
            "КодНоменклатуры",
            "НазваниеНоменклатуры",
            "Коллекция",
            "Взаимодействие",
            "ДатаВзаимодействия",
        ]
    )


def _empty_recommendations():
    return pd.DataFrame(
        columns=[
            "КодНоменклатуры",
            "НазваниеНоменклатуры",
            "Коллекция",
            "Коэффициент",
            "Конверсия",
            "Остаток",
        ]
    )


@pytest.fixture
def search_harness(monkeypatch):
    window = SimpleNamespace(
        client_filter_field=_Choice("MindboxID"),
        mb_input=_TextField("A"),
        recs_topk=_Choice("Топ-10"),
        purchases_table=_Table(),
        recs_table=_Table(),
        status_label=_TextField(),
        status_icon=_TextField(),
        _img_gen=0,
        _img_queue=deque(),
        _img_targets={},
        _img_inflight=set(),
        _img_retry_count={},
    )
    for field in CLIENT_FIELDS:
        setattr(window, field, _TextField())

    control = SimpleNamespace(
        not_found=False,
        photo_error=None,
        interaction_errors={},
        recommendation_errors={},
        photo_url_errors={},
        interaction_frames={},
        recommendation_frames={},
        info_by_client={},
        resolved_values=[],
        loaded_info=[],
        loaded_interactions=[],
        loaded_recommendations=[],
        photo_lookups=[],
        status_events=[],
        reset_delays=[],
        messages=[],
        photo_cells=[],
    )

    def resolve(field, value):
        control.resolved_values.append((field, value))
        return [] if control.not_found else [value]

    def load_info(client):
        control.loaded_info.append(client)
        return control.info_by_client.get(client, _client_info(client))

    def ensure_photo_map(aboba):
        if control.photo_error is not None:
            raise control.photo_error

    def load_interactions(aboba, client):
        control.loaded_interactions.append(client)
        if client in control.interaction_errors:
            raise control.interaction_errors[client]
        return control.interaction_frames.get(client, _interactions(client)).copy()

    def load_recommendations(aboba, client, topk):
        control.loaded_recommendations.append((client, topk))
        if client in control.recommendation_errors:
            raise control.recommendation_errors[client]
        return control.recommendation_frames.get(client, _recommendations(client)).copy()

    def photo_url_for_code(aboba, code):
        control.photo_lookups.append(code)
        if code in control.photo_url_errors:
            raise control.photo_url_errors[code]
        return f"https://example.test/{code}.jpg"

    monkeypatch.setattr(create_results_tab, "QApplication", _Application)
    monkeypatch.setattr(create_results_tab, "QLineEdit", _TextField)
    monkeypatch.setattr(create_results_tab, "QTableWidgetItem", _TableItem)
    monkeypatch.setattr(create_results_tab, "_resolve_mindbox_ids", resolve)
    monkeypatch.setattr(create_results_tab, "_load_client_info", load_info)
    monkeypatch.setattr(create_results_tab, "_ensure_photo_map", ensure_photo_map)
    monkeypatch.setattr(create_results_tab, "_load_client_interactions", load_interactions)
    monkeypatch.setattr(
        create_results_tab,
        "_load_recommendations_from_excel",
        load_recommendations,
    )
    monkeypatch.setattr(
        create_results_tab,
        "_photo_url_for_code",
        photo_url_for_code,
        raising=False,
    )
    monkeypatch.setattr(
        create_results_tab,
        "_set_photo_cell",
        lambda aboba, table, row, code, generation, photo_url=None: control.photo_cells.append(
            (table, row, code, generation, photo_url)
        ),
    )
    monkeypatch.setattr(
        create_results_tab,
        "set_status_processing",
        lambda aboba, text: control.status_events.append(("processing", text)),
    )
    monkeypatch.setattr(
        create_results_tab,
        "set_status_ok",
        lambda aboba, text: control.status_events.append(("success", text)),
    )
    monkeypatch.setattr(
        create_results_tab,
        "set_status_error",
        lambda aboba, text: control.status_events.append(("error", text)),
    )
    monkeypatch.setattr(
        create_results_tab,
        "schedule_status_reset",
        lambda aboba, seconds: control.reset_delays.append(seconds),
    )
    monkeypatch.setattr(
        create_results_tab,
        "show_custom_message",
        lambda aboba, title, text, image_path=None: control.messages.append(
            (title, text, image_path)
        ),
    )

    return window, control


def _set_search(window, client, field="MindboxID"):
    window.client_filter_field.setCurrentText(field)
    window.mb_input.setText(client)


def _result_snapshot(window):
    return {
        "panel": {field: getattr(window, field).text() for field in CLIENT_FIELDS},
        "purchases": window.purchases_table.values(range(1, 6)),
        "recommendations": window.recs_table.values(range(1, 7)),
    }


def _show(window):
    create_results_tab.show_purchase_history_clicked(window)


def test_successful_search_populates_complete_result(search_harness):
    window, control = search_harness

    _show(window)

    assert _result_snapshot(window) == {
        "panel": {
            "le_mb": "A",
            "le_card": "card-A",
            "le_email": "a@example.test",
            "le_phone": "phone-A",
            "le_gender": "gender-A",
            "le_fio": "Name A",
            "le_age": "age-A",
            "le_agegrp": "group-A",
        },
        "purchases": [
            (
                "purchase-code-A",
                "purchase-name-A",
                "purchase-collection-A",
                "Покупка",
                "01.01.2026",
            )
        ],
        "recommendations": [
            (
                "rec-code-A",
                "rec-name-A",
                "rec-collection-A",
                "coef-A",
                "conversion-A",
                "stock-A",
            )
        ],
    }
    assert window._img_gen == 1
    assert [event[0] for event in control.status_events] == ["processing", "success"]
    assert control.reset_delays == [5]


def test_recommendation_error_preserves_previous_result_and_image_state(search_harness):
    window, control = search_harness
    _show(window)
    previous_result = _result_snapshot(window)

    window._img_queue.extend(["a-queued"])
    window._img_targets["a-url"] = [(window.purchases_table, 0)]
    window._img_inflight.add("a-url")
    window._img_retry_count["a-url"] = 1
    previous_generation = window._img_gen
    previous_queue = list(window._img_queue)
    previous_targets = dict(window._img_targets)
    previous_inflight = set(window._img_inflight)
    previous_retries = dict(window._img_retry_count)

    control.status_events.clear()
    control.reset_delays.clear()
    control.recommendation_errors["B"] = OSError("synthetic recommendations error")
    _set_search(window, "B")

    _show(window)

    assert _result_snapshot(window) == previous_result
    assert window._img_gen == previous_generation
    assert list(window._img_queue) == previous_queue
    assert window._img_targets == previous_targets
    assert window._img_inflight == previous_inflight
    assert window._img_retry_count == previous_retries
    assert [event[0] for event in control.status_events] == ["processing", "error"]
    assert all(event[0] != "success" for event in control.status_events)
    assert "synthetic recommendations error" in control.messages[-1][1]


def test_photo_map_error_preserves_previous_result(search_harness):
    window, control = search_harness
    _show(window)
    previous_result = _result_snapshot(window)
    previous_generation = window._img_gen

    control.status_events.clear()
    control.photo_error = OSError("synthetic nomenclature error")
    _set_search(window, "B")

    _show(window)

    assert _result_snapshot(window) == previous_result
    assert window._img_gen == previous_generation
    assert [event[0] for event in control.status_events] == ["processing", "error"]


def test_interactions_error_preserves_previous_result(search_harness):
    window, control = search_harness
    _show(window)
    previous_result = _result_snapshot(window)
    previous_generation = window._img_gen

    control.status_events.clear()
    control.interaction_errors["B"] = OSError("synthetic interactions error")
    _set_search(window, "B")

    _show(window)

    assert _result_snapshot(window) == previous_result
    assert window._img_gen == previous_generation
    assert [event[0] for event in control.status_events] == ["processing", "error"]


def test_empty_history_commits_client_and_recommendations_without_success(search_harness):
    window, control = search_harness
    control.interaction_frames["B"] = _empty_interactions()
    _set_search(window, "B")

    _show(window)

    result = _result_snapshot(window)
    assert result["panel"]["le_mb"] == "B"
    assert result["purchases"] == []
    assert result["recommendations"][0][0] == "rec-code-B"
    assert window._img_gen == 1
    assert [event[0] for event in control.status_events] == ["processing", "error"]
    assert control.reset_delays == [5]
    assert all(event[0] != "success" for event in control.status_events)


def test_successful_b_completely_replaces_a(search_harness):
    window, control = search_harness
    _show(window)
    control.status_events.clear()
    _set_search(window, "B")

    _show(window)

    result_text = repr(_result_snapshot(window))
    assert "purchase-code-B" in result_text
    assert "rec-code-B" in result_text
    assert "card-B" in result_text
    assert "purchase-code-A" not in result_text
    assert "rec-code-A" not in result_text
    assert "card-A" not in result_text
    assert window._img_gen == 2
    assert [event[0] for event in control.status_events] == ["processing", "success"]


def test_successful_b_with_empty_recommendations_clears_a_recommendations(search_harness):
    window, control = search_harness
    _show(window)
    control.status_events.clear()
    control.recommendation_frames["B"] = _empty_recommendations()
    _set_search(window, "B")

    _show(window)

    result = _result_snapshot(window)
    assert result["panel"]["le_mb"] == "B"
    assert result["purchases"][0][0] == "purchase-code-B"
    assert result["recommendations"] == []
    assert [event[0] for event in control.status_events] == ["processing", "success"]


def test_client_not_found_clears_previous_result(search_harness):
    window, control = search_harness
    _show(window)
    control.status_events.clear()
    control.not_found = True
    _set_search(window, "missing-phone", field="Телефон")

    _show(window)

    assert all(value == "" for value in _result_snapshot(window)["panel"].values())
    assert window.purchases_table.rowCount() == 0
    assert window.recs_table.rowCount() == 0
    assert [event[0] for event in control.status_events] == ["processing", "error"]
    assert all(event[0] != "success" for event in control.status_events)


def test_empty_input_clears_previous_result(search_harness):
    window, control = search_harness
    _show(window)
    control.status_events.clear()
    _set_search(window, "   ")

    _show(window)

    assert all(value == "" for value in _result_snapshot(window)["panel"].values())
    assert window.purchases_table.rowCount() == 0
    assert window.recs_table.rowCount() == 0
    assert [event[0] for event in control.status_events] == ["error"]
    assert all(event[0] != "success" for event in control.status_events)


def test_unknown_nonempty_mindbox_id_reaches_empty_interactions(search_harness):
    window, control = search_harness
    control.info_by_client["UNKNOWN"] = {}
    control.interaction_frames["UNKNOWN"] = _empty_interactions()
    control.recommendation_frames["UNKNOWN"] = _empty_recommendations()
    _set_search(window, "UNKNOWN")

    _show(window)

    assert control.resolved_values == [("MindboxID", "UNKNOWN")]
    assert control.loaded_info == ["UNKNOWN"]
    assert control.loaded_interactions == ["UNKNOWN"]
    assert control.loaded_recommendations == [("UNKNOWN", 10)]
    assert any("взаимодействий не найдено" in message[1] for message in control.messages)
    assert all("клиент не найден" not in message[1].lower() for message in control.messages)


def test_row_preparation_error_does_not_partially_commit_b(search_harness):
    window, control = search_harness
    _show(window)
    previous_result = _result_snapshot(window)
    previous_generation = window._img_gen

    control.status_events.clear()
    control.interaction_frames["B"] = pd.DataFrame(
        [{"КодНоменклатуры": "incomplete-B"}]
    )
    _set_search(window, "B")

    _show(window)

    assert _result_snapshot(window) == previous_result
    assert window._img_gen == previous_generation
    assert [event[0] for event in control.status_events] == ["processing", "error"]


def test_commit_uses_prepared_photo_urls_without_catalog_lookup(
    search_harness, monkeypatch
):
    window, control = search_harness
    _show(window)
    control.status_events.clear()
    control.photo_lookups.clear()
    control.photo_cells.clear()
    _set_search(window, "B")

    lookup_allowed = True
    missing_url = object()
    real_apply = create_results_tab._apply_client_search_result

    def guarded_lookup(aboba, code):
        if not lookup_allowed:
            raise AssertionError("photo lookup after commit")
        control.photo_lookups.append(code)
        return f"https://example.test/{code}.jpg"

    def guarded_photo_cell(
        aboba, table, row, code, generation, photo_url=missing_url
    ):
        if photo_url is missing_url:
            photo_url = create_results_tab._photo_url_for_code(aboba, code)
        control.photo_cells.append((table, row, code, generation, photo_url))

    def mark_commit_started(*args, **kwargs):
        nonlocal lookup_allowed
        lookup_allowed = False
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(create_results_tab, "_photo_url_for_code", guarded_lookup)
    monkeypatch.setattr(create_results_tab, "_set_photo_cell", guarded_photo_cell)
    monkeypatch.setattr(
        create_results_tab,
        "_apply_client_search_result",
        mark_commit_started,
    )

    _show(window)

    result = _result_snapshot(window)
    assert result["panel"]["le_mb"] == "B"
    assert result["purchases"][0][0] == "purchase-code-B"
    assert result["recommendations"][0][0] == "rec-code-B"
    assert control.photo_lookups == ["purchase-code-B", "rec-code-B"]
    assert [cell[4] for cell in control.photo_cells] == [
        "https://example.test/purchase-code-B.jpg",
        "https://example.test/rec-code-B.jpg",
    ]
    assert window._img_gen == 2
    assert [event[0] for event in control.status_events] == ["processing", "success"]


def test_photo_url_preparation_error_preserves_previous_result(search_harness):
    window, control = search_harness
    _show(window)
    previous_result = _result_snapshot(window)
    previous_generation = window._img_gen

    window._img_queue.extend(["a-queued"])
    window._img_targets["a-url"] = [(window.purchases_table, 0)]
    previous_queue = list(window._img_queue)
    previous_targets = dict(window._img_targets)

    control.status_events.clear()
    control.photo_url_errors["purchase-code-B"] = PermissionError(
        "synthetic photo lookup error"
    )
    _set_search(window, "B")

    _show(window)

    assert _result_snapshot(window) == previous_result
    assert window._img_gen == previous_generation
    assert list(window._img_queue) == previous_queue
    assert window._img_targets == previous_targets
    assert [event[0] for event in control.status_events] == ["processing", "error"]
    assert all(event[0] != "success" for event in control.status_events)


class _PhotoLabel:
    def __init__(self):
        self.text = ""

    def setAlignment(self, alignment):
        pass

    def setContentsMargins(self, *margins):
        pass

    def setFixedSize(self, width, height):
        pass

    def setStyleSheet(self, stylesheet):
        pass

    def setPixmap(self, pixmap):
        pass

    def clear(self):
        pass

    def setText(self, text):
        self.text = text


class _PhotoPixmap:
    def __init__(self, width, height):
        pass

    def fill(self, color):
        pass


class _PhotoTable:
    def setCellWidget(self, row, column, widget):
        self.widget = widget

    def setRowHeight(self, row, height):
        pass

    def setColumnWidth(self, column, width):
        pass


def _photo_window():
    return SimpleNamespace(
        _img_cache={
            "legacy-url": None,
            "prepared-url": None,
        },
        _img_targets={},
        _img_inflight=set(),
        _img_queue=deque(),
    )


def test_set_photo_cell_legacy_caller_still_uses_lookup(monkeypatch):
    lookups = []
    monkeypatch.setattr(photo_processing, "QLabel", _PhotoLabel)
    monkeypatch.setattr(photo_processing, "QPixmap", _PhotoPixmap)
    monkeypatch.setattr(
        photo_processing,
        "_photo_url_for_code",
        lambda aboba, code: lookups.append(code) or "legacy-url",
    )

    photo_processing._set_photo_cell(_photo_window(), _PhotoTable(), 0, "code", 1)

    assert lookups == ["code"]


def test_set_photo_cell_prepared_url_skips_lookup(monkeypatch):
    monkeypatch.setattr(photo_processing, "QLabel", _PhotoLabel)
    monkeypatch.setattr(photo_processing, "QPixmap", _PhotoPixmap)
    monkeypatch.setattr(
        photo_processing,
        "_photo_url_for_code",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected lookup")),
    )

    photo_processing._set_photo_cell(
        _photo_window(),
        _PhotoTable(),
        0,
        "code",
        1,
        photo_url="prepared-url",
    )


@pytest.mark.parametrize("prepared_no_photo", [None, ""])
def test_set_photo_cell_prepared_no_photo_skips_lookup_and_shows_placeholder(
    monkeypatch, prepared_no_photo
):
    monkeypatch.setattr(photo_processing, "QLabel", _PhotoLabel)
    monkeypatch.setattr(photo_processing, "QPixmap", _PhotoPixmap)
    monkeypatch.setattr(
        photo_processing,
        "_photo_url_for_code",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected lookup")),
    )
    table = _PhotoTable()

    photo_processing._set_photo_cell(
        _photo_window(),
        table,
        0,
        "code",
        1,
        photo_url=prepared_no_photo,
    )

    assert table.widget.text == "Нет фото"
