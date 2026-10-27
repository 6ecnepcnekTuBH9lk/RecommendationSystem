import json
import traceback

import pytest

from Application.mindbox import schema_profiler as profiler
from scripts import mindbox_schema_report as cli


def write_part(directory, export_name, items, number=1):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{export_name}_part_{number:03d}.json"
    path.write_text(json.dumps({profiler.EXPORT_ROOTS[export_name]: items}), encoding="utf-8")
    return path


def test_recursive_paths_optional_null_and_multiple_types(tmp_path):
    write_part(tmp_path, "orders", [
        {"customer": {"ids": {"mindboxId": 1}}, "optional": None, "mixed": "synthetic"},
        {"customer": {"ids": {"externalId": "synthetic-id"}}, "mixed": 2},
        {"mixed": False},
    ])
    report = profiler.profile_export("orders", input_dir=tmp_path)
    paths = report["paths"]
    assert report["file_count"] == 1
    assert report["total_objects"] == 3
    assert paths["orders[].customer.ids"]["present_records"] == 2
    assert paths["orders[].customer.ids"]["presence_percent"] == 66.67
    assert paths["orders[].optional"]["null_count"] == 1
    assert paths["orders[].optional"]["types"] == ["null"]
    assert paths["orders[].mixed"]["type_counts"] == {"boolean": 1, "integer": 1, "string": 1}
    assert "multiple_types" in paths["orders[].mixed"]["variation_reasons"]
    assert "different_object_fields" in paths["orders[].customer.ids"]["variation_reasons"]
    assert "orders[].optional" in report["variable_paths"]
    assert paths["orders"]["scope"] == "part_files"
    assert paths["orders"]["present_records"] == 1


def test_arrays_count_unique_export_objects_not_array_elements(tmp_path):
    write_part(tmp_path, "orders", [
        {"lines": [{"price": None}, {"price": None}, {"price": 1.5}, {}]},
        {"lines": []},
        {},
    ])
    report = profiler.profile_export("orders", input_dir=tmp_path)
    paths = report["paths"]
    price = paths["orders[].lines[].price"]
    assert price["occurrences"] == 3
    assert price["present_records"] == 1
    assert price["presence_percent"] == 33.33
    assert price["null_count"] == 2
    assert price["null_records"] == 1
    assert price["type_counts"] == {"null": 2, "number": 1}
    assert paths["orders[].lines"]["array"] == {
        "observed": True, "count": 2, "empty_count": 1, "min_length": 0, "max_length": 4,
    }
    assert paths["orders[].lines[]"]["occurrences"] == 4
    assert "different_object_fields" in paths["orders[].lines[]"]["variation_reasons"]
    assert all(stats["presence_percent"] <= 100 for stats in paths.values())


def test_nested_arrays_and_empty_objects(tmp_path):
    write_part(tmp_path, "customers", [{"nested": [[True, None], []], "empty": {}}])
    paths = profiler.profile_export("customers", input_dir=tmp_path)["paths"]
    assert paths["customers[].nested[]"]["array"]["empty_count"] == 1
    assert paths["customers[].nested[][]"]["type_counts"] == {"boolean": 1, "null": 1}
    assert paths["customers[].empty"]["object_fields"] == []


def test_part_files_are_processed_sequentially(tmp_path, monkeypatch):
    first = write_part(tmp_path, "customers", [{"first": 1}])
    second = write_part(tmp_path, "customers", [{"second": 2}, {"second": None}], 2)
    events = []
    original_load = profiler.json.load
    original_walk = profiler._Profiler.walk

    def load(stream, **kwargs):
        events.append(("load", stream.name))
        return original_load(stream, **kwargs)

    def walk(instance, value, path, record):
        if path == "customers[]":
            events.append(("record", record))
        return original_walk(instance, value, path, record)

    monkeypatch.setattr(profiler.json, "load", load)
    monkeypatch.setattr(profiler._Profiler, "walk", walk)
    report = profiler.profile_export("customers", input_dir=tmp_path)
    assert events == [("load", str(first)), ("record", 1), ("load", str(second)), ("record", 2), ("record", 3)]
    assert report["file_count"] == 2
    assert report["total_objects"] == 3
    assert report["paths"]["customers"]["array"]["min_length"] == 1
    assert report["paths"]["customers"]["array"]["max_length"] == 2
    assert report["paths"]["customers[].second"]["null_count"] == 1


def test_action_templates_and_related_entities(tmp_path):
    def action(name, **entities):
        return {"actionTemplate": {"ids": {"systemName": name}}, **entities}

    write_part(tmp_path, "actions", [
        action("Synthetic.A", product={"ids": {"externalId": "private"}}, customer={"ids": {"mindboxId": 12}}),
        action("Synthetic.A", products=[], customFields={"custom": "private"}, anotherEntity={}),
        action("Synthetic.B", order={}),
        {}, {"actionTemplate": {"ids": {"systemName": None}}},
        {"actionTemplate": {"ids": {"systemName": 123}}},
    ])
    report = profiler.profile_export("actions", input_dir=tmp_path)
    a, b = report["action_templates"]
    assert a["system_name"] == "Synthetic.A"
    assert a["count"] == 2
    assert a["percent"] == 33.33
    assert set(a["related_entities"]) == {"product", "products", "customer", "customFields", "anotherEntity"}
    assert a["related_entities"]["product"]["present_records"] == 1
    assert a["related_entities"]["product"]["presence_percent"] == 50.0
    assert b["count"] == 1
    assert report["actions_without_system_name"] == {"missing": 1, "null": 1, "invalid_type": 1}


def test_custom_fields_names_counts_and_identifier_namespaces(tmp_path):
    write_part(tmp_path, "orders", [{
        "ids": {"mindboxId": "private-order"}, "customFields": {"orderField": None},
        "lines": [{"product": {"ids": {"offline1C": "private-rf", "kanzlerKz": "private-kz"}},
                   "customFields": {"size": "private-size"}}, {"customFields": {"size": None}}],
    }])
    report = profiler.profile_export("orders", input_dir=tmp_path)
    assert report["identifier_namespaces"] == {
        "orders[].ids": ["mindboxId"], "orders[].lines[].product.ids": ["kanzlerKz", "offline1C"],
    }
    size = report["custom_fields"]["orders[].lines[].customFields"]["fields"]["size"]
    assert size == {"types": ["null", "string"], "occurrences": 2, "present_records": 1, "null_count": 1, "null_records": 1}
    assert report["sections"]["orders[].lines[].product"]["observed"]
    assert not report["sections"]["orders[].lines[].status"]["observed"]


@pytest.mark.parametrize("name", list(profiler.EXPORT_ROOTS))
def test_security_regression_no_values_in_json_markdown_stdout_or_logs(tmp_path, name, capsys, caplog):
    source, output = tmp_path / "source", tmp_path / "reports"
    secrets = ["SECRET_FIRST_NAME", "secret@example.test", "+79999999999", "SECRET_ID",
               "SECRET_PRODUCT", "SECRET_CUSTOM_VALUE", "SECRET_CARD", "SECRET_PROMO", "SECRET_ADDRESS",
               "SECRET_MERGE_METHOD"]
    record = {
        "firstName": secrets[0], "email": secrets[1], "mobilePhone": secrets[2],
        "ids": {"mindboxId": secrets[3]}, "product": {"ids": {"externalId": secrets[4]}},
        "customFields": {"privateField": secrets[5], "nested": {"private": [secrets[1], secrets[3]]}},
        "discountCards": [{"number": secrets[6]}], "appliedPromotions": [{"code": secrets[7]}],
        "address": secrets[8], "method": secrets[9], "resultingCustomer": {"ids": {"mindboxId": secrets[3]}},
        "mergedCustomers": [{"ids": {"mindboxId": secrets[3]}}],
        "actionTemplate": {"ids": {"systemName": "Synthetic.Allowed", "externalId": secrets[3]}},
    }
    raw = write_part(source, name, [record])
    original = raw.read_bytes()
    assert cli.main([name, "--input-dir", str(source), "--output-dir", str(output)]) == 0
    stdout, stderr = capsys.readouterr()
    combined = stdout + stderr + caplog.text + "".join(p.read_text(encoding="utf-8") for p in output.iterdir())
    for secret in secrets:
        assert secret not in combined
    assert ("Synthetic.Allowed" in combined) == (name == "actions")
    assert raw.read_bytes() == original
    assert len(list(source.iterdir())) == 1


def test_paths_escape_ambiguous_keys_and_markdown_markup(tmp_path):
    write_part(tmp_path, "actions", [{"a.b": None, "a": {"b": True}, "x[]": 1, "x": [],
        "customFields": {"a|b\n<script>": "PRIVATE"},
        "actionTemplate": {"ids": {"systemName": "<script>|`test`\nnext"}},
    }])
    report = profiler.profile_export("actions", input_dir=tmp_path)
    assert 'customerActions[]["a.b"]' in report["paths"]
    assert "customerActions[].a.b" in report["paths"]
    assert 'customerActions[]["x[]"]' in report["paths"]
    assert "customerActions[].x[]" not in report["paths"]
    markdown = profiler.render_markdown(report)
    assert "<script>" not in markdown
    assert "&#124;" in markdown
    assert "PRIVATE" not in markdown


def test_latest_timestamp_ignores_staging_and_compares_suffix_numerically(tmp_path):
    for directory in ("20260101_120000", "20260102_120000", "20260102_120000_999",
                      "20260102_120000_1000", ".staging-later", ".20260103_120000.reserve",
                      "20261399_120000", "not-a-timestamp"):
        write_part(tmp_path / "orders" / directory, "orders", [{}])
    selected = profiler.select_export_directory("orders", raw_root=tmp_path)
    assert selected.name == "20260102_120000_1000"
    assert profiler.profile_export("orders", raw_root=tmp_path)["total_objects"] == 1


def test_bad_latest_export_is_not_silently_replaced_with_old(tmp_path):
    write_part(tmp_path / "orders" / "20260101_000000", "orders", [{}])
    (tmp_path / "orders" / "20260102_000000").mkdir()
    with pytest.raises(profiler.SchemaProfileError, match="part"):
        profiler.profile_export("orders", raw_root=tmp_path)


def test_explicit_input_directory_overrides_latest(tmp_path):
    old = tmp_path / "orders" / "20260101_000000"
    write_part(old, "orders", [{}, {}])
    write_part(tmp_path / "orders" / "20260102_000000", "orders", [{}])
    report = profiler.profile_export("orders", raw_root=tmp_path, input_dir=old)
    assert report["total_objects"] == 2


@pytest.mark.parametrize("payload", [
    '{"customers": [{"email": "secret@example.test"}',
    '{"WRONG_SECRET_ROOT": []}', '{"customers": {}}', '{"customers": [null]}',
    '{"customers": [], "SECRET_EXTRA": 1}', '{"customers": [{"x": NaN}]}',
    '{"customers": [{"x": 1, "x": 2}]}',
])
def test_invalid_json_and_root_errors_do_not_leak_values(tmp_path, payload):
    (tmp_path / "customers_part_001.json").write_text(payload, encoding="utf-8")
    with pytest.raises(profiler.SchemaProfileError) as error:
        profiler.profile_export("customers", input_dir=tmp_path)
    text = "".join(traceback.format_exception(error.value))
    assert "secret@example.test" not in text
    assert "WRONG_SECRET_ROOT" not in text
    assert "SECRET_EXTRA" not in text


def test_empty_export_is_valid_and_missing_parts_are_error(tmp_path):
    with pytest.raises(profiler.SchemaProfileError):
        profiler.profile_export("customers", input_dir=tmp_path)
    write_part(tmp_path, "customers", [])
    report = profiler.profile_export("customers", input_dir=tmp_path)
    assert report["total_objects"] == 0
    assert report["paths"]["customers"]["array"]["empty_count"] == 1
    assert "customers[]" not in report["paths"]
    assert not report["sections"]["customers[].birthDate"]["observed"]
    assert report["custom_fields"]["customers[].customFields"] == {"observed": False, "fields": {}}
    assert "Объектов: 0" in profiler.render_markdown(report)


@pytest.mark.parametrize("extra", ["customers_part_003.json", "customers_part_001.json.tmp",
                                   "customers_part_001.json.download", "orders_part_001.json"])
def test_incomplete_or_unexpected_part_set_is_rejected(tmp_path, extra):
    write_part(tmp_path, "customers", [])
    (tmp_path / extra).write_text("{}", encoding="utf-8")
    with pytest.raises(profiler.SchemaProfileError):
        profiler.profile_export("customers", input_dir=tmp_path)


def test_all_mode(tmp_path, capsys):
    raw, output = tmp_path / "raw", tmp_path / "reports"
    for name in profiler.EXPORT_ROOTS:
        write_part(raw / name / "20260101_000000", name, [{}])
    assert cli.main(["all", "--input-dir", str(raw), "--output-dir", str(output)]) == 0
    assert len(list(output.iterdir())) == 8
    for name in profiler.EXPORT_ROOTS:
        report = json.loads((output / f"{name}_schema.json").read_text(encoding="utf-8"))
        assert report["total_objects"] == 1
        assert report["export_name"] == name
    assert "objects=1" in capsys.readouterr().out


def test_all_mode_failure_does_not_publish_partial_reports(tmp_path, capsys):
    raw, output = tmp_path / "raw", tmp_path / "reports"
    for name in profiler.EXPORT_ROOTS:
        write_part(raw / name / "20260101_000000", name, [{}])
    (raw / "orders" / "20260101_000000" / "orders_part_001.json").write_text("PRIVATE_BAD_JSON", encoding="utf-8")
    assert cli.main(["all", "--input-dir", str(raw), "--output-dir", str(output)]) == 1
    assert not output.exists()
    assert "PRIVATE_BAD_JSON" not in capsys.readouterr().err


def test_report_publication_uses_temporary_file_and_preserves_existing_on_error(tmp_path, monkeypatch):
    source, output = tmp_path / "raw", tmp_path / "reports"
    write_part(source, "orders", [{}])
    report = profiler.profile_export("orders", input_dir=source)
    paths = profiler.write_report(report, output)
    before = paths[0].read_bytes()

    def fail(source, target):
        assert source.suffix == ".tmp"
        assert json.loads(source.read_text(encoding="utf-8"))["export_name"] == "orders"
        raise OSError("private filename or data")

    monkeypatch.setattr(profiler.os, "replace", fail)
    with pytest.raises(profiler.SchemaProfileError) as error:
        profiler.write_report(report, output)
    assert "private filename or data" not in str(error.value)
    assert paths[0].read_bytes() == before
    assert not list(output.glob("*.tmp"))


def test_output_inside_raw_is_rejected(tmp_path, capsys):
    write_part(tmp_path, "orders", [{}])
    assert cli.main(["orders", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "reports")]) == 1
    assert not (tmp_path / "reports").exists()
    assert "отдельно" in capsys.readouterr().err


@pytest.mark.parametrize("name,fields", [
    ("customers", {"ids": {}, "birthDate": None, "balances": [], "subscriptions": [], "lastActivatedCard": {}}),
    ("customer_merges", {"id": "private", "dateTimeUtc": "private", "method": "private-enum",
                         "resultingCustomer": {"ids": {"mindboxId": 1}}, "mergedCustomers": [{"ids": {"externalId": 2}}]}),
    ("orders", {"firstAction": {}, "payments": [], "appliedPromotions": [], "bonusPointsInfoPerBalanceTypes": [],
                "lines": [{"status": {}, "product": {"ids": {}}, "appliedPromotions": []}]}),
])
def test_special_sections_include_actual_structure(tmp_path, name, fields):
    write_part(tmp_path, name, [fields])
    report = profiler.profile_export(name, input_dir=tmp_path)
    root = profiler.EXPORT_ROOTS[name] + "[]"
    for field in fields:
        if root + "." + field in report["sections"]:
            assert report["sections"][root + "." + field]["observed"]
    if name == "customer_merges":
        assert report["identifier_namespaces"][root + ".mergedCustomers[].ids"] == ["externalId"]
    assert "private-enum" not in profiler.render_markdown(report)
