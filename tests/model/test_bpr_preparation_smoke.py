from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import socket

import pytest

from Application.mindbox import adapters, raw_reader
from Application.mindbox.records import ActionRecord, OrderLineRecord, ProductKey
from scripts.bpr_preparation_smoke import main


@pytest.fixture
def local_data(tmp_path, monkeypatch):
    path = tmp_path / "catalog.csv"
    path.write_text("КодНоменклатуры\n001234\n", encoding="utf-8-sig")
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    action = ActionRecord(
        action_id="SECRET_ACTION", action_system_name="ProsmotrProdukta",
        event_datetime_utc=timestamp, creation_datetime_utc=timestamp,
        source_customer_id="SECRET_SOURCE", customer_id="secret@example.test",
        products=(ProductKey("offline1C", "001234_SECRET_PRODUCT"),),
    )
    line = OrderLineRecord(
        order_id="SECRET_ORDER", order_datetime_utc=timestamp, channel_external_id="SECRET_CHANNEL",
        channel_name="SECRET_CHANNEL_NAME", source_customer_id="SECRET_SOURCE", customer_id="+79999999999",
        line_id="SECRET_LINE", line_number=1, product=ProductKey("kanzlerKz", "001234_SECRET_PRODUCT"),
        product_name="SECRET_NAME", quantity=Decimal("1.5"), base_price_per_item=Decimal("10"),
        price_of_line=Decimal("15"), line_status="CP",
    )
    data = {"customer_merges": [], "actions": [action, replace(action,
             action_system_name="DobavlenieProduktaVSpisokVOperaciiDobavlenie")], "orders": [(line,)]}
    monkeypatch.setattr(raw_reader, "iter_export", lambda name, **kwargs: iter(data[name]))
    monkeypatch.setattr(adapters, "adapt_action", lambda record, resolver: record)
    monkeypatch.setattr(adapters, "adapt_order", lambda records, resolver: records)

    def forbidden(*args, **kwargs):
        pytest.fail("API requests are forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    return path, data


@pytest.mark.parametrize("problem", [None, "malformed", "unresolved", "unsupported"])
@pytest.mark.parametrize("diagnose", [True, False])
def test_safe_smoke_counts_exit_and_no_output_files(local_data, capsys, problem, diagnose):
    path, data = local_data
    action = data["actions"][0]
    if problem:
        products = () if problem == "malformed" else (ProductKey(
            "SECRET_NAMESPACE" if problem == "unsupported" else "offline1C", "999999_SECRET"),)
        data["actions"].insert(0, replace(action, products=products))
    before = {file: file.read_bytes() for file in path.parent.iterdir()}
    args = ["--catalog", str(path), "--raw-root", str(path.parent)]
    assert main(args + (["--diagnose"] if diagnose else [])) == int(problem is not None)
    output = capsys.readouterr()
    for secret in ("SECRET", "001234", "999999", "secret@example.test", "+79999999999"):
        assert secret not in output.out + output.err
    if problem is None or diagnose:
        assert "Prepared input validation: OK" in output.out
        for label in ("Resolved interactions: 3", "VIEW: 1", "FAVORITE: 1", "PURCHASE: 1", "TOTAL: 3",
                      "Unique users: 2", "Unique items: 1", "Eval events: 0",
                      "Train events before aggregation: 3", "Train user-item pairs after aggregation: 2",
                      "Total train weight: 17.1"):
            assert label in output.out
        assert f"Upstream malformed actions: {int(problem == 'malformed')}" in output.out
        assert f"Unresolved products: {int(problem in ('unresolved', 'unsupported'))}" in output.out
    else:
        assert output.out == ""
    assert before == {file: file.read_bytes() for file in path.parent.iterdir()}


def test_upstream_error_not_suppressed_or_leaked(local_data, monkeypatch, capsys):
    path, _ = local_data

    def broken(*args):
        raise adapters.AdapterError("SECRET 001234 secret@example.test")

    monkeypatch.setattr(adapters, "adapt_action", broken)
    assert main(["--catalog", str(path), "--diagnose"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "AdapterError" in output.err
    assert "SECRET" not in output.err


def test_full_timestamp_cli_explicit_opt_in(local_data, capsys):
    path, _ = local_data
    assert main(["--catalog", str(path), "--date-mode", "FULL_TIMESTAMP"]) == 0
    assert "Date mode: FULL_TIMESTAMP" in capsys.readouterr().out
