from datetime import datetime, timedelta

import pytest

from scripts import mindbox_export_smoke as smoke


def test_actions_default_payload_is_recent_hour():
    args = smoke.build_parser().parse_args(["actions"])
    payload = smoke.build_payload(args)
    since = datetime.strptime(payload["sinceDateTimeUtc"], "%Y-%m-%d %H:%M")
    until = datetime.strptime(payload["tillDateTimeUtc"], "%Y-%m-%d %H:%M")
    assert until - since == timedelta(hours=1)


def test_explicit_actions_period():
    args = smoke.build_parser().parse_args(["actions", "--since", "2026-01-01 00:00", "--until", "2026-01-01 01:00"])
    assert smoke.build_payload(args) == {"sinceDateTimeUtc": "2026-01-01 00:00", "tillDateTimeUtc": "2026-01-01 01:00"}


@pytest.mark.parametrize("name", ["orders", "customers", "customer_merges"])
def test_no_implicit_period_for_other_operations(name):
    assert smoke.build_payload(smoke.build_parser().parse_args([name])) == {}


@pytest.mark.parametrize("argv", [
    ["actions", "--since", "2026-01-01 00:00"],
    ["actions", "--since", "invalid", "--until", "invalid"],
    ["actions", "--since", "2026-01-02 00:00", "--until", "2026-01-01 00:00"],
    ["orders", "--since", "2026-01-01 00:00", "--until", "2026-01-01 01:00"],
])
def test_invalid_period_rejected(argv):
    with pytest.raises(ValueError):
        smoke.build_payload(smoke.build_parser().parse_args(argv))


def test_arbitrary_payload_file(tmp_path):
    payload_file = tmp_path / "synthetic.json"
    payload_file.write_text('{"custom": {"synthetic": [1, 2]}}', encoding="utf-8")
    args = smoke.build_parser().parse_args(["customers", "--payload-file", str(payload_file)])
    assert smoke.build_payload(args) == {"custom": {"synthetic": [1, 2]}}


def test_smoke_flow_without_live_api(tmp_path, monkeypatch, transport, response, config, capsys):
    from Application import mindbox
    from Application.mindbox import storage as storage_module

    transport.api.responses.extend([
        response({"status": "Success", "exportId": "123"}),
        response({"status": "Success", "exportResult": {"processingStatus": "Ready", "urls": ["https://files.example.test/one"]}}),
    ])
    transport.files.responses.append(response(body=b'{"synthetic": true}'))
    client = transport.client
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *args: config)
    monkeypatch.setattr(mindbox, "MindboxClient", lambda *args: client)
    monkeypatch.setattr("Application.mindbox.client.RawExportStorage", lambda: storage_module.RawExportStorage(tmp_path))
    assert smoke.main(["actions"]) == 0
    output = capsys.readouterr().out
    assert "Operation: CustomerActionsAPI" in output
    assert "Export started: 123" in output
    assert "Parts: 1" in output
    assert "actions_part_001.json" in output
    assert config.secret_key not in output


def test_smoke_config_error_has_nonzero_exit_and_no_secret(monkeypatch, config, capsys):
    from Application import mindbox

    def fail(*args):
        raise mindbox.MindboxConfigError("Не заданы обязательные переменные: MINDBOX_SECRET_KEY")

    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", fail)
    assert smoke.main(["actions"]) == 1
    output = capsys.readouterr()
    assert "MINDBOX_SECRET_KEY" in output.err
    assert config.secret_key not in output.err
    assert output.out == ""
