import json
import traceback

import pytest
import requests

from Application.mindbox import (
    MindboxApiError, MindboxExportCancelledError, MindboxExportTimeoutError, RawExportStorage,
)


URLS = ["https://files.example.test/one", "https://files.example.test/two"]


def test_start_export_sends_contract_and_preserves_payload(transport, response, config):
    transport.api.responses.append(response({"status": "Success", "exportId": "123", "isDuplicate": True}))
    payload = {"sinceDateTimeUtc": "2026-01-01 00:00", "nested": {"custom": [1, 2]}}
    assert transport.client.start_export("CustomerActionsAPI", payload) == "123"
    request, options = transport.api.calls[0]
    assert request.method == "POST"
    assert request.url == "https://api.example.test/v3/operations/sync?endpointId=synthetic-endpoint&operation=CustomerActionsAPI"
    assert json.loads(request.body) == payload
    assert payload["nested"] == {"custom": [1, 2]}
    assert request.headers["Authorization"] == "SecretKey " + config.secret_key
    assert request.headers["Accept"] == "application/json"
    assert request.headers["Content-Type"] == "application/json"
    assert options["timeout"] == (10.0, 30.0)


def test_not_ready_then_ready_sends_only_export_id(transport, response, clock):
    transport.api.responses.extend([
        response({"status": "Success", "exportResult": {"processingStatus": "NotReady"}}),
        response({"status": "Success", "exportResult": {"processingStatus": "Ready", "urls": URLS}}),
    ])
    assert transport.client.wait_for_export("OrdersAPI", "123", poll_interval=2) == URLS
    assert clock.sleeps == [2]
    assert all(json.loads(call[0].body) == {"exportId": "123"} for call in transport.api.calls)


def test_cancelled_includes_reason(transport, response):
    transport.api.responses.append(response({"status": "Success", "exportResult": {
        "processingStatus": "Cancelled", "cancellationReason": "synthetic cancellation",
    }}))
    with pytest.raises(MindboxExportCancelledError, match="synthetic cancellation"):
        transport.client.wait_for_export("OrdersAPI", "123")
    assert len(transport.api.calls) == 1


@pytest.mark.parametrize("status", ["ValidationError", "ProtocolError", "InternalServerError"])
def test_operation_error_stops_polling(transport, response, status):
    transport.api.responses.append(response({"status": status, "errorMessage": "synthetic diagnostic"}))
    with pytest.raises(MindboxApiError, match="synthetic diagnostic"):
        transport.client.wait_for_export("OrdersAPI", "123")
    assert len(transport.api.calls) == 1


def test_polling_timeout_clamps_sleep(transport, response, clock):
    transport.api.responses.append(response({"status": "Success", "exportResult": {"processingStatus": "NotReady"}}))
    with pytest.raises(MindboxExportTimeoutError):
        transport.client.wait_for_export("OrdersAPI", "123", timeout=1, poll_interval=5)
    assert clock.now == 1
    assert len(transport.api.calls) == 1
    assert transport.api.calls[0][1]["timeout"] == (0.5, 0.5)


def test_polling_deadline_includes_retry_backoff(transport, response, clock):
    transport.api.responses.append(response(status=429, headers={"Retry-After": "20"}))
    with pytest.raises(MindboxExportTimeoutError):
        transport.client.wait_for_export("OrdersAPI", "123", timeout=2)
    assert clock.sleeps == [2]
    assert len(transport.api.calls) == 1


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_http_retry(transport, response, clock, code):
    transport.api.responses.extend([response(status=code), response(status=code),
                                   response({"status": "Success", "exportId": 123})])
    assert transport.client.start_export("OrdersAPI") == "123"
    assert clock.sleeps == [1, 2]
    assert all(item[1]["stream"] for item in transport.api.calls)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 302])
def test_permanent_http_no_retry_and_no_redirect(transport, response, clock, code):
    transport.api.responses.append(response(status=code, headers={"Location": URLS[0]}))
    with pytest.raises(MindboxApiError, match=f"HTTP {code}"):
        transport.client.start_export("OrdersAPI")
    assert len(transport.api.calls) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("error_type", [requests.Timeout, requests.ConnectionError,
    requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError])
def test_transient_network_retry(transport, response, error_type):
    transport.api.responses.extend([error_type("synthetic"), response({"status": "Success", "exportId": "123"})])
    assert transport.client.start_export("OrdersAPI") == "123"
    assert len(transport.api.calls) == 2


def test_retry_exhaustion(transport):
    transport.api.responses.extend([requests.Timeout("synthetic") for _ in range(3)])
    with pytest.raises(MindboxApiError, match="Timeout.*3"):
        transport.client.start_export("OrdersAPI")
    assert len(transport.api.calls) == 3


def test_retry_after_is_respected(transport, response, clock):
    transport.api.responses.extend([response(status=429, headers={"Retry-After": "7"}),
                                   response({"status": "Success", "exportId": "123"})])
    assert transport.client.start_export("OrdersAPI") == "123"
    assert clock.sleeps == [7]


def test_late_ready_is_not_accepted(transport, response, clock):
    item = response({"status": "Success", "exportResult": {"processingStatus": "Ready", "urls": URLS}})
    original = item.iter_content

    def late(chunk_size):
        clock.now = 3
        yield from original(chunk_size)

    item.iter_content = late
    transport.api.responses.append(item)
    with pytest.raises(MindboxExportTimeoutError):
        transport.client.wait_for_export("OrdersAPI", "123", timeout=2)
    assert len(transport.api.calls) == 1


def test_netrc_cannot_override_api_or_authorize_download(transport, response, tmp_path, monkeypatch, config):
    def fail(*args):
        pytest.fail("Транспорт не должен искать credentials в .netrc")

    monkeypatch.setattr(requests.sessions, "get_netrc_auth", fail)
    transport.api.responses.append(response({"status": "Success", "exportId": "123"}))
    transport.files.responses.append(response(body=b'{"synthetic": true}'))
    transport.client.start_export("OrdersAPI")
    transport.client.download_export("orders", URLS[:1], storage=RawExportStorage(tmp_path))
    assert transport.api.calls[0][0].headers["Authorization"] == "SecretKey " + config.secret_key
    assert "Authorization" not in transport.files.calls[0][0].headers


def test_tls_failure_is_not_retried(transport, config):
    transport.api.responses.append(requests.exceptions.SSLError(config.secret_key))
    with pytest.raises(MindboxApiError, match="TLS") as error:
        transport.client.start_export("OrdersAPI")
    assert config.secret_key not in "".join(traceback.format_exception(error.value))
    assert len(transport.api.calls) == 1


@pytest.mark.parametrize("payload", [{"exportId": "123"}, {"value": float("nan")}, {"value": object()}])
def test_invalid_start_payload_never_sends_request(transport, payload):
    with pytest.raises(MindboxApiError):
        transport.client.start_export("OrdersAPI", payload)
    assert not transport.api.calls


def test_invalid_json_has_safe_error(transport, response, config):
    transport.api.responses.append(response(body=config.secret_key.encode()))
    with pytest.raises(MindboxApiError, match="JSON") as error:
        transport.client.start_export("OrdersAPI")
    assert config.secret_key not in "".join(traceback.format_exception(error.value))
    assert len(transport.api.calls) == 1


@pytest.mark.parametrize("data", [None, [], {}, {"status": "Success"},
    {"status": "Success", "exportId": None}, {"status": "Success", "exportId": True},
    {"status": "Success", "exportId": []}, {"status": "Success", "exportId": ""}])
def test_invalid_start_response(transport, response, data):
    transport.api.responses.append(response(data))
    with pytest.raises(MindboxApiError):
        transport.client.start_export("OrdersAPI")


@pytest.mark.parametrize("result", [None, {}, {"processingStatus": "Unexpected"},
    {"processingStatus": "Ready"}, {"processingStatus": "Ready", "urls": []},
    {"processingStatus": "Ready", "urls": "https://files.example.test/one"},
    {"processingStatus": "Ready", "urls": ["http://files.example.test/one"]},
    {"processingStatus": "Ready", "urls": [None]}])
def test_invalid_polling_response(transport, response, result):
    transport.api.responses.append(response({"status": "Success", "exportResult": result}))
    with pytest.raises(MindboxApiError):
        transport.client.wait_for_export("OrdersAPI", "123")
    assert len(transport.api.calls) == 1


@pytest.mark.parametrize("kind", ["operation", "cancelled", "network"])
def test_secret_never_in_exception_repr_traceback_or_logs(transport, response, config, caplog, kind):
    secret = config.secret_key
    if kind == "operation":
        transport.api.responses.append(response({"status": "ValidationError", "validationMessages": [{"message": secret}]}))
    elif kind == "cancelled":
        transport.api.responses.append(response({"status": "Success", "exportResult": {
            "processingStatus": "Cancelled", "cancellationReason": secret,
        }}))
    else:
        transport.api.responses.extend([requests.ConnectionError(secret) for _ in range(3)])
    with pytest.raises(MindboxApiError) as error:
        transport.client.wait_for_export("OrdersAPI", "123")
    assert secret not in str(error.value)
    assert secret not in repr(error.value)
    assert secret not in "".join(traceback.format_exception(error.value))
    assert secret not in repr(config)
    assert secret not in repr(transport.client)
    assert secret not in caplog.text


@pytest.mark.parametrize("name", ["actions", "orders", "customers", "customer_merges"])
def test_full_flow_for_every_operation(tmp_path, transport, response, config, name):
    transport.api.responses.extend([
        response({"status": "Success", "exportId": "123"}),
        response({"status": "Success", "exportResult": {"processingStatus": "Ready", "urls": URLS}}),
    ])
    raw = b'{ "synthetic": [1, 2] }\r\n'
    transport.files.responses.extend([response(body=raw), response(body=raw)])
    paths = transport.client.export(name, storage=RawExportStorage(tmp_path))
    assert [p.name for p in paths] == [f"{name}_part_001.json", f"{name}_part_002.json"]
    assert all(p.read_bytes() == raw for p in paths)
    assert config.operations[name] in transport.api.calls[0][0].url
    assert all("Authorization" not in call[0].headers for call in transport.files.calls)
    assert transport.client._files.auth is not None


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_before_export(transport, value):
    with pytest.raises(ValueError):
        transport.client.export("actions", timeout=value)
    assert not transport.api.calls
