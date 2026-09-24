import io
import json
from types import SimpleNamespace

import pytest
import requests
from urllib3.response import HTTPResponse

from Application.mindbox import MindboxClient, MindboxConfig
from Application.mindbox import client as client_module


class MemoryAdapter(requests.adapters.BaseAdapter):
    """Ответы проходят через настоящие requests.Response и urllib3 gzip decoder."""

    def __init__(self):
        self.responses = []
        self.calls = []

    def send(self, request, **kwargs):
        self.calls.append((request, kwargs))
        assert self.responses, "Неожиданный запрос: сценарий исчерпан"
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        response.request = request
        return response

    def close(self):
        pass


@pytest.fixture(autouse=True)
def forbid_live_http(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Unit tests не должны выполнять реальные HTTP-запросы")
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fail)


@pytest.fixture
def config():
    return MindboxConfig(
        api_url="https://api.example.test",
        endpoint_id="synthetic-endpoint",
        secret_key="synthetic-test-key-not-a-real-secret",
        operations={"actions": "CustomerActionsAPI", "orders": "OrdersAPI",
                    "customers": "CustomersAPI", "customer_merges": "CustomerMergesAPI"},
    )


@pytest.fixture
def response():
    def make(data=None, *, body=None, status=200, headers=None):
        result = requests.Response()
        result.status_code = status
        result.headers.update(headers or {})
        if body is None:
            body = json.dumps(data).encode("utf-8")
        result.raw = HTTPResponse(
            body=io.BytesIO(body), headers=headers or {}, preload_content=False,
            decode_content=False, enforce_content_length=True,
        )
        return result
    return make


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        state.sleeps.append(seconds)
        state.now += seconds

    monkeypatch.setattr(client_module, "time", SimpleNamespace(
        monotonic=lambda: state.now, sleep=sleep,
    ))
    return state


@pytest.fixture
def transport(config, clock):
    with MindboxClient(config) as client:
        api, files = MemoryAdapter(), MemoryAdapter()
        client._api.mount("https://", api)
        client._files.mount("https://", files)
        yield SimpleNamespace(client=client, api=api, files=files)
