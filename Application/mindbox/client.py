"""Универсальный синхронный транспорт Mindbox V3, независимый от UI."""

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import quote, urlsplit

import requests

from .config import MindboxConfig
from .exceptions import (
    MindboxApiError,
    MindboxDownloadError,
    MindboxError,
    MindboxExportCancelledError,
    MindboxExportTimeoutError,
)
from .storage import RawExportStorage


T = TypeVar("T")
CHUNK_SIZE = 64 * 1024


class _KeepHeadersAuth(requests.auth.AuthBase):
    """Сохраняет явные headers и отключает подстановку credentials из .netrc."""

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        return request


class MindboxClient:
    def __init__(
        self,
        config: MindboxConfig,
        *,
        request_timeout: tuple[float, float] = (10.0, 30.0),
        max_attempts: int = 3,
        backoff_base: float = 1.0,
        max_backoff: float = 60.0,
        download_timeout: float = 600.0,
    ) -> None:
        for value in (*request_timeout, backoff_base, max_backoff, download_timeout):
            self._positive(value)
        if len(request_timeout) != 2 or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("Нужны два HTTP timeout и max_attempts >= 1")
        self.config = config
        self.request_timeout = request_timeout
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self.download_timeout = download_timeout
        self._api = requests.Session()
        self._files = requests.Session()
        # Прокси и CA из environment работают как в requests, а .netrc не
        # подменяет SecretKey и не передаёт сторонние credentials файловому хосту.
        self._api.auth = _KeepHeadersAuth()
        self._files.auth = _KeepHeadersAuth()

    def __enter__(self) -> "MindboxClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self._api.close()
        self._files.close()

    def safe_text(self, value: Any) -> str:
        """Для внешней диагностики: редактируем секрет до ограничения длины."""
        text = str(value)
        secret = self.config.secret_key
        for variant in (secret, quote(secret, safe=""), json.dumps(secret)[1:-1]):
            text = text.replace(variant, "[REDACTED]")
        return text[:1000]

    def start_export(self, operation: str, payload: Mapping[str, Any] | None = None) -> str:
        """Передаёт параметры операции без добавления периодов или бизнес-полей."""
        body = dict(payload) if payload is not None else {}
        if "exportId" in body:
            raise MindboxApiError("start_export ожидает условия экспорта без exportId")
        data = self._operation(operation, body)
        export_id = data.get("exportId")
        if (not isinstance(export_id, (str, int)) or isinstance(export_id, bool)
                or not str(export_id).strip()):
            raise MindboxApiError("В ответе запуска отсутствует корректный exportId")
        return str(export_id)

    def get_export_status(
        self, operation: str, export_id: str, *, deadline: float | None = None,
    ) -> dict[str, Any]:
        """deadline — абсолютное значение time.monotonic(), используемое polling."""
        if not isinstance(export_id, str) or not export_id.strip():
            raise MindboxApiError("Для проверки статуса необходим exportId")
        data = self._operation(operation, {"exportId": export_id}, deadline=deadline)
        result = data.get("exportResult")
        if not isinstance(result, dict):
            raise MindboxApiError("В ответе отсутствует объект exportResult")
        status = result.get("processingStatus")
        if status == "Cancelled":
            reason = self.safe_text(result.get("cancellationReason") or "причина не указана")
            raise MindboxExportCancelledError(f"Экспорт отменён Mindbox: {reason}")
        if status not in ("NotReady", "Ready"):
            raise MindboxApiError("Неизвестный или отсутствующий exportResult.processingStatus")
        if status == "Ready":
            self._validate_urls(result.get("urls"))
        return result

    def wait_for_export(
        self, operation: str, export_id: str, *, poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> list[str]:
        self._positive(poll_interval)
        self._positive(timeout)
        deadline = time.monotonic() + timeout
        while True:
            self._remaining(deadline, MindboxExportTimeoutError)
            result = self.get_export_status(operation, export_id, deadline=deadline)
            self._remaining(deadline, MindboxExportTimeoutError)
            if result["processingStatus"] == "Ready":
                return list(result["urls"])
            self._sleep(poll_interval, deadline, MindboxExportTimeoutError)

    def download_export(
        self, export_name: str, urls: Sequence[str], *, storage: RawExportStorage | None = None,
    ) -> list[Path]:
        self._validate_urls(urls)
        return (storage or RawExportStorage()).save_export(export_name, urls, self._download_part)

    def export(
        self, export_name: str, payload: Mapping[str, Any] | None = None, *,
        storage: RawExportStorage | None = None, poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> list[Path]:
        """Полный цикл для одного из четырёх настроенных экспортов."""
        if export_name not in self.config.operations:
            raise MindboxApiError("Неизвестное имя экспорта в конфигурации")
        self._positive(poll_interval)
        self._positive(timeout)
        operation = self.config.operations[export_name]
        export_id = self.start_export(operation, payload)
        urls = self.wait_for_export(operation, export_id, poll_interval=poll_interval, timeout=timeout)
        return self.download_export(export_name, urls, storage=storage)

    def _operation(
        self, operation: str, payload: dict[str, Any], *, deadline: float | None = None,
    ) -> dict[str, Any]:
        if not isinstance(operation, str) or not operation.strip():
            raise MindboxApiError("Не задано системное имя операции")
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError):
            raise MindboxApiError("Payload операции должен быть корректным JSON-объектом") from None
        return self._request(
            self._api, "POST", self.config.api_url.rstrip("/") + "/v3/operations/sync",
            lambda response: self._read_operation(response, deadline), MindboxApiError,
            deadline=deadline, timeout_error=MindboxExportTimeoutError,
            params={"endpointId": self.config.endpoint_id, "operation": operation},
            headers={
                "Authorization": "SecretKey " + self.config.secret_key,
                "Accept": "application/json", "Content-Type": "application/json",
            },
            json=payload,
        )

    def _read_operation(
        self, response: requests.Response, deadline: float | None = None,
    ) -> dict[str, Any]:
        chunks = []
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            self._remaining(deadline, MindboxExportTimeoutError)
            chunks.append(chunk)
        try:
            data = json.loads(b"".join(chunks))
        except ValueError:
            raise MindboxApiError("Mindbox вернул некорректный JSON-ответ") from None
        if not isinstance(data, dict) or not isinstance(data.get("status"), str):
            raise MindboxApiError("В JSON-ответе Mindbox отсутствует строковое поле status")
        if data["status"] != "Success":
            details = {key: data[key] for key in (
                "status", "errorMessage", "validationMessages", "errorId", "message",
            ) if key in data}
            raise MindboxApiError("Ошибка операции Mindbox: " + self.safe_text(details))
        return data

    def _download_part(self, url: str, destination: Path) -> None:
        deadline = time.monotonic() + self.download_timeout

        def save(response: requests.Response) -> None:
            # iter_content снимает Content-Encoding; storage проверяет реальные
            # байты после этой обработки, а не заголовок и не расширение URL.
            with destination.open("wb") as target:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    self._remaining(deadline, MindboxDownloadError)
                    target.write(chunk)

        self._request(
            self._files, "GET", url, save, MindboxDownloadError,
            deadline=deadline, timeout_error=MindboxDownloadError,
            headers={"Accept": "application/json, application/octet-stream"},
        )

    def _request(
        self, session: requests.Session, method: str, url: str,
        consume: Callable[[requests.Response], T], error_type: type[MindboxError], *,
        deadline: float | None = None, timeout_error: type[MindboxError] = MindboxApiError,
        **kwargs: Any,
    ) -> T:
        reason = ""
        for attempt in range(self.max_attempts):
            remaining = self._remaining(deadline, timeout_error)
            timeout = self.request_timeout
            if remaining is not None:
                timeout = tuple(min(value, remaining / 2) for value in timeout)
            retry_after = 0.0
            try:
                # Redirect не должен перенести Authorization на другой адрес.
                with session.request(method, url, timeout=timeout, stream=True,
                                     allow_redirects=False, **kwargs) as response:
                    code = response.status_code
                    if code == 429 or 500 <= code < 600:
                        reason = f"HTTP {code}"
                        retry_after = self._retry_after(response.headers.get("Retry-After"))
                    elif not 200 <= code < 300:
                        raise error_type(f"Mindbox: HTTP {code}; запрос не повторяется")
                    else:
                        result = consume(response)
                        self._remaining(deadline, timeout_error)
                        return result
            except requests.exceptions.SSLError:
                raise error_type("Mindbox: ошибка проверки TLS; запрос не повторяется") from None
            except (requests.Timeout, requests.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ContentDecodingError) as exc:
                # requests исключения могут содержать PreparedRequest, ключ и URL.
                # Сохраняем категорию ошибки без небезопасной цепочки исключений.
                reason = type(exc).__name__
            except requests.RequestException as exc:
                raise error_type(f"Mindbox: ошибка HTTP-транспорта ({type(exc).__name__})") from None
            self._remaining(deadline, timeout_error)
            if attempt + 1 < self.max_attempts:
                delay = min(self.max_backoff, max(self.backoff_base * 2 ** attempt, retry_after))
                self._sleep(delay, deadline, timeout_error)
        raise error_type(f"Mindbox: {reason}; исчерпаны попытки ({self.max_attempts})") from None

    @staticmethod
    def _validate_urls(urls: Any) -> None:
        if not isinstance(urls, (list, tuple)) or not urls:
            raise MindboxApiError("Ready должен содержать непустой список exportResult.urls")
        for value in urls:
            try:
                url = urlsplit(value) if isinstance(value, str) else None
                valid = (url is not None and url.scheme == "https" and url.hostname
                         and not url.username and not url.password and not url.fragment)
                if url is not None:
                    url.port
            except ValueError:
                valid = False
            if not valid:
                raise MindboxApiError("Экспорт содержит некорректный HTTPS URL части")

    @staticmethod
    def _positive(value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Интервалы и timeout должны быть конечными положительными числами")

    @staticmethod
    def _remaining(deadline: float | None, error_type: type[MindboxError]) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise error_type("Mindbox: истекло максимальное время ожидания")
        return remaining

    def _sleep(self, delay: float, deadline: float | None, error_type: type[MindboxError]) -> None:
        remaining = self._remaining(deadline, error_type)
        time.sleep(delay if remaining is None else min(delay, remaining))
        self._remaining(deadline, error_type)

    def _retry_after(self, value: str | None) -> float:
        if not value:
            return 0.0
        try:
            seconds = float(value)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                seconds = (date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return 0.0
        return max(0.0, min(seconds, self.max_backoff)) if math.isfinite(seconds) else 0.0
