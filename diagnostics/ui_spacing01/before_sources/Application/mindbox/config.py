"""Чтение конфигурации без изменения environment и действий при импорте."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from .exceptions import MindboxConfigError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_ENV_VARS = {
    "actions": "MINDBOX_OPERATION_ACTIONS",
    "orders": "MINDBOX_OPERATION_ORDERS",
    "customers": "MINDBOX_OPERATION_CUSTOMERS",
    "customer_merges": "MINDBOX_OPERATION_MERGES",
}


@dataclass(frozen=True)
class MindboxConfig:
    api_url: str
    endpoint_id: str
    secret_key: str = field(repr=False)
    operations: dict[str, str]

    def __post_init__(self) -> None:
        required = {
            "MINDBOX_API_URL": self.api_url,
            "MINDBOX_ENDPOINT_ID": self.endpoint_id,
            "MINDBOX_SECRET_KEY": self.secret_key,
            **{env: self.operations.get(name) for name, env in EXPORT_ENV_VARS.items()},
        }
        missing = [name for name, value in required.items()
                   if not isinstance(value, str) or not value.strip()]
        if missing:
            raise MindboxConfigError("Не заданы обязательные переменные: " + ", ".join(missing))
        try:
            url = urlsplit(self.api_url)
            valid_url = (
                url.scheme == "https" and bool(url.hostname) and not url.username
                and not url.password and not url.query and not url.fragment
                and url.path in ("", "/")
            )
            url.port  # Проверка корректности порта без вывода URL в ошибку.
        except ValueError:
            valid_url = False
        if not valid_url:
            raise MindboxConfigError("MINDBOX_API_URL должен быть базовым HTTPS URL без credentials и пути")
        if any(char.isspace() for char in self.secret_key):
            raise MindboxConfigError("MINDBOX_SECRET_KEY содержит пробельные символы")
        try:
            self.secret_key.encode("ascii")
        except UnicodeEncodeError:
            raise MindboxConfigError("MINDBOX_SECRET_KEY должен содержать ASCII-символы") from None

    @classmethod
    def from_env(cls, env_file: str | Path | None = PROJECT_ROOT / ".env") -> "MindboxConfig":
        """Environment имеет приоритет; None отключает чтение .env."""
        values: dict[str, str | None] = {}
        if env_file is not None:
            try:
                with Path(env_file).open(encoding="utf-8-sig") as stream:
                    values.update(dotenv_values(stream=stream, interpolate=False))
            except FileNotFoundError:
                pass  # Для конфигурации только через environment файл необязателен.
            except (OSError, UnicodeError):
                raise MindboxConfigError("Не удалось прочитать локальный файл .env") from None
        values.update(os.environ)
        return cls(
            api_url=values.get("MINDBOX_API_URL") or "",
            endpoint_id=values.get("MINDBOX_ENDPOINT_ID") or "",
            secret_key=values.get("MINDBOX_SECRET_KEY") or "",
            operations={name: values.get(env) or "" for name, env in EXPORT_ENV_VARS.items()},
        )
