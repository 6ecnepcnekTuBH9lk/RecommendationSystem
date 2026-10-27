import dataclasses

import pytest

from Application.mindbox import MindboxConfig, MindboxConfigError
from Application.mindbox.config import EXPORT_ENV_VARS


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("MINDBOX_API_URL", "MINDBOX_ENDPOINT_ID", "MINDBOX_SECRET_KEY", *EXPORT_ENV_VARS.values()):
        monkeypatch.delenv(name, raising=False)


def test_missing_secret_has_clear_error(config, monkeypatch, clean_env):
    monkeypatch.setenv("MINDBOX_API_URL", config.api_url)
    monkeypatch.setenv("MINDBOX_ENDPOINT_ID", config.endpoint_id)
    for name, env in EXPORT_ENV_VARS.items():
        monkeypatch.setenv(env, config.operations[name])
    with pytest.raises(MindboxConfigError, match="MINDBOX_SECRET_KEY"):
        MindboxConfig.from_env(None)


def test_dotenv_environment_precedence_and_no_mutation(tmp_path, config, monkeypatch, clean_env):
    import os
    env_file = tmp_path / ".env"
    values = {"MINDBOX_API_URL": config.api_url, "MINDBOX_ENDPOINT_ID": config.endpoint_id,
              "MINDBOX_SECRET_KEY": config.secret_key,
              **{env: config.operations[name] for name, env in EXPORT_ENV_VARS.items()}}
    env_file.write_text("\n".join(f'{key}="{value}"' for key, value in values.items()), encoding="utf-8-sig")
    monkeypatch.setenv("MINDBOX_OPERATION_ACTIONS", "OverriddenOperation")
    loaded = MindboxConfig.from_env(env_file)
    assert loaded.secret_key == config.secret_key
    assert loaded.operations["actions"] == "OverriddenOperation"
    assert loaded.operations["customer_merges"] == "CustomerMergesAPI"
    assert "MINDBOX_SECRET_KEY" not in os.environ
    assert config.secret_key not in repr(loaded)
    monkeypatch.setenv("MINDBOX_SECRET_KEY", "")
    with pytest.raises(MindboxConfigError, match="MINDBOX_SECRET_KEY"):
        MindboxConfig.from_env(env_file)


@pytest.mark.parametrize("url", ["http://api.example.test", "https://user:password@api.example.test",
                                     "https://api.example.test/path", "https://api.example.test?key=x",
                                     "https://[", "https://api.example.test:bad"])
def test_invalid_base_url_is_not_echoed(config, url):
    with pytest.raises(MindboxConfigError) as error:
        dataclasses.replace(config, api_url=url)
    assert url not in str(error.value)


def test_invalid_secret_is_not_echoed(config):
    with pytest.raises(MindboxConfigError) as error:
        dataclasses.replace(config, secret_key=config.secret_key + "\r\n")
    assert config.secret_key not in str(error.value)


def test_missing_env_file_reports_required_variables(tmp_path, clean_env):
    with pytest.raises(MindboxConfigError, match="MINDBOX_ENDPOINT_ID"):
        MindboxConfig.from_env(tmp_path / "absent.env")
