import pytest
from pydantic import SecretStr

from harness.settings import Settings


def test_allowed_chat_ids_are_parsed_and_deduplicated() -> None:
    settings = Settings(
        telegram_allowed_chat_ids="123, -456,123",
        _env_file=None,
    )
    assert settings.allowed_chat_ids == (123, -456)


def test_modal_proxy_credentials_are_required_together() -> None:
    with pytest.raises(RuntimeError, match="MODAL_TOKEN"):
        Settings(_env_file=None).require_modal_proxy_credentials()

    settings = Settings(
        modal_token_id=SecretStr("wk-fixture"),
        modal_token_secret=SecretStr("ws-fixture"),
        _env_file=None,
    )
    assert settings.require_modal_proxy_credentials() == ("wk-fixture", "ws-fixture")


def test_vast_endpoint_and_optional_api_key_stay_in_runtime_settings() -> None:
    with pytest.raises(RuntimeError, match="VAST_ENDPOINT_URL"):
        Settings(_env_file=None).require_vast_endpoint_url()

    settings = Settings(
        vast_endpoint_url="http://127.0.0.1:18000",
        vast_vllm_api_key=SecretStr("fixture-key"),
        _env_file=None,
    )
    assert settings.require_vast_endpoint_url() == "http://127.0.0.1:18000"
    assert settings.vast_vllm_api_key_value() == "fixture-key"
    assert (
        Settings(vast_vllm_api_key=SecretStr(""), _env_file=None).vast_vllm_api_key_value() is None
    )
