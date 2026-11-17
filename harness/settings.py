from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = "postgresql+asyncpg://research_harness:CHANGE_ME@localhost/research_harness"
    artifact_root: Path = Path("./artifacts")
    scheduler_poll_seconds: float = Field(default=2, gt=0, le=300)
    scheduler_abandoned_after_seconds: int = Field(default=3_600, ge=60)

    modal_app_name: str = Field(default="research-harness-sandboxes", min_length=1)
    modal_token_id: SecretStr | None = None
    modal_token_secret: SecretStr | None = None
    modal_qwen36_endpoint_url: str | None = None
    modal_gemma4_endpoint_url: str | None = None

    modal_gemma4_vision_endpoint_url: str | None = None
    modal_gpt_oss_judge_endpoint_url: str | None = None

    vast_endpoint_url: str | None = None
    vast_vllm_api_key: SecretStr | None = None

    telegram_bot_token: SecretStr | None = None
    telegram_allowed_chat_ids: str = ""
    telegram_poll_timeout_seconds: int = Field(default=30, ge=1, le=50)
    private_trace_base_url: str | None = None

    enable_subscription_baselines: bool = False

    @property
    def allowed_chat_ids(self) -> tuple[int, ...]:
        if not self.telegram_allowed_chat_ids.strip():
            return ()
        values: list[int] = []
        for item in self.telegram_allowed_chat_ids.split(","):
            values.append(int(item.strip()))
        return tuple(dict.fromkeys(values))

    def require_telegram_token(self) -> str:
        if self.telegram_bot_token is None:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
        return self.telegram_bot_token.get_secret_value()

    def require_modal_proxy_credentials(self) -> tuple[str, str]:
        if self.modal_token_id is None or self.modal_token_secret is None:
            raise RuntimeError("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are required")
        return (
            self.modal_token_id.get_secret_value(),
            self.modal_token_secret.get_secret_value(),
        )

    def require_vast_endpoint_url(self) -> str:
        if self.vast_endpoint_url is None:
            raise RuntimeError("VAST_ENDPOINT_URL is required")
        return self.vast_endpoint_url

    def vast_vllm_api_key_value(self) -> str | None:
        if self.vast_vllm_api_key is None:
            return None
        return self.vast_vllm_api_key.get_secret_value() or None
