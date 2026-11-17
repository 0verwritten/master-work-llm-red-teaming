from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from harness.agents.native.modal_provider import ModalOpenAIClient, ModalServingProfile
from harness.experiments.bilingual_manifest import StudyCell
from harness.settings import Settings


@dataclass(frozen=True)
class RoleEndpoint:
    role: str
    profile: ModalServingProfile
    endpoint_url: str


@dataclass(frozen=True)
class PreflightResult:
    ready: bool
    missing: tuple[str, ...]
    present: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "missing": list(self.missing),
            "present": list(self.present),
        }


def preflight(
    settings: Settings | None = None,
    *,
    required_roles: tuple[str, ...] = ("target", "judge"),
) -> PreflightResult:

    settings = settings or Settings()
    missing: list[str] = []
    present: list[str] = []

    (present if settings.database_url else missing).append("DATABASE_URL")

    if settings.modal_token_id and settings.modal_token_secret:
        present.append("modal proxy token pair")
    else:
        missing.append("MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (proxy auth: wk-… / ws-…)")

    url_by_role = {
        "target": settings.modal_qwen36_endpoint_url,
        "judge": settings.modal_gpt_oss_judge_endpoint_url,
        "attacker": settings.modal_qwen36_endpoint_url,
    }
    for role in required_roles:
        name = {
            "target": "MODAL_QWEN36_ENDPOINT_URL",
            "judge": "MODAL_GPT_OSS_JUDGE_ENDPOINT_URL",
            "attacker": "MODAL_QWEN36_ENDPOINT_URL",
        }[role]
        (present if url_by_role[role] else missing).append(f"{name} ({role})")

    return PreflightResult(
        ready=not missing, missing=tuple(dict.fromkeys(missing)), present=tuple(present)
    )


@dataclass
class ModalClientFactory:
    endpoints: dict[str, RoleEndpoint]
    proxy_token_id: str
    proxy_token_secret: str
    http_client: httpx.AsyncClient

    def __call__(self, *, role: str, cell: StudyCell) -> ModalOpenAIClient:
        del cell
        endpoint = self.endpoints.get(role) or self.endpoints["target"]
        return ModalOpenAIClient(
            profile=endpoint.profile,
            endpoint_url=endpoint.endpoint_url,
            proxy_token_id=self.proxy_token_id,
            proxy_token_secret=self.proxy_token_secret,
            http_client=self.http_client,
        )
