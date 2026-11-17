from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from harness.agents.native.modal_provider import (
    GEMMA4_31B_CANDIDATE,
    GPT_OSS_20B_JUDGE_CANDIDATE,
    MISTRAL_SMALL_32_24B_CANDIDATE,
    QWEN36_27B_FP8_CANDIDATE,
    ModalServingProfile,
)
from harness.experiments.adversarial_matrix import FrozenModelProfile


def freeze_modal_profile(
    *,
    key: str,
    profile: ModalServingProfile,
) -> FrozenModelProfile:
    return FrozenModelProfile(
        key=key,
        provider="modal",
        model=None,
        endpoint_profile=profile.profile_id,
        revision=profile.checkpoint_revision,
    )


@dataclass(frozen=True)
class FrozenModelRoster:
    version: str
    matrix_profiles: tuple[FrozenModelProfile, ...]
    judge_profile: FrozenModelProfile

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("roster version is required")
        if len(self.matrix_profiles) != 3:
            raise ValueError("the primary matrix requires exactly three profiles")
        if len({profile.key for profile in self.matrix_profiles}) != 3:
            raise ValueError("matrix profile keys must be unique")
        if any(profile.provider != "modal" for profile in self.matrix_profiles):
            raise ValueError("matrix profiles must use the frozen Modal serving path")
        if self.judge_profile.provider != "modal":
            raise ValueError("judge must use the frozen Modal serving path")
        matrix_identities = {profile.identity_sha256 for profile in self.matrix_profiles}
        if self.judge_profile.identity_sha256 in matrix_identities:
            raise ValueError("judge must be independent of every matrix profile")

    def manifest_json(self) -> dict[str, Any]:
        return {
            "judge_profile": self.judge_profile.as_json(),
            "matrix_profiles": [
                profile.as_json()
                for profile in sorted(self.matrix_profiles, key=lambda value: value.key)
            ],
            "roster_sha256": self.roster_sha256,
            "version": self.version,
        }

    @property
    def roster_sha256(self) -> str:
        payload = {
            "judge_profile": self.judge_profile.as_json(),
            "matrix_profiles": [
                profile.as_json()
                for profile in sorted(self.matrix_profiles, key=lambda value: value.key)
            ],
            "version": self.version,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


PHASE2_MODEL_ROSTER_V1 = FrozenModelRoster(
    version="phase2-model-roster-v1",
    matrix_profiles=(
        freeze_modal_profile(
            key="qwen36-27b-fp8",
            profile=QWEN36_27B_FP8_CANDIDATE,
        ),
        freeze_modal_profile(
            key="gemma4-31b-it",
            profile=GEMMA4_31B_CANDIDATE,
        ),
        freeze_modal_profile(
            key="mistral-small-3.2-24b",
            profile=MISTRAL_SMALL_32_24B_CANDIDATE,
        ),
    ),
    judge_profile=freeze_modal_profile(
        key="gpt-oss-20b-judge",
        profile=GPT_OSS_20B_JUDGE_CANDIDATE,
    ),
)
