from __future__ import annotations

from harness.agents.native.modal_provider import (
    GEMMA4_31B_CANDIDATE,
    GPT_OSS_20B_JUDGE_CANDIDATE,
    MISTRAL_SMALL_32_24B_CANDIDATE,
    QWEN36_27B_FP8_CANDIDATE,
)
from preliminary_study.provider import PreliminaryServingProfile

VISION_ENGINE_VERSION = "0.27.1"
VISION_TRANSFORMERS_PIN = "transformers==5.15.0"

IMAGE_LIMITS = '{"image":1,"video":0,"audio":0}'
NO_IMAGE_LIMITS = '{"image":0,"video":0,"audio":0}'

QWEN36_35B_A3B_VISION = PreliminaryServingProfile(
    **{
        **QWEN36_27B_FP8_CANDIDATE.model_dump(),
        "name": "modal-qwen36-35b-a3b-fp8-vision-v1",
        "app_name": "research-harness-prelim-qwen36-35b-vision",
        "served_model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "checkpoint_revision": "95a723d08a9490559dae23d0cff1d9466213d989",
        "config_sha256": "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845",
        "chat_template_sha256": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
        "inference_engine_version": VISION_ENGINE_VERSION,
        "extra_pip_packages": (VISION_TRANSFORMERS_PIN,),
        "server_args": ("--limit-mm-per-prompt", IMAGE_LIMITS),
        "server_env": {"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_DEEP_GEMM_WARMUP": "skip"},
        "request_overrides": {"chat_template_kwargs": {"enable_thinking": False}},
    }
)

GEMMA4_31B_TEXT_ONLY = PreliminaryServingProfile(
    **{
        **GEMMA4_31B_CANDIDATE.model_dump(),
        "name": "modal-gemma4-31b-it-prelim-text-v1",
        "app_name": "research-harness-prelim-gemma4-31b",
        "inference_engine_version": VISION_ENGINE_VERSION,
        "extra_pip_packages": ("transformers==5.13.0",),
        "server_args": ("--language-model-only", "--limit-mm-per-prompt", NO_IMAGE_LIMITS),
    }
)

MISTRAL_SMALL_32_VISION = PreliminaryServingProfile(
    **{
        **MISTRAL_SMALL_32_24B_CANDIDATE.model_dump(),
        "name": "modal-mistral-small-3.2-24b-vision-v1",
        "app_name": "research-harness-prelim-mistral-small-32",
        "inference_engine_version": VISION_ENGINE_VERSION,
        "extra_pip_packages": (VISION_TRANSFORMERS_PIN,),
        "server_args": (
            "--tokenizer-mode",
            "mistral",
            "--config-format",
            "mistral",
            "--load-format",
            "mistral",
            "--limit-mm-per-prompt",
            IMAGE_LIMITS,
        ),
    }
)

GEMMA4_31B_VISION_HF = PreliminaryServingProfile(
    **{
        **GEMMA4_31B_CANDIDATE.model_dump(),
        "name": "modal-gemma4-31b-it-vision-hf-v1",
        "app_name": "research-harness-prelim-gemma4-31b-hf",
        "inference_engine": "transformers",
        "inference_engine_version": "5.15.0",
        "extra_pip_packages": ("transformers==5.15.0",),
        "server_args": (),
        "server_env": {"HF_XET_HIGH_PERFORMANCE": "1"},
        "estimated_active_cost_usd_per_second": 0.00150,
        "sampling_scope": "batch",
    }
)

GPT_OSS_20B_TEXT_ONLY = PreliminaryServingProfile(
    **{
        **GPT_OSS_20B_JUDGE_CANDIDATE.model_dump(),
        "name": "modal-gpt-oss-20b-prelim-text-v1",
        "app_name": "research-harness-prelim-gpt-oss-20b",
        "inference_engine_version": VISION_ENGINE_VERSION,
        "extra_pip_packages": (VISION_TRANSFORMERS_PIN,),
        "server_args": ("--reasoning-parser", "openai_gptoss"),
    }
)


class RosterEntry:
    def __init__(
        self,
        *,
        key: str,
        profile: PreliminaryServingProfile,
        supports_vision: bool,
        vision_note: str,
        judge_max_output_tokens: int | None = None,
    ) -> None:
        self.key = key
        self.profile = profile
        self.supports_vision = supports_vision
        self.vision_note = vision_note

        self.judge_max_output_tokens = judge_max_output_tokens

    def public_manifest(self) -> dict[str, object]:
        return {
            "key": self.key,
            "served_model": self.profile.served_model,
            "checkpoint_revision": self.profile.checkpoint_revision,
            "serving_profile_id": self.profile.profile_id,
            "supports_vision": self.supports_vision,
            "vision_note": self.vision_note,
            "judge_max_output_tokens_override": self.judge_max_output_tokens,
        }


ROSTER: dict[str, RosterEntry] = {
    entry.key: entry
    for entry in (
        RosterEntry(
            key="qwen36-35b",
            profile=QWEN36_35B_A3B_VISION,
            supports_vision=True,
            vision_note="checkpoint declares vision_config; served with one image per prompt",
        ),
        RosterEntry(
            key="gemma4-31b",
            profile=GEMMA4_31B_TEXT_ONLY,
            supports_vision=False,
            vision_note=(
                "checkpoint declares vision_config, but vLLM cannot serve it: multimodal "
                "profiling aborts inside Gemma4Processor ('Found 1 <|image|> tokens in the "
                "text but no images were passed') on vllm 0.21.0+transformers 5.13.0 and on "
                "vllm 0.27.1+transformers 5.15.0. Text arms only."
            ),
        ),
        RosterEntry(
            key="gemma4-31b-hf",
            profile=GEMMA4_31B_VISION_HF,
            supports_vision=True,
            vision_note=(
                "same checkpoint as gemma4-31b, served through transformers instead of vLLM. "
                "Hugging Face serves the vision tower that vLLM's multimodal profiling cannot, "
                "so the image arm is available here. Not pooled with the vLLM-served run: the "
                "engine differs, and it is part of the serving identity."
            ),
        ),
        RosterEntry(
            key="mistral-small-32",
            profile=MISTRAL_SMALL_32_VISION,
            supports_vision=True,
            vision_note="checkpoint declares vision_config; served with one image per prompt",
        ),
        RosterEntry(
            key="gpt-oss-20b",
            profile=GPT_OSS_20B_TEXT_ONLY,
            supports_vision=False,
            vision_note=(
                "checkpoint has no vision_config and no image token; the model is text-only, "
                "so its image arm does not exist. Text arms only."
            ),
            judge_max_output_tokens=1024,
        ),
    )
}

VISION_KEYS = tuple(key for key, entry in ROSTER.items() if entry.supports_vision)
TEXT_ONLY_KEYS = tuple(key for key, entry in ROSTER.items() if not entry.supports_vision)


def get_entry(key: str) -> RosterEntry:
    if key not in ROSTER:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(ROSTER)}")
    return ROSTER[key]
