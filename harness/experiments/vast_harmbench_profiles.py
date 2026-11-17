from __future__ import annotations

from harness.agents.native.modal_provider import (
    GPT_OSS_20B_JUDGE_CANDIDATE,
    MISTRAL_SMALL_32_24B_CANDIDATE,
    QWEN36_27B_FP8_CANDIDATE,
)
from harness.agents.native.vast_provider import VastServingProfile
from harness.experiments.adversarial_matrix import FrozenModelProfile

VLLM_IMAGE = (
    "docker.io/vllm/vllm-openai@"
    "sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9"
)
VLLM_IMAGE_SHA256 = "a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9"

VAST_QWEN_H100 = VastServingProfile(
    name="vast-qwen36-27b-fp8-h100-v1",
    served_model=QWEN36_27B_FP8_CANDIDATE.served_model,
    checkpoint_revision=QWEN36_27B_FP8_CANDIDATE.checkpoint_revision,
    config_sha256=QWEN36_27B_FP8_CANDIDATE.config_sha256,
    chat_template_sha256=QWEN36_27B_FP8_CANDIDATE.chat_template_sha256,
    inference_engine="vllm",
    inference_engine_version="0.21.0",
    container_image=VLLM_IMAGE,
    container_image_sha256=VLLM_IMAGE_SHA256,
    cuda_version="13.0.2",
    gpu_name="H100 SXM",
    num_gpus=1,
    tensor_parallel_size=1,
    dtype="auto",
    max_model_len=32_768,
    max_output_tokens=2_048,
    request_timeout_seconds=900,
    request_overrides={"chat_template_kwargs": {"enable_thinking": False}},
    server_args=(
        "--revision",
        QWEN36_27B_FP8_CANDIDATE.checkpoint_revision,
        "--served-model-name",
        QWEN36_27B_FP8_CANDIDATE.served_model,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--gdn-prefill-backend",
        "triton",
        "--safetensors-load-strategy",
        "prefetch",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--generation-config",
        "vllm",
        "--seed",
        "0",
        "--limit-mm-per-prompt",
        '{"image": 0, "video": 0, "audio": 0}',
        "--enforce-eager",
    ),
    server_env={
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_DEEP_GEMM_WARMUP": "skip",
    },
)

VAST_GPT_OSS_A6000 = VastServingProfile(
    name="vast-gpt-oss-20b-a6000-judge-v1",
    served_model=GPT_OSS_20B_JUDGE_CANDIDATE.served_model,
    checkpoint_revision=GPT_OSS_20B_JUDGE_CANDIDATE.checkpoint_revision,
    config_sha256=GPT_OSS_20B_JUDGE_CANDIDATE.config_sha256,
    chat_template_sha256=GPT_OSS_20B_JUDGE_CANDIDATE.chat_template_sha256,
    inference_engine="vllm",
    inference_engine_version="0.21.0",
    container_image=VLLM_IMAGE,
    container_image_sha256=VLLM_IMAGE_SHA256,
    cuda_version="13.0.2",
    gpu_name="RTX A6000",
    num_gpus=1,
    tensor_parallel_size=1,
    dtype="auto",
    max_model_len=32_768,
    max_output_tokens=2_048,
    request_timeout_seconds=900,
    request_overrides={"reasoning_effort": "low"},
    server_args=(
        "--revision",
        GPT_OSS_20B_JUDGE_CANDIDATE.checkpoint_revision,
        "--served-model-name",
        GPT_OSS_20B_JUDGE_CANDIDATE.served_model,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--safetensors-load-strategy",
        "prefetch",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--generation-config",
        "vllm",
        "--seed",
        "0",
        "--enforce-eager",
        "--reasoning-parser",
        "openai_gptoss",
    ),
    server_env={
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_DEEP_GEMM_WARMUP": "skip",
    },
)

VAST_GPT_OSS_L40S = VastServingProfile.model_validate(
    {
        **VAST_GPT_OSS_A6000.model_dump(mode="json"),
        "name": "vast-gpt-oss-20b-l40s-judge-v1",
        "gpu_name": "L40S",
    }
)

VAST_GPT_OSS_RTX6000_ADA = VastServingProfile.model_validate(
    {
        **VAST_GPT_OSS_A6000.model_dump(mode="json"),
        "name": "vast-gpt-oss-20b-rtx6000ada-judge-v1",
        "gpu_name": "RTX 6000Ada",
    }
)

VAST_MISTRAL_SMALL_H100 = VastServingProfile(
    name="vast-mistral-small-3.2-24b-h100-attacker-v1",
    served_model=MISTRAL_SMALL_32_24B_CANDIDATE.served_model,
    checkpoint_revision=MISTRAL_SMALL_32_24B_CANDIDATE.checkpoint_revision,
    config_sha256=MISTRAL_SMALL_32_24B_CANDIDATE.config_sha256,
    chat_template_sha256=MISTRAL_SMALL_32_24B_CANDIDATE.chat_template_sha256,
    inference_engine="vllm",
    inference_engine_version="0.21.0",
    container_image=VLLM_IMAGE,
    container_image_sha256=VLLM_IMAGE_SHA256,
    cuda_version="13.0.2",
    gpu_name="H100 SXM",
    num_gpus=1,
    tensor_parallel_size=1,
    dtype="auto",
    max_model_len=32_768,
    max_output_tokens=2_048,
    request_timeout_seconds=900,
    server_args=(
        "--revision",
        MISTRAL_SMALL_32_24B_CANDIDATE.checkpoint_revision,
        "--served-model-name",
        MISTRAL_SMALL_32_24B_CANDIDATE.served_model,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--safetensors-load-strategy",
        "prefetch",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--generation-config",
        "vllm",
        "--seed",
        "0",
        "--enforce-eager",
        "--tokenizer-mode",
        "mistral",
        "--config-format",
        "mistral",
        "--load-format",
        "mistral",
        "--language-model-only",
    ),
    server_env={
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_DEEP_GEMM_WARMUP": "skip",
    },
)


def freeze_vast_profile(*, key: str, profile: VastServingProfile) -> FrozenModelProfile:
    return FrozenModelProfile(
        key=key,
        provider="vast.ai",
        model=None,
        endpoint_profile=profile.profile_id,
        revision=profile.checkpoint_revision,
    )


VAST_QWEN_H100_FROZEN = freeze_vast_profile(key="qwen36-27b-fp8", profile=VAST_QWEN_H100)
VAST_MISTRAL_SMALL_H100_FROZEN = freeze_vast_profile(
    key="mistral-small-3.2-24b", profile=VAST_MISTRAL_SMALL_H100
)
VAST_GPT_OSS_A6000_FROZEN = freeze_vast_profile(key="gpt-oss-20b-judge", profile=VAST_GPT_OSS_A6000)
VAST_GPT_OSS_L40S_FROZEN = freeze_vast_profile(key="gpt-oss-20b-judge", profile=VAST_GPT_OSS_L40S)
VAST_GPT_OSS_RTX6000_ADA_FROZEN = freeze_vast_profile(
    key="gpt-oss-20b-judge", profile=VAST_GPT_OSS_RTX6000_ADA
)
