"""Print the Vast onstart command or create-instance fields for a frozen profile.

The served command is derived from ``VastServingProfile.server_args`` so a launch
cannot drift from the content-addressed identity the runner later verifies. The
vLLM API key is never part of the output: vLLM reads ``VLLM_API_KEY`` from the
container environment.
"""

from __future__ import annotations

import argparse
import shlex

from harness.agents.native.vast_provider import VastServingProfile
from harness.experiments.vast_harmbench_profiles import (
    VAST_GPT_OSS_L40S,
    VAST_GPT_OSS_RTX6000_ADA,
    VAST_MISTRAL_SMALL_H100,
    VAST_QWEN_H100,
)

PROFILES: dict[str, VastServingProfile] = {
    "qwen-h100": VAST_QWEN_H100,
    "gptoss-rtx6000ada": VAST_GPT_OSS_RTX6000_ADA,
    "gptoss-l40s": VAST_GPT_OSS_L40S,
    "mistral-h100": VAST_MISTRAL_SMALL_H100,
}


def onstart_command(profile: VastServingProfile) -> str:
    environment = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(profile.server_env.items())
    )
    serve = shlex.join(["vllm", "serve", profile.served_model, *profile.server_args])
    return f"nohup env {environment} {serve} > /root/vllm.log 2>&1 &"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=sorted(PROFILES))
    parser.add_argument("field", choices=("onstart", "image", "gpu-query", "profile-id"))
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    if args.field == "onstart":
        print(onstart_command(profile))
    elif args.field == "image":
        print(profile.container_image)
    elif args.field == "gpu-query":
        print(profile.gpu_name.replace(" ", "_"))
    else:
        print(profile.profile_id)


if __name__ == "__main__":
    main()
