from __future__ import annotations

import modal

app = modal.App("research-harness-prefetch")
hf_cache = modal.Volume.from_name("research-harness-huggingface-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache},
    cpu=4.0,
    memory=16384,
    timeout=4 * 60 * 60,
)
def prefetch(repo_id: str, revision: str) -> dict[str, object]:

    from pathlib import Path

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir="/root/.cache/huggingface",
        max_workers=8,
    )
    files = sorted(p.name for p in Path(path).glob("*.safetensors"))
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    hf_cache.commit()
    return {
        "repo": repo_id,
        "revision": revision,
        "safetensors_files": len(files),
        "bytes": total,
    }


@app.local_entrypoint()
def main(key: str = "") -> None:

    from harness.agents.native.modal_provider import (
        GEMMA4_31B_JUDGE_V2,
        GPT_OSS_20B_JUDGE_CANDIDATE,
        MISTRAL_SMALL_32_24B_CANDIDATE,
        QWEN36_27B_FP8_CANDIDATE,
    )

    profiles = {
        "qwen36": QWEN36_27B_FP8_CANDIDATE,
        "gemma4": GEMMA4_31B_JUDGE_V2,
        "gpt-oss-judge": GPT_OSS_20B_JUDGE_CANDIDATE,
        "mistral32": MISTRAL_SMALL_32_24B_CANDIDATE,
    }
    if key and key not in profiles:
        raise SystemExit(f"unknown profile {key!r}; expected one of {sorted(profiles)}")
    wanted = [profiles[key]] if key else list(profiles.values())
    for profile in wanted:
        print(f"prefetching {profile.served_model} @ {profile.checkpoint_revision}", flush=True)
        result = prefetch.remote(profile.served_model, profile.checkpoint_revision)
        gib = float(result["bytes"]) / 1024**3  # type: ignore[arg-type]
        print(f"  done: {result['safetensors_files']} safetensors, {gib:.1f} GiB", flush=True)
