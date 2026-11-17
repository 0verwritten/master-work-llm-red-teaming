from __future__ import annotations

import json
import subprocess
from typing import Any

import modal

from harness.agents.native.modal_provider import QWEN36_27B_FP8_CANDIDATE
from harness.agents.native.modal_readiness import wait_for_http_server

PROFILE = QWEN36_27B_FP8_CANDIDATE
VLLM_PORT = 8000
STARTUP_TIMEOUT_SECONDS = 15 * 60
READINESS_TIMEOUT_SECONDS = 14 * 60

app = modal.App(PROFILE.app_name)
hf_cache = modal.Volume.from_name("research-harness-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("research-harness-vllm-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={PROFILE.inference_engine_version}")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_DEEP_GEMM_WARMUP": PROFILE.deep_gemm_warmup,
            "VLLM_LOG_STATS_INTERVAL": "10",
        }
    )
    .add_local_python_source("harness")
)


@app.server(
    image=image,
    gpu=PROFILE.gpu,
    min_containers=0,
    max_containers=1,
    target_concurrency=1,
    scaledown_window=60,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    port=VLLM_PORT,
    routing_region="eu-west",
    unauthenticated=False,
)
class Server:
    process: subprocess.Popen[Any]

    @modal.enter()
    def start(self) -> None:
        command = [
            "vllm",
            "serve",
            PROFILE.served_model,
            "--revision",
            PROFILE.checkpoint_revision,
            "--served-model-name",
            PROFILE.served_model,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--dtype",
            PROFILE.dtype,
            "--gdn-prefill-backend",
            PROFILE.gdn_prefill_backend,
            "--safetensors-load-strategy",
            PROFILE.safetensors_load_strategy,
            "--tensor-parallel-size",
            str(PROFILE.tensor_parallel_size),
            "--max-model-len",
            str(PROFILE.max_model_len),
            "--generation-config",
            "vllm",
            "--seed",
            "0",
            "--limit-mm-per-prompt",
            json.dumps({"image": 0, "video": 0, "audio": 0}),
        ]
        if PROFILE.enforce_eager:
            command.append("--enforce-eager")
        self.process = subprocess.Popen(command)
        wait_for_http_server(
            self.process,
            health_url=f"http://127.0.0.1:{VLLM_PORT}/health",
            timeout_seconds=READINESS_TIMEOUT_SECONDS,
        )

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
