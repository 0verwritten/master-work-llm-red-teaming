from __future__ import annotations

import subprocess
from typing import Any

import modal

from harness.agents.native.modal_provider import GEMMA4_31B_JUDGE_V2
from harness.agents.native.modal_readiness import wait_for_http_server

PROFILE = GEMMA4_31B_JUDGE_V2
VLLM_PORT = 8000
READINESS_TIMEOUT_SECONDS = 14 * 60

app = modal.App(PROFILE.app_name)
hf_cache = modal.Volume.from_name("research-harness-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("research-harness-vllm-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={PROFILE.inference_engine_version}", *PROFILE.extra_pip_packages)
    .env(PROFILE.server_env)
    .add_local_python_source("harness")
)


@app.server(
    image=image,
    gpu=PROFILE.gpu,
    min_containers=PROFILE.min_containers,
    max_containers=PROFILE.max_containers,
    scaledown_window=PROFILE.scaledown_window_seconds,
    startup_timeout=15 * 60,
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
            "--tensor-parallel-size",
            str(PROFILE.tensor_parallel_size),
            "--max-model-len",
            str(PROFILE.max_model_len),
            "--generation-config",
            "vllm",
            "--seed",
            "0",
            "--safetensors-load-strategy",
            PROFILE.safetensors_load_strategy,
            *PROFILE.server_args,
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
