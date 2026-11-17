from __future__ import annotations

import subprocess
from typing import Any

import modal

from harness.agents.native.modal_readiness import wait_for_http_server
from preliminary_study.provider import PreliminaryServingProfile

VLLM_PORT = 8000
READINESS_TIMEOUT_SECONDS = 14 * 60
STARTUP_TIMEOUT_SECONDS = 20 * 60

MAX_CONCURRENT_INPUTS = 8


def build_app(profile: PreliminaryServingProfile) -> tuple[modal.App, type]:

    app = modal.App(profile.app_name)
    hf_cache = modal.Volume.from_name("research-harness-huggingface-cache", create_if_missing=True)
    vllm_cache = modal.Volume.from_name("research-harness-vllm-cache", create_if_missing=True)

    image = (
        modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .uv_pip_install(
            f"vllm=={profile.inference_engine_version}",
            *profile.extra_pip_packages,
            "httpx",
            "pydantic",
        )
        .env(profile.server_env)
        .add_local_python_source("harness", "preliminary_study")
    )

    @app.cls(
        image=image,
        gpu=profile.gpu,
        serialized=True,
        min_containers=0,
        max_containers=profile.max_containers,
        scaledown_window=profile.scaledown_window_seconds,
        timeout=60 * 60,
        volumes={
            "/root/.cache/huggingface": hf_cache,
            "/root/.cache/vllm": vllm_cache,
        },
    )
    @modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
    class Server:
        process: subprocess.Popen[Any]

        @modal.enter()
        def start(self) -> None:
            command = [
                "vllm",
                "serve",
                profile.served_model,
                "--revision",
                profile.checkpoint_revision,
                "--served-model-name",
                profile.served_model,
                "--host",
                "127.0.0.1",
                "--port",
                str(VLLM_PORT),
                "--dtype",
                profile.dtype,
                "--tensor-parallel-size",
                str(profile.tensor_parallel_size),
                "--max-model-len",
                str(profile.max_model_len),
                "--generation-config",
                "vllm",
                "--seed",
                "0",
                "--safetensors-load-strategy",
                profile.safetensors_load_strategy,
                *profile.server_args,
            ]
            if profile.enforce_eager:
                command.append("--enforce-eager")
            self.process = subprocess.Popen(command)
            wait_for_http_server(
                self.process,
                health_url=f"http://127.0.0.1:{VLLM_PORT}/health",
                timeout_seconds=READINESS_TIMEOUT_SECONDS,
            )

        @modal.method()
        def chat(self, payload: dict[str, Any]) -> dict[str, Any]:

            import httpx

            response = httpx.post(
                f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
                json=payload,
                timeout=profile.request_timeout_seconds,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

        @modal.exit()
        def stop(self) -> None:
            self.process.terminate()

    return app, Server
