from __future__ import annotations

from typing import Any

import modal

from preliminary_study.provider import PreliminaryServingProfile

MAX_BATCH_SIZE = 8

BATCH_WAIT_MS = 2_000


def build_hf_app(profile: PreliminaryServingProfile) -> tuple[modal.App, type]:

    app = modal.App(profile.app_name)
    hf_cache = modal.Volume.from_name("research-harness-huggingface-cache", create_if_missing=True)

    image = (
        modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .uv_pip_install(
            *profile.extra_pip_packages,
            "torch",
            "torchvision",
            "accelerate",
            "pillow",
            "hf_transfer",
            "pydantic",
        )
        .env({**profile.server_env, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
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
        volumes={"/root/.cache/huggingface": hf_cache},
    )
    class Server:
        @modal.enter()
        def start(self) -> None:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            self.torch = torch
            self.processor = AutoProcessor.from_pretrained(
                profile.served_model,
                revision=profile.checkpoint_revision,
            )

            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                profile.served_model,
                revision=profile.checkpoint_revision,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.eval()

        def _decode_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:

            import base64
            import io

            from PIL import Image

            converted: list[dict[str, Any]] = []
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    converted.append({"role": message["role"], "content": content})
                    continue
                parts: list[dict[str, Any]] = []
                for part in content or []:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if not url.startswith("data:"):
                            raise ValueError("only inlined data: image URLs are accepted")
                        raw = base64.b64decode(url.split(",", 1)[1])
                        parts.append(
                            {
                                "type": "image",
                                "image": Image.open(io.BytesIO(raw)).convert("RGB"),
                            }
                        )
                    else:
                        parts.append({"type": "text", "text": part.get("text", "")})
                converted.append({"role": message["role"], "content": parts})
            return converted

        def _generate_group(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:

            import hashlib

            torch = self.torch
            conversations = [self._decode_messages(p["messages"]) for p in payloads]
            first = payloads[0]
            temperature = float(first.get("temperature", 0.0))
            max_new_tokens = int(first.get("max_tokens", 512))
            top_p = float(first.get("top_p", 1.0))

            inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            prompt_width = int(inputs["input_ids"].shape[-1])

            digest = hashlib.sha256(
                ",".join(str(int(p.get("seed", 0))) for p in payloads).encode()
            ).digest()
            torch.manual_seed(int.from_bytes(digest[:4], "big"))

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            results: list[dict[str, Any]] = []
            for index, payload in enumerate(payloads):
                new_tokens = generated[index][prompt_width:]
                text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
                results.append(
                    {
                        "id": f"hf-{payload.get('seed', 0)}",
                        "model": profile.served_model,
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": (
                                    "length" if len(new_tokens) >= max_new_tokens else "stop"
                                ),
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_width,
                            "completion_tokens": len(new_tokens),
                        },
                    }
                )
            return results

        @modal.batched(max_batch_size=MAX_BATCH_SIZE, wait_ms=BATCH_WAIT_MS)
        def chat(self, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:

            groups: dict[tuple[float, float, int], list[int]] = {}
            for index, item in enumerate(payload):
                key = (
                    float(item.get("temperature", 0.0)),
                    float(item.get("top_p", 1.0)),
                    int(item.get("max_tokens", 512)),
                )
                groups.setdefault(key, []).append(index)

            results: list[dict[str, Any] | None] = [None] * len(payload)
            for indices in groups.values():
                generated = self._generate_group([payload[i] for i in indices])
                for position, index in enumerate(indices):
                    results[index] = generated[position]

            missing = [i for i, value in enumerate(results) if value is None]
            if missing:
                raise RuntimeError(f"batch produced no result for positions {missing}")
            return [value for value in results if value is not None]

    return app, Server
