from __future__ import annotations

import json
from typing import Any, get_args

from harness.analysis.phase4_instruments import (
    REGEX_SHA256,
    REGEX_VERSION,
    regex_label,
    verify_regex_identity,
)
from preliminary_study.items import BenchmarkItem
from preliminary_study.protocol import Arm, Protocol, Verdict
from preliminary_study.provider import ChatRequest, image_content_part
from preliminary_study.rendering import RenderedImage

JUDGE_PARSER_VERSION = "preliminary-study-judge-parser-v1"

verify_regex_identity()

VERDICT_VALUES: tuple[str, ...] = get_args(Verdict)


def regex_refusal(response: str) -> bool:

    return regex_label(response) == "refused"


def parse_verdict(raw: str) -> tuple[Verdict, str]:

    stripped = raw.strip()
    fenced = _strip_code_fence(stripped)
    try:
        parsed = json.loads(fenced)
    except json.JSONDecodeError:
        return ("UNPARSED", stripped[:400])
    if not isinstance(parsed, dict):
        return ("UNPARSED", stripped[:400])
    verdict = str(parsed.get("verdict", "")).strip().upper()
    reason = str(parsed.get("reason", "")).strip()
    if verdict in {"REFUSED", "PARTIAL", "COMPLIED"}:
        return (verdict, reason)  # type: ignore[return-value]
    return ("UNPARSED", stripped[:400])


def _strip_code_fence(value: str) -> str:
    if not value.startswith("```"):
        return value
    body = value.split("\n", 1)[1] if "\n" in value else ""
    return body.rsplit("```", 1)[0].strip()


def target_request(
    *,
    item: BenchmarkItem,
    arm: Arm,
    repetition: int,
    protocol: Protocol,
    rendered: RenderedImage | None,
) -> ChatRequest:

    content: list[dict[str, Any]] | str
    if arm is Arm.IMAGE:
        if rendered is None:
            raise ValueError("image arm requires a rendered payload")
        content = [
            {"type": "text", "text": protocol.carrier_prompt},
            image_content_part(rendered.png_bytes),
        ]
    elif arm is Arm.TEXT_CARRIER:
        content = protocol.text_carrier_template.format(payload=item.payload)
    else:
        content = item.payload

    return ChatRequest(
        messages=[{"role": "user", "content": content}],
        seed=protocol.request_seed(item_id=item.item_id, arm=arm, repetition=repetition),
        temperature=protocol.temperature,
        max_output_tokens=protocol.max_output_tokens,
    )


def judge_request(
    *,
    response_text: str,
    protocol: Protocol,
    seed: int,
    max_output_tokens: int | None = None,
) -> ChatRequest:

    return ChatRequest(
        messages=[
            {"role": "system", "content": protocol.judge_system_prompt},
            {
                "role": "user",
                "content": protocol.judge_user_template.format(response=response_text),
            },
        ],
        seed=seed,
        temperature=protocol.judge_temperature,
        max_output_tokens=max_output_tokens or protocol.judge_max_output_tokens,
    )


__all__ = [
    "JUDGE_PARSER_VERSION",
    "REGEX_SHA256",
    "REGEX_VERSION",
    "VERDICT_VALUES",
    "judge_request",
    "parse_verdict",
    "regex_refusal",
    "target_request",
]
