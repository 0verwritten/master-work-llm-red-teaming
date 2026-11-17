"""One benign request against the vision deployment, before the paid batch.

It proves four things that would otherwise only be discovered 576 calls in: the
Modal method call works, the vision tower is loaded, an inlined PNG data URI is
accepted, and the served model identity matches the frozen profile.

The payload is a harmless word-echo instruction, not a benchmark item, so this
costs one cold start and one short generation and touches no research content.
"""

from __future__ import annotations

import argparse
import asyncio

from preliminary_study.protocol import DEFAULT_PROTOCOL, Arm
from preliminary_study.provider import ChatRequest, ModalFunctionChatProvider, image_content_part
from preliminary_study.rendering import render_payload
from preliminary_study.roster import ROSTER, get_entry

SMOKE_PAYLOAD = "Reply with exactly the word BANANA and nothing else."

BATCH_PAYLOADS = [
    ("BANANA", "Reply with exactly the word BANANA and nothing else."),
    ("CRIMSON", "Reply with exactly the word CRIMSON and nothing else."),
    ("LANTERN", "Reply with exactly the word LANTERN and nothing else."),
    ("MERIDIAN", "Reply with exactly the word MERIDIAN and nothing else."),
]


async def _run(model_key: str) -> int:
    entry = get_entry(model_key)
    print(f"model {entry.key}  ({entry.profile.served_model})")
    provider = ModalFunctionChatProvider(profile=entry.profile)

    if entry.supports_vision:
        rendered = render_payload(SMOKE_PAYLOAD, DEFAULT_PROTOCOL)
        print(f"smoke image {rendered.sha256[:16]} ({len(rendered.png_bytes)} bytes)")
        content: object = [
            {"type": "text", "text": DEFAULT_PROTOCOL.carrier_prompt},
            image_content_part(rendered.png_bytes),
        ]
        arm = Arm.IMAGE
    else:
        print(f"text-only target: {entry.vision_note}")
        content = SMOKE_PAYLOAD
        arm = Arm.TEXT_PLAIN

    request = ChatRequest(
        messages=[{"role": "user", "content": content}],
        seed=DEFAULT_PROTOCOL.request_seed(item_id="smoke", arm=arm, repetition=1),
        temperature=0.0,
        max_output_tokens=64,
    )
    print("issuing one request (a cold start takes several minutes)...")
    completion = await provider.complete(request)

    print(f"\nlatency        {completion.latency_seconds:.1f}s")
    print(f"prompt tokens  {completion.prompt_tokens}")
    print(f"output tokens  {completion.completion_tokens}")
    print(f"finish reason  {completion.finish_reason}")
    print(f"est. cost      USD {completion.estimated_cost_usd:.4f}")
    print(f"response       {completion.content!r}")

    if "BANANA" not in completion.content.upper():
        print("\nFAIL: the model did not follow the instruction it was given.")
        return 1
    print("\nPASS: serving works and the payload arrived.")
    return 0


async def _run_batch(model_key: str) -> int:

    entry = get_entry(model_key)
    provider = ModalFunctionChatProvider(profile=entry.profile)
    print(f"model {entry.key}: firing {len(BATCH_PAYLOADS)} concurrent requests")

    async def one(index: int, expected: str, text: str) -> tuple[str, str]:
        if entry.supports_vision:
            rendered = render_payload(text, DEFAULT_PROTOCOL)
            content: object = [
                {"type": "text", "text": DEFAULT_PROTOCOL.carrier_prompt},
                image_content_part(rendered.png_bytes),
            ]
            arm = Arm.IMAGE
        else:
            content = text
            arm = Arm.TEXT_PLAIN
        request = ChatRequest(
            messages=[{"role": "user", "content": content}],
            seed=DEFAULT_PROTOCOL.request_seed(item_id=f"smoke{index}", arm=arm, repetition=1),
            temperature=0.0,
            max_output_tokens=64,
        )
        completion = await provider.complete(request)
        return expected, completion.content

    results = await asyncio.gather(*(one(i, e, t) for i, (e, t) in enumerate(BATCH_PAYLOADS)))

    failures = []
    for expected, actual in results:
        ok = expected in actual.upper()
        print(f"  expected {expected:<9} got {actual!r} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(expected)
    if failures:
        print(f"\nFAIL: {len(failures)} of {len(results)} rows wrong - batching corrupts rows.")
        return 1
    print("\nPASS: every row in the batch got its own correct answer.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(ROSTER))
    parser.add_argument(
        "--batch",
        action="store_true",
        help="fire several distinct requests at once to verify batched serving",
    )
    args = parser.parse_args()
    runner = _run_batch(args.model) if args.batch else _run(args.model)
    raise SystemExit(asyncio.run(runner))


if __name__ == "__main__":
    main()
