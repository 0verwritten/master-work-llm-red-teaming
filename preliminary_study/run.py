"""Execute the corrected preliminary study.

Execution is two strictly ordered passes over the same trace file:

1. **Target pass** — one request per (item, arm, repetition), each appended as a
   `target` record.
2. **Judge pass** — iterate the *persisted target records* and score each one.
   The judge prompt is built from the record's own `response_text`, and the
   resulting record stores that record's `call_id` and `response_sha256`.

The legacy pipeline instead held `text_response` and `image_response` in local
variables and formatted the judge prompt twice from the same one. Here there is
no second variable to get wrong: the judge pass is driven by the records
themselves, and a judge record that does not name a real target response fails
schema validation before it is ever written.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preliminary_study.instruments import (
    JUDGE_PARSER_VERSION,
    REGEX_VERSION,
    judge_request,
    target_request,
)
from preliminary_study.items import BenchmarkItem, benchmark_identity_sha256, execution_items
from preliminary_study.protocol import (
    DEFAULT_PROTOCOL,
    Arm,
    Protocol,
    arms_for,
    text_sha256,
)
from preliminary_study.provider import (
    ChatCompletion,
    ChatProvider,
    ChatRequest,
    MockChatProvider,
    ModalFunctionChatProvider,
)
from preliminary_study.rendering import (
    FONT_SHA256,
    RENDERER_VERSION,
    RenderedImage,
    render_payload,
)
from preliminary_study.roster import ROSTER, RosterEntry, get_entry
from preliminary_study.trace import (
    VAULT_RUNS,
    RunLock,
    TraceRecord,
    TraceStore,
    make_call_id,
    utcnow_iso,
    write_json_atomic,
)


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def trace(self) -> Path:
        return self.root / "trace.jsonl"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def lock(self) -> Path:
        return self.root / "run.lock"


def build_manifest(
    *,
    run_id: str,
    protocol: Protocol,
    items: tuple[BenchmarkItem, ...],
    provider: ChatProvider,
    arms: tuple[Arm, ...],
    entry: RosterEntry | None = None,
) -> dict[str, Any]:

    return {
        "run_id": run_id,
        "model_key": entry.key if entry is not None else None,
        "supports_vision": entry.supports_vision if entry is not None else (Arm.IMAGE in arms),
        "vision_note": entry.vision_note if entry is not None else "",
        "created_at": utcnow_iso(),
        "protocol_identity_sha256": protocol.identity_sha256,
        "protocol": protocol.canonical(),
        "benchmark_identity_sha256": benchmark_identity_sha256(items),
        "items": [item.public_manifest() for item in items],
        "arms": [arm.value for arm in arms],
        "provider": provider.provider_name,
        "serving_profile": provider.profile.public_manifest(),
        "serving_profile_id": provider.profile.profile_id,
        "sampling_scope": getattr(provider.profile, "sampling_scope", "request"),
        "renderer_version": RENDERER_VERSION,
        "font_sha256": FONT_SHA256,
        "regex_version": REGEX_VERSION,
        "judge_parser_version": JUDGE_PARSER_VERSION,
        "judge_max_output_tokens_override": (
            entry.judge_max_output_tokens if entry is not None else None
        ),
        "planned_target_calls": len(items) * len(arms) * protocol.repetitions,
        "planned_judge_calls": len(items) * len(arms) * protocol.repetitions,
    }


@dataclass(frozen=True)
class _Unit:
    call_id: str
    purpose: str
    item_id: str
    arm: Arm
    repetition: int
    request: ChatRequest
    image_sha256: str | None = None
    judged_call_id: str | None = None
    judged_response_sha256: str | None = None


async def _complete_with_retry(
    provider: ChatProvider,
    request: ChatRequest,
    *,
    attempts: int = 3,
) -> ChatCompletion:

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await provider.complete(request)
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(2.0 * (attempt + 1))
    assert last is not None
    raise last


async def _issue(
    *,
    unit: _Unit,
    run_id: str,
    provider: ChatProvider,
    store: TraceStore,
    semaphore: asyncio.Semaphore,
    counter: list[int],
    total: int,
    verbose: bool,
) -> None:
    async with semaphore:
        completion = await _complete_with_retry(provider, unit.request)
    store.append(
        TraceRecord(
            call_id=unit.call_id,
            run_id=run_id,
            purpose=unit.purpose,  # type: ignore[arg-type]
            item_id=unit.item_id,
            arm=unit.arm,
            repetition=unit.repetition,
            request_sha256=unit.request.request_sha256,
            image_sha256=unit.image_sha256,
            seed=unit.request.seed,
            temperature=unit.request.temperature,
            max_output_tokens=unit.request.max_output_tokens,
            sampling_scope=getattr(provider.profile, "sampling_scope", "request"),
            judged_call_id=unit.judged_call_id,
            judged_response_sha256=unit.judged_response_sha256,
            provider=provider.provider_name,
            endpoint_profile=provider.profile.profile_id,
            served_model=provider.profile.served_model,
            checkpoint_revision=provider.profile.checkpoint_revision,
            response_text=completion.content,
            response_sha256=text_sha256(completion.content),
            finish_reason=completion.finish_reason,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_seconds=completion.latency_seconds,
            estimated_cost_usd=completion.estimated_cost_usd,
            created_at=utcnow_iso(),
        )
    )
    counter[0] += 1
    if verbose:
        print(
            f"{unit.purpose:<6} {counter[0]:>4}/{total}  "
            f"{unit.item_id:<24} {unit.arm.value:<13} rep{unit.repetition}"
        )


async def _drain(
    *,
    units: list[_Unit],
    run_id: str,
    provider: ChatProvider,
    store: TraceStore,
    concurrency: int,
    verbose: bool,
) -> int:

    if not units:
        return 0
    semaphore = asyncio.Semaphore(concurrency)
    counter = [0]
    await asyncio.gather(
        *(
            _issue(
                unit=unit,
                run_id=run_id,
                provider=provider,
                store=store,
                semaphore=semaphore,
                counter=counter,
                total=len(units),
                verbose=verbose,
            )
            for unit in units
        )
    )
    return len(units)


async def run_target_pass(
    *,
    run_id: str,
    protocol: Protocol,
    items: tuple[BenchmarkItem, ...],
    provider: ChatProvider,
    store: TraceStore,
    renders: dict[str, RenderedImage],
    arms: tuple[Arm, ...],
    concurrency: int,
    verbose: bool,
) -> int:
    done = store.completed_call_ids()
    units: list[_Unit] = []
    for item in items:
        for arm in arms:
            for repetition in range(1, protocol.repetitions + 1):
                call_id = make_call_id(
                    run_id=run_id,
                    purpose="target",
                    item_id=item.item_id,
                    arm=arm,
                    repetition=repetition,
                )
                if call_id in done:
                    continue
                rendered = renders[item.item_id] if arm is Arm.IMAGE else None
                units.append(
                    _Unit(
                        call_id=call_id,
                        purpose="target",
                        item_id=item.item_id,
                        arm=arm,
                        repetition=repetition,
                        request=target_request(
                            item=item,
                            arm=arm,
                            repetition=repetition,
                            protocol=protocol,
                            rendered=rendered,
                        ),
                        image_sha256=rendered.sha256 if rendered is not None else None,
                    )
                )
    return await _drain(
        units=units,
        run_id=run_id,
        provider=provider,
        store=store,
        concurrency=concurrency,
        verbose=verbose,
    )


async def run_judge_pass(
    *,
    run_id: str,
    protocol: Protocol,
    provider: ChatProvider,
    store: TraceStore,
    concurrency: int,
    verbose: bool,
    judge_tokens: int | None = None,
) -> int:

    records = store.load()
    targets = [record for record in records if record.purpose == "target"]
    done = {record.call_id for record in records if record.purpose == "judge"}

    units: list[_Unit] = []
    for target in targets:
        call_id = make_call_id(
            run_id=run_id,
            purpose="judge",
            item_id=target.item_id,
            arm=target.arm,
            repetition=target.repetition,
        )
        if call_id in done:
            continue
        units.append(
            _Unit(
                call_id=call_id,
                purpose="judge",
                item_id=target.item_id,
                arm=target.arm,
                repetition=target.repetition,
                request=judge_request(
                    response_text=target.response_text,
                    protocol=protocol,
                    seed=target.seed,
                    max_output_tokens=judge_tokens,
                ),
                judged_call_id=target.call_id,
                judged_response_sha256=target.response_sha256,
            )
        )
    return await _drain(
        units=units,
        run_id=run_id,
        provider=provider,
        store=store,
        concurrency=concurrency,
        verbose=verbose,
    )


async def execute(
    *,
    run_id: str,
    provider: ChatProvider,
    protocol: Protocol = DEFAULT_PROTOCOL,
    root: Path = VAULT_RUNS,
    concurrency: int = 8,
    save_images: bool = True,
    verbose: bool = True,
    entry: RosterEntry | None = None,
    arms: tuple[Arm, ...] | None = None,
) -> RunPaths:
    items = execution_items()
    paths = RunPaths(root / run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    store = TraceStore(paths.trace)

    if arms is None:
        supports_vision = entry.supports_vision if entry is not None else True
        arms = arms_for(supports_vision=supports_vision)

    renders = (
        {item.item_id: render_payload(item.payload, protocol) for item in items}
        if Arm.IMAGE in arms
        else {}
    )
    if save_images and renders:
        paths.images.mkdir(parents=True, exist_ok=True)
        for item_id, rendered in renders.items():
            safe = item_id.replace(":", "_")
            (paths.images / f"{safe}.png").write_bytes(rendered.png_bytes)

    write_json_atomic(
        paths.manifest,
        build_manifest(
            run_id=run_id,
            protocol=protocol,
            items=items,
            provider=provider,
            arms=arms,
            entry=entry,
        ),
    )

    with RunLock(paths.lock):
        targets, judges = await _run_passes(
            run_id=run_id,
            protocol=protocol,
            items=items,
            provider=provider,
            store=store,
            renders=renders,
            arms=arms,
            entry=entry,
            concurrency=concurrency,
            verbose=verbose,
        )
    if verbose:
        print(f"\nissued {targets} target calls and {judges} judge calls -> {paths.trace}")
    return paths


async def _run_passes(
    *,
    run_id: str,
    protocol: Protocol,
    items: tuple[BenchmarkItem, ...],
    provider: ChatProvider,
    store: TraceStore,
    renders: dict[str, RenderedImage],
    arms: tuple[Arm, ...],
    entry: RosterEntry | None,
    concurrency: int,
    verbose: bool,
) -> tuple[int, int]:
    targets = await run_target_pass(
        run_id=run_id,
        protocol=protocol,
        items=items,
        provider=provider,
        store=store,
        renders=renders,
        arms=arms,
        concurrency=concurrency,
        verbose=verbose,
    )
    judges = await run_judge_pass(
        run_id=run_id,
        protocol=protocol,
        provider=provider,
        concurrency=concurrency,
        store=store,
        verbose=verbose,
        judge_tokens=entry.judge_max_output_tokens if entry is not None else None,
    )
    return targets, judges


def _build_provider(kind: str, entry: RosterEntry) -> ChatProvider:
    if kind == "mock":
        return MockChatProvider(profile=entry.profile)

    return ModalFunctionChatProvider(profile=entry.profile)


async def _main(args: argparse.Namespace) -> None:
    entry = get_entry(args.model)
    run_id = args.run_id or f"{entry.key}-01"
    arms = arms_for(supports_vision=entry.supports_vision)

    print(f"model      {entry.key}  ({entry.profile.served_model})")
    print(f"arms       {', '.join(arm.value for arm in arms)}")
    if not entry.supports_vision:
        print(f"no image arm: {entry.vision_note}")
    print(f"run id     {run_id}\n")

    await execute(
        run_id=run_id,
        provider=_build_provider(args.provider, entry),
        concurrency=args.concurrency,
        entry=entry,
        arms=arms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(ROSTER),
        help="target model from the roster; decides which arms are available",
    )
    parser.add_argument(
        "--run-id",
        help="run directory under the vault; defaults to <model>-01",
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "modal"),
        default="mock",
        help="mock costs nothing; modal issues paid GPU calls via Modal account credentials",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="in-flight requests; higher batches better on vLLM and costs less GPU time",
    )
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
