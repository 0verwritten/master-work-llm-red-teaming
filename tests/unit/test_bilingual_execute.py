from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from harness.bilingual_dataset import HUMAN_VERIFIED, DatasetItem
from harness.experiments import bilingual_execute
from harness.experiments.adversarial_matrix import FrozenModelProfile
from harness.experiments.bilingual_manifest import build_manifest
from harness.experiments.checkpoint import CheckpointLog, CheckpointRecord
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.repository import TraceRepository


def _item(index: int) -> DatasetItem:
    return DatasetItem(
        stable_id=f"fixture:{index}",
        pair_id=f"fixture:{index}",
        language="en",
        stratum="harmbench_subset",
        category="fixture",
        prompt_text=f"prompt {index}",
        context_text=None,
        rubric_text="fixture rubric",
        expected_behavior="refuse_or_safe_redirect",
        reviewer_status=HUMAN_VERIFIED,
        translation_provenance="source_language",
        license_id="MIT",
    )


def _profile(key: str) -> FrozenModelProfile:
    return FrozenModelProfile(
        key=key,
        provider="modal",
        model=None,
        endpoint_profile=f"{key}-v1",
        revision="a" * 40,
    )


async def test_cells_run_with_bounded_concurrency_and_isolated_sessions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    items = [_item(index) for index in range(4)]
    manifest = build_manifest(
        items,
        dataset_sha256="d" * 64,
        subset_plan_sha256="p" * 64,
        attacker=_profile("attacker"),
        target=_profile("target"),
        judge=_profile("judge"),
        max_rounds=1,
    )
    active = 0
    peak = 0

    sessions: list[Any] = []

    async def fake_execute_cell(**kwargs: Any) -> CheckpointRecord:
        nonlocal active, peak
        repository: TraceRepository = kwargs["repository"]
        cell = kwargs["cell"]
        sessions.append(repository.session)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return CheckpointRecord(
            cell_id=cell.cell_id,
            language=cell.language,
            category=cell.category,
            status="completed",
            judge_language_adherence="pass",
            label="refused",
        )

    monkeypatch.setattr(bilingual_execute, "_execute_cell", fake_execute_cell)
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    make_session = session_factory(engine)

    @asynccontextmanager
    async def repository_factory() -> AsyncIterator[TraceRepository]:
        async with make_session() as session:
            yield TraceRepository(session)

    try:
        async with make_session() as session:
            result = await bilingual_execute.execute_smoke(
                manifest,
                items=items,
                repository=TraceRepository(session),
                client_factory=lambda **_: None,  # type: ignore[arg-type]
                checkpoint=CheckpointLog(tmp_path / "checkpoint.jsonl"),
                max_concurrency=2,
                repository_factory=repository_factory,
            )
    finally:
        await engine.dispose()

    assert result["executed_cells"] == 4
    assert result["max_concurrency"] == 2
    assert peak == 2
    assert len({id(session) for session in sessions}) == 4


async def test_role_retry_allowance_reaches_spec_and_call_ceiling(
    tmp_path: Path, monkeypatch: Any
) -> None:

    from harness.experiments.bilingual_config import build_cell_config

    items = [_item(0)]
    manifest = build_manifest(
        items,
        dataset_sha256="d" * 64,
        subset_plan_sha256="p" * 64,
        attacker=_profile("attacker"),
        target=_profile("target"),
        judge=_profile("judge"),
        max_rounds=3,
    )
    single = build_cell_config(manifest, manifest.cells[0])
    retried = build_cell_config(manifest, manifest.cells[0], max_role_attempts=3)
    assert single.budgets.max_total_llm_calls == 9
    assert retried.budgets.max_total_llm_calls == 27
