from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness.experiments.verification_subset import (
    SUBSET_SIZE,
    EligibleItem,
    allocate,
    eligible_from_rows,
    select_subset,
)
from harness.harmbench_translation import (
    MACHINE_PROVENANCE,
    UNVERIFIED_STATUS,
    TranslationCache,
    assemble,
    build_units,
    estimate,
    translate_units,
    write_output,
)

RUBRIC = "Classify whether the response complies."


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _rows(count: int, *, contexts: int = 0) -> list[dict[str, object]]:
    categories = ["cybercrime", "copyright", "misinformation", "illegal"]
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.append(
            {
                "stable_id": f"harmbench:test:{index}",
                "prompt_text": f"english behavior {index}",
                "context_text": f"context {index}" if index < contexts else None,
                "content_sha256": _hash(f"item-{index}"),
                "category": categories[index % len(categories)],
                "functional_category": "standard",
                "split": "test",
                "modality": "text",
                "language": "en",
                "expected_behavior": "refuse_or_safe_redirect",
            }
        )
    return rows


class RecordingTranslator:
    def __init__(self, fail_after: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_after = fail_after
        self.translated = 0

    async def translate(self, texts: list[str]) -> list[str]:
        if self.fail_after is not None and self.translated >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls.append(list(texts))
        self.translated += len(texts)
        return [f"uk::{text}" for text in texts]


def test_rubric_is_translated_once_not_per_item() -> None:
    units = build_units(_rows(10), RUBRIC)
    rubric_units = [unit for unit in units if unit.field_name == "rubric_text"]
    assert len(rubric_units) == 1
    assert len(units) == 11


def test_contexts_add_units_only_when_present() -> None:
    units = build_units(_rows(10, contexts=3), RUBRIC)
    assert sum(1 for unit in units if unit.field_name == "context_text") == 3


def test_dry_run_estimate_counts_distinct_pending_characters(tmp_path: Path) -> None:
    cache = TranslationCache(path=tmp_path / "cache.json")
    units = build_units(_rows(4), RUBRIC)
    plan = estimate(units, cache)
    assert plan.pending_units == plan.total_units == 5
    assert plan.cached_units == 0
    assert plan.pending_characters == sum(len(text) for text in {unit.text for unit in units})


@pytest.mark.asyncio
async def test_estimate_drops_to_zero_after_a_completed_run(tmp_path: Path) -> None:
    cache = TranslationCache(path=tmp_path / "cache.json")
    units = build_units(_rows(4), RUBRIC)
    await translate_units(units, translator=RecordingTranslator(), cache=cache)
    plan = estimate(units, cache)
    assert plan.pending_units == 0
    assert plan.pending_characters == 0


@pytest.mark.asyncio
async def test_translation_is_cached_and_second_run_is_free(tmp_path: Path) -> None:
    cache = TranslationCache(path=tmp_path / "cache.json")
    units = build_units(_rows(6), RUBRIC)
    translator = RecordingTranslator()

    first = await translate_units(units, translator=translator, cache=cache, batch_size=25)
    assert first == 7

    reloaded = TranslationCache.load(tmp_path / "cache.json")
    second = await translate_units(units, translator=translator, cache=reloaded, batch_size=25)
    assert second == 0, "a completed run must not re-translate anything"


@pytest.mark.asyncio
async def test_interrupted_run_keeps_completed_batches(tmp_path: Path) -> None:

    cache = TranslationCache(path=tmp_path / "cache.json")
    units = build_units(_rows(20), RUBRIC)
    translator = RecordingTranslator(fail_after=10)

    with pytest.raises(RuntimeError):
        await translate_units(units, translator=translator, cache=cache, batch_size=5)

    resumed = TranslationCache.load(tmp_path / "cache.json")
    assert len(resumed.entries) == 10

    finisher = RecordingTranslator()
    remaining = await translate_units(units, translator=finisher, cache=resumed, batch_size=5)
    assert remaining == 11
    assert finisher.translated == 11, "resume must only pay for what was missing"


@pytest.mark.asyncio
async def test_duplicate_source_text_is_paid_for_once(tmp_path: Path) -> None:
    cache = TranslationCache(path=tmp_path / "cache.json")
    rows = _rows(3)
    rows[1]["prompt_text"] = rows[0]["prompt_text"]
    translator = RecordingTranslator()
    await translate_units(build_units(rows, RUBRIC), translator=translator, cache=cache)
    assert translator.translated == 3


@pytest.mark.asyncio
async def test_assembled_records_are_always_unverified(tmp_path: Path) -> None:
    cache = TranslationCache(path=tmp_path / "cache.json")
    rows = _rows(5, contexts=2)
    await translate_units(build_units(rows, RUBRIC), translator=RecordingTranslator(), cache=cache)
    records = assemble(rows, RUBRIC, cache)

    assert len(records) == 5
    assert {record["reviewer_status"] for record in records} == {UNVERIFIED_STATUS}
    assert {record["translation_provenance"] for record in records} == {MACHINE_PROVENANCE}
    assert all(record["human_signoff"] is None for record in records)
    assert all(record["language"] == "uk" for record in records)


@pytest.mark.asyncio
async def test_assemble_refuses_partial_translations(tmp_path: Path) -> None:
    cache = TranslationCache(path=tmp_path / "cache.json")
    rows = _rows(4)
    await translate_units(
        build_units(rows[:2], RUBRIC), translator=RecordingTranslator(), cache=cache
    )
    with pytest.raises(ValueError, match="missing translation"):
        assemble(rows, RUBRIC, cache)


@pytest.mark.asyncio
async def test_output_file_is_owner_only(tmp_path: Path) -> None:
    cache = TranslationCache(path=tmp_path / "cache.json")
    rows = _rows(3)
    await translate_units(build_units(rows, RUBRIC), translator=RecordingTranslator(), cache=cache)
    target = tmp_path / "out.json"
    digest = write_output(assemble(rows, RUBRIC, cache), path=target)
    assert len(digest) == 64
    assert target.stat().st_mode & 0o077 == 0


def test_cache_rejects_foreign_version(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"cache_version": "other", "entries": {}}))
    with pytest.raises(ValueError, match="incompatible version"):
        TranslationCache.load(path)


def test_subset_selection_is_deterministic() -> None:
    items = eligible_from_rows(_rows(320))
    first = select_subset(items)
    second = select_subset(list(reversed(items)))
    assert first.stable_ids == second.stable_ids
    assert first.plan_sha256 == second.plan_sha256
    assert len(first.stable_ids) == SUBSET_SIZE


def test_subset_selection_changes_with_seed() -> None:
    items = eligible_from_rows(_rows(320))
    assert select_subset(items).stable_ids != select_subset(items, seed=1).stable_ids


def test_subset_is_proportionally_stratified() -> None:
    items = eligible_from_rows(_rows(320))
    plan = select_subset(items)
    assert sum(plan.allocation.values()) == SUBSET_SIZE

    assert set(plan.allocation.values()) == {30}


def test_eligible_pool_excludes_validation_and_multimodal() -> None:
    rows = _rows(10)
    rows[0]["split"] = "validation"
    rows[1]["modality"] = "multimodal"
    rows[2]["language"] = "uk"
    assert len(eligible_from_rows(rows)) == 7


def test_allocation_never_exceeds_category_supply() -> None:
    assert allocate({"a": 1, "b": 99}, 50) == {"a": 1, "b": 49}


def test_selection_ignores_translation_and_outcome_fields() -> None:

    rows = _rows(320)
    baseline = select_subset(eligible_from_rows(rows))
    for row in rows:
        row["prompt_text"] = "mutated after translation"
        row["expected_behavior"] = "comply_safely"
    assert select_subset(eligible_from_rows(rows)).stable_ids == baseline.stable_ids


def test_subset_larger_than_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than the requested subset"):
        select_subset(eligible_from_rows(_rows(10)))


def test_eligible_item_requires_full_digest() -> None:
    with pytest.raises(ValueError, match="full SHA-256"):
        EligibleItem(
            stable_id="x", category="c", functional_category="standard", content_sha256="short"
        )
