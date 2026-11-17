from __future__ import annotations

import pytest

from harness.bilingual_dataset import (
    ASSUMED_VALID,
    HUMAN_VERIFIED,
    UNVERIFIED,
    build,
    require_verified,
)

RUBRIC_EN = "English rubric."
RUBRIC_UK = "Український рубрикатор."
BENIGN_RUBRIC_EN = "English benign rubric."


def _english(count: int) -> list[dict[str, object]]:
    return [
        {
            "stable_id": f"harmbench:test:{index}",
            "prompt_text": f"english {index}",
            "context_text": f"context {index}" if index % 3 == 0 else None,
            "category": ["cybercrime", "copyright"][index % 2],
            "content_sha256": f"{index:064d}",
            "split": "test",
            "modality": "text",
            "language": "en",
        }
        for index in range(count)
    ]


def _ukrainian(count: int) -> list[dict[str, object]]:
    return [
        {
            "source_stable_id": f"harmbench:test:{index}",
            "prompt_text": f"українська {index}",
            "context_text": f"контекст {index}" if index % 3 == 0 else None,
            "rubric_text": RUBRIC_UK,
        }
        for index in range(count)
    ]


def _benign_en(count: int = 4) -> list[dict[str, object]]:
    return [
        {
            "stable_id": f"legacy-benign:adversarial_pairs:p{index}",
            "prompt_text": f"benign {index}",
            "category": "violence",
        }
        for index in range(count)
    ]


def _benign_uk(count: int = 4) -> list[dict[str, object]]:
    return [
        {
            "source_stable_id": f"legacy-benign:adversarial_pairs:p{index}",
            "prompt_text": f"безпечний {index}",
            "rubric_text": RUBRIC_UK,
        }
        for index in range(count)
    ]


def _build(count: int = 6, status: str = UNVERIFIED):
    english = _english(count)
    return build(
        english_rows=english,
        english_rubric=RUBRIC_EN,
        ukrainian_records=_ukrainian(count),
        subset_stable_ids=[str(row["stable_id"]) for row in english],
        subset_plan_sha256="a" * 64,
        benign_english_rows=_benign_en(),
        benign_english_rubric=BENIGN_RUBRIC_EN,
        benign_ukrainian_records=_benign_uk(),
        ukrainian_status=status,
    )


def test_every_item_is_paired_across_languages() -> None:
    dataset = _build(6)
    assert len(dataset.items) == 20
    assert len(dataset.pair_ids) == 10
    assert len(dataset.by_language("en")) == len(dataset.by_language("uk")) == 10


def test_pairs_share_category_and_expected_behaviour() -> None:
    dataset = _build(6)
    for pair_id in dataset.pair_ids:
        sides = [item for item in dataset.items if item.pair_id == pair_id]
        assert len({item.category for item in sides}) == 1
        assert len({item.expected_behavior for item in sides}) == 1


def test_benign_controls_expect_safe_compliance() -> None:
    dataset = _build(6)
    benign = [item for item in dataset.items if item.stratum == "benign_control"]
    assert len(benign) == 8
    assert {item.expected_behavior for item in benign} == {"comply_safely"}


def test_harmful_and_benign_strata_are_distinguishable() -> None:
    dataset = _build(6)
    manifest = dataset.manifest()
    assert manifest["by_stratum"] == {"benign_control": 8, "harmbench_subset": 12}


def test_manifest_counts_only_ukrainian_verification() -> None:

    manifest = _build(6, status=ASSUMED_VALID).manifest()
    assert manifest["ukrainian_item_count"] == 10
    assert manifest["ukrainian_human_verified_count"] == 0
    assert manifest["assumed_validation"] is True


def test_assumed_status_is_recorded_in_the_manifest_identity() -> None:

    assumed = _build(6, status=ASSUMED_VALID).manifest()
    unverified = _build(6, status=UNVERIFIED).manifest()
    assert assumed["dataset_sha256"] != unverified["dataset_sha256"]
    assert unverified["assumed_validation"] is False


def test_guard_rejects_unverified_by_default() -> None:
    dataset = _build(6, status=UNVERIFIED)
    with pytest.raises(ValueError, match="not verified"):
        require_verified(list(dataset.items))


def test_guard_rejects_assumed_without_opt_in() -> None:
    dataset = _build(6, status=ASSUMED_VALID)
    with pytest.raises(ValueError, match="allow_assumed=False"):
        require_verified(list(dataset.items))


def test_guard_accepts_assumed_only_with_explicit_opt_in() -> None:
    dataset = _build(6, status=ASSUMED_VALID)
    require_verified(list(dataset.items), allow_assumed=True)


def test_guard_never_accepts_plain_unverified_even_with_opt_in() -> None:
    dataset = _build(6, status=UNVERIFIED)
    with pytest.raises(ValueError, match="not verified"):
        require_verified(list(dataset.items), allow_assumed=True)


def test_guard_passes_when_genuinely_verified() -> None:
    require_verified(list(_build(6, status=HUMAN_VERIFIED).items), allow_assumed=False)


def test_missing_translation_is_rejected() -> None:
    english = _english(6)
    with pytest.raises(ValueError, match="Ukrainian translation missing"):
        build(
            english_rows=english,
            english_rubric=RUBRIC_EN,
            ukrainian_records=_ukrainian(3),
            subset_stable_ids=[str(row["stable_id"]) for row in english],
            subset_plan_sha256="a" * 64,
            benign_english_rows=_benign_en(),
            benign_english_rubric=BENIGN_RUBRIC_EN,
            benign_ukrainian_records=_benign_uk(),
        )


def test_missing_benign_translation_is_rejected() -> None:
    english = _english(4)
    with pytest.raises(ValueError, match="benign translation missing"):
        build(
            english_rows=english,
            english_rubric=RUBRIC_EN,
            ukrainian_records=_ukrainian(4),
            subset_stable_ids=[str(row["stable_id"]) for row in english],
            subset_plan_sha256="a" * 64,
            benign_english_rows=_benign_en(4),
            benign_english_rubric=BENIGN_RUBRIC_EN,
            benign_ukrainian_records=_benign_uk(2),
        )


def test_duplicate_subset_ids_are_rejected() -> None:
    english = _english(4)
    ids = [str(row["stable_id"]) for row in english]
    with pytest.raises(ValueError, match="duplicate"):
        build(
            english_rows=english,
            english_rubric=RUBRIC_EN,
            ukrainian_records=_ukrainian(4),
            subset_stable_ids=ids + ids[:1],
            subset_plan_sha256="a" * 64,
            benign_english_rows=_benign_en(),
            benign_english_rubric=BENIGN_RUBRIC_EN,
            benign_ukrainian_records=_benign_uk(),
        )


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown reviewer status"):
        _build(4, status="approved")


def test_ukrainian_items_never_claim_source_language_provenance() -> None:
    dataset = _build(6, status=ASSUMED_VALID)
    for item in dataset.by_language("uk"):
        assert item.translation_provenance == "machine_deepl"
        assert item.reviewer_status != HUMAN_VERIFIED
