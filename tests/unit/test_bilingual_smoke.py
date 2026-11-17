from __future__ import annotations

import pytest

from harness.bilingual_dataset import ASSUMED_VALID, HUMAN_VERIFIED, UNVERIFIED, DatasetItem
from harness.experiments.adversarial_matrix import FrozenModelProfile
from harness.experiments.bilingual_manifest import build_manifest, check_affordable
from harness.experiments.bilingual_smoke import (
    JudgeLanguageObservation,
    observation_from_evaluation,
    select_smoke_pairs,
    smoke_items,
    summarise,
)

CATEGORIES = ["copyright", "cybercrime", "illegal", "misinformation"]


def _profile(key: str) -> FrozenModelProfile:
    return FrozenModelProfile(
        key=key, provider="modal", model=None, endpoint_profile=f"{key}-v1", revision="a" * 40
    )


def _item(index: int, language: str, status: str, stratum: str = "harmbench_subset") -> DatasetItem:
    return DatasetItem(
        stable_id=f"item:{index}" + (":uk" if language == "uk" else ""),
        pair_id=f"item:{index}",
        language=language,
        stratum=stratum,
        category=CATEGORIES[index % len(CATEGORIES)],
        prompt_text=f"prompt {index} {language}",
        context_text=None,
        rubric_text=f"rubric {language}",
        expected_behavior="refuse_or_safe_redirect",
        reviewer_status=HUMAN_VERIFIED if language == "en" else status,
        translation_provenance="source_language" if language == "en" else "machine_deepl",
        license_id="MIT",
    )


def _dataset(pairs: int = 120, status: str = ASSUMED_VALID) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for index in range(pairs):
        items.append(_item(index, "en", status))
        items.append(_item(index, "uk", status))
    return items


def test_sample_is_paired_across_languages() -> None:
    items = _dataset()
    pair_ids = select_smoke_pairs(items, pairs=30)
    selected = smoke_items(items, pair_ids)
    assert len(pair_ids) == 30
    assert len(selected) == 60
    assert sum(1 for item in selected if item.language == "en") == 30
    assert sum(1 for item in selected if item.language == "uk") == 30


def test_sample_is_deterministic() -> None:
    items = _dataset()
    assert select_smoke_pairs(items, pairs=30) == select_smoke_pairs(
        list(reversed(items)), pairs=30
    )


def test_sample_changes_with_seed() -> None:
    items = _dataset()
    assert select_smoke_pairs(items, pairs=30) != select_smoke_pairs(items, pairs=30, seed=1)


def test_sample_ignores_translation_content() -> None:

    items = _dataset()
    baseline = select_smoke_pairs(items, pairs=30)
    mutated = [
        DatasetItem(**{**item.__dict__, "prompt_text": "changed"})
        if item.language == "uk"
        else item
        for item in items
    ]
    assert select_smoke_pairs(mutated, pairs=30) == baseline


def test_sample_spreads_across_categories() -> None:
    items = _dataset()
    selected = smoke_items(items, select_smoke_pairs(items, pairs=30))
    categories = {item.category for item in selected}
    assert categories == set(CATEGORIES)


def test_cells_are_ordered_by_category() -> None:
    items = _dataset()
    selected = smoke_items(items, select_smoke_pairs(items, pairs=30))
    assert [item.category for item in selected] == sorted(item.category for item in selected)


def test_incomplete_pairs_are_rejected() -> None:
    items = [item for item in _dataset() if item.language == "en"]
    with pytest.raises(ValueError, match="need 30"):
        select_smoke_pairs(items, pairs=30)


def test_manifest_refuses_unverified_ukrainian_by_default() -> None:
    items = smoke_items(_dataset(status=UNVERIFIED), select_smoke_pairs(_dataset(), pairs=30))
    with pytest.raises(ValueError, match="not verified"):
        build_manifest(
            items,
            dataset_sha256="d" * 64,
            subset_plan_sha256="p" * 64,
            attacker=_profile("attacker"),
            target=_profile("target"),
            judge=_profile("judge"),
        )


def test_manifest_accepts_assumed_only_with_opt_in() -> None:
    items = smoke_items(_dataset(), select_smoke_pairs(_dataset(), pairs=30))
    kwargs = dict(
        dataset_sha256="d" * 64,
        subset_plan_sha256="p" * 64,
        attacker=_profile("attacker"),
        target=_profile("target"),
        judge=_profile("judge"),
        max_rounds=1,
    )
    with pytest.raises(ValueError, match="allow_assumed=False"):
        build_manifest(items, **kwargs)  # type: ignore[arg-type]
    manifest = build_manifest(items, allow_assumed=True, **kwargs)  # type: ignore[arg-type]
    assert manifest.assumed_validation is True


def test_envelope_matches_cells_times_rounds_times_roles() -> None:
    items = smoke_items(_dataset(), select_smoke_pairs(_dataset(), pairs=30))
    manifest = build_manifest(
        items,
        dataset_sha256="d" * 64,
        subset_plan_sha256="p" * 64,
        attacker=_profile("attacker"),
        target=_profile("target"),
        judge=_profile("judge"),
        max_rounds=1,
        allow_assumed=True,
    )
    assert manifest.envelope.cells == 60
    assert manifest.envelope.worst_case_calls == 60 * 1 * 3
    assert manifest.envelope.worst_case_usd_high > manifest.envelope.worst_case_usd_low


def test_repeat_rejection_changes_identity_envelope_and_budget_only_when_enabled() -> None:
    from harness.experiments.bilingual_config import build_cell_config

    items = smoke_items(_dataset(), select_smoke_pairs(_dataset(), pairs=30))
    kwargs = {
        "dataset_sha256": "d" * 64,
        "subset_plan_sha256": "p" * 64,
        "attacker": _profile("attacker"),
        "target": _profile("target"),
        "judge": _profile("judge"),
        "max_rounds": 3,
        "allow_assumed": True,
    }
    original = build_manifest(items, **kwargs)  # type: ignore[arg-type]
    explicit_zero = build_manifest(items, max_attack_regenerations=0, **kwargs)  # type: ignore[arg-type]
    rejecting = build_manifest(items, max_attack_regenerations=2, **kwargs)  # type: ignore[arg-type]

    assert explicit_zero.manifest_sha256 == original.manifest_sha256
    assert "attacker_protocol" not in original.as_json()
    assert rejecting.as_json()["attacker_protocol"] == {
        "max_attack_regenerations": 2,
        "reject_repeated_attacks": True,
    }
    assert {cell.cell_id for cell in rejecting.cells}.isdisjoint(
        cell.cell_id for cell in original.cells
    )
    assert original.envelope.worst_case_calls == 60 * 3 * 3
    assert rejecting.envelope.worst_case_calls == 60 * 3 * 5
    assert build_cell_config(original, original.cells[0]).budgets.max_total_llm_calls == 9
    assert build_cell_config(rejecting, rejecting.cells[0]).budgets.max_total_llm_calls == 15


def test_local_baseline_method_is_content_addressed_and_reserves_two_calls_per_round() -> None:
    from harness.experiments.bilingual_config import build_cell_config

    items = smoke_items(_dataset(), select_smoke_pairs(_dataset(), pairs=30))
    kwargs = {
        "dataset_sha256": "d" * 64,
        "subset_plan_sha256": "p" * 64,
        "attacker": _profile("attacker"),
        "target": _profile("target"),
        "judge": _profile("judge"),
        "max_rounds": 3,
        "allow_assumed": True,
    }
    pair = build_manifest(items, **kwargs)  # type: ignore[arg-type]
    static = build_manifest(items, attack_method="static", **kwargs)  # type: ignore[arg-type]

    assert static.as_json()["attack_method"] == "static"
    assert pair.manifest_sha256 != static.manifest_sha256
    assert {cell.cell_id for cell in pair.cells}.isdisjoint(cell.cell_id for cell in static.cells)
    assert static.envelope.worst_case_calls == 60 * 3 * 2
    assert build_cell_config(static, static.cells[0]).budgets.max_total_llm_calls == 6


def test_repeat_rejection_is_refused_for_non_pair_method() -> None:
    items = smoke_items(_dataset(), select_smoke_pairs(_dataset(), pairs=30))
    with pytest.raises(ValueError, match="only for pair"):
        build_manifest(
            items,
            dataset_sha256="d" * 64,
            subset_plan_sha256="p" * 64,
            attacker=_profile("attacker"),
            target=_profile("target"),
            judge=_profile("judge"),
            max_attack_regenerations=2,
            attack_method="single_shot",
            allow_assumed=True,
        )


def test_unaffordable_batch_is_refused() -> None:
    items = smoke_items(_dataset(), select_smoke_pairs(_dataset(), pairs=30))
    manifest = build_manifest(
        items,
        dataset_sha256="d" * 64,
        subset_plan_sha256="p" * 64,
        attacker=_profile("attacker"),
        target=_profile("target"),
        judge=_profile("judge"),
        max_rounds=1,
        allow_assumed=True,
    )
    with pytest.raises(ValueError, match="exceeds the remaining"):
        check_affordable(manifest, remaining_usd=1.0)
    check_affordable(manifest, remaining_usd=100.0)


def test_manifest_identity_changes_with_the_sample() -> None:
    dataset = _dataset()
    common = dict(
        dataset_sha256="d" * 64,
        subset_plan_sha256="p" * 64,
        attacker=_profile("attacker"),
        target=_profile("target"),
        judge=_profile("judge"),
        max_rounds=1,
        allow_assumed=True,
    )
    a = build_manifest(smoke_items(dataset, select_smoke_pairs(dataset, pairs=30)), **common)  # type: ignore[arg-type]
    b = build_manifest(smoke_items(dataset, select_smoke_pairs(dataset, pairs=20)), **common)  # type: ignore[arg-type]
    assert a.manifest_sha256 != b.manifest_sha256


def _obs(language: str, adherence: str, index: int = 0) -> JudgeLanguageObservation:
    return JudgeLanguageObservation(
        cell_id=f"c{index}", language=language, category="illegal", adherence=adherence
    )


def test_verdict_is_go_when_ukrainian_holds() -> None:
    report = summarise([_obs("uk", "pass", i) for i in range(30)])
    assert report.adherence_rate["uk"] == 1.0
    assert report.verdict == "go"


def test_verdict_reproduces_the_adr_0020_no_go() -> None:

    report = summarise([_obs("uk", "fail", i) for i in range(6)])
    assert report.adherence_rate["uk"] == 0.0
    assert report.verdict == "no_go_judge_cannot_sustain_ukrainian"


def test_marginal_band_defers_to_the_operator() -> None:
    observations = [_obs("uk", "pass", i) for i in range(17)] + [
        _obs("uk", "fail", 90 + i) for i in range(3)
    ]
    assert summarise(observations).verdict == "marginal_operator_decision"


def test_indeterminate_never_counts_as_a_pass() -> None:
    report = summarise([_obs("uk", "indeterminate", i) for i in range(10)])
    assert report.adherence_rate["uk"] == 0.0
    assert report.by_language["uk"]["indeterminate"] == 10


def test_languages_are_reported_separately() -> None:
    report = summarise(
        [_obs("en", "pass", i) for i in range(5)] + [_obs("uk", "fail", 50 + i) for i in range(5)]
    )
    assert report.adherence_rate == {"en": 1.0, "uk": 0.0}


def test_english_only_report_does_not_claim_a_ukrainian_failure() -> None:
    report = summarise([_obs("en", "pass", i) for i in range(3)])
    assert report.verdict == "not_applicable_ukrainian_not_sampled"


def test_persisted_diagnostic_is_preferred_over_recomputation() -> None:
    observation = observation_from_evaluation(
        cell_id="c1",
        language="uk",
        category="illegal",
        rationale="English text that would otherwise fail",
        raw_json={"judge_language_adherence": "pass"},
    )
    assert observation.adherence == "pass"


def test_missing_diagnostic_falls_back_to_recomputation() -> None:
    observation = observation_from_evaluation(
        cell_id="c1",
        language="uk",
        category="illegal",
        rationale="English text with no Cyrillic at all",
        raw_json=None,
    )
    assert observation.adherence == "fail"
