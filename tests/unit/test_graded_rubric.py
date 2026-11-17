from __future__ import annotations

import pytest

from harness.experiments.graded_rubric import (
    GRADED_RUBRIC_VERSION,
    rubric_for,
    rubric_sha256,
)


def test_all_three_levels_are_defined_in_both_languages() -> None:

    for language in ("en", "uk"):
        text = rubric_for(language)
        for level in ("refused", "partial", "complied"):
            assert level in text, f"{level} undefined in {language}"


def test_partial_gets_a_decision_test_not_just_a_name() -> None:

    assert "meaningfully closer" in rubric_for("en")
    assert "відчутно ближчим" in rubric_for("uk")


def test_label_tokens_stay_english_in_the_ukrainian_rubric() -> None:

    text = rubric_for("uk")
    assert "англійським токеном" in text


def test_tone_is_explicitly_excluded_from_the_decision() -> None:

    assert "not by its tone" in rubric_for("en")
    assert "не за її тоном" in rubric_for("uk")


def test_rubric_identity_is_content_addressed() -> None:
    assert len(rubric_sha256()) == 64
    assert rubric_sha256() == rubric_sha256()


def test_unknown_language_is_rejected() -> None:
    with pytest.raises(ValueError, match="no graded rubric"):
        rubric_for("de")


def test_version_is_recorded() -> None:
    assert GRADED_RUBRIC_VERSION == "graded-three-level-v1"


def test_both_languages_define_the_levels_in_the_same_order() -> None:

    for language in ("en", "uk"):
        text = rubric_for(language)
        positions = [text.index(level) for level in ("refused", "partial", "complied")]
        assert positions == sorted(positions), f"{language} defines levels out of order"


def test_both_languages_carry_the_same_decision_criteria() -> None:

    english, ukrainian = rubric_for("en"), rubric_for("uk")

    assert "withholding" in english and "приховуючи" in ukrainian
    assert "caveated" in english and "застережній" in ukrainian
    assert "begins to comply and stops" in english and "починає виконувати" in ukrainian
