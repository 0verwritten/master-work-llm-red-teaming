# ruff: noqa: RUF001  # Intentional Ukrainian rationale text.

from __future__ import annotations

from harness.experiments.rejudge import REJUDGE_VERSION, strip_code_fence, summarise


def test_unwraps_gemma_style_fenced_json() -> None:

    fenced = '```json\n{"label": "refused", "rationale": "бо", "confidence": 1}\n```'
    assert strip_code_fence(fenced).startswith("{")
    assert strip_code_fence(fenced).endswith("}")


def test_unwraps_a_bare_fence() -> None:
    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_leaves_unfenced_json_untouched() -> None:

    bare = '{"label": "complied", "rationale": "x", "confidence": 0.9}'
    assert strip_code_fence(bare) == bare


def test_does_not_mangle_json_containing_backticks() -> None:
    body = '{"rationale": "use ``` in code"}'
    assert strip_code_fence(body) == body


def test_summary_reports_partial_counts_per_language() -> None:

    records = [
        {"language": "en", "original_label": "partial", "rejudge_label": "partial", "agree": True},
        {"language": "en", "original_label": "refused", "rejudge_label": "refused", "agree": True},
        {"language": "uk", "original_label": "refused", "rejudge_label": "partial", "agree": False},
        {"language": "uk", "original_label": "refused", "rejudge_label": "refused", "agree": True},
    ]
    out = summarise(records)  # type: ignore[arg-type]
    assert out["rejudge_version"] == REJUDGE_VERSION
    assert out["en"]["original_partial"] == 1
    assert out["uk"]["original_partial"] == 0
    assert out["uk"]["rejudge_partial"] == 1, "a second judge finding partial in uk is the signal"
    assert out["uk"]["agreement"] == 0.5


def test_summary_handles_a_single_language() -> None:
    records = [
        {"language": "en", "original_label": "refused", "rejudge_label": "refused", "agree": True}
    ]
    out = summarise(records)  # type: ignore[arg-type]
    assert out["total"] == 1
    assert "uk" not in out
