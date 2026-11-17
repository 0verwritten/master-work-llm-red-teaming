from __future__ import annotations

from typing import Any

import pytest

from harness.legacy_audit import audit_payloads


def _row(
    *,
    item_id: str,
    repetition: int,
    text_refused: bool,
    text_verdict: str,
    image_refused: bool,
    image_verdict: str,
    same_reason: bool,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "repetition": repetition,
        "model": "recorded-model",
        "source": "fixture",
        "category": "safe_fixture",
        "timestamp": f"2026-01-01T00:00:0{repetition}.000000",
        "text_refused": text_refused,
        "self_judge_text_verdict": text_verdict,
        "image_refused": image_refused,
        "self_judge_image_verdict": image_verdict,
        "self_judge_text_reason": "same" if same_reason else "text",
        "self_judge_image_reason": "same" if same_reason else "image",
    }


def test_audit_payloads_recomputes_counts_and_percentages() -> None:
    rows = [
        _row(
            item_id="one",
            repetition=1,
            text_refused=True,
            text_verdict="REFUSED",
            image_refused=False,
            image_verdict="PARTIAL",
            same_reason=True,
        ),
        _row(
            item_id="one",
            repetition=2,
            text_refused=True,
            text_verdict="COMPLIED",
            image_refused=True,
            image_verdict="REFUSED",
            same_reason=False,
        ),
    ]

    result = audit_payloads("extended-raw_results-file-claim.json", rows)

    assert result["filename_model_claim"] == "file-claim"
    assert result["record_count"] == 2
    assert result["unique_item_count"] == 1
    assert result["unique_item_repetition_count"] == 2
    assert result["item_repetition_multiplicity_counts"] == {"1": 2}
    assert result["repetitions"] == [1, 2]
    assert result["text_heuristic"]["implied_asr_pct"] == 0.0
    assert result["image_heuristic"]["implied_asr_pct"] == 50.0
    assert result["image_judge_replica_1"]["verdict_counts"] == {
        "REFUSED": 1,
        "PARTIAL": 1,
        "COMPLIED": 0,
    }
    assert result["image_judge_replica_1"]["strict_asr_pct"] == 0.0
    assert result["image_judge_replica_1"]["lenient_asr_pct"] == 50.0
    assert result["image_judge_replica_1"]["image_heuristic_agreement_pct"] == 100.0
    assert result["image_judge_replica_2_misnamed_text"]["strict_asr_pct"] == 50.0
    assert result["matching_image_judge_verdict_pct"] == 0.0
    assert result["identical_image_judge_reason_pct"] == 50.0


def test_audit_payloads_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        audit_payloads("empty.json", [])
