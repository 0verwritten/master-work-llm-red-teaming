from __future__ import annotations

from typing import Any

from harness.spend.ledger import SpendEntry, SpendLedger

HISTORICAL_ROWS: tuple[dict[str, Any], ...] = (
    {
        "provider": "modal",
        "account": "window-1",
        "kind": "call_batch",
        "basis": "observed",
        "amount_usd": 2.88664720,
        "reference": "modal:qwen-serving-smoke",
        "purpose": "Pinned Qwen candidate deployment and one benign serving smoke",
        "evidence": "docs/thesis/modal-serving-smoke-plan.md; ADR-0013",
    },
    {
        "provider": "modal",
        "account": "window-1",
        "kind": "call_batch",
        "basis": "observed",
        "amount_usd": 2.16502838,
        "reference": "modal:judge-candidate-smokes",
        "purpose": "Gemma 4, Mistral 3.2, and GPT-OSS candidate deployability smokes",
        "evidence": "docs/thesis/model-selection-serving-plan.md; ADR-0015",
    },
    {
        "provider": "modal",
        "account": "window-1",
        "kind": "call_batch",
        "basis": "observed",
        "amount_usd": 8.85567294,
        "reference": "modal:phase3-bilingual-preflight",
        "purpose": "Phase 3 bilingual preflight and abort; 63 persisted calls",
        "evidence": "ADR-0020; ADR-0023; result-phase3.html",
    },
    {
        "provider": "modal",
        "account": "window-1",
        "kind": "call_batch",
        "basis": "observed",
        "amount_usd": 0.57440696,
        "reference": "modal:phase4-self-judge-census",
        "purpose": "Phase 4 target self-judge census over 21 frozen responses",
        "evidence": "ADR-0021; result-phase4.html",
    },
    {
        "provider": "modal",
        "account": "window-1-mknssh-2025",
        "kind": "call_batch",
        "basis": "estimated",
        "amount_usd": 3.9631,
        "reference": "modal:judge-language-smoke-partial",
        "purpose": "Judge-language smoke, first 35 of 60 cells; stopped by spend limit",
        "evidence": "ADR-0026; docs/thesis/judge-language-smoke-result.md",
    },
    {
        "provider": "modal",
        "account": "window-2-kn-2021",
        "kind": "call_batch",
        "basis": "observed",
        "amount_usd": 17.95,
        "reference": "modal:judge-language-smoke-complete-and-bilingual-start",
        "purpose": "Judge-language smoke completed 60/60, then bilingual run to 30 cells",
        "evidence": "docs/thesis/judge-language-smoke-result.md",
    },
    {
        "provider": "modal",
        "account": "window-3-jadovirlyvyjlox",
        "kind": "call_batch",
        "basis": "estimated",
        "amount_usd": 32.87,
        "reference": "modal:bilingual-half-dataset-partial",
        "purpose": "Bilingual half-dataset run to 104 of 140 cells",
        "evidence": "docs/thesis/results-bilingual-half-dataset.md",
    },
    {
        "provider": "modal",
        "account": "window-4-passmaak",
        "kind": "call_batch",
        "basis": "estimated",
        "amount_usd": 41.33,
        "reference": "modal:bilingual-half-dataset-complete",
        "supersedes": "modal:bilingual-half-dataset-partial",
        "purpose": "Bilingual half-dataset run completed 140/140 cells, 70/70 matched pairs",
        "evidence": "docs/thesis/results-bilingual-half-dataset.md",
    },
    {
        "provider": "modal",
        "account": "window-4-passmaak",
        "kind": "call_batch",
        "basis": "estimated",
        "amount_usd": 2.59,
        "reference": "modal:second-judge-rescoring",
        "purpose": "Second-judge rescoring, 309 distinct responses across two sessions",
        "evidence": "docs/thesis/results-second-judge-rescoring.md",
    },
    {
        "provider": "modal",
        "account": "window-4-passmaak",
        "kind": "call_batch",
        "basis": "estimated",
        "amount_usd": 4.96,
        "reference": "modal:graded-rubric-rescoring",
        "purpose": "Graded-rubric rescoring, 337 responses, 389/390 attempts succeeded",
        "evidence": "docs/thesis/results-graded-rubric-rescoring.md",
    },
    {
        "provider": "modal",
        "account": "window-1",
        "kind": "adjustment",
        "basis": "estimated",
        "amount_usd": 0.0,
        "reference": "modal:preliminary-multimodal-study",
        "purpose": "Corrected preliminary multimodal study, 2,112 persisted calls",
        "evidence": (
            "docs/thesis/preliminary-study-rerun-plan.md; amount unreconciled, see detail.problem"
        ),
        "detail": {
            "problem": (
                "Per-record estimated_cost_usd sums to roughly USD 12, but that field "
                "multiplies each request's latency by the GPU rate and counts overlapping "
                "GPU-seconds up to eight times at client concurrency 8. Billed cost follows "
                "container wall-clock and is substantially lower. Recorded at 0.00 until "
                "reconciled against the Modal dashboard."
            ),
            "unreconciled": True,
        },
    },
)


def backfill(ledger: SpendLedger, *, dry_run: bool = False) -> dict[str, Any]:

    existing = {entry.reference for entry in ledger.entries()}
    appended: list[str] = []
    skipped: list[str] = []
    for row in HISTORICAL_ROWS:
        entry = SpendEntry.model_validate(row)
        if entry.reference in existing:
            skipped.append(entry.reference)
            continue
        if not dry_run:
            ledger.append(entry)
        appended.append(entry.reference)
    return {
        "dry_run": dry_run,
        "appended": appended,
        "already_present": skipped,
        "totals": ledger.totals().model_dump(mode="json"),
    }
