from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from harness.agents.native.adversarial import AdversarialRunner, ExperimentLanguage
from harness.agents.native.gateway import TracedModelGateway
from harness.language import detect_language_adherence
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    Evaluation,
    Run,
)
from harness.persistence.repository import TraceRepository

REJUDGE_VERSION = "rejudge-v1"

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_code_fence(content: str) -> str:

    match = _FENCE.match(content)
    return match.group(1) if match else content


@dataclass(frozen=True)
class FrozenResponse:
    round_id: Any
    cell_id: str
    language: str
    rubric_text: str
    benchmark_prompt: str
    target_response: str
    original_label: str
    original_adherence: str | None
    original_judge_profile: str

    @property
    def response_sha256(self) -> str:
        return hashlib.sha256(self.target_response.encode("utf-8")).hexdigest()


async def load_frozen_responses(
    repository: TraceRepository, prompts_by_cell: dict[str, str]
) -> list[FrozenResponse]:

    rows = (
        await repository.session.execute(
            select(AdversarialSession, AdversarialRound, Evaluation)
            .join(AdversarialRound, AdversarialRound.session_id == AdversarialSession.id)
            .join(Evaluation, Evaluation.round_id == AdversarialRound.id)
            .order_by(AdversarialRound.round_number)
        )
    ).all()

    frozen: list[FrozenResponse] = []
    for session, round_, evaluation in rows:
        cell_id = str((session.spec_json or {}).get("cell_id", ""))
        prompt = prompts_by_cell.get(cell_id)
        if not cell_id or prompt is None or not round_.target_response:
            continue
        frozen.append(
            FrozenResponse(
                round_id=round_.id,
                cell_id=cell_id,
                language=str(session.language),
                rubric_text=str(session.rubric_text),
                benchmark_prompt=prompt,
                target_response=str(round_.target_response),
                original_label=str(evaluation.label),
                original_adherence=(evaluation.raw_json or {}).get("judge_language_adherence"),
                original_judge_profile=str(evaluation.judge_endpoint_profile),
            )
        )
    return frozen


async def rescore(
    frozen: FrozenResponse,
    *,
    gateway: TracedModelGateway,
    run: Run,
    repository: TraceRepository,
    temperature: float = 0.0,
    step_index: int = 0,
    rubric_override: str | None = None,
) -> dict[str, Any]:

    language = ExperimentLanguage(frozen.language)
    messages = AdversarialRunner._judge_messages(
        benchmark_prompt=frozen.benchmark_prompt,
        target_response=frozen.target_response,
        rubric=rubric_override or frozen.rubric_text,
        language=language,
    )
    step = await repository.create_step(
        run_id=run.id,
        sequence_number=step_index,
        step_type="judge",
        purpose="rejudge",
    )
    response = await gateway.request(
        run_id=run.id,
        step_id=step.id,
        purpose="rejudge",
        messages=messages,
        tools=[],
        parameters={"temperature": temperature},
    )
    decision, error = AdversarialRunner._parse_judge(strip_code_fence(response.content))
    label = decision.label if decision else "parse_error"
    rationale = decision.rationale if decision else str(error)
    adherence = (
        detect_language_adherence(rationale, frozen.language) if decision else "indeterminate"
    )

    await repository.session.commit()

    return {
        "cell_id": frozen.cell_id,
        "language": frozen.language,
        "response_sha256": frozen.response_sha256,
        "original_label": frozen.original_label,
        "rejudge_label": label,
        "agree": label == frozen.original_label,
        "judge_language_adherence": adherence,
        "estimated_cost_usd": response.estimated_cost_usd,
    }


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:

    out: dict[str, Any] = {"rejudge_version": REJUDGE_VERSION, "total": len(records)}
    for language in ("en", "uk"):
        subset = [r for r in records if r["language"] == language]
        if not subset:
            continue
        original: dict[str, int] = {}
        rejudged: dict[str, int] = {}
        for record in subset:
            original[record["original_label"]] = original.get(record["original_label"], 0) + 1
            rejudged[record["rejudge_label"]] = rejudged.get(record["rejudge_label"], 0) + 1
        agree = sum(1 for record in subset if record["agree"])
        out[language] = {
            "n": len(subset),
            "original_labels": dict(sorted(original.items())),
            "rejudge_labels": dict(sorted(rejudged.items())),
            "agreement": round(agree / len(subset), 4),
            "original_partial": original.get("partial", 0),
            "rejudge_partial": rejudged.get("partial", 0),
        }
    return out
