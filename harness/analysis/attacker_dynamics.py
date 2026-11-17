from __future__ import annotations

import itertools
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import comb
from typing import Any, Literal

Verdict = Literal["refused", "partial", "complied"]
VERDICTS: tuple[Verdict, ...] = ("refused", "partial", "complied")

TOKENIZER_VERSION = "unicode-nfc-casefold-alnum-v1"
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)

NON_REFUSAL: frozenset[str] = frozenset({"partial", "complied"})


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(unicodedata.normalize("NFC", text or "").casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


@dataclass(frozen=True)
class RoundObservation:
    session_id: str
    round_number: int
    language: str
    attack_text: str
    verdict: Verdict | None

    def __post_init__(self) -> None:
        if self.round_number < 1:
            raise ValueError("round numbers are one-based")
        if self.verdict is not None and self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}")


@dataclass(frozen=True)
class NoveltySummary:
    transitions: int
    mean_similarity: float
    median_similarity: float
    exact_repeat_transitions: int

    def as_json(self) -> dict[str, Any]:
        return {
            "transitions": self.transitions,
            "mean_similarity": self.mean_similarity,
            "median_similarity": self.median_similarity,
            "exact_repeat_transitions": self.exact_repeat_transitions,
        }


@dataclass(frozen=True)
class RoundPoint:
    round_number: int
    total: int
    verdicts: dict[str, int]
    non_refusal_rate: float

    def as_json(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "total": self.total,
            "verdicts": dict(sorted(self.verdicts.items())),
            "non_refusal_rate": self.non_refusal_rate,
        }


@dataclass(frozen=True)
class EscalationSummary:
    complete_sessions: int
    rounds_required: int
    points: tuple[RoundPoint, ...]
    first_round_only: int
    last_round_only: int
    both_rounds: int
    neither_round: int
    mcnemar_p: float

    def as_json(self) -> dict[str, Any]:
        return {
            "complete_sessions": self.complete_sessions,
            "rounds_required": self.rounds_required,
            "points": [point.as_json() for point in self.points],
            "first_round_only": self.first_round_only,
            "last_round_only": self.last_round_only,
            "both_rounds": self.both_rounds,
            "neither_round": self.neither_round,
            "mcnemar_p": self.mcnemar_p,
        }


@dataclass(frozen=True)
class RepetitionSummary:
    sessions: int
    with_repeat: int
    repeat_rate: float
    success_with_repeat: int
    success_without_repeat: int

    def as_json(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "with_repeat": self.with_repeat,
            "repeat_rate": self.repeat_rate,
            "success_with_repeat": self.success_with_repeat,
            "success_without_repeat": self.success_without_repeat,
        }


@dataclass(frozen=True)
class LanguageStratum:
    language: str
    novelty: NoveltySummary
    escalation: EscalationSummary
    repetition: RepetitionSummary

    def as_json(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "novelty": self.novelty.as_json(),
            "escalation": self.escalation.as_json(),
            "repetition": self.repetition.as_json(),
        }


@dataclass(frozen=True)
class AttackerDynamicsReport:
    tokenizer_version: str
    rounds_required: int
    strata: dict[str, LanguageStratum]

    def as_json(self) -> dict[str, Any]:
        return {
            "tokenizer_version": self.tokenizer_version,
            "rounds_required": self.rounds_required,
            "strata": {name: value.as_json() for name, value in sorted(self.strata.items())},
        }


def exact_mcnemar(first_only: int, second_only: int) -> float:

    total = first_only + second_only
    if total == 0:
        return 1.0
    smaller = min(first_only, second_only)
    tail: float = sum(comb(total, k) for k in range(smaller + 1)) / (2**total)
    return min(1.0, 2 * tail)


def _novelty(sessions: dict[str, list[RoundObservation]]) -> NoveltySummary:
    similarities: list[float] = []
    exact = 0
    for observations in sessions.values():
        ordered = sorted(observations, key=lambda value: value.round_number)
        for previous, current in itertools.pairwise(ordered):
            if previous.attack_text == current.attack_text:
                exact += 1
            similarities.append(
                _jaccard(_tokens(previous.attack_text), _tokens(current.attack_text))
            )
    similarities.sort()
    if not similarities:
        return NoveltySummary(0, 0.0, 0.0, 0)
    return NoveltySummary(
        transitions=len(similarities),
        mean_similarity=sum(similarities) / len(similarities),
        median_similarity=similarities[len(similarities) // 2],
        exact_repeat_transitions=exact,
    )


def _escalation(
    sessions: dict[str, list[RoundObservation]], rounds_required: int
) -> EscalationSummary:
    complete = {
        session_id: observations
        for session_id, observations in sessions.items()
        if sorted(value.round_number for value in observations)
        == list(range(1, rounds_required + 1))
    }
    points: list[RoundPoint] = []
    for number in range(1, rounds_required + 1):
        counts: Counter[str] = Counter()
        for observations in complete.values():
            for value in observations:
                if value.round_number == number and value.verdict is not None:
                    counts[value.verdict] += 1
        total = sum(counts.values())
        non_refusal = total - counts["refused"]
        points.append(
            RoundPoint(
                round_number=number,
                total=total,
                verdicts={verdict: counts[verdict] for verdict in VERDICTS},
                non_refusal_rate=_rate(non_refusal, total),
            )
        )

    first_only = last_only = both = neither = 0
    for observations in complete.values():
        by_round = {value.round_number: value.verdict for value in observations}
        first = by_round.get(1) in NON_REFUSAL
        last = by_round.get(rounds_required) in NON_REFUSAL
        if first and not last:
            first_only += 1
        elif last and not first:
            last_only += 1
        elif first and last:
            both += 1
        else:
            neither += 1

    return EscalationSummary(
        complete_sessions=len(complete),
        rounds_required=rounds_required,
        points=tuple(points),
        first_round_only=first_only,
        last_round_only=last_only,
        both_rounds=both,
        neither_round=neither,
        mcnemar_p=exact_mcnemar(first_only, last_only),
    )


def _repetition(sessions: dict[str, list[RoundObservation]]) -> RepetitionSummary:
    with_repeat = success_with = success_without = considered = 0
    for observations in sessions.values():
        if len(observations) < 2:
            continue
        considered += 1
        texts = [value.attack_text for value in observations]
        repeated = len(set(texts)) < len(texts)
        succeeded = any(value.verdict in NON_REFUSAL for value in observations)
        if repeated:
            with_repeat += 1
            success_with += int(succeeded)
        else:
            success_without += int(succeeded)
    return RepetitionSummary(
        sessions=considered,
        with_repeat=with_repeat,
        repeat_rate=_rate(with_repeat, considered),
        success_with_repeat=success_with,
        success_without_repeat=success_without,
    )


def analyse(
    observations: list[RoundObservation], *, rounds_required: int = 3
) -> AttackerDynamicsReport:

    if rounds_required < 2:
        raise ValueError("attacker dynamics need at least two rounds to compare")
    by_language: dict[str, dict[str, list[RoundObservation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for value in observations:
        by_language[value.language][value.session_id].append(value)

    strata = {
        language: LanguageStratum(
            language=language,
            novelty=_novelty(sessions),
            escalation=_escalation(sessions, rounds_required),
            repetition=_repetition(sessions),
        )
        for language, sessions in by_language.items()
    }
    return AttackerDynamicsReport(
        tokenizer_version=TOKENIZER_VERSION,
        rounds_required=rounds_required,
        strata=strata,
    )
