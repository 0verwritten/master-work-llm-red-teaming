from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

Verdict = Literal["refused", "partial", "complied"]
VERDICTS: tuple[Verdict, ...] = ("refused", "partial", "complied")

TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)

CYRILLIC_PATTERN = re.compile(r"[Ѐ-ӿԀ-ԯ]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")

CEILING = 0.99

PERMUTATIONS = 20_000
PERMUTATION_SEED = 20260829


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


@dataclass(frozen=True)
class JudgeResponse:
    judge_pass: str
    language: str
    label: Verdict
    confidence: float | None
    rationale: str

    def __post_init__(self) -> None:
        if self.label not in VERDICTS:
            raise ValueError(f"unknown verdict {self.label!r}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie between zero and one")
        if not self.rationale.strip():
            raise ValueError("a judge response carries a rationale")

    @property
    def word_count(self) -> int:
        return len(TOKEN_PATTERN.findall(unicodedata.normalize("NFC", self.rationale)))

    @property
    def cyrillic_share(self) -> float:
        cyrillic = len(CYRILLIC_PATTERN.findall(self.rationale))
        latin = len(LATIN_PATTERN.findall(self.rationale))
        return _rate(cyrillic, cyrillic + latin)


@dataclass(frozen=True)
class ScriptAdherence:
    total: int
    median_cyrillic_share: float
    majority_cyrillic: int
    majority_cyrillic_rate: float
    mixed_script: int

    def as_json(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "median_cyrillic_share": self.median_cyrillic_share,
            "majority_cyrillic": self.majority_cyrillic,
            "majority_cyrillic_rate": self.majority_cyrillic_rate,
            "mixed_script": self.mixed_script,
        }


@dataclass(frozen=True)
class ConfidenceProfile:
    total: int
    mean: float
    median: float
    minimum: float
    at_ceiling: int
    at_ceiling_rate: float
    distinct_values: int

    @property
    def is_constant(self) -> bool:

        return self.distinct_values <= 1

    def as_json(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "mean": self.mean,
            "median": self.median,
            "minimum": self.minimum,
            "at_ceiling": self.at_ceiling,
            "at_ceiling_rate": self.at_ceiling_rate,
            "distinct_values": self.distinct_values,
            "is_constant": self.is_constant,
        }


@dataclass(frozen=True)
class LengthByLabel:
    label: str
    total: int
    mean_words: float
    median_words: float

    def as_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "total": self.total,
            "mean_words": self.mean_words,
            "median_words": self.median_words,
        }


@dataclass(frozen=True)
class DisagreementProfile:
    agreed: int
    disagreed: int
    mean_confidence_agreed: float
    mean_confidence_disagreed: float
    ceiling_rate_agreed: float
    ceiling_rate_disagreed: float
    difference: float
    permutation_p: float

    def as_json(self) -> dict[str, Any]:
        return {
            "agreed": self.agreed,
            "disagreed": self.disagreed,
            "mean_confidence_agreed": self.mean_confidence_agreed,
            "mean_confidence_disagreed": self.mean_confidence_disagreed,
            "ceiling_rate_agreed": self.ceiling_rate_agreed,
            "ceiling_rate_disagreed": self.ceiling_rate_disagreed,
            "difference": self.difference,
            "permutation_p": self.permutation_p,
        }


@dataclass(frozen=True)
class JudgeStratum:
    judge_pass: str
    language: str
    script: ScriptAdherence
    confidence: ConfidenceProfile | None
    lengths: tuple[LengthByLabel, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "judge_pass": self.judge_pass,
            "language": self.language,
            "script": self.script.as_json(),
            "confidence": None if self.confidence is None else self.confidence.as_json(),
            "lengths": [value.as_json() for value in self.lengths],
        }


@dataclass(frozen=True)
class JudgeBehaviourReport:
    strata: tuple[JudgeStratum, ...]

    def as_json(self) -> dict[str, Any]:
        return {"strata": [value.as_json() for value in self.strata]}


def script_adherence(responses: list[JudgeResponse]) -> ScriptAdherence:

    shares = [value.cyrillic_share for value in responses]
    majority = sum(1 for share in shares if share > 0.5)
    mixed = sum(1 for share in shares if 0.05 < share < 0.95)
    return ScriptAdherence(
        total=len(responses),
        median_cyrillic_share=_median(shares),
        majority_cyrillic=majority,
        majority_cyrillic_rate=_rate(majority, len(responses)),
        mixed_script=mixed,
    )


def confidence_profile(responses: list[JudgeResponse]) -> ConfidenceProfile | None:

    values = [value.confidence for value in responses if value.confidence is not None]
    if not values:
        return None
    at_ceiling = sum(1 for value in values if value >= CEILING)
    return ConfidenceProfile(
        total=len(values),
        mean=sum(values) / len(values),
        median=_median(values),
        minimum=min(values),
        at_ceiling=at_ceiling,
        at_ceiling_rate=_rate(at_ceiling, len(values)),
        distinct_values=len(set(values)),
    )


def rationale_lengths(responses: list[JudgeResponse]) -> tuple[LengthByLabel, ...]:

    grouped: dict[str, list[int]] = defaultdict(list)
    for value in responses:
        grouped[value.label].append(value.word_count)
    return tuple(
        LengthByLabel(
            label=label,
            total=len(grouped[label]),
            mean_words=sum(grouped[label]) / len(grouped[label]),
            median_words=_median([float(v) for v in grouped[label]]),
        )
        for label in VERDICTS
        if grouped[label]
    )


def permutation_difference(
    left: list[float],
    right: list[float],
    *,
    iterations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> tuple[float, float]:

    if not left or not right:
        raise ValueError("both groups need at least one observation")
    observed = abs(sum(left) / len(left) - sum(right) / len(right))
    pool = list(left) + list(right)
    size = len(left)
    generator = random.Random(seed)
    hits = 0
    for _ in range(iterations):
        generator.shuffle(pool)
        candidate = abs(sum(pool[:size]) / size - sum(pool[size:]) / (len(pool) - size))
        if candidate >= observed - 1e-12:
            hits += 1

    return observed, (hits + 1) / (iterations + 1)


def disagreement_profile(
    agreed: list[JudgeResponse], disagreed: list[JudgeResponse]
) -> DisagreementProfile:

    left = [value.confidence for value in agreed if value.confidence is not None]
    right = [value.confidence for value in disagreed if value.confidence is not None]
    if not left or not right:
        raise ValueError("both arms need at least one confidence value")
    difference, probability = permutation_difference(left, right)
    return DisagreementProfile(
        agreed=len(left),
        disagreed=len(right),
        mean_confidence_agreed=sum(left) / len(left),
        mean_confidence_disagreed=sum(right) / len(right),
        ceiling_rate_agreed=_rate(sum(1 for v in left if v >= CEILING), len(left)),
        ceiling_rate_disagreed=_rate(sum(1 for v in right if v >= CEILING), len(right)),
        difference=difference,
        permutation_p=probability,
    )


def analyse(responses: list[JudgeResponse]) -> JudgeBehaviourReport:

    if not responses:
        raise ValueError("no judge responses to analyse")
    grouped: dict[tuple[str, str], list[JudgeResponse]] = defaultdict(list)
    for value in responses:
        grouped[(value.judge_pass, value.language)].append(value)
    strata = tuple(
        JudgeStratum(
            judge_pass=judge_pass,
            language=language,
            script=script_adherence(values),
            confidence=confidence_profile(values),
            lengths=rationale_lengths(values),
        )
        for (judge_pass, language), values in sorted(grouped.items())
    )
    return JudgeBehaviourReport(strata=strata)


def label_mix(responses: list[JudgeResponse]) -> dict[str, int]:

    counts = Counter(value.label for value in responses)
    return {label: counts[label] for label in VERDICTS}
