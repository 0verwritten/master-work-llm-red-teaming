from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

Verdict = Literal["refused", "partial", "complied"]
VERDICTS: tuple[Verdict, ...] = ("refused", "partial", "complied")

BOOTSTRAP_ITERATIONS = 20_000

BOOTSTRAP_SEED = 20260829


@dataclass(frozen=True)
class ConcordanceReport:
    total: int
    observed_agreement: float
    expected_agreement: float
    kappa: float
    kappa_ci_low: float
    kappa_ci_high: float
    row_marginals: dict[str, int]
    column_marginals: dict[str, int]

    def as_json(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "observed_agreement": self.observed_agreement,
            "expected_agreement": self.expected_agreement,
            "kappa": self.kappa,
            "kappa_ci_low": self.kappa_ci_low,
            "kappa_ci_high": self.kappa_ci_high,
            "row_marginals": dict(sorted(self.row_marginals.items())),
            "column_marginals": dict(sorted(self.column_marginals.items())),
        }


def _validate(matrix: list[list[int]]) -> None:
    size = len(matrix)
    if size < 2:
        raise ValueError("a confusion matrix needs at least two categories")
    if any(len(row) != size for row in matrix):
        raise ValueError("the confusion matrix must be square")
    if any(value < 0 for row in matrix for value in row):
        raise ValueError("confusion counts cannot be negative")
    if sum(sum(row) for row in matrix) == 0:
        raise ValueError("the confusion matrix is empty")


def agreement(matrix: list[list[int]]) -> tuple[float, float, float]:

    _validate(matrix)
    size = len(matrix)
    total = sum(sum(row) for row in matrix)
    observed = sum(matrix[index][index] for index in range(size)) / total
    rows = [sum(row) for row in matrix]
    columns = [sum(matrix[index][column] for index in range(size)) for column in range(size)]
    expected = sum(rows[index] * columns[index] for index in range(size)) / (total * total)
    if expected == 1.0:
        raise ValueError("kappa is undefined when chance agreement is total")
    return observed, expected, (observed - expected) / (1 - expected)


def collapse_to_binary(matrix: list[list[int]]) -> list[list[int]]:

    _validate(matrix)
    if len(matrix) != 3:
        raise ValueError("collapsing expects the three-level verdict scale")
    held = matrix[0][0]
    row_broke = matrix[0][1] + matrix[0][2]
    column_broke = matrix[1][0] + matrix[2][0]
    both_broke = matrix[1][1] + matrix[1][2] + matrix[2][1] + matrix[2][2]
    return [[held, row_broke], [column_broke, both_broke]]


def _bootstrap(matrix: list[list[int]], iterations: int, seed: int) -> tuple[float, float]:
    generator = random.Random(seed)
    size = len(matrix)
    cells = [(row, column) for row in range(size) for column in range(size)]
    weights = [matrix[row][column] for row, column in cells]
    total = sum(weights)
    estimates: list[float] = []
    for _ in range(iterations):
        sample = [[0] * size for _ in range(size)]
        for row, column in generator.choices(cells, weights=weights, k=total):
            sample[row][column] += 1
        try:
            estimates.append(agreement(sample)[2])
        except (ValueError, ZeroDivisionError):
            continue
    if not estimates:
        raise ValueError("every bootstrap resample was degenerate")
    estimates.sort()
    low = estimates[int(0.025 * len(estimates))]
    high = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]
    return low, high


def analyse(
    matrix: list[list[int]],
    *,
    labels: tuple[str, ...] = VERDICTS,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> ConcordanceReport:

    observed, expected, kappa = agreement(matrix)
    if len(labels) != len(matrix):
        raise ValueError("label count must match the confusion matrix")
    low, high = _bootstrap(matrix, iterations, seed)
    size = len(matrix)
    return ConcordanceReport(
        total=sum(sum(row) for row in matrix),
        observed_agreement=observed,
        expected_agreement=expected,
        kappa=kappa,
        kappa_ci_low=low,
        kappa_ci_high=high,
        row_marginals={labels[index]: sum(matrix[index]) for index in range(size)},
        column_marginals={
            labels[column]: sum(matrix[row][column] for row in range(size))
            for column in range(size)
        },
    )
