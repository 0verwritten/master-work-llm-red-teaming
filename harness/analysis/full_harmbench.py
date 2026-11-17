from __future__ import annotations

from math import comb, sqrt
from random import Random
from statistics import NormalDist


def rate(numerator: int, denominator: int) -> float | None:

    return numerator / denominator if denominator else None


def wilson_interval(successes: int, total: int, *, confidence: float = 0.95) -> tuple[float, float]:

    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    observed = successes / total
    denominator = 1 + z * z / total
    centre = (observed + z * z / (2 * total)) / denominator
    half_width = (
        z * sqrt(observed * (1 - observed) / total + z * z / (4 * total * total)) / denominator
    )
    return centre - half_width, centre + half_width


def exact_mcnemar(left_only: int, right_only: int) -> float:

    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(comb(discordant, value) for value in range(smaller + 1)) / float(2**discordant)
    return min(1.0, 2 * tail)


def paired_risk_difference_interval(
    pairs: list[tuple[bool, bool]],
    *,
    confidence: float = 0.95,
    resamples: int = 20_000,
    seed: int = 20260914,
) -> tuple[float, float]:

    if not pairs:
        raise ValueError("at least one pair is required")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    random = Random(seed)
    size = len(pairs)

    differences: list[float] = []
    for _ in range(resamples):
        total = 0
        for _ in range(size):
            left, right = pairs[random.randrange(size)]
            total += int(left) - int(right)
        differences.append(total / size)
    differences.sort()
    alpha = (1 - confidence) / 2
    low = differences[int(alpha * (resamples - 1))]
    high = differences[int((1 - alpha) * (resamples - 1))]
    return low, high


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:

    if min(a, b, c, d) < 0:
        raise ValueError("cell counts cannot be negative")
    row_one = a + b
    column_one = a + c
    total = a + b + c + d
    if total == 0:
        return 1.0

    def weight(value: int) -> int:
        return comb(column_one, value) * comb(total - column_one, row_one - value)

    lower = max(0, row_one - (total - column_one))
    upper = min(row_one, column_one)
    observed = weight(a)
    tail = sum(
        candidate for value in range(lower, upper + 1) if (candidate := weight(value)) <= observed
    )
    return tail / comb(total, row_one)


def holm_adjust(p_values: list[float]) -> list[float]:

    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda value: value[1])
    adjusted = [1.0] * count
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted
