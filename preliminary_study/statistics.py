from __future__ import annotations

import random
from dataclasses import dataclass
from math import comb, sqrt


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


def wilson_interval(successes: int, trials: int, z: float = 1.959963985) -> Interval:

    if trials == 0:
        return Interval(0.0, 0.0)
    phat = successes / trials
    denominator = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    margin = (z / denominator) * sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
    return Interval(max(0.0, centre - margin), min(1.0, centre + margin))


def binomial_two_sided_p(successes: int, trials: int, probability: float = 0.5) -> float:

    if trials == 0:
        return 1.0
    probabilities = [
        comb(trials, k) * probability**k * (1 - probability) ** (trials - k)
        for k in range(trials + 1)
    ]
    observed = probabilities[successes]

    tolerance = observed * 1e-7
    return min(1.0, sum(p for p in probabilities if p <= observed + tolerance))


def exact_mcnemar(*, only_first: int, only_second: int) -> float:

    return binomial_two_sided_p(only_first, only_first + only_second, 0.5)


def sign_test(differences: list[float]) -> float:

    positives = sum(1 for value in differences if value > 0)
    negatives = sum(1 for value in differences if value < 0)
    return binomial_two_sided_p(positives, positives + negatives, 0.5)


def bootstrap_mean_difference_ci(
    paired: list[tuple[float, float]],
    *,
    iterations: int = 20_000,
    seed: int = 20260817,
    alpha: float = 0.05,
) -> Interval:

    if not paired:
        return Interval(0.0, 0.0)
    rng = random.Random(seed)
    size = len(paired)
    differences = [first - second for first, second in paired]
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(size):
            total += differences[rng.randrange(size)]
        means.append(total / size)
    means.sort()
    low_index = int((alpha / 2) * iterations)
    high_index = min(iterations - 1, int((1 - alpha / 2) * iterations))
    return Interval(means[low_index], means[high_index])


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:

    if not pairs:
        return 0.0
    labels = sorted({label for pair in pairs for label in pair})
    total = len(pairs)
    observed = sum(1 for first, second in pairs if first == second) / total
    expected = 0.0
    for label in labels:
        first_share = sum(1 for first, _ in pairs if first == label) / total
        second_share = sum(1 for _, second in pairs if second == label) / total
        expected += first_share * second_share
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)
