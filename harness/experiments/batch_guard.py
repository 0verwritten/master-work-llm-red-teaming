from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from harness.agents.native.types import TransientModelError

DEFAULT_ATTEMPTS = 4
DEFAULT_BASE_DELAY_SECONDS = 5.0
DEFAULT_CONSECUTIVE_LIMIT = 3


async def with_retries[T](
    factory: Callable[[], Awaitable[T]],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:

    if attempts < 1:
        raise ValueError("attempts must be positive")
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await factory()
        except TransientModelError as exc:
            last = exc
            if attempt == attempts - 1:
                break
            await sleep(base_delay_seconds * (2**attempt))
    raise TransientModelError(f"still failing after {attempts} attempts: {last}") from last


@dataclass
class ConsecutiveFailureBreaker:
    limit: int = DEFAULT_CONSECUTIVE_LIMIT
    consecutive: int = 0
    total_failures: int = 0
    reasons: list[str] = field(default_factory=list)

    def record_success(self) -> None:
        self.consecutive = 0

    def record_failure(self, reason: str) -> None:
        self.consecutive += 1
        self.total_failures += 1
        if len(self.reasons) < 5:
            self.reasons.append(reason[:200])

    @property
    def tripped(self) -> bool:
        return self.consecutive >= self.limit

    def as_json(self) -> dict[str, Any]:
        return {
            "consecutive_failures": self.consecutive,
            "total_failures": self.total_failures,
            "limit": self.limit,
            "tripped": self.tripped,
            "sample_reasons": list(self.reasons),
        }
