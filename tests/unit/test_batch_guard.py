from __future__ import annotations

import pytest

from harness.agents.native.types import TransientModelError
from harness.experiments.batch_guard import ConsecutiveFailureBreaker, with_retries
from harness.experiments.diagnose import _classify
from harness.experiments.modal_cli import ModalCli


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_retries_survive_a_container_restart() -> None:

    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientModelError("modal endpoint returned an empty body")
        return "ok"

    assert await with_retries(flaky, sleep=_no_sleep) == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_gives_up_after_the_attempt_budget() -> None:
    async def always() -> str:
        raise TransientModelError("empty body")

    with pytest.raises(TransientModelError, match="still failing after 4 attempts"):
        await with_retries(always, sleep=_no_sleep)


@pytest.mark.asyncio
async def test_non_transient_errors_are_not_retried() -> None:

    calls = {"n": 0}

    async def broken() -> str:
        calls["n"] += 1
        raise ValueError("malformed response schema")

    with pytest.raises(ValueError):
        await with_retries(broken, sleep=_no_sleep)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_backoff_grows(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    async def always() -> str:
        raise TransientModelError("x")

    with pytest.raises(TransientModelError):
        await with_retries(always, attempts=4, base_delay_seconds=5.0, sleep=record)
    assert waits == [5.0, 10.0, 20.0]


def test_breaker_trips_only_on_consecutive_failures() -> None:
    breaker = ConsecutiveFailureBreaker(limit=3)
    breaker.record_failure("a")
    breaker.record_failure("b")
    breaker.record_success()
    breaker.record_failure("c")
    assert breaker.tripped is False, "an isolated blip must not stop the batch"
    breaker.record_failure("d")
    breaker.record_failure("e")
    assert breaker.tripped is True
    assert breaker.total_failures == 5


def test_breaker_reports_sample_reasons() -> None:
    breaker = ConsecutiveFailureBreaker()
    for i in range(9):
        breaker.record_failure(f"reason {i}")
    payload = breaker.as_json()
    assert payload["tripped"] is True
    assert len(payload["sample_reasons"]) == 5


def test_diagnosis_names_a_spend_limit() -> None:
    assert "spend limit" in _classify({"credit_exhausted": True})


def test_diagnosis_names_a_crash_loop() -> None:
    assert "failed to start" in _classify({"logs_healthy": False, "logs_reason": "oom"})


def test_diagnosis_recognises_a_recovered_restart() -> None:

    verdict = _classify({"restarted_and_recovered": True})
    assert "retry is safe" in verdict


def test_diagnosis_names_a_missing_app() -> None:
    assert "no longer deployed" in _classify({"still_deployed": False})


def test_diagnose_never_raises_on_a_broken_cli() -> None:
    from harness.experiments.diagnose import diagnose

    class Failing:
        def __call__(self, args, *, timeout):
            return (1, "", "boom")

    report = diagnose("some-app", cli=ModalCli(runner=Failing()))
    assert "likely_cause" in report
