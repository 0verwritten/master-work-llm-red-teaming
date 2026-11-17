from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest

from harness.agents.native import modal_readiness


class _Process:
    def __init__(self, return_codes: list[int | None]) -> None:
        self._return_codes = iter(return_codes)
        self.terminated = False

    def poll(self) -> int | None:
        return next(self._return_codes)

    def terminate(self) -> None:
        self.terminated = True


class _Response:
    status = 200


def test_wait_for_http_server_retries_until_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process([None, None])
    attempts = iter([OSError("not ready"), nullcontext(_Response())])

    def open_fixture(*_args: Any, **_kwargs: Any) -> Any:
        result = next(attempts)
        if isinstance(result, BaseException):
            raise result
        return result

    ticks = iter([0.0, 0.1, 0.2])
    monkeypatch.setattr(
        "harness.agents.native.modal_readiness.time.monotonic",
        lambda: next(ticks),
    )
    monkeypatch.setattr(
        "harness.agents.native.modal_readiness.time.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(modal_readiness, "urlopen", open_fixture)

    modal_readiness.wait_for_http_server(
        process,
        health_url="http://127.0.0.1:8000/health",
        timeout_seconds=1,
    )

    assert process.terminated is False


def test_wait_for_http_server_fails_immediately_when_child_exits() -> None:
    process = _Process([17])

    with pytest.raises(RuntimeError, match="exited before readiness: 17"):
        modal_readiness.wait_for_http_server(
            process,
            health_url="http://127.0.0.1:8000/health",
            timeout_seconds=1,
        )
