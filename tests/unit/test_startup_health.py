from __future__ import annotations

from pathlib import Path

from harness.experiments.modal_cli import ModalCli
from harness.experiments.startup_health import (
    TRACEBACK_LOOP_THRESHOLD,
    check_app,
    inspect_logs,
)

REAL_GEMMA_FAILURE = """
(APIServer pid=4) transformers.integrations.heterogeneity.configuration_utils.
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute
Traceback (most recent call last):
  File "/root/app.py", line 76, in start
    wait_for_http_server(
RuntimeError: model server exited before readiness: 1
Runner failed with exception: RuntimeError('model server exited before readiness: 1')
"""


def test_detects_the_real_gemma_crash_loop() -> None:

    health = inspect_logs(REAL_GEMMA_FAILURE)
    assert health.healthy is False
    assert health.reason == "exited before readiness"
    assert health.excerpt and "head_dim" in health.excerpt


def test_slow_startup_is_not_flagged() -> None:

    logs = "\n".join(
        [
            "INFO Loading safetensors checkpoint shards: 12% Completed",
            "INFO Loading safetensors checkpoint shards: 48% Completed",
            "INFO Starting vLLM API server on http://0.0.0.0:8000",
        ]
    )
    assert inspect_logs(logs).healthy is True


def test_single_traceback_is_tolerated() -> None:

    assert inspect_logs("Traceback (most recent call last):\nsomething\n").healthy is True


def test_repeated_tracebacks_read_as_a_restart_loop() -> None:
    logs = "Traceback (most recent call last):\n" * TRACEBACK_LOOP_THRESHOLD
    health = inspect_logs(logs)
    assert health.healthy is False
    assert "restart loop" in (health.reason or "")


def test_oom_is_fatal() -> None:
    assert inspect_logs("torch.cuda.OutOfMemoryError: CUDA out of memory").healthy is False


def test_missing_dependency_is_fatal() -> None:

    assert inspect_logs("DeserializationError: pydantic not available").healthy is False


def test_ansi_styling_does_not_hide_a_failure() -> None:
    styled = "\x1b[31m(APIServer)\x1b[0m Runner failed with exception: boom"
    assert inspect_logs(styled).healthy is False


def test_log_fetch_failure_is_never_treated_as_fatal() -> None:

    class Failing:
        def __call__(self, args, *, timeout):
            return (1, "", "network hiccup")

    assert check_app("some-app", cli=ModalCli(runner=Failing())).healthy is True


def test_healthy_app_passes_through_the_cli() -> None:
    class Ok:
        def __call__(self, args, *, timeout):
            return (0, "INFO Loading safetensors checkpoint shards: 90%", "")

    assert check_app("some-app", cli=ModalCli(runner=Ok())).healthy is True


def test_warmup_wires_the_health_check() -> None:
    source = Path("harness/experiments/warmup.py").read_text(encoding="utf-8")
    assert "app_name" in source
    assert "check_app" in source
    assert "not waiting out the timeout" in source
