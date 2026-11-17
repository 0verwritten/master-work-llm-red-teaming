from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from harness.experiments.modal_cli import DEFAULT_TIMEOUT_SECONDS, ModalCli, clean_output

FATAL_MARKERS: tuple[str, ...] = (
    "exited before readiness",
    "Runner failed with exception",
    "Task exited with code",
    "Error: No such",
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "ModuleNotFoundError",
    "DeserializationError",
    "ImportError",
)

_TRACEBACK = "Traceback (most recent call last)"
TRACEBACK_LOOP_THRESHOLD = 3


@dataclass(frozen=True)
class StartupHealth:
    healthy: bool
    reason: str | None = None
    excerpt: str | None = None


def inspect_logs(text: str) -> StartupHealth:

    plain = clean_output(text)
    for marker in FATAL_MARKERS:
        if marker in plain:
            index = plain.find(marker)
            return StartupHealth(
                healthy=False,
                reason=marker,
                excerpt=plain[max(0, index - 200) : index + 300].strip(),
            )
    if plain.count(_TRACEBACK) >= TRACEBACK_LOOP_THRESHOLD:
        return StartupHealth(
            healthy=False,
            reason=f"{plain.count(_TRACEBACK)} tracebacks (restart loop)",
            excerpt=plain[-400:].strip(),
        )
    return StartupHealth(healthy=True)


def check_app(
    app_name: str,
    *,
    cli: ModalCli | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> StartupHealth:

    cli = cli or ModalCli()
    try:
        text = cli._run("app", "logs", app_name, timeout=timeout)
    except RuntimeError:
        return StartupHealth(healthy=True)
    return inspect_logs(text)


def summarise_markers(text: str) -> Sequence[str]:
    plain = clean_output(text)
    return tuple(sorted({m for m in FATAL_MARKERS if m in plain}))


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
