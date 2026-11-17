from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

MODAL_BIN = ".venv/bin/modal"
DEPLOY_TIMEOUT_SECONDS = 1_800
DEFAULT_TIMEOUT_SECONDS = 180

_ENDPOINT_RE = re.compile(r"https://[A-Za-z0-9._-]+\.modal\.(?:direct|run)")


class CommandRunner(Protocol):
    def __call__(self, args: Sequence[str], *, timeout: int) -> tuple[int, str, str]: ...


def run_command(
    args: Sequence[str], *, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> tuple[int, str, str]:

    env = {**os.environ, "COLUMNS": "1000", "TERM": "dumb"}
    completed = subprocess.run(
        list(args), capture_output=True, text=True, timeout=timeout, check=False, env=env
    )
    return completed.returncode, completed.stdout, completed.stderr


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean_output(text: str) -> str:

    return re.sub(r"\n\s+", "", _ANSI_RE.sub("", text))


@dataclass
class ModalCli:
    binary: str = MODAL_BIN
    runner: CommandRunner = run_command
    commands: list[list[str]] = field(default_factory=list)

    def _run(self, *args: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
        command = [self.binary, *args]
        self.commands.append(command)
        code, stdout, stderr = self.runner(command, timeout=timeout)
        if code != 0:
            raise RuntimeError(f"modal command failed ({code}): {' '.join(args)}\n{stderr.strip()}")
        return stdout

    def profile(self) -> str:
        return self._run("profile", "current").strip()

    def deploy(self, app_path: str) -> str:

        stdout = self._run("deploy", app_path, timeout=DEPLOY_TIMEOUT_SECONDS)
        match = _ENDPOINT_RE.search(clean_output(stdout))
        if match is None:
            raise RuntimeError(
                f"no *.modal.direct or *.modal.run endpoint in deploy output for {app_path}"
            )
        return match.group(0)

    def stop(self, app_name: str) -> None:

        self._run("app", "stop", "-y", app_name)

    def list_apps(self) -> str:
        return self._run("app", "list")

    def billing_summary(self) -> str:
        return self._run("billing", "summary")

    def running_app_names(self) -> list[str]:

        names: list[str] = []
        for line in self.list_apps().splitlines():
            plain = _ANSI_RE.sub("", line)
            if "deployed" not in plain or "│" not in plain:
                continue
            cells = [cell.strip() for cell in plain.split("│") if cell.strip()]
            if len(cells) >= 2:
                names.append(cells[1])
        return names


@dataclass(frozen=True)
class DeploymentRecord:
    role: str
    app_path: str
    endpoint_url: str

    def as_json(self) -> dict[str, Any]:
        return {"role": self.role, "app_path": self.app_path, "endpoint_url": self.endpoint_url}


def deploy_roles(cli: ModalCli, app_paths: dict[str, str]) -> list[DeploymentRecord]:

    records: list[DeploymentRecord] = []
    for role in sorted(app_paths):
        records.append(
            DeploymentRecord(
                role=role, app_path=app_paths[role], endpoint_url=cli.deploy(app_paths[role])
            )
        )
    return records


def stop_all(cli: ModalCli, app_names: Sequence[str]) -> list[str]:

    stopped: list[str] = []
    errors: list[str] = []
    for name in app_names:
        try:
            cli.stop(name)
            stopped.append(name)
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("failed to stop: " + "; ".join(errors))
    return stopped


def summarise_commands(cli: ModalCli) -> str:
    return json.dumps([" ".join(command) for command in cli.commands], indent=2)
