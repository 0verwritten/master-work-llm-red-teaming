from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class SubscriptionRuntime(StrEnum):
    CODEX = "codex"
    CLAUDE = "claude"


@dataclass(frozen=True)
class BaselineResult:
    runtime: SubscriptionRuntime
    runtime_version: str
    status: str
    events: list[dict[str, Any]]
    final_output: str | None
    usage: dict[str, Any] | None
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    invocation: dict[str, Any]
    error_summary: str | None = None


class SubscriptionBaselineRunner:
    def __init__(self, *, timeout_seconds: int = 600) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    async def status() -> dict[str, dict[str, Any]]:
        return {
            "codex": await _codex_status(),
            "claude": await _claude_status(),
        }

    async def run(
        self,
        runtime: SubscriptionRuntime,
        *,
        prompt: str,
        model: str | None = None,
    ) -> BaselineResult:
        if os.getenv("ENABLE_SUBSCRIPTION_BASELINES", "").lower() not in {"1", "true", "yes"}:
            raise RuntimeError(
                "subscription baselines are disabled; set ENABLE_SUBSCRIPTION_BASELINES=true"
            )
        executable = shutil.which(runtime.value)
        if executable is None:
            raise RuntimeError(f"{runtime.value} CLI is not installed")
        version = await _version(executable)
        started_at = datetime.now(UTC)
        monotonic_started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"research-{runtime.value}-") as temp_dir:
            args = self._args(runtime, executable, Path(temp_dir), model)
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=temp_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(prompt.encode()), timeout=self.timeout_seconds
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                finished_at = datetime.now(UTC)
                return BaselineResult(
                    runtime=runtime,
                    runtime_version=version,
                    status="timed_out",
                    events=[],
                    final_output=None,
                    usage=None,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=int((time.monotonic() - monotonic_started) * 1000),
                    invocation=self._invocation(args, prompt, model),
                    error_summary=f"runtime exceeded {self.timeout_seconds} seconds",
                )

        finished_at = datetime.now(UTC)
        events, final_output, usage = self._parse(runtime, stdout.decode(errors="replace"))
        error_text = stderr.decode(errors="replace").strip()
        return BaselineResult(
            runtime=runtime,
            runtime_version=version,
            status="succeeded" if process.returncode == 0 else "failed",
            events=events,
            final_output=final_output,
            usage=usage,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=int((time.monotonic() - monotonic_started) * 1000),
            invocation=self._invocation(args, prompt, model),
            error_summary=error_text[-4_000:] or None,
        )

    @staticmethod
    def _args(
        runtime: SubscriptionRuntime,
        executable: str,
        working_directory: Path,
        model: str | None,
    ) -> list[str]:
        if runtime == SubscriptionRuntime.CODEX:
            args = [
                executable,
                "exec",
                "--json",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                str(working_directory),
            ]
            if model:
                args.extend(["--model", model])
            args.append("-")
            return args
        args = [
            executable,
            "--print",
            "--output-format",
            "json",
            "--safe-mode",
            "--tools",
            "",
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
        ]
        if model:
            args.extend(["--model", model])
        return args

    @staticmethod
    def _invocation(args: list[str], prompt: str, model: str | None) -> dict[str, Any]:
        return {
            "executable": Path(args[0]).name,
            "arguments": args[1:],
            "model_requested": model,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "trace_completeness": "opaque_runtime",
        }

    @staticmethod
    def _parse(
        runtime: SubscriptionRuntime, stdout: str
    ) -> tuple[list[dict[str, Any]], str | None, dict[str, Any] | None]:
        if runtime == SubscriptionRuntime.CLAUDE:
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError:
                return [], stdout.strip() or None, None
            events = [payload] if isinstance(payload, dict) else []
            if not isinstance(payload, dict):
                return events, None, None
            return (
                events,
                _as_optional_string(payload.get("result")),
                _as_dict(payload.get("usage")),
            )

        events = []
        final_output = None
        usage = None
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)
            if event.get("type") == "item.completed":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    final_output = _as_optional_string(item.get("text"))
            if event.get("type") == "turn.completed":
                usage = _as_dict(event.get("usage"))
        return events, final_output, usage


def _as_optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_dict(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


async def _version(executable: str) -> str:
    process = await asyncio.create_subprocess_exec(
        executable,
        "--version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await process.communicate()
    return stdout.decode(errors="replace").strip()


async def _codex_status() -> dict[str, Any]:
    executable = shutil.which("codex")
    if executable is None:
        return {"installed": False, "authenticated": False}
    process = await asyncio.create_subprocess_exec(
        executable,
        "login",
        "status",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await process.communicate()
    text = stdout.decode(errors="replace").strip()
    return {
        "installed": True,
        "version": await _version(executable),
        "authenticated": process.returncode == 0 and "logged in" in text.lower(),
        "auth_kind": "chatgpt" if "chatgpt" in text.lower() else "unknown",
    }


async def _claude_status() -> dict[str, Any]:
    executable = shutil.which("claude")
    if executable is None:
        return {"installed": False, "authenticated": False}
    process = await asyncio.create_subprocess_exec(
        executable,
        "auth",
        "status",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await process.communicate()
    try:
        raw = json.loads(stdout.decode(errors="replace"))
    except json.JSONDecodeError:
        raw = {}
    return {
        "installed": True,
        "version": await _version(executable),
        "authenticated": raw.get("loggedIn") is True,
        "auth_kind": raw.get("authMethod", "unknown"),
        "subscription_type": raw.get("subscriptionType", "unknown"),
    }
