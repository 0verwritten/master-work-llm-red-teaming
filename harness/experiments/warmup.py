from __future__ import annotations

import asyncio
import time

MODAL_BIN = ".venv/bin/modal"


async def wait_until_ready(
    endpoint_url: str,
    *,
    binary: str = MODAL_BIN,
    timeout_seconds: float = 2_700.0,
    poll_interval_seconds: float = 15.0,
    app_name: str | None = None,
    health_every: int = 4,
) -> float:

    started = time.monotonic()
    deadline = started + timeout_seconds
    url = f"{endpoint_url.rstrip('/')}/v1/models"
    last = "no attempt"
    polls = 0
    while time.monotonic() < deadline:
        polls += 1
        if app_name and polls % health_every == 0:
            from harness.experiments.startup_health import check_app

            health = check_app(app_name)
            if not health.healthy:
                raise RuntimeError(
                    f"{app_name} failed during startup ({health.reason}); "
                    f"not waiting out the timeout.\n{health.excerpt or ''}"
                )
        process = await asyncio.create_subprocess_exec(
            binary,
            "curl",
            "-sS",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        code = stdout.decode(errors="replace").strip()
        if process.returncode == 0 and code == "200":
            return time.monotonic() - started
        last = code or stderr.decode(errors="replace").strip()[:120]
        await asyncio.sleep(poll_interval_seconds)
    raise RuntimeError(f"endpoint not ready within {timeout_seconds}s (last: {last})")
