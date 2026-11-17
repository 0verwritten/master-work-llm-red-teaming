from __future__ import annotations

import time
from typing import Protocol
from urllib.error import URLError
from urllib.request import urlopen


class ManagedProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


def wait_for_http_server(
    process: ManagedProcess,
    *,
    health_url: str,
    timeout_seconds: float,
    poll_interval_seconds: float = 1.0,
) -> None:

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"model server exited before readiness: {return_code}")
        try:
            with urlopen(health_url, timeout=poll_interval_seconds) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(poll_interval_seconds)
    process.terminate()
    raise TimeoutError(f"model server did not become healthy within {timeout_seconds}s")
