from __future__ import annotations

import asyncio
import signal

from harness.persistence.database import create_engine, session_factory
from harness.scheduler import Scheduler
from harness.settings import Settings


async def serve() -> None:
    settings = Settings()
    engine = create_engine(settings.database_url)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        scheduler = Scheduler(
            sessions=session_factory(engine),
            abandoned_after_seconds=settings.scheduler_abandoned_after_seconds,
        )
        await scheduler.run_forever(poll_seconds=settings.scheduler_poll_seconds, stop=stop)
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
