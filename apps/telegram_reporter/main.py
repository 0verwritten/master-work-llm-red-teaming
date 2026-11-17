from __future__ import annotations

import asyncio
import signal

from harness.persistence.database import create_engine, session_factory
from harness.reporting import TelegramClient, TelegramReporter
from harness.settings import Settings


async def serve() -> None:
    settings = Settings()
    engine = create_engine(settings.database_url)
    client = TelegramClient(settings.require_telegram_token())
    reporter = TelegramReporter(
        client=client,
        sessions=session_factory(engine),
        allowed_chat_ids=settings.allowed_chat_ids,
        private_trace_base_url=settings.private_trace_base_url,
        poll_timeout_seconds=settings.telegram_poll_timeout_seconds,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        await reporter.run_forever(stop=stop)
    finally:
        await client.close()
        await engine.dispose()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
