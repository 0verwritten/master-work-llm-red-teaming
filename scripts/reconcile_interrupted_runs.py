"""Mark runs left mid-flight by an interrupted session as interrupted observations.

A session that dies takes its in-flight cells with it: their runs stay `running`
and their last requests stay `started`. Those are neither results nor absent
work. This marks them terminal with the retryable `role_request_error` stop
reason so a resume re-executes exactly those cells, and closes their dangling
call rows as failed. Rows that already reached a terminal state are untouched,
and no row is deleted, so the one-request/one-row invariant is preserved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from sqlalchemy import select, update

from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import (
    AdversarialSession,
    CallStatus,
    LLMCall,
    Run,
    RunStatus,
)

INTERRUPTED = "role_request_error"


async def reconcile(database_url: str, *, apply: bool) -> dict[str, Any]:
    engine = create_engine(database_url)
    make_session = session_factory(engine)
    try:
        async with make_session() as session:
            run_ids = list(
                (await session.scalars(select(Run.id).where(Run.status == RunStatus.RUNNING))).all()
            )
            call_ids = list(
                (
                    await session.scalars(
                        select(LLMCall.id)
                        .join(Run, Run.id == LLMCall.run_id)
                        .where(LLMCall.status == CallStatus.STARTED)
                        .where(Run.status == RunStatus.RUNNING)
                    )
                ).all()
            )
            session_ids = list(
                (
                    await session.scalars(
                        select(AdversarialSession.id).where(
                            AdversarialSession.run_id.in_(run_ids)
                            if run_ids
                            else AdversarialSession.run_id.is_(None)
                        )
                    )
                ).all()
            )
            if apply and run_ids:
                await session.execute(
                    update(Run)
                    .where(Run.id.in_(run_ids))
                    .values(status=RunStatus.FAILED, stop_reason=INTERRUPTED)
                )
                await session.execute(
                    update(AdversarialSession)
                    .where(AdversarialSession.id.in_(session_ids))
                    .values(status=RunStatus.FAILED)
                )
                if call_ids:
                    await session.execute(
                        update(LLMCall)
                        .where(LLMCall.id.in_(call_ids))
                        .values(status=CallStatus.FAILED)
                    )
                await session.commit()
            return {
                "applied": apply,
                "runs_marked_interrupted": len(run_ids),
                "sessions_marked_failed": len(session_ids),
                "calls_closed_failed": len(call_ids),
                "stop_reason": INTERRUPTED,
            }
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--apply", action="store_true", help="write the changes")
    arguments = parser.parse_args()
    print(
        json.dumps(asyncio.run(reconcile(arguments.database_url, apply=arguments.apply)), indent=2)
    )


if __name__ == "__main__":
    main()
