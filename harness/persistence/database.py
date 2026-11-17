from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine as sqlalchemy_create_async_engine

from harness.persistence.models import Base


def create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    return sqlalchemy_create_async_engine(database_url, echo=echo)


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_schema(engine: AsyncEngine) -> None:

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
