import logging
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from config.settings import settings

logger = logging.getLogger(__name__)

engine: AsyncEngine | None = None


async def init_db() -> None:
    global engine
    engine = create_async_engine(
        settings.postgres_dsn,
        pool_size=2,
        max_overflow=8,
        pool_pre_ping=True,
    )
    logger.info("Database engine initialized")


async def get_db_conn() -> AsyncGenerator[AsyncConnection, None]:
    if engine is None:
        raise RuntimeError(
            "Database engine is not initialised — call init_db() on startup")
    async with engine.connect() as conn:
        yield conn


async def check_db() -> None:
    if engine is None:
        raise RuntimeError(
            "Database engine is not initialised — call init_db() on startup")
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("Database health check passed")


async def close_db() -> None:
    logger.info("Closing database engine")
    if engine is not None:
        await engine.dispose()
