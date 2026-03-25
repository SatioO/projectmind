import logging
from typing import AsyncGenerator

import psycopg
from psycopg_pool import AsyncConnectionPool

from config.settings import settings

logger = logging.getLogger(__name__)

pg_pool: AsyncConnectionPool | None = None


async def init_db() -> None:
    global pg_pool
    pg_pool = AsyncConnectionPool(
        conninfo=settings.postgres_dsn,
        min_size=2,
        max_size=10,
        open=False,
    )
    await pg_pool.open()


async def get_db_conn() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    async with pg_pool.connection() as conn:
        yield conn


async def check_db() -> None:
    async with pg_pool.connection() as conn:
        await conn.execute("SELECT 1")


async def close_db() -> None:
    logger.info("cleaning up postgres connection")
    if pg_pool:
        await pg_pool.close()
