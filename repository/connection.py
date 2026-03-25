import logging
from typing import AsyncGenerator

import psycopg
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from config.settings import settings

logger = logging.getLogger(__name__)

pg_pool: AsyncConnectionPool | None = None

async_session_maker = async_sessionmaker(
    bind=pg_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


def _on_reconnect_failed(pool: AsyncConnectionPool) -> None:
    """Called by the pool when it cannot re-establish a lost connection."""
    logger.critical(
        "Pool '%s' failed to reconnect to the database — "
        "all connections may be unavailable",
        pool.name,
    )


async def _check_connection(conn: psycopg.AsyncConnection) -> None:
    """Lightweight liveness probe run on each connection checkout."""
    await conn.execute("SELECT 1")


async def init_db() -> None:
    global pg_pool
    pg_pool = AsyncConnectionPool(
        conninfo=settings.postgres_dsn,
        min_size=2,
        max_size=10,
        max_waiting=100,
        check=_check_connection,
        reconnect_failed=_on_reconnect_failed,
        kwargs={"application_name": "graphmind"},
        open=False,
    )
    await pg_pool.open(wait=True)
    logger.info("Database connection pool opened (min=%d, max=%d)", 2, 10)


async def get_db_conn() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    if pg_pool is None:
        raise RuntimeError("Database pool is not initialised — call init_db() on startup")
    async with pg_pool.connection() as conn:
        yield conn


async def check_db() -> None:
    if pg_pool is None:
        raise RuntimeError("Database pool is not initialised — call init_db() on startup")
    try:
        async with pg_pool.connection() as conn:
            await conn.execute("SELECT 1")
        logger.info("Database health check passed")
    except PoolTimeout as exc:
        raise RuntimeError("Database health check timed out") from exc


async def close_db() -> None:
    logger.info("Closing database connection pool")
    if pg_pool is not None:
        await pg_pool.close()
