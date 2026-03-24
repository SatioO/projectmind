import logging
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config.settings import settings

logger = logging.getLogger(__name__)


pg_engine = create_async_engine(
    str(settings.postgres_dsn),  # IMPORTANT: cast to str
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
)

# ---- Session Factory ----
SessionLocal = async_sessionmaker(
    bind=pg_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def check_db():
    async with pg_engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def close_db() -> None:
    logging.info("cleaning up postgres connection")
    await pg_engine.dispose()

Base = declarative_base()
