from uuid import UUID

import psycopg
from fastapi import Depends
from psycopg.rows import class_row
from dataclasses import dataclass
from datetime import datetime

from models.nodes import IngestDocument
from models.core import Category
from core.utils import sanitize_filename
from repository.connection import get_db_conn


@dataclass
class IngestionJob:
    id: UUID
    doc_id: str
    status: str
    created_at: datetime


class IngestionRepository:
    def __init__(self, conn: psycopg.AsyncConnection = Depends(get_db_conn)):
        self.conn = conn

    async def create_job(
        self,
        doc_id: str,
        project_id: str,
        agent_id: str | None,
        data: IngestDocument,
    ) -> IngestionJob:
        scope = "agent" if agent_id else "project"
        async with self.conn.cursor(row_factory=class_row(IngestionJob)) as cur:
            await cur.execute(
                """
                INSERT INTO doc_ingestion_jobs
                    (doc_id, project_id, agent_id, category, scope, filename)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, doc_id, status, created_at
                """,
                (
                    doc_id,
                    project_id,
                    agent_id,
                    data.category,
                    scope,
                    sanitize_filename(filename=data.filename),
                ),
            )
            row = await cur.fetchone()
        await self.conn.commit()
        return row


# --- Standalone helpers used by background tasks (outside request scope) ---

async def mark_job_failed(
    conn: psycopg.AsyncConnection,
    doc_id: str,
    error: str,
) -> None:
    await conn.execute(
        """
        UPDATE doc_ingestion_jobs
           SET status = 'failed', error = %s, updated_at = now()
         WHERE doc_id = %s
        """,
        (error, doc_id),
    )
    await conn.commit()


async def mark_job_done(
    conn: psycopg.AsyncConnection,
    doc_id: str,
    chunks_total: int,
) -> None:
    await conn.execute(
        """
        UPDATE doc_ingestion_jobs
           SET status = 'done',
               chunks_total = %s,
               chunks_done  = %s,
               updated_at   = now()
         WHERE doc_id = %s
        """,
        (chunks_total, chunks_total, doc_id),
    )
    await conn.commit()
