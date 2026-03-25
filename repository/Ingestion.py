from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from models.nodes import IngestDocument
from repository.connection import get_db_conn


@dataclass
class IngestionJob:
    id: UUID
    doc_id: str
    status: str
    created_at: datetime


class IngestionJobRepository:
    def __init__(self, conn: AsyncConnection = Depends(get_db_conn)) -> None:
        self.conn = conn

    async def create_job(
        self,
        doc_id: str,
        project_id: str,
        agent_id: str | None,
        filename: str,
        data: IngestDocument,
    ) -> IngestionJob:
        scope = "agent" if agent_id else "project"
        result = await self.conn.execute(
            text("""
                INSERT INTO doc_ingestion_jobs
                    (doc_id, project_id, agent_id, category, scope, filename)
                VALUES (:doc_id, :project_id, :agent_id, :category, :scope, :filename)
                ON CONFLICT (doc_id) DO UPDATE SET
                    status       = 'processing',
                    chunks_total = NULL,
                    chunks_done  = 0,
                    error        = NULL,
                    updated_at   = now()
                RETURNING id, doc_id, status, created_at
            """),
            {
                "doc_id": doc_id,
                "project_id": project_id,
                "agent_id": agent_id,
                "category": data.category,
                "scope": scope,
                "filename": filename,
            },
        )
        await self.conn.commit()
        row = result.fetchone()
        return IngestionJob(id=row.id, doc_id=row.doc_id, status=row.status, created_at=row.created_at)

async def mark_job_failed(
    conn: AsyncConnection,
    doc_id: str,
    error: str,
) -> None:
    await conn.execute(
        text("""
            UPDATE doc_ingestion_jobs
            SET status = 'failed', error = :error, updated_at = now()
            WHERE doc_id = :doc_id
        """),
        {"error": error, "doc_id": doc_id},
    )
    await conn.commit()

async def mark_job_done(
    conn: AsyncConnection,
    doc_id: str,
    chunks_total: int,
) -> None:
    await conn.execute(
        text("""
            UPDATE doc_ingestion_jobs
            SET status = 'done',
                chunks_total = :chunks_total,
                chunks_done  = :chunks_total,
                updated_at   = now()
            WHERE doc_id = :doc_id
        """),
        {"chunks_total": chunks_total, "doc_id": doc_id},
    )
    await conn.commit()
