import psycopg
from fastapi import Depends

from models.nodes import IngestDocument
from core.utils import sanitize_filename
from repository.connection import get_db_conn


class IngestionRepository:
    def __init__(self, conn: psycopg.AsyncConnection = Depends(get_db_conn)):
        self.conn = conn

    async def ingest_document(self, project_id: str, agent_id: str | None, data: IngestDocument):
        scope = "agent" if agent_id else "project"
        await self.conn.execute(
            """
            INSERT INTO doc_ingestion_jobs (project_id, agent_id, category, scope, filename)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                project_id,
                agent_id,
                data.category,
                scope,
                sanitize_filename(filename=data.filename),
            ),
        )
        await self.conn.commit()
