import logging
from uuid import UUID

from core.chunker import chunker
from core.store import vectorstore
from models.nodes import IngestDocument
from repository.connection import pg_pool
from repository.Ingestion import mark_job_done, mark_job_failed

logger = logging.getLogger(__name__)


async def run_ingestion(
    doc_id: str,
    project_id: str,
    agent_id: str | None,
    data: IngestDocument,
) -> None:
    """Run the full ingestion pipeline for a single document.

    Safe to call multiple times for the same doc_id — deletes old chunks first.
    Marks the job 'failed' in doc_ingestion_jobs if any step raises.
    """
    async with pg_pool.connection() as conn:
        try:
            store = await vectorstore.get_store(category=data.category)

            semantic_chunks = chunker.split_documents(
                data.content,
                metadata={
                    "doc_id": doc_id,
                    "project_id": project_id,
                    "agent_id": agent_id,
                    "category": data.category,
                    **data.metadata,
                },
            )

            logger.info("doc_id=%s chunks=%d", doc_id, len(semantic_chunks))

            await mark_job_done(conn, doc_id, chunks_total=len(semantic_chunks))

        except Exception as exc:
            logger.exception("Ingestion failed for doc_id=%s", doc_id)
            await mark_job_failed(conn, doc_id, error=str(exc))
            raise
