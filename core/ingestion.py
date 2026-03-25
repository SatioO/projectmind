import logging

from config.settings import settings
from core.chunker import chunker
from core.embedding import embedding
from models.nodes import IngestDocument
from repository import connection as db
from repository.chunks import delete_chunks, insert_chunks
from repository.Ingestion import mark_job_done, mark_job_failed

logger = logging.getLogger(__name__)


async def run_ingestion(
    doc_id: str,
    project_id: str,
    agent_id: str | None,
    data: IngestDocument,
) -> None:
    """Run the full ingestion pipeline for a single document.

    Embedding generation happens before the connection is opened so the
    pool slot is not held idle during model inference.

    delete → insert → mark_done all run on the same connection — fully
    atomic. On any failure the transaction rolls back, old chunks are
    preserved, and the job is marked failed.
    """
    embedding_model = embedding.get_embedding_model(provider=settings.embedding_provider)

    chunks = chunker.split_documents(
        data.content,
        metadata={
            "doc_id": doc_id,
            "project_id": project_id,
            "agent_id": agent_id,
            "category": data.category,
            **data.metadata,
        },
    )
    logger.info("doc_id=%s chunks=%d", doc_id, len(chunks))

    embeddings = await embedding_model.aembed_documents(
        [chunk.page_content for chunk in chunks]
    )

    async with db.engine.connect() as conn:
        try:
            await delete_chunks(conn, data.category, doc_id)
            await insert_chunks(conn, data.category, chunks, embeddings)
            await mark_job_done(conn, doc_id, chunks_total=len(chunks))
            await conn.commit()

        except Exception as exc:
            logger.exception("Ingestion failed for doc_id=%s", doc_id)
            await mark_job_failed(conn, doc_id, error=str(exc))
            await conn.commit()
            raise
