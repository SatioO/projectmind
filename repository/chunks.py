import json

from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from models.core import Category

# Columns stored explicitly on the table — excluded from the JSONB metadata blob
_EXPLICIT_COLUMNS = {"doc_id", "project_id", "agent_id"}


async def delete_chunks(
    conn: AsyncConnection,
    category: Category,
    doc_id: str,
) -> None:
    """Hard-delete all existing chunks for a doc_id before re-ingestion."""
    await conn.execute(
        text(f"DELETE FROM rag_{category}_chunks WHERE doc_id = :doc_id"),
        {"doc_id": doc_id},
    )


async def insert_chunks(
    conn: AsyncConnection,
    category: Category,
    chunks: list[Document],
    embeddings: list[list[float]],
) -> None:
    """Batch-insert chunks with pre-computed embeddings.

    Uses executemany — asyncpg prepares the statement once and pipelines
    all rows in a single round-trip. CAST(...) is used instead of ::type
    syntax because SQLAlchemy text() parses :: as part of a named param.

    Table name is interpolated from category (Literal type — not user input).
    All other values use bound parameters.
    """
    if not chunks:
        return

    rows = [
        {
            "doc_id": chunk.metadata["doc_id"],
            "project_id": chunk.metadata["project_id"],
            "agent_id": chunk.metadata.get("agent_id"),
            "content": chunk.page_content,
            "metadata": json.dumps(
                {k: v for k, v in chunk.metadata.items() if k not in _EXPLICIT_COLUMNS}
            ),
            "embedding": str(embedding),  # '[0.1, 0.2, ...]' — pgvector literal format
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    await conn.execute(
        text(f"""
            INSERT INTO rag_{category}_chunks
                (doc_id, project_id, agent_id, content, metadata, embedding)
            VALUES
                (:doc_id, :project_id, :agent_id, :content, CAST(:metadata AS jsonb), CAST(:embedding AS vector))
        """),
        rows,
    )
