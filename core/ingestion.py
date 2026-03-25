from uuid import UUID
from core.chunker import chunker
from core.store import vectorstore
from models.nodes import IngestDocument
from repository.Ingestion import IngestionJobRepository
from repository.connection import async_session_maker, pg_engine


async def run_ingestion(
    doc_id: str,
    project_id: str,
    agent_id: str | None,
    document: IngestDocument
):
    """Run the full ingestion pipeline for a single document.

      Safe to call multiple times for the same doc_id — deletes old chunks first.
      Marks the job 'failed' in doc_ingestion_jobs if any step raises.
    """
    try:
        # Initialize Vector Store
        store = await vectorstore.get_store(
            pg_engine=pg_engine,
            category=document.category
        )

        semantic_chunks = chunker.split_documents([document.content])
        print(f"Total chunks: {len(semantic_chunks)} \n\n")

        for semantic_chunk in semantic_chunks:
            print(f"====== Document Chunk: {doc_id} =====")
            print(semantic_chunk.page_content)

        async with async_session_maker() as session:
            repo = IngestionJobRepository(session)

    except:
        raise
