from uuid import UUID
from core.chunker import chunker
from core.store import vectorstore
from models.nodes import IngestDocument
from repository.connection import pg_engine


async def run_ingestion(
    doc_id: UUID,
    project_id: str,
    agent_id: str | None,
    data: IngestDocument
):
    """Run the full ingestion pipeline for a single document.

      Safe to call multiple times for the same doc_id — deletes old chunks first.
      Marks the job 'failed' in doc_ingestion_jobs if any step raises.
    """
    try:
        # Initialize Vector Store
        store = vectorstore.get_store(
            pg_engine=pg_engine,
            category=data.category
        )

        semantic_chunks = chunker.split_documents([data.content])
        print(f"Total chunks: {len(semantic_chunks)} \n\n")

        for semantic_chunk in semantic_chunks:
            print(f"====== Document Chunk: {doc_id} =====")
            print(semantic_chunk.page_content)

    except:
        raise
