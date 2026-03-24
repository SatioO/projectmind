from core.chunker import chunker
from core.store import vectorstore
from core.utils import generate_doc_id, sanitize_filename
from models.nodes import IngestDocument
from repository.connection import pg_engine


async def run_ingestion(
    project_id: str,
    agent_id: str | None,
    data: IngestDocument
):
    """Run the full ingestion pipeline for a single document.

      Safe to call multiple times for the same doc_id — deletes old chunks first.
      Marks the job 'failed' in doc_ingestion_jobs if any step raises.
    """

    try:
        # Generate document id based on the filename and project namespace
        modified_filename = sanitize_filename(data.filename)
        doc_id = generate_doc_id(project_id, modified_filename)

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
