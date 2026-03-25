from typing import Optional
from fastapi import Depends
from fastapi.background import BackgroundTasks

from core.ingestion import run_ingestion
from models.nodes import IngestDocument, IngestDocumentResponse
from repository.Ingestion import IngestionJobRepository
from core.utils import generate_doc_id, sanitize_filename


class IngestionService:
    def __init__(self, ingestion_repo: IngestionJobRepository = Depends(IngestionJobRepository)) -> None:
        self.ingestion_repo = ingestion_repo

    async def ingest_document(self, project_id: str, agent_id: Optional[str], document: IngestDocument, background_tasks: BackgroundTasks):
       # Generate document id based on the filename and project namespace
        modified_filename = sanitize_filename(document.filename)
        doc_id = generate_doc_id(project_id, modified_filename)

        # Inserts new job and runs it in background for further processing
        await self.ingestion_repo.ingest_job(str(doc_id), project_id, agent_id, document)
        background_tasks.add_task(
            run_ingestion,
            doc_id=str(doc_id),
            project_id=project_id,
            agent_id=agent_id,
            document=document
        )

        return IngestDocumentResponse(doc_id=doc_id)
