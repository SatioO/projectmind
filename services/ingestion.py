from typing import Optional
from fastapi import Depends
from fastapi.background import BackgroundTasks

from core.ingestion import run_ingestion
from models.nodes import IngestDocument, IngestDocumentResponse
from repository.Ingestion import IngestionRepository


class IngestionService:
    def __init__(self, ingestion_repo: IngestionRepository = Depends(IngestionRepository)) -> None:
        self.ingestion_repo = ingestion_repo

    async def ingest_document(self, project_id: str, agent_id: Optional[str], data: IngestDocument, background_tasks: BackgroundTasks):
        await run_ingestion(project_id, agent_id, data)

        return IngestDocumentResponse(doc_id="123", status="processing")
