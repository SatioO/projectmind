from fastapi import APIRouter, Depends
from fastapi.background import BackgroundTasks

from models.nodes import IngestDocument
from services.ingestion import IngestionService

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post("/projects/{project_id}/documents", status_code=201)
async def ingest_data(project_id: str, data: IngestDocument, background_tasks: BackgroundTasks, ingestion_svc: IngestionService = Depends(IngestionService)):
    return await ingestion_svc.ingest_document(project_id, None, data, background_tasks)
