from typing import Literal
from uuid import UUID
from models.core import Category
from pydantic import BaseModel


class IngestDocument(BaseModel):
    category: Category
    filename: str
    content: str                  # raw text; caller handles PDF/DOCX extraction
    metadata: dict = {}           # optional: {component, language, env, spec_id}


class IngestDocumentResponse(BaseModel):
    doc_id: UUID
    status: Literal["processing"] = "processing"
