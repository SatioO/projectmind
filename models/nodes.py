from pydantic import BaseModel
from pydantic.types import Literal

Category = Literal["prd", "arch", "code", "tasks", "ops"]


class IngestDocument(BaseModel):
    category: Category
    filename: str
    content: str                  # raw text; caller handles PDF/DOCX extraction
    metadata: dict = {}           # optional: {component, language, env, spec_id}


class IngestDocumentResponse(BaseModel):
    status: Literal["processing"] = "processing"
