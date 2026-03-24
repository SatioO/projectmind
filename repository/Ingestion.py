from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy import (
    Column,
    Text,
    Integer,
    CheckConstraint,
    Index,
    text,
)

from models.nodes import IngestDocument
from core.utils import sanitize_filename
from repository.connection import Base, get_db_session

# --- ENUMS(must match Postgres enums) - --
doc_category_enum = ENUM(
    "prd", "arch", "code", "tasks", "ops",
    name="doc_category",
    create_type=False,   # already created via DDL
)

ingestion_scope_enum = ENUM(
    "project", "agent",
    name="ingestion_scope",
    create_type=False,
)

ingestion_status_enum = ENUM(
    "processing", "done", "failed", "deleted",
    name="ingestion_status",
    create_type=False,
)


class DocIngestionJob(Base):
    __tablename__ = "doc_ingestion_jobs"

    # --- Primary Key ---
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    # --- Core Identity ---
    project_id = Column(Text, nullable=False)
    agent_id = Column(Text, nullable=True)

    # --- Classification ---
    category = Column(doc_category_enum, nullable=False)
    scope = Column(ingestion_scope_enum, nullable=False)

    # --- File info ---
    filename = Column(Text, nullable=True)

    # --- Status ---
    status = Column(
        ingestion_status_enum,
        nullable=False,
        server_default=text("'processing'::ingestion_status"),
    )

    # --- Progress ---
    chunks_total = Column(Integer, nullable=True)
    chunks_done = Column(Integer, nullable=False, server_default="0")

    # --- Error tracking ---
    error = Column(Text, nullable=True)

    # --- Timestamps ---
    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # --- Constraints ---
    __table_args__ = (
        # chunks_total >= 0
        CheckConstraint(
            "chunks_total IS NULL OR chunks_total >= 0",
            name="check_chunks_total_positive",
        ),

        # chunks_done >= 0
        CheckConstraint(
            "chunks_done >= 0",
            name="check_chunks_done_positive",
        ),

        # chunks_done <= chunks_total
        CheckConstraint(
            "chunks_total IS NULL OR chunks_done <= chunks_total",
            name="check_chunks_progress_valid",
        ),

        # scope-agent consistency
        CheckConstraint(
            "(scope = 'project' AND agent_id IS NULL) OR "
            "(scope = 'agent' AND agent_id IS NOT NULL)",
            name="check_agent_scope_consistency",
        ),

        # --- Indexes ---
        Index("idx_ingestion_project_created", "project_id", "created_at"),
        Index(
            "idx_ingestion_active",
            "project_id",
            postgresql_where=text("status = 'processing'"),
        ),
        Index(
            "idx_ingestion_agent",
            "agent_id",
            postgresql_where=text("agent_id IS NOT NULL"),
        ),
    )


class IngestionRepository:
    def __init__(self, session: AsyncSession = Depends(get_db_session)):
        self.session = session

    async def ingest_document(self, project_id: str, agent_id: str, data: IngestDocument):
        try:
            ingestion_job = DocIngestionJob(
                project_id=project_id,
                agent_id=agent_id,
                category=data.category,
                scope="project",
                filename=sanitize_filename(filename=data.filename))
            self.session.add(ingestion_job)
            await self.session.commit()
        except:
            self.session.rollback()
            raise
        finally:
            self.session.close()
