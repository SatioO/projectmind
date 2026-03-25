import asyncio
from typing import Dict

from langchain_postgres import PGEngine, PGVectorStore

from config.settings import settings
from core.embedding import embedding
from models.core import Category
from repository import connection as db


class VectorStore:
    def __init__(self) -> None:
        self._stores: Dict[Category, PGVectorStore] = {}
        self._lock = asyncio.Lock()
        self._embedding_model = embedding.get_embedding_model(
            provider=settings.embedding_provider
        )
        self._pg_engine: PGEngine | None = None

    def _get_pg_engine(self) -> PGEngine:
        """Lazily wrap the shared SQLAlchemy engine in a PGEngine.

        Cannot be done at __init__ time because db.engine is None until
        init_db() runs during application startup.
        """
        if self._pg_engine is None:
            self._pg_engine = PGEngine.from_engine(db.engine)
        return self._pg_engine

    async def get_store(self, category: Category) -> PGVectorStore:
        if settings.postgres_plugin != "pgvector":
            raise NotImplementedError(f"{settings.postgres_plugin} is not implemented")

        if category in self._stores:
            return self._stores[category]

        async with self._lock:
            if category in self._stores:
                return self._stores[category]

            self._stores[category] = await PGVectorStore.create(
                engine=self._get_pg_engine(),
                table_name=f"rag_{category}_chunks",
                embedding_service=self._embedding_model,
                id_column="id",
                content_column="content",
                embedding_column="embedding",
                metadata_columns=["doc_id", "project_id", "agent_id"],
                metadata_json_column="metadata",
            )

        return self._stores[category]


vectorstore = VectorStore()
