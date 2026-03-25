import asyncio
from typing import Dict

from langchain_postgres import PGEngine, PGVectorStore

from models.core import Category
from core.embedding import embedding
from config.settings import settings


class VectorStore:
    def __init__(self) -> None:
        self._stores: Dict[Category, PGVectorStore] = {}
        self._lock = asyncio.Lock()
        self._embedding_model = embedding.get_embedding_model(
            provider=settings.embedding_provider
        )
        self._engine = PGEngine.from_connection_string(settings.postgres_dsn_sqlalchemy)

    async def get_store(self, category: Category) -> PGVectorStore:
        if settings.postgres_plugin != "pgvector":
            raise NotImplementedError(f"{settings.postgres_plugin} is not implemented")

        if category in self._stores:
            return self._stores[category]

        async with self._lock:
            if category in self._stores:
                return self._stores[category]

            self._stores[category] = await PGVectorStore.create(
                engine=self._engine,
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
