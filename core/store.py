from typing import Dict
from langchain_postgres import PGVectorStore
from sqlalchemy.ext.asyncio import AsyncEngine

from models.nodes import Category
from core.embedding import embedding
from config.settings import settings


class VectorStore:
    def __init__(self) -> None:
        self._stores: Dict[str, PGVectorStore] = {}
        self._embedding_model = embedding.get_embedding_model(
            provider=settings.embedding_provider
        )

    def _get_table_name(self, category: Category) -> str:
        return f"rag_{category}_chunks"

    async def get_store(
        self,
        pg_engine: AsyncEngine,
        category: Category,
    ) -> PGVectorStore:
        """
        Returns a cached PGVectorStore per category.
        Lazily initializes on first access.
        """
        if settings.postgres_plugin != "pgvector":
            raise NotImplementedError(
                f"{settings.postgres_plugin} is not implemented"
            )

        if category in self._stores:
            return self._stores[category]

        table_name = self._get_table_name(category)

        store = await PGVectorStore.create(
            engine=pg_engine,
            table_name=table_name,
            embedding_service=self._embedding_model,
        )

        self._stores[category] = store
        return store


vectorstore = VectorStore()
