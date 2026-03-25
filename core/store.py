from typing import Dict
from langchain_postgres import PGVectorStore

from models.core import Category
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

    async def get_store(self, category: Category) -> PGVectorStore:
        if settings.postgres_plugin != "pgvector":
            raise NotImplementedError(
                f"{settings.postgres_plugin} is not implemented"
            )

        if category in self._stores:
            return self._stores[category]

        table_name = self._get_table_name(category)

        store = await PGVectorStore.create(
            connection=settings.postgres_dsn,
            table_name=table_name,
            embedding_service=self._embedding_model,
        )

        self._stores[category] = store
        return store


vectorstore = VectorStore()
