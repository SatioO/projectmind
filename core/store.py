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
        self._engine = PGEngine.from_connection_string(settings.postgres_dsn)

    def _get_table_name(self, category: Category) -> str:
        return f"rag_{category}_chunks"

    async def get_store(self, category: Category) -> PGVectorStore:
        if settings.postgres_plugin != "pgvector":
            raise NotImplementedError(
                f"{settings.postgres_plugin} is not implemented"
            )

        if category in self._stores:
            return self._stores[category]

        async with self._lock:
            # Re-check inside the lock to avoid double initialisation
            if category in self._stores:
                return self._stores[category]

            store = await PGVectorStore.create(
                engine=self._engine,
                table_name=self._get_table_name(category),
                embedding_service=self._embedding_model,
            )
            self._stores[category] = store

        return self._stores[category]


vectorstore = VectorStore()
