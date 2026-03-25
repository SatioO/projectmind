import asyncio
import json
from typing import Dict, List

from langchain_core.documents import Document

from models.core import Category
from core.embedding import embedding
from config.settings import settings
from repository import connection as db


class CategoryStore:
    """Direct psycopg3-backed vector store for a single document category."""

    def __init__(self, category: Category, embedding_model) -> None:
        self._table = f"rag_{category}_chunks"
        self._embedding_model = embedding_model

    async def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        vectors = await self._embedding_model.aembed_documents(texts)

        async with db.pg_pool.connection() as conn:
            async with conn.cursor() as cur:
                for doc, vec in zip(documents, vectors):
                    meta = dict(doc.metadata)
                    doc_id = meta.pop("doc_id", None)
                    project_id = meta.pop("project_id", None)
                    agent_id = meta.pop("agent_id", None)
                    meta.pop("category", None)
                    vec_str = "[" + ",".join(str(v) for v in vec) + "]"

                    await cur.execute(
                        f"INSERT INTO {self._table}"
                        " (doc_id, project_id, agent_id, content, metadata, embedding)"
                        " VALUES (%s, %s, %s, %s, %s, %s::vector)",
                        (doc_id, project_id, agent_id, doc.page_content,
                         json.dumps(meta), vec_str),
                    )


class VectorStore:
    def __init__(self) -> None:
        self._stores: Dict[Category, CategoryStore] = {}
        self._lock = asyncio.Lock()
        self._embedding_model = embedding.get_embedding_model(
            provider=settings.embedding_provider
        )

    async def get_store(self, category: Category) -> CategoryStore:
        if settings.postgres_plugin != "pgvector":
            raise NotImplementedError(f"{settings.postgres_plugin} is not implemented")

        if category in self._stores:
            return self._stores[category]

        async with self._lock:
            if category in self._stores:
                return self._stores[category]

            self._stores[category] = CategoryStore(
                category=category,
                embedding_model=self._embedding_model,
            )

        return self._stores[category]


vectorstore = VectorStore()
