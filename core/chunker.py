from typing import Dict, List, Union
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from config.settings import settings
from core.embedding import embedding


class Chunker:
    def __init__(self) -> None:
        self._embedding_model = embedding.get_embedding_model(
            provider=settings.embedding_provider
        )
        self._chunker = self._build_chunker()

    def split_documents(
        self,
        content: Union[str, List[str]],
        metadata: Dict | None = None,
    ) -> List[Document]:
        """
        Splits content into semantically coherent chunks.

        Args:
            content: raw string or list of strings
            metadata: optional metadata to attach to each chunk

        Returns:
            List[Document]
        """
        texts = self._normalize_input(content)
        docs = self._chunker.create_documents(texts)

        if metadata:
            for doc in docs:
                doc.metadata.update(metadata)

        return docs

    def _build_chunker(self) -> SemanticChunker:
        return SemanticChunker(
            self._embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            buffer_size=1,
        )

    def _normalize_input(self, content: Union[str, List[str]]) -> List[str]:
        if isinstance(content, str):
            return [content]
        return content


chunker = Chunker()
