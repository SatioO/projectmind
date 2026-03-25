from typing import Dict, List, Union
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from models.core import DocumentChunkingStrategy
from config.settings import settings
from core.embedding import embedding


class Chunker:
    def __init__(self) -> None:
        self._embedding_model = embedding.get_embedding_model(
            provider=settings.embedding_provider
        )
        self._chunker = self._build_chunker(
            strategy=settings.chunking_strategy)

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
        chunks = self._chunker.create_documents(texts)

        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        return chunks

    def _build_chunker(self, strategy: DocumentChunkingStrategy) -> Union[RecursiveCharacterTextSplitter, CharacterTextSplitter, SemanticChunker]:
        if strategy == "character_based_fixed_size":
            return RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
            )

        if strategy == "token_based_fixed_size":
            return CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=500, chunk_overlap=0
            )

        if strategy == "semantic":
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
