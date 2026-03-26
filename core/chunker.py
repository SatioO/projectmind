from typing import Dict, List, Union
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from models.core import DocumentChunkingStrategy
from config.settings import CHUNK_SIZE_RECOMMENDATIONS, settings
from core.embedding import embedding


class DocumentChunker:
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
            chunks = self._enrich_context(chunks, metadata)

        return chunks

    def _enrich_context(self, chunks: List[Document], metadata: Dict,  include_neighbors: bool = True) -> List[Document]:
        """
            Enrich chunks with contextual information for better retrieval.

            Args:
                chunks: List of document chunks
                include_neighbors: Whether to include references to neighboring chunks

            Returns:
                Enriched document chunks
        """
        enriched_chunks: List[Document] = []

        for i, chunk in enumerate(chunks):
            # Create enhanced metadata
            enhanced_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "is_first_chunk": i == 0,
                "is_last_chunk": i == len(chunks) - 1
            }

            # Optionally add neighbor context
            if include_neighbors:
                if i > 0:
                    enhanced_metadata["previous_chunks_preview"] = \
                        chunks[i - 1].page_content[:100]
                if i < len(chunks) - 1:
                    enhanced_metadata["next_chunk_preview"] = \
                        chunks[i + 1].page_content[:100]

            chunk.metadata.update(enhanced_metadata)

        return enriched_chunks

    def _build_chunker(self, strategy: DocumentChunkingStrategy) -> Union[RecursiveCharacterTextSplitter, CharacterTextSplitter, SemanticChunker]:
        chunk_size = CHUNK_SIZE_RECOMMENDATIONS.get(
            settings.embedding_model, 500)

        # Overlap should typically be 10-20 % of chunk size
        chunk_overlap = int(chunk_size * 0.15)

        if strategy == "character_based_fixed_size":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        if strategy == "token_based_fixed_size":
            return CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap
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


chunker = DocumentChunker()
