from typing import Dict, List, Union
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from models.core import DocumentChunkingStrategy
from config.settings import settings
from core.embedding import embedding


class DocumentChunker:
    def __init__(self) -> None:
        self._chunker = self._build_chunker(strategy=settings.chunking_strategy)

    def split_documents(
        self,
        content: Union[str, List[str]],
        metadata: Dict | None = None,
    ) -> List[Document]:
        texts = self._normalize_input(content)
        chunks = self._chunker.create_documents(texts)
        if metadata:
            chunks = self._enrich_context(chunks, metadata)
        return chunks

    def _enrich_context(
        self,
        chunks: List[Document],
        metadata: Dict,
        include_neighbors: bool = True,
    ) -> List[Document]:
        enriched_chunks: List[Document] = []

        for i, chunk in enumerate(chunks):
            enhanced_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "is_first_chunk": i == 0,
                "is_last_chunk": i == len(chunks) - 1,
            }

            if include_neighbors:
                if i > 0:
                    enhanced_metadata["previous_chunk_preview"] = chunks[i - 1].page_content[:100]
                if i < len(chunks) - 1:
                    enhanced_metadata["next_chunk_preview"] = chunks[i + 1].page_content[:100]

            chunk.metadata.update(enhanced_metadata)
            enriched_chunks.append(chunk)

        return enriched_chunks

    def _get_embedding_model(self):
        return embedding.get_embedding_model(provider=settings.embedding_provider)

    def _build_chunker(
        self, strategy: DocumentChunkingStrategy
    ) -> Union[RecursiveCharacterTextSplitter, CharacterTextSplitter, SemanticChunker]:
        if strategy == "character_based_fixed_size":
            cfg = settings.character_chunker
            return RecursiveCharacterTextSplitter(
                chunk_size=cfg.chunk_size,
                chunk_overlap=int(cfg.chunk_size * cfg.chunk_overlap_pct),
            )

        if strategy == "token_based_fixed_size":
            cfg = settings.token_chunker
            return CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=cfg.encoding_name,
                chunk_size=cfg.chunk_size,
                chunk_overlap=int(cfg.chunk_size * cfg.chunk_overlap_pct),
            )

        if strategy == "semantic":
            cfg = settings.semantic_chunker
            return SemanticChunker(
                self._get_embedding_model(),
                breakpoint_threshold_type=cfg.breakpoint_threshold_type,
                breakpoint_threshold_amount=cfg.breakpoint_threshold_amount,
                buffer_size=cfg.buffer_size,
            )

        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    def _normalize_input(self, content: Union[str, List[str]]) -> List[str]:
        if isinstance(content, str):
            return [content]
        return content


chunker = DocumentChunker()
