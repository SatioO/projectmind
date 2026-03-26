import pytest
from unittest.mock import patch
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document


def test_enrich_context_returns_all_chunks():
    """_enrich_context must return the same number of chunks it receives."""
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)  # skip __init__
    docs = [Document(page_content=f"chunk {i}") for i in range(3)]
    result = chunker._enrich_context(docs, {"doc_id": "test"})
    assert len(result) == 3


def test_enrich_context_sets_chunk_index():
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)
    docs = [Document(page_content=f"chunk {i}") for i in range(3)]
    result = chunker._enrich_context(docs, {"doc_id": "test"})
    assert result[0].metadata["chunk_index"] == 0
    assert result[1].metadata["chunk_index"] == 1
    assert result[2].metadata["chunk_index"] == 2


def test_enrich_context_boundary_flags():
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)
    docs = [Document(page_content=f"chunk {i}") for i in range(3)]
    result = chunker._enrich_context(docs, {"doc_id": "test"})
    assert result[0].metadata["is_first_chunk"] is True
    assert result[0].metadata["is_last_chunk"] is False
    assert result[2].metadata["is_first_chunk"] is False
    assert result[2].metadata["is_last_chunk"] is True


def test_build_chunker_character_returns_recursive_splitter():
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)
    splitter = chunker._build_chunker("character_based_fixed_size")
    assert isinstance(splitter, RecursiveCharacterTextSplitter)


def test_build_chunker_token_returns_character_splitter():
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)
    splitter = chunker._build_chunker("token_based_fixed_size")
    assert isinstance(splitter, CharacterTextSplitter)


def test_build_chunker_unknown_raises():
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunker._build_chunker("not_a_strategy")


def test_embedding_model_not_loaded_for_character_strategy(monkeypatch):
    """Embedding model should NOT be instantiated for non-semantic strategies."""
    from core import chunker as chunker_module
    call_count = {"n": 0}

    original = chunker_module.embedding.get_embedding_model

    def counting_get(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(chunker_module.embedding, "get_embedding_model", counting_get)

    from models.core import CharacterChunkerConfig
    with patch("core.chunker.settings") as mock_settings:
        mock_settings.chunking_strategy = "character_based_fixed_size"
        mock_settings.character_chunker = CharacterChunkerConfig()
        from core.chunker import DocumentChunker
        DocumentChunker()

    assert call_count["n"] == 0, "Embedding model should not be loaded for character strategy"
