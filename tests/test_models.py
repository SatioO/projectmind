from models.core import (
    CharacterChunkerConfig,
    TokenChunkerConfig,
    SemanticChunkerConfig,
    DocumentChunkingStrategy,
)


def test_character_chunker_config_defaults():
    cfg = CharacterChunkerConfig()
    assert cfg.chunk_size == 500
    assert cfg.chunk_overlap_pct == 0.15


def test_token_chunker_config_defaults():
    cfg = TokenChunkerConfig()
    assert cfg.chunk_size == 500
    assert cfg.chunk_overlap_pct == 0.15
    assert cfg.encoding_name == "cl100k_base"


def test_semantic_chunker_config_defaults():
    cfg = SemanticChunkerConfig()
    assert cfg.breakpoint_threshold_type == "percentile"
    assert cfg.breakpoint_threshold_amount == 95.0
    assert cfg.buffer_size == 1


def test_semantic_chunker_config_threshold_type_validation():
    from pydantic import ValidationError
    import pytest
    with pytest.raises(ValidationError):
        SemanticChunkerConfig(breakpoint_threshold_type="invalid")


def test_recursive_not_in_chunking_strategy():
    import typing
    args = typing.get_args(DocumentChunkingStrategy)
    assert "recursive" not in args


def test_valid_chunking_strategies():
    import typing
    args = typing.get_args(DocumentChunkingStrategy)
    assert set(args) == {"character_based_fixed_size", "token_based_fixed_size", "semantic"}
