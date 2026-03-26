import pytest
from pydantic import ValidationError


def test_settings_has_character_chunker():
    from config.settings import settings
    from models.core import CharacterChunkerConfig
    assert isinstance(settings.character_chunker, CharacterChunkerConfig)


def test_settings_has_token_chunker():
    from config.settings import settings
    from models.core import TokenChunkerConfig
    assert isinstance(settings.token_chunker, TokenChunkerConfig)


def test_settings_has_semantic_chunker():
    from config.settings import settings
    from models.core import SemanticChunkerConfig
    assert isinstance(settings.semantic_chunker, SemanticChunkerConfig)


def test_settings_embedding_provider_default():
    from config.settings import settings
    assert settings.embedding_provider == "huggingface"


def test_settings_embedding_provider_valid_values():
    from config.settings import Settings
    s = Settings(embedding_provider="openai")
    assert s.embedding_provider == "openai"


def test_settings_open_ai_is_invalid():
    from config.settings import Settings
    with pytest.raises(ValidationError):
        Settings(embedding_provider="open_ai")


def test_chunk_size_recommendations_removed():
    import config.settings as s
    assert not hasattr(s, "CHUNK_SIZE_RECOMMENDATIONS")
