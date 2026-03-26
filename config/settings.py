from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

from models.core import (
    DocumentChunkingStrategy,
    EmbeddingModel,
    CharacterChunkerConfig,
    TokenChunkerConfig,
    SemanticChunkerConfig,
)


class Settings(BaseSettings):
    # === POSTGRES ===
    postgres_dsn: str = "postgresql+asyncpg://postgres:password@localhost:5432/graphmind"
    postgres_plugin: Literal["pgvector", "pgsearch"] = "pgvector"
    log_level: str = "INFO"

    # === Chunking ===
    chunking_strategy: DocumentChunkingStrategy = "character_based_fixed_size"
    character_chunker: CharacterChunkerConfig = CharacterChunkerConfig()
    token_chunker: TokenChunkerConfig = TokenChunkerConfig()
    semantic_chunker: SemanticChunkerConfig = SemanticChunkerConfig()

    # === Embedding ===
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: EmbeddingModel = "BAAI/bge-base-en-v1.5"
    openai_api_key: str = ""
    embedding_dimensions: int = 1536

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


settings = Settings()
