from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

from models.core import DocumentChunkingStrategy, EmbeddingModel

CHUNK_SIZE_RECOMMENDATIONS = {
    "text-embedding-ada-002": 500,      # OpenAI
    "text-embedding-3-small": 512,      # OpenAI
    "text-embedding-3-large": 1024,     # OpenAI
    # "all-MiniLM-L6-v2": 256,            # Sentence Transformers
    # "all-mpnet-base-v2": 384,           # Sentence Transformers
    "BAAI/bge-base-en-v1.5": 512
}


class Settings(BaseSettings):
    # === POSTGRES ===
    postgres_dsn: str = "postgresql+asyncpg://postgres:password@localhost:5432/graphmind"
    # TODO: bring in pgsearch support for BM25 search capabilities
    postgres_plugin: Literal["pgvector", "pgsearch"] = "pgvector"
    log_level: str = "INFO"

    # === Chunking ===
    chunking_strategy:  DocumentChunkingStrategy = "character_based_fixed_size"
    embedding_provider: Literal["huggingface", "open_ai"] = "huggingface"
    # embedding_model: str = "text-embedding-3-small"
    embedding_model: EmbeddingModel = "BAAI/bge-base-en-v1.5"
    openai_api_key: str = ""          # Optional: required for openai models
    embedding_dimensions: int = 1536  # Optional: required for openai models

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
