from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # === POSTGRES ===
    postgres_dsn: str = "postgresql://postgres:password@localhost:5432/graphmind"
    # TODO: bring in pgsearch support for BM25 search capabilities
    postgres_plugin: Literal["pgvector", "pgsearch"] = "pgvector"
    log_level: str = "INFO"

    # === Chunking ===
    embedding_provider: Literal["huggingface", "open_ai"] = "huggingface"
    # embedding_model: str = "text-embedding-3-small"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    openai_api_key: str = ""          # Optional: required for openai models
    embedding_dimensions: int = 1536  # Optional: required for openai models

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
