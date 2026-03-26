from typing import Literal
from pydantic import BaseModel

Category = Literal["prd", "arch", "code", "tasks", "ops"]
EmbeddingProvider = Literal["openai", "huggingface"]
DocumentChunkingStrategy = Literal["character_based_fixed_size", "token_based_fixed_size", "semantic"]

EmbeddingModel = Literal[
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "BAAI/bge-base-en-v1.5"
]


class CharacterChunkerConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap_pct: float = 0.15


class TokenChunkerConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap_pct: float = 0.15
    encoding_name: str = "cl100k_base"


class SemanticChunkerConfig(BaseModel):
    breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile"] = "percentile"
    breakpoint_threshold_amount: float = 95.0
    buffer_size: int = 1
