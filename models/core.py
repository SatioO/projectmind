from typing import Literal

Category = Literal["prd", "arch", "code", "tasks", "ops"]
EmbeddingProvider = Literal["openai", "huggingface"]
DocumentChunkingStrategy = Literal["character_based_fixed_size", "token_based_fixed_size",
                                   "semantic", "recursive"]

EmbeddingModel = Literal[
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "BAAI/bge-base-en-v1.5"
]
