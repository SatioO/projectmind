from typing import Literal

Category = Literal["prd", "arch", "code", "tasks", "ops"]
EmbeddingProvider = Literal["openai", "huggingface"]
DocumentChunkingStrategy = Literal["character_based_fixed_size", "token_based_fixed_size",
                                   "semantic", "recursive"]
