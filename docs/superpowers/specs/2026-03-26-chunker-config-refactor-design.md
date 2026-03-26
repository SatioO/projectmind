# Chunker Config Refactor Design

**Date:** 2026-03-26
**Status:** Approved

## Problem

The multi-chunker strategy added in commit `bad66b7` introduced several issues:

1. **Bug:** `_enrich_context` in `core/chunker.py` builds `enriched_chunks` list but never appends to it — always returns an empty list, so metadata enrichment is silently lost.
2. **Bug:** `_build_chunker` has no fallback branch — if `"recursive"` (a value present in `DocumentChunkingStrategy`) were passed, `None` would be returned and `create_documents` would crash at runtime.
3. **Dead type:** `"recursive"` is in `DocumentChunkingStrategy` but never handled. It was never user-facing (no `.env` default, no docs) and can be dropped. After removal, any env file that sets `CHUNKING_STRATEGY=recursive` will fail at startup with a Pydantic validation error — this is the intended behavior.
4. **Smell:** Embedding model is instantiated unconditionally in `DocumentChunker.__init__` — wasteful for non-semantic strategies.
5. **Smell:** All chunker parameters (overlap %, semantic thresholds, tokenizer encoding name) are hardcoded — no way to tune without editing source.
6. **Inconsistency:** `config/settings.py` uses `Literal["huggingface", "open_ai"]` for `embedding_provider` but `models/core.py` `EmbeddingProvider` correctly uses `"openai"`. Any existing env file using `EMBEDDING_PROVIDER=open_ai` must be updated to `EMBEDDING_PROVIDER=openai`.

## Approach

**Per-strategy config dataclasses.** Each chunking strategy gets its own Pydantic config model with typed, default-valued fields. `Settings` holds one instance of each. `DocumentChunker._build_chunker` reads from the relevant config. Strategy remains global config — no per-document switching.

## Architecture

### Config Models (`models/core.py`)

Three new `pydantic.BaseModel` classes added. Names match the strategy they configure:

```python
class CharacterChunkerConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap_pct: float = 0.15          # fraction; converted to int(chunk_size * pct) at build time

class TokenChunkerConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap_pct: float = 0.15          # preserves existing behavior (was also int(chunk_size * 0.15))
    encoding_name: str = "cl100k_base"

class SemanticChunkerConfig(BaseModel):
    breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile"] = "percentile"
    breakpoint_threshold_amount: float = 95.0
    buffer_size: int = 1
```

`DocumentChunkingStrategy` drops `"recursive"` (never implemented, never user-facing).
`EmbeddingProvider` in `models/core.py` is already correct (`"openai"`); no change needed there.

### Settings (`config/settings.py`)

```python
class Settings(BaseSettings):
    chunking_strategy: DocumentChunkingStrategy = "character_based_fixed_size"
    character_chunker: CharacterChunkerConfig = CharacterChunkerConfig()
    token_chunker: TokenChunkerConfig = TokenChunkerConfig()
    semantic_chunker: SemanticChunkerConfig = SemanticChunkerConfig()
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"  # fix "open_ai" → "openai"
    # ... rest unchanged

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",           # required for nested model env vars to work
        extra="ignore",
    )
```

`CHUNK_SIZE_RECOMMENDATIONS` dict is removed — only referenced in `core/chunker.py` and `config/settings.py`, safe to delete. Default `chunk_size` values now live in the per-strategy config dataclasses. Nested fields overridable via `.env` using double-underscore delimiter (e.g. `CHARACTER_CHUNKER__CHUNK_SIZE=800`).

### DocumentChunker (`core/chunker.py`)

**`_build_chunker` refactor** — reads from config objects, no magic numbers:

```python
def _build_chunker(self, strategy: DocumentChunkingStrategy):
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
```

`chunk_overlap_pct` is converted to an absolute integer via `int(chunk_size * chunk_overlap_pct)` before being passed to splitters (both require `int`).

**Lazy embedding model** — `__init__` no longer instantiates the embedding model. `_get_embedding_model()` is a thin helper that delegates to `embedding.get_embedding_model(provider=settings.embedding_provider)` (the existing singleton in `core/embedding.py` which already caches by provider). No separate `self._embedding_model` attribute is needed. It is only called from the `"semantic"` branch of `_build_chunker`. The `settings.embedding_provider` fix (problem #6) is a prerequisite for this delegation to be type-safe.

**`_enrich_context` bug fix** — append each mutated `chunk` to `enriched_chunks` inside the loop:

```python
for i, chunk in enumerate(chunks):
    chunk.metadata.update(enhanced_metadata)
    enriched_chunks.append(chunk)   # was missing — caused empty return
return enriched_chunks
```

**Explicit fallback** — `raise ValueError` at end of `_build_chunker` instead of implicit `None` return.

## File Changes

| File | Change |
|---|---|
| `models/core.py` | Add `CharacterChunkerConfig`, `TokenChunkerConfig`, `SemanticChunkerConfig`; drop `"recursive"` from `DocumentChunkingStrategy` |
| `config/settings.py` | Add three nested config fields (`character_chunker`, `token_chunker`, `semantic_chunker`); add `env_nested_delimiter="__"` to `SettingsConfigDict`; fix `embedding_provider` literal (`"open_ai"` → `"openai"`); remove `CHUNK_SIZE_RECOMMENDATIONS` |
| `core/chunker.py` | Refactor `_build_chunker` to read from config objects; make embedding model lazy via `_get_embedding_model()` delegation; fix `_enrich_context` empty-return bug; add `ValueError` fallback; remove `CHUNK_SIZE_RECOMMENDATIONS` import |

No changes to `core/ingestion.py`, `services/`, `routes/`, or `repository/` — public interface unchanged.

## Behavior Changes

- All chunker parameter defaults match current hardcoded values — no behavior change for existing deployments.
- `TokenChunkerConfig.chunk_overlap_pct = 0.15` preserves existing behavior (current code uses `int(chunk_size * 0.15)` for both strategies).
- `"recursive"` removed from `DocumentChunkingStrategy` — any env with `CHUNKING_STRATEGY=recursive` will now fail at startup with a Pydantic validation error (previously returned `None` and crashed at call time).
- `EMBEDDING_PROVIDER=open_ai` in env files must be updated to `EMBEDDING_PROVIDER=openai`.

## Out of Scope

- Per-document chunking strategy selection (remains global config)
- New chunking strategies
- Changes to the embedding layer
