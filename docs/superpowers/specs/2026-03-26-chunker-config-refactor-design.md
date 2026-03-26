# Chunker Config Refactor Design

**Date:** 2026-03-26
**Status:** Approved

## Problem

The multi-chunker strategy added in commit `bad66b7` introduced several issues:

1. **Bug:** `_enrich_context` builds `enriched_chunks` but never appends to it — always returns an empty list.
2. **Bug:** `_build_chunker` returns `None` for unhandled strategies (including `"recursive"` which is declared in the type but never handled), crashing downstream.
3. **Smell:** Embedding model is instantiated unconditionally at `DocumentChunker.__init__` time — wasteful for non-semantic strategies.
4. **Smell:** All chunker parameters (overlap %, semantic thresholds, tokenizer name) are hardcoded — no way to tune without editing source.
5. **Inconsistency:** `EmbeddingProvider` in `models/core.py` uses `"openai"` but `Settings.embedding_provider` uses `"open_ai"`.
6. **Dead type:** `"recursive"` in `DocumentChunkingStrategy` is never handled by the chunker.

## Approach

**Per-strategy config dataclasses (Approach B).** Each chunking strategy gets its own Pydantic config model with typed, default-valued fields. `Settings` holds one instance of each. `DocumentChunker._build_chunker` reads from the relevant config. No strategy-switching per-document — strategy remains a global config.

## Architecture

### Config Layer (`models/core.py`)

Three new Pydantic `BaseModel` classes:

```python
class RecursiveChunkerConfig(BaseModel):
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
```

`DocumentChunkingStrategy` drops `"recursive"` (it was never implemented — `character_based_fixed_size` already uses `RecursiveCharacterTextSplitter`). `EmbeddingProvider` is corrected to `"openai"` (was `"open_ai"`).

### Settings Layer (`config/settings.py`)

```python
chunking_strategy: DocumentChunkingStrategy = "character_based_fixed_size"
recursive_chunker: RecursiveChunkerConfig = RecursiveChunkerConfig()
token_chunker: TokenChunkerConfig = TokenChunkerConfig()
semantic_chunker: SemanticChunkerConfig = SemanticChunkerConfig()
embedding_provider: Literal["huggingface", "openai"] = "huggingface"
```

`CHUNK_SIZE_RECOMMENDATIONS` dict is removed — default `chunk_size` values per strategy live in the config dataclasses. All fields are overridable via `.env` using Pydantic nested env syntax (e.g. `RECURSIVE_CHUNKER__CHUNK_SIZE=800`).

### DocumentChunker (`core/chunker.py`)

- `_build_chunker` reads from the appropriate config object per strategy branch.
- Embedding model is **lazy-loaded** via `_get_embedding_model()` — only called inside the `"semantic"` branch of `_build_chunker`, not in `__init__`.
- Explicit `raise ValueError(f"Unknown chunking strategy: {strategy!r}")` at the end of `_build_chunker` instead of silently returning `None`.
- `_enrich_context` bug fixed: append each `chunk` to `enriched_chunks` inside the loop before returning it.

## File Changes

| File | Change |
|---|---|
| `models/core.py` | Add `RecursiveChunkerConfig`, `TokenChunkerConfig`, `SemanticChunkerConfig`; fix `EmbeddingProvider` (`"open_ai"` → `"openai"`); drop `"recursive"` from `DocumentChunkingStrategy` |
| `config/settings.py` | Add three nested config fields; fix `embedding_provider` literal; remove `CHUNK_SIZE_RECOMMENDATIONS` |
| `core/chunker.py` | Refactor `_build_chunker` to use config objects; lazy-load embedding model; fix `_enrich_context`; add `ValueError` fallback |

No changes to `core/ingestion.py`, `services/`, `routes/`, or `repository/` — public interface unchanged.

## Out of Scope

- Per-document chunking strategy selection (remains global config)
- New chunking strategies
- Changes to the embedding layer
