# Chunker Config Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded chunker parameters with per-strategy Pydantic config models, fix two silent bugs in `_enrich_context` and `_build_chunker`, and make the embedding model lazy-loaded.

**Architecture:** Three new `BaseModel` config classes (`CharacterChunkerConfig`, `TokenChunkerConfig`, `SemanticChunkerConfig`) are added to `models/core.py` and held as nested fields on `Settings`. `DocumentChunker._build_chunker` reads from the active config instead of hardcoded values. The embedding model is only instantiated when the `"semantic"` strategy is selected.

**Tech Stack:** Python 3.13, FastAPI, pydantic-settings v2, LangChain (`langchain-text-splitters`, `langchain-experimental`), pytest

**Spec:** `docs/superpowers/specs/2026-03-26-chunker-config-refactor-design.md`

---

### Task 1: Set Up pytest

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add pytest to pyproject.toml**

Add under `[project]` dependencies or create a dev dependency group:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
```

- [ ] **Step 2: Add pytest config to pyproject.toml**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Install dev dependencies**

```bash
uv sync --group dev
```

Expected: dependencies resolved, `.venv` updated.

- [ ] **Step 4: Create test scaffolding**

```bash
mkdir -p tests
touch tests/__init__.py tests/conftest.py
```

- [ ] **Step 5: Verify pytest works**

```bash
uv run pytest --collect-only
```

Expected: `no tests ran` (no errors).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/
git commit -m "chore: add pytest with asyncio support"
```

---

### Task 2: Add Config Models to `models/core.py`

**Files:**
- Modify: `models/core.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models.py`:

```python
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
    # "recursive" should no longer be a valid strategy
    import typing
    args = typing.get_args(DocumentChunkingStrategy)
    assert "recursive" not in args


def test_valid_chunking_strategies():
    import typing
    args = typing.get_args(DocumentChunkingStrategy)
    assert set(args) == {"character_based_fixed_size", "token_based_fixed_size", "semantic"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_models.py -v
```

Expected: `ImportError` — `CharacterChunkerConfig` not yet defined.

- [ ] **Step 3: Update `models/core.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_models.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add models/core.py tests/test_models.py
git commit -m "feat: add per-strategy chunker config models, drop 'recursive' strategy type"
```

---

### Task 3: Update `config/settings.py`

**Files:**
- Modify: `config/settings.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_settings.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_settings.py -v
```

Expected: failures on missing `character_chunker`, `CHUNK_SIZE_RECOMMENDATIONS` still present, `"open_ai"` still valid.

- [ ] **Step 3: Update `config/settings.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_settings.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add config/settings.py tests/test_settings.py
git commit -m "feat: add nested chunker config to Settings, fix embedding_provider literal"
```

---

### Task 4: Refactor `core/chunker.py`

**Files:**
- Modify: `core/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_chunker.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document


def test_enrich_context_returns_all_chunks():
    """_enrich_context must return the same number of chunks it receives."""
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)  # skip __init__
    docs = [Document(page_content=f"chunk {i}") for i in range(3)]
    result = chunker._enrich_context(docs, {"doc_id": "test"})
    assert len(result) == 3


def test_enrich_context_sets_chunk_index():
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)
    docs = [Document(page_content=f"chunk {i}") for i in range(3)]
    result = chunker._enrich_context(docs, {"doc_id": "test"})
    assert result[0].metadata["chunk_index"] == 0
    assert result[1].metadata["chunk_index"] == 1
    assert result[2].metadata["chunk_index"] == 2


def test_enrich_context_boundary_flags():
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)
    docs = [Document(page_content=f"chunk {i}") for i in range(3)]
    result = chunker._enrich_context(docs, {"doc_id": "test"})
    assert result[0].metadata["is_first_chunk"] is True
    assert result[0].metadata["is_last_chunk"] is False
    assert result[2].metadata["is_first_chunk"] is False
    assert result[2].metadata["is_last_chunk"] is True


def test_build_chunker_character_returns_recursive_splitter():
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)
    splitter = chunker._build_chunker("character_based_fixed_size")
    assert isinstance(splitter, RecursiveCharacterTextSplitter)


def test_build_chunker_token_returns_character_splitter():
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)
    splitter = chunker._build_chunker("token_based_fixed_size")
    assert isinstance(splitter, CharacterTextSplitter)


def test_build_chunker_unknown_raises():
    from core.chunker import DocumentChunker
    chunker = DocumentChunker.__new__(DocumentChunker)
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunker._build_chunker("not_a_strategy")


def test_embedding_model_not_loaded_for_character_strategy(monkeypatch):
    """Embedding model should NOT be instantiated for non-semantic strategies."""
    from core import chunker as chunker_module
    call_count = {"n": 0}

    original = chunker_module.embedding.get_embedding_model

    def counting_get(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(chunker_module.embedding, "get_embedding_model", counting_get)

    from config.settings import Settings
    from models.core import CharacterChunkerConfig
    with patch("core.chunker.settings") as mock_settings:
        mock_settings.chunking_strategy = "character_based_fixed_size"
        mock_settings.character_chunker = CharacterChunkerConfig()
        from core.chunker import DocumentChunker
        DocumentChunker()

    assert call_count["n"] == 0, "Embedding model should not be loaded for character strategy"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_chunker.py -v
```

Expected: `test_enrich_context_returns_all_chunks` FAILS (returns 0 chunks), `test_build_chunker_unknown_raises` FAILS (returns None), `test_embedding_model_not_loaded_for_character_strategy` FAILS.

Note: `prev_chunk_preview` / `next_chunk_preview` are intentionally retained in Plan 1 — they are removed in Plan 2 (neighbor context retrieval). Do not remove them here.

- [ ] **Step 3: Rewrite `core/chunker.py`**

```python
from typing import Dict, List, Union
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from models.core import DocumentChunkingStrategy
from config.settings import settings
from core.embedding import embedding


class DocumentChunker:
    def __init__(self) -> None:
        self._chunker = self._build_chunker(strategy=settings.chunking_strategy)

    def split_documents(
        self,
        content: Union[str, List[str]],
        metadata: Dict | None = None,
    ) -> List[Document]:
        texts = self._normalize_input(content)
        chunks = self._chunker.create_documents(texts)
        if metadata:
            chunks = self._enrich_context(chunks, metadata)
        return chunks

    def _enrich_context(
        self,
        chunks: List[Document],
        metadata: Dict,
        include_neighbors: bool = True,
    ) -> List[Document]:
        enriched_chunks: List[Document] = []

        for i, chunk in enumerate(chunks):
            enhanced_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "is_first_chunk": i == 0,
                "is_last_chunk": i == len(chunks) - 1,
            }

            if include_neighbors:
                if i > 0:
                    enhanced_metadata["previous_chunk_preview"] = chunks[i - 1].page_content[:100]
                if i < len(chunks) - 1:
                    enhanced_metadata["next_chunk_preview"] = chunks[i + 1].page_content[:100]

            chunk.metadata.update(enhanced_metadata)
            enriched_chunks.append(chunk)  # fix: was missing

        return enriched_chunks

    def _get_embedding_model(self):
        return embedding.get_embedding_model(provider=settings.embedding_provider)

    def _build_chunker(
        self, strategy: DocumentChunkingStrategy
    ) -> Union[RecursiveCharacterTextSplitter, CharacterTextSplitter, SemanticChunker]:
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

    def _normalize_input(self, content: Union[str, List[str]]) -> List[str]:
        if isinstance(content, str):
            return [content]
        return content


chunker = DocumentChunker()
```

Note: `previous_chunk_preview` / `next_chunk_preview` are retained here intentionally — they will be removed in the **neighbor context retrieval plan** (Plan 2), which also removes these fields from `_enrich_context`. Do not remove them in this task.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_chunker.py -v
```

Expected: all tests PASS except `test_enrich_context_neighbor_previews_absent` — this test is intentionally for Plan 2. Skip it for now by marking `@pytest.mark.skip(reason="removed in Plan 2")`.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add core/chunker.py tests/test_chunker.py
git commit -m "feat: refactor chunker to use config models, fix _enrich_context bug, lazy embedding"
```

---

### Task 5: Final Verification

- [ ] **Step 1: Run full test suite once more**

```bash
uv run pytest -v
```

Expected: all tests PASS, no warnings about unknown strategies.

- [ ] **Step 2: Verify the app starts without errors**

```bash
uv run python -c "from core.chunker import chunker; print('chunker ok:', type(chunker._chunker).__name__)"
```

Expected output: `chunker ok: RecursiveCharacterTextSplitter`

- [ ] **Step 3: Verify env_nested_delimiter works**

```bash
CHARACTER_CHUNKER__CHUNK_SIZE=800 uv run python -c "from config.settings import Settings; s = Settings(); print(s.character_chunker.chunk_size)"
```

Expected output: `800`
