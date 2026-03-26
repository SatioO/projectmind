# Neighbor Context Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 100-char neighbor preview fields with proper retrieval-time context expansion — promoting `chunk_index` to a real DB column and adding a `fetch_chunks_with_context()` function that fetches a chunk plus its neighbors in a single query.

**Architecture:** `chunk_index` moves from the JSONB metadata blob to a dedicated integer column with a `(doc_id, chunk_index)` composite index. `prev_chunk_preview` / `next_chunk_preview` are removed from `_enrich_context`. A new `fetch_chunks_with_context()` function in `repository/chunks.py` fetches the matched chunk plus ±`window` neighbors by `doc_id` + `chunk_index` range. Window size is configurable via `Settings.retrieval_window_size`.

**Tech Stack:** Python 3.13, FastAPI, asyncpg, SQLAlchemy async, PostgreSQL (pgvector), pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-26-neighbor-context-retrieval-design.md`

**Prerequisite:** Plan 1 (chunker config refactor) must be implemented first — this plan assumes `_enrich_context` bug is already fixed and pytest is already configured.

---

### Task 1: Write and Run the DB Migration

**Files:**
- Create: `migrations/add_chunk_index.sql`

- [ ] **Step 1: Create migrations directory and SQL file**

```bash
mkdir -p migrations
```

Create `migrations/add_chunk_index.sql`:

```sql
-- Migration: promote chunk_index to a real column on all rag_*_chunks tables
-- Run once per category table. Safe for existing data via 4-phase approach.

-- ============================================================
-- rag_prd_chunks
-- ============================================================
ALTER TABLE rag_prd_chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0;

UPDATE rag_prd_chunks c
SET chunk_index = sub.rn - 1
FROM (
    SELECT id, row_number() OVER (PARTITION BY doc_id ORDER BY id) AS rn
    FROM rag_prd_chunks
) sub
WHERE c.id = sub.id;

ALTER TABLE rag_prd_chunks ALTER COLUMN chunk_index SET NOT NULL;
ALTER TABLE rag_prd_chunks ALTER COLUMN chunk_index DROP DEFAULT;
CREATE INDEX IF NOT EXISTS idx_prd_chunks_doc_chunk ON rag_prd_chunks (doc_id, chunk_index);

-- ============================================================
-- rag_arch_chunks
-- ============================================================
ALTER TABLE rag_arch_chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0;

UPDATE rag_arch_chunks c
SET chunk_index = sub.rn - 1
FROM (
    SELECT id, row_number() OVER (PARTITION BY doc_id ORDER BY id) AS rn
    FROM rag_arch_chunks
) sub
WHERE c.id = sub.id;

ALTER TABLE rag_arch_chunks ALTER COLUMN chunk_index SET NOT NULL;
ALTER TABLE rag_arch_chunks ALTER COLUMN chunk_index DROP DEFAULT;
CREATE INDEX IF NOT EXISTS idx_arch_chunks_doc_chunk ON rag_arch_chunks (doc_id, chunk_index);

-- ============================================================
-- rag_code_chunks
-- ============================================================
ALTER TABLE rag_code_chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0;

UPDATE rag_code_chunks c
SET chunk_index = sub.rn - 1
FROM (
    SELECT id, row_number() OVER (PARTITION BY doc_id ORDER BY id) AS rn
    FROM rag_code_chunks
) sub
WHERE c.id = sub.id;

ALTER TABLE rag_code_chunks ALTER COLUMN chunk_index SET NOT NULL;
ALTER TABLE rag_code_chunks ALTER COLUMN chunk_index DROP DEFAULT;
CREATE INDEX IF NOT EXISTS idx_code_chunks_doc_chunk ON rag_code_chunks (doc_id, chunk_index);

-- ============================================================
-- rag_tasks_chunks
-- ============================================================
ALTER TABLE rag_tasks_chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0;

UPDATE rag_tasks_chunks c
SET chunk_index = sub.rn - 1
FROM (
    SELECT id, row_number() OVER (PARTITION BY doc_id ORDER BY id) AS rn
    FROM rag_tasks_chunks
) sub
WHERE c.id = sub.id;

ALTER TABLE rag_tasks_chunks ALTER COLUMN chunk_index SET NOT NULL;
ALTER TABLE rag_tasks_chunks ALTER COLUMN chunk_index DROP DEFAULT;
CREATE INDEX IF NOT EXISTS idx_tasks_chunks_doc_chunk ON rag_tasks_chunks (doc_id, chunk_index);

-- ============================================================
-- rag_ops_chunks
-- ============================================================
ALTER TABLE rag_ops_chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0;

UPDATE rag_ops_chunks c
SET chunk_index = sub.rn - 1
FROM (
    SELECT id, row_number() OVER (PARTITION BY doc_id ORDER BY id) AS rn
    FROM rag_ops_chunks
) sub
WHERE c.id = sub.id;

ALTER TABLE rag_ops_chunks ALTER COLUMN chunk_index SET NOT NULL;
ALTER TABLE rag_ops_chunks ALTER COLUMN chunk_index DROP DEFAULT;
CREATE INDEX IF NOT EXISTS idx_ops_chunks_doc_chunk ON rag_ops_chunks (doc_id, chunk_index);
```

- [ ] **Step 2: Run the migration**

```bash
psql $DATABASE_URL -f migrations/add_chunk_index.sql
```

Or connect via your preferred client and execute the file. Expected: no errors, each table gets the column + index.

- [ ] **Step 3: Verify the column exists**

```bash
psql $DATABASE_URL -c "\d rag_prd_chunks"
```

Expected: `chunk_index` column of type `integer not null` is listed.

- [ ] **Step 4: Commit**

```bash
git add migrations/add_chunk_index.sql
git commit -m "feat: add chunk_index column and composite index to all rag_*_chunks tables"
```

---

### Task 2: Remove Preview Fields from `_enrich_context`

**Files:**
- Modify: `core/chunker.py`
- Modify: `tests/test_chunker.py`

- [ ] **Step 1: Remove the skip marker and enable the test**

In `tests/test_chunker.py`, find `test_enrich_context_neighbor_previews_absent` and remove the `@pytest.mark.skip` decorator if present (it was deferred from Plan 1).

- [ ] **Step 2: Run the test to confirm it fails**

```bash
uv run pytest tests/test_chunker.py::test_enrich_context_neighbor_previews_absent -v
```

Expected: FAIL — `prev_chunk_preview` is currently present in metadata.

- [ ] **Step 3: Remove preview fields from `_enrich_context` in `core/chunker.py`**

Remove the `if include_neighbors:` block entirely. The updated loop body:

```python
for i, chunk in enumerate(chunks):
    enhanced_metadata = {
        **metadata,
        "chunk_index": i,
        "total_chunks": len(chunks),
        "is_first_chunk": i == 0,
        "is_last_chunk": i == len(chunks) - 1,
    }
    chunk.metadata.update(enhanced_metadata)
    enriched_chunks.append(chunk)
```

Also remove the `include_neighbors: bool = True` parameter from the method signature since it's no longer used.

- [ ] **Step 4: Run the full chunker test suite**

```bash
uv run pytest tests/test_chunker.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/chunker.py tests/test_chunker.py
git commit -m "feat: remove neighbor preview fields from _enrich_context"
```

---

### Task 3: Promote `chunk_index` in `repository/chunks.py`

**Files:**
- Modify: `repository/chunks.py`
- Create: `tests/test_chunks_repository.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_chunks_repository.py`:

```python
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, call
from langchain_core.documents import Document

from repository.chunks import (
    insert_chunks,
    _EXPLICIT_COLUMNS,
    fetch_chunks_with_context,
)


def test_chunk_index_in_explicit_columns():
    assert "chunk_index" in _EXPLICIT_COLUMNS


async def test_insert_chunks_includes_chunk_index():
    """chunk_index must appear in the row dict passed to execute."""
    conn = MagicMock()
    conn.execute = AsyncMock()

    chunks = [
        Document(
            page_content="hello world",
            metadata={
                "doc_id": "doc1",
                "project_id": "proj1",
                "agent_id": None,
                "chunk_index": 0,
                "total_chunks": 1,
                "is_first_chunk": True,
                "is_last_chunk": True,
            },
        )
    ]
    embeddings = [[0.1, 0.2, 0.3]]

    await insert_chunks(conn, "prd", chunks, embeddings)

    rows = conn.execute.call_args[0][1]  # second positional arg to execute()
    assert rows[0]["chunk_index"] == 0


async def test_insert_chunks_chunk_index_not_in_metadata_json():
    """chunk_index must NOT appear in the JSONB metadata blob."""
    conn = MagicMock()
    conn.execute = AsyncMock()

    chunks = [
        Document(
            page_content="hello world",
            metadata={
                "doc_id": "doc1",
                "project_id": "proj1",
                "agent_id": None,
                "chunk_index": 2,
                "total_chunks": 5,
            },
        )
    ]
    embeddings = [[0.1, 0.2, 0.3]]

    await insert_chunks(conn, "prd", chunks, embeddings)

    rows = conn.execute.call_args[0][1]
    metadata = json.loads(rows[0]["metadata"])
    assert "chunk_index" not in metadata


@pytest.mark.asyncio
async def test_fetch_chunks_with_context_basic():
    """fetch_chunks_with_context returns content strings in order."""
    conn = MagicMock()
    mock_rows = [
        MagicMock(content="prev chunk"),
        MagicMock(content="matched chunk"),
        MagicMock(content="next chunk"),
    ]
    conn.execute = AsyncMock(return_value=MagicMock(fetchall=lambda: mock_rows))

    result = await fetch_chunks_with_context(
        conn, "prd", "doc1", chunk_index=1, total_chunks=3, window=1
    )

    assert result == ["prev chunk", "matched chunk", "next chunk"]


@pytest.mark.asyncio
async def test_fetch_chunks_with_context_clamps_at_start():
    """start index should not go below 0."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=MagicMock(fetchall=lambda: []))

    await fetch_chunks_with_context(
        conn, "prd", "doc1", chunk_index=0, total_chunks=10, window=2
    )

    params = conn.execute.call_args[0][1]
    assert params["start"] == 0


@pytest.mark.asyncio
async def test_fetch_chunks_with_context_clamps_at_end():
    """end index should not exceed total_chunks - 1."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=MagicMock(fetchall=lambda: []))

    await fetch_chunks_with_context(
        conn, "prd", "doc1", chunk_index=9, total_chunks=10, window=2
    )

    params = conn.execute.call_args[0][1]
    assert params["end"] == 9  # total_chunks - 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_chunks_repository.py -v
```

Expected: `chunk_index` not in `_EXPLICIT_COLUMNS`, `fetch_chunks_with_context` not found.

- [ ] **Step 3: Update `repository/chunks.py`**

```python
import json

from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from models.core import Category

_EXPLICIT_COLUMNS = {"doc_id", "project_id", "agent_id", "chunk_index"}


async def delete_chunks(
    conn: AsyncConnection,
    category: Category,
    doc_id: str,
) -> None:
    """Hard-delete all existing chunks for a doc_id before re-ingestion."""
    await conn.execute(
        text(f"DELETE FROM rag_{category}_chunks WHERE doc_id = :doc_id"),
        {"doc_id": doc_id},
    )


async def insert_chunks(
    conn: AsyncConnection,
    category: Category,
    chunks: list[Document],
    embeddings: list[list[float]],
) -> None:
    """Batch-insert chunks with pre-computed embeddings."""
    if not chunks:
        return

    rows = [
        {
            "doc_id": chunk.metadata["doc_id"],
            "project_id": chunk.metadata["project_id"],
            "agent_id": chunk.metadata.get("agent_id"),
            "chunk_index": chunk.metadata["chunk_index"],
            "content": chunk.page_content,
            "metadata": json.dumps(
                {k: v for k, v in chunk.metadata.items() if k not in _EXPLICIT_COLUMNS}
            ),
            "embedding": str(embedding),
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    await conn.execute(
        text(f"""
            INSERT INTO rag_{category}_chunks
                (doc_id, project_id, agent_id, chunk_index, content, metadata, embedding)
            VALUES
                (:doc_id, :project_id, :agent_id, :chunk_index, :content, CAST(:metadata AS jsonb), CAST(:embedding AS vector))
        """),
        rows,
    )


async def fetch_chunks_with_context(
    conn: AsyncConnection,
    category: Category,
    doc_id: str,
    chunk_index: int,
    total_chunks: int,
    window: int = 1,
) -> list[str]:
    """
    Fetch the matched chunk plus up to `window` neighbors on each side.
    Returns content strings ordered by chunk_index.
    """
    start = max(0, chunk_index - window)
    end = min(total_chunks - 1, chunk_index + window)

    result = await conn.execute(
        text(f"""
            SELECT content FROM rag_{category}_chunks
            WHERE doc_id = :doc_id
              AND chunk_index BETWEEN :start AND :end
            ORDER BY chunk_index
        """),
        {"doc_id": doc_id, "start": start, "end": end},
    )
    return [row.content for row in result.fetchall()]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_chunks_repository.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add repository/chunks.py tests/test_chunks_repository.py
git commit -m "feat: promote chunk_index to explicit column, add fetch_chunks_with_context"
```

---

### Task 4: Add `retrieval_window_size` to Settings

**Files:**
- Modify: `config/settings.py`
- Modify: `tests/test_settings.py`

- [ ] **Step 1: Add test**

Append to `tests/test_settings.py`:

```python
def test_retrieval_window_size_default():
    from config.settings import settings
    assert settings.retrieval_window_size == 1


def test_retrieval_window_size_env_override(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_WINDOW_SIZE", "3")
    from config.settings import Settings
    s = Settings()
    assert s.retrieval_window_size == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_settings.py::test_retrieval_window_size_default -v
```

Expected: FAIL — `retrieval_window_size` attribute does not exist.

- [ ] **Step 3: Add field to `config/settings.py`**

Add inside `Settings`:

```python
retrieval_window_size: int = 1
```

Place it under the `# === Chunking ===` section.

- [ ] **Step 4: Run all tests**

```bash
uv run pytest -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add config/settings.py tests/test_settings.py
git commit -m "feat: add retrieval_window_size setting"
```

---

### Task 5: Final Verification

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Verify `chunk_index` is excluded from JSONB metadata at runtime**

```bash
uv run python -c "
from repository.chunks import _EXPLICIT_COLUMNS
print('explicit:', _EXPLICIT_COLUMNS)
assert 'chunk_index' in _EXPLICIT_COLUMNS
print('ok')
"
```

Expected: `explicit: {'doc_id', 'project_id', 'agent_id', 'chunk_index'}` and `ok`.

- [ ] **Step 3: Verify `fetch_chunks_with_context` boundary clamping manually**

```bash
uv run python -c "
from repository.chunks import fetch_chunks_with_context
import inspect
src = inspect.getsource(fetch_chunks_with_context)
assert 'max(0' in src and 'min(total_chunks' in src
print('boundary clamping present')
"
```

Expected: `boundary clamping present`.
