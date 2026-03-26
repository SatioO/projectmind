# Neighbor Context Retrieval Design

**Date:** 2026-03-26
**Status:** Approved

## Problem

The current `_enrich_context` method stores 100-character previews of neighboring chunks (`next_chunk_preview`, `prev_chunk_preview`) in JSONB metadata at index time. This approach has three problems:

1. **Too short to be useful** — 100 chars is rarely meaningful context for long-form documents (5–50 pages).
2. **Doesn't improve retrieval** — the embedding of the chunk still has no awareness of its neighbors; previews don't affect vector search ranking.
3. **Orphaned data** — nothing in the retrieval pipeline consumes these fields; they bloat metadata storage with no effect.

The industry-standard solution for long-form documents is **parent document / neighbor retrieval**: retrieve on small chunks for precision, then expand context from neighboring chunks before passing to the LLM.

## Approach

**Retrieval-time neighbor expansion using `chunk_index` + `doc_id`.** The chunks table already stores `doc_id` as an explicit column and `chunk_index` in JSONB metadata. By promoting `chunk_index` to a real integer column and adding a composite index, neighbor chunks can be fetched in a single query at retrieval time — no ingestion changes, no duplicate content storage, configurable window size.

## Architecture

### Ingestion Changes

**Remove preview fields from `_enrich_context` (`core/chunker.py`):**

The `prev_chunk_preview` and `next_chunk_preview` metadata fields are removed. The remaining fields are retained:
- `chunk_index` — position of chunk within document (0-based)
- `total_chunks` — total chunk count for the document
- `is_first_chunk` — boundary flag
- `is_last_chunk` — boundary flag

**Promote `chunk_index` to an explicit column (`repository/chunks.py`):**

Add `"chunk_index"` to `_EXPLICIT_COLUMNS` so it is written as a dedicated integer column rather than into the JSONB blob. Update the INSERT statement and the row-building dict in `insert_chunks`:

```sql
INSERT INTO rag_{category}_chunks
    (doc_id, project_id, agent_id, chunk_index, content, metadata, embedding)
VALUES
    (:doc_id, :project_id, :agent_id, :chunk_index, :content, CAST(:metadata AS jsonb), CAST(:embedding AS vector))
```

The row dict construction must also include `chunk_index`:

```python
rows = [
    {
        "doc_id": chunk.metadata["doc_id"],
        "project_id": chunk.metadata["project_id"],
        "agent_id": chunk.metadata.get("agent_id"),
        "chunk_index": chunk.metadata["chunk_index"],   # promoted from JSONB
        "content": chunk.page_content,
        "metadata": json.dumps(
            {k: v for k, v in chunk.metadata.items() if k not in _EXPLICIT_COLUMNS}
        ),
        "embedding": str(embedding),
    }
    for chunk, embedding in zip(chunks, embeddings)
]
```

**DB migration** — add to each `rag_{category}_chunks` table. Use a two-phase approach to handle existing rows safely:

```sql
-- Phase 1: add nullable with a temporary default for backfill
ALTER TABLE rag_{category}_chunks ADD COLUMN chunk_index INTEGER DEFAULT 0;

-- Phase 2: backfill existing rows using row_number() ordered by id within each doc
UPDATE rag_{category}_chunks c
SET chunk_index = sub.rn - 1
FROM (
    SELECT id, row_number() OVER (PARTITION BY doc_id ORDER BY id) AS rn
    FROM rag_{category}_chunks
) sub
WHERE c.id = sub.id;

-- Phase 3: enforce NOT NULL and drop temporary default
ALTER TABLE rag_{category}_chunks ALTER COLUMN chunk_index SET NOT NULL;
ALTER TABLE rag_{category}_chunks ALTER COLUMN chunk_index DROP DEFAULT;

-- Phase 4: add the composite index
CREATE INDEX ON rag_{category}_chunks (doc_id, chunk_index);
```

The composite index `(doc_id, chunk_index)` makes neighbor lookups an index scan regardless of table size. Run this migration for each category: `prd`, `arch`, `code`, `tasks`, `ops`.

### Retrieval Layer

New function in `repository/chunks.py`:

```python
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

The caller joins the returned strings and passes the combined text to the LLM. `total_chunks` (already in chunk metadata) is used for boundary clamping — no extra query needed.

### Configuration

Add to `Settings` (`config/settings.py`):

```python
retrieval_window_size: int = 1   # neighbors to fetch on each side; override via RETRIEVAL_WINDOW_SIZE env var
```

Default of `1` returns up to 3 chunks (prev + matched + next). Increase for very long documents where wider context is needed.

## File Changes

| File | Change |
|---|---|
| `core/chunker.py` | Remove `prev_chunk_preview` / `next_chunk_preview` from `_enrich_context`; retain `chunk_index`, `total_chunks`, `is_first_chunk`, `is_last_chunk` |
| `repository/chunks.py` | Add `"chunk_index"` to `_EXPLICIT_COLUMNS`; add `chunk_index` to `INSERT`; add `fetch_chunks_with_context()` |
| `config/settings.py` | Add `retrieval_window_size: int = 1` |
| DB migration | `ALTER TABLE` + `CREATE INDEX` for each `rag_{category}_chunks` table |

No changes to `core/ingestion.py`, `services/`, `routes/`, or `repository/Ingestion.py`.

## Behavior Changes

- `prev_chunk_preview` and `next_chunk_preview` are removed from chunk metadata — no downstream consumer existed.
- `chunk_index` moves from JSONB blob to a real column — JSONB metadata no longer contains it.
- `fetch_chunks_with_context` is a new addition with no existing callers; it is ready for use when the retrieval/query endpoint is built.

## Out of Scope

- Retrieval/query endpoint implementation
- Re-ranking or scoring of expanded context
- Multi-hop neighbor expansion (e.g., fetching neighbors-of-neighbors)
