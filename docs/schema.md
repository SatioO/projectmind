# 5.1 Five chunk tables (identical schema, different `table_name`)

```sql
-- Repeated for: rag_prd_chunks, rag_arch_chunks, rag_code_chunks,
-- rag_tasks_chunks, rag_ops_chunks
CREATE TABLE rag_ < category > _chunks(
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id      TEXT        NOT NULL,          -- parent document ULID
    version     INTEGER     NOT NULL DEFAULT 1,
    project_id  TEXT        NOT NULL,          -- hard isolation boundary
    agent_id    TEXT,                          -- NULL=project-scoped
    content     TEXT        NOT NULL,          -- raw chunk text
    summary     TEXT,                          -- generated at index time(cheap LLM)
    metadata    JSONB       NOT NULL DEFAULT '{}',
    -- metadata shape:
    -- {
        -- "project_id": "...",
        -- "agent_id": "..." | null,
        -- "doc_id": "...",
        -- "version": 3,
        -- "category": "arch",
        -- "component": "auth-service",   ← optional, from upload metadata
        - -   "language": "rust",            ← optional
        - -   "env": "prod",                 ← optional
        - -   "spec_id": "JIRA-1234"         ← optional
        - -}
    source      TEXT,                          -- e.g. "upload:prd_v3.pdf"
    embedding   vector(1536) NOT NULL,
    fts         tsvector GENERATED ALWAYS AS(to_tsvector('english', content)) STORED,
    deleted_at  TIMESTAMPTZ,                   -- NULL=active
    soft-delete on re-index
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)

-- Vector index: HNSW, cosine distance
CREATE INDEX ON rag_ < category > _chunks
USING hnsw(embedding vector_cosine_ops)
WITH(m=16, ef_construction=64)

-- Keyword index: GIN on generated tsvector
CREATE INDEX ON rag_ < category > _chunks USING GIN(fts)

-- Pre-filter index: btree on(project_id, deleted_at) — used before every ANN scan
CREATE INDEX ON rag_ < category > _chunks(project_id, deleted_at)

-- Versioning/lookup index
CREATE INDEX ON rag_ <category>_chunks (doc_id, version);



### 5.2 Ingestion jobs table

```sql
CREATE TABLE doc_ingestion_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          TEXT        NOT NULL UNIQUE,
    project_id      TEXT        NOT NULL,
    agent_id        TEXT,                      -- NULL = project-scoped
    category        TEXT        NOT NULL,      -- prd | arch | code | tasks | ops
    scope           TEXT        NOT NULL,      -- project | agent
    filename        TEXT,
    status          TEXT        NOT NULL DEFAULT 'processing',
    -- status values: processing | done | failed
    chunks_total    INTEGER,                   -- set once chunking completes
    chunks_done     INTEGER     NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON doc_ingestion_jobs (project_id, status);
CREATE INDEX ON doc_ingestion_jobs (doc_id);