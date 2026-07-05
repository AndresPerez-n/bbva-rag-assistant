# Databases — how to inspect them visually

The system uses **two data stores**, each with the right visual tool:

| Store | Contents | Visual tool |
|---|---|---|
| **SQLite** (`data/history.db`) | conversation history + feedback | **DBeaver** (native SQLite) |
| **Qdrant** (`bbva_web` collection) | 305 embedded vectors + payloads | **Qdrant dashboard** (built-in) |

DBeaver has no Qdrant driver (Qdrant is a vector DB, not relational), so the
relational conversation store is shown in DBeaver and the vector store in
Qdrant's own dashboard — which is the better visual for vectors anyway.

---

## 1. Conversation DB in DBeaver (SQLite)

The DB lives on the host at `./data/history.db` (mounted from the container),
so DBeaver opens it directly — no server needed.

1. **Database → New Database Connection → SQLite.**
2. **Path:** browse to `…\bbva-rag-assistant\data\history.db`.
   - If DBeaver offers to download the SQLite driver, accept.
3. **Finish**, then expand **Tables**:
   - **`messages`** — `id, session_id, role, content, created_at, metadata (JSON)`
     - assistant rows carry metadata: `confidence`, `top_score`, `num_chunks`,
       `sources`, `latency_ms`, `model`, `crosscheck_agree`.
   - **`feedback`** — `id, session_id, message_id, rating (1/-1), created_at`.

Sample queries to run live in DBeaver:

```sql
-- full transcript of a session
SELECT id, role, substr(content,1,80) AS preview, created_at
FROM messages WHERE session_id = 'demo' ORDER BY id;

-- answers with their retrieval confidence and latency
SELECT id,
       json_extract(metadata,'$.confidence')  AS confidence,
       json_extract(metadata,'$.top_score')   AS top_score,
       json_extract(metadata,'$.latency_ms')  AS latency_ms,
       json_extract(metadata,'$.crosscheck_agree') AS faiss_agree
FROM messages WHERE role = 'assistant' ORDER BY id DESC;

-- feedback tally
SELECT rating, COUNT(*) FROM feedback GROUP BY rating;
```

> This is the same data the analytics feature aggregates
> (`python -m scripts.run_analytics`).

---

## 2. Vector DB in the Qdrant dashboard

While the stack is running (`docker compose up -d`):

1. Open **http://localhost:6333/dashboard**.
2. Open the **`bbva_web`** collection → **~305 points**.
3. Each point shows its **payload** (`text`, `url`, `title`, `chunk_index`) and
   you can browse/scroll the vectors and run visual similarity queries.

This is the same collection the retriever searches and the FAISS cross-check
mirrors in memory.
