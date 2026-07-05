# BBVA RAG Assistant

A Retrieval-Augmented Generation (RAG) system that lets internal users ask
questions about the content published on a bank's website
(<https://www.bbva.com.co/>) without searching it manually.

The system **scrapes** the site, **stores** the raw and clean data locally,
**vectorizes and indexes** it in a vector database, and exposes a **conversational
interface** (web + CLI) that answers grounded in the scraped content, with source
citations, per-session persisted history (last *N* messages, configurable), and a
**conversation-analytics** report.

---

## Table of contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Quick start (Docker)](#quick-start-docker)
4. [Using the assistant](#using-the-assistant)
5. [Conversation analytics](#conversation-analytics)
6. [Design patterns](#design-patterns)
7. [Tech stack and rationale](#tech-stack-and-rationale)
8. [Externalized configuration](#externalized-configuration)
9. [Error handling](#error-handling)
10. [Known limitations and design decisions](#known-limitations-and-design-decisions)
11. [Future improvements](#future-improvements)

---

## Architecture

```
                    ┌──────────────┐     raw HTML (crudos)  ┌─────────────┐
  bbva.com.co  ───► │   Scraper    │ ─────────────────────► │  data/raw   │
                    │ (Playwright) │     clean JSON (limpios)│  data/clean │
                    └──────┬───────┘ ───────────────────────►└─────┬───────┘
                           │                                       │
                           ▼                                       ▼
                    ┌──────────────┐   chunk → embed → upsert ┌─────────────┐
                    │  Ingestion   │ ───────────────────────► │   Qdrant    │
                    │  (chunker +  │                          │ (vector DB) │
                    │   embedder)  │                          └─────┬───────┘
                    └──────────────┘                                │ search
                                                                    ▼
   ┌──────────┐   query   ┌──────────────┐   retrieve   ┌───────────────────┐
   │ Web UI / │ ────────► │   Chatbot    │ ───────────► │ Retriever+Reranker│
   │   CLI    │ ◄──────── │ (orchestrator)│ ◄─────────── │ (two-stage)       │
   └──────────┘  answer   └──────┬───────┘   context     └───────────────────┘
                                 │  ▲                              │
                       history   │  │ last N msgs                  ▼
                                 ▼  │                       ┌────────────┐
                          ┌──────────────┐                  │ LLM (OpenAI)│
                          │  SQLite      │                  └────────────┘
                          │ (history +   │
                          │  feedback)   │ ──► Analytics (metrics over history)
                          └──────────────┘
```

Pipeline: **scrape → store raw+clean → chunk → embed → index (Qdrant) →
retrieve → rerank → generate (grounded, cited) → persist history → analytics.**

---

## Prerequisites

- **Docker** and **Docker Compose** (v2). Nothing else is required to run the system.
- An **OpenAI API key** (GPT is the default generation model). The rest of
  the stack — embeddings, reranker, vector DB — is free / self-hosted.
  - You can switch to Claude by setting `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`.
- Outbound internet access from the containers (to reach the bank site, the
  LLM API, and to download the local embedding/reranker models on first run).

Environment variables are documented in [`.env.example`](.env.example). The only
one you *must* set is `OPENAI_API_KEY`.

---

## Quick start (Docker)

Everything runs with Docker; the vector database and the app come up together.

```bash
# 1. Clone
git clone https://github.com/AndresPerez-n/bbva-rag-assistant.git
cd bbva-rag-assistant

# 2. Configure environment
cp .env.example .env
#   then edit .env and set OPENAI_API_KEY=sk-...

# 3. Build and start all services (app + Qdrant)
docker compose up -d --build

# 4. Populate the knowledge base: scrape the site, then index it.
#    (one-off commands run inside the app container)
docker compose exec app python -m scripts.bootstrap
#    ^ equivalent to: run_scrape then run_ingest

# 5. Open the chat UI
#    http://localhost:8000
```

That's it. The `bootstrap` step scrapes `SCRAPE_START_URL` into `data/raw`
(raw HTML) and `data/clean` (clean JSON), then chunks, embeds, and indexes the
content into Qdrant.

> **Tip:** the Qdrant dashboard is available at
> <http://localhost:6333/dashboard> to inspect the indexed collection.

To stop everything: `docker compose down` (add `-v` to also remove the vector
and cache volumes).

---

## Using the assistant

### Web interface

Open <http://localhost:8000>. Type a question; the answer streams token-by-token,
shows a **confidence badge**, lists **source links**, and offers **👍 / 👎 feedback**.
Change the **session id** field (top-right) to keep separate conversation
histories — the assistant remembers the last *N* messages of the active session.

### CLI

```bash
docker compose exec app python -m src.cli --session my-session
```

Slash commands: `/session <id>`, `/history`, `/up`, `/down`, `/analytics`,
`/help`, `/quit`.

### API (for integration)

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`  | `/health` | status + indexed chunk count |
| `POST` | `/chat` | non-streaming answer (JSON) |
| `POST` | `/chat/stream` | streaming answer (Server-Sent Events) |
| `POST` | `/feedback` | thumbs up/down |
| `GET`  | `/sessions` / `/sessions/{id}` | list / read history |
| `GET`  | `/analytics` | aggregated metrics (JSON) |

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué tarjetas de crédito ofrece el banco?", "session_id": "demo"}'
```

Interactive API docs (Swagger) at <http://localhost:8000/docs>.

---

## Conversation analytics

A dedicated feature walks the persisted conversation history and extracts metrics
and impact values:

```bash
docker compose exec app python -m scripts.run_analytics          # formatted tables
docker compose exec app python -m scripts.run_analytics --json   # raw JSON
```

Reported metrics:

- **Volume** — sessions, messages, user turns, average turns per session.
- **Quality** — confidence distribution (high/medium/low/out-of-scope),
  out-of-scope rate, average retrieval score.
- **Performance** — average / p95 / max answer latency.
- **Satisfaction** — thumbs up/down counts and satisfaction rate.
- **Content** — top questions asked, most-cited source URLs.

The same data is available programmatically at `GET /analytics` and from the CLI
via `/analytics`.

---

## Design patterns

Five patterns are applied where they earn their place (the test asks for at least
three):

| Pattern | Type | Where | Why |
|---|---|---|---|
| **Singleton** | Creational | [`src/config.py`](src/config.py) — `Settings` / `get_settings()` | One immutable, process-wide source of configuration read once from the environment; avoids re-parsing env and scattering `os.getenv` calls. |
| **Factory Method** | Creational | [`src/ingestion/embedder.py`](src/ingestion/embedder.py) — `EmbedderFactory`; [`src/rag/llm.py`](src/rag/llm.py) — `LLMFactory` | The provider/model is chosen from config; callers depend only on the `Embedder` / `LLMClient` interface, so swapping Claude↔OpenAI or the embedding model is a config change, not a code change. |
| **Template Method** | Behavioral | [`src/scraper/base.py`](src/scraper/base.py) — `BaseScraper.run()` | The crawl/persist algorithm is fixed; only the `fetch()` step varies. `PlaywrightScraper` and `HttpScraper` supply that one step and inherit the whole workflow. |
| **Strategy** | Behavioral | [`src/rag/reranker.py`](src/rag/reranker.py) — `Reranker` (`NoOpReranker` / `CrossEncoderReranker`) | Re-ranking is an interchangeable algorithm selected from config; the retriever never branches on which one is active. |
| **Repository** | Structural | [`src/rag/vector_store.py`](src/rag/vector_store.py) — `VectorStore` / `QdrantVectorStore` | A storage-agnostic interface over the vector DB (also reads as an **Adapter** over the Qdrant client); the database could be swapped without touching ingestion or retrieval. |

---

## Tech stack and rationale

| Layer | Choice | Why |
|---|---|---|
| **Language** | Python 3.12 | Required. |
| **Scraping** | Playwright (headless Chromium) + BeautifulSoup | BBVA's site is client-rendered behind bot protection; a plain HTTP GET returns an empty shell, so a real browser is needed. BeautifulSoup handles clean-text extraction. A stdlib HTTP fallback is included. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local) | Free, runs on CPU, no API cost; 384-dim vectors, normalized so cosine = dot product. Preferred (per the brief) over paid embedding APIs. |
| **Vector DB** | **Qdrant** (self-hosted service) | A real, production-grade vector database that runs as its own container (fits "bring up the vector DB with one command"), free/self-hosted, with native cosine search and metadata payloads for citations and future access filters. |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) | Bonus. A cross-encoder re-scores (query, chunk) pairs jointly — far more precise than first-stage bi-encoder similarity — and is free/local. |
| **LLM** | **GPT (OpenAI)**, model-swappable | Strong grounded-answer quality and instruction-following for a citation-strict assistant. Externalized via `LLM_PROVIDER` / `LLM_MODEL`; Claude (Anthropic) is supported as an alternative through the same factory. |
| **Backend** | FastAPI + Uvicorn | Async, automatic OpenAPI docs, Pydantic validation, native streaming (SSE). |
| **History** | SQLite (stdlib) | Persists conversation history and feedback keyed by session id with zero extra services; queryable by the analytics module. |
| **Frontend** | Single-file HTML/JS (no build step) | Minimal, functional, dependency-free; streams tokens over SSE and renders sources + feedback. |
| **Orchestration** | Docker + Docker Compose | One command brings up the app and the Qdrant service. |

---

## Externalized configuration

Every tunable parameter is read from the environment (`.env`); nothing is
hard-coded. See [`.env.example`](.env.example). Highlights:

- `CONVERSATION_WINDOW` — **N** previous messages kept per session (configurable).
- `LLM_PROVIDER`, `LLM_MODEL`, `LLM_MAX_TOKENS` — generation backend and model.
- `CHUNK_SIZE`, `CHUNK_OVERLAP` — chunking.
- `TOP_K_RETRIEVAL`, `TOP_K_FINAL`, `SIMILARITY_THRESHOLD` — retrieval.
- `RERANKER_ENABLED`, `RERANKER_MODEL` — reranker (bonus).
- `EMBEDDING_MODEL`, `QDRANT_URL`, `QDRANT_COLLECTION` — index.
- `SCRAPE_START_URL`, `SCRAPE_MAX_PAGES`, `SCRAPE_MAX_DEPTH` — crawl scope.

---

## Error handling

- **Scraper** — a failed page fetch is logged and skipped, never aborts the crawl;
  Playwright falls back to a stdlib HTTP scraper if Chromium is unavailable.
- **Retrieval / out-of-scope** — if nothing clears the similarity threshold, the
  assistant returns a clear "not in the site content" message **without** calling
  the LLM (no token spend, no hallucination from model memory).
- **LLM** — generation errors are caught and surfaced as a readable message rather
  than a crash; the reranker degrades to a no-op if its model can't load.
- **API** — Pydantic validates every request; endpoints return `503` until the
  chatbot has finished loading.

---

## Known limitations and design decisions

Honest notes on shortcuts and assumptions:

- **Assumption — scope of "the site".** The crawler is bounded by
  `SCRAPE_MAX_PAGES` / `SCRAPE_MAX_DEPTH` (defaults 40 / 2) so the demo indexes a
  representative slice of the site, not its entirety. Raise them to index more.
- **BBVA anti-bot / JS rendering.** BBVA Colombia is client-rendered and sits
  behind Akamai bot management, which returns HTTP 403 to a naked headless
  browser. The scraper defeats this with realistic client-hint headers and a
  small stealth init script (spoofing `navigator.webdriver`/languages/plugins) —
  verified returning HTTP 200 and full content. This is inherently fragile: if
  BBVA tightens its WAF the headers may need updating, and the brief explicitly
  allows another bank (`SCRAPE_START_URL` is a single env var).
- **Embeddings and Spanish.** The default `all-MiniLM-L6-v2` is English-first, so
  Spanish similarity scores are compressed (which weakens the similarity-based
  out-of-scope gate — the grounding prompt is the real safety net and refuses
  off-topic questions correctly). A **multilingual** embedding model
  (`paraphrase-multilingual-MiniLM-L12-v2`, see `.env.example`) ranks Spanish
  content noticeably better and is recommended where the ~470MB download is viable.
- **Chunking / parsing.** Text-only extraction; tables and PDFs linked from the
  site are not parsed structurally (would need e.g. Unstructured.io).
- **Confidence bands** are calibrated to the MiniLM cosine-score distribution; a
  different embedding model would need re-calibration.
- **No authentication.** The internal-user access perimeter is out of scope for
  this prototype; Qdrant payload metadata is in place to support per-department
  filtering later.
- **Image size.** The app image is based on the Playwright image (Chromium
  included), which is large but makes in-container scraping reliable.

---

## Future improvements

- **Incremental / scheduled re-scraping** to keep the index fresh (webhook or cron).
- **Hybrid retrieval** (BM25 + vector) before re-ranking for better recall on rare terms.
- **Access control** via Qdrant metadata filters + authentication (JWT / SSO).
- **Evaluation harness** (RAGAS: faithfulness, context precision/recall) on a
  golden Q&A set, run in CI.
- **Structured parsing** of tables/PDFs (Unstructured.io / Document AI).
- **Observability** (Langfuse tracing) and a semantic cache (Redis) for latency/cost.
- **Two-tier memory**: verbatim recent turns + a summarized long-term history.
```
