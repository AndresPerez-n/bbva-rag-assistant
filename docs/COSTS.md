# Cost Model — current solution

What it costs to run this system. The design keeps **everything except the LLM
free**: embeddings and the reranker run locally, and the vector database is
self-hosted. The only per-query cost is the LLM API; the only fixed cost is
compute to host the containers.

> Prices below are OpenAI `gpt-4o-mini` list prices at the time of writing
> (~$0.15 / 1M input tokens, ~$0.60 / 1M output tokens). **Verify current
> pricing** — model prices change. All figures are estimates.

## Cost per component

| Component | What it is | Cost |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2`, local on CPU | **$0** (no API) |
| Reranker | cross-encoder, local on CPU | **$0** (no API) |
| Vector DB | Qdrant, self-hosted container | **$0** software (compute only) |
| FAISS cross-check | in-memory, local | **$0** |
| Conversation history | SQLite, local file | **$0** |
| **LLM generation** | GPT `gpt-4o-mini` via API | **per-token, see below** |
| Hosting | app + Qdrant containers | **fixed compute, see below** |

## LLM cost per query

Typical tokens for one grounded answer:

| Part | Tokens |
|---|---|
| System prompt | ~250 |
| Retrieved context (6 chunks) | ~1,200 |
| Conversation history (last N) | ~600 |
| User question | ~30 |
| **Input total** | **~2,080** |
| Answer (output) | ~250 |

- Input: 2,080 × $0.15 / 1M = **$0.00031**
- Output: 250 × $0.60 / 1M = **$0.00015**
- **≈ $0.00046 per query** (~0.05 US cents)

## Monthly LLM cost by usage

Assuming ~22 working days/month, 1,000 internal users:

| Scenario | Queries/user/day | Queries/month | LLM cost/month |
|---|---|---|---|
| Light pilot | ~0.05 (1k total/day) | ~22,000 | **~$10** |
| Typical | 3 | ~66,000 | **~$30** |
| Heavy | 10 | ~220,000 | **~$100** |

## Hosting (fixed)

The app is a stateless container; Qdrant is one small stateful service.

| Option | Approx. cost/month |
|---|---|
| Single small VM (2 vCPU / 4 GB) running both via compose | **~$25–40** |
| Cloud Run (app, scale-to-zero) + small managed Qdrant/VM | **~$15–35** |

CPU is enough — embeddings and the reranker are small models; no GPU required.

## All-in estimate

| Usage | LLM | Hosting | **Total/month** |
|---|---|---|---|
| Light | ~$10 | ~$30 | **~$40** |
| Typical | ~$30 | ~$30 | **~$60** |
| Heavy | ~$100 | ~$40 | **~$140** |

## Levers to reduce cost further

- **Free LLM path:** `LLM_PROVIDER` can point at a local model (e.g. Ollama) →
  **$0 API**, only compute. The factory makes this a config change.
- **Model tiering:** route simple questions to the cheapest model, complex ones
  to a stronger one.
- **Semantic cache:** cache answers to repeated questions (banking FAQs repeat a
  lot) to cut LLM calls.
- **Smaller context:** fewer/tighter chunks reduce input tokens (the largest
  share of per-query cost).

**Takeaway:** at typical internal-bank usage this runs at roughly **$40–60/month
all-in**, and can be driven to compute-only cost by swapping in a local LLM.
