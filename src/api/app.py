"""FastAPI server: conversational RAG endpoints + a minimal chat UI.

Endpoints
---------
GET  /                 -> the single-page chat interface
GET  /health           -> liveness + index size
POST /chat             -> non-streaming answer (JSON)
POST /chat/stream      -> streaming answer (Server-Sent Events)
POST /feedback         -> thumbs up/down against a message id
GET  /sessions         -> list session ids
GET  /sessions/{id}    -> full history for a session
GET  /analytics        -> aggregated conversation metrics

The heavy objects (embedder, reranker, LLM client) are built once at startup and
shared across requests.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.config import get_settings
from src.memory.history import ConversationStore
from src.rag.chatbot import Chatbot, build_chatbot

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="default")


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., description="1 for thumbs up, -1 for thumbs down")
    message_id: Optional[int] = None


# --- Response models (typed output, auto-documented in Swagger) -------------
class SourceModel(BaseModel):
    url: str
    title: str
    score: float                       # Qdrant cosine similarity
    rerank_score: Optional[float] = None  # cross-encoder relevance


class CrossHit(BaseModel):
    url: Optional[str] = None
    score: Optional[float] = None


class CrossCheckModel(BaseModel):
    faiss_indexed: int
    qdrant_top: Optional[CrossHit] = None
    faiss_top: Optional[CrossHit] = None
    agree: bool


class ChatResponse(BaseModel):
    answer: str
    confidence: str
    out_of_scope: bool
    sources: List[SourceModel]
    crosscheck: Optional[CrossCheckModel] = None
    latency_ms: int
    message_id: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    provider: str
    collection: str
    indexed_chunks: int
    faiss_crosscheck_vectors: int
    conversation_window: int


class FeedbackResponse(BaseModel):
    status: str


class _State:
    chatbot: Optional[Chatbot] = None
    store: Optional[ConversationStore] = None


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Building chatbot (loading embedder/reranker)...")
    state.store = ConversationStore(settings.history_db_path)
    state.chatbot = build_chatbot(settings, store=state.store)
    logger.info("Chatbot ready")
    yield
    if state.store:
        state.store.close()


app = FastAPI(title="BBVA RAG Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/")
def index():
    index_file = _STATIC_DIR / "index.html"
    if not index_file.exists():
        return {"message": "Chat UI not found. POST /chat or /chat/stream instead."}
    # no-cache so UI updates are always picked up (avoids stale cached page)
    return FileResponse(index_file, headers={"Cache-Control": "no-cache, max-age=0"})


@app.get("/health", response_model=HealthResponse)
def health():
    settings = get_settings()
    indexed = 0
    faiss_indexed = 0
    try:
        indexed = state.chatbot.retriever.store.count() if state.chatbot else 0
        cc = state.chatbot.retriever.faiss_crosscheck if state.chatbot else None
        faiss_indexed = cc.size if cc else 0
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok",
        "model": settings.llm_model,
        "provider": settings.llm_provider,
        "collection": settings.qdrant_collection,
        "indexed_chunks": indexed,
        "faiss_crosscheck_vectors": faiss_indexed,
        "conversation_window": settings.conversation_window,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if state.chatbot is None:
        raise HTTPException(503, "Chatbot not ready")
    answer = state.chatbot.answer(req.session_id, req.message)
    return {
        "answer": answer.text,
        "confidence": answer.confidence,
        "out_of_scope": answer.out_of_scope,
        "sources": [s.__dict__ for s in answer.sources],
        "crosscheck": answer.crosscheck,
        "latency_ms": answer.latency_ms,
        "message_id": answer.message_id,
    }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if state.chatbot is None:
        raise HTTPException(503, "Chatbot not ready")

    def event_gen():
        for event in state.chatbot.stream_answer(req.session_id, req.message):
            if event["type"] == "done":
                ans = event["answer"]
                payload = {
                    "type": "done",
                    "confidence": ans.confidence,
                    "out_of_scope": ans.out_of_scope,
                    "sources": [s.__dict__ for s in ans.sources],
                    "crosscheck": ans.crosscheck,
                    "message_id": ans.message_id,
                    "latency_ms": ans.latency_ms,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    if state.store is None:
        raise HTTPException(503, "Store not ready")
    state.store.add_feedback(req.session_id, req.rating, req.message_id)
    return {"status": "recorded"}


@app.get("/sessions")
def sessions():
    return {"sessions": state.store.list_sessions() if state.store else []}


@app.get("/sessions/{session_id}")
def session_history(session_id: str):
    if state.store is None:
        raise HTTPException(503, "Store not ready")
    return {
        "session_id": session_id,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at, "metadata": m.metadata}
            for m in state.store.get_session(session_id)
        ],
    }


@app.get("/analytics")
def analytics():
    from src.analytics.metrics import ConversationAnalytics

    if state.store is None:
        raise HTTPException(503, "Store not ready")
    return ConversationAnalytics(state.store).summary()
