"""RAG orchestration.

Ties the pieces together for one turn:
1. retrieve (vector recall + rerank),
2. short-circuit out-of-scope questions *without* calling the LLM (saves tokens
   and prevents answering from model memory instead of the site content),
3. assemble a grounded, cited prompt with the last N messages of history,
4. generate (streaming or not) with the configured LLM,
5. persist both turns (with metadata) to the conversation store.

Answers are grounded in the scraped bank-site content and cite their sources.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from src.config import Settings, get_settings
from src.memory.history import ConversationStore, Message
from src.rag.llm import LLMClient, LLMFactory
from src.rag.retriever import RetrievalResult, Retriever, build_retriever
from src.rag.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Eres un asistente interno del banco. Respondes preguntas de empleados usando "
    "EXCLUSIVAMENTE la informacion del CONTEXTO proporcionado, que proviene del sitio "
    "web del banco.\n"
    "Reglas:\n"
    "1. Responde solo con datos presentes en el contexto. No inventes ni uses conocimiento externo.\n"
    "2. Si la respuesta no esta en el contexto, dilo claramente y sugiere consultar el canal oficial.\n"
    "3. Cita las fuentes relevantes al final usando el formato [Fuente: URL].\n"
    "4. Responde en espanol, de forma clara y concisa.\n"
    "5. Si la pregunta es ambigua, pide una aclaracion breve."
)

REFUSAL = (
    "No encontre esa informacion en el contenido disponible del sitio del banco. "
    "Te sugiero consultar el canal oficial o reformular la pregunta."
)


@dataclass
class Source:
    url: str
    title: str
    score: float


@dataclass
class Answer:
    text: str
    confidence: str
    out_of_scope: bool = False
    sources: List[Source] = field(default_factory=list)
    latency_ms: int = 0
    message_id: Optional[int] = None


class Chatbot:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        store: ConversationStore,
        conversation_window: int,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.store = store
        self.conversation_window = conversation_window

    # --- public API --------------------------------------------------------
    def answer(self, session_id: str, query: str) -> Answer:
        started = time.time()
        self.store.add_message(session_id, "user", query)
        result = self.retriever.retrieve(query)

        if result.is_empty:
            return self._finish_out_of_scope(session_id, started)

        system, messages = self._build_prompt(session_id, query, result)
        try:
            text = self.llm.generate(system, messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM generation failed")
            text = f"Ocurrio un error al generar la respuesta: {exc}"

        return self._persist_answer(session_id, text, result, started)

    def stream_answer(self, session_id: str, query: str) -> Iterator[Dict]:
        """Yield events: {'type': 'sources'|'token'|'done', ...}."""
        started = time.time()
        self.store.add_message(session_id, "user", query)
        result = self.retriever.retrieve(query)

        if result.is_empty:
            answer = self._finish_out_of_scope(session_id, started)
            yield {"type": "token", "text": answer.text}
            yield {"type": "done", "answer": answer}
            return

        sources = self._sources(result.chunks)
        yield {"type": "sources", "sources": sources, "confidence": result.confidence}

        system, messages = self._build_prompt(session_id, query, result)
        collected: List[str] = []
        try:
            for token in self.llm.stream(system, messages):
                collected.append(token)
                yield {"type": "token", "text": token}
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM streaming failed")
            err = f"\n[Error al generar la respuesta: {exc}]"
            collected.append(err)
            yield {"type": "token", "text": err}

        answer = self._persist_answer(session_id, "".join(collected), result, started)
        yield {"type": "done", "answer": answer}

    # --- internals ---------------------------------------------------------
    def _build_prompt(self, session_id: str, query: str, result: RetrievalResult):
        context_blocks = []
        for chunk in result.chunks:
            tag = chunk.source_url or chunk.title or "sitio"
            context_blocks.append(f"[Fuente: {tag}]\n{chunk.text}")
        context = "\n\n".join(context_blocks)

        history = self._history_messages(session_id)
        current = (
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA DEL USUARIO:\n{query}"
        )
        messages = [*history, {"role": "user", "content": current}]
        return SYSTEM_PROMPT, messages

    def _history_messages(self, session_id: str) -> List[Dict[str, str]]:
        # Pull the window then drop the just-added current user message (last row).
        window = self.store.get_window(session_id, self.conversation_window + 1)
        prior = window[:-1] if window else []
        msgs = [{"role": m.role, "content": m.content} for m in prior]
        # Chat APIs require the first message to be from the user.
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        return msgs

    def _sources(self, chunks: List[RetrievedChunk]) -> List[Dict]:
        seen = set()
        out = []
        for c in chunks:
            key = c.source_url or c.title
            if key in seen:
                continue
            seen.add(key)
            out.append({"url": c.source_url, "title": c.title, "score": round(c.score, 4)})
        return out

    def _persist_answer(
        self, session_id: str, text: str, result: RetrievalResult, started: float
    ) -> Answer:
        latency_ms = int((time.time() - started) * 1000)
        sources = [Source(s["url"], s["title"], s["score"]) for s in self._sources(result.chunks)]
        metadata = {
            "confidence": result.confidence,
            "top_score": round(result.top_score, 4),
            "num_chunks": len(result.chunks),
            "sources": [s.url for s in sources],
            "latency_ms": latency_ms,
            "out_of_scope": False,
            "model": get_settings().llm_model,
        }
        msg_id = self.store.add_message(session_id, "assistant", text, metadata)
        return Answer(
            text=text,
            confidence=result.confidence,
            out_of_scope=False,
            sources=sources,
            latency_ms=latency_ms,
            message_id=msg_id,
        )

    def _finish_out_of_scope(self, session_id: str, started: float) -> Answer:
        latency_ms = int((time.time() - started) * 1000)
        metadata = {
            "confidence": "out_of_scope",
            "top_score": 0.0,
            "num_chunks": 0,
            "sources": [],
            "latency_ms": latency_ms,
            "out_of_scope": True,
            "model": None,
        }
        msg_id = self.store.add_message(session_id, "assistant", REFUSAL, metadata)
        return Answer(
            text=REFUSAL,
            confidence="out_of_scope",
            out_of_scope=True,
            sources=[],
            latency_ms=latency_ms,
            message_id=msg_id,
        )


def build_chatbot(settings: Settings | None = None, store: Optional[ConversationStore] = None) -> Chatbot:
    settings = settings or get_settings()
    store = store or ConversationStore(settings.history_db_path)
    return Chatbot(
        retriever=build_retriever(settings),
        llm=LLMFactory.create(settings),
        store=store,
        conversation_window=settings.conversation_window,
    )
