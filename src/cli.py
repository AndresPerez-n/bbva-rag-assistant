"""Interactive command-line chat interface.

A minimal REPL over the same :class:`Chatbot` the API uses. Supports slash
commands for switching sessions, inspecting history, giving feedback, and
printing analytics.

Run:
    python -m src.cli
    python -m src.cli --session my-session
"""
from __future__ import annotations

import argparse
import sys

from src.config import get_settings
from src.memory.history import ConversationStore
from src.rag.chatbot import build_chatbot

HELP = """
Comandos:
  /session <id>   cambiar de sesión
  /history        mostrar el historial de la sesión actual
  /up  | /down    calificar la última respuesta (pulgar arriba/abajo)
  /analytics      métricas del histórico de conversaciones
  /help           mostrar esta ayuda
  /quit           salir
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="BBVA RAG Assistant — CLI")
    parser.add_argument("--session", default="cli", help="conversation session id")
    args = parser.parse_args()

    settings = get_settings()
    store = ConversationStore(settings.history_db_path)
    print("Cargando modelo de recuperación y reranker...")
    chatbot = build_chatbot(settings, store=store)

    session_id = args.session
    last_message_id = None

    print(f"\nBBVA RAG Assistant (sesión: {session_id}). Escribe /help para comandos.\n")

    while True:
        try:
            query = input("tú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not query:
            continue

        if query.startswith("/"):
            cmd, *rest = query[1:].split(maxsplit=1)
            arg = rest[0] if rest else ""

            if cmd in ("quit", "exit"):
                print("Hasta luego.")
                break
            if cmd == "help":
                print(HELP); continue
            if cmd == "session":
                session_id = arg or session_id
                print(f"[sesión cambiada a: {session_id}]"); continue
            if cmd == "history":
                for m in store.get_session(session_id):
                    print(f"  [{m.role}] {m.content[:120]}")
                continue
            if cmd in ("up", "down"):
                if last_message_id is None:
                    print("[no hay respuesta reciente que calificar]"); continue
                store.add_feedback(session_id, 1 if cmd == "up" else -1, last_message_id)
                print("[feedback registrado]"); continue
            if cmd == "analytics":
                from src.analytics.metrics import ConversationAnalytics
                ConversationAnalytics(store).print_report()
                continue
            print("[comando desconocido — usa /help]")
            continue

        # Normal question -> streamed answer.
        print("bot > ", end="", flush=True)
        answer = None
        for event in chatbot.stream_answer(session_id, query):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "done":
                answer = event["answer"]
        print()
        if answer:
            last_message_id = answer.message_id
            if answer.sources:
                srcs = ", ".join(s.url or s.title for s in answer.sources)
                print(f"    [confianza: {answer.confidence} | fuentes: {srcs}]")
            else:
                print(f"    [confianza: {answer.confidence}]")
        print()

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
