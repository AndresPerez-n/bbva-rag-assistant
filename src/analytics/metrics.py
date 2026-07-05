"""Aggregate metrics over the persisted conversation history.

Requirement: a feature that walks the conversation history to extract metrics and
impact values. This reads the same SQLite store the chatbot writes to and reports:

- volume (sessions, messages, turns, avg turns/session),
- quality (confidence distribution, out-of-scope rate, avg retrieval score),
- performance (avg/percentile answer latency),
- satisfaction (thumbs up/down, satisfaction rate),
- content (top questions, most-cited source URLs).

Exposed both as a dict (``summary()`` — consumed by the API) and a printed report
(``print_report()`` — used by the CLI and the analytics script).
"""
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, List

from src.memory.history import ConversationStore, Message


class ConversationAnalytics:
    def __init__(self, store: ConversationStore) -> None:
        self.store = store
        self._messages: List[Message] = store.all_messages()
        self._feedback = store.all_feedback()

    # --- aggregation -------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        msgs = self._messages
        user_msgs = [m for m in msgs if m.role == "user"]
        bot_msgs = [m for m in msgs if m.role == "assistant"]
        sessions = {m.session_id for m in msgs}

        confidences = [m.metadata.get("confidence") for m in bot_msgs if m.metadata]
        conf_dist = dict(Counter(c for c in confidences if c))
        out_of_scope = sum(1 for m in bot_msgs if m.metadata.get("out_of_scope"))

        latencies = [m.metadata.get("latency_ms") for m in bot_msgs if m.metadata.get("latency_ms")]
        scores = [m.metadata.get("top_score") for m in bot_msgs
                  if isinstance(m.metadata.get("top_score"), (int, float)) and m.metadata.get("top_score")]

        up = sum(1 for f in self._feedback if f["rating"] > 0)
        down = sum(1 for f in self._feedback if f["rating"] < 0)

        source_counter: Counter = Counter()
        for m in bot_msgs:
            for url in m.metadata.get("sources", []) or []:
                if url:
                    source_counter[url] += 1

        question_counter = Counter(m.content.strip().lower() for m in user_msgs)

        return {
            "totals": {
                "sessions": len(sessions),
                "messages": len(msgs),
                "user_messages": len(user_msgs),
                "assistant_messages": len(bot_msgs),
                "avg_turns_per_session": round(len(user_msgs) / len(sessions), 2) if sessions else 0,
            },
            "quality": {
                "confidence_distribution": conf_dist,
                "out_of_scope": out_of_scope,
                "out_of_scope_rate": round(out_of_scope / len(bot_msgs), 3) if bot_msgs else 0,
                "avg_retrieval_score": round(mean(scores), 3) if scores else 0,
            },
            "performance": {
                "avg_latency_ms": round(mean(latencies)) if latencies else 0,
                "p95_latency_ms": self._percentile(latencies, 95),
                "max_latency_ms": max(latencies) if latencies else 0,
            },
            "satisfaction": {
                "thumbs_up": up,
                "thumbs_down": down,
                "total_feedback": up + down,
                "satisfaction_rate": round(up / (up + down), 3) if (up + down) else None,
            },
            "top_questions": question_counter.most_common(5),
            "top_sources": source_counter.most_common(5),
        }

    @staticmethod
    def _percentile(values: List[float], pct: int) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        k = max(0, min(len(ordered) - 1, int(round((pct / 100) * len(ordered)) - 1)))
        return int(ordered[k])

    # --- presentation ------------------------------------------------------
    def print_report(self) -> None:
        try:
            from tabulate import tabulate
        except ImportError:  # pragma: no cover
            tabulate = None

        s = self.summary()
        if s["totals"]["messages"] == 0:
            print("\nNo hay conversaciones registradas todavía.\n")
            return

        def table(title, rows):
            print(f"\n== {title} ==")
            if tabulate:
                print(tabulate(rows, tablefmt="github"))
            else:
                for k, v in rows:
                    print(f"  {k}: {v}")

        t = s["totals"]
        table("Volumen", [
            ["Sesiones", t["sessions"]],
            ["Mensajes totales", t["messages"]],
            ["Preguntas de usuario", t["user_messages"]],
            ["Respuestas del asistente", t["assistant_messages"]],
            ["Turnos promedio por sesión", t["avg_turns_per_session"]],
        ])
        q = s["quality"]
        table("Calidad", [
            ["Distribución de confianza", q["confidence_distribution"]],
            ["Fuera de alcance", q["out_of_scope"]],
            ["Tasa fuera de alcance", q["out_of_scope_rate"]],
            ["Score de recuperación promedio", q["avg_retrieval_score"]],
        ])
        p = s["performance"]
        table("Rendimiento", [
            ["Latencia promedio (ms)", p["avg_latency_ms"]],
            ["Latencia p95 (ms)", p["p95_latency_ms"]],
            ["Latencia máxima (ms)", p["max_latency_ms"]],
        ])
        sat = s["satisfaction"]
        table("Satisfacción", [
            ["Pulgar arriba", sat["thumbs_up"]],
            ["Pulgar abajo", sat["thumbs_down"]],
            ["Tasa de satisfacción", sat["satisfaction_rate"]],
        ])
        if s["top_questions"]:
            table("Preguntas más frecuentes", [[c, question] for question, c in s["top_questions"]])
        if s["top_sources"]:
            table("Fuentes más citadas", [[c, url] for url, c in s["top_sources"]])
        print()
