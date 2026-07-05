"""Entrypoint: print conversation analytics from the persisted history.

Usage:
    python -m scripts.run_analytics
    python -m scripts.run_analytics --json
"""
from __future__ import annotations

import argparse
import json
import sys

from src.analytics.metrics import ConversationAnalytics
from src.config import get_settings
from src.memory.history import ConversationStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Conversation analytics")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of tables")
    args = parser.parse_args()

    settings = get_settings()
    store = ConversationStore(settings.history_db_path)
    analytics = ConversationAnalytics(store)

    if args.json:
        print(json.dumps(analytics.summary(), ensure_ascii=False, indent=2))
    else:
        analytics.print_report()

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
