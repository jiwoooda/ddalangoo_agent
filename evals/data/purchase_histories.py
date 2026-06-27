"""Compatibility view for eval mock purchase histories.

Canonical mock data lives in evals/data/mock_db.json.
"""
import json
from pathlib import Path


_MOCK_DB = json.loads((Path(__file__).with_name("mock_db.json")).read_text(encoding="utf-8"))

MOCK_PURCHASE_HISTORY = _MOCK_DB.get("purchase_history", {})
MOCK_PURCHASE_HISTORIES = [
    {"user_id": user_id, **item}
    for user_id, histories in MOCK_PURCHASE_HISTORY.items()
    for item in histories
]

