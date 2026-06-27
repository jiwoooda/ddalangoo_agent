"""Compatibility view for eval mock users.

Canonical mock data lives in evals/data/mock_db.json.
"""
import json
from pathlib import Path


_MOCK_DB = json.loads((Path(__file__).with_name("mock_db.json")).read_text(encoding="utf-8"))

MOCK_USERS = [
    {"id": user_id, **user}
    for user_id, user in _MOCK_DB.get("users", {}).items()
]
