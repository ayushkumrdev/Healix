#!/usr/bin/env python3
"""
Lightweight audit logging helpers for unified UI/backend usage.
"""
import os
import time
import json
from typing import Dict, Any

_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
os.makedirs(_LOG_DIR, exist_ok=True)
_AUDIT_PATH = os.path.join(_LOG_DIR, "audit_log.jsonl")
_MAX_BYTES = int(os.getenv("BAYMAX_AUDIT_MAX_BYTES", str(10 * 1024 * 1024)) or 10485760)


def _rotate_if_needed(path: str):
    try:
        if os.path.exists(path) and os.path.getsize(path) >= _MAX_BYTES:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            rotated = path.replace(".jsonl", f"_{ts}.jsonl")
            os.replace(path, rotated)
    except Exception:
        pass


def append_jsonl(path: str, record: Dict[str, Any]):
    _rotate_if_needed(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def audit(event: str, payload: Dict[str, Any]):
    rec = {"event": event, "ts": time.time(), **payload}
    try:
        append_jsonl(_AUDIT_PATH, rec)
    except Exception:
        pass
    return rec
