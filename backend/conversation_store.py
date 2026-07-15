#!/usr/bin/env python3
"""
Persistent conversation storage for Healix.

Each conversation is a JSON file under data/conversations/. This gives the web
app ChatGPT-style persistent history: conversations survive restarts, can be
listed, reopened, renamed, and deleted, and provide full context to the model.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_DIR = Path(os.getenv("HEALIX_CONV_DIR", os.path.join("data", "conversations")))
_LOCK = threading.Lock()


def _ensure() -> None:
    _DIR.mkdir(parents=True, exist_ok=True)


def _path(cid: str) -> Path:
    return _DIR / f"{cid}.json"


def list_conversations() -> List[Dict[str, Any]]:
    _ensure()
    items: List[Dict[str, Any]] = []
    for p in _DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            items.append({
                "id": d["id"],
                "title": d.get("title") or "New chat",
                "updated": d.get("updated", 0),
                "message_count": len(d.get("messages", [])),
            })
        except Exception:
            continue
    items.sort(key=lambda x: x["updated"], reverse=True)
    return items


def get_conversation(cid: str) -> Optional[Dict[str, Any]]:
    p = _path(cid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save(conv: Dict[str, Any]) -> None:
    _ensure()
    with _LOCK:
        _path(conv["id"]).write_text(
            json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def create_conversation(title: str = "New chat") -> Dict[str, Any]:
    cid = uuid.uuid4().hex[:12]
    now = time.time()
    conv = {"id": cid, "title": title, "created": now, "updated": now, "messages": []}
    _save(conv)
    return conv


def append_message(cid: str, role: str, content: str, **extra: Any) -> Dict[str, Any]:
    """Append a message to a conversation (creating it if missing)."""
    conv = get_conversation(cid)
    if conv is None:
        conv = {"id": cid, "title": "New chat", "created": time.time(),
                "updated": time.time(), "messages": []}
    msg: Dict[str, Any] = {"role": role, "content": content, "ts": time.time()}
    for k, v in extra.items():
        if v is not None:
            msg[k] = v
    conv["messages"].append(msg)
    conv["updated"] = time.time()
    if role == "user" and conv.get("title") in (None, "", "New chat") \
            and isinstance(content, str) and content.strip():
        conv["title"] = content.strip()[:60]
    _save(conv)
    return conv


def delete_conversation(cid: str) -> bool:
    p = _path(cid)
    if p.exists():
        try:
            p.unlink()
            return True
        except Exception:
            return False
    return False


def rename_conversation(cid: str, title: str) -> bool:
    conv = get_conversation(cid)
    if not conv:
        return False
    conv["title"] = (title or "Untitled")[:80]
    conv["updated"] = time.time()
    _save(conv)
    return True


def context_messages(cid: str) -> List[Dict[str, str]]:
    """Return messages as [{role, content}] for the orchestrator's memory builder."""
    conv = get_conversation(cid)
    if not conv:
        return []
    out = []
    for m in conv.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            out.append({"role": m.get("role", "user"), "content": c})
    return out
