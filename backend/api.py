#!/usr/bin/env python3
"""
Healix web backend (FastAPI).

Serves the custom chat UI (frontend/web) and exposes:
  GET  /                          -> chat app
  GET  /api/conversations         -> list saved conversations
  POST /api/conversations         -> create a conversation
  GET  /api/conversations/{id}    -> full conversation
  DELETE /api/conversations/{id}  -> delete
  POST /api/chat                  -> Server-Sent-Events stream of the reply

Heavy services (retriever, orchestrator, vision) load lazily on first use so the
UI and conversation history are instant. All generation runs locally.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))  # allow `services`, `conversation_store`

import conversation_store as store
from services.retriever import MedicalRetriever
from services.advanced_orchestrator import AdvancedMedicalOrchestrator
from services.scope_gate import ScopeGate

try:
    from services.vision import analyze_image as _analyze_image
except Exception:
    _analyze_image = None

try:
    from services.moe_router import ExpertRouter
except Exception:
    ExpertRouter = None

try:
    from services.hyde import generate_hypotheticals
except Exception:
    generate_hypotheticals = None

try:
    from services.raptor import RaptorIndex, raptor_enabled as _raptor_enabled
except Exception:
    RaptorIndex = None

    def _raptor_enabled():
        return False

MOE_ENABLED = str(os.getenv("HEALIX_MOE", "1")).lower() in ("1", "true", "yes")
HYBRID_ENABLED = str(os.getenv("HEALIX_HYBRID", "1")).lower() in ("1", "true", "yes")

try:
    from services.audit import audit as audit_log
except Exception:
    def audit_log(event, payload):
        return {"event": event, **payload}

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "frontend" / "web"

app = FastAPI(title="Healix")

# ---- lazy singletons ------------------------------------------------------
_retriever: Optional[MedicalRetriever] = None
_orch: Optional[AdvancedMedicalOrchestrator] = None
_router = None
_gate: Optional[ScopeGate] = None
_load_lock = threading.Lock()


def get_router(retriever):
    global _router
    if _router is None and MOE_ENABLED and ExpertRouter is not None:
        _router = ExpertRouter(retriever)
    return _router


def get_gate(retriever) -> ScopeGate:
    global _gate
    if _gate is None:
        _gate = ScopeGate(retriever)
    return _gate


_raptor = None


def get_raptor():
    global _raptor
    if _raptor is None and RaptorIndex is not None and _raptor_enabled():
        _raptor = RaptorIndex()
    return _raptor


def get_services():
    global _retriever, _orch
    if _retriever is None or _orch is None:
        with _load_lock:
            if _retriever is None:
                _retriever = MedicalRetriever(index_dir=str(ROOT / "data"))
            if _orch is None:
                kwargs = {}
                if os.getenv("BAYMAX_GGUF_MODEL_NAME"):
                    kwargs["model_name"] = os.getenv("BAYMAX_GGUF_MODEL_NAME")
                if os.getenv("BAYMAX_GGUF_MODEL_DIR"):
                    kwargs["model_path"] = os.getenv("BAYMAX_GGUF_MODEL_DIR")
                _orch = AdvancedMedicalOrchestrator(**kwargs)
    return _retriever, _orch


# ---- models ---------------------------------------------------------------
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str = ""
    image: Optional[str] = None  # data URL or base64
    modality: str = "auto"


# ---- conversation endpoints ----------------------------------------------
@app.get("/api/conversations")
def list_convs():
    return store.list_conversations()


@app.post("/api/conversations")
def new_conv():
    return store.create_conversation()


@app.get("/api/conversations/{cid}")
def get_conv(cid: str):
    conv = store.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "not found")
    return conv


@app.delete("/api/conversations/{cid}")
def del_conv(cid: str):
    return {"ok": store.delete_conversation(cid)}


# ---- chat (SSE stream) ----------------------------------------------------
def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _scripted_response(cid: str, text: str, kind: str) -> StreamingResponse:
    """Stream a canned scope-gate reply with a light typing cadence, then
    persist it — same SSE contract as the full pipeline, zero model time."""
    def gen():
        for i, word in enumerate(text.split(" ")):
            yield _sse("token", {"t": word if i == 0 else " " + word})
            time.sleep(0.012)
        conv = store.append_message(cid, "assistant", text)
        audit_log("chat_answer", {"intent": kind, "scope_gated": True})
        yield _sse("done", {
            "text": text, "conversation_id": cid, "title": conv.get("title"),
            "sources": [], "timings": {"total_s": 0.0}, "experts": [],
        })
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _decode_image(data: str) -> Optional[bytes]:
    if not data:
        return None
    try:
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        return base64.b64decode(data)
    except Exception:
        return None


def _strip_opening_echo(text: str, prev_reply: str) -> str:
    """Drop an opening paragraph that near-duplicates the previous assistant
    reply. Small local models often begin a follow-up answer by re-stating
    their last message; prompt instructions reduce but don't eliminate it."""
    if not text or not prev_reply:
        return text
    paras = [p for p in text.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return text  # never strip a whole reply
    import difflib
    import re as _re
    first = _re.sub(r"\s+", " ", paras[0]).strip().lower()
    prev = _re.sub(r"\s+", " ", prev_reply).strip().lower()
    if not first or len(first) > len(prev) + 80:
        return text
    window = prev[: len(first) + 80]
    ratio = difflib.SequenceMatcher(None, first, window).ratio()
    if ratio >= 0.75 or first in prev:
        return "\n\n".join(paras[1:]).strip()
    return text


@app.post("/api/chat")
def chat(req: ChatRequest):
    retriever, orch = get_services()

    cid = req.conversation_id or store.create_conversation()["id"]
    user_text = (req.message or "").strip()

    # Image analysis (if provided)
    image_block = ""
    findings: List[Dict[str, Any]] = []
    modality_label = ""
    image_modality = ""
    img_bytes = _decode_image(req.image) if req.image else None
    if img_bytes is not None and _analyze_image is not None:
        try:
            from io import BytesIO
            from PIL import Image
            result = _analyze_image(Image.open(BytesIO(img_bytes)).convert("RGB"),
                                    modality=req.modality)
            if result.get("available"):
                findings = result.get("findings", [])[:8]
                image_block = result.get("prompt_block", "") or ""
                modality_label = result.get("modality_label", "")
                image_modality = result.get("modality", "")
            if not user_text:
                user_text = f"Explain what this {modality_label or 'medical image'} shows, in plain language."
        except Exception:
            pass

    if not user_text:
        user_text = "Hello"

    # Persist the user turn (store a small thumbnail flag, not the raw bytes)
    store.append_message(cid, "user", user_text,
                         image=("attached" if img_bytes else None))

    context = store.context_messages(cid)

    # Domain gate: greetings/identity get a warm canned line, clearly
    # non-medical asks get a polite redirect — no retrieval, no LLM call.
    # context includes the just-persisted user turn, hence > 1 for "ongoing".
    scope = get_gate(retriever).classify(
        user_text, has_image=bool(img_bytes), in_conversation=len(context) > 1)
    if scope.reply and not findings:
        return _scripted_response(cid, scope.reply, scope.kind)

    def stream():
        if findings:
            yield _sse("findings", {"modality": modality_label, "findings": findings})

        try:
            complexity = orch.classify_query_complexity(user_text)
            t0 = time.time()
            routing: List[Dict[str, Any]] = []
            expert_directive = None
            router = get_router(retriever)
            # Short follow-ups ("what about at night?") retrieve poorly on
            # their own; fold the previous user message into the search query.
            # context already contains the just-persisted current turn.
            retrieval_query = user_text
            if len(user_text.split()) < 8:
                prior = [m for m in context[:-1]
                         if m.get("role") == "user" and isinstance(m.get("content"), str)]
                if prior:
                    retrieval_query = (prior[-1]["content"][:300] + "\n" + user_text).strip()

            # HyDE: draft what a reference would say (two abstraction levels).
            # The specific line joins the leaf search query; the broad line
            # searches the RAPTOR overview tree. Fail-open on any error.
            hyde = {}
            if generate_hypotheticals is not None and complexity != "greeting":
                hyde = generate_hypotheticals(retrieval_query) or {}
            leaf_query = retrieval_query
            if hyde.get("specific"):
                leaf_query = f"{retrieval_query}\n{hyde['specific']}"

            if complexity == "greeting":
                passages = []
            elif router is not None:
                # Agentic MoE: gate to specialist experts, then gather
                # lens-specific evidence from each before a single synthesis.
                routing, activated = router.route(retrieval_query, image_modality or None)
                passages = router.gather_evidence(leaf_query, activated)
                expert_directive = router.directive(routing)
            elif HYBRID_ENABLED and hasattr(retriever, "hybrid_retrieve"):
                passages = retriever.hybrid_retrieve(query=leaf_query, k=10)
            else:
                passages = retriever.retrieve(query=leaf_query, k=10, min_score=0.25)

            # RAPTOR: altitude-matched overview evidence for broad questions —
            # topic/domain summary nodes retrieved with the broad hypothetical.
            raptor = get_raptor()
            if raptor is not None and raptor.available and complexity != "greeting":
                overview = raptor.search(hyde.get("broad") or retrieval_query,
                                         retriever, k=2)
                if overview:
                    passages = overview + (passages or [])
            t1 = time.time()
            if routing:
                yield _sse("routing", {"experts": routing})
            full_prompt, query_type, specialties = orch.prepare_stream_prompt(
                user_text=user_text, retrieved_passages=passages, symptom_data=None,
                conversation_context=context, user_mode="patient",
                image_findings=image_block or None, expert_directive=expert_directive)

            try:
                max_tok = int(os.getenv("BAYMAX_GEN_MAX_TOKENS", "900") or 900)
            except Exception:
                max_tok = 900

            buf = ""
            t2 = time.time()
            for tok in orch.stream_generate(
                    full_prompt, max_tokens=max_tok,
                    temp=float(os.getenv("BAYMAX_GEN_TEMP", "0.4") or 0.4),
                    top_k=int(os.getenv("BAYMAX_GEN_TOP_K", "10") or 10),
                    top_p=float(os.getenv("BAYMAX_GEN_TOP_P", "0.9") or 0.9)):
                buf += tok
                yield _sse("token", {"t": tok})
            t3 = time.time()

            # Gentle cleanup only — must preserve markdown and line breaks.
            cleaned = orch.clean_stream_text(buf) if buf else ""
            cleaned = cleaned or (buf.strip() if buf else "No response generated.")
            prev_assistant = next(
                (m.get("content") for m in reversed(context)
                 if m.get("role") == "assistant" and isinstance(m.get("content"), str)), "")
            cleaned = _strip_opening_echo(cleaned, prev_assistant) or cleaned

            sources = sorted({p.get("source", "") for p in passages if p.get("source")})
            timings = {"retrieval_s": round(t1 - t0, 2),
                       "generation_s": round(t3 - t2, 2),
                       "total_s": round((t1 - t0) + (t3 - t2), 2)}

            conv = store.append_message(cid, "assistant", cleaned,
                                        findings=findings or None,
                                        sources=sources or None, timings=timings,
                                        experts=routing or None)
            audit_log("chat_answer", {"intent": query_type, **timings})
            yield _sse("done", {
                "text": cleaned, "conversation_id": cid, "title": conv.get("title"),
                "sources": sources, "timings": timings, "experts": routing,
            })
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


_DEVICE_CACHE: Optional[str] = None


@app.get("/api/info")
def info():
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        _DEVICE_CACHE = "CPU"
        try:
            import torch
            _DEVICE_CACHE = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        except Exception:
            pass
    device = _DEVICE_CACHE
    docs = None
    try:
        cfg = json.loads((ROOT / "data" / "index_config.json").read_text(encoding="utf-8"))
        docs = cfg.get("total_vectors")
    except Exception:
        pass
    if (os.getenv("BAYMAX_LLM_BACKEND", "").lower() == "ollama"):
        model = os.getenv("BAYMAX_OLLAMA_MODEL", "llama3")
    else:
        model = os.getenv("BAYMAX_GGUF_MODEL_NAME", "mistral-7b")
    return {"device": device, "docs": docs, "model": model}


# ---- static frontend (mounted last so /api wins) --------------------------
@app.get("/")
def index():
    idx = WEB_DIR / "index.html"
    if not idx.exists():
        return JSONResponse({"error": "frontend not built"}, status_code=500)
    return FileResponse(str(idx))


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
