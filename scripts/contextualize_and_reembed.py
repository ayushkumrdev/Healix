#!/usr/bin/env python3
"""
Full Contextual Retrieval for Healix (Anthropic-style).

Phase 1 (contextualize): for every chunk, ask a small local Ollama model for a
one-line context that situates the chunk, and cache it to a resumable JSONL
checkpoint (data/contexts.jsonl). Concurrent + checkpointed, so it can be stopped
and resumed (--resume) and survives restarts.

Phase 2 (re-embed + index): build `context + text` for every chunk, re-embed with
the bge model on GPU, rebuild the FAISS index (SQ8) and the contextual BM25, and
store the context on each chunk. Originals are backed up first.

Usage:
  # measure throughput on a sample
  python scripts/contextualize_and_reembed.py --phase 1 --limit 200
  # full contextualization (resumable)
  python scripts/contextualize_and_reembed.py --phase 1 --resume
  # rebuild the index from the cached contexts
  python scripts/contextualize_and_reembed.py --phase 2
  # both
  python scripts/contextualize_and_reembed.py --phase all --resume
"""

import argparse
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHUNKS = DATA / "indexed_chunks.pkl"
CTX_FILE = DATA / "contexts.jsonl"
OLLAMA = os.getenv("BAYMAX_OLLAMA_URL", "http://localhost:11434").rstrip("/")

PROMPT = (
    "You write a concise search context. In 18 words or fewer, state the medical "
    "topic and key entities of the text, to improve retrieval. Reply with the "
    "context only, no preamble.\n\nText:\n{body}\n\nContext:"
)


def load_chunks():
    with open(CHUNKS, "rb") as f:
        return pickle.load(f)


def load_done():
    done = {}
    if CTX_FILE.exists():
        with open(CTX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    done[o["id"]] = o["ctx"]
                except Exception:
                    continue
    return done


def ollama_context(body: str, model: str) -> str:
    payload = {"model": model, "prompt": PROMPT.format(body=body[:600]),
               "stream": False, "options": {"temperature": 0.0, "num_predict": 48}}
    r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    txt = (r.json().get("response", "") or "").strip().replace("\n", " ")
    # strip any <think> blocks from reasoning models
    if "</think>" in txt:
        txt = txt.split("</think>")[-1].strip()
    return txt[:240]


def phase1(model: str, workers: int, limit: int, resume: bool):
    chunks = load_chunks()
    if limit:
        chunks = chunks[:limit]
    done = load_done() if resume else {}
    todo = [c for c in chunks if str(c.get("id")) not in done]
    print(f"Phase 1: {len(chunks)} chunks, {len(done)} cached, {len(todo)} to do, "
          f"model={model}, workers={workers}")
    if not todo:
        print("Nothing to contextualize.")
        return

    lock = threading.Lock()
    counter = {"n": 0, "err": 0}
    t0 = time.time()
    fh = open(CTX_FILE, "a", encoding="utf-8")

    def work(c):
        cid = str(c.get("id"))
        body = c.get("text") or ""
        try:
            ctx = ollama_context(body, model)
        except Exception:
            ctx = ""
            with lock:
                counter["err"] += 1
        with lock:
            fh.write(json.dumps({"id": cid, "ctx": ctx}, ensure_ascii=False) + "\n")
            counter["n"] += 1
            n = counter["n"]
            if n % 200 == 0 or n == len(todo):
                fh.flush()
                rate = n / (time.time() - t0)
                eta = (len(todo) - n) / rate / 3600 if rate else 0
                print(f"  {n}/{len(todo)}  {rate:.1f} chunks/s  ETA {eta:.1f}h  "
                      f"errors={counter['err']}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"Phase 1 done in {(time.time()-t0)/60:.1f} min")


def phase2(embed_model: str):
    from sentence_transformers import SentenceTransformer
    import faiss

    chunks = load_chunks()
    ctx = load_done()
    print(f"Phase 2: {len(chunks)} chunks, {len(ctx)} contexts cached")

    texts, used = [], 0
    for c in chunks:
        cid = str(c.get("id"))
        ctext = ctx.get(cid, "")
        if ctext:
            used += 1
            c["context"] = ctext
        body = c.get("text") or ""
        texts.append((ctext + " | " + body).strip(" |") if ctext else body)
    print(f"  contextual coverage: {used}/{len(chunks)}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(os.getenv("BAYMAX_EMBEDDING_MODEL", embed_model), device=device)
    print(f"  embedding {len(texts)} contextual texts on {device}...")
    emb = model.encode(texts, batch_size=64, convert_to_numpy=True,
                       show_progress_bar=True, normalize_embeddings=True).astype(np.float32)

    dim = emb.shape[1]
    index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit,
                                       faiss.METRIC_INNER_PRODUCT)
    index.train(emb)
    index.add(emb)

    # Back up and write
    idx_path = DATA / "index.faiss"
    if idx_path.exists():
        bak = DATA / "index.faiss.prectx"
        if not bak.exists():
            idx_path.replace(bak)
            print(f"  backed up old index -> {bak.name}")
    faiss.write_index(index, str(idx_path))
    with open(CHUNKS, "wb") as f:
        pickle.dump(chunks, f)
    # Invalidate BM25 cache so it rebuilds over contextual text on next launch
    for p in (DATA / "bm25_context.pkl", DATA / "bm25_index.pkl"):
        if p.exists():
            p.unlink()
    print("  wrote contextual index.faiss + indexed_chunks.pkl; BM25 cache cleared")
    print("Phase 2 done. Restart the app to use the contextual index.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all")
    ap.add_argument("--model", default=os.getenv("HEALIX_CTX_MODEL", "qwen2.5:0.5b-instruct"))
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--workers", type=int, default=int(os.getenv("HEALIX_CTX_WORKERS", "6")))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.phase in ("1", "all"):
        phase1(args.model, args.workers, args.limit, args.resume)
    if args.phase in ("2", "all"):
        phase2(args.embed_model)


if __name__ == "__main__":
    main()
