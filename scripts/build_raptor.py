#!/usr/bin/env python3
"""Build the RAPTOR summary tree over the existing chunk index.

Levels:
  0  the 239k leaf chunks (existing data/index.faiss — untouched)
  1  ~topic nodes: k-means clusters of leaf embeddings, each summarized by the
     local Ollama model into a neutral "topic overview" paragraph
  2  ~domain nodes: k-means over the level-1 summaries, summarized again

Leaf embeddings are reconstructed from the FAISS index (no re-embedding).
Summaries are appended to data/raptor_l{1,2}.jsonl as they finish, so the
build is fully resumable — rerun the same command to continue, or pass
--cap N to do N summaries per run. Clusters are processed largest-first so
partial builds cover the biggest topics. `--finalize` embeds whatever nodes
exist into data/raptor.faiss + data/raptor_nodes.pkl for the runtime.

Full build (approx. 2h of local Ollama time):
  .venv/Scripts/python.exe scripts/build_raptor.py --topics 1500 --domains 120
Validation build:
  .venv/Scripts/python.exe scripts/build_raptor.py --topics 1500 --cap 24
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import faiss

# Windows consoles default to cp1252; medical titles carry Greek letters
# (β-blocker, α-synuclein) and other non-Latin-1 glyphs. Make every print
# tolerant so a progress line can never crash the build (data is UTF-8 JSONL).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def ollama(prompt: str, num_predict: int = 280) -> str:
    import requests
    url = os.getenv("BAYMAX_OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("BAYMAX_OLLAMA_MODEL", "qwen2.5:7b-instruct")
    r = requests.post(f"{url}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": num_predict},
    }, timeout=300)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def summarize(texts, kind: str) -> dict:
    joined = "\n".join(f"- {t}" for t in texts)[:5000]
    out = ollama(
        "You are compiling a medical reference. The excerpts below all belong "
        f"to one {kind}.\n\n{joined}\n\n"
        "Write exactly two labeled lines:\n"
        "TITLE: a 3-8 word name for this topic.\n"
        "OVERVIEW: a 4-6 sentence neutral, factual overview of the topic that "
        "these excerpts cover — themes, typical conditions, treatments or "
        "mechanisms involved. Encyclopedic prose only: no advice, no bullet "
        "points, no questions.")
    import re
    title = ""
    body = out
    m = re.search(r"TITLE\s*:\s*(.+)", out, flags=re.IGNORECASE)
    if m:
        title = m.group(1).strip().splitlines()[0].strip()[:80]
    m = re.search(r"OVERVIEW\s*:\s*(.+)", out, flags=re.IGNORECASE | re.DOTALL)
    if m:
        body = m.group(1).strip()
    body = " ".join(body.split())[:1500]
    return {"title": title, "text": body}


def load_done(path: Path) -> dict:
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done[obj["id"]] = obj
                except Exception:
                    continue
    return done


def cluster(vectors: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    km = faiss.Kmeans(vectors.shape[1], k, niter=15, seed=seed, verbose=True,
                      max_points_per_centroid=10_000_000)
    km.train(vectors)
    _, assign = km.index.search(vectors, 1)
    return assign.ravel()


def rep_texts(chunks, member_idx, vectors, n=8):
    """Pick up to n members nearest the cluster centroid; prefer questions."""
    sub = vectors[member_idx]
    centroid = sub.mean(axis=0, keepdims=True)
    faiss.normalize_L2(centroid)
    sims = (sub @ centroid.T).ravel()
    order = np.argsort(-sims)[: n * 2]
    texts, seen = [], set()
    for j in order:
        c = chunks[member_idx[j]]
        q = (c.get("question") or "").strip()
        t = q if q else (c.get("text") or "")[:200]
        key = t.lower()[:80]
        if t and key not in seen:
            seen.add(key)
            texts.append(t[:300])
        if len(texts) >= n:
            break
    return texts


def build_level(vectors, source_texts_fn, k, out_path, cap, kind):
    assign = cluster(vectors, k)
    sizes = np.bincount(assign, minlength=k)
    order = np.argsort(-sizes)  # biggest topics first for capped runs
    done = load_done(out_path)
    todo = [int(c) for c in order if sizes[c] >= 3 and f"{kind}_{int(c)}" not in done]
    print(f"[{kind}] clusters: {k}, already summarized: {len(done)}, todo: {len(todo)}")
    if cap:
        todo = todo[:cap]
    with open(out_path, "a", encoding="utf-8") as f:
        for n, c in enumerate(todo, 1):
            member_idx = np.where(assign == c)[0]
            texts = source_texts_fn(member_idx)
            if not texts:
                continue
            try:
                s = summarize(texts, kind)
            except Exception as e:
                print(f"  summarize failed for cluster {c}: {e}")
                continue
            node = {"id": f"{kind}_{c}", "level": 1 if kind == "topic" else 2,
                    "title": s["title"], "text": s["text"], "size": int(sizes[c])}
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{n}/{len(todo)}] {node['id']} ({node['size']} members): {s['title']}")
    return assign


def finalize():
    """Embed all summarized nodes into data/raptor.faiss + raptor_nodes.pkl."""
    nodes = list(load_done(DATA / "raptor_l1.jsonl").values()) + \
            list(load_done(DATA / "raptor_l2.jsonl").values())
    nodes = [n for n in nodes if n.get("text")]
    if not nodes:
        print("No nodes to finalize.")
        return
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.getenv("BAYMAX_EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5"))
    texts = [f"{n.get('title', '')}. {n['text']}" for n in nodes]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(DATA / "raptor.faiss"))
    with open(DATA / "raptor_nodes.pkl", "wb") as f:
        pickle.dump(nodes, f)
    print(f"Finalized {len(nodes)} nodes -> data/raptor.faiss + data/raptor_nodes.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=1500)
    ap.add_argument("--domains", type=int, default=0,
                    help="build level-2 domain nodes (needs level 1 complete)")
    ap.add_argument("--cap", type=int, default=0, help="max summaries this run")
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()

    if args.finalize_only:
        finalize()
        return

    print("Loading chunks + reconstructing leaf embeddings from FAISS...")
    with open(DATA / "indexed_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(str(DATA / "index.faiss"))
    vectors = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    print(f"  {len(chunks)} chunks, {vectors.shape} vectors")

    build_level(vectors, lambda mi: rep_texts(chunks, mi, vectors),
                args.topics, DATA / "raptor_l1.jsonl", args.cap, "topic")

    if args.domains:
        l1 = list(load_done(DATA / "raptor_l1.jsonl").values())
        if len(l1) < args.domains * 2:
            print(f"Level 1 too small ({len(l1)}) for {args.domains} domains; skip.")
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(os.getenv("BAYMAX_EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5"))
            emb = model.encode([n["text"] for n in l1], convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(emb)
            build_level(emb, lambda mi: [l1[i]["text"][:300] for i in mi[:8]],
                        args.domains, DATA / "raptor_l2.jsonl", args.cap, "domain")

    finalize()


if __name__ == "__main__":
    main()
