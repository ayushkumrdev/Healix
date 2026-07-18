#!/usr/bin/env python3
"""Query-variation retrieval evaluation for the Healix paper.

The corpus questions are verbatim in the index, so evaluating with the raw
question is circular (dense trivially self-matches). Instead we paraphrase each
sampled question into colloquial patient phrasing, keep the *original chunk* as
the gold target, and measure whether each retrieval method recovers it. This
tests exactly the robustness-to-phrasing that hybrid/HyDE claim to improve.

Conditions:
  dense        - MedicalRetriever.retrieve (FAISS + cross-encoder rerank)
  hybrid       - hybrid_retrieve (dense + contextual BM25, RRF, rerank)
  hybrid+hyde  - hybrid_retrieve on [paraphrase + HyDE specific hypothetical]

Metrics: Recall@1/5/10, MRR@10, mean latency. Results -> data/eval_retrieval.json
Usage:  BAYMAX_OLLAMA_MODEL=qwen2.5:7b-instruct .venv/Scripts/python.exe scripts/eval_retrieval.py --n 150
"""
import argparse, json, os, pickle, random, time, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
ROOT = Path(__file__).resolve().parent.parent


def paraphrase(question: str, model: str, url: str) -> str:
    import requests
    prompt = ("Rewrite this medical question the casual way a worried patient "
              "would type it to a health app — plain words, no jargon, keep the "
              "medical meaning. Output only the rewritten question.\n\n"
              f"Question: {question}\nRewritten:")
    try:
        r = requests.post(f"{url}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0.7, "num_predict": 60, "top_p": 0.9}}, timeout=60)
        r.raise_for_status()
        out = (r.json().get("response") or "").strip().splitlines()
        return out[0].strip().strip('"') if out else ""
    except Exception:
        return ""


def rank_of(gold_id, results):
    for i, r in enumerate(results):
        if r.get("chunk_id") == gold_id:
            return i + 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--hyde_n", type=int, default=0,
                    help="cap the (slow) HyDE arm to first N queries; 0 = all")
    args = ap.parse_args()

    url = os.getenv("BAYMAX_OLLAMA_URL", "http://localhost:11434").rstrip("/")
    para_model = os.getenv("BAYMAX_OLLAMA_MODEL", "qwen2.5:7b-instruct")

    print("Loading chunks + retriever...")
    chunks = pickle.load(open(ROOT / "data" / "indexed_chunks.pkl", "rb"))
    qa = [c for c in chunks if c.get("type") == "QA_pair" and c.get("question")
          and c.get("id") and len(str(c.get("question", "")).split()) >= 4]
    random.seed(args.seed)
    sample = random.sample(qa, min(args.n, len(qa)))

    from services.retriever import MedicalRetriever
    from services import hyde
    retr = MedicalRetriever(index_dir=str(ROOT / "data"))
    retr.retrieve("warmup", k=3)  # warm embedder/reranker

    conds = ["dense", "hybrid", "hybrid+hyde"]
    agg = {c: {"r1": 0, "r5": 0, "r10": 0, "mrr": 0.0, "lat": 0.0, "n": 0} for c in conds}
    para_ok = 0

    for i, ch in enumerate(sample):
        gold = ch["id"]
        q = paraphrase(ch["question"], para_model, url) or ch["question"]
        if q != ch["question"]:
            para_ok += 1

        # dense
        t = time.time(); res = retr.retrieve(q, k=args.k, min_score=0.0); lat = time.time() - t
        _score(agg["dense"], rank_of(gold, res), lat, args.k)
        # hybrid
        t = time.time(); res = retr.hybrid_retrieve(query=q, k=args.k); lat = time.time() - t
        _score(agg["hybrid"], rank_of(gold, res), lat, args.k)
        # hybrid + hyde (latency includes the HyDE generation, as in production)
        if not args.hyde_n or i < args.hyde_n:
            t = time.time()
            h = hyde.generate_hypotheticals(q) or {}
            q2 = f"{q}\n{h['specific']}" if h.get("specific") else q
            res = retr.hybrid_retrieve(query=q2, k=args.k); lat = time.time() - t
            _score(agg["hybrid+hyde"], rank_of(gold, res), lat, args.k)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample)} done")

    report = {"n": len(sample), "k": args.k, "seed": args.seed,
              "paraphrase_rate": round(para_ok / max(1, len(sample)), 3),
              "paraphrase_model": para_model, "conditions": {}}
    for c in conds:
        a = agg[c]; n = max(1, a["n"])
        report["conditions"][c] = {
            "n": a["n"],
            "recall@1": round(a["r1"] / n, 3),
            "recall@5": round(a["r5"] / n, 3),
            "recall@10": round(a["r10"] / n, 3),
            "mrr@10": round(a["mrr"] / n, 3),
            "mean_latency_s": round(a["lat"] / n, 3),
        }
    out = ROOT / "data" / "eval_retrieval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== RESULTS ===")
    print(json.dumps(report, indent=2))
    print("saved ->", out)


def _score(a, rank, lat, k):
    a["n"] += 1
    a["lat"] += lat
    if rank:
        if rank <= 1: a["r1"] += 1
        if rank <= 5: a["r5"] += 1
        if rank <= k: a["r10"] += 1
        a["mrr"] += 1.0 / rank


if __name__ == "__main__":
    main()
