#!/usr/bin/env python3
"""First-stage vs reranked retrieval: does hybrid (dense+BM25+RRF) beat pure
dense before a cross-encoder reranker equalizes them?

Pre-registered hypothesis (from the IR literature, stated before running):
  H1: hybrid first-stage recall > dense first-stage recall.
  H2: after cross-encoder reranking the two converge.
If H1+H2 hold, hybrid's value is concentrated in the reranker-free regime, which
is exactly the latency-constrained on-device setting Healix targets.

Protocol: query-variation. Sample n QA chunks, paraphrase each question into
colloquial patient phrasing once (cached), keep the source chunk as gold, and
measure Recall@1/5/10 and MRR@10 under four conditions:
  dense_raw    - FAISS only (reranker disabled)
  hybrid_raw   - dense+BM25 fused by RRF (reranker disabled)
  dense_rr     - FAISS pool + cross-encoder rerank
  hybrid_rr    - RRF pool + cross-encoder rerank
We bootstrap a 95% CI on the hybrid_raw - dense_raw Recall@10 gap.

Usage:
  BAYMAX_OLLAMA_MODEL=qwen2.5:7b-instruct .venv/Scripts/python.exe \
    scripts/eval_firststage.py --n 300 --seed 13
"""
import argparse, json, os, pickle, random, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
ROOT = Path(__file__).resolve().parent.parent


def paraphrase(question, model, url):
    import requests
    prompt = ("Rewrite this medical question the casual way a worried patient "
              "would type it to a health app - plain words, no jargon, keep the "
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


def rank_of(gold, results):
    for i, r in enumerate(results):
        if r.get("chunk_id") == gold:
            return i + 1
    return 0


def metrics(ranks, k=10):
    n = len(ranks)
    r1 = sum(1 for r in ranks if 0 < r <= 1) / n
    r5 = sum(1 for r in ranks if 0 < r <= 5) / n
    r10 = sum(1 for r in ranks if 0 < r <= k) / n
    mrr = sum((1.0 / r) if r else 0.0 for r in ranks) / n
    return {"recall@1": round(r1, 3), "recall@5": round(r5, 3),
            "recall@10": round(r10, 3), "mrr@10": round(mrr, 3)}


def bootstrap_ci(hit_h, hit_d, iters=5000, seed=1):
    """95% CI on mean(hit_h) - mean(hit_d), paired bootstrap over queries."""
    rng = random.Random(seed)
    n = len(hit_h)
    diffs = []
    idx = list(range(n))
    for _ in range(iters):
        s = [rng.choice(idx) for _ in range(n)]
        dh = sum(hit_h[i] for i in s) / n
        dd = sum(hit_d[i] for i in s) / n
        diffs.append(dh - dd)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters)]
    point = sum(hit_h) / n - sum(hit_d) / n
    return round(point, 3), round(lo, 3), round(hi, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    url = os.getenv("BAYMAX_OLLAMA_URL", "http://localhost:11434").rstrip("/")
    para_model = os.getenv("BAYMAX_OLLAMA_MODEL", "qwen2.5:7b-instruct")

    print("Loading chunks...")
    chunks = pickle.load(open(ROOT / "data" / "indexed_chunks.pkl", "rb"))
    qa = [c for c in chunks if c.get("type") == "QA_pair" and c.get("question")
          and c.get("id") and len(str(c.get("question", "")).split()) >= 4]
    random.seed(args.seed)
    sample = random.sample(qa, min(args.n, len(qa)))

    # Paraphrase once, cache to disk keyed by (seed, n).
    cache_path = ROOT / "data" / f"paraphrase_cache_{args.seed}_{args.n}.json"
    if cache_path.exists():
        para = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(para)} cached paraphrases")
    else:
        print("Paraphrasing queries (one-time)...")
        para = {}
        for i, ch in enumerate(sample):
            para[ch["id"]] = paraphrase(ch["question"], para_model, url) or ch["question"]
            if (i + 1) % 25 == 0:
                print(f"  paraphrased {i+1}/{len(sample)}")
        cache_path.write_text(json.dumps(para, ensure_ascii=False), encoding="utf-8")

    from services.retriever import MedicalRetriever
    retr = MedicalRetriever(index_dir=str(ROOT / "data"))
    retr.retrieve("warmup", k=3)

    # ---- Phase 1: reranker DISABLED (first-stage) ----
    print("Phase 1: first-stage (reranker off)...")
    saved_reranker, saved_name = retr.reranker, retr.reranker_model_name
    retr.reranker, retr.reranker_model_name = None, ""
    ranks = {"dense_raw": [], "hybrid_raw": []}
    for ch in sample:
        q = para[ch["id"]]
        ranks["dense_raw"].append(rank_of(ch["id"], retr.retrieve(q, k=args.k, min_score=0.0)))
        ranks["hybrid_raw"].append(rank_of(ch["id"], retr.hybrid_retrieve(query=q, k=args.k)))

    # ---- Phase 2: reranker ENABLED ----
    print("Phase 2: reranked...")
    retr.reranker, retr.reranker_model_name = saved_reranker, saved_name
    if retr.reranker is None and retr.reranker_model_name:
        retr._load_reranker()
    ranks["dense_rr"], ranks["hybrid_rr"] = [], []
    for ch in sample:
        q = para[ch["id"]]
        ranks["dense_rr"].append(rank_of(ch["id"], retr.retrieve(q, k=args.k, min_score=0.0)))
        ranks["hybrid_rr"].append(rank_of(ch["id"], retr.hybrid_retrieve(query=q, k=args.k)))

    report = {"n": len(sample), "k": args.k, "seed": args.seed,
              "paraphrase_model": para_model, "conditions": {}}
    for c, rk in ranks.items():
        report["conditions"][c] = metrics(rk, args.k)

    # Significance: hybrid_raw vs dense_raw on Recall@10 (paired bootstrap)
    hit_h = [1 if 0 < r <= args.k else 0 for r in ranks["hybrid_raw"]]
    hit_d = [1 if 0 < r <= args.k else 0 for r in ranks["dense_raw"]]
    pt, lo, hi = bootstrap_ci(hit_h, hit_d)
    report["firststage_recall10_gap"] = {"hybrid_minus_dense": pt, "ci95": [lo, hi],
                                         "significant": lo > 0 or hi < 0}
    # Rescue rate: gold found by hybrid_raw but missed by dense_raw (and vice versa)
    resc = sum(1 for h, d in zip(hit_h, hit_d) if h and not d)
    lost = sum(1 for h, d in zip(hit_h, hit_d) if d and not h)
    report["firststage_rescue"] = {"hybrid_rescued": resc, "hybrid_lost": lost,
                                   "net": resc - lost}

    out = ROOT / "data" / "eval_firststage.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== RESULTS ===")
    print(json.dumps(report, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
