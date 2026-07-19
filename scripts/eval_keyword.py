#!/usr/bin/env python3
"""Keyword-query retrieval: dense vs hybrid, first-stage and reranked.

Motivation and pre-registered hypothesis (stated before running): dense encoders
are trained on natural-language and underperform on terse keyword queries, while
BM25 excels at exact lexical matching. Many real users type keywords ("glaucoma
treatment") rather than full sentences. We therefore hypothesize:
  H1: on keyword queries, hybrid first-stage recall > dense first-stage recall.
  H2: the gap is larger for keyword queries than for natural-language queries.

Deterministic keyword extraction (no LLM) lets us run a large, cheap sample.
Gold = the source chunk. Four conditions as in eval_firststage. We report a
paired bootstrap 95% CI on the hybrid_raw - dense_raw Recall@10 gap and the
rescue counts.

Usage:  .venv/Scripts/python.exe scripts/eval_keyword.py --n 500 --seed 13
"""
import argparse, json, os, pickle, random, re, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
ROOT = Path(__file__).resolve().parent.parent

STOP = set(("what whats what's is are was were be been being the a an of for to in on "
            "with and or how do does did can could would should i my me you your our "
            "when where which who whom that this these those about it its as at by from "
            "have has had will there their they them but not any some most more will "
            "if then than so such also into over under out up down between during").split())


def to_keywords(question: str) -> str:
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-']+", (question or "").lower())
    kws = [t for t in toks if t not in STOP and len(t) > 2]
    return " ".join(kws) if kws else (question or "")


def rank_of(gold, results):
    for i, r in enumerate(results):
        if r.get("chunk_id") == gold:
            return i + 1
    return 0


def metrics(ranks, k=10):
    n = len(ranks)
    return {"recall@1": round(sum(1 for r in ranks if 0 < r <= 1) / n, 3),
            "recall@5": round(sum(1 for r in ranks if 0 < r <= 5) / n, 3),
            "recall@10": round(sum(1 for r in ranks if 0 < r <= k) / n, 3),
            "mrr@10": round(sum((1.0 / r) if r else 0.0 for r in ranks) / n, 3)}


def bootstrap_ci(hit_h, hit_d, iters=5000, seed=1):
    rng = random.Random(seed); n = len(hit_h); diffs = []
    for _ in range(iters):
        s = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(hit_h[i] for i in s) / n - sum(hit_d[i] for i in s) / n)
    diffs.sort()
    return (round(sum(hit_h)/n - sum(hit_d)/n, 3),
            round(diffs[int(0.025*iters)], 3), round(diffs[int(0.975*iters)], 3))


def run_conditions(retr, sample, querymap, k):
    ranks = {"dense_raw": [], "hybrid_raw": [], "dense_rr": [], "hybrid_rr": []}
    saved_r, saved_n = retr.reranker, retr.reranker_model_name
    retr.reranker, retr.reranker_model_name = None, ""
    for ch in sample:
        q = querymap[ch["id"]]
        ranks["dense_raw"].append(rank_of(ch["id"], retr.retrieve(q, k=k, min_score=0.0)))
        ranks["hybrid_raw"].append(rank_of(ch["id"], retr.hybrid_retrieve(query=q, k=k)))
    retr.reranker, retr.reranker_model_name = saved_r, saved_n
    if retr.reranker is None and retr.reranker_model_name:
        retr._load_reranker()
    for ch in sample:
        q = querymap[ch["id"]]
        ranks["dense_rr"].append(rank_of(ch["id"], retr.retrieve(q, k=k, min_score=0.0)))
        ranks["hybrid_rr"].append(rank_of(ch["id"], retr.hybrid_retrieve(query=q, k=k)))
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    print("Loading chunks...")
    chunks = pickle.load(open(ROOT / "data" / "indexed_chunks.pkl", "rb"))
    qa = [c for c in chunks if c.get("type") == "QA_pair" and c.get("question")
          and c.get("id") and len(str(c.get("question", "")).split()) >= 4]
    random.seed(args.seed)
    sample = random.sample(qa, min(args.n, len(qa)))
    kw = {ch["id"]: to_keywords(ch["question"]) for ch in sample}
    print("example keyword queries:")
    for ch in sample[:3]:
        print(f"   '{ch['question']}' -> '{kw[ch['id']]}'")

    from services.retriever import MedicalRetriever
    retr = MedicalRetriever(index_dir=str(ROOT / "data"))
    retr.retrieve("warmup", k=3)

    print(f"Running 4 conditions x {len(sample)} keyword queries...")
    ranks = run_conditions(retr, sample, kw, args.k)

    report = {"query_style": "keyword", "n": len(sample), "k": args.k, "seed": args.seed,
              "conditions": {c: metrics(rk, args.k) for c, rk in ranks.items()}}
    hh = [1 if 0 < r <= args.k else 0 for r in ranks["hybrid_raw"]]
    hd = [1 if 0 < r <= args.k else 0 for r in ranks["dense_raw"]]
    pt, lo, hi = bootstrap_ci(hh, hd)
    report["firststage_recall10_gap"] = {"hybrid_minus_dense": pt, "ci95": [lo, hi],
                                         "significant": lo > 0 or hi < 0}
    report["firststage_rescue"] = {
        "hybrid_rescued": sum(1 for a, b in zip(hh, hd) if a and not b),
        "hybrid_lost": sum(1 for a, b in zip(hh, hd) if b and not a)}
    # also MRR gap first-stage
    import statistics
    mrr_h = statistics.mean((1.0/r if r else 0.0) for r in ranks["hybrid_raw"])
    mrr_d = statistics.mean((1.0/r if r else 0.0) for r in ranks["dense_raw"])
    report["firststage_mrr_gap"] = round(mrr_h - mrr_d, 3)

    out = ROOT / "data" / "eval_keyword.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== RESULTS ===")
    print(json.dumps(report, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
