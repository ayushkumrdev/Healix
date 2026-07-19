#!/usr/bin/env python3
"""Rigorous latency benchmark for the HyDE small-model optimization.

HyDE runs a language model *before* first-stage retrieval, so its cost is on the
critical path to the first generated token. We measure the HyDE call latency
distribution for the full generation model vs. a sub-billion-parameter model
across a fixed query set, and the end-to-end retrieval phase each implies.

Reports mean, median, p95 and a paired bootstrap 95% CI on the per-query speedup.

Usage:  .venv/Scripts/python.exe scripts/eval_latency.py --n 25
"""
import argparse, json, os, statistics, sys, time, random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
ROOT = Path(__file__).resolve().parent.parent

QUERIES = [
    "why do i get headaches in the afternoon", "what causes high blood pressure",
    "i feel dizzy when i stand up", "how does stress affect sleep",
    "what are symptoms of low iron", "my knee hurts when i climb stairs",
    "is it normal to feel tired all the time", "what helps with acid reflux",
    "how much water should i drink daily", "what causes muscle cramps at night",
    "why is my heart racing", "what are early signs of diabetes",
    "how do i lower my cholesterol", "what causes lower back pain",
    "is coffee bad for anxiety", "how does smoking affect the lungs",
    "what are symptoms of dehydration", "why do i bruise easily",
    "what causes frequent urination", "how does menopause affect the body",
    "what helps a sore throat", "why do i feel short of breath",
    "what are signs of vitamin d deficiency", "how does alcohol affect the liver",
    "what causes ringing in the ears",
]


def hyde_latency(model, queries):
    from services import hyde
    os.environ["HEALIX_HYDE_MODEL"] = model
    hyde.generate_hypotheticals("warmup query for model load")  # warm
    lats, ok = [], 0
    for q in queries:
        t = time.time()
        h = hyde.generate_hypotheticals(q)
        lats.append(time.time() - t)
        if h.get("specific"):
            ok += 1
    return lats, ok


def summ(lats):
    s = sorted(lats)
    return {"mean": round(statistics.mean(lats), 2),
            "median": round(statistics.median(lats), 2),
            "p95": round(s[int(0.95 * (len(s) - 1))], 2),
            "min": round(s[0], 2), "max": round(s[-1], 2)}


def bootstrap_speedup(slow, fast, iters=5000, seed=1):
    rng = random.Random(seed); n = len(slow); diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(slow[i] for i in idx)/n - sum(fast[i] for i in idx)/n)
    diffs.sort()
    return (round(statistics.mean(slow) - statistics.mean(fast), 2),
            round(diffs[int(0.025*iters)], 2), round(diffs[int(0.975*iters)], 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--big", default=os.getenv("BAYMAX_OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    ap.add_argument("--small", default="qwen2.5:0.5b-instruct")
    args = ap.parse_args()
    queries = QUERIES[: args.n]

    print(f"Measuring HyDE latency: big={args.big} vs small={args.small} on {len(queries)} queries")
    big_lat, big_ok = hyde_latency(args.big, queries)
    small_lat, small_ok = hyde_latency(args.small, queries)

    pt, lo, hi = bootstrap_speedup(big_lat, small_lat)
    report = {"n": len(queries), "big_model": args.big, "small_model": args.small,
              "big_hyde": summ(big_lat), "small_hyde": summ(small_lat),
              "big_parse_ok": big_ok, "small_parse_ok": small_ok,
              "speedup_seconds_per_query": {"point": pt, "ci95": [lo, hi]},
              "speedup_factor": round(statistics.mean(big_lat) / max(1e-6, statistics.mean(small_lat)), 2)}
    out = ROOT / "data" / "eval_latency.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
