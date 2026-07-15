#!/usr/bin/env python3
"""Build the BM25 index (if needed) and compare dense vs hybrid retrieval.

Shows, for a few queries, the top hits from dense-only vs hybrid (dense + BM25
fused with RRF), highlighting where the lexical arm pulls in exact-term matches
the embeddings miss. Run once to warm the BM25 cache before serving.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.retriever import create_retriever


QUERIES = [
    "metformin side effects",
    "can I take ibuprofen with lisinopril",
    "symptoms of pneumothorax",
    "what is hashimoto thyroiditis",
]


def short(r):
    return (r.get("source", "?"), round(r.get("score", 0.0), 3), (r.get("text", "")[:70]).replace("\n", " "))


def main():
    r = create_retriever()
    print("Warming BM25 + comparing dense vs hybrid\n" + "=" * 50)
    for q in QUERIES:
        dense = r.retrieve(q, k=3, min_score=0.0)
        hybrid = r.hybrid_retrieve(q, k=3)
        print(f"\nQ: {q}")
        print("  dense :")
        for d in dense:
            print("    -", short(d))
        print("  hybrid:")
        for h in hybrid:
            print("    -", short(h))
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
