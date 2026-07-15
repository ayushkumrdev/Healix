#!/usr/bin/env python3
"""Smoke test for the Healix Mixture-of-Experts router.

Routes a few sample queries and prints which specialist experts the gate
activates and with what weights. Loads only the retriever's embedding model
(no LLM), so it is fast.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.retriever import create_retriever
from services.moe_router import ExpertRouter


SAMPLES = [
    ("crushing chest pain and palpitations", None),
    ("pounding headache and dizziness for three days", None),
    ("a dark irregular mole that changed shape", "skin"),
    ("shortness of breath, here is my scan", "chest_xray"),
    ("can I take ibuprofen with my blood pressure pills", None),
    ("constant stress and I can't sleep", None),
]


def main():
    retriever = create_retriever()
    router = ExpertRouter(retriever)
    print("MoE router — sample routing\n" + "=" * 40)
    for q, modality in SAMPLES:
        routing, _ = router.route(q, modality)
        tag = f" [+image:{modality}]" if modality else ""
        experts = ", ".join(f"{r['name']} {r['weight']:.0%}" for r in routing)
        print(f"\nQ: {q}{tag}\n   -> {experts}")
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
