#!/usr/bin/env python3
"""
Smoke test for the medical scope gate (backend/services/scope_gate.py).

Runs heuristics-only by default (fast, no model load):
    .venv\\Scripts\\python.exe scripts\\check_scope.py

Add --full to also exercise the embedding tiebreaker through the retriever:
    .venv\\Scripts\\python.exe scripts\\check_scope.py --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from services.scope_gate import ScopeGate  # noqa: E402

# (query, expected_kind, in_conversation)
CASES = [
    # courtesy — allowed
    ("hi", "greeting", False),
    ("Hello there", "greeting", False),
    ("good morning healix", "greeting", False),
    ("hey, how are you?", "greeting", False),
    ("how's it going", "greeting", False),
    ("thanks a lot", "thanks", False),
    ("thank you so much", "thanks", False),
    ("bye", "farewell", False),
    ("good night", "farewell", False),
    ("who are you?", "identity", False),
    ("what can you do", "identity", False),
    ("are you a bot", "identity", False),
    # medical — must reach the pipeline
    ("hip pain", "medical", False),                       # old substring bug
    ("hi doctor, my chest hurts when I breathe", "medical", False),
    ("I've had a tension headache for three days", "medical", False),
    ("why do I feel jittery after coffee", "medical", False),
    ("what does ibuprofen do", "medical", False),
    ("is 140/90 blood pressure bad", "medical", False),
    ("how much sleep do adults need", "medical", False),
    ("what should I eat to lose belly fat", "medical", False),
    ("my mole changed color recently", "medical", False),
    ("can stress cause stomach ulcers?", "medical", False),
    ("paracetamol vs ibuprofen for fever", "medical", False),
    # follow-ups inside a conversation — never refused by heuristics
    ("how long will it take?", "medical", True),
    ("and what about my dad?", "medical", True),
    ("more detail please", "medical", True),
    # out of scope — politely redirected
    ("write me a python function to sort a list", "out_of_scope", False),
    ("what is the capital of France?", "out_of_scope", False),
    ("who won the cricket world cup", "out_of_scope", False),
    ("best netflix series this year", "out_of_scope", False),
    ("should I invest in bitcoin", "out_of_scope", False),
    ("plan a trip to Goa", "out_of_scope", False),
    ("write an essay about climate change", "out_of_scope", False),
    ("solve this equation for x", "out_of_scope", False),
    ("recommend a good laptop", "out_of_scope", False),
    ("tell me a joke", "out_of_scope", False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also load the retriever for the embedding tiebreaker")
    args = ap.parse_args()

    retriever = None
    if args.full:
        from services.retriever import MedicalRetriever
        retriever = MedicalRetriever(index_dir=str(ROOT / "data"))

    gate = ScopeGate(retriever)
    passed = failed = 0
    for query, expected, in_conv in CASES:
        r = gate.classify(query, in_conversation=in_conv)
        ok = r.kind == expected
        passed += ok
        failed += (not ok)
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {query!r:55s} -> {r.kind:13s} (want {expected}; {r.reason})")

    if args.full:
        print("\n  embedding tiebreaker (no keywords either way):")
        for q in ["tell me about the french revolution",
                  "my legs feel heavy every morning lately"]:
            r = gate.classify(q)
            print(f"       {q!r:55s} -> {r.kind} ({r.reason})")

    print(f"\n  {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
