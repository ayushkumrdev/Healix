#!/usr/bin/env python3
"""
Prepare SFT dataset from FAISS chunks for medical instruction tuning.

Creates JSONL with fields: instruction, input, output, system, tags.
Sources used:
- data/chunks/pubmedqa_chunks.jsonl (QA)
- data/chunks/all_chunks.jsonl (QA)

Outputs:
- data/sft/train_med.jsonl
- data/sft/dev_med.jsonl

Notes:
- We only include items that have explicit question+answer to ensure correct supervision.
- The system prompt enforces professional medical tone and a strict JSON output.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import random

BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
OUT_DIR = BASE_DIR / "data" / "sft"

SOURCES = [
    CHUNKS_DIR / "pubmedqa_chunks.jsonl",
    CHUNKS_DIR / "all_chunks.jsonl",
]

SYSTEM_PROMPT = (
    "You are a clinical assistant. Use a professional medical tone, evidence-based and conservative language. "
    "Respond ONLY in JSON with keys: answer (string), confidence (one of: low, medium, high), citations (array of objects with fields: source, category, file, url)."
)

INSTRUCTION_PROMPT = (
    "Answer the medical question strictly in the required JSON format. "
    "If the provided context is insufficient, set confidence='low' and keep citations empty."
)


def build_record(rec: Dict[str, Any], source_tag: str) -> Optional[Dict[str, Any]]:
    q = rec.get("question")
    a = rec.get("answer")
    if not q or not a:
        return None

    text = rec.get("text", "")
    source = rec.get("source", "")
    category = rec.get("category", "")
    file = rec.get("file", "")
    url = rec.get("url", "")

    # Build input and output
    the_input = f"Question: {q}\nContext: {text[:1500]}" if text else f"Question: {q}"

    output_obj = {
        "answer": a,
        "confidence": "medium",
        "citations": []
    }
    if any([source, category, file, url]):
        output_obj["citations"].append({
            "source": source,
            "category": category,
            "file": file,
            "url": url,
        })

    return {
        "instruction": INSTRUCTION_PROMPT,
        "input": the_input,
        "output": json.dumps(output_obj, ensure_ascii=False),
        "system": SYSTEM_PROMPT,
        "tags": [source_tag]
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = []

    for src in SOURCES:
        if not src.exists():
            continue
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ex = build_record(rec, src.stem)
                if ex:
                    examples.append(ex)

    # Shuffle and split
    random.seed(42)
    random.shuffle(examples)

    n = len(examples)
    dev_size = max(500, int(0.01 * n)) if n > 10000 else max(100, int(0.02 * n))
    dev_size = min(dev_size, max(2000, int(0.05 * n)))  # cap

    dev = examples[:dev_size]
    train = examples[dev_size:]

    train_path = OUT_DIR / "train_med.jsonl"
    dev_path = OUT_DIR / "dev_med.jsonl"

    with train_path.open("w", encoding="utf-8") as w:
        for ex in train:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with dev_path.open("w", encoding="utf-8") as w:
        for ex in dev:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Prepared SFT dataset: {len(train)} train, {len(dev)} dev")
    print(f"Train: {train_path}")
    print(f"Dev  : {dev_path}")


if __name__ == "__main__":
    main()

