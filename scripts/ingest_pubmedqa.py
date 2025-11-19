#!/usr/bin/env python3
"""
Ingest PubMed QA dataset into RAG chunks for FAISS.

- Reads JSON (array) or JSONL records from the input path.
- Normalizes records to QA chunk schema (source=PubMedQA, category=PubMed_QA, type=QA_pair).
- Writes data/chunks/pubmedqa_chunks.jsonl

Default input path:
  datasets/roxan_data/qa/pubmed/ori_pqaa.json

Usage:
  python scripts/ingest_pubmedqa.py [--input PATH] [--output PATH]
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Iterable
import re

try:
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
except Exception:
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        t = text if len(text) <= 4000 else text[:4000]
        if _TOKENIZER is not None:
            return len(_TOKENIZER.encode(t, truncation=True, max_length=512))
    except Exception:
        pass
    return max(1, len(text) // 4)


def chunk_text(text: str, max_tokens: int = 220, overlap_tokens: int = 40) -> List[str]:
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]
    chunks: List[str] = []
    cur: List[str] = []
    for w in words:
        test = cur + [w]
        ttxt = " ".join(test)
        if count_tokens(ttxt) > max_tokens:
            if cur:
                chunks.append(" ".join(cur))
                overlap = cur[-overlap_tokens:] if len(cur) > overlap_tokens else cur
                cur = overlap + [w]
            else:
                cur = [w]
        else:
            cur = test
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _clean(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s.strip())
    return s


def _iter_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    # Detect JSON array, dict-of-records, or JSONL
    with path.open("r", encoding="utf-8") as f:
        # Read first non-whitespace character to detect structure
        head = None
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                head = ch
                break
        f.seek(0)
        if head in ("[", "{"):
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        if isinstance(rec, dict):
                            yield rec
                    return
                if isinstance(data, dict):
                    # Some PubMedQA variants are dict keyed by id -> record
                    for rec in data.values():
                        if isinstance(rec, dict):
                            yield rec
                    return
            except Exception:
                # Fall through to JSONL parsing
                f.seek(0)
        # Else treat as JSONL
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    yield rec
            except Exception:
                continue


def _get_question(rec: Dict[str, Any]) -> str:
    for k in ("question", "q", "Question", "QUESTION"):
        if k in rec:
            return _clean(str(rec[k]))
    return ""


def _get_answer(rec: Dict[str, Any]) -> str:
    # PubMedQA often has final_decision / long_answer
    for k in ("answer", "a", "final_decision", "long_answer", "LongAnswer", "LONG_ANSWER"):
        if k in rec:
            return _clean(str(rec[k]))
    return ""


def _get_context(rec: Dict[str, Any]) -> str:
    # Try common keys
    ctx_fields = [
        "context", "contexts", "abstract", "abstracts", "passages", "evidence", "article", "docs"
    ]
    for k in ctx_fields:
        if k in rec:
            v = rec[k]
            if isinstance(v, list):
                # Join list of strings
                items = []
                for it in v:
                    if isinstance(it, str):
                        items.append(it)
                    elif isinstance(it, dict):
                        # common shapes: {text: ...}
                        txt = it.get("text") or it.get("abstract") or it.get("content")
                        if txt:
                            items.append(str(txt))
                return _clean(" \n".join(items))
            elif isinstance(v, str):
                return _clean(v)
            else:
                try:
                    return _clean(json.dumps(v)[:1200])
                except Exception:
                    return ""
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path("datasets") / "roxan_data" / "qa" / "pubmed" / "ori_pqaa.json"))
    parser.add_argument("--output", default=str(Path("data") / "chunks" / "pubmedqa_chunks.jsonl"))
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return

    print("Healix - PubMedQA Ingestion")
    print("=" * 30)
    print(f"Input: {in_path}")
    print(f"Output: {out_path}")

    total_records = 0
    chunk_id = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in _iter_json_records(in_path):
            total_records += 1
            q = _get_question(rec)
            a = _get_answer(rec)
            ctx = _get_context(rec)

            if not q and not a and not ctx:
                continue

            # Build combined text (cap context to avoid huge prompts)
            ctx_cap = ctx[:1500] if ctx else ""
            combined = f"Question: {q}\nAnswer: {a}" + (f"\nContext: {ctx_cap}" if ctx_cap else "")

            pieces = chunk_text(combined, max_tokens=220, overlap_tokens=40)
            for ci, piece in enumerate(pieces):
                row = {
                    "id": f"pmqa_chunk_{chunk_id:07d}",
                    "doc_id": f"pmqa_doc_{total_records:07d}",
                    "chunk_index": ci,
                    "text": piece,
                    "source": "PubMedQA",
                    "category": "PubMed_QA",
                    "file": in_path.name,
                    "url": "",
                    "type": "QA_pair",
                    "token_count": count_tokens(piece)
                }
                if q:
                    row["question"] = q
                if a:
                    row["answer"] = a
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                chunk_id += 1

    print(f"Records processed: {total_records}")
    print(f"Chunks written: {chunk_id}")


if __name__ == "__main__":
    main()

