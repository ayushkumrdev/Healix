#!/usr/bin/env python3
"""
Ingest drug database into RAG chunks for FAISS
- Reads one or more local drug datasets
- Normalizes into the existing chunk schema used by build_faiss.py
- Writes data/chunks/drug_chunks.jsonl

Sources supported by default:
- data/enhanced/drug_data/drug_database.json (array of records with `content`)
- data/enhanced_comprehensive/drug_database.json (array of records with `text`)

This script does NOT fetch from the internet. Place your drug datasets locally
in the above paths or extend `load_local_drug_sources()` accordingly.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
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
    # Fallback estimate (~4 chars per token)
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


def clean(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text.strip())
    return t


def load_local_drug_sources(base_dir: Path) -> List[Tuple[str, List[Dict]]]:
    sources: List[Tuple[str, List[Dict]]] = []

    p1 = base_dir / "data" / "enhanced" / "drug_data" / "drug_database.json"
    if p1.exists():
        try:
            data = json.loads(p1.read_text(encoding="utf-8"))
            if isinstance(data, list):
                sources.append((str(p1), data))
        except Exception as e:
            print(f"Warning: failed reading {p1}: {e}")

    p2 = base_dir / "data" / "enhanced_comprehensive" / "drug_database.json"
    if p2.exists():
        try:
            data = json.loads(p2.read_text(encoding="utf-8"))
            if isinstance(data, list):
                sources.append((str(p2), data))
        except Exception as e:
            print(f"Warning: failed reading {p2}: {e}")

    return sources


def normalize_docs(file_path: str, records: List[Dict]) -> List[Dict]:
    docs: List[Dict] = []
    for rec in records:
        # Prefer rich 'content' then 'text'
        text = rec.get("content") or rec.get("text") or ""
        text = clean(text)
        if not text:
            continue
        source = rec.get("source") or "Drug Database"
        # Normalize RAG category to a single namespace to improve retrieval filtering
        category = "Medications"
        drug_name = rec.get("name") or rec.get("drug_name") or ""
        drug_cat = rec.get("category") or rec.get("drug_category") or ""
        url = rec.get("url") or ""
        type_ = rec.get("type") or "medication_info"

        doc = {
            "text": text,
            "source": source,
            "category": category,
            "file": os.path.basename(file_path),
            "url": url,
            "type": type_,
            "drug_name": drug_name,
            "drug_category": drug_cat,
        }
        docs.append(doc)
    return docs


def write_chunks(docs: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_id = 0
    with out_path.open("w", encoding="utf-8") as f:
        for doc_idx, doc in enumerate(docs):
            pieces = chunk_text(doc["text"], max_tokens=220, overlap_tokens=40)
            for c_idx, piece in enumerate(pieces):
                row = {
                    "id": f"drug_chunk_{chunk_id:06d}",
                    "doc_id": f"drug_doc_{doc_idx:06d}",
                    "chunk_index": c_idx,
                    "text": piece,
                    "source": doc["source"],
                    "category": doc["category"],
                    "file": doc["file"],
                    "url": doc["url"],
                    "type": doc["type"],
                    "token_count": count_tokens(piece),
                }
                # Preserve drug fields in chunk for provenance
                if doc.get("drug_name"):
                    row["drug_name"] = doc["drug_name"]
                if doc.get("drug_category"):
                    row["drug_category"] = doc["drug_category"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                chunk_id += 1
    print(f"Wrote {chunk_id} drug chunks to {out_path}")


def main():
    base_dir = Path(__file__).parent.parent
    out_file = base_dir / "data" / "chunks" / "drug_chunks.jsonl"

    print("Healix - Drug Database Ingestion")
    print("=" * 36)
    sources = load_local_drug_sources(base_dir)
    if not sources:
        print("No local drug sources found. Place your datasets into data/enhanced/drug_data/drug_database.json or data/enhanced_comprehensive/drug_database.json")
        return

    all_docs: List[Dict] = []
    total = 0
    for file_path, records in sources:
        docs = normalize_docs(file_path, records)
        print(f"Loaded {len(docs)} records from {file_path}")
        all_docs.extend(docs)
        total += len(docs)

    if not all_docs:
        print("No normalized docs to ingest.")
        return

    write_chunks(all_docs, out_file)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()

