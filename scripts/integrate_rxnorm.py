#!/usr/bin/env python3
"""
Integrate RxNorm chunks into an existing FAISS index without a full rebuild.

Inputs:
- data/chunks/rxnorm_medications.jsonl (from scripts/ingest_rxnorm.py)
- Existing FAISS index at data/ (index.faiss, indexed_chunks.pkl, faiss_metadata.pkl)

Usage (PowerShell):
  .venv/Scripts/python.exe scripts/integrate_rxnorm.py \
    --chunks data/chunks/rxnorm_medications.jsonl --index-dir data
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', default='data/chunks/rxnorm_medications.jsonl')
    ap.add_argument('--index-dir', default='data')
    ap.add_argument('--model', default='all-MiniLM-L6-v2')
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index_path = index_dir / 'index.faiss'
    chunks_pkl = index_dir / 'indexed_chunks.pkl'
    meta_pkl = index_dir / 'faiss_metadata.pkl'

    if not index_path.exists() or not chunks_pkl.exists() or not meta_pkl.exists():
        raise FileNotFoundError('Existing FAISS index not found under data/. Build it once before integrating.')

    rx_path = Path(args.chunks)
    if not rx_path.exists():
        raise FileNotFoundError(f'RxNorm chunks not found: {rx_path}. Run ingest_rxnorm.py first.')

    # Load existing index
    print(f'Loading index: {index_path}')
    index = faiss.read_index(str(index_path))

    import pickle
    with chunks_pkl.open('rb') as f:
        existing_chunks: List[Dict] = pickle.load(f)
    with meta_pkl.open('rb') as f:
        existing_meta: List[Dict] = pickle.load(f)

    # Load new chunks
    print(f'Loading RxNorm chunks: {rx_path}')
    new_rows = load_jsonl(rx_path)
    if not new_rows:
        print('No RxNorm rows loaded; aborting.')
        return

    # Check duplicates by id
    existing_ids = set(r.get('id') for r in existing_chunks)
    add_rows = [r for r in new_rows if r.get('id') not in existing_ids]
    if not add_rows:
        print('All RxNorm rows already present; nothing to add.')
        return

    # Compute embeddings
    print('Loading embedding model:', args.model)
    model = SentenceTransformer(args.model)

    texts = [r.get('text', '') for r in add_rows]
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embs)

    # Add to index
    print(f'Adding {len(add_rows)} embeddings to index...')
    index.add(embs.astype(np.float32))

    # Append to data
    existing_chunks.extend(add_rows)
    new_meta: List[Dict] = []
    for r in add_rows:
        new_meta.append({
            'id': r.get('id'),
            'doc_id': r.get('doc_id', r.get('id')),
            'source': r.get('source'),
            'category': r.get('category'),
            'type': r.get('type'),
        })
    existing_meta.extend(new_meta)

    # Save back
    print('Saving updated index and metadata...')
    faiss.write_index(index, str(index_path))
    with chunks_pkl.open('wb') as f:
        pickle.dump(existing_chunks, f)
    with meta_pkl.open('wb') as f:
        pickle.dump(existing_meta, f)

    # Update index_config.json if present
    config_path = index_dir / 'index_config.json'
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text('utf-8'))
        except Exception:
            cfg = {}
    else:
        cfg = {}

    cfg['total_vectors'] = index.ntotal
    cfg['model_name'] = args.model
    cfg['last_updated'] = __import__('datetime').datetime.utcnow().isoformat() + 'Z'

    config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')

    print('Integration complete!')
    print('New total vectors:', index.ntotal)


if __name__ == '__main__':
    main()

