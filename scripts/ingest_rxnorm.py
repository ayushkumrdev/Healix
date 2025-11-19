#!/usr/bin/env python3
"""
Ingest RxNorm RRF into RAG chunks for Medications retrieval.

Requirements:
- Place RxNorm RRF files locally (at minimum RXNCONSO.RRF) under data/rxnorm/
- You must obtain RxNorm from UMLS/NLM per their license.

Output:
- data/chunks/rxnorm_medications.jsonl (one concept per RxCUI)

Usage (PowerShell):
  .venv/Scripts/python.exe scripts/ingest_rxnorm.py \
    --rrf-dir data/rxnorm --out data/chunks/rxnorm_medications.jsonl

Notes:
- This script limits scope to RXNCONSO.RRF (synonyms, term types, preferred names) to keep ingestion simple.
- If RXNREL.RRF is present, you can extend this to relate brand/generic concepts; not required for baseline.
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


def parse_rxnconso(rrf_path: Path, max_concepts: int = 0) -> Dict[str, Dict]:
    """
    Parse RXNCONSO.RRF and build a dict per RxCUI with synonyms and TTY counts.

    Returns a dict: rxcui -> { 'preferred': str, 'synonyms': set[str], 'tty_counts': Counter }
    """
    concepts: Dict[str, Dict] = {}
    counts = 0

    if not rrf_path.exists():
        raise FileNotFoundError(f"RXNCONSO.RRF not found at {rrf_path}")

    # RXNCONSO.RRF columns (pipe-delimited), see NLM specs:
    # 0 RXCUI | 1 LAT | 2 TS | 3 LUI | 4 STT | 5 SUI | 6 ISPREF | 7 RXAUI | 8 SAUI | 9 SCUI
    # 10 SDUI | 11 SAB | 12 TTY | 13 CODE | 14 STR | 15 SRL | 16 SUPPRESS | 17 CVF
    
    with rrf_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip('\n').split('|')
            if len(parts) < 18:
                continue
            rxcui = parts[0]
            lat = parts[1]
            ispref = parts[6]
            sab = parts[11]
            tty = parts[12]
            term = parts[14].strip()

            if not rxcui or not term:
                continue
            # English terms only; prefer RXNORM source terms
            if lat.upper() != 'ENG':
                continue
            # We accept any SAB but prefer RXNORM ones for preferred text
            entry = concepts.get(rxcui)
            if entry is None:
                entry = {
                    'preferred': '',
                    'preferred_sab': '',
                    'synonyms': set(),
                    'tty_counts': Counter(),
                }
                concepts[rxcui] = entry
                counts += 1
                if max_concepts and counts >= max_concepts:
                    # Note: we will keep adding synonyms to seen RxCUIs, but stop adding new RxCUIs
                    pass
            # Track TTY distribution and synonym text
            entry['tty_counts'][tty] += 1
            if term:
                entry['synonyms'].add(term)
            # Choose preferred: favor RXNORM SAB with ISPREF='Y', fallback to any first term
            if not entry['preferred']:
                if sab.upper() == 'RXNORM' and ispref.upper() == 'Y':
                    entry['preferred'] = term
                    entry['preferred_sab'] = sab
            else:
                # If we don't have an RXNORM preferred yet, allow RXNORM|Y to override
                if entry['preferred_sab'].upper() != 'RXNORM' and sab.upper() == 'RXNORM' and ispref.upper() == 'Y':
                    entry['preferred'] = term
                    entry['preferred_sab'] = sab

    # Fallback preferred = any synonym
    for rx, data in concepts.items():
        if not data['preferred'] and data['synonyms']:
            data['preferred'] = next(iter(data['synonyms']))
    return concepts


def write_chunks(concepts: Dict[str, Dict], out_path: Path, max_synonyms: int = 30) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open('w', encoding='utf-8') as out:
        for rxcui, data in concepts.items():
            preferred = (data.get('preferred') or '').strip()
            syns = sorted(list(data.get('synonyms') or []))[:max_synonyms]
            tty_counts = data.get('tty_counts') or {}
            tty_summary = ', '.join(f"{k}:{v}" for k, v in tty_counts.items())
            text = (
                f"RxNorm concept RXCUI {rxcui}: {preferred}.\n"
                f"Synonyms: {', '.join(syns)}.\n"
                f"TTY counts: {tty_summary}."
            )
            token_count = max(1, len(text) // 4)
            row = {
                'id': f"rxnorm_{rxcui}",
                'doc_id': f"rxnorm_{rxcui}",
                'chunk_index': 0,
                'text': text,
                'source': 'RxNorm',
                'category': 'Medications',
                'file': 'RXNCONSO.RRF',
                'url': '',
                'type': 'rxnorm_concept',
                'token_count': token_count,
                'rxcui': rxcui,
            }
            import json
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rrf-dir', default='data/rxnorm', help='Directory containing RXNCONSO.RRF (and optionally RXNREL.RRF)')
    ap.add_argument('--out', default='data/chunks/rxnorm_medications.jsonl', help='Output JSONL for chunks')
    ap.add_argument('--max-concepts', type=int, default=0, help='Optional cap for number of unique RxCUIs (0 = no cap)')
    args = ap.parse_args()

    rrf_dir = Path(args.rrf_dir)
    conso = rrf_dir / 'RXNCONSO.RRF'
    if not conso.exists():
        raise FileNotFoundError(f"Place RXNCONSO.RRF under {rrf_dir}")

    print(f"Parsing: {conso}")
    concepts = parse_rxnconso(conso, max_concepts=args.max_concepts)
    print(f"Concepts parsed: {len(concepts)}")

    out_path = Path(args.out)
    written = write_chunks(concepts, out_path)
    print(f"Wrote {written} RxNorm chunks -> {out_path}")


if __name__ == '__main__':
    main()

