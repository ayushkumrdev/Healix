#!/usr/bin/env python3
"""
Shrink the Healix FAISS index without re-embedding.

The original index is an IndexFlatIP (float32, ~703 MB for 239k x 768).
Because the source chunk JSONL files are no longer on disk, this script
reconstructs the stored vectors directly from the existing flat index and
rebuilds them as an 8-bit scalar-quantized index (SQ8).

SQ8 keeps brute-force (exact-scan) search, so the returned inner-product
scores stay very close to the original cosine scores. That matters because
retriever.py filters on absolute score thresholds (min_score, rerank skip);
lossy product quantization would shift those scores and break filtering.

Result: ~4x smaller index, ~99% top-k recall, no retriever code changes.

Usage:
  python scripts/shrink_faiss_index.py                # build + validate only
  python scripts/shrink_faiss_index.py --apply        # also swap into place
  python scripts/shrink_faiss_index.py --apply --min-recall 0.97
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import faiss


def build_sq8(xb: np.ndarray, dim: int) -> faiss.Index:
    index = faiss.IndexScalarQuantizer(
        dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    print(f"Training SQ8 quantizer on {xb.shape[0]} vectors...")
    index.train(xb)
    print("Adding vectors to SQ8 index...")
    index.add(xb)
    return index


def validate(flat: faiss.Index, sq: faiss.Index, xb: np.ndarray,
             k: int = 10, sample: int = 500, seed: int = 0) -> dict:
    """Use a random sample of DB vectors as queries; compare neighbor sets."""
    rng = np.random.default_rng(seed)
    n = xb.shape[0]
    qidx = rng.choice(n, size=min(sample, n), replace=False)
    q = xb[qidx]

    _, flat_nn = flat.search(q, k)
    sq_scores, sq_nn = sq.search(q, k)
    flat_scores, _ = flat.search(q, k)

    overlaps = [
        len(set(flat_nn[i]) & set(sq_nn[i])) / k
        for i in range(len(qidx))
    ]
    recall = float(np.mean(overlaps))
    # Mean absolute score error on the rank-matched top hits
    score_err = float(np.mean(np.abs(flat_scores - sq_scores)))
    return {"recall_at_k": recall, "k": k, "samples": len(qidx),
            "mean_abs_score_err": score_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--apply", action="store_true", help="swap the new index into place")
    ap.add_argument("--min-recall", type=float, default=0.95,
                    help="minimum top-k recall required to --apply")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    src = data_dir / "index.faiss"
    candidate = data_dir / "index.faiss.sq8"
    backup = data_dir / "index.faiss.flatbak"
    config_path = data_dir / "index_config.json"

    if not src.exists():
        raise SystemExit(f"Index not found: {src}")

    src_mb = src.stat().st_size / 1e6
    print(f"Loading flat index: {src} ({src_mb:.1f} MB)")
    flat = faiss.read_index(str(src))  # full read (need reconstruct)
    n, d = flat.ntotal, flat.d
    print(f"Vectors: {n}, dim: {d}")

    print("Reconstructing vectors from flat index...")
    xb = flat.reconstruct_n(0, n).astype(np.float32)

    sq = build_sq8(xb, d)

    print("Validating SQ8 against flat (neighbor recall)...")
    metrics = validate(flat, sq, xb, k=args.k)
    print(f"  recall@{metrics['k']}: {metrics['recall_at_k']:.4f} "
          f"({metrics['samples']} sample queries)")
    print(f"  mean |score| error: {metrics['mean_abs_score_err']:.5f}")

    faiss.write_index(sq, str(candidate))
    cand_mb = candidate.stat().st_size / 1e6
    print(f"Wrote candidate: {candidate} ({cand_mb:.1f} MB, "
          f"{src_mb / cand_mb:.1f}x smaller)")

    if not args.apply:
        print("\nDry run complete. Re-run with --apply to swap into place.")
        return

    if metrics["recall_at_k"] < args.min_recall:
        raise SystemExit(
            f"Recall {metrics['recall_at_k']:.4f} below --min-recall "
            f"{args.min_recall}; NOT applying. Candidate left at {candidate}."
        )

    if backup.exists():
        raise SystemExit(f"Backup already exists: {backup}. Remove it first.")
    print(f"Backing up original -> {backup}")
    shutil.move(str(src), str(backup))
    print(f"Installing SQ8 index -> {src}")
    shutil.move(str(candidate), str(src))

    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg["index_type"] = "IndexScalarQuantizer(QT_8bit, METRIC_INNER_PRODUCT)"
        cfg["quantization"] = "sq8"
        cfg["shrunk_from"] = "IndexFlatIP"
        cfg["recall_at_10"] = round(metrics["recall_at_k"], 4)
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"Updated {config_path}")

    print(f"\nDone. Original preserved at {backup} "
          f"(safe to delete once verified).")


if __name__ == "__main__":
    main()
