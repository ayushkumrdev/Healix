#!/usr/bin/env python3
"""
Sparse BM25 retriever + Reciprocal Rank Fusion for Healix hybrid RAG.

Dense embedding search (FAISS) is strong on meaning but blurs exact tokens
(drug names, gene symbols, rare conditions). A classic BM25 lexical index nails
those. Fusing both with Reciprocal Rank Fusion (RRF) gives consistently better
recall than either alone — the core of modern "hybrid" retrieval — at negligible
cost (BM25 is CPU-cheap and computed over a sparse matrix).

BM25 is implemented on top of scikit-learn's CountVectorizer sparse matrix, so it
adds no heavy dependency and stays memory-efficient. The index builds once from
the existing chunk texts and is cached to disk.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class SparseBM25:
    """Memory-efficient BM25 over a scikit-learn sparse term-frequency matrix."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.tf_csc = None          # docs x vocab, term counts (CSC for column access)
        self.idf: Optional[np.ndarray] = None
        self.doc_len: Optional[np.ndarray] = None
        self.avgdl: float = 0.0

    def fit(self, docs: List[str]) -> "SparseBM25":
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(lowercase=True, stop_words="english",
                              max_features=200_000, dtype=np.float32)
        tf = vec.fit_transform(docs)                       # CSR docs x vocab
        n_docs = tf.shape[0]
        df = np.asarray((tf > 0).sum(axis=0)).ravel()      # doc frequency per term
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.doc_len = np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = float(self.doc_len.mean()) or 1.0
        self.vectorizer = vec
        self.tf_csc = tf.tocsc()
        return self

    def query(self, text: str, top_n: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectorizer is None:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        qv = self.vectorizer.transform([text or ""])
        terms = qv.indices
        if terms.size == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        scores = np.zeros(self.tf_csc.shape[0], dtype=np.float32)
        denom = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        for t in terms:
            col = self.tf_csc.getcol(t)
            idx = col.indices
            tf = col.data.astype(np.float32)
            scores[idx] += self.idf[t] * (tf * (self.k1 + 1.0)) / (tf + denom[idx])
        nz = np.flatnonzero(scores)
        if nz.size == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        order = nz[np.argsort(scores[nz])[::-1][:top_n]]
        return order, scores[order]

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"k1": self.k1, "b": self.b, "vectorizer": self.vectorizer,
                         "tf_csc": self.tf_csc, "idf": self.idf,
                         "doc_len": self.doc_len, "avgdl": self.avgdl}, f)

    @classmethod
    def load(cls, path: str) -> "SparseBM25":
        with open(path, "rb") as f:
            d = pickle.load(f)
        o = cls(d["k1"], d["b"])
        o.vectorizer = d["vectorizer"]; o.tf_csc = d["tf_csc"]
        o.idf = d["idf"]; o.doc_len = d["doc_len"]; o.avgdl = d["avgdl"]
        return o


def build_or_load_bm25(docs: List[str], cache_path: str) -> SparseBM25:
    p = Path(cache_path)
    if p.exists():
        try:
            bm = SparseBM25.load(str(p))
            if bm.tf_csc is not None and bm.tf_csc.shape[0] == len(docs):
                return bm
        except Exception:
            pass
    t0 = time.time()
    print(f"[hybrid] building BM25 index over {len(docs)} chunks...")
    bm = SparseBM25().fit(docs)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        bm.save(str(p))
    except Exception as e:
        print(f"[hybrid] could not cache BM25 index: {e}")
    print(f"[hybrid] BM25 ready in {time.time() - t0:.1f}s")
    return bm


def reciprocal_rank_fusion(ranked_lists: List[List[int]], k: int = 60) -> List[int]:
    """Fuse multiple ranked id lists with RRF; returns ids sorted by fused score."""
    scores: dict = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
