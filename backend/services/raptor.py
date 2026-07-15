"""RAPTOR summary-tree retrieval (runtime half).

Loads the topic/domain overview nodes built by scripts/build_raptor.py and
searches them with the *broad* HyDE hypothetical (see hyde.py) — abstraction-
matched retrieval: detail questions hit the leaf chunk index, broad questions
additionally get topic-level overview evidence that no single leaf contains.

Fail-open: if the artifacts are missing the index reports unavailable and the
pipeline behaves exactly as before.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"


def raptor_enabled() -> bool:
    return str(os.getenv("HEALIX_RAPTOR", "1")).lower() in ("1", "true", "yes")


class RaptorIndex:
    def __init__(self, data_dir: Optional[Path] = None):
        self.available = False
        self._index = None
        self._nodes: List[Dict] = []
        d = Path(data_dir) if data_dir else DATA
        try:
            import faiss
            idx_path, nodes_path = d / "raptor.faiss", d / "raptor_nodes.pkl"
            if idx_path.exists() and nodes_path.exists():
                self._index = faiss.read_index(str(idx_path))
                with open(nodes_path, "rb") as f:
                    self._nodes = pickle.load(f)
                self.available = self._index.ntotal == len(self._nodes) > 0
        except Exception as e:
            print(f"RAPTOR index unavailable: {e}")

    def search(self, text: str, retriever, k: int = 2,
               min_score: float = 0.35) -> List[Dict]:
        """Return top overview nodes shaped like retriever results.

        Uses the retriever's own embedding model so no second encoder loads.
        """
        if not (self.available and text and text.strip()):
            return []
        try:
            import faiss
            import numpy as np
            if retriever.model is None:
                retriever._load_model()
            emb = retriever.model.encode([text], convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(emb)
            scores, idxs = self._index.search(emb, min(k, self._index.ntotal))
            out = []
            for score, i in zip(scores[0], idxs[0]):
                if i == -1 or score < min_score:
                    continue
                n = self._nodes[i]
                out.append({
                    "score": float(score),
                    "chunk_id": n.get("id"),
                    "text": f"Topic overview — {n.get('title', '')}: {n.get('text', '')}",
                    "source": "Healix knowledge tree",
                    "category": f"raptor_level_{n.get('level', 1)}",
                    "url": "", "type": "overview",
                    "token_count": None,
                })
            return out
        except Exception as e:
            print(f"RAPTOR search failed: {e}")
            return []


if __name__ == "__main__":
    import sys
    sys.path.append(str(ROOT / "backend"))
    from services.retriever import MedicalRetriever
    r = MedicalRetriever(index_dir=str(DATA))
    ri = RaptorIndex()
    print("available:", ri.available, "nodes:", len(ri._nodes))
    for q in ("how does stress affect the body overall",
              "what should i know about managing diabetes"):
        print("\nQ:", q)
        for n in ri.search(q, r, k=2):
            print(f"  [{n['score']:.3f}] {n['text'][:160]}")
