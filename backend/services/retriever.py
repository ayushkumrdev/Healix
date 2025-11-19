#!/usr/bin/env python3
"""
Retrieval service for Healthcare Super-Assistant
Provides fast, relevant document retrieval using FAISS index with provenance
Optionally supports cross-encoder reranking for higher evidence quality.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
# Load .env for direct Python runs (optional)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    # Optional cross-encoder for reranking
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None  # type: ignore


class MedicalRetriever:
    """Fast medical document retriever with FAISS index and optional reranking"""
    
    def __init__(self, index_dir: str, model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize the retriever
        
        Args:
            index_dir: Directory containing FAISS index and metadata files
            model_name: Name of the sentence transformer model to use
        """
        self.index_dir = Path(index_dir)
        # Allow environment override for embedding model
        self.model_name = os.getenv("BAYMAX_EMBEDDING_MODEL", model_name)
        self.model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        # Optional exclusion of categories (comma-separated) via env var
        exclude = os.getenv("BAYMAX_RETRIEVER_EXCLUDE_CATEGORIES", "").strip()
        self.exclude_categories = set([c.strip() for c in exclude.split(",") if c.strip()])
        
        # Reranker configuration (optional)
        self.reranker_model_name = os.getenv("BAYMAX_RERANKER", "").strip()
        if self.reranker_model_name.lower() in ("none", "0", "false"):
            self.reranker_model_name = ""
        self.reranker_topn = int(os.getenv("BAYMAX_RERANK_TOPN", "50").strip() or 50)
        self.reranker_batch = int(os.getenv("BAYMAX_RERANK_BATCH", "16").strip() or 16)
        self.reranker = None
        # Simple LRU cache for query embeddings to reduce recomputation
        self._embed_cache = {}
        self._embed_cache_order = []
        try:
            self._embed_cache_max = int(os.getenv("BAYMAX_EMBED_CACHE", "256"))
        except Exception:
            self._embed_cache_max = 256
        
        # File paths
        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "indexed_chunks.pkl"
        self.metadata_path = self.index_dir / "faiss_metadata.pkl"
        
        # Load index immediately; defer model/reranker until needed to speed startup
        self._load_index()
        # self._load_model()  # Lazy-load on first retrieve
        # self._load_reranker()  # Only if enabled and when needed
        
    def _load_model(self):
        """Load the sentence transformer model"""
        print(f"Loading embedding model: {self.model_name}")
        # Prefer GPU if available for faster query encoding
        device = "cpu"
        try:
            import torch  # Local import to avoid hard dependency if not installed
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"
        self.model = SentenceTransformer(self.model_name, device=device)
        
    def _load_reranker(self):
        """Optionally load cross-encoder reranker"""
        if not self.reranker_model_name:
            return
        if CrossEncoder is None:
            print("Reranker requested but sentence_transformers.CrossEncoder not available; skipping rerank.")
            return
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"
        try:
            print(f"Loading reranker model: {self.reranker_model_name}")
            self.reranker = CrossEncoder(self.reranker_model_name, device=device)
            print("Reranker loaded successfully")
        except Exception as e:
            print(f"Failed to load reranker '{self.reranker_model_name}': {e}")
            self.reranker = None
        
    def _load_index(self):
        """Load FAISS index and associated data"""
        
        # Check if all required files exist
        required_files = [self.index_path, self.chunks_path, self.metadata_path]
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            missing_str = ", ".join(str(f) for f in missing_files)
            raise FileNotFoundError(f"Missing required files: {missing_str}")
        
        # Load FAISS index
        print(f"Loading FAISS index from: {self.index_path}")
        # Prefer memory-mapped loading to reduce RAM and speed startup
        try:
            self.index = faiss.read_index(str(self.index_path), faiss.IO_FLAG_MMAP)
            print("FAISS index loaded with memory-mapping (IO_FLAG_MMAP)")
        except TypeError:
            # Older faiss versions may not support the flag
            self.index = faiss.read_index(str(self.index_path))
        print(f"Index loaded: {self.index.ntotal} vectors, dimension {self.index.d}")
        
        # Optional: move index to GPU if available and requested
        try:
            use_gpu = str(os.getenv("BAYMAX_FAISS_GPU", "0")).lower() in ("1", "true", "yes")
            device_id = int(os.getenv("BAYMAX_CUDA_DEVICE", "0"))
            if use_gpu and hasattr(faiss, "StandardGpuResources"):
                print(f"Moving FAISS index to GPU:{device_id} ...")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, device_id, self.index)
                self._faiss_gpu_resources = res  # keep a ref
                print(f"FAISS index now on GPU:{device_id}")
            else:
                if use_gpu:
                    print("FAISS GPU requested but not available in this build; using CPU index")
        except Exception as e:
            print(f"FAISS GPU offload failed: {e}. Continuing with CPU index.")
        
        # Load chunks (with full text)
        print(f"Loading indexed chunks from: {self.chunks_path}")
        with open(self.chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Loaded {len(self.chunks)} chunks")
        
        # Load metadata (lighter version without text)
        print(f"Loading metadata from: {self.metadata_path}")
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"Loaded {len(self.metadata)} metadata entries")
        
        # Verify consistency
        if len(self.chunks) != len(self.metadata) or len(self.chunks) != self.index.ntotal:
            raise ValueError("Mismatch between index, chunks, and metadata sizes")
            
    def retrieve(self, query: str, k: int = 10, min_score: float = 0.0) -> List[Dict]:
        """
        Retrieve top-k most relevant passages for a query
        
        Args:
            query: Search query string
            k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved passages with metadata
        """
        if not query.strip():
            return []
            
        # Encode query (with small LRU cache)
        key = query.strip()
        if key and key in getattr(self, "_embed_cache", {}):
            query_embedding = self._embed_cache[key]
        else:
            # Lazy-load embedding model on first use
            if self.model is None:
                self._load_model()
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity
            try:
                if key:
                    self._embed_cache[key] = query_embedding
                    self._embed_cache_order.append(key)
                    if len(self._embed_cache_order) > getattr(self, "_embed_cache_max", 256):
                        old = self._embed_cache_order.pop(0)
                        self._embed_cache.pop(old, None)
            except Exception:
                pass
        
        # Lazy-load reranker if configured
        if self.reranker is None and self.reranker_model_name:
            try:
                self._load_reranker()
            except Exception:
                self.reranker = None

        # Decide pre-retrieval size if reranker is enabled
        k_pre = max(k, self.reranker_topn) if self.reranker is not None else k
        
        # Search index
        scores, indices = self.index.search(query_embedding.astype(np.float32), k_pre)
        
        # Prepare results
        prelim_results = []
        for score, idx in zip(scores[0], indices[0]):
            # Skip invalid indices or scores below threshold
            if idx == -1 or score < min_score:
                continue
                
            if idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx]
            meta = self.metadata[idx]
            
            # Apply optional category exclusion
            if self.exclude_categories and str(chunk.get('category', '')).strip() in self.exclude_categories:
                continue

            result = {
                'score': float(score),  # embedding similarity (cosine)
                'chunk_id': chunk['id'],
                'text': chunk['text'],
                'source': chunk['source'],
                'category': chunk['category'],
                'url': chunk['url'],
                'type': chunk['type'],
                'token_count': chunk['token_count'],
                'metadata': meta
            }
            # Include RxNorm identifiers when present
            if 'rxcui' in chunk:
                result['rxcui'] = chunk.get('rxcui')
            
            # Include original Q&A for QA pairs
            if chunk['type'] == 'QA_pair':
                result['question'] = chunk['question']
                result['answer'] = chunk['answer']
                
            prelim_results.append(result)
        
        # Optional reranking with cross-encoder
        if self.reranker is not None and prelim_results:
            try:
                # Skip reranking if the top embedding match is already very strong
                skip_thr = float(os.getenv("BAYMAX_RERANK_SKIP_THRESHOLD", "0.86"))
                if prelim_results and prelim_results[0].get('score', 0.0) >= skip_thr:
                    return prelim_results[:k]
                pairs = [[query, r['text']] for r in prelim_results]
                rerank_scores = self.reranker.predict(pairs, batch_size=self.reranker_batch, show_progress_bar=False)
                for r, s in zip(prelim_results, rerank_scores):
                    r['rerank'] = float(s)
                prelim_results.sort(key=lambda x: x.get('rerank', 0.0), reverse=True)
            except Exception as e:
                print(f"Reranking failed: {e}. Proceeding with embedding scores.")
        
        # Return top-k
        return prelim_results[:k]

    def search_by_category(self, query: str, category: str, k: int = 2, min_score: float = 0.0) -> List[Dict]:
        """Retrieve top-k passages restricted to a specific category (exact match)."""
        if not query.strip():
            return []
        # Retrieve a slightly larger pool, then filter by category. No fallback expansion.
        pool_k = max(k * 5, k)
        results = self.retrieve(query=query, k=pool_k, min_score=min_score)
        filtered = [r for r in results if str(r.get('category', '')).strip() == str(category).strip()]
        return filtered[:k]
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        if not self.chunks:
            return {}
            
        # Count by category
        categories = {}
        types = {}
        
        for chunk in self.chunks:
            cat = chunk.get('category', 'unknown')
            typ = chunk.get('type', 'unknown')
            
            categories[cat] = categories.get(cat, 0) + 1
            types[typ] = types.get(typ, 0) + 1
            
        return {
            'total_chunks': len(self.chunks),
            'index_vectors': self.index.ntotal,
            'embedding_dimension': self.index.d,
            'categories': categories,
            'types': types,
            'model_name': self.model_name
        }
    
    def health_check(self) -> Dict:
        """Check if retriever is working correctly"""
        try:
            # Test with a simple query
            results = self.retrieve("fever", k=1)
            
            return {
                'status': 'healthy',
                'index_loaded': self.index is not None,
                'model_loaded': self.model is not None,
                'chunks_loaded': len(self.chunks) if self.chunks else 0,
                'test_query_results': len(results)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


def create_retriever(base_dir: Optional[str] = None) -> MedicalRetriever:
    """
    Convenience function to create a retriever with default paths
    
    Args:
        base_dir: Base directory containing data folder, defaults to project root
        
    Returns:
        Configured MedicalRetriever instance
    """
    if base_dir is None:
        # Default to project root
        current_file = Path(__file__)
        base_dir = current_file.parent.parent.parent
        
    index_dir = Path(base_dir) / "data"
    
    return MedicalRetriever(str(index_dir))


# Example usage and testing
if __name__ == "__main__":
    # Test the retriever
    print("Healthcare Super-Assistant - Retrieval Service Test")
    print("=" * 55)
    
    try:
        # Create retriever
        retriever = create_retriever()
        
        # Show stats
        stats = retriever.get_stats()
        print("\\nRetriever Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Categories: {len(stats['categories'])}")
        print(f"  Document types: {stats['types']}")
        
        # Test queries
        test_queries = [
            "What are the symptoms of diabetes?",
            "How to treat high blood pressure?",
            "Side effects of chemotherapy",
            "Heart failure symptoms"
        ]
        
        for query in test_queries:
            print(f"\\nQuery: '{query}'")
            results = retriever.retrieve(query, k=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.4f}")
                print(f"     Source: {result['source']} - {result['category']}")
                text_preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
                print(f"     Text: {text_preview}")
                print()
        
        # Health check
        health = retriever.health_check()
        print(f"Health check: {health['status']}")
        
    except Exception as e:
        print(f"Error testing retriever: {e}")
        import traceback
        traceback.print_exc()
