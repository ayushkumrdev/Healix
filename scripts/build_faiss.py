#!/usr/bin/env python3
"""
FAISS index builder script for Healthcare Super-Assistant
Computes embeddings for chunked documents and builds a searchable FAISS index
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

def load_chunks(chunks_file: str) -> List[Dict]:
    """Load chunked documents from JSONL file"""
    chunks = []
    
    if not os.path.exists(chunks_file):
        print(f"Chunks file not found: {chunks_file}")
        return chunks
    
    print(f"Loading chunks from: {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk_data = json.loads(line.strip())
                chunks.append(chunk_data)
                
                if line_num % 10000 == 0:
                    print(f"  Loaded {line_num} chunks...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Total chunks loaded: {len(chunks)}")
    return chunks

def compute_embeddings(chunks: List[Dict], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32) -> tuple:
    """Compute embeddings for all chunk texts using sentence-transformers"""
    
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Extract text from chunks
    texts = [chunk['text'] for chunk in chunks]
    
    print(f"Computing embeddings for {len(texts)} text chunks with batch_size={batch_size}...")
    print("This may take several minutes depending on your hardware...")
    
    # Compute embeddings with progress bar
    embeddings = model.encode(
        texts, 
        convert_to_numpy=True, 
        show_progress_bar=True,
        batch_size=batch_size  # Adjustable for memory constraints
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, model.get_sentence_embedding_dimension()

def build_faiss_index(embeddings: np.ndarray, dimension: int) -> faiss.Index:
    """Build FAISS index for fast similarity search"""
    
    print(f"Building FAISS index with dimension: {dimension}")
    
    # Normalize embeddings for cosine similarity
    print("Normalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index (Inner Product on normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    
    print(f"Adding {embeddings.shape[0]} embeddings to index...")
    index.add(embeddings.astype(np.float32))
    
    print(f"Index built successfully. Total vectors: {index.ntotal}")
    
    return index

def save_index_and_metadata(index: faiss.Index, chunks: List[Dict], output_dir: str, chunk_files_names: List[str], embed_model_name: str):
    """Save FAISS index and associated metadata"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    index_path = os.path.join(output_dir, "index.faiss")
    print(f"Saving FAISS index to: {index_path}")
    faiss.write_index(index, index_path)
    
    # Prepare metadata (remove text to save space, keep for retrieval lookup)
    metadata = []
    for chunk in chunks:
        meta = {
            'id': chunk.get('id'),
            'doc_id': chunk.get('doc_id', chunk.get('id', '')),
            'chunk_index': chunk.get('chunk_index', 0),
            'source': chunk.get('source', ''),
            'category': chunk.get('category', ''),
            'file': chunk.get('file', ''),
            'url': chunk.get('url', ''),
            'type': chunk.get('type', ''),
            'token_count': chunk.get('token_count', None)
        }
        
        # Include original Q&A for QA pairs when present
        if chunk.get('type') == 'QA_pair':
            if 'question' in chunk:
                meta['question'] = chunk.get('question')
            if 'answer' in chunk:
                meta['answer'] = chunk.get('answer')
        
        metadata.append(meta)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "faiss_metadata.pkl")
    print(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save chunks (with text) for retrieval
    chunks_path = os.path.join(output_dir, "indexed_chunks.pkl")
    print(f"Saving indexed chunks to: {chunks_path}")
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    # Save index configuration
    config = {
        'index_type': 'IndexFlatIP',
        'dimension': index.d,
        'total_vectors': index.ntotal,
        'model_name': embed_model_name,
        'chunks_files': chunk_files_names,
        'created_at': datetime.utcnow().isoformat() + 'Z'
    }
    
    config_path = os.path.join(output_dir, "index_config.json")
    print(f"Saving index configuration to: {config_path}")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("Index and metadata saved successfully!")

def test_index(index: faiss.Index, chunks: List[Dict], sample_queries: List[str], embed_model_name: str):
    """Test the index with sample queries"""
    
    print("\nTesting index with sample queries...")
    
    model = SentenceTransformer(embed_model_name)
    
    for query in sample_queries:
        print(f"\nQuery: '{query}'")
        
        # Compute query embedding
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search index
        k = 3  # Top 3 results
        scores, indices = index.search(query_embedding.astype(np.float32), k)
        
        print("Top results:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(chunks):
                chunk = chunks[idx]
                text_preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                print(f"  {i+1}. Score: {score:.4f}")
                print(f"     Source: {chunk['source']} - {chunk['category']}")
                print(f"     Text: {text_preview}")
                print(f"     URL: {chunk.get('url', 'N/A')}")
                print()

def main():
    """Main function to build FAISS index"""
    
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument('--embed-model', default=os.getenv('BAYMAX_EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))
    ap.add_argument('--batch-size', type=int, default=int(os.getenv('BAYMAX_EMBED_BATCH', '16')))
    args = ap.parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    chunks_dir = base_dir / "data" / "chunks"
    output_dir = base_dir / "data"
    
    print("Healthcare Super-Assistant - FAISS Index Builder")
    print("=" * 55)
    print(f"Chunks directory: {chunks_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Embedding model: {args.embed_model}")
    print()
    
    # Step 1: Load chunks from all *.jsonl in chunks_dir
    if not chunks_dir.exists():
        print(f"Chunks directory not found: {chunks_dir}")
        return
    chunk_files = sorted([p for p in chunks_dir.glob('*.jsonl') if p.is_file()])
    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}")
        return
    print("Discovered chunk files:")
    for cf in chunk_files:
        print(f" - {cf.name}")
    chunks = []
    for cf in chunk_files:
        loaded = load_chunks(str(cf))
        if loaded:
            print(f"Merging {len(loaded)} from {cf.name}")
            chunks.extend(loaded)
    if not chunks:
        print("No chunks loaded. Please run the ingestion scripts first.")
        return
    
    print()
    
    # Step 2: Compute embeddings
    embeddings, dimension = compute_embeddings(chunks, model_name=args.embed_model, batch_size=args.batch_size)
    print()
    
    # Step 3: Build FAISS index
    index = build_faiss_index(embeddings, dimension)
    print()
    
    # Step 4: Save index and metadata
    save_index_and_metadata(index, chunks, str(output_dir), [p.name for p in chunk_files], args.embed_model)
    print()
    
    # Step 5: Test index with sample queries
    sample_queries = [
        "What are the symptoms of diabetes?",
        "How is cancer treated?",
        "Side effects of chemotherapy",
        "Signs of heart disease",
        "Treatment for high blood pressure"
    ]
    
    test_index(index, chunks, sample_queries, args.embed_model)
    
    print("\nFAISS index building completed successfully!")
    print("\nFiles created:")
    print(f"  - {output_dir}/index.faiss")
    print(f"  - {output_dir}/faiss_metadata.pkl") 
    print(f"  - {output_dir}/indexed_chunks.pkl")
    print(f"  - {output_dir}/index_config.json")

if __name__ == "__main__":
    main()
