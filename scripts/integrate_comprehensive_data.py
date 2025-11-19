#!/usr/bin/env python3
"""
Integrate comprehensive medical data into existing FAISS index
"""

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def integrate_comprehensive_data():
    """Integrate comprehensive medical data into FAISS index"""
    
    print("🔄 Integrating comprehensive medical data into FAISS index...")
    
    # Load existing FAISS index and data
    data_dir = Path("data")
    index_path = data_dir / "index.faiss"
    chunks_path = data_dir / "indexed_chunks.pkl" 
    metadata_path = data_dir / "faiss_metadata.pkl"
    
    if not all(p.exists() for p in [index_path, chunks_path, metadata_path]):
        print("❌ Existing FAISS index not found. Please build the base index first.")
        return
    
    print("📂 Loading existing FAISS index...")
    index = faiss.read_index(str(index_path))
    
    with open(chunks_path, 'rb') as f:
        existing_chunks = pickle.load(f)
    
    with open(metadata_path, 'rb') as f:
        existing_metadata = pickle.load(f)
    
    print(f"📊 Existing index: {len(existing_chunks)} chunks")
    
    # Load comprehensive medical data
    enhanced_dir = Path("data/enhanced_comprehensive")
    all_new_data = []
    
    # Load all databases
    databases = [
        "drug_database.json",
        "diagnostic_database.json", 
        "treatment_database.json",
        "symptom_database.json",
        "interactions_database.json"
    ]
    
    for db_file in databases:
        db_path = enhanced_dir / db_file
        if db_path.exists():
            print(f"📋 Loading {db_file}...")
            with open(db_path, 'r') as f:
                data = json.load(f)
                all_new_data.extend(data)
        else:
            print(f"⚠️ {db_file} not found, skipping...")
    
    print(f"📊 New medical data: {len(all_new_data)} entries")
    
    if not all_new_data:
        print("❌ No new data to integrate")
        return
    
    # Initialize sentence transformer (same model as existing)
    print("🤖 Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Process new data into chunks
    print("🔄 Processing new data...")
    new_chunks = []
    new_metadata = []
    
    for i, item in enumerate(all_new_data):
        chunk_id = f"enhanced_{i:06d}"
        
        chunk = {
            "id": chunk_id,
            "text": item["text"],
            "source": item["source"],
            "category": item["category"],
            "url": "N/A",
            "type": item["type"],
            "token_count": len(item["text"].split())
        }
        
        # Add drug-specific fields if available
        if "drug_name" in item:
            chunk["drug_name"] = item["drug_name"]
            chunk["drug_category"] = item["drug_category"]
        
        # Add condition-specific fields if available
        if "condition_name" in item:
            chunk["condition_name"] = item["condition_name"]
        
        new_chunks.append(chunk)
        
        # Create metadata entry
        metadata = {
            "chunk_id": chunk_id,
            "source": item["source"],
            "category": item["category"],
            "type": item["type"]
        }
        
        new_metadata.append(metadata)
    
    # Generate embeddings for new data
    print("🧠 Generating embeddings for new data...")
    new_texts = [chunk["text"] for chunk in new_chunks]
    new_embeddings = model.encode(new_texts, convert_to_numpy=True, show_progress_bar=True)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(new_embeddings.astype(np.float32))
    
    # Add to existing index
    print("📚 Adding to FAISS index...")
    index.add(new_embeddings.astype(np.float32))
    
    # Combine with existing data
    combined_chunks = existing_chunks + new_chunks
    combined_metadata = existing_metadata + new_metadata
    
    # Save updated index and data
    print("💾 Saving updated index...")
    faiss.write_index(index, str(index_path))
    
    with open(chunks_path, 'wb') as f:
        pickle.dump(combined_chunks, f)
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(combined_metadata, f)
    
    # Update index config
    config = {
        "total_chunks": len(combined_chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "index_type": "IndexFlatIP",
        "original_chunks": len(existing_chunks),
        "enhanced_chunks": len(new_chunks),
        "categories": list(set([chunk["category"] for chunk in combined_chunks])),
        "last_updated": "2025-01-07"
    }
    
    with open(data_dir / "index_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Integration complete!")
    print(f"📊 Total chunks: {len(combined_chunks)}")
    print(f"🆕 Added: {len(new_chunks)} new medical entries")
    print(f"🗂️ Categories: {len(config['categories'])}")
    
    # Test the updated index
    print("\n🧪 Testing updated index...")
    test_queries = [
        "headache medication dosage",
        "high blood pressure treatment", 
        "chest pain diagnosis",
        "drug interactions"
    ]
    
    for query in test_queries:
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding.astype(np.float32), 3)
        
        print(f"\n🔍 Query: '{query}'")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(combined_chunks):
                chunk = combined_chunks[idx]
                print(f"  {i+1}. ({score:.3f}) {chunk['category']}: {chunk['text'][:100]}...")

if __name__ == "__main__":
    integrate_comprehensive_data()
