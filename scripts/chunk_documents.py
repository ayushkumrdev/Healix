#!/usr/bin/env python3
"""
Document chunking script for Healthcare Super-Assistant
Processes MedQuAD XML files and converts them to chunked JSONL format
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from typing import List, Dict
from transformers import AutoTokenizer

# Initialize tokenizer for counting tokens
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def count_tokens(text: str) -> int:
    """Count tokens in text using the same tokenizer as embeddings"""
    try:
        # Truncate very long texts to avoid tokenizer issues
        if len(text) > 2000:  # rough char limit
            text = text[:2000]
        return len(tokenizer.encode(text, truncation=True, max_length=512))
    except Exception:
        # Fallback: estimate tokens as roughly 4 characters per token
        return len(text) // 4

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove XML artifacts
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.,;:!?()\-\'"]', '', text)
    
    return text

def chunk_text(text: str, max_tokens: int = 250, overlap_tokens: int = 50) -> List[str]:
    """
    Split text into overlapping chunks based on token count
    """
    if not text:
        return []
    
    words = text.split()
    if not words:
        return []
    
    # If text is short enough, return as single chunk
    if count_tokens(text) <= max_tokens:
        return [text]
    
    chunks = []
    current_chunk = []
    
    for word in words:
        test_chunk = current_chunk + [word]
        test_text = " ".join(test_chunk)
        
        if count_tokens(test_text) > max_tokens:
            if current_chunk:  # Save current chunk if not empty
                chunks.append(" ".join(current_chunk))
                
                # Create overlap for next chunk
                overlap_words = current_chunk[-overlap_tokens:] if len(current_chunk) > overlap_tokens else current_chunk
                current_chunk = overlap_words + [word]
            else:
                # Single word is too long, add it anyway
                current_chunk = [word]
        else:
            current_chunk = test_chunk
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def parse_medquad_xml(xml_path: str) -> List[Dict]:
    """Parse MedQuAD XML file and extract QA pairs"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        qa_pairs = []
        
        # Find all QA pairs in the XML
        for qa in root.findall('.//QAPair'):
            question_elem = qa.find('Question')
            answer_elem = qa.find('Answer')
            
            if question_elem is not None and answer_elem is not None:
                question = clean_text(question_elem.text)
                answer = clean_text(answer_elem.text)
                
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'file': os.path.basename(xml_path)
                    })
        
        return qa_pairs
        
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error processing {xml_path}: {e}")
        return []

def process_medquad_directory(medquad_dir: str) -> List[Dict]:
    """Process all MedQuAD directories and extract documents"""
    documents = []
    medquad_path = Path(medquad_dir)
    
    if not medquad_path.exists():
        print(f"MedQuAD directory not found: {medquad_dir}")
        return documents
    
    # Process all subdirectories
    for subdir in medquad_path.iterdir():
        if subdir.is_dir() and subdir.name.endswith('_QA'):
            print(f"Processing directory: {subdir.name}")
            
            xml_count = 0
            for xml_file in subdir.glob('*.xml'):
                qa_pairs = parse_medquad_xml(str(xml_file))
                
                for qa_pair in qa_pairs:
                    # Create a combined document from Q&A
                    combined_text = f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}"
                    
                    documents.append({
                        'text': combined_text,
                        'source': 'MedQuAD',
                        'category': subdir.name,
                        'file': qa_pair['file'],
                        'url': f"https://github.com/abachaa/MedQuAD/tree/master/{subdir.name}/{qa_pair['file']}",
                        'type': 'QA_pair',
                        'question': qa_pair['question'],
                        'answer': qa_pair['answer']
                    })
                
                xml_count += 1
                if xml_count % 100 == 0:
                    print(f"  Processed {xml_count} XML files from {subdir.name}")
            
            print(f"  Total: {xml_count} XML files from {subdir.name}")
    
    return documents

def chunk_documents(documents: List[Dict], output_dir: str):
    """Chunk documents and save as JSONL"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "all_chunks.jsonl")
    
    chunk_id = 0
    total_chunks = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_idx, doc in enumerate(documents):
            text = doc['text']
            
            # Chunk the document text
            chunks = chunk_text(text, max_tokens=250)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_data = {
                    'id': f"chunk_{chunk_id:06d}",
                    'doc_id': f"doc_{doc_idx:06d}",
                    'chunk_index': chunk_idx,
                    'text': chunk,
                    'source': doc['source'],
                    'category': doc['category'],
                    'file': doc['file'],
                    'url': doc['url'],
                    'type': doc['type'],
                    'token_count': count_tokens(chunk)
                }
                
                # Include original question/answer for QA pairs
                if doc['type'] == 'QA_pair':
                    chunk_data['question'] = doc['question']
                    chunk_data['answer'] = doc['answer']
                
                f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
                chunk_id += 1
            
            total_chunks += len(chunks)
            
            if (doc_idx + 1) % 1000 == 0:
                print(f"Processed {doc_idx + 1} documents, created {total_chunks} chunks")
    
    print(f"\nCompleted chunking:")
    print(f"  - Total documents: {len(documents)}")
    print(f"  - Total chunks: {total_chunks}")
    print(f"  - Output file: {output_file}")

def main():
    """Main function to process MedQuAD data"""
    # Paths
    base_dir = Path(__file__).parent.parent
    medquad_dir = base_dir / "data" / "sources" / "MedQuAD"
    output_dir = base_dir / "data" / "chunks"
    
    print("Healthcare Super-Assistant - Document Chunking")
    print("=" * 50)
    print(f"MedQuAD directory: {medquad_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process MedQuAD documents
    print("Step 1: Processing MedQuAD documents...")
    documents = process_medquad_directory(str(medquad_dir))
    
    if not documents:
        print("No documents found. Please check the MedQuAD directory.")
        return
    
    print(f"Found {len(documents)} documents")
    print()
    
    # Chunk documents
    print("Step 2: Chunking documents...")
    chunk_documents(documents, str(output_dir))
    
    print("\nDocument chunking completed successfully!")

if __name__ == "__main__":
    main()
