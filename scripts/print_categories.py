#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from services.retriever import MedicalRetriever

def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    r = MedicalRetriever(index_dir=os.path.join(base, 'data'))
    stats = r.get_stats()
    cats = stats.get('categories', {})
    print('Total chunks:', stats.get('total_chunks'))
    # Print top 30 categories
    for i, (k, v) in enumerate(cats.items()):
        if i >= 30:
            break
        print(f"{k}: {v}")

if __name__ == '__main__':
    main()

