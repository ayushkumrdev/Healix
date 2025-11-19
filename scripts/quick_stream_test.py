#!/usr/bin/env python3
import os, sys, time
sys.path.append('backend')
os.environ.setdefault('BAYMAX_USE_LLAMA_CPP','1')
os.environ.setdefault('BAYMAX_GPU_LAYERS','99999')
from services.advanced_orchestrator import AdvancedMedicalOrchestrator
from services.retriever import MedicalRetriever

def main():
    orch = AdvancedMedicalOrchestrator()
    ret = MedicalRetriever(index_dir='data')
    q = 'What is aspirin used for?'
    passages = ret.retrieve(q, k=3, min_score=0.2)
    full, qt, specs = orch.prepare_stream_prompt(q, passages, None, None, 'patient')
    buf=''
    t0=time.time()
    for tok in orch.stream_generate(full, max_tokens=120, temp=0.2, top_k=20, top_p=0.9):
        buf += tok
    dt=time.time()-t0
    print('len=', len(buf), 'time=', round(dt,3))
    print('snippet=', buf[:200].replace('\n',' '))

if __name__ == '__main__':
    main()
