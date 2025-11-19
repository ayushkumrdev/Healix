#!/usr/bin/env python3
import os, sys
sys.path.append('backend')
from services.advanced_orchestrator import AdvancedMedicalOrchestrator
from services.retriever import MedicalRetriever

os.environ['BAYMAX_GGUF_MODEL_DIR'] = 'artifacts'
os.environ['BAYMAX_GGUF_MODEL_NAME'] = 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'

retriever = MedicalRetriever(index_dir='data')
orch = AdvancedMedicalOrchestrator()
q='hello'
passages = retriever.retrieve(q, k=1, min_score=0.0)
res = orch.synthesize_advanced_response(user_text=q, retrieved_passages=passages, symptom_data=None, conversation_context=None, user_mode='patient')
print('\nKeys:', list(res.keys()))
print('Emergency:', res.get('emergency'))
resp = res.get('response', {})
print('Assessment len:', len((resp.get('medical_assessment') or '')))
print('Assessment preview:', (resp.get('medical_assessment') or '')[:200])

