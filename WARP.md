# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Healix (Baymax.v1)** is a local-first medical AI assistant providing ChatGPT-level healthcare information. It uses:
- **LLM**: Mistral-7B-Instruct (Q4_K_M quantized, 4.4GB) via GPT4All
- **Retrieval**: FAISS vector search over 239,644 medical document chunks
- **NER**: Clinical BERT for symptom extraction
- **Stack**: Python, Streamlit, sentence-transformers
- **Privacy**: 100% local processing, no external API calls

## Common Commands

### Development & Testing

```powershell
# Activate virtual environment
.\.venv\Scripts\activate

# Start the Streamlit UI (uses professional UI by default)
.\start_system.bat

# Run CLI demo interface
.\.venv\Scripts\python.exe demo_chat.py

# End-to-end smoke test
.\.venv\Scripts\python.exe scripts\e2e_smoke.py

# Rebuild FAISS index (if data changes)
.\.venv\Scripts\python.exe scripts\build_faiss.py --embed-model BAAI/bge-base-en-v1.5
```

### Running Tests

```powershell
# Test orchestrator directly
.\.venv\Scripts\python.exe test_advanced_orchestrator.py

# Test dynamic token allocation
.\.venv\Scripts\python.exe test_dynamic_tokens.py

# Test latency profile
.\.venv\Scripts\python.exe test_latency_profile.py

# Test UI integration
.\.venv\Scripts\python.exe test_ui_simple.py
```

### CUDA/GPU Setup

```powershell
# Verify CUDA installation
.\verify_cuda_install.bat

# Fix CUDA path issues (if needed)
.\fix_cuda_path.bat
```

## Architecture Overview

### Core Request Flow

```
User Input
    ↓
[Emergency Detection] → Alert if CRITICAL/HIGH severity
    ↓
[Intent Routing] → medication | symptom | reasoning | general
    ↓
[Symptom Extraction] (Clinical NER with BERT)
    ↓
[FAISS Retrieval] (k=5, BAAI/bge-base-en-v1.5 embeddings)
    ↓
[LLM Generation] (Mistral-7B streaming with dynamic tokens)
    ↓
[Safety Checks] + [Audit Logging]
    ↓
Streaming Response → UI
```

### Key Singletons (Module-Level)

The system uses **singleton patterns** to avoid reloading heavy models:
- `_GPT4ALL_SINGLETON` in `advanced_orchestrator.py` (LLM)
- `_RETRIEVER_SINGLETON` in `advanced_orchestrator.py` (FAISS retriever)
- `_SYMPTOM_SINGLETON` in `advanced_orchestrator.py` (NER extractor)

**Critical**: When modifying these services, understand that they're loaded once per process. Environment variables must be set **before** first import.

### Directory Structure

```
backend/services/
├── advanced_orchestrator.py   # Main LLM reasoning (70KB, ~2000 lines)
├── retriever.py                # FAISS semantic search
├── symptom_extractor.py        # Clinical NER (biomedical BERT)
├── safety_eval.py              # Emergency detection + audit
├── medication_support.py       # Drug decision support (stubs)
├── clinical_safety.py          # Safety checks (stubs for future)
├── config.py                   # Centralized config defaults
└── xai.py                      # Explainability package

frontend/
└── app_professional.py         # Professional UI (active)

data/
├── index.faiss                 # FAISS index (736MB)
├── indexed_chunks.pkl          # Document chunks (206MB)
├── faiss_metadata.pkl          # Metadata (131MB)
└── chunks/                     # Raw JSONL chunks

artifacts/
└── mistral-7b-instruct-v0.2.Q4_K_M.gguf  # LLM model (4.4GB)
```

## Environment Variables Reference

### Critical Configuration (start_system.bat)

The `start_system.bat` file sets all environment variables. Key ones:

**Model & Generation:**
- `BAYMAX_GGUF_MODEL_NAME` - Model filename (default: mistral-7b-instruct-v0.2.Q4_K_M.gguf)
- `BAYMAX_GGUF_MODEL_DIR` - Model directory (default: artifacts)
- `BAYMAX_GEN_THREADS` - CPU threads (default: 12)
- `BAYMAX_GEN_MAX_TOKENS` - Max tokens for complex queries (default: 500)
- `BAYMAX_GEN_MIN_TOKENS` - Min tokens for simple queries (default: 30)
- `BAYMAX_GEN_TEMP` - Temperature (default: 0.8)
- `BAYMAX_GEN_STYLE` - abstractive | extractive (default: abstractive)

**Performance:**
- `BAYMAX_FAST_MODE` - Enable speed optimizations (default: true)
- `BAYMAX_POST_ABSTRACT` - Enable paraphrase post-pass (default: 0, adds latency)
- `BAYMAX_GENERATE_ALL` - Force generation for all inputs (default: 1)
- `BAYMAX_NO_FALLBACKS` - Disable fallback responses (default: 1)
- `BAYMAX_ENABLE_XAI` - Enable explainability (default: 0, adds latency)

**Retrieval:**
- `BAYMAX_EMBEDDING_MODEL` - Embedding model (default: BAAI/bge-base-en-v1.5)
- `BAYMAX_RAG_STRICT` - Strict RAG mode, context-only (default: 1)
- `BAYMAX_RERANKER` - Cross-encoder reranker (set to "none" for speed)

**GPU Acceleration (Optional):**
- `LLAMA_CUBLAS=1` - Enable CUDA for llama.cpp
- `CUDA_VISIBLE_DEVICES=0` - GPU device ID
- `BAYMAX_FAISS_GPU=1` - Enable FAISS GPU offload

**IMPORTANT**: Environment variables are read when services first load. Changes require restart of the UI/process.

## Development Guidelines

### Working with the Advanced Orchestrator

The `AdvancedMedicalOrchestrator` is the heart of the system:

1. **Initialization is expensive** (~3-5s to load model). Use singleton pattern.
2. **Thread safety**: Uses `_gen_lock` for model generation (only one generation at a time)
3. **Emergency detection**: Runs BEFORE retrieval to catch critical conditions immediately
4. **Intent routing**: Determines flow (medication → DRAFT output, others → LLM generation)
5. **Dynamic token allocation**: Adjusts response length based on query complexity

**Key method**: `synthesize_advanced_response(user_text, retrieved_passages, symptom_data, user_mode)`

### Working with FAISS Retrieval

The retriever (`backend/services/retriever.py`) provides semantic search:

1. **Index is memory-mapped** (IO_FLAG_MMAP) for fast startup
2. **Embeddings cached** (LRU cache, default 256 entries)
3. **Cosine similarity**: Uses IndexFlatIP on normalized vectors
4. **Retrieval params**: k=5 (top results), min_score=0.3

**To rebuild index**: Run `scripts\build_faiss.py` after updating data in `data/chunks/`

### Symptom Extraction (Clinical NER)

Uses `samrawal/bert-base-uncased_clinical-ner` model:

1. **Hybrid approach**: NER model + rule-based patterns
2. **Extracts**: symptoms, duration (e.g., "3 days"), severity (mild/moderate/severe)
3. **CUDA support**: Will use GPU if available
4. **Cached**: LRU cache (64 entries) for repeated queries

### Emergency Detection System

Multi-category severity scoring across 8 categories:
- Cardiac, Respiratory, Neurological, Trauma, Allergic, Mental Health, Metabolic, Obstetric

**Severity Levels:**
- **CRITICAL** (score ≥10): "CALL 911 IMMEDIATELY"
- **HIGH** (score ≥7): "Go to nearest emergency department"
- **MODERATE** (score ≥3): "Seek medical evaluation within hours"
- **LOW** (score <3): No emergency alert

**Negation handling**: 10-char window for "no", "not", "denies" to reduce false positives

### UI Development

Single UI:
- **app_professional.py** (ACTIVE) - Default UI launched by start_system.bat

**Streamlit caching**: Use `@st.cache_resource` for service loading to prevent reloads on UI rerun

### Audit & Safety Logging

Two logging systems:
1. **audit_log.jsonl** - Lightweight JSONL logs (event, timestamp, timing)
2. **audit_YYYYMMDD.log** - Detailed safety logs (emergency detection, risk levels)

**Logged events**: chat_answer, chat_draft, emergency_detected

## Known Performance Issues

### CPU-Only Inference Bottleneck

**Problem**: Generation takes 10-40s (5-10x slower than expected)

**Root Cause**: Missing CUDA DLLs prevent GPU acceleration
- Model runs on CPU only
- Expected: cublas64_12.dll, cublasLt64_12.dll, cudart64_12.dll

**Solution**: Install NVIDIA CUDA Toolkit 12.x
```powershell
# After installing CUDA, verify:
where cublas64_12.dll
# Should show path in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
```

**Workarounds** (if GPU unavailable):
1. Reduce token limits (edit start_system.bat):
   ```batch
   set BAYMAX_GEN_MAX_TOKENS=160
   set BAYMAX_GEN_MIN_TOKENS=30
   ```
2. Use smaller model (phi-2.Q4_0.gguf, 1.6GB) - lower quality but faster
3. Accept slower performance (~10-20s responses)

### Fast Mode Optimizations

Enabled by default (`BAYMAX_FAST_MODE=true`):
- Short queries (≤16 chars, no medical keywords) skip retrieval
- Dynamic token limits (30-500 based on query length)
- Single reference in fast mode (k=1 in prompt, not retrieval)
- Disabled reranker and post-abstractive pass

## Data Pipeline

### Adding New Medical Data

1. **Create chunks**: Add JSONL files to `data/chunks/`
   - Format: `{"id": "...", "text": "...", "source": "...", "category": "...", ...}`
   - See existing files: all_chunks.jsonl, drug_chunks.jsonl, pubmedqa_chunks.jsonl

2. **Rebuild FAISS index**:
   ```powershell
   .\.venv\Scripts\python.exe scripts\build_faiss.py --embed-model BAAI/bge-base-en-v1.5
   ```

3. **Restart services** to load new index

### Chunking Guidelines

- **Q&A pairs**: Keep question + answer together (type: "QA_pair")
- **Guidelines**: Chunk by section/paragraph (~300 words)
- **Medications**: Separate chunks for indication, dosing, side effects
- **Token limit**: Aim for 200-500 tokens per chunk

## Testing Strategy

### Health Checks

All services provide `health_check()` method:

```python
from backend.services.retriever import MedicalRetriever
retriever = MedicalRetriever(index_dir="data")
health = retriever.health_check()
# Returns: {"status": "healthy", "chunks_loaded": 239644, ...}
```

### Integration Testing

Use `scripts/e2e_smoke.py` for end-to-end validation:
- Tests emergency detection (chest pain query)
- Tests complex reasoning (necrotizing fasciitis query)
- Validates full pipeline: extraction → retrieval → generation

### Unit Testing

Test files in root directory (test_*.py):
- `test_advanced_orchestrator.py` - LLM generation
- `test_latency_profile.py` - Performance profiling
- `test_extraction.py` - NER extraction
- `test_ui_simple.py` - UI integration

## Clinical Safety & Disclaimers

### Medical Device Status

**NOT a medical device**. For informational/educational use only.
- No FDA clearance
- Not a substitute for professional medical advice
- Always requires healthcare professional review

### Medication Safety

Medication queries generate **DRAFT** outputs (decision support, NOT prescriptions):
- Shows drug candidates with reasoning
- Safety checks are stubs (ready for clinical integration)
- UI displays "DRAFT — Decision Support Only" banner

### Emergency Detection

Best-effort detection, not guaranteed:
- Multi-category scoring across 8 emergency types
- Negation handling to reduce false positives
- **Always advise**: "For emergencies, call emergency services immediately"

## Future Enhancements (from PROJECT_KNOWLEDGE.md)

### High Priority
- GPU acceleration (install CUDA DLLs)
- Complete RxNorm integration (drug vocabulary)
- DailyMed SPL integration (FDA drug labels)
- Clinical guidelines (WHO, CDC, NICE)

### Medium Priority
- Multi-turn conversation (persistent state)
- Differential diagnosis engine (Bayesian reasoning)
- Real-time safety checks (allergy DB, drug interactions)
- Advanced RAG (query expansion, hybrid retrieval)

### Long-Term
- EHR/EMR integration (HL7/FHIR)
- Multimodal support (OCR, medical images, voice)
- Telemedicine features
- Clinical validation & FDA clearance path

## Project-Specific Rules

1. **Single Project Focus**: This is one unified medical AI system. Do not create parallel backends or duplicate services.

2. **Singleton Awareness**: Services use module-level singletons. Environment variables must be set before first import. Restart required for config changes.

3. **Privacy First**: All processing is local. No external API calls. Logs may contain PHI - handle per HIPAA/GDPR.

4. **Safety Disclaimers**: Always include medical disclaimers. Emergency detection is best-effort, not guaranteed.

5. **Performance Context**: CPU-only inference is the current bottleneck (missing CUDA DLLs). This is a hardware issue, not a code issue.

6. **No Fallbacks Policy**: User explicitly requested no fallback responses (BAYMAX_NO_FALLBACKS=1). System generates medical responses or returns empty.

7. **Strict RAG Mode**: Default is strict RAG (BAYMAX_RAG_STRICT=1) - answers only from retrieved context, no general advice.

## Troubleshooting

### Model not found
- Check `BAYMAX_GGUF_MODEL_NAME` and `BAYMAX_GGUF_MODEL_DIR` in start_system.bat
- Verify file exists: `artifacts\mistral-7b-instruct-v0.2.Q4_K_M.gguf`

### FAISS index missing
- Run: `.\.venv\Scripts\python.exe scripts\build_faiss.py`
- Check: `data\index.faiss` exists (736MB)

### Slow responses (10-40s)
- **Root cause**: CPU-only inference (missing CUDA DLLs)
- **Quick fix**: Reduce tokens in start_system.bat
- **Proper fix**: Install CUDA Toolkit 12.x

### GPU not detected
- Install NVIDIA CUDA Toolkit
- Verify DLLs in PATH: `where cublas64_12.dll`
- Set `LLAMA_CUBLAS=1` in start_system.bat

### Services fail to load
- Activate venv: `.\.venv\Scripts\activate`
- Check dependencies: `pip list | grep -E "gpt4all|faiss|transformers|streamlit"`
- Review logs in `logs/` directory

### Environment variables not working
- **Don't run Python directly** (`python test.py`)
- **Use start_system.bat** which sets all env vars
- Direct Python calls don't inherit batch file environment

### UI stuck at "System Online" / Chat input not appearing
- **Root cause**: Services take 10-15 seconds to load (model + FAISS index)
- **Fixed**: UI now shows spinner during service loading
- **Wait**: Give it 15 seconds on first launch
- **Verify**: Chat input box should appear after "Services loaded!" message
- If still stuck after 30s, check logs for errors
- Make sure you're using `start_system.bat` not direct Python

---

**Last Updated**: November 5, 2025  
**Project Status**: Operational (CPU-only), needs GPU acceleration  
**Version**: 1.0-RC (Release Candidate)
