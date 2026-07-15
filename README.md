# Healix

Healix is a **local-first medical reasoning companion** — an AI healthcare
assistant that runs entirely on your own machine. It pairs retrieval-augmented
generation (RAG) over ~240k medical knowledge chunks with a local LLM (via
Ollama) to give calm, context-aware, psychophysiology-aware explanations of what
your body and mind may be signaling.

No cloud calls, no external APIs — all reasoning happens locally.

> Deep dives: **[PROJECT_KNOWLEDGE.md](PROJECT_KNOWLEDGE.md)** (architecture,
> retrieval, config, operations) and **[RESEARCH.md](RESEARCH.md)** (the agentic
> Mixture-of-Experts router and the RAPTOR × HyDE retrieval design).

---

## What you get

- **Local LLM via Ollama** — defaults to `qwen2.5:7b-instruct`; swap any pulled
  model with `BAYMAX_OLLAMA_MODEL`.
- **Hybrid RAG** — a FAISS index of 239,644 chunks (MedQuAD, PubMedQA, drug
  data) with `BAAI/bge-base-en-v1.5` embeddings, fused with contextual BM25 via
  Reciprocal Rank Fusion and reranked with `ms-marco-MiniLM`.
- **RAPTOR × HyDE retrieval** — two-altitude query expansion searching a
  summary tree for broad questions and the leaf index for specific ones
  (RESEARCH.md §9). Fail-open.
- **Agentic Mixture-of-Experts** — an embedding-gated router sends each query to
  the top medical specialists; consulted experts are shown per reply.
- **Medical computer vision** — attach a chest X-ray or skin photo and Healix
  analyzes it locally (TorchXRayVision / a skin-lesion classifier) and explains
  the findings.
- **Custom web app** — FastAPI + SSE streaming + a hand-built chat UI with
  persistent multi-conversation history.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10–3.12** | 3.12 recommended. |
| **[Ollama](https://ollama.com)** | Provides the local LLM. Install and keep it running. |
| **~6 GB disk** | ~5.5 GB venv (mostly PyTorch) + ~0.4 GB knowledge index. |
| **NVIDIA GPU + CUDA 12.4** | *Optional.* Speeds up embeddings/vision. Works CPU-only. |
| **Git** | To clone. Use `--depth 1` for a fast, lightweight clone. |

---

## Setup (fresh machine)

### 1. Clone

```bash
git clone --depth 1 https://github.com/ayushkumrdev/Healix.git
cd Healix
```

### 2. Create the virtualenv and install dependencies

`requirements.txt` pins **CUDA 12.4** PyTorch wheels (`torch==2.6.0+cu124`).
Pick the path that matches your hardware:

**GPU (NVIDIA, CUDA 12.4):**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

**CPU-only (no NVIDIA GPU — including macOS):** install CPU PyTorch first, then
the rest:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
grep -viE '^torch(vision|audio)?==' requirements.txt | pip install -r /dev/stdin
```

> On CPU-only machines, leave `BAYMAX_FAISS_GPU=1` as-is — the retriever
> automatically falls back to the CPU FAISS index.

### 3. Install the LLM (Ollama)

Install Ollama from [ollama.com](https://ollama.com), then pull the default
model:

```bash
ollama pull qwen2.5:7b-instruct
```

Ollama runs as a background server on `http://localhost:11434`. Verify with
`ollama list`.

### 4. Get the knowledge index

Healix needs two files in `data/` that are **not** in the git repo (too large):

- `data/index.faiss` — the FAISS vector index (~175 MB)
- `data/indexed_chunks.pkl` — the chunk text (~197 MB)

**Option A — download the prebuilt index (recommended).** Grab both files from
the project's [GitHub Releases](https://github.com/ayushkumrdev/Healix/releases)
and drop them into `data/`:

```bash
mkdir -p data
# download index.faiss and indexed_chunks.pkl into data/
```

**Option B — rebuild from public sources (advanced).** The ingestion scripts
pull public datasets (MedQuAD, PubMedQA) and re-embed them. This needs a GPU for
reasonable speed and is not a one-command path — expect to adapt it:

```bash
python scripts/download_medical_data.py   # fetch public corpora
python scripts/chunk_documents.py         # chunk into passages
python scripts/build_faiss.py             # embed + build data/index.faiss
```

### 5. Configure environment

```bash
cp .env.example .env
```

The defaults work out of the box. `.env` is read automatically on startup
(via `python-dotenv`); edit it to change the model, ports, or retrieval flags.
`.env` is gitignored — never commit real secrets.

### 6. Run

**Windows:**

```bat
start_system.bat
```

**Any OS:**

```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8504
```

Then open **http://127.0.0.1:8504**.

### 7. Verify

```bash
python backend/services/retriever.py   # retriever self-test (loads the index)
python scripts/check_moe.py            # MoE routing smoke test
python scripts/check_vision.py         # image analyzer smoke test
```

---

## Using it

- Type a health question — replies stream in, with the consulted specialist
  experts shown as chips.
- Attach a medical image with the composer's **+** button; findings render as
  probability bars and the model explains them.
- Ask "show sources" for a short source list.
- Conversations persist across restarts (saved under `data/conversations/`).

### Optional: build the RAPTOR summary tree

For better answers to broad questions, build the overview tree once (uses your
local Ollama model, ~2 h, resumable):

```bash
python scripts/build_raptor.py --topics 1500 --domains 120
```

Until built, RAPTOR fails open and Healix behaves as standard hybrid RAG.

### Optional: fine-tuning

LoRA/SFT training deps live in `requirements-train.txt`; see
`scripts/train_sft.py` and `scripts/merge_lora.py`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip` can't find `torch==2.6.0+cu124` | You skipped the CUDA index URL — use the GPU command in step 2, or the CPU path. |
| `Connection refused` to `:11434` | Ollama isn't running. Start it (`ollama serve`) and `ollama pull qwen2.5:7b-instruct`. |
| Retriever error about `index.faiss` | The knowledge index isn't in `data/` — complete step 4. |
| Replies are slow on first token | Ollama is loading the model; subsequent turns are warm (`keep_alive=30m`). |
| Model name mismatch | Ensure `BAYMAX_OLLAMA_MODEL` matches a model from `ollama list`. |

---

## Repository layout

| Path | What |
|---|---|
| `backend/api.py` | FastAPI app: SSE chat, conversations, image analysis |
| `backend/services/` | orchestrator, retriever, hybrid/RAPTOR/HyDE, MoE, vision, scope gate |
| `backend/conversation_store.py` | persistent conversation history |
| `frontend/web/` | custom chat UI (`index.html`, `styles.css`, `app.js`) |
| `healix_main_prompt.txt` | the single behavioral prompt (edit to change tone) |
| `scripts/` | data ingest, index build, RAPTOR build, smoke tests, training |
| `data/` | FAISS index, chunk text, conversations (gitignored) |
| `artifacts/` | optional GGUF model weights (gitignored) |
| `start_system.bat` | Windows launcher (sets env, starts the app on :8504) |
| `.env.example` | environment template — copy to `.env` |

---

## Disclaimer

Healix is for educational and informational purposes. It is not a medical device
and does not provide diagnosis or emergency triage.

## License

See [LICENSE](LICENSE).
