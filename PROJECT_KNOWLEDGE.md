# Healix — Project Knowledge (canonical)

**Last updated:** June 5, 2026
**Status:** Operational, local-first medical reasoning companion

This is the single source of truth for how Healix works. It supersedes the
former `HEALIX_FULL_KNOWLEDGE.txt` and `CHECKPOINT.md` (now removed) and the
older draft of this file, both of which had drifted from the running system.

> Naming note: the project was originally "Baymax.v1". Every environment
> variable is still prefixed `BAYMAX_`. Healix is the product name.

---

## 1. What it is

A local-first medical companion that runs entirely on the user's device. It
combines:

- Retrieval-Augmented Generation (RAG) over a FAISS index of ~240k medical chunks
- Optional cross-encoder reranking (MiniLM)
- A summarization (abstraction) pass that fuses the top hits into one understanding
- A single external prompt template (`healix_main_prompt.txt`) as the source of truth
- Calm, human-paced, psychophysiology-aware explanations — no legal disclaimers,
  no emergency directives
- GPU-accelerated llama.cpp (cuBLAS) preferred, GPT4All (CPU) fallback
- An agentic Mixture-of-Experts router over medical specialties (see RESEARCH.md)
- A custom web app (FastAPI + streaming + hand-built chat UI) with persistent
  multi-conversation history and multi-modality medical imaging

---

## 2. Architecture

```
User input (+ optional image)
   -> scope gate (medical-only domain guard; greetings/identity/out-of-scope
      answered instantly with canned lines, no retrieval, no LLM)
   -> [image] vision analysis (chest X-ray / skin) -> findings
   -> MoE gate -> top-k specialist experts (+ modality affinity)
   -> per-expert lens retrieval (multi-query) -> pooled evidence
   -> prompt composition (single template + panel directive + findings)
   -> LLM generation (llama.cpp GPU, else GPT4All), streamed via SSE
   -> persist turn (conversation_store) -> response + routing + findings
```

- **Web app:** `backend/api.py` (FastAPI) serves `frontend/web/` (custom
  `index.html` / `styles.css` / `app.js`) and exposes SSE chat, conversation CRUD,
  and `/api/info`. The UI is a ChatGPT/Claude-style chat: persistent history
  sidebar, inline "+" image attach, expert chips, findings bars, copy buttons.
- **Conversations:** `backend/conversation_store.py` — JSON-per-conversation
  persistence so history survives restarts and provides model context.
- **MoE router:** `backend/services/moe_router.py` — embedding-gated sparse routing
  to specialist experts + lens-conditioned retrieval (see RESEARCH.md).
- **Orchestrator:** `backend/services/advanced_orchestrator.py` — loads the one
  prompt template, composes prompts (evidence + image findings + expert directive)
  for streaming and non-streaming, handles identity/greeting, optional XAI.
- **Retriever:** `backend/services/retriever.py` — FAISS search + optional
  cross-encoder rerank; lazy-loads the embedding model and reranker.
- **Vision:** `backend/services/vision.py` — local multi-modality image analysis,
  chest X-ray + skin lesion, auto-routed (see section 4a).
- **Aux services:** `safety_eval.py` (internal risk/audit), `audit.py`
  (JSONL logging), `medication_support.py` (FastAPI clinician/prescription
  decision-support flow), `xai.py` (explainability), `symptom_extractor.py`
  (optional clinical NER, not required at runtime), `config.py` (fallback defaults).

---

## 3. Single main prompt (source of truth)

- File: `healix_main_prompt.txt` (project root). Edit it to change behavior
  instantly — there is no large embedded prompt in code.
- Wiring: `start_system.bat` and `scripts/run_frontend.ps1` set
  `BAYMAX_MAIN_PROMPT_FILE` to this path. The orchestrator loads it via
  `_load_main_prompt_template()` and fills it with `_compose_main_prompt()` for
  both streaming and non-streaming paths.
- Placeholders: `{CONVERSATION_MEMORY}`, `{EVIDENCE_BLOCK}`, `{SYMPTOM_LINE}`,
  `{USER_QUESTION}`, optional `{REASONING_SUMMARY_INSTRUCTION}`. The template
  is filled via `format_map` with a safe dict — never put other literal
  `{braces}` in it. A `<<<USER TURN>>>` line splits system vs user content
  (see section 5).

Behavioral highlights (see the prompt file for the full directive): conversational
humanity tone, psychophysiology awareness (sympathetic/parasympathetic/vagal/HPA),
personal-baseline reasoning, intervention-trial tracking, optional one-line causal
chain, references only when explicitly requested (no URLs/brackets), no emojis,
no exclamation marks.

---

## 4. Retrieval & evidence

- **Index:** `data/index.faiss` — `IndexScalarQuantizer` (SQ8, inner product),
  768-dim, **239,644 vectors, ~175 MB**. Originally an `IndexFlatIP` (~736 MB);
  shrunk in-place to 8-bit scalar quantization via
  `scripts/shrink_faiss_index.py` (98.6% top-10 recall vs flat, mean score error
  1.7e-4 — negligible, so the score thresholds below stay valid). Loaded with
  `IO_FLAG_MMAP`.
- **Embeddings:** `BAAI/bge-base-en-v1.5` (768-dim), GPU if available.
- **Chunks:** `data/indexed_chunks.pkl` (~197 MB) holds the full passage text.
  `faiss_metadata.pkl` was removed — it duplicated chunk fields and was never read
  at runtime. The retriever loads it only if present.
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, lazy-loaded, pre-pool
  `BAYMAX_RERANK_TOPN`, batch `BAYMAX_RERANK_BATCH`; skipped when the top FAISS
  score exceeds `BAYMAX_RERANK_SKIP_THRESHOLD` (0.86). Set `BAYMAX_RERANKER=none`
  to disable.
- **Hybrid retrieval (`HEALIX_HYBRID=1`):** `hybrid_retrieve` fuses dense FAISS
  with a sparse **BM25** index (`backend/services/hybrid_retriever.py`, a
  scikit-learn sparse matrix, cached under `data/`) using **Reciprocal Rank
  Fusion**, then reranks. Catches exact terms (drug names, rare conditions) dense
  search blurs. Used by the MoE per-expert lens queries and the fallback path.
- **Contextual Retrieval (`HEALIX_CONTEXTUAL=1`):** the BM25 arm is built over
  *context-enriched* chunks — each chunk is prefixed with its situating context
  (source / category / question) before indexing (the free, lexical half of
  Anthropic's Contextual Retrieval; the LLM-context dense pass is the roadmap in
  RESEARCH.md §8). Cached to `data/bm25_context.pkl`.
- **Summarization:** the non-streaming path summarizes the top hits into one
  integrated understanding injected as `{EVIDENCE_BLOCK}`; the streaming path uses
  compact evidence lines for latency.
- **Score thresholds:** retrieval filters on absolute cosine scores
  (`min_score`, default 0.25 from the UI; rerank skip 0.86). This is why the index
  uses scalar quantization (near-exact scores) rather than lossy product
  quantization, which would shift these scores.

**Data sources in the index:** MedQuAD (~47k Q&A, NIH/NLM), PubMedQA, drug info,
plus a small set of enhanced/medication entries. RxNorm is partially integrated
(`scripts/ingest_rxnorm.py`, `integrate_rxnorm.py`).

> The original `data/chunks/*.jsonl` sources are no longer on disk, so
> `scripts/build_faiss.py` cannot rebuild from scratch. `shrink_faiss_index.py`
> works by reconstructing vectors directly from the existing index, and the chunk
> text survives in `indexed_chunks.pkl`.

---

## 4a. Medical computer vision (imaging)

`backend/services/vision.py` adds local, multi-modality medical image analysis.
Images are **auto-routed** by a saturation heuristic (grayscale → radiograph,
colorful → skin/derm photo) or by an explicit `modality` argument.

- **Chest X-ray:** TorchXRayVision `densenet121-res224-all` — a peer-reviewed
  DenseNet (NIH/PadChest/CheXpert/MIMIC/…) estimating 18 thoracic pathologies
  (atelectasis, cardiomegaly, consolidation, edema, effusion, emphysema, fibrosis,
  fracture, hernia, infiltration, lung lesion, lung opacity, mass, nodule, pleural
  thickening, pneumonia, pneumothorax, …).
- **Skin lesion:** a Hugging Face image-classification model
  (`HEALIX_SKIN_MODEL`, default `Anwarkh1/Skin_Cancer-Image_Classification`),
  loaded via `transformers` pipeline.
- **Flow:** the vision model performs *perception* (what is visible) and returns
  a structured findings list + a compact `prompt_block`. That block is injected
  into the LLM prompt via `prepare_stream_prompt(..., image_findings=...)`, so the
  text pipeline *explains* the findings in Healix's calm voice, grounded by RAG.
- **Safeguards:** a saturation heuristic rejects non-radiograph images
  (calibrated for chest X-rays); everything degrades gracefully — if the library
  or weights are missing, analysis returns an "unavailable" result instead of
  raising, so the app never breaks. Findings are framed as screening estimates,
  not diagnoses.
- **Runtime:** lazy-loaded and cached; uses GPU if available. Weights (~tens of
  MB) download once to `~/.torchxrayvision/` then run offline. The download
  progress bar is redirected to an in-memory buffer to avoid a Windows console
  Unicode crash.
- **UI:** images are attached inline via the chat composer's "+" button. The
  image appears as a thumbnail in the user's message, the findings render as
  probability bars in the reply, and the model explains them.
- **Smoke test:** `python scripts/check_vision.py`.

---

## 4b. Agentic Mixture-of-Experts (MoME)

`backend/services/moe_router.py` adds a system-level sparse MoE over medical
specialists (Cardiology, Pulmonology, Neurology, Dermatology, Gastroenterology,
Pharmacology, Psychophysiology, General).

- **Gate:** cosine similarity between the query embedding and per-expert prototype
  embeddings, plus keyword priors and an imaging-modality affinity, sharpened by a
  softmax temperature; the top-k (default 2) experts above a floor are activated.
- **Experts:** each activated expert reformulates the query through its retrieval
  *lens* and pulls specialty-specific evidence (multi-query expansion); the union
  grounds one synthesis pass framed as an integrated panel (`expert_directive`).
- **Explainability:** the activated experts and gate weights stream to the UI as
  "Consulted" chips and are persisted on each reply.
- **Config:** `HEALIX_MOE=1` (default; `0` to A/B against plain RAG),
  `HEALIX_MOE_TOPK`, `HEALIX_MOE_TEMP`. Smoke test: `python scripts/check_moe.py`.
- **Full design, novelty, and evaluation plan:** see `RESEARCH.md`.

---

## 4c. Altitude-matched retrieval: RAPTOR × HyDE

Broad questions need evidence no single chunk contains. Two coupled upgrades
(full design + novelty: RESEARCH.md §9):

- **HyDE (`backend/services/hyde.py`, `HEALIX_HYDE=1`):** one ~4 s Ollama call
  per medical query drafts what a reference text would say at two abstraction
  levels — a SPECIFIC clinical-detail line and a BROAD topic line. The specific
  line is appended to the leaf search query (dense + BM25 + MoE lenses); the
  broad line searches the RAPTOR tree. Fail-open: on any error retrieval uses
  the raw query.
- **RAPTOR tree (`scripts/build_raptor.py` offline, `backend/services/raptor.py`
  runtime, `HEALIX_RAPTOR=1`):** k-means over leaf embeddings (reconstructed
  from the SQ8 index) → ~1,500 topic nodes summarized by the local model
  (largest-first, resumable JSONL) → ~120 domain nodes. Artifacts:
  `data/raptor.faiss` + `data/raptor_nodes.pkl`. Top overview nodes are
  prepended to the evidence block as "Healix knowledge tree" passages.
- **Build:** `python scripts/build_raptor.py --topics 1500 --domains 120`
  (~2 h local Ollama time, resumable; `--cap N` for partial runs,
  `--finalize-only` to re-embed whatever nodes exist).

---

## 4d. Medical-only scope gate

`backend/services/scope_gate.py` resolves every message to `greeting | thanks |
farewell | identity | out_of_scope | medical` before any retrieval or
generation. Courtesy kinds and clear non-medical asks (code, trivia, sports,
finance, entertainment, travel, ...) stream a canned house-voice line over the
same SSE contract with zero model time; only `medical` runs the full pipeline.

- **Precision-first refusals:** word-boundary matching only (the old substring
  check classified "hip pain" as a greeting via "hi"); broad medical vocabulary
  always wins; short mid-conversation follow-ups are never refused.
- **Embedding tiebreaker:** for ambiguous standalone messages the gate reuses
  the retriever's encoder against medical vs non-medical anchor prototypes and
  refuses only with a clear margin. Any failure degrades to `medical`.
- **Defense in depth:** `healix_main_prompt.txt` carries a strict
  "Domain" directive so anything that slips through is declined by the LLM;
  the non-streaming path (`synthesize_advanced_response`) also applies
  `_is_healthcare_topic` -> polite refusal.
- Smoke test: `python scripts/check_scope.py` (add `--full` to exercise the
  embedding tiebreaker).

---

## 5. Generation, streaming, fallbacks

- One prompt composition for both paths via `_compose_main_prompt`.
- The template contains a `<<<USER TURN>>>` divider: text above it is sent to
  Ollama as the **system message** (stronger instruction adherence), text below
  (memory, evidence, question) as the user prompt. Backends without a system
  slot strip the marker. The Ollama payload also pins `num_ctx` (from
  `BAYMAX_GEN_CONTEXT`) so long prompts are never silently truncated from the
  top, and `keep_alive=30m` (`BAYMAX_OLLAMA_KEEP_ALIVE`) to avoid reload lag.
- Token heuristics (runtime values): `GEN_MAX_TOKENS=900`, `GEN_MIN_TOKENS=60`,
  `GEN_SHORT_TOKENS=256`, context `8192`.
- Short follow-up messages (<8 words) are retrieved with the previous user
  message folded into the search query (`backend/api.py`), so "what about at
  night?" still finds on-topic evidence; the prompt itself is unchanged.
- **Backends (pluggable):** `BAYMAX_LLM_BACKEND=ollama` (default) streams from the
  local Ollama server (`BAYMAX_OLLAMA_MODEL`, e.g. `llama3:latest`) and **skips the
  GGUF load entirely** — faster startup and stronger models. Ollama is now the
  only bundled backend; the GGUF stack (`llama-cpp-python`/`gpt4all` + model) was
  removed to save ~5.7 GB. To restore an offline GGUF fallback, reinstall those
  packages, drop a `.gguf` in `artifacts/`, and clear `BAYMAX_LLM_BACKEND` — the
  optional-import code paths are still in place.
- If streaming yields zero tokens, the system falls back to a non-streaming call.
- `config.py` holds conservative *fallback* defaults (e.g. `k=5`, `temp=0.25`)
  used only when the corresponding env var is unset; the launchers set the real
  runtime values below.

---

## 6. Safety posture (by design)

- **No legal/medical disclaimers** anywhere.
- **No emergency/urgent directives** ("call 911"); neutral phrasing only.
  `_get_advanced_emergency_message` returns an empty string.
- Emergency *detection* still runs internally (`_detect_advanced_emergency`) and
  is logged to `logs/audit_log.jsonl` for analytics — it just produces no
  user-facing alert.
- Medication guidance: general info and classes/examples; patient-specific dosing
  only when explicitly requested. The `medication_support.py` clinician flow emits
  DRAFT decision-support output, not prescriptions; its allergy/interaction checks
  are stubs awaiting real clinical-DB integration.

---

## 7. Configuration & launch

Two launch paths exist and **must be kept in sync**:

| Launcher | Sets env how | When |
|---|---|---|
| `start_system.bat` / `scripts/run_frontend.ps1` | exports `BAYMAX_*` before launch | normal use |
| `.env` (via `load_dotenv`) | only fills vars **not already set** | bare `python` runs |

Because `load_dotenv()` does not override existing variables, the launcher wins
when you use it, and `.env` applies only to direct `python` invocations. As of
this cleanup both define the same retrieval config (reranker on,
`BAYMAX_RAG_STRICT=0`) so behavior is identical either way. See the header of
`.env` for the precedence note.

Key runtime variables (full set in `start_system.bat`): model
`mistral-7b-instruct-v0.2.Q4_K_M.gguf` in `artifacts/`, embeddings
`BAAI/bge-base-en-v1.5`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`,
`BAYMAX_GEN_TEMP=0.4`, `BAYMAX_GEN_TOP_K=10`, `BAYMAX_GEN_TOP_P=0.9`,
`BAYMAX_GEN_STYLE=abstractive`, `BAYMAX_ENABLE_XAI=1`, GPU flags
(`LLAMA_CUBLAS=1`, `CUDA_VISIBLE_DEVICES=0`, `BAYMAX_FAISS_GPU=1`), UI port 8504.
The script also prepends the CUDA 12.6 `bin` to `PATH` for cuBLAS DLLs.

---

## 8. Dependencies & footprint

- `requirements.txt` — runtime (app + retrieval + inference), including the
  computer-vision deps (`torchxrayvision`, `scikit-image`, `pydicom`, `pillow`).
  `altair` is an optional extra that enables the UI progress charts (lazy-imported).
- `requirements-train.txt` — training-only deps (`peft`, `trl`, `datasets`,
  `accelerate`, `bitsandbytes`) for `scripts/train_sft.py` and
  `scripts/merge_lora.py`. Not needed to run the app.

Approximate on-disk footprint (Ollama-only): **~5.9 GB total** — `.venv` ~5.5 GB
(mostly CUDA torch), `data/` ~0.4 GB (index 175 MB + chunks 197 MB + contextual
BM25 ~42 MB). The bundled GGUF model + `llama-cpp-python`/`gpt4all` +
`third_party/llama.cpp` were removed (~5.7 GB) in favor of Ollama. `data/`,
`datasets/`, `.venv/` are gitignored.

---

## 9. Running it

```bat
:: Windows (preferred) — launches the web app, then open http://127.0.0.1:8504
start_system.bat
```

```powershell
# Or run the server directly
.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8504
```

Quick checks:

```powershell
.venv\Scripts\python.exe backend\services\retriever.py   # retriever self-test
.venv\Scripts\python.exe scripts\check_vision.py         # image analyzer smoke
.venv\Scripts\python.exe scripts\check_moe.py            # MoE routing smoke
.venv\Scripts\python.exe scripts\shrink_faiss_index.py   # SQ8 index (dry run)
```

In the chat, attach a medical image with the "+" button; each reply shows the
consulted experts (MoE) and, for images, the findings bars. Ask "show sources"
for a short source list.

---

## 10. File map

```
healix_main_prompt.txt              single prompt template (edit me)
start_system.bat                    Windows launcher (env vars, GPU, runs web app)
backend/api.py                      FastAPI app: SSE chat, conversations, /api/info
backend/conversation_store.py       persistent conversation history (JSON per chat)
frontend/web/                       custom chat UI (index.html, styles.css, app.js)
RESEARCH.md                         agentic MoE architecture + evaluation plan
scripts/shrink_faiss_index.py       SQ8 index shrink/rebuild (reconstruct-based)
scripts/build_faiss.py              full index build (needs data/chunks/*.jsonl)
scripts/check_vision.py             image analyzer smoke test
scripts/check_moe.py                MoE routing smoke test
scripts/check_hybrid.py             build BM25 + dense-vs-hybrid comparison
scripts/contextualize_and_reembed.py  full Contextual Retrieval (LLM ctx + re-embed)
scripts/train_sft.py, merge_lora.py LoRA/SFT training (see requirements-train.txt)
backend/services/advanced_orchestrator.py  prompt composition, summarization, XAI
backend/services/moe_router.py      agentic Mixture-of-Experts router
backend/services/hybrid_retriever.py  sparse BM25 + Reciprocal Rank Fusion
backend/services/retriever.py       dense FAISS + hybrid_retrieve + rerank
backend/services/hyde.py            two-altitude HyDE hypotheticals (runtime)
backend/services/raptor.py          RAPTOR overview-tree search (runtime)
scripts/build_raptor.py             RAPTOR tree build (k-means + Ollama, resumable)
backend/services/vision.py          multi-modality imaging (chest X-ray + skin)
backend/services/*.py               safety, audit, medication, xai, symptom NER
data/index.faiss                    SQ8 FAISS index (~175 MB)
data/indexed_chunks.pkl             chunk text (~197 MB)
data/conversations/                 saved chats (gitignored)
artifacts/*.gguf                    LLM weights (4.4 GB)
third_party/llama.cpp               inference backend
requirements.txt / requirements-train.txt  runtime / training deps
WARP.md                             Warp terminal project notes
```

---

## 11. Known trade-offs & roadmap

**Trade-offs (intentional):** no emergency instructions; no legal disclaimers;
dosing only on explicit request; references hidden unless asked; personalization
is lightweight (optional CSV logging), not model-driven.

**Index:** SQ8 gives ~4x size reduction at ~99% recall. Going smaller (product
quantization, ~20-30 MB) is possible but would shift the absolute similarity
scores the retriever filters on, so it needs threshold re-tuning + recall
validation first.

**Roadmap candidates:** richer baseline-capture UI with per-user persistence;
optional structured "clinician mode"; complete RxNorm/DailyMed integration; real
(non-stub) drug-interaction/allergy checks; multi-turn memory; quantitative
validation of psychophysiology inference.

---

## 12. Recent maintenance (June 2026 cleanup)

- Cleared the user pip wheel cache (~10.3 GB, outside the repo).
- Uninstalled unused/training-only venv packages: `gradio` + `gradio_client`
  (UI is Streamlit), `bitsandbytes`, `spacy` (~470 MB).
- Shrank the FAISS index 736 MB -> 175 MB (IndexFlatIP -> SQ8) and removed the
  redundant `faiss_metadata.pkl` (~125 MB); retriever made metadata-optional.
- Aligned `.env` with `start_system.bat` (reranker on, `RAG_STRICT=0`) and
  documented launch precedence.
- Split `requirements.txt` into runtime + `requirements-train.txt`; fixed a
  duplicate import; consolidated four overlapping docs into this file + README.

**Feature work (same round):**
- Added local medical computer vision (`backend/services/vision.py`) — chest
  X-ray analysis via TorchXRayVision, with findings injected into the LLM prompt
  (`prepare_stream_prompt(image_findings=...)`).
- Rebuilt the Streamlit UI as a ChatGPT/Claude-style chat: empty-state greeting,
  clean message bubbles, and a single composer with an inline "+" attach button
  for images (native `st.chat_input(accept_file=...)`). No emojis, no disclaimer
  clutter. Attached chest X-rays show as a thumbnail; findings render as
  probability bars in the reply.
- Replaced the empty/alarming emergency banner with the calm LLM path (detection
  retained for the audit log only), aligning the UI with the no-emergency-directive
  design.

**Web rebuild + research (later in the round):**
- Replaced Streamlit with a custom web app: `backend/api.py` (FastAPI, SSE
  streaming) serving a hand-built chat UI in `frontend/web/`.
- Added persistent multi-conversation history (`backend/conversation_store.py`),
  fixing the "forgets past chats" problem, and widened the per-turn memory window
  (`BAYMAX_MEMORY_TURNS=10`, `BAYMAX_MEMORY_CHARS=2400`).
- Added an agentic Mixture-of-Experts router (`backend/services/moe_router.py`,
  section 4b, `RESEARCH.md`); routing chips shown per reply.
- Extended vision to multi-modality (chest X-ray + skin lesion, auto-routed).
- Pointed `start_system.bat` at uvicorn; the Streamlit app and `.streamlit/`
  theme are superseded (`frontend/app_professional.py` left in place but unused).

**Hybrid RAG + brand refresh (later in the round):**
- Added hybrid retrieval — dense FAISS + sparse BM25 fused with Reciprocal Rank
  Fusion, reranked (`backend/services/hybrid_retriever.py`); wired into the MoE
  experts and the fallback path (`HEALIX_HYBRID=1`). RESEARCH.md §8.
- New logo (hexagonal node + vitals pulse) + `favicon.svg`; UI polish
  (depth gradient, refined empty state).

**Ollama brain + black/white UI + contextual RAG (later in the round):**
- Added a pluggable **Ollama** generation backend (`BAYMAX_LLM_BACKEND=ollama`,
  `BAYMAX_OLLAMA_MODEL=llama3:latest`) that streams from the local Ollama server
  and skips the GGUF load; falls back to the local model if unreachable.
- Redesigned the UI to a **monochrome black-and-white** theme (no glow, Claude-
  grade), monochrome logo/favicon, and a looping "thinking" logo animation while
  the model generates.
- Added **Contextual Retrieval** (lexical half) — context-enriched BM25
  (`HEALIX_CONTEXTUAL=1`).

**Generation-quality upgrade (July 2026):**
- Rewrote `healix_main_prompt.txt` for the 7B Ollama model and validated it live
  (3 runs per scenario against the real model): evidence-noise handling,
  accuracy rule, one consolidated question quota, anti-boilerplate bans,
  length targets. Lesson: never put quotable example sentences in rules —
  the model parrots them verbatim.
- Split the prompt into system vs user via a `<<<USER TURN>>>` marker
  (`PROMPT_SPLIT_MARKER` in the orchestrator); Ollama now gets the rules as a
  true system message. Conversation memory lives in the system half with
  historical labels ("You already replied:") because a `User:/Healix:`
  transcript in the user turn made the model re-state its last reply.
- Fixed two silent bugs: the SSE chat path hardcoded `temp=0.15` (now reads
  `BAYMAX_GEN_TEMP`, set to 0.4), and the Ollama payload never pinned
  `num_ctx`, so prompts longer than the 4096 default were truncated from the
  top — chopping the rules first. Now pinned from `BAYMAX_GEN_CONTEXT` (8192).
- Follow-up-aware retrieval: short messages (<8 words) fold the previous user
  message into the search query so "what about at night?" retrieves on-topic.
- Deterministic post-processing: an opening paragraph that near-duplicates the
  previous reply is stripped (`_strip_opening_echo` in `api.py`), and stray
  exclamation marks are normalized away in `clean_stream_text`.
- Evidence snippets now cut at sentence boundaries and dedupe; Ollama
  `keep_alive=30m` avoids model reload lag between turns.

**Altitude-matched retrieval: RAPTOR × HyDE (July 2026):**
- Implemented roadmap item HyDE and merged it with a RAPTOR summary tree in a
  novel coupling: hypotheticals generated at two abstraction levels, each
  searching the matching index tier (section 4c, RESEARCH.md §9).
- New: `backend/services/hyde.py`, `backend/services/raptor.py`,
  `scripts/build_raptor.py`; wiring in `backend/api.py`; flags `HEALIX_HYDE`,
  `HEALIX_RAPTOR` in both launchers. Everything fail-open.
