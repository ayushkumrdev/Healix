# Healix Research: An Agentic Mixture-of-Medical-Experts (MoME)

**Status:** prototype implemented (`backend/services/moe_router.py`), enabled by
default (`HEALIX_MOE=1`). This document describes the architecture, what is novel,
the agentic control loop, and an evaluation plan.

---

## 1. Problem

Healix runs a single quantized 7B model locally. We want **specialist-grade
medical reasoning** and **explainability** without hosting N specialist models or
paying for token-level MoE inside the transformer. Plain RAG over one shared
index also tends to retrieve generically — a cardiac question and a dermatologic
question pull from the same undifferentiated neighborhood.

## 2. Idea: move Mixture-of-Experts up a level

Classic MoE routes *tokens* to expert FFNs via a gating network. We apply the
same sparse-gating principle at the **system/agent level**:

> A learned-prototype gate routes each query (and any image modality) to the
> top-k **medical specialist experts**. Each expert is a small agent that pulls
> its own lens-specific evidence; the union grounds a single LLM synthesis pass,
> framed as an integrated specialist panel.

Experts implemented: Cardiology, Pulmonology, Neurology, Dermatology,
Gastroenterology, Pharmacology, Psychophysiology, and a default General Internal
Medicine expert.

### 2.1 The gate (sparse, explainable)

For query *q* and optional image modality *m*:

```
score_i = cos(E(q), P_i)               # E = bge embedding; P_i = expert prototype embedding
        + λ_kw · 1[keyword hit_i]      # cheap lexical prior
        + λ_mod · 1[m == modality_i]   # imaging affinity (X-ray→Pulmonology, derm→Dermatology)
w = softmax(score / τ)                  # temperature-sharpened gate
A = { i : i ∈ top-k(w) and w_i ≥ floor } ∪ {default if A = ∅}
```

τ controls sparsity: confident queries collapse to one expert; genuinely
cross-domain queries (e.g. "ibuprofen with my blood-pressure pills") split across
two (Cardiology + Pharmacology). The weights are returned and shown in the UI —
the routing is *interpretable*, unlike opaque token gating.

### 2.2 The experts (lens-conditioned retrieval)

Each activated expert reformulates the query through its **lens** (e.g. Cardiology
→ "cardiac and cardiovascular causes of: {q}") and retrieves a few passages. This
is a **multi-query retrieval expansion**: the union covers more relevant evidence
than a single query, and each passage is tagged with its originating expert.

### 2.3 Single synthesis (efficiency)

Rather than run k expensive generations, the activated experts and their pooled
evidence are composed into **one** prompt with a panel directive ("integrate these
weighted perspectives into one calm, unified answer"). One generation → low
latency on a single local model, while still benefiting from specialist routing
and broadened retrieval.

## 3. The agentic control loop

The router is step one of a plan→act→synthesize→verify agent:

1. **Plan** — gate to experts (above).
2. **Act / tools** — experts may invoke tools:
   - Imaging expert → `services.vision` (chest-X-ray / skin classifiers); findings
     are injected into the prompt and shown as probability bars.
   - Pharmacology expert → drug-interaction / formulary checks
     (`services.clinical_safety`, `medication_support`).
3. **Gather** — lens-specific multi-query retrieval.
4. **Synthesize** — single grounded generation in Healix's voice.
5. **Verify / refine** *(roadmap)* — a critic pass scores groundedness and
   uncertainty; on low confidence the controller can re-route (raise k, add the
   General expert, or widen retrieval) and regenerate once.

Steps 1–4 are implemented today; step 5 is specified for the next iteration.

## 4. Why this is novel (vs. related work)

- **Token-level MoE** (Switch/Mixtral): experts are FFNs, gating is opaque, needs
  expert weights. Ours is model-agnostic, wraps one model, and is interpretable.
- **Plain RAG**: single query, undifferentiated retrieval. Ours adds gated,
  lens-conditioned multi-query expansion.
- **Multi-agent debate / panels**: usually N full LLM calls. Ours keeps a single
  synthesis pass, so it is practical on a local 7B.
- **Router-LLMs / model routing**: route *between models*. Ours routes between
  *specialist retrieval+persona experts* over one model, and fuses imaging-modality
  signal into the gate.

The specific combination — embedding-gated specialist routing + imaging-modality
gating + lens-conditioned retrieval expansion + single-model panel synthesis +
explainable weights, all local-first — is the contribution.

## 5. Roadmap toward "real" MoE

1. **Learned gate** — replace the prototype/keyword gate with a small classifier
   trained on specialty-labeled queries (MedQuAD categories, MedMCQA subjects).
2. **True expert weights** — per-specialty **LoRA adapters** over the base GGUF,
   hot-swapped for the top-k experts → genuine expert parameters, still one base
   model on disk. (Training scaffolding already exists: `scripts/train_sft.py`,
   `scripts/merge_lora.py`.)
3. **Expert-parallel decoding** — speculative/parallel drafts from 2 adapters,
   verified by the base model.
4. **Calibrated verify loop** — groundedness + abstention thresholds driving
   re-routing.

## 6. Evaluation plan

- **Datasets:** MedQA / MedMCQA (specialty-labeled), PubMedQA (groundedness),
  held-out MedQuAD by category.
- **Routing accuracy:** gate top-1/top-2 vs. ground-truth specialty labels.
- **Retrieval uplift:** recall@k of lens-expanded multi-query vs. single query.
- **Answer quality:** win-rate (LLM-judge + clinician spot-check) MoME vs.
  no-MoME and vs. k=1.
- **Groundedness:** fraction of claims supported by retrieved evidence.
- **Cost:** added latency from routing + multi-query (target < 0.3 s) and the
  one-generation invariant.
- **Ablations:** gate components (embedding only / +keywords / +modality),
  k ∈ {1,2,3}, with/without lens expansion, with/without verify loop.

## 7. Try it

```bash
python scripts/check_moe.py          # see sample routing
HEALIX_MOE=0 ...                     # disable to A/B against plain RAG
```

In the web app, each answer shows a "Consulted" row with the activated experts
and their gate weights.

---

## 8. Retrieval stack: hybrid RAG (implemented) → "god-level" (roadmap)

Dense embedding search alone has a known failure mode: it captures meaning but
blurs exact tokens (drug names, gene symbols, rare conditions, dosages). The
established fix — and the basis of Anthropic's *Contextual Retrieval* results — is
to add a lexical arm and fuse.

### Implemented

- **Hybrid dense + sparse + RRF** (`backend/services/hybrid_retriever.py`,
  `MedicalRetriever.hybrid_retrieve`): FAISS dense search and a BM25 sparse index
  (built over the existing chunk texts on a scikit-learn sparse matrix) are fused
  with **Reciprocal Rank Fusion**, then reranked by the cross-encoder. BM25 builds
  in ~9 s over 239k chunks and is cached to disk; query cost is negligible.
- **MoE × hybrid:** each specialist expert's lens query runs through hybrid
  retrieval, so routing and lexical/semantic fusion compound.
- **Contextual Retrieval (lexical half), `HEALIX_CONTEXTUAL=1`:** each chunk is
  prefixed with its situating context (source / category / question) before the
  BM25 index is built, so document-level terms become matchable. This is the free
  part of Anthropic's method (no LLM); the LLM-context dense pass is roadmap #1.
- **Toggle:** `HEALIX_HYBRID=1` (default). Smoke/A-B: `scripts/check_hybrid.py`.

Why this is the efficient win: BM25 is CPU-cheap and needs no re-embedding of the
corpus, yet it recovers exactly the queries dense retrieval drops. RRF needs no
score calibration between the two arms (rank-based), so it is robust.

### Roadmap to "god-level"

1. **Contextual Retrieval (full, in progress)** — `scripts/contextualize_and_reembed.py`
   generates an LLM context line per chunk (small local Ollama model, concurrent +
   resumable checkpoint), then re-embeds `context + text` and rebuilds the FAISS
   SQ8 index and contextual BM25. The lexical half already ships; this adds the
   dense half (Anthropic's biggest single gain) as a one-time offline pass.
2. **HyDE** — ~~embed a hypothetical answer generated by the LLM to retrieve for
   under-specified queries~~ — implemented, and extended into the
   altitude-matched design of §9.
3. **CRAG / Self-RAG (corrective loop)** — grade retrieved evidence; on low
   confidence, rewrite the query or widen retrieval and regenerate once. This is
   the same agentic verify/refine loop described in §3.5, applied to retrieval.
4. **GraphRAG** — build an entity/relation graph (drugs ↔ conditions ↔ symptoms)
   with community summaries for multi-hop questions ("which of my meds interacts
   with X and why").
5. **Late-interaction (ColBERT-style)** reranking for token-level precision.
6. **Matryoshka / quantized embeddings** for cheaper dense recall at scale.

Evaluation mirrors §6: recall@k and answer-quality uplift per stage, with the
one-generation latency budget held fixed.

## 9. Altitude-matched retrieval: RAPTOR × HyDE (implemented)

Flat RAG has a structural blind spot: broad questions ("how does stress affect
the body?") need evidence that *no single leaf chunk contains* — the answer
lives in the aggregate. RAPTOR (Sarthi et al., 2024) fixes the index side by
adding recursive summary nodes; HyDE (Gao et al., 2022) fixes the query side by
searching with a hypothetical answer instead of the question. Healix merges
them with a coupling neither paper describes: **the hypothetical is generated
at two abstraction levels, and each level searches the matching tier of the
tree.**

### Design

- **Tree** (`scripts/build_raptor.py`, offline, resumable): leaf level is the
  existing 239k-chunk FAISS index, untouched. Level 1 clusters leaf embeddings
  (reconstructed from the SQ8 index — no re-embedding) into ~1,500 topics via
  k-means; the local Ollama model writes a titled 4-6 sentence overview per
  cluster, largest topics first. Level 2 clusters those summaries into ~120
  domains. Nodes land in `data/raptor.faiss` + `raptor_nodes.pkl`.
- **Two-altitude HyDE** (`backend/services/hyde.py`, runtime, one LLM call
  ~4 s): drafts a SPECIFIC line (concrete clinical details) and a BROAD line
  (topic-level framing) of the ideal reference text.
- **Matched search** (`backend/api.py`): the specific line joins the leaf
  query through the existing hybrid + MoE machinery (dense, BM25, RRF,
  rerank); the broad line searches the summary tree
  (`backend/services/raptor.py`). Top overview nodes are *prepended* to the
  evidence block, so the model reads the frame first, then the details.
- **MoE stays gated on the raw query** — routing reflects the user's intent,
  not the hypothetical's vocabulary; only evidence-gathering sees HyDE text.

### Why altitude matching (the novel bit)

Classic HyDE embeds one hypothetical against one flat index; classic RAPTOR
searches its collapsed tree with the raw query. Both leave a mismatch: a broad
query's embedding sits far from any leaf, and a specific query wastes budget
on summary nodes. Generating the hypothetical *at the abstraction level of
each index tier* aligns query and corpus geometry on both axes at once — and
the same single LLM call powers both, so the marginal cost of the second
altitude is zero.

### Properties

- Fail-open everywhere: no tree artifacts → flat RAG; Ollama down → raw-query
  retrieval. `HEALIX_HYDE=0` / `HEALIX_RAPTOR=0` for A/B.
- Latency: +~4 s per medical query (one 120-token generation, warm model).
- Evaluation plan: A/B answer quality on broad vs specific question sets;
  recall@k on leaf retrieval with/without the specific hypothetical; ablate
  altitude matching by crossing the lines (broad→leaves, specific→tree).
