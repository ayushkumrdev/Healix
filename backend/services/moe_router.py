#!/usr/bin/env python3
"""
Healix Mixture-of-Medical-Experts (MoE) router  —  agentic, system-level MoE.

Classic MoE routes tokens to expert FFNs inside a transformer. Here we apply the
same sparse-gating idea one level up: a learned-prototype gate routes each query
(and any image modality) to the top-k medical *specialist experts*. Each expert
is a lightweight agent defined by:

  - a semantic prototype (for embedding-based gating),
  - a retrieval "lens" (a query reformulation that pulls specialty-specific
    evidence from the shared FAISS index),
  - keyword priors and an optional imaging modality affinity.

The gate produces sparse weights (softmax over expert scores, top-k kept). The
activated experts each fetch lens-specific evidence (a multi-query retrieval
expansion); the union grounds a single synthesis pass in the LLM, framed as an
integrated specialist panel. Routing weights are returned for explainability.

This is intentionally model-agnostic: it wraps the existing single GGUF model and
FAISS retriever, so it runs locally with no extra weights. See RESEARCH.md for
the full architecture, the agentic verify/refine loop, and the evaluation plan.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Expert:
    name: str
    lens: str            # query reformulation; "{q}" is replaced by the user query
    prototype: str       # text encoded once for gating
    keywords: Tuple[str, ...] = ()
    modality: Optional[str] = None   # imaging affinity: "chest_xray" | "skin"
    is_default: bool = False


EXPERTS: List[Expert] = [
    Expert("Cardiology",
           "cardiac and cardiovascular causes and management of: {q}",
           "heart chest pain pressure palpitations arrhythmia blood pressure "
           "cardiovascular circulation angina",
           ("heart", "chest pain", "palpitation", "blood pressure", "pulse", "cardiac")),
    Expert("Pulmonology",
           "respiratory, lung and breathing aspects of: {q}",
           "lungs breathing shortness of breath cough wheeze pneumonia asthma "
           "chest x-ray respiratory oxygen",
           ("breath", "cough", "lung", "wheeze", "chest x-ray", "pneumonia", "asthma"),
           modality="chest_xray"),
    Expert("Neurology",
           "neurological causes and patterns of: {q}",
           "headache migraine dizziness numbness tingling seizure nerve brain "
           "vertigo cognition",
           ("headache", "migraine", "dizz", "numb", "tingl", "seizure", "vertigo")),
    Expert("Dermatology",
           "skin, lesion and dermatological aspects of: {q}",
           "skin rash lesion mole itch acne eczema dermatitis pigmentation "
           "dermatoscopic",
           ("skin", "rash", "lesion", "mole", "itch", "acne", "eczema"),
           modality="skin"),
    Expert("Gastroenterology",
           "digestive and gastrointestinal aspects of: {q}",
           "stomach nausea acid reflux bowel diarrhea constipation abdominal "
           "digestion gut",
           ("stomach", "nausea", "acid", "reflux", "bowel", "abdomen", "diarrhea")),
    Expert("Pharmacology",
           "medication, dosing, side effects and interactions for: {q}",
           "medication drug dose side effect interaction contraindication "
           "prescription pharmacology",
           ("drug", "medication", "dose", "side effect", "interaction", "pill", "tablet")),
    Expert("Psychophysiology",
           "nervous-system, stress and mind-body aspects of: {q}",
           "stress anxiety sleep fatigue panic nervous system vagal sympathetic "
           "cortisol mood mind body",
           ("stress", "anxiety", "sleep", "fatigue", "panic", "burnout", "mood")),
    Expert("General Internal Medicine",
           "{q}",
           "general health symptoms wellness primary care evaluation",
           (), is_default=True),
]


class ExpertRouter:
    """Embedding-gated sparse router over medical specialist experts."""

    def __init__(self, retriever, top_k: int = 2):
        self.retriever = retriever
        self.top_k = int(os.getenv("HEALIX_MOE_TOPK", str(top_k)))
        self.temperature = float(os.getenv("HEALIX_MOE_TEMP", "0.35"))
        self.kw_boost = 0.18
        self.modality_boost = 0.45
        self.floor = 0.12  # min normalized weight to activate a non-default expert
        self._proto: Optional[np.ndarray] = None

    # ---- embedding helpers --------------------------------------------
    def _encode(self, texts: List[str]) -> np.ndarray:
        if getattr(self.retriever, "model", None) is None:
            self.retriever._load_model()
        embs = self.retriever.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)

    def _prototypes(self) -> np.ndarray:
        if self._proto is None:
            self._proto = self._encode([e.prototype for e in EXPERTS])
        return self._proto

    # ---- gating --------------------------------------------------------
    def route(self, query: str, image_modality: Optional[str] = None
              ) -> Tuple[List[Dict], List[Expert]]:
        """Return (routing[{name,weight}], activated_experts)."""
        q = (query or "").strip() or "general health question"
        qv = self._encode([q])[0]
        sims = self._prototypes() @ qv  # cosine (both normalized)

        ql = q.lower()
        scores = []
        for i, e in enumerate(EXPERTS):
            s = float(sims[i])
            if any(k in ql for k in e.keywords):
                s += self.kw_boost
            if image_modality and e.modality == image_modality:
                s += self.modality_boost
            scores.append(s)

        scores = np.asarray(scores, dtype=np.float32)
        # sparse softmax gating
        z = np.exp((scores - scores.max()) / max(1e-6, self.temperature))
        weights = z / z.sum()

        order = list(np.argsort(weights)[::-1])
        chosen = [i for i in order[: self.top_k] if weights[i] >= self.floor]
        if not chosen:
            # fall back to the default/general expert
            default_idx = next((i for i, e in enumerate(EXPERTS) if e.is_default), order[0])
            chosen = [default_idx]
        # If imaging is present, guarantee the matching modality expert is active.
        if image_modality:
            for i, e in enumerate(EXPERTS):
                if e.modality == image_modality and i not in chosen:
                    chosen = [i] + chosen[: self.top_k - 1]

        w = np.array([weights[i] for i in chosen], dtype=np.float32)
        w = w / w.sum()
        routing = [{"name": EXPERTS[i].name, "weight": round(float(wi), 3)}
                   for i, wi in zip(chosen, w)]
        activated = [EXPERTS[i] for i in chosen]
        return routing, activated

    # ---- agentic evidence gathering -----------------------------------
    def gather_evidence(self, query: str, experts: List[Expert],
                        per_expert: int = 4, max_total: int = 12) -> List[Dict]:
        """Multi-query retrieval: each expert pulls evidence through its lens."""
        use_hybrid = str(os.getenv("HEALIX_HYBRID", "1")).lower() in ("1", "true", "yes")
        seen = set()
        pooled: List[Dict] = []
        for e in experts:
            subq = e.lens.replace("{q}", query)
            try:
                if use_hybrid and hasattr(self.retriever, "hybrid_retrieve"):
                    hits = self.retriever.hybrid_retrieve(query=subq, k=per_expert)
                else:
                    hits = self.retriever.retrieve(query=subq, k=per_expert, min_score=0.2)
            except Exception:
                hits = []
            for h in hits:
                key = (h.get("chunk_id") or h.get("text", ""))[:120]
                if key in seen:
                    continue
                seen.add(key)
                h = dict(h)
                h["_expert"] = e.name
                pooled.append(h)
        pooled.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return pooled[:max_total]

    # ---- prompt directive ---------------------------------------------
    @staticmethod
    def directive(routing: List[Dict]) -> str:
        if not routing:
            return ""
        parts = ", ".join(f"{r['name']} ({r['weight']:.0%})" for r in routing)
        return (
            "Specialist perspectives to integrate, by relevance: " + parts + ". "
            "Reason from these angles and merge them into one calm, unified "
            "explanation in your own voice. Do not name the specialties, address a "
            "panel, or mention routing."
        )
