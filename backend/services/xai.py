#!/usr/bin/env python3
"""
Explainability (XAI) utilities for Healix backend.
Provides evidence highlighting, reasoning outline extraction, confidence scoring,
counterfactual suggestions, and a lightweight knowledge graph from symptoms → conditions → treatments.

All functions are local and deterministic (no external calls, no extra model calls).
"""

from typing import List, Dict, Any, Tuple
import re
import math

# Minimal English stopword list for keyword extraction
_STOPWORDS = set(
    """
a an and are as at be by for from has have he her hers him his i in is it its of on or our she that the their them they this to was were will with you your yours we us about into over under then than not no yes if else when while more most less few one two three can may should would could might just also only even very like such than so because due via per 
    """.split()
)

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()

def extract_keywords(text: str, topn: int = 20) -> List[str]:
    """Extract simple keywords from text by token filtering.
    Not a full NLP pipeline; intended for lightweight alignment and confidence.
    """
    t = _normalize_text(text)
    # Keep alphanumeric words 3+ chars
    toks = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", t)
    # Filter stopwords and numeric-like tokens
    keys = []
    for tok in toks:
        if tok in _STOPWORDS:
            continue
        if tok.isdigit():
            continue
        keys.append(tok)
    # Frequency sort
    freq = {}
    for k in keys:
        freq[k] = freq.get(k, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:topn]]

def build_evidence_alignment(passages: List[Dict[str, Any]], response_text: str) -> List[Dict[str, Any]]:
    """Align key terms from the response text to retrieved passages.
    Returns a list of evidence entries per passage:
    [{ passage_index, source, url, matched_keywords: [...], support_score: 0..1 }]
    """
    if not passages:
        return []
    keys = extract_keywords(response_text, topn=25)
    if not keys:
        return []
    results = []
    for i, p in enumerate(passages):
        text = _normalize_text(p.get("text", ""))
        matched = [k for k in keys if k in text]
        support = len(matched) / max(1, len(keys))
        results.append({
            "passage_index": i,
            "source": p.get("source"),
            "url": p.get("url"),
            "matched_keywords": matched[:15],
            "support_score": round(float(support), 3)
        })
    # Sort by support score desc
    results.sort(key=lambda x: x["support_score"], reverse=True)
    return results[:5]

def build_reasoning_outline(response_text: str) -> List[str]:
    """Extract a short reasoning outline from the response text using heuristics.
    We prefer sentences containing clinical markers (likely, consider, recommend, because).
    """
    text = (response_text or "").strip()
    if not text:
        return []
    # Split into sentences (lightweight)
    sents = re.split(r"(?<=[.!?])\s+", text)
    priorities = []
    for s in sents:
        s_l = s.lower()
        weight = 0
        for kw in ("likely", "consider", "recommend", "because", "suggest", "red flag", "warning", "follow-up"):
            if kw in s_l:
                weight += 1
        if len(s) > 12:
            priorities.append((weight, s.strip()))
    # Take top 4-6 sentences by weight then by order
    priorities.sort(key=lambda x: (-x[0]))
    outline = [s for w, s in priorities[:6] if w > 0]
    if not outline:
        outline = sents[:4]
    return [o.strip() for o in outline if o.strip()]

def compute_confidence(passages: List[Dict[str, Any]], response_text: str, emergency_level: str = None) -> float:
    """Compute a conservative confidence score 0..1 based on evidence support and scenario.
    - Evidence support: fraction of response keywords present in passages.
    - Emergency adjustment: reduce confidence in emergencies to avoid overconfidence.
    """
    keys = extract_keywords(response_text, topn=20)
    if not keys:
        return 0.5
    # Support: union of matches across passages
    seen = set()
    union_text = " ".join(_normalize_text(p.get("text", "")) for p in passages)
    for k in keys:
        if k in union_text:
            seen.add(k)
    coverage = len(seen) / max(1, len(keys))
    # Map coverage to base confidence [0.45, 0.9]
    base = 0.45 + 0.45 * coverage
    # Emergency adjustment
    if emergency_level in ("CRITICAL", "HIGH"):
        base -= 0.1
    return round(max(0.3, min(0.92, base)), 3)

def generate_counterfactuals(symptoms: Dict[str, Any], response_text: str) -> List[str]:
    """Generate 2-4 counterfactual what-if statements using simple clinical heuristics."""
    cfs: List[str] = []
    text_l = (response_text or "").lower()
    has_fever = False
    has_chest = False
    has_breath = False
    if symptoms and isinstance(symptoms.get("symptoms"), list):
        for ent in symptoms["symptoms"]:
            t = (ent.get("text") or "").lower()
            if "fever" in t:
                has_fever = True
            if "chest" in t:
                has_chest = True
            if "breath" in t or "shortness of breath" in t:
                has_breath = True
    # Heuristic counterfactuals
    if has_fever or "fever" in text_l:
        cfs.append("If the fever persisted beyond 72 hours or exceeded 39.5°C (103.1°F), escalation to in-person evaluation would be indicated.")
    if has_chest or "chest pain" in text_l:
        cfs.append("If chest pain were associated with shortness of breath or exertion, the risk profile would increase and urgent assessment would be recommended.")
    if has_breath or "shortness of breath" in text_l:
        cfs.append("If shortness of breath worsened when lying flat or was associated with leg swelling, cardiac causes would move higher on the differential.")
    # Generic safety counterfactuals
    cfs.append("If new red-flag symptoms (e.g., fainting, severe headache, neurological deficits) appeared, the recommended action would shift to urgent/emergency care.")
    # Cap at 4 items
    return cfs[:4]

def build_knowledge_graph(symptoms: Dict[str, Any], passages: List[Dict[str, Any]], response_text: str) -> Dict[str, Any]:
    """Construct a lightweight knowledge graph: symptoms → conditions → treatments.
    - Symptom nodes from symptom extractor
    - Condition nodes inferred from response text keywords and top passages
    - Treatment nodes inferred from passages in category 'Medications'
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Symptom nodes
    symptom_ids = {}
    if symptoms and isinstance(symptoms.get("symptoms"), list):
        for i, ent in enumerate(symptoms["symptoms"][:10]):
            s_text = ent.get("text") or "symptom"
            sid = f"sym_{i}"
            symptom_ids[s_text.lower()] = sid
            nodes.append({"id": sid, "label": s_text, "type": "symptom"})

    # Condition candidates from response text (naive)
    # Capture words after 'likely', 'possible', 'consider'
    cond_candidates = set()
    for m in re.finditer(r"(?:likely|possible|consider)\s+([a-zA-Z][a-zA-Z\s\-]{2,30})", response_text or "", flags=re.IGNORECASE):
        cand = m.group(1).strip().rstrip(".,;:)")
        if cand and len(cand) <= 40:
            cond_candidates.add(cand.title())
    # Also glean common conditions from passages (first 3)
    for p in passages[:3]:
        txt = p.get("text", "")
        for m in re.finditer(r"\b([A-Z][a-z]+\s(?:[A-Z][a-z]+|[a-z]+){0,2})\b", txt):
            phrase = m.group(1).strip()
            if 3 <= len(phrase) <= 32:
                cond_candidates.add(phrase)
    cond_ids = {}
    for j, c in enumerate(list(cond_candidates)[:8]):
        cid = f"cond_{j}"
        cond_ids[c.lower()] = cid
        nodes.append({"id": cid, "label": c, "type": "condition"})

    # Treatment nodes from medications passages
    treat_candidates = set()
    for p in passages:
        if (p.get("category") or "").lower() == "medications":
            # Extract drug names (TitleCase tokens)
            for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})?)\b", p.get("text", "")):
                name = m.group(1).strip()
                if 3 <= len(name) <= 30:
                    treat_candidates.add(name)
    treat_ids = {}
    for k, t in enumerate(list(treat_candidates)[:8]):
        tid = f"treat_{k}"
        treat_ids[t.lower()] = tid
        nodes.append({"id": tid, "label": t, "type": "treatment"})

    # Edges: symptom→condition, condition→treatment (heuristic: connect all-to-all within caps)
    for s_txt, sid in list(symptom_ids.items())[:5]:
        for c_txt, cid in list(cond_ids.items())[:5]:
            edges.append({"source": sid, "target": cid, "label": "suggests"})
    for c_txt, cid in list(cond_ids.items())[:5]:
        for t_txt, tid in list(treat_ids.items())[:5]:
            edges.append({"source": cid, "target": tid, "label": "managed_by"})

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "symptoms": len(symptom_ids),
            "conditions": len(cond_ids),
            "treatments": len(treat_ids)
        }
    }

def build_xai_package(user_text: str,
                       response_text: str,
                       passages: List[Dict[str, Any]],
                       symptoms: Dict[str, Any],
                       emergency_level: str = None) -> Dict[str, Any]:
    """Assemble the full XAI package for a response."""
    evidence = build_evidence_alignment(passages, response_text)
    outline = build_reasoning_outline(response_text)
    conf = compute_confidence(passages, response_text, emergency_level)
    counter = generate_counterfactuals(symptoms, response_text)
    kg = build_knowledge_graph(symptoms, passages, response_text)
    return {
        "evidence": evidence,
        "reasoning_outline": outline,
        "confidence": conf,
        "counterfactuals": counter,
        "knowledge_graph": kg
    }

