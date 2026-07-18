"""HyDE (Hypothetical Document Embeddings) — query-side retrieval upgrade.

At query time one small, fast LLM call drafts what an ideal reference passage
would say, at two abstraction levels:

  specific — one clinical-detail sentence; concatenated with the raw query it
             searches the leaf chunk index (dense + BM25), bridging the gap
             between casual patient phrasing and reference-style corpus text.
  broad    — one sentence naming the overall topic; it searches the RAPTOR
             summary levels (see backend/services/raptor.py), matching the
             abstraction "altitude" of those nodes.

Fail-open by design: any error, timeout, or disabled flag returns {} and
retrieval proceeds with the raw query untouched.
"""

import os
import re
from typing import Dict

_PROMPT = (
    'A patient asked: "{q}"\n\n'
    "Write exactly two lines describing what a medical reference text would "
    "say about this, in neutral encyclopedic prose (no advice, no questions):\n"
    "SPECIFIC: one sentence with the concrete clinical details that answer it.\n"
    "BROAD: one sentence naming the overall medical topic and the themes it involves."
)


def hyde_enabled() -> bool:
    return str(os.getenv("HEALIX_HYDE", "1")).lower() in ("1", "true", "yes")


def generate_hypotheticals(query: str, timeout: float = 12.0) -> Dict[str, str]:
    """Return {"specific": ..., "broad": ...} (either key may be missing)."""
    if not hyde_enabled():
        return {}
    q = (query or "").strip()
    if not q:
        return {}
    try:
        import requests
        url = os.getenv("BAYMAX_OLLAMA_URL", "http://localhost:11434").rstrip("/")
        # HyDE only needs a plausible, topical hypothetical for embedding — not a
        # strong model. Use HEALIX_HYDE_MODEL (e.g. a fast 0.5B) when set; it cuts
        # this blocking pre-retrieval call from ~10s to ~3s. Falls back to the
        # main generation model.
        model = (os.getenv("HEALIX_HYDE_MODEL", "").strip()
                 or os.getenv("BAYMAX_OLLAMA_MODEL", "qwen2.5:7b-instruct"))
        r = requests.post(f"{url}/api/generate", json={
            "model": model,
            "prompt": _PROMPT.format(q=q[:400]),
            "stream": False,
            "keep_alive": os.getenv("BAYMAX_OLLAMA_KEEP_ALIVE", "30m"),
            "options": {"temperature": 0.3, "num_predict": 120, "top_p": 0.9},
        }, timeout=timeout)
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for label, key in (("SPECIFIC", "specific"), ("BROAD", "broad")):
        m = re.search(rf"{label}\s*:\s*(.+)", text, flags=re.IGNORECASE)
        if m:
            line = m.group(1).strip().splitlines()[0].strip()
            if len(line) > 15:
                out[key] = line[:400]
    # Lenient fallback: small/fast models often ignore the labels and just write
    # a plain paragraph. That prose is still a fine leaf-retrieval hypothetical —
    # use it as "specific" so HyDE never silently no-ops. (broad stays empty; the
    # caller falls back to the raw query for the RAPTOR arm.)
    if not out.get("specific"):
        plain = re.sub(r"(?i)\b(specific|broad)\s*:\s*", " ", text)
        # Drop a leading prompt-echo preamble ("A patient ... asks: '...'")
        # that small models sometimes emit before the real hypothetical.
        plain = re.sub(r'(?is)^.*?\basks?\s*:\s*"[^"]*"\s*[.,:;-]*', "", plain)
        plain = re.sub(r"(?i)^\s*(specifically|in general)[,:]?\s*", "", plain)
        plain = " ".join(plain.split())
        if len(plain) > 25:
            out["specific"] = plain[:400]
    return out


if __name__ == "__main__":
    import json
    for q in ("i keep getting dull headaches at work",
              "how does stress affect the body overall"):
        print(q, "->", json.dumps(generate_hypotheticals(q), indent=1))
