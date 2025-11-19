#!/usr/bin/env python3
"""
Centralized configuration for Healix services.
These defaults are imported by services to keep behavior consistent.
Override via environment variables where exposed.
"""

import os

# Retrieval defaults
RAG_DEFAULT_K: int = int(os.getenv("BAYMAX_RAG_K", "5"))
RAG_MIN_SCORE: float = float(os.getenv("BAYMAX_RAG_MIN_SCORE", "0.3"))

# Generation defaults (GPT4All)
GEN_MAX_TOKENS: int = int(os.getenv("BAYMAX_GEN_TOKENS") or os.getenv("BAYMAX_GEN_MAX_TOKENS", "500"))
GEN_MIN_TOKENS: int = int(os.getenv("BAYMAX_GEN_MIN_TOKENS", "30"))
GEN_TEMP: float = float(os.getenv("BAYMAX_GEN_TEMP", "0.25"))
GEN_TOP_K: int = int(os.getenv("BAYMAX_GEN_TOP_K", "40"))
GEN_TOP_P: float = float(os.getenv("BAYMAX_GEN_TOP_P", "0.9"))
# Legacy/deprecated (for backward compat)
GEN_SHORT_TOKENS: int = int(os.getenv("BAYMAX_GEN_SHORT_TOKENS", "80"))

# Retrieval behavior
RAG_STRICT: bool = str(os.getenv("BAYMAX_RAG_STRICT", "0")).lower() in ("1", "true", "yes")

