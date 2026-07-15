#!/usr/bin/env python3
"""
Advanced LLM orchestrator for ChatGPT-level Healthcare Super-Assistant
Uses larger, more capable models for sophisticated medical reasoning
"""

import json
import re
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
try:
    import gpt4all  # optional; only used for the local GGUF fallback backend
except Exception:
    gpt4all = None
# Load .env for direct Python runs (optional)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
# Optional llama.cpp backend (CUDA via cuBLAS)
try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # llama-cpp-python may be missing
    Llama = None  # type: ignore
from datetime import datetime
import logging
import os
import threading
from .xai import build_xai_package
from .config import GEN_MAX_TOKENS, GEN_TEMP, GEN_TOP_K, GEN_TOP_P, GEN_SHORT_TOKENS

# Optional divider inside the composed prompt: text above it is sent to the
# LLM as the system message (stronger instruction adherence), text below as
# the user turn. Backends that have no system slot just remove the line.
PROMPT_SPLIT_MARKER = "<<<USER TURN>>>"
from .retriever import create_retriever
from .symptom_extractor import create_symptom_extractor

# Module-level GPT4All singleton to avoid reloading per-orchestrator
_GPT4ALL_SINGLETON = None
_GPT4ALL_SINGLETON_LOCK = threading.Lock()

# Module-level retriever and symptom extractor singletons
_RETRIEVER_SINGLETON = None
_RETRIEVER_LOCK = threading.Lock()
_SYMPTOM_SINGLETON = None
_SYMPTOM_LOCK = threading.Lock()

def _get_retriever_singleton():
    global _RETRIEVER_SINGLETON
    with _RETRIEVER_LOCK:
        if _RETRIEVER_SINGLETON is None:
            _RETRIEVER_SINGLETON = create_retriever()
        return _RETRIEVER_SINGLETON

def _get_symptom_singleton():
    global _SYMPTOM_SINGLETON
    with _SYMPTOM_LOCK:
        if _SYMPTOM_SINGLETON is None:
            _SYMPTOM_SINGLETON = create_symptom_extractor()
        return _SYMPTOM_SINGLETON

def _get_llm_singleton(model_name: str, model_path: Optional[str], threads: int):
    """Return a module-wide GPT4All instance initialized once."""
    global _GPT4ALL_SINGLETON
    with _GPT4ALL_SINGLETON_LOCK:
        if _GPT4ALL_SINGLETON is None:
            if model_path:
                _GPT4ALL_SINGLETON = gpt4all.GPT4All(
                    model_name,
                    model_path=model_path,
                    allow_download=False,
                    n_threads=threads,
                    verbose=False
                )
            else:
                _GPT4ALL_SINGLETON = gpt4all.GPT4All(
                    model_name,
                    allow_download=False,
                    n_threads=threads,
                    verbose=False
                )
        return _GPT4ALL_SINGLETON


class AdvancedMedicalOrchestrator:
    """
    Advanced medical LLM orchestrator with ChatGPT-level reasoning capabilities
    Uses larger models and sophisticated prompting for medical expertise
    """
    
    def __init__(self, 
                 model_name: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                 model_path: Optional[str] = "artifacts",
                 enable_advanced_reasoning: bool = True,
                 enable_xai: bool = True):
        """
        Initialize the advanced orchestrator
        
        Args:
            model_name: Name of the GPT4All model to use (larger models for better reasoning)
            model_path: Optional path to model file
            enable_advanced_reasoning: Enable multi-step reasoning capabilities
        """
        # Allow environment variable overrides without introducing a secondary fallback system
        env_model_name = os.getenv("BAYMAX_GGUF_MODEL_NAME")
        env_model_dir = os.getenv("BAYMAX_GGUF_MODEL_DIR")
        
        self.model_name = env_model_name or model_name
        self.model_path = env_model_dir or model_path
        self.model = None
        self.backend = "gpt4all"  # or "llama_cpp"
        self._llama = None
        # Optional Ollama backend: when BAYMAX_LLM_BACKEND=ollama we call the local
        # Ollama server and skip loading the GGUF entirely (faster, stronger models).
        self.llm_backend = (os.getenv("BAYMAX_LLM_BACKEND", "") or "").strip().lower()
        self.ollama_url = os.getenv("BAYMAX_OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("BAYMAX_OLLAMA_MODEL", "llama3:latest")
        self.enable_advanced_reasoning = enable_advanced_reasoning
        # Allow env to disable XAI for latency: BAYMAX_ENABLE_XAI=0/false
        env_xai = os.getenv("BAYMAX_ENABLE_XAI")
        if env_xai is not None and str(env_xai).lower() in ("0", "false", "no"):
            enable_xai = False
        self.enable_xai = enable_xai
        
        # Single main prompt template (can be overridden via env/file)
        self.main_prompt_template = self._load_main_prompt_template()
        
        # Advanced medical knowledge patterns
        self.emergency_keywords = self._load_comprehensive_emergency_keywords()
        self.medical_specialties = self._load_medical_specialties()
        self.drug_interaction_flags = self._load_drug_interaction_flags()
        
        # Conversation state for multi-turn reasoning
        self.conversation_history = []
        self.current_differential = []
        
        # Concurrency guard for model generation
        self._gen_lock = threading.Lock()
        
        # Defer heavy model load until first use to speed UI startup
        # self._initialize_advanced_model()
        
        print(f"Advanced Medical Orchestrator initialized with {self.model_name}")
        
    def _ensure_model_loaded(self):
        """Load the local model lazily on first use (skipped for the Ollama backend)."""
        if self.llm_backend == "ollama":
            return  # Ollama runs as a server; nothing to load locally
        if self.model is None:
            self._initialize_advanced_model()
    
    def _initialize_advanced_model(self):
        """Initialize the medical AI model with performance optimization"""
        print(f"Loading medical AI model: {self.model_name}")
        print("Optimizing for fast responses...")
        
        try:
            threads_env = os.getenv("BAYMAX_GEN_THREADS") or os.getenv("BAYMAX_GPT4ALL_THREADS")
            default_threads = str(max(2, min(os.cpu_count() or 4, 16)))
            threads = int(threads_env) if threads_env else int(default_threads)

            # Prefer llama.cpp GPU backend if requested and available
            use_llama = str(os.getenv("BAYMAX_USE_LLAMA_CPP", os.getenv("BAYMAX_LLM_BACKEND", ""))).lower() in ("1", "true", "yes", "llama_cpp")
            if use_llama and Llama is not None:
                model_file = self.model_name
                if self.model_path and not os.path.isabs(self.model_name):
                    model_file = os.path.join(self.model_path, self.model_name)
                n_gpu_layers = int(os.getenv("BAYMAX_GPU_LAYERS", "99999") or 99999)
                try:
                    self._llama = Llama(
                        model_path=model_file,
                        n_threads=threads,
                        n_gpu_layers=n_gpu_layers,
                        seed=-1,
                        logits_all=False,
                        embedding=False,
                        n_ctx=int(os.getenv("BAYMAX_GEN_CONTEXT", "4096") or 4096),
                        use_mmap=True,
                        use_mlock=False
                    )
                    self.backend = "llama_cpp"
                    print("Medical AI model loaded successfully (llama.cpp, GPU)")
                    print(f"Model: {self.model_name} | n_gpu_layers={n_gpu_layers}")
                    return
                except Exception as e:
                    print(f"llama.cpp GPU backend failed: {e}; falling back to GPT4All")
                    self._llama = None
                    self.backend = "gpt4all"

            # Default GPT4All backend (CPU unless cuBLAS DLLs available)
            self.model = _get_llm_singleton(self.model_name, self.model_path, threads)
            print("Medical AI model loaded successfully (GPT4All)")
            print(f"Model: {self.model_name}")
            print(f"Performance: Optimized for fast medical responses")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Tip: Set BAYMAX_GGUF_MODEL_NAME and BAYMAX_GGUF_MODEL_DIR env vars for local models")
            raise
    
    def _load_comprehensive_emergency_keywords(self) -> Dict[str, List[str]]:
        """Load comprehensive emergency detection patterns"""
        return {
            "cardiac": [
                "chest pain", "heart attack", "cardiac arrest", "palpitations",
                "irregular heartbeat", "heart pounding", "chest pressure",
                "crushing chest pain", "radiating pain", "left arm pain",
                "jaw pain", "myocardial infarction", "angina", "heart racing"
            ],
            "respiratory": [
                "difficulty breathing", "shortness of breath", "can't breathe",
                "wheezing", "gasping", "choking", "respiratory distress",
                "blue lips", "cyanosis", "oxygen", "ventilator", "intubation",
                "pulmonary embolism", "asthma attack", "pneumonia"
            ],
            "neurological": [
                "stroke", "seizure", "unconscious", "unresponsive", "paralysis",
                "severe headache", "confusion", "memory loss", "slurred speech",
                "face drooping", "arm weakness", "sudden numbness",
                "brain injury", "concussion", "meningitis", "encephalitis"
            ],
            "trauma": [
                "severe bleeding", "heavy bleeding", "major injury", "trauma",
                "broken bone", "head injury", "burns", "motorcycle accident",
                "car crash", "fall from height", "gunshot", "stab wound",
                "hemorrhage", "internal bleeding", "fracture", "spinal injury"
            ],
            "allergic": [
                "anaphylaxis", "severe allergic reaction", "swelling",
                "hives", "difficulty swallowing", "throat closing",
                "epinephrine", "epipen", "allergic shock", "food allergy",
                "drug allergy", "bee sting", "severe itching"
            ],
            "mental_health": [
                "suicide", "self harm", "want to die", "end it all",
                "kill myself", "suicidal thoughts", "overdose", "poison",
                "mental health crisis", "psychosis", "hallucinations"
            ],
            "metabolic": [
                "diabetic coma", "ketoacidosis", "severe hypoglycemia",
                "blood sugar", "insulin shock", "diabetic emergency",
                "severe dehydration", "electrolyte imbalance"
            ],
            "obstetric": [
                "severe pregnancy complications", "preeclampsia", "eclampsia",
                "pregnancy bleeding", "premature labor", "miscarriage",
                "ectopic pregnancy", "placental abruption"
            ]
        }
    
    def _load_medical_specialties(self) -> Dict[str, List[str]]:
        """Load medical specialty keywords for routing"""
        return {
            "cardiology": ["heart", "cardiac", "cardio", "coronary", "artery", "blood pressure"],
            "pulmonology": ["lung", "respiratory", "breathing", "asthma", "copd", "pneumonia"],
            "neurology": ["brain", "neurological", "seizure", "stroke", "headache", "migraine"],
            "gastroenterology": ["stomach", "digestive", "intestinal", "liver", "pancreas", "bowel"],
            "endocrinology": ["diabetes", "thyroid", "hormone", "insulin", "glucose", "endocrine"],
            "infectious_disease": ["infection", "virus", "bacteria", "antibiotic", "fever", "sepsis"],
            "oncology": ["cancer", "tumor", "malignant", "chemotherapy", "radiation", "oncology"],
            "orthopedics": ["bone", "joint", "fracture", "orthopedic", "muscle", "ligament"],
            "dermatology": ["skin", "rash", "dermatitis", "eczema", "psoriasis", "melanoma"],
            "psychiatry": ["mental", "psychiatric", "depression", "anxiety", "bipolar", "schizophrenia"]
        }
    
    def _load_drug_interaction_flags(self) -> List[str]:
        """Load drug interaction warning flags"""
        return [
            "warfarin", "coumadin", "blood thinner", "anticoagulant",
            "lithium", "digoxin", "phenytoin", "carbamazepine",
            "monoamine oxidase inhibitor", "maoi", "ssri", "antidepressant",
            "multiple medications", "drug interaction", "pharmacy check"
        ]
    
    def _create_advanced_medical_prompt(self, query_type: str = "general") -> str:
        """Minimal default directive used only if no external main prompt template is provided.
        Keep this short to avoid multiple embedded prompts in code.
        """
        return ("You are Healix, a calm medical companion. Answer only health questions. "
                "Begin directly with the answer — never introduce yourself or greet. "
                "Use clear markdown: short paragraphs, and numbered lists with bold lead-ins "
                "for plans or multi-part guidance. No exclamation marks, no emojis, no disclaimers.")
    
    def _detect_advanced_emergency(self, text: str) -> Dict[str, Any]:
        """Advanced emergency detection with severity assessment"""
        text_lower = text.lower()
        emergency_matches = {}
        severity_score = 0
        
        # Check each emergency category
        negations = {"no", "not", "denies", "denied", "without"}
        
        for category, keywords in self.emergency_keywords.items():
            matches = []
            for keyword in keywords:
                kw = keyword.lower()
                start = 0
                while True:
                    idx = text_lower.find(kw, start)
                    if idx == -1:
                        break
                    # Simple negation window: last ~10 characters before match
                    window = text_lower[max(0, idx-10):idx]
                    window_tokens = window.split()
                    negated = any(tok in negations for tok in window_tokens[-3:])
                    if not negated:
                        matches.append(keyword)
                        # Weight severity based on keyword criticality
                        if keyword in ["cardiac arrest", "respiratory distress", "unconscious", 
                                      "severe bleeding", "anaphylaxis", "stroke"]:
                            severity_score += 10
                        elif keyword in ["chest pain", "difficulty breathing", "seizure"]:
                            severity_score += 7
                        else:
                            severity_score += 3
                    start = idx + len(kw)
            
            if matches:
                emergency_matches[category] = matches
        
        is_emergency = len(emergency_matches) > 0
        
        # Determine emergency level
        if severity_score >= 10:
            emergency_level = "CRITICAL"
        elif severity_score >= 7:
            emergency_level = "HIGH"
        elif severity_score >= 3:
            emergency_level = "MODERATE"
        else:
            emergency_level = "LOW"
        
        return {
            "is_emergency": is_emergency,
            "emergency_level": emergency_level,
            "severity_score": severity_score,
            "categories_detected": emergency_matches,
            "emergency_message": self._get_advanced_emergency_message(emergency_level, emergency_matches) if is_emergency else None
        }
    
    def _get_advanced_emergency_message(self, level: str, categories: Dict) -> str:
        """Return a neutral placeholder (we do not output urgent directives)."""
        return ""
    
    def _identify_medical_specialty(self, text: str) -> List[str]:
        """Identify relevant medical specialties for the query"""
        text_lower = text.lower()
        relevant_specialties = []
        
        for specialty, keywords in self.medical_specialties.items():
            if any(keyword in text_lower for keyword in keywords):
                relevant_specialties.append(specialty)
        
        return relevant_specialties
    
    def _is_greeting_or_smalltalk(self, text: str) -> bool:
        """Detect simple greetings or small talk that should elicit a professional intake greeting."""
        t = (text or "").strip().lower()
        if len(t) <= 2:
            return True
        patterns = [
            r"^hi$", r"^hello$", r"^hey$", r"^yo$", r"^how\s+are\s+you\??$",
            r"^good\s+(morning|afternoon|evening)$", r"^what's\s+up\??$", r"^sup\??$",
        ]
        for p in patterns:
            if re.match(p, t):
                return True
        # Also treat very short, non-medical inputs as greetings
        if len(t.split()) <= 3 and not any(k in t for k in [
            "pain","fever","cough","dose","medication","diagnosis","symptom","rash","bleeding","breath","chest","headache","nausea","vomit","diarrhea"
        ]):
            return True
        return False

    def _build_professional_greeting_response(self, user_text: str) -> Dict[str, Any]:
        """Return a single natural, human-sounding greeting line per GREETING VARIATION RULE (no emojis)."""
        t = (user_text or "").strip().lower()
        # Tone heuristics
        is_playful = any(p in t for p in ["what's up", "whats up", "sup", "wassup", "yo"]) or ("?" in t and "up" in t)
        is_friendly = any(p in t for p in ["hi there", "hello there", "good morning", "good afternoon", "good evening"]) or ("!" in t and len(t.split()) <= 4)
        is_tired = any(p in t for p in ["...", "…", "ugh", "sigh", "not great", "meh", "tired", "exhausted", "drained", "rough"]) or t.endswith("...")

        if is_playful:
            msg = "Hey. What's on your mind?"
        elif is_tired:
            msg = "Hey. How are you holding up?"
        elif is_friendly:
            msg = "Hello. How are you doing?"
        else:
            msg = "Hi, I'm here."

        response_structure = {
            "medical_assessment": msg,
            "recommendations": [],
            "conditions": [],
            "patient_education": "",
            "confidence_level": 0.9
        }
        emergency_check = {"is_emergency": False, "emergency_level": "LOW", "severity_score": 0, "categories_detected": {}}
        return {
            "emergency": False,
            "response": response_structure,
            "specialties": [],
            "query_type": "general",
            "passages_used": 0,
            "reasoning_quality": "fast",
            "model_used": self.model_name,
            "emergency_assessment": emergency_check
        }

    def _is_identity_query(self, text: str) -> bool:
        """Detect identity questions like 'who are you?' or 'what are you?'"""
        t = (text or "").strip().lower()
        if not t:
            return False
        patterns = [
            r"^who\s+are\s+you\??$",
            r"^what\s+are\s+you\??$",
            r"^who\s+is\s+healix\??$",
            r"^what\s+is\s+healix\??$",
            r"^explain\s+yourself\.?$",
            r"^tell\s+me\s+about\s+you\.?$",
            r"^tell\s+me\s+about\s+yourself\.?$",
            r"^describe\s+yourself\.?$",
        ]
        for p in patterns:
            try:
                if re.match(p, t):
                    return True
            except Exception:
                continue
        # Heuristic keywords
        if any(kw in t for kw in ["who are you", "what are you", "explain yourself", "about yourself", "about you", "what is healix", "who is healix"]):
            return True
        return False

    def _build_identity_response(self) -> Dict[str, Any]:
        """Return a minimal identity one-liner."""
        msg = "I'm Healix, a medical companion here to help you understand your health."
        return {
            "emergency": False,
            "response": {
                "medical_assessment": msg,
                "recommendations": [],
                "conditions": [],
                "patient_education": "",
                "confidence_level": 0.98,
            },
            "specialties": [],
            "query_type": "identity",
            "passages_used": 0,
            "reasoning_quality": "fast",
            "model_used": self.model_name,
            "emergency_assessment": {"is_emergency": False, "emergency_level": "LOW", "severity_score": 0, "categories_detected": {}}
        }

    def _is_healthcare_topic(self, text: str) -> bool:
        """Heuristic check: true if the query is within healthcare domain."""
        t = (text or "").lower()
        medical_terms = [
            "pain","fever","cough","dose","medication","medicine","drug","diagnosis","symptom","rash","bleeding",
            "breath","chest","headache","nausea","vomit","diarrhea","clinic","doctor","hospital","injury","infection",
            "blood","pressure","glucose","heart","lung","kidney","liver","thyroid","anxiety","depression","sleep",
            "nutrition","diet","exercise","vaccine","allergy","side effect","contraindication","therapy","treatment"
        ]
        non_med_markers = [
            "code","program","compile","deploy","finance","stock","crypto","tax","legal","contract","movie","music",
            "recipe","travel","hotel","flight","game","sports","weather","news","politics","math","physics","chemistry"
        ]
        has_med = any(k in t for k in medical_terms)
        has_non_med = any(k in t for k in non_med_markers)
        # If it clearly looks non-medical and lacks medical terms, treat as out-of-domain
        return has_med or (not has_non_med)

    def _build_non_medical_refusal_response(self, user_text: str) -> Dict[str, Any]:
        """Politely decline non-medical queries while keeping schema consistent."""
        msg = (
            "I’m focused on healthcare topics. Please ask a health-related question — for example, symptoms, medications, lab results, "
            "prevention, sleep, nutrition, or general wellness."
        )
        return {
            "emergency": False,
            "response": {
                "medical_assessment": msg,
                "recommendations": [],
                "conditions": [],
                "patient_education": "",
                "confidence_level": 0.95
            },
            "specialties": [],
            "query_type": "general",
            "passages_used": 0,
            "reasoning_quality": "fast",
            "model_used": self.model_name,
            "emergency_assessment": {"is_emergency": False, "emergency_level": "LOW", "severity_score": 0, "categories_detected": {}}
        }

    def synthesize_advanced_response(self,
                                   user_text: str,
                                   retrieved_passages: List[Dict],
                                   symptom_data: Optional[Dict] = None,
                                   conversation_context: Optional[List[Dict]] = None,
                                   user_mode: str = "patient",
                                   max_tokens_override: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate advanced medical response with ChatGPT-level reasoning
        
        Args:
            user_text: User's medical query
            retrieved_passages: Relevant medical literature
            symptom_data: Extracted symptom information
            conversation_context: Previous conversation for multi-turn reasoning
            user_mode: patient or clinician mode for response complexity
            
        Returns:
            Comprehensive medical analysis and recommendations
        """
        # Step 1: Advanced emergency detection
        emergency_check = self._detect_advanced_emergency(user_text)
        # Do not short-circuit on emergencies; continue with structured reasoning per policy.
        
        # Step 2: Identify medical specialties and query type
        # If generate-all is disabled and input is small talk, short-circuit with a professional intake greeting.
        generate_all = str(os.getenv("BAYMAX_GENERATE_ALL", "0")).lower() in ("1", "true", "yes")
        if (not generate_all) and self._is_greeting_or_smalltalk(user_text):
            return self._build_professional_greeting_response(user_text)

        # Identity response rule (minimal one-liner)
        if self._is_identity_query(user_text):
            return self._build_identity_response()

        # Domain rule: politely decline clearly non-medical queries.
        if not self._is_healthcare_topic(user_text):
            return self._build_non_medical_refusal_response(user_text)

        specialties = self._identify_medical_specialty(user_text)
        query_type = self._classify_query_type(user_text, symptom_data)
        
        # Step 3: Prepare comprehensive medical context
        safe_passages = self._convert_to_json_safe(retrieved_passages)
        safe_symptoms = self._convert_to_json_safe(symptom_data) if symptom_data else None
        
        # Step 4: Compose single main prompt (use summarization if available)
        summarized_context = ""
        try:
            if safe_passages:
                summarized_context = self._summarize_evidence(user_text, safe_passages)
        except Exception:
            summarized_context = ""
        conv_mem = self._build_conversation_memory_text(conversation_context, current_user_text=user_text)
        full_prompt = self._compose_main_prompt(
            user_text=user_text,
            passages=safe_passages,
            symptoms=safe_symptoms,
            summarized_context=summarized_context or None,
            conversation_memory=conv_mem or "",
        )
        
        # Step 5: Generate advanced medical response
        try:
            # Ensure model is loaded before generation
            self._ensure_model_loaded()
            print(f"Generating advanced medical response using {self.model_name}")
            print(f"Query type: {query_type}")
            print(f"Relevant specialties: {', '.join(specialties) if specialties else 'General Medicine'}")
            
            # Extreme speed optimization + style tuning
            # Dynamic fast-mode tuning for latency; allow 'abstractive' style to raise temperature
            fast_mode = str(os.getenv("BAYMAX_FAST_MODE", "1")).lower() in ("1", "true", "yes")
            style = str(os.getenv("BAYMAX_GEN_STYLE", "")).lower()  # e.g., 'abstractive'
            dyn_max_tokens = GEN_MAX_TOKENS
            if isinstance(max_tokens_override, int) and max_tokens_override > 0:
                dyn_max_tokens = min(dyn_max_tokens, max_tokens_override)
            dyn_temp = GEN_TEMP
            dyn_top_k = GEN_TOP_K
            dyn_top_p = GEN_TOP_P
            try:
                if fast_mode:
                    # Short inputs or general info -> fewer tokens for speed
                    if len(user_text) <= 120:
                        dyn_max_tokens = min(GEN_MAX_TOKENS, GEN_SHORT_TOKENS)
                    else:
                        dyn_max_tokens = min(GEN_MAX_TOKENS, int(GEN_MAX_TOKENS * 0.75))
                    if style == 'abstractive':
                        # Encourage paraphrasing/abstraction
                        dyn_temp = max(GEN_TEMP, 0.8)
                        dyn_top_k = max(20, min(GEN_TOP_K, 40))
                        dyn_top_p = max(0.9, GEN_TOP_P)
                    else:
                        dyn_temp = min(GEN_TEMP, 0.2)
                        dyn_top_k = min(GEN_TOP_K, 20)
                        dyn_top_p = min(GEN_TOP_P, 0.90)
            except Exception:
                pass

            with self._gen_lock:
                raw_response = self._llm_generate_text(
                    full_prompt,
                    max_tokens=dyn_max_tokens,
                    temp=dyn_temp,
                    repeat_penalty=1.18,
                    top_k=dyn_top_k,
                    top_p=dyn_top_p
                )
            
            # Step 7: Clean up response to extract conversational text
            cleaned_response = self._extract_conversational_text(raw_response)
            
            # Optional abstractive post-pass to enforce paraphrasing
            try:
                post_abstract = str(os.getenv("BAYMAX_POST_ABSTRACT", "0")).lower() in ("1", "true", "yes")
            except Exception:
                post_abstract = False
            text_for_output = (cleaned_response or "").strip()
            if post_abstract and len(text_for_output) > 20:
                try:
                    rewrite_prompt = (
                        "Rewrite the following medical answer in your own words while preserving all facts. "
                        "Avoid copying phrases; paraphrase clearly and concisely.\n\n" + text_for_output
                    )
                    with self._gen_lock:
                        rewritten = self._llm_generate_text(
                            rewrite_prompt,
                            max_tokens=min(GEN_MAX_TOKENS, 512),
                            temp=max(GEN_TEMP, 0.75),
                            repeat_penalty=1.05,
                            top_k=max(20, GEN_TOP_K),
                            top_p=max(0.9, GEN_TOP_P)
                        )
                    cleaned_rewrite = self._extract_conversational_text(rewritten)
                    if cleaned_rewrite and len(cleaned_rewrite) > 20:
                        text_for_output = cleaned_rewrite.strip()
                except Exception:
                    pass

            # Create simple response structure
            # Basic response structure with unified schema expected by UI/safety modules
            citations = [f"Ref {i}" for i in range(min(len(safe_passages), 10))]
            evidence_sources = [
                f"Ref {i}: {p.get('source', 'Unknown')}" for i, p in enumerate(safe_passages[:10])
            ]
            no_fallbacks = str(os.getenv("BAYMAX_NO_FALLBACKS", "0")).lower() in ("1", "true", "yes")
            response_structure = {
                "medical_assessment": text_for_output,
                "recommendations": [],
                "conditions": [],
                "patient_education": "",
                "confidence_level": 0.8,
                "confidence_overall": 0.8,
                "citations": citations,
                "evidence_sources": evidence_sources
            }
            if (not no_fallbacks) and len(text_for_output) < 20:
                response_structure = self._create_advanced_fallback_response(safe_passages, specialties)
            
            # Build XAI package if enabled
            xai = None
            if self.enable_xai:
                try:
                    xai = build_xai_package(
                        user_text=user_text,
                        response_text=response_structure.get("medical_assessment", ""),
                        passages=safe_passages,
                        symptoms=safe_symptoms or {},
                        emergency_level=emergency_check.get("emergency_level")
                    )
                except Exception as _:
                    xai = {"error": str(_)}
            
            safe_emergency = dict(emergency_check)
            if "emergency_message" in safe_emergency:
                safe_emergency["emergency_message"] = None
            # Conditional references output when user explicitly asks
            request_sources = False
            lt = (user_text or "").strip().lower()
            if any(kw in lt for kw in ["show sources", "where is this from", "cite this", "references", "source list"]):
                request_sources = True
            if request_sources:
                # Short, natural-language reference list (no URLs, no brackets)
                unique_sources = []
                seen = set()
                for i, p in enumerate(safe_passages[:10]):
                    name = p.get('source') or 'Unknown'
                    cat = p.get('category') or ''
                    key = f"{name}|{cat}"
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_sources.append(f"• {name}" + (f" — {cat}" if cat else ""))
                ref_text = "\n".join(unique_sources) if unique_sources else "No sources available."
                response_structure["medical_assessment"] = ref_text

            result = {
                "emergency": False,
                "response": response_structure,
                "specialties": specialties,
                "query_type": query_type,
                "passages_used": len(safe_passages),
                "reasoning_quality": "fast",
                "model_used": self.model_name,
                "emergency_assessment": safe_emergency
            }
            if xai is not None:
                result["xai"] = xai
            return result
            
        except Exception as e:
            print(f"Error in advanced response generation: {e}")
            return {
                "error": True,
                "message": f"Advanced reasoning error: {str(e)}",
                "specialties": specialties,
                "emergency": emergency_check["is_emergency"]
            }
    
    def _classify_query_type(self, text: str, symptoms: Optional[Dict]) -> str:
        """Classify the type of medical query for appropriate response handling"""
        text_lower = text.lower()
        
        # Check for specific query patterns
        if any(phrase in text_lower for phrase in ["what is", "what are", "define", "explain"]):
            return "medical_information"
        elif any(phrase in text_lower for phrase in ["treat", "treatment", "therapy", "medicine", "medication"]):
            return "treatment_planning"  
        elif any(phrase in text_lower for phrase in ["diagnose", "diagnosis", "what do i have", "what's wrong"]):
            return "differential_diagnosis"
        elif any(phrase in text_lower for phrase in ["drug", "medication", "pill", "dose", "side effect"]):
            return "drug_information"
        elif any(phrase in text_lower for phrase in ["cause", "causes", "etiology", "risk factor", "risk factors"]):
            return "differential_diagnosis"
        elif symptoms and symptoms.get('entities_count', 0) > 0:
            return "differential_diagnosis"
        else:
            return "general"
    
    def _build_medical_consultation_prompt(self, 
                                         user_text: str,
                                         passages: List[Dict],
                                         symptoms: Optional[Dict],
                                         specialties: List[str],
                                         context: Optional[List[Dict]],
                                         user_mode: str) -> str:
        """Deprecated: use _compose_main_prompt instead (kept for compatibility)."""
        conv_mem = self._build_conversation_memory_text(context, current_user_text=user_text)
        return self._compose_main_prompt(
            user_text=user_text,
            passages=passages,
            symptoms=symptoms,
            summarized_context=None,
            conversation_memory=conv_mem or "",
        )
    
    def _truncate_prompt_if_needed(self, prompt: str, max_tokens: int = 1800) -> str:
        """Truncate prompt to fit within context window, preserving important sections"""
        # Rough token estimation (4 chars = 1 token)
        estimated_tokens = len(prompt) // 4
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # If too long, prioritize: query, symptoms, emergency info, and first few references
        lines = prompt.split('\n')
        important_lines = []
        reference_count = 0
        
        for line in lines:
            if any(key in line for key in ['MEDICAL LITERATURE CONTEXT:', 'Question:']):
                important_lines.append(line)
            elif 'Ref[' in line and reference_count < 3:  # Keep only first 3 references
                important_lines.append(line)
                reference_count += 1
            elif len(' '.join(important_lines)) // 4 < max_tokens * 0.8:  # Keep under 80% of limit
                important_lines.append(line)
        
        truncated = '\n'.join(important_lines)
        if len(truncated) // 4 > max_tokens:
            # Final truncation if still too long
            truncated = truncated[:max_tokens * 4]
        
        return truncated + "\n\n[Note: Prompt truncated to fit context window]\n"
    
    def _format_medical_literature(self, passages: List[Dict]) -> str:
        """Format medical literature for clinical context"""
        if not passages:
            return "No specific medical literature retrieved for this query."
        
        formatted = []
        try:
            limit_chars = int(os.getenv("BAYMAX_CONTEXT_CHARS", "400"))
        except Exception:
            limit_chars = 400
        for i, passage in enumerate(passages):
            text = passage.get('text', '')
            source = passage.get('source', 'Unknown source')
            category = passage.get('category', 'General')
            url = passage.get('url', 'N/A')
            
            snippet = text[:limit_chars]
            formatted.append(f"""
Ref[{i}]: {source} ({category}): {snippet}...
""")
        
        return "\n".join(formatted)
    
    def _parse_advanced_medical_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse advanced medical AI response with sophisticated error handling"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                parsed = json.loads(json_text)
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Advanced fallback parsing for medical content
        return self._extract_structured_medical_content(raw_response)
    
    def _extract_structured_medical_content(self, text: str) -> Dict[str, Any]:
        """Extract structured medical information from free text"""
        # This is a sophisticated text parser for medical content
        # It can extract key medical concepts even when JSON parsing fails
        
        lines = text.split('\n')
        content = {
            "medical_assessment": "",
            "recommendations": [],
            "conditions": [],
            "patient_education": "",
            "follow_up_care": "",
            "warning_signs": [],
            "confidence_level": 0.6,
        }
        
        current_section = "assessment"
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections based on medical terminology
            if any(keyword in line.lower() for keyword in ["assessment", "diagnosis", "clinical impression"]):
                if current_text:
                    content["medical_assessment"] = " ".join(current_text)
                current_section = "assessment"
                current_text = []
            elif any(keyword in line.lower() for keyword in ["recommend", "treatment", "management", "therapy"]):
                current_section = "recommendations"
                if not line.lower().startswith(('recommend', 'treatment', 'management')):
                    content["recommendations"].append(line)
            elif any(keyword in line.lower() for keyword in ["education", "patient should", "important to"]):
                current_section = "education"
                current_text = [line]
            elif any(keyword in line.lower() for keyword in ["warning", "red flag", "seek immediate", "emergency"]):
                current_section = "warnings"
                content["warning_signs"].append(line)
            else:
                if current_section == "assessment":
                    current_text.append(line)
                elif current_section == "recommendations" and line:
                    content["recommendations"].append(line)
                elif current_section == "education":
                    current_text.append(line)
        
        # Finalize assessment
        if current_text and current_section == "assessment":
            content["medical_assessment"] = " ".join(current_text)
        elif current_text and current_section == "education":
            content["patient_education"] = " ".join(current_text)
        
        # Ensure we have some content
        if not content["medical_assessment"] and text:
            content["medical_assessment"] = text[:500] + "..." if len(text) > 500 else text
        
        return content
    
    def _enhance_with_medical_reasoning(self, 
                                      response: Dict[str, Any],
                                      passages: List[Dict],
                                      specialties: List[str],
                                      query_type: str) -> Dict[str, Any]:
        """Enhance response with additional medical reasoning and validation"""
        
        # Add evidence citations
        response["evidence_citations"] = [
            f"Reference [{i}]: {p.get('source', 'Unknown')}" 
            for i, p in enumerate(passages[:5])
        ]
        
        # Add specialty context
        response["relevant_specialties"] = specialties
        
        # Add reasoning metadata
        response["clinical_reasoning_applied"] = True
        response["query_type"] = query_type
        response["response_timestamp"] = datetime.now().isoformat()
        
        # Validate medical content
        response["content_validation"] = self._validate_medical_content(response, passages)
        
        return response
    
    def _validate_medical_content(self, response: Dict, passages: List[Dict]) -> Dict[str, Any]:
        """Validate medical content against source materials"""
        validation = {
            "source_alignment": "good",
            "medical_accuracy_confidence": 0.8,
            "citation_completeness": "adequate",
            "safety_appropriate": True
        }
        
        # Check if response aligns with retrieved passages
        response_text = json.dumps(response).lower()
        source_concepts = set()
        
        for passage in passages:
            text = passage.get('text', '').lower()
            # Extract key medical concepts (simplified)
            words = text.split()
            source_concepts.update([w for w in words if len(w) > 4])
        
        # Simple alignment check
        response_words = response_text.split()
        common_concepts = len([w for w in response_words if w in source_concepts])
        
        if common_concepts > 10:
            validation["source_alignment"] = "excellent"
        elif common_concepts > 5:
            validation["source_alignment"] = "good"
        else:
            validation["source_alignment"] = "moderate"
        
        return validation
    
    def _generate_appropriate_disclaimer(self, query_type: str, specialties: List[str]) -> str:
        """Generate appropriate medical disclaimer based on query type"""
        base_disclaimer = "This information is provided for educational purposes only and does not constitute medical advice."
        
        if query_type == "differential_diagnosis":
            return base_disclaimer + " Only a qualified healthcare provider can provide an accurate diagnosis after proper medical evaluation."
        elif query_type == "treatment_planning":
            return base_disclaimer + " Treatment decisions should always be made in consultation with qualified healthcare professionals who can assess your individual medical situation."
        elif query_type == "drug_information":
            return base_disclaimer + " Medication information should be verified with healthcare providers and pharmacists. Never start or stop medications without professional guidance."
        elif "cardiology" in specialties or "neurology" in specialties:
            return base_disclaimer + " Cardiovascular and neurological conditions require specialized medical evaluation and care. Please consult appropriate specialists."
        else:
            return base_disclaimer + " Always consult qualified healthcare professionals for medical decisions, diagnosis, and treatment planning."
    
    def _create_advanced_fallback_response(self, passages: List[Dict], specialties: List[str]) -> Dict[str, Any]:
        """Create sophisticated fallback response when AI generation fails"""
        return {
            "medical_assessment": "I was unable to generate a complete structured response, but I have retrieved relevant medical information from authoritative sources.",
            "retrieved_information": f"Found {len(passages)} relevant medical references",
            "relevant_specialties": specialties,
            "recommendations": [
                "Review the retrieved medical literature below",
                "Consult with a healthcare provider for personalized medical advice",
                "Consider seeking evaluation from relevant medical specialists" + (f" ({', '.join(specialties)})" if specialties else "")
            ],
            "follow_up_care": "Schedule an appointment with your healthcare provider to discuss these findings and your specific medical situation",
            "confidence_level": 0.4,
            "reasoning_quality": "fallback"
        }
    
    def _convert_to_json_safe(self, obj):
        """Convert numpy types and other non-JSON serializable objects to JSON-safe types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_safe(item) for item in obj]
        else:
            return obj
    
    def _summarize_evidence(self, user_text: str, passages: List[Dict]) -> str:
        """Summarize retrieved passages into one unified medical understanding (abstraction pass)."""
        all_texts = []
        try:
            ctx_chars = int(os.getenv("BAYMAX_CONTEXT_CHARS", "800"))
        except Exception:
            ctx_chars = 800
        for p in passages[:10]:
            txt = p.get('text') or ''
            if not isinstance(txt, str):
                txt = str(txt)
            if txt:
                all_texts.append(txt[:ctx_chars])
        concatenated = "\n\n".join(all_texts)
        if not concatenated:
            return ""
        summarization_prompt = (
            "Integrate the following medical excerpts into one concise, clinically relevant summary. No quotes or lists.\n\n"
            f"TEXT:\n{concatenated}\n\n"
            "Summary:"
        )
        with self._gen_lock:
            summary = self._llm_generate_text(
                summarization_prompt,
                max_tokens=min(GEN_MAX_TOKENS, 512),
                temp=max(GEN_TEMP, 0.4),
                repeat_penalty=1.10,
                top_k=max(20, GEN_TOP_K),
                top_p=max(0.9, GEN_TOP_P),
            )
        cleaned = self._extract_conversational_text(summary)
        return cleaned or ""

    def _build_conversation_memory_text(self,
                                        conversation_context: Optional[List[Dict]],
                                        current_user_text: Optional[str] = None) -> str:
        """Build a brief, compact conversation memory from recent turns.
        - Never include long verbatim replies; trim aggressively.
        - Prefer the last few turns (user/assistant) and cap total characters.
        """
        try:
            max_turns = int(os.getenv("BAYMAX_MEMORY_TURNS", "4"))  # total recent messages (e.g., 2 pairs)
        except Exception:
            max_turns = 4
        try:
            char_limit = int(os.getenv("BAYMAX_MEMORY_CHARS", "800"))
        except Exception:
            char_limit = 800
        if not conversation_context or max_turns <= 0 or char_limit <= 0:
            return ""
        lines: List[str] = []
        per_turn_limit = max(80, min(220, char_limit // max(1, max_turns)))
        for m in conversation_context[-max_turns:]:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").strip()
            content = m.get("content")
            # Skip the current user text to avoid duplication in prompt
            if current_user_text and role == "user" and isinstance(content, str):
                if content.strip() == (current_user_text or "").strip():
                    continue
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict):
                # Prefer nested response.medical_assessment
                resp = content.get("response") if isinstance(content.get("response"), dict) else None
                if resp and isinstance(resp.get("medical_assessment"), str):
                    text = resp.get("medical_assessment", "")
                elif isinstance(content.get("medical_assessment"), str):
                    text = content.get("medical_assessment", "")
                elif isinstance(content.get("message"), str):
                    text = content.get("message", "")
                else:
                    text = ""
            text = re.sub(r"\s+", " ", str(text or "")).strip()
            if not text:
                continue
            if len(text) > per_turn_limit:
                text = text[:per_turn_limit].rstrip() + "..."
            # Historical framing ("you already told them") rather than a
            # transcript ("User:/Healix:") — small models continue transcripts
            # by re-stating the last assistant line as their opening.
            role_label = "The user said" if role == "user" else "You already replied"
            lines.append(f"{role_label}: {text}")
        memory = "\n".join(lines[-max_turns:]).strip()
        if len(memory) > char_limit:
            memory = memory[-char_limit:]
            # trim to first newline to avoid starting mid-line
            nl = memory.find("\n")
            if nl != -1:
                memory = memory[nl+1:]
        return memory

    class _SafeDict(dict):
        def __missing__(self, key):
            return ""

    def _load_main_prompt_template(self) -> str:
        """Load a single main prompt template from env/file. If none provided, use a minimal fallback.
        Placeholders allowed: {EVIDENCE_BLOCK}, {SYMPTOM_LINE}, {USER_QUESTION}, {CONVERSATION_MEMORY}, {REASONING_SUMMARY_INSTRUCTION}.
        """
        # 1) File path
        for env_key in ("BAYMAX_MAIN_PROMPT_FILE", "HEALIX_MAIN_PROMPT_FILE"):
            p = os.getenv(env_key, "").strip()
            if p:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return f.read()
                except Exception:
                    pass
        # 2) Inline env fallback
        for env_key in ("BAYMAX_MAIN_PROMPT", "HEALIX_MAIN_PROMPT"):
            t = os.getenv(env_key)
            if t and isinstance(t, str) and len(t.strip()) > 0:
                return t
        # 3) Project-root healix_main_prompt.txt, so bare `uvicorn`/`python`
        #    runs get the real prompt even when no launcher set the env var
        try:
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "healix_main_prompt.txt")
            if os.path.exists(default_path):
                with open(default_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception:
            pass
        # 4) Minimal synthesized default to avoid embedding long prompts here
        directive = self._create_advanced_medical_prompt("general").strip()
        tail = (
            "\n\nCONVERSATION MEMORY:\n{CONVERSATION_MEMORY}\n\n"
            "BACKGROUND EVIDENCE:\n{EVIDENCE_BLOCK}\n\n"
            "{SYMPTOM_LINE}\n\n"
            "User question:\n{USER_QUESTION}"
        )
        return directive + tail

    def _compose_main_prompt(self,
                             user_text: str,
                             passages: List[Dict],
                             symptoms: Optional[Dict],
                             summarized_context: Optional[str] = None,
                             conversation_memory: Optional[str] = None,
                             image_findings: Optional[str] = None,
                             expert_directive: Optional[str] = None) -> str:
        # Evidence block
        if summarized_context:
            evidence_block = summarized_context.strip()
        else:
            lines = []
            try:
                ctx_chars = int(os.getenv("BAYMAX_CONTEXT_CHARS", "400"))
            except Exception:
                ctx_chars = 400
            seen = set()
            for p in (passages or [])[:10]:
                txt = p.get('text') or ''
                if not isinstance(txt, str):
                    txt = str(txt)
                snippet = txt[:ctx_chars].strip()
                if len(txt) > ctx_chars:
                    # Cut at the last sentence end so the model never sees a
                    # mid-sentence fragment it might imitate or misread.
                    cut = max(snippet.rfind('. '), snippet.rfind('? '), snippet.rfind('! '))
                    if cut >= ctx_chars // 3:
                        snippet = snippet[:cut + 1]
                key = snippet.lower()
                if snippet and key not in seen:
                    seen.add(key)
                    lines.append(f"- {snippet}")
            evidence_block = "\n".join(lines)

        # Prepend automated imaging findings (if any) so the model grounds its
        # explanation in what the vision model perceived, without diagnosing.
        if image_findings and image_findings.strip():
            evidence_block = (image_findings.strip() + "\n" + evidence_block).strip()

        # Prepend the activated specialist-panel directive (MoE routing), so the
        # model integrates the consulted experts' perspectives into one answer.
        if expert_directive and expert_directive.strip():
            evidence_block = (expert_directive.strip() + "\n" + evidence_block).strip()
        
        # Symptom line
        symptom_line = ""
        try:
            if symptoms and isinstance(symptoms.get('symptoms'), list) and symptoms['symptoms']:
                ents = [e.get('text') for e in symptoms['symptoms'] if isinstance(e, dict) and e.get('text')]
                if ents:
                    symptom_line = f"Symptoms noted: {', '.join(ents[:5])}"
        except Exception:
            symptom_line = ""
        
        # Optional concise reasoning summary toggle (never reveal chain-of-thought)
        try:
            include_reasoning = str(os.getenv("BAYMAX_INCLUDE_REASONING_SUMMARY", "0")).lower() in ("1", "true", "yes")
        except Exception:
            include_reasoning = False
        reasoning_inst = (
            "Include a final 'Why this' section with 2–4 short bullet points summarizing the key factors considered and how they connect. "
            "Avoid step-by-step internal reasoning; do not expose chain-of-thought."
        ) if include_reasoning else ""
        
        # Compose
        tmpl = self.main_prompt_template or "{USER_QUESTION}"
        params = self._SafeDict({
            "EVIDENCE_BLOCK": evidence_block,
            "SYMPTOM_LINE": symptom_line,
            "USER_QUESTION": user_text,
            "REASONING_SUMMARY_INSTRUCTION": reasoning_inst,
            "CONVERSATION_MEMORY": (conversation_memory or ""),
        })
        try:
            return tmpl.format_map(params)
        except Exception:
            return f"{tmpl}\n\n{evidence_block}\n\n{user_text}"

    def clean_stream_text(self, raw_response: str) -> str:
        """Light cleanup for streamed replies: trim stray field headers and
        wrapping quotes while PRESERVING markdown and line breaks.

        The heavier _extract_conversational_text collapses every newline and
        strips bold/lists — fine for the legacy JSON-style non-streaming path,
        but it was flattening well-formatted streamed answers into a single
        wall of text before they were rendered and persisted.
        """
        text = (raw_response or "").strip()
        if not text:
            return ""
        # If the whole reply is a JSON object, pull the best text field.
        if re.match(r'^\s*\{.*\}\s*$', text, flags=re.DOTALL):
            try:
                import json as _json
                parsed = _json.loads(text)
                for key in ("medical_assessment", "answer", "response", "summary"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        text = val.strip()
                        break
            except Exception:
                pass
        # Strip a leading field header if the model emitted one.
        text = re.sub(r'^"?(?:medical_assessment|answer|response)"?\s*:\s*"?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(?:Response:|Answer:|Medical Assessment:)\s*', '', text, flags=re.IGNORECASE)
        # Normalize runs of blank lines but keep paragraph/list structure.
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        if len(text) > 1 and text[0] == '"' and text[-1] == '"':
            text = text[1:-1].strip()
        # House style bans exclamation marks; the model still slips one in
        # occasionally, so enforce it here.
        text = re.sub(r'!+', '.', text)
        return text

    def _extract_conversational_text(self, raw_response: str) -> str:
        """Extract clean conversational text from model output, removing JSON or structured formatting.
        If cleaning produces empty text but raw text exists, fall back to a trimmed raw response.
        """
        response = (raw_response or "").strip()
        if not response:
            return ""
        
        # First, clean any obvious malformed JSON field headers at the start
        response = re.sub(r'^"?medical_assessment"?:\s*"?', '', response)
        response = re.sub(r'^"?[a-z_]+"?:\s*"?', '', response)
        
        # If the whole response looks like a JSON object, try to parse and extract fields
        try:
            import json
            if re.match(r'^\s*\{.*\}\s*$', response, flags=re.DOTALL):
                parsed = json.loads(response)
                for key in ("medical_assessment", "answer", "response", "summary", "condition_summary", "diagnosis"):
                    val = parsed.get(key)
                    if isinstance(val, str) and len(val.strip()) > 0:
                        response = val.strip()
                        break
        except Exception:
            pass
        
        # General cleanup (keep braces inside normal text; don't nuke everything)
        response = re.sub(r'^(Return structured JSON response:|Response:|Answer:|Medical Assessment:)\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```[^`]*```', '', response)  # Remove code blocks
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Bold
        response = re.sub(r'\*(.*?)\*', r'\1', response)        # Italic
        response = re.sub(r'\n\n+', ' ', response)               # Collapse blank lines
        response = re.sub(r'\s+', ' ', response)                  # Normalize whitespace
        cleaned = response.strip().strip('"')
        
        # De-duplicate obvious repeated short phrases (up to 4 words)
        try:
            cleaned = re.sub(r'(\b\w+(?:\s+\w+){0,3})\s*(?:\1\s*){2,}', r'\1 ', cleaned, flags=re.IGNORECASE)
        except Exception:
            pass
        
        # Fallback: if cleaning removed everything, return a safe slice of raw text
        if not cleaned and raw_response and raw_response.strip():
            fallback = re.sub(r'\s+', ' ', raw_response).strip()
            return fallback[:2000]
        return cleaned

    def prepare_stream_prompt(self,
                              user_text: str,
                              retrieved_passages: List[Dict],
                              symptom_data: Optional[Dict] = None,
                              conversation_context: Optional[List[Dict]] = None,
                              user_mode: str = "patient",
                              image_findings: Optional[str] = None,
                              expert_directive: Optional[str] = None) -> Tuple[str, str, List[str]]:
        """Prepare the full prompt and metadata for streaming, using a single main prompt template.

        image_findings: optional text block of automated imaging findings (e.g.
        from backend.services.vision) to ground the explanation.
        expert_directive: optional MoE specialist-panel directive (from
        backend.services.moe_router) describing the activated experts.
        """
        safe_passages = self._convert_to_json_safe(retrieved_passages)
        safe_symptoms = self._convert_to_json_safe(symptom_data) if symptom_data else None
        specialties = self._identify_medical_specialty(user_text)
        query_type = self._classify_query_type(user_text, symptom_data)
        conv_mem = self._build_conversation_memory_text(conversation_context, current_user_text=user_text)
        full_prompt = self._compose_main_prompt(
            user_text=user_text,
            passages=safe_passages,
            symptoms=safe_symptoms,
            summarized_context=None,
            conversation_memory=conv_mem or "",
            image_findings=image_findings,
            expert_directive=expert_directive,
        )
        return full_prompt, query_type, specialties

    def estimate_needed_tokens(self, query: str, query_type: str = "general", context_length: int = 0) -> int:
        """Dynamically estimate needed tokens based on query complexity (ChatGPT-style)."""
        query_len = len(query.split())
        
        # Base allocation by query type
        base_tokens = {
            "greeting": 30,                    # "Hi" → short greeting back
            "simple_fact": 80,                 # "What is aspirin?" → brief definition
            "explanation": 150,                # "How does aspirin work?" → detailed explanation
            "comparison": 200,                 # "Difference between X and Y?" → comparison
            "differential_diagnosis": 250,     # "I have headache and fever" → analysis
            "treatment_planning": 300,         # "How to treat condition X?" → comprehensive plan
            "complex_medical": 400,            # Multi-part medical questions → thorough answer
        }.get(query_type, 120)
        
        # Adjust based on query length (longer questions = more detailed expected)
        if query_len < 5:
            multiplier = 0.5    # Very short query → brief answer
        elif query_len < 10:
            multiplier = 0.8    # Short query → moderate answer
        elif query_len < 20:
            multiplier = 1.0    # Medium query → normal answer
        elif query_len < 40:
            multiplier = 1.3    # Long query → detailed answer
        else:
            multiplier = 1.5    # Very long query → comprehensive answer
        
        # Adjust for context (more context = more detailed answer possible)
        if context_length > 500:
            multiplier *= 1.2
        elif context_length > 1000:
            multiplier *= 1.4
        
        # Calculate final tokens
        estimated = int(base_tokens * multiplier)
        
        # Clamp to reasonable bounds
        min_tokens = int(os.getenv("BAYMAX_GEN_MIN_TOKENS", "30"))
        max_tokens_cap = int(os.getenv("BAYMAX_GEN_MAX_TOKENS", "500"))
        
        return max(min_tokens, min(estimated, max_tokens_cap))
    
    def classify_query_complexity(self, query: str) -> str:
        """Classify query into complexity categories for token allocation."""
        query_lower = query.lower().strip()
        query_words = query.split()
        
        # Greetings and small talk (word-boundary match: "hi" must not match
        # inside "hip pain", which used to skip retrieval for short queries)
        if len(query_words) <= 4 and re.search(
                r"\b(?:hi|hiya|hello|hey|yo|thanks|thank\s+you|bye|goodbye|good\s+(?:morning|afternoon|evening|night))\b",
                query_lower):
            return "greeting"
        
        # Simple factual queries
        if query_lower.startswith(("what is", "define", "meaning of")) and len(query_words) < 10:
            return "simple_fact"
        
        # Explanatory queries
        if any(word in query_lower for word in ["how does", "how do", "explain", "why", "what causes"]):
            return "explanation"
        
        # Comparison queries
        if any(word in query_lower for word in ["difference between", "compare", "versus", "vs", "better than"]):
            return "comparison"
        
        # Symptom-based (differential diagnosis)
        if any(word in query_lower for word in ["i have", "i feel", "experiencing", "symptoms", "suffering from"]):
            return "differential_diagnosis"
        
        # Treatment planning
        if any(word in query_lower for word in ["how to treat", "treatment for", "cure for", "manage", "therapy"]):
            return "treatment_planning"
        
        # Complex multi-part questions
        if len(query_words) > 25 or query.count("?") > 1 or any(word in query_lower for word in ["and", "also", "additionally", "furthermore"]):
            return "complex_medical"
        
        # Default: explanation level
        return "explanation"
    
    def _llm_generate_text(self,
                            prompt: str,
                            max_tokens: int = GEN_MAX_TOKENS,
                            temp: float = GEN_TEMP,
                            top_k: int = GEN_TOP_K,
                            top_p: float = GEN_TOP_P) -> str:
        """Generate text (non-streaming) using Ollama, llama.cpp (GPU), or GPT4All."""
        if self.llm_backend == "ollama":
            try:
                return self._ollama_generate(prompt, max_tokens=max_tokens, temp=temp,
                                             top_k=top_k, top_p=top_p)
            except Exception as e:
                print(f"Ollama generate failed: {e}; falling back to local model")
        prompt = prompt.replace(PROMPT_SPLIT_MARKER, "")
        self._ensure_model_loaded()
        # Try llama.cpp first if available
        if self.backend == "llama_cpp" and self._llama is not None:
            try:
                out = self._llama.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=float(temp),
                    top_p=float(top_p),
                    stream=False,
                )
                msg = out["choices"][0].get("message", {}).get("content", "")
                return msg
            except Exception as e:
                print(f"llama.cpp generation failed: {e}; falling back to GPT4All")
                # fallthrough to GPT4All
        # GPT4All path
        try:
            return self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temp=temp,
                repeat_penalty=1.0,
                top_k=top_k,
                top_p=top_p,
            )
        except TypeError:
            return self.model.generate(
                prompt,
                n_predict=max_tokens,
                temp=temp,
                repeat_penalty=1.0,
                top_k=top_k,
                top_p=top_p,
            )

    def _llm_generate_stream(self,
                             prompt: str,
                             max_tokens: int = GEN_MAX_TOKENS,
                             temp: float = GEN_TEMP,
                             top_k: int = GEN_TOP_K,
                             top_p: float = GEN_TOP_P):
        """Stream tokens using either llama.cpp (GPU) or GPT4All."""
        prompt = prompt.replace(PROMPT_SPLIT_MARKER, "")
        self._ensure_model_loaded()
        # llama.cpp streaming
        if self.backend == "llama_cpp" and self._llama is not None:
            try:
                for out in self._llama.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=float(temp),
                    top_p=float(top_p),
                    stream=True,
                ):
                    # Newer API yields deltas
                    delta = out["choices"][0].get("delta") or {}
                    tok = delta.get("content", "")
                    if not tok:
                        # Fallback to text
                        tok = out["choices"][0].get("text", "")
                    if tok:
                        yield tok
                return
            except Exception as e:
                print(f"llama.cpp streaming failed: {e}; falling back to GPT4All")
                # fallthrough
        # GPT4All streaming
        try:
            gen = self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temp=temp,
                repeat_penalty=1.0,
                top_k=top_k,
                top_p=top_p,
                streaming=True,
            )
            for tok in gen:
                if tok:
                    yield tok
            return
        except TypeError:
            pass
        except Exception:
            pass

    def _ollama_stream(self, prompt: str, max_tokens: int = GEN_MAX_TOKENS,
                       temp: float = GEN_TEMP, top_k: int = GEN_TOP_K,
                       top_p: float = GEN_TOP_P):
        """Stream tokens from a local Ollama server (/api/generate)."""
        import json as _json
        import requests
        system_text = ""
        if PROMPT_SPLIT_MARKER in prompt:
            system_text, prompt = prompt.split(PROMPT_SPLIT_MARKER, 1)
            system_text, prompt = system_text.strip(), prompt.strip()
        try:
            num_ctx = int(os.getenv("BAYMAX_GEN_CONTEXT", "8192"))
        except Exception:
            num_ctx = 8192
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": os.getenv("BAYMAX_OLLAMA_KEEP_ALIVE", "30m"),
            "options": {"temperature": float(temp), "top_p": float(top_p),
                        "top_k": int(top_k), "num_predict": int(max_tokens),
                        "num_ctx": num_ctx},
        }
        if system_text:
            payload["system"] = system_text
        with requests.post(f"{self.ollama_url}/api/generate", json=payload,
                           stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = _json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                tok = obj.get("response", "")
                if tok:
                    yield tok
                if obj.get("done"):
                    break

    def _ollama_generate(self, prompt: str, max_tokens: int = GEN_MAX_TOKENS,
                         temp: float = GEN_TEMP, top_k: int = GEN_TOP_K,
                         top_p: float = GEN_TOP_P) -> str:
        return "".join(self._ollama_stream(prompt, max_tokens=max_tokens, temp=temp,
                                           top_k=top_k, top_p=top_p))

    def stream_generate(self,
                        prompt: str,
                        max_tokens: int = GEN_MAX_TOKENS,
                        temp: float = GEN_TEMP,
                        top_k: int = GEN_TOP_K,
                        top_p: float = GEN_TOP_P):
        """Stream tokens directly using the unified backend; fallback to buffered split."""
        # Ollama backend: stream from the server, skip local model entirely.
        if self.llm_backend == "ollama":
            try:
                produced = False
                with self._gen_lock:
                    for tok in self._ollama_stream(prompt, max_tokens=max_tokens,
                                                   temp=temp, top_k=top_k, top_p=top_p):
                        if tok:
                            produced = True
                            yield tok
                if produced:
                    return
            except Exception as e:
                print(f"Ollama backend failed: {e}; falling back to local model")
        # Ensure model is loaded before streaming
        self._ensure_model_loaded()
        with self._gen_lock:
            # Use unified generator
            try:
                gen = self._llm_generate_stream(
                    prompt,
                    max_tokens=max_tokens,
                    temp=temp,
                    top_k=top_k,
                    top_p=top_p,
                )
                yielded = False
                for tok in gen:
                    if tok:
                        yielded = True
                        yield tok
                if yielded:
                    return
            except TypeError:
                # Older versions may not support streaming
                pass
            except Exception:
                # Fall back to non-streaming path below
                pass

            # Non-streaming fallback with whitespace-based chunking
            try:
                try:
                    text = self._llm_generate_text(
                        prompt,
                        max_tokens=max_tokens,
                        temp=temp,
                        repeat_penalty=1.0,
                        top_k=top_k,
                        top_p=top_p,
                    )
                except TypeError:
                    # Fallback parameter name
                    text = self.model.generate(
                        prompt,
                        n_predict=max_tokens,
                        temp=temp,
                        repeat_penalty=1.0,
                        top_k=top_k,
                        top_p=top_p,
                    )
            except Exception as _:
                text = ""
            import re as _re
            for piece in _re.split(r'(\s+)', text or ""):
                if piece:
                    yield piece

    def build_xai(self,
                   user_text: str,
                   response_text: str,
                   passages: List[Dict],
                   symptoms: Optional[Dict],
                   emergency_level: Optional[str] = None) -> Dict[str, Any]:
        """Public method to build the XAI package for a given response (no generation)."""
        try:
            return build_xai_package(
                user_text=user_text,
                response_text=response_text or "",
                passages=self._convert_to_json_safe(passages) if passages else [],
                symptoms=self._convert_to_json_safe(symptoms) if symptoms else {},
                emergency_level=emergency_level
            )
        except Exception as e:
            return {"error": True, "message": str(e)}

    def generate_medication_draft(self,
                                  user_text: str,
                                  retrieved_passages: List[Dict],
                                  patient_context: Optional[str] = "",
                                  symptoms: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate a DRAFT medication recommendation JSON for clinician review only.
        This DOES NOT prescribe; it returns structured candidates and checks to run.
        """
        # Prepare evidence snippets (short, varied)
        try:
            ctx_chars = int(os.getenv("BAYMAX_CONTEXT_CHARS", "400"))
        except Exception:
            ctx_chars = 400
        refs = []
        for i, p in enumerate(retrieved_passages[:3]):
            txt = (p.get('text') or '')
            if not isinstance(txt, str):
                txt = str(txt)
            refs.append({
                "ref": i,
                "source": p.get('source', 'Unknown'),
                "snippet": txt[:ctx_chars]
            })
        context_block = "\n\n".join([f"[Ref {r['ref']}] {r['snippet']}" for r in refs])
        prompt = (
            "You are a clinical decision support assistant. Produce a DRAFT medication recommendation for clinician review.\n"
            "Return a JSON object with fields: \n"
            " - candidate_drugs: list of {drug_name, reason, evidence_snippets, confidence_estimate}\n"
            " - checks_to_run: list of checks the clinician must perform (allergy, interactions, pregnancy, renal)\n"
            " - sources: list of citations (titles and source IDs)\n"
            "Do NOT produce a prescription, dosing, or administration instructions. Mark this output as DRAFT.\n"
            f"Patient context: {patient_context or ''}\n"
            f"Question: {user_text}\n"
            f"Evidence:\n{context_block}\n"
            "Output JSON only."
        )
        try:
            # Ensure model is loaded before generation
            self._ensure_model_loaded()
            with self._gen_lock:
                raw = self._llm_generate_text(
                    prompt,
                    max_tokens=min(GEN_MAX_TOKENS, 512),
                    temp=max(GEN_TEMP, 0.7),
                    top_k=max(20, GEN_TOP_K),
                    top_p=max(0.9, GEN_TOP_P)
                )
            # Extract JSON
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                import json as _json
                parsed = _json.loads(m.group(0))
            else:
                parsed = {}
        except Exception:
            parsed = {}
        # Normalize structure and label DRAFT
        draft = {
            "status": "DRAFT",
            "candidate_drugs": parsed.get("candidate_drugs", []),
            "checks_to_run": parsed.get("checks_to_run", ["allergy", "interactions", "pregnancy", "renal"]),
            "sources": parsed.get("sources", []),
            "llm_model": self.model_name,
        }
        return draft

    # -------------------- Intent routing and unified response --------------------
    def route_intent(self, query: str) -> str:
        q = (query or "").lower()
        if any(x in q for x in ["medicine", "drug", "dose", "prescribe", "tablet"]):
            return "medication"
        if any(x in q for x in ["symptom", "pain", "fever", "what could be", "cause of"]):
            return "symptom"
        if any(x in q for x in ["explain", "why", "how does"]):
            return "reasoning"
        return "qa"

    def generate_response(self, query: str, conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Route the query and produce a unified response with simple timings."""
        import time as _time
        t0 = _time.time()
        intent = self.route_intent(query)
        retriever = _get_retriever_singleton()
        symptoms = None
        # Retrieval (k=3)
        if intent == "medication":
            passages = retriever.search_by_category(query, category="Medications", k=3)
            if not passages:
                passages = retriever.retrieve(query, k=3, min_score=0.2)
        else:
            passages = retriever.retrieve(query, k=3, min_score=0.2)
        t1 = _time.time()

        # Optional NER for symptom intent
        t_ner = 0.0
        if intent == "symptom":
            try:
                extractor = _get_symptom_singleton()
                t_ner_s = _time.time()
                symptoms = extractor.extract_symptoms(query)
                t_ner = _time.time() - t_ner_s
            except Exception:
                symptoms = None

        # Generation
        t_gen_s = _time.time()
        if intent == "medication":
            resp = self.generate_medication_draft(user_text=query, retrieved_passages=passages, patient_context="")
        elif intent in ("symptom", "reasoning", "qa"):
                resp = self.synthesize_advanced_response(
                user_text=query,
                retrieved_passages=passages,
                symptom_data=symptoms,
                conversation_context=conversation_context,
                user_mode="patient",
                max_tokens_override=int(os.getenv("BAYMAX_GEN_MAX_TOKENS", "900") or 900)
            )
        else:
            resp = {"error": True, "message": "Unknown intent"}
        t2 = _time.time()

        # Attach timings
        try:
            resp.setdefault("timings", {})
            resp["timings"].update({
                "retrieval_s": round(t1 - t0, 3),
                "ner_s": round(t_ner, 3) if t_ner else 0.0,
                "generation_s": round(t2 - t_gen_s, 3),
                "total_s": round(t2 - t0, 3)
            })
            resp.setdefault("intent", intent)
        except Exception:
            pass
        return resp

    def health_check(self) -> Dict[str, Any]:
        """Advanced health check for the medical orchestrator"""
        try:
            # Ensure model is loaded before health check
            self._ensure_model_loaded()
            # Test advanced reasoning with a medical query
            with self._gen_lock:
                test_response = self._llm_generate_text(
                    "Explain the pathophysiology of myocardial infarction in 2 sentences.",
                    max_tokens=100,
                    temp=0.1
                )
            
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "model_name": self.model_name,
                "advanced_reasoning_enabled": self.enable_advanced_reasoning,
                "emergency_detection_categories": len(self.emergency_keywords),
                "medical_specialties_supported": list(self.medical_specialties.keys()),
                "test_generation_success": len(test_response) > 20,
                "reasoning_capabilities": "ChatGPT-level medical AI",
                "medical_knowledge_domains": list(self.medical_specialties.keys())
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model_loaded": self.model is not None,
                "fallback_available": True
            }


# Example usage and testing
if __name__ == "__main__":
    print("Advanced Healthcare Super-Assistant - ChatGPT-Level Medical AI Test")
    print("=" * 75)
    
    try:
        # Initialize advanced orchestrator
        print("Initializing advanced medical AI...")
        orchestrator = AdvancedMedicalOrchestrator()
        
        # Health check
        health = orchestrator.health_check()
        print(f"\nHealth Status: {health['status']}")
        print(f"Model: {health['model_name']}")
        specialties = health.get('medical_specialties_supported', [])
        print(f"Medical Specialties: {len(specialties)} ({', '.join(specialties[:3])}{', ...' if len(specialties) > 3 else ''})")
        print(f"Reasoning Level: {health.get('reasoning_capabilities', 'Standard')}")
        
        if health['status'] == 'healthy':
            # Test advanced medical reasoning
            test_cases = [
                {
                    "query": "I have chest pain radiating to my left arm with sweating",
                    "description": "Emergency cardiac symptoms"
                },
                {
                    "query": "What are the current treatment guidelines for type 2 diabetes?",
                    "description": "Treatment planning query"
                },
                {
                    "query": "Explain the differential diagnosis for acute abdominal pain",
                    "description": "Differential diagnosis request"
                }
            ]
            
            for i, test in enumerate(test_cases, 1):
                print(f"\n{'='*60}")
                print(f"Advanced Medical AI Test {i}: {test['description']}")
                print(f"Query: \"{test['query']}\"")
                print("="*60)
                
                # Mock retrieved passages for testing
                mock_passages = [
                    {"text": f"Mock medical literature relevant to: {test['query'][:50]}...", 
                     "source": "Medical Journal", "category": "Clinical"}
                ]
                
                result = orchestrator.synthesize_advanced_response(
                    user_text=test['query'],
                    retrieved_passages=mock_passages,
                    user_mode="patient"
                )
                
                print(f"Emergency Detected: {result.get('emergency', False)}")
                print(f"Specialties: {', '.join(result.get('specialties', []))}")
                print(f"Query Type: {result.get('query_type', 'general')}")
                print(f"Reasoning Quality: {result.get('reasoning_quality', 'standard')}")
                
                if result.get('emergency'):
                    print(f"Emergency Level: {result.get('emergency_level', 'N/A')}")
                
                print("✅ Advanced medical reasoning test completed")
        
        print(f"\n🎯 Advanced Healthcare AI System Ready!")
        print("🧠 Capabilities: ChatGPT-level medical reasoning, comprehensive emergency detection, multi-specialty support")
        
    except Exception as e:
        print(f"❌ Error testing advanced orchestrator: {e}")
        import traceback
        traceback.print_exc()
