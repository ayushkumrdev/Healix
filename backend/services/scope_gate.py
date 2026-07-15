#!/usr/bin/env python3
"""
Healix scope gate — keeps the assistant strictly inside the medical domain.

Every chat message passes through `ScopeGate.classify()` before retrieval or
generation. The gate resolves each message to one of:

  greeting     pure greetings / "how are you"            -> warm canned line
  thanks       gratitude                                  -> canned line
  farewell     goodbyes                                   -> canned line
  identity     "who are you", "what can you do"           -> canonical identity line
  out_of_scope clearly non-medical asks (code, trivia,    -> polite redirect
               sports, finance, entertainment, ...)
  medical      everything else                            -> full RAG + LLM pipeline

Design principles:
- Refusals must be HIGH PRECISION. A health question wrongly refused is far
  worse than a non-medical question slipping through to the LLM (the main
  prompt carries a strict domain directive as the second line of defense).
- Word-boundary matching only; substring matching caused real bugs before
  ("hi" inside "hip pain" classified short medical queries as greetings).
- Short follow-ups inside an ongoing conversation ("how long will it last?",
  "what about my dad?") are never refused on heuristics alone.
- When heuristics are inconclusive, an optional embedding check (reusing the
  retriever's encoder, the same trick as the MoE gate) compares the query
  against medical vs non-medical anchor prototypes. It refuses only with a
  clear margin; any failure degrades to "medical" (lenient).

Canned replies follow the house voice: calm, no emojis, no exclamation marks,
no disclaimers.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ScopeResult:
    kind: str                 # greeting | thanks | farewell | identity | out_of_scope | medical
    reply: Optional[str]      # canned reply to stream, or None to run the pipeline
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# canned voice
# ---------------------------------------------------------------------------

GREETING_REPLIES = [
    "Hello. How are you feeling today?",
    "Hi. What can I help you understand about your health?",
    "Hey. How is your body doing today?",
    "Hello. What's on your mind, health-wise?",
]

HOWAREYOU_REPLIES = [
    "Running steady, thank you. More importantly — how are you feeling?",
    "I'm well. How about you — anything your body has been telling you lately?",
]

THANKS_REPLIES = [
    "You're welcome. I'm here if anything else comes up.",
    "Glad I could help. Take care of yourself.",
    "Anytime. I'm here whenever something feels off.",
]

FAREWELL_REPLIES = [
    "Take care. I'll be here when you need me.",
    "Be well. I'm always here if something changes.",
    "Take care of yourself. Come back anytime.",
]

IDENTITY_REPLY = (
    "I'm Healix, a medical companion that runs entirely on your device. "
    "I help you understand your health — symptoms, medications, sleep, stress, "
    "nutrition, lab results, or anything happening in your body."
)

OUT_OF_SCOPE_REPLIES = [
    "That's outside my focus — I'm built purely for health. Ask me about symptoms, "
    "medications, sleep, stress, nutrition, or anything happening in your body.",
    "I stay in one lane, and that lane is your health. If something is bothering you "
    "physically or mentally, that's where I can actually help.",
    "I can't help with that one — medicine is my whole world. Bring me a symptom, a "
    "medication question, or anything about how your body works.",
]


# ---------------------------------------------------------------------------
# heuristics
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"(?:hi|hiya|hello|helo|hey+|yo|sup|what'?s\s+up|wassup|howdy|namaste|hola|greetings|"
    r"good\s+(?:morning|afternoon|evening|day))"
    r"(?:[\s,!.]*(?:healix|there|doc|doctor|buddy|man|again))?[\s!.?]*",
    re.IGNORECASE,
)

_HOWAREYOU_RE = re.compile(
    r"(?:(?:hi|hello|hey)[\s,!.]*)?"
    r"(?:how\s+are\s+you(?:\s+doing|\s+today)?|how'?s\s+it\s+going|how\s+do\s+you\s+do|you\s+ok|you\s+good)"
    r"[\s!.?]*",
    re.IGNORECASE,
)

_THANKS_RE = re.compile(
    r"(?:ok(?:ay)?[\s,!.]*)?(?:many\s+)?(?:thanks|thank\s+you|thankyou|thx|ty|tysm|thanks\s+a\s+lot|"
    r"thank\s+you\s+so\s+much|appreciate\s+it|appreciated)(?:[\s,!.]*(?:healix|doc|doctor|so\s+much|a\s+lot))?[\s!.?]*",
    re.IGNORECASE,
)

_FAREWELL_RE = re.compile(
    r"(?:ok(?:ay)?[\s,!.]*)?(?:bye+|goodbye|good\s+night|goodnight|see\s+you(?:\s+later)?|see\s+ya|"
    r"take\s+care|catch\s+you\s+later|gtg|gotta\s+go|talk\s+later)(?:[\s,!.]*(?:healix|doc|doctor))?[\s!.?]*",
    re.IGNORECASE,
)

_IDENTITY_RE = re.compile(
    r"(?:who|what)\s+(?:are|r)\s+(?:you|u)\b|who\s+is\s+healix|what\s+is\s+healix|"
    r"introduce\s+yourself|tell\s+me\s+about\s+yourself|describe\s+yourself|"
    r"what\s+can\s+(?:you|u)\s+do|what\s+do\s+(?:you|u)\s+do|what'?s\s+your\s+name|your\s+name\b|"
    r"are\s+you\s+(?:a\s+)?(?:bot|ai|robot|human|doctor)",
    re.IGNORECASE,
)

# Broad health vocabulary. Terms act as prefixes ("diagnos" matches diagnosis /
# diagnose / diagnostic). Any hit means the message goes to the full pipeline.
_MEDICAL_TERMS = (
    "pain ache aching hurt hurts sore soreness fever feverish chills cough cold flu covid "
    "headache migraine nausea nauseous vomit diarrhea diarrhoea constipat bloat cramp spasm "
    "rash itch itchy hives swell swollen inflam dizz vertigo faint fatigue tired exhaust "
    "insomnia sleep sleepy drowsy snor apnea stress anxiet anxious depress panic mood burnout "
    "trauma therapy therapist breath breathing breathless wheez asthma chest heart cardiac "
    "palpitation pulse blood pressure hypertension sugar glucose diabet cholesterol thyroid "
    "kidney renal liver hepat lung pulmonary stomach abdomen abdominal bowel intestin gut "
    "digest acid reflux heartburn ulcer gastritis skin acne eczema psoriasis mole lesion wart "
    "wound bruise burn injury injur sprain strain fracture broken bone joint muscle muscular "
    "back neck shoulder knee hip ankle wrist elbow spine arthritis osteo allerg sinus congestion "
    "infection infect virus viral bacteria bacterial fungal antibiotic antiviral medicine "
    "medication med meds drug pill tablet capsule dose dosage dosing side effect prescription "
    "pharmacy paracetamol acetaminophen ibuprofen aspirin antacid antihistamine insulin statin "
    "vaccine vaccin immuniz booster vitamin supplement mineral protein nutrition nutrient diet "
    "calorie weight obese obesity bmi appetite fasting hydrat dehydrat electrolyte exercise "
    "workout cardio stretch posture ergonomic period menstrual menstruation cramps pms pcos "
    "pregnan fertility ovulation menopause hormone hormonal testosterone estrogen cortisol "
    "adrenal immune immunity inflammation cancer tumor tumour cyst symptom symptoms diagnos "
    "prognosis treatment cure heal healing recover recovery rehab surgery surgical operation "
    "doctor physician clinic clinician hospital nurse patient specialist x-ray xray radiograph "
    "mri ct scan ultrasound ecg ekg eeg biopsy bloodwork screening checkup test results "
    "eye eyes vision blurry earache ear pain ear infection hearing tinnitus throat tonsil nose "
    "nosebleed tooth teeth toothache dental gum jaw hair loss dandruff urine urinary bladder "
    "prostate sexual libido erectile std sti hiv hepatitis herpes seizure epilep stroke "
    "concussion numb tingl tremor twitch memory brain nerve neural neuro cognitive dementia "
    "alzheimer parkinson autism adhd wellness wellbeing health healthy unwell sick sickness "
    "illness disease disorder syndrome chronic acute smoking nicotine alcohol hangover caffeine "
    "addiction withdrawal overdose poison toxin anemia anaemia deficiency jaundice malaria "
    "dengue typhoid eat eating belly fat keto vegan vegetarian fiber fibre gluten lactose "
    "hemoglobin haemoglobin platelet sodium potassium calcium iron b12 melatonin"
).split()

_MEDICAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(set(_MEDICAL_TERMS), key=len, reverse=True)) + r")",
    re.IGNORECASE,
)

# High-precision non-medical intent. A match here (with zero medical terms)
# refuses immediately.
_NON_MEDICAL_RES = [
    # programming & tech work
    re.compile(r"\b(?:python|javascript|typescript|java|c\+\+|c#|rust|golang|html|css|sql|regex|"
               r"code|coding|program|programming|debug|compile|deploy|api|algorithm|function|script|"
               r"software|frontend|backend|database|server|website|web\s*app|machine\s+learning|neural\s+network)\b", re.I),
    # creative writing / homework
    re.compile(r"\b(?:write|compose|draft|generate|make)\b.{0,40}\b(?:essay|poem|poetry|story|song|lyrics|"
               r"email|letter|blog|article|tweet|caption|resume|cover\s+letter|speech|script)\b", re.I),
    re.compile(r"\b(?:tell|crack)\s+(?:me\s+)?a\s+joke|\briddle\b|\brap\s+battle\b", re.I),
    re.compile(r"\b(?:solve|calculate|evaluate)\b.{0,30}\b(?:equation|integral|derivative|theorem|expression)\b|"
               r"\bhomework\b|\bassignment\b", re.I),
    # trivia / general knowledge
    re.compile(r"\b(?:capital\s+of|president\s+of|prime\s+minister|population\s+of|currency\s+of|"
               r"who\s+invented|who\s+discovered|who\s+founded|world\s+war|in\s+what\s+year|"
               r"history\s+of\s+(?!medicine|disease|illness))", re.I),
    # entertainment & celebrities
    re.compile(r"\b(?:movie|movies|film|films|netflix|song|songs|album|singer|actor|actress|celebrity|"
               r"tv\s+show|series|anime|manga|kpop|bollywood|hollywood|concert|playlist)\b", re.I),
    # sports
    re.compile(r"\b(?:football|soccer|cricket|basketball|baseball|tennis|hockey|ipl|fifa|nba|nfl|ufc|"
               r"olympics|premier\s+league|world\s+cup|tournament|championship)\b", re.I),
    # money & finance
    re.compile(r"\b(?:stock|stocks|share\s+market|crypto|bitcoin|ethereum|nft|invest|investing|investment|"
               r"trading|forex|mutual\s+fund|tax|taxes|loan|mortgage|salary|paycheck|budget(?:ing)?)\b", re.I),
    # politics & news
    re.compile(r"\b(?:election|politics|political|parliament|congress|senate|geopolit|war\s+in)\b", re.I),
    # travel & weather
    re.compile(r"\b(?:flight|flights|hotel|hotels|visa|itinerary|tourist|vacation|sightseeing|"
               r"travel\s+to|trip\s+to|weather|forecast)\b", re.I),
    # gaming, gadgets, shopping, vehicles
    re.compile(r"\b(?:video\s+game|gaming|minecraft|fortnite|valorant|gta|playstation|xbox|"
               r"gpu|graphics\s+card|iphone|smartphone|laptop|which\s+phone|car\s+to\s+buy|bike\s+to\s+buy)\b", re.I),
    # cooking-as-cooking (health-flavoured recipe questions carry medical terms)
    re.compile(r"\b(?:recipe|how\s+to\s+(?:cook|bake)|restaurant)\b", re.I),
]

# Short conversational continuations are passed through when a conversation is
# already underway — context lives in the LLM prompt, not in this gate.
_CONTINUATION_RE = re.compile(
    r"^(?:and|also|but|so|what\s+about|how\s+about|why|how|when|where|which|what|can|could|should|"
    r"would|does|do|did|is|are|was|it|that|this|they|he|she|my|ok|okay|yes|yeah|no|nope|sure|"
    r"more|explain|elaborate|continue|go\s+on|tell\s+me\s+more)\b",
    re.IGNORECASE,
)


def _fullmatch(rx: re.Pattern, text: str) -> bool:
    return bool(rx.fullmatch(text.strip()))


class ScopeGate:
    """Domain gate. `retriever` is optional and only used for the embedding
    tiebreaker; pass the shared MedicalRetriever to reuse its encoder."""

    # anchor prototypes for the embedding tiebreaker
    _MED_ANCHORS = [
        "symptoms, causes and treatment of a medical condition",
        "pain, fever, cough, fatigue, nausea and other body symptoms",
        "medication dosage, side effects and drug interactions",
        "anatomy, organs and how the human body works",
        "mental health, stress, anxiety, sleep and mood",
        "diet, nutrition, exercise and a healthy lifestyle",
        "lab tests, scans, diagnosis and clinical care",
        "injury, wound healing, recovery and rehabilitation",
    ]
    _NONMED_ANCHORS = [
        "writing computer code, software programming and debugging",
        "movies, music, celebrities and entertainment",
        "sports teams, matches, scores and tournaments",
        "politics, elections, government and current news",
        "money, investing, stocks, crypto, banking and taxes",
        "travel destinations, flights, hotels and vacation planning",
        "cooking recipes and restaurant recommendations",
        "history, geography, trivia and general knowledge facts",
        "school homework, mathematics, physics and essay writing",
        "video games, gadgets, cars and online shopping",
    ]

    def __init__(self, retriever=None):
        self.retriever = retriever
        self._anchor_embs = None  # (med_matrix, nonmed_matrix)

    # ---- public ---------------------------------------------------------
    def classify(self, text: str, has_image: bool = False,
                 in_conversation: bool = False) -> ScopeResult:
        t = (text or "").strip()
        low = t.lower()
        words = low.split()

        # An attached medical image is always in-domain.
        if has_image:
            return ScopeResult("medical", None, 1.0, "image attached")
        if not t:
            return ScopeResult("greeting", random.choice(GREETING_REPLIES), 1.0, "empty input")

        # Courtesy fast-paths (full-string matches only, so "hi, my chest hurts"
        # can never land here).
        if len(words) <= 7:
            if _fullmatch(_HOWAREYOU_RE, t):
                return ScopeResult("greeting", random.choice(HOWAREYOU_REPLIES), 1.0, "how-are-you")
            if _fullmatch(_GREETING_RE, t):
                return ScopeResult("greeting", random.choice(GREETING_REPLIES), 1.0, "greeting")
            if _fullmatch(_THANKS_RE, t):
                return ScopeResult("thanks", random.choice(THANKS_REPLIES), 1.0, "thanks")
            if _fullmatch(_FAREWELL_RE, t):
                return ScopeResult("farewell", random.choice(FAREWELL_REPLIES), 1.0, "farewell")
        if len(words) <= 10 and _IDENTITY_RE.search(low):
            return ScopeResult("identity", IDENTITY_REPLY, 1.0, "identity")

        # Any medical vocabulary -> straight to the pipeline.
        if _MEDICAL_RE.search(low):
            return ScopeResult("medical", None, 1.0, "medical vocabulary")

        # Clear non-medical intent with zero medical vocabulary -> redirect.
        for rx in _NON_MEDICAL_RES:
            if rx.search(low):
                return ScopeResult("out_of_scope", random.choice(OUT_OF_SCOPE_REPLIES),
                                   0.95, f"non-medical pattern: {rx.pattern[:40]}")

        # Mid-conversation follow-ups stay with the model (it has the context).
        if in_conversation and (len(words) <= 12 or _CONTINUATION_RE.match(low)):
            return ScopeResult("medical", None, 0.6, "conversational continuation")

        # Ambiguous standalone message: semantic tiebreaker if an encoder is
        # available, otherwise be lenient (the prompt guard handles the rest).
        if len(words) >= 4:
            verdict = self._embedding_verdict(t)
            if verdict == "out_of_scope":
                return ScopeResult("out_of_scope", random.choice(OUT_OF_SCOPE_REPLIES),
                                   0.7, "embedding margin")
        return ScopeResult("medical", None, 0.5, "default lenient")

    # ---- embedding tiebreaker -------------------------------------------
    def _encode(self, texts: List[str]):
        if self.retriever is None:
            return None
        try:
            if getattr(self.retriever, "model", None) is None:
                self.retriever._load_model()
            import numpy as np
            embs = self.retriever.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(embs, dtype="float32")
        except Exception:
            return None

    def _embedding_verdict(self, query: str) -> str:
        try:
            if self._anchor_embs is None:
                med = self._encode(self._MED_ANCHORS)
                non = self._encode(self._NONMED_ANCHORS)
                if med is None or non is None:
                    return "medical"
                self._anchor_embs = (med, non)
            med, non = self._anchor_embs
            qv = self._encode([query])
            if qv is None:
                return "medical"
            q = qv[0]
            med_sim = float((med @ q).max())
            non_sim = float((non @ q).max())
            # refuse only with a clear margin AND weak medical affinity
            if non_sim - med_sim > 0.05 and med_sim < 0.45:
                return "out_of_scope"
        except Exception:
            pass
        return "medical"


if __name__ == "__main__":  # tiny self-check (heuristics only)
    gate = ScopeGate()
    for q in ["hi", "hip pain", "who are you", "write me a python function",
              "I've had a headache for three days", "capital of france?", "thanks"]:
        r = gate.classify(q)
        print(f"{q!r:45s} -> {r.kind:13s} ({r.reason})")
