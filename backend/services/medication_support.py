import os
import time
import uuid
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel

# NOTE: This module intentionally avoids issuing prescriptions.
# It only returns non-prescriptive info and drafts. Final prescriptions
# must be created via /submit_prescription by an authenticated clinician.

app = FastAPI(title="Baymax Medication Support API", version="0.1.0")

# --- PERSISTENCE (replace with real DB in production) ---
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "..", "logs")
LOG_DIR = os.path.abspath(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
AUDIT_PATH = os.path.join(LOG_DIR, "audit_log.jsonl")
RX_PATH = os.path.join(LOG_DIR, "prescriptions.jsonl")

PRESCRIPTION_DB = []  # in-memory mirror
AUDIT_LOG = []

# --- MODELS ---
class MedInfoRequest(BaseModel):
    drug_name: str

class DraftRequest(BaseModel):
    patient_id: str
    chief_complaint: str
    context_notes: str = ""

class PrescriptionSubmission(BaseModel):
    patient_id: str
    draft_id: str
    clinician_id: str
    clinician_signature: str  # e.g., SSO token or signed payload
    confirmation_notes: str = ""

# --- helpers ---
def _append_jsonl(path: str, record: Dict[str, Any]):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _audit(event: str, payload: Dict[str, Any]):
    rec = {"event": event, "ts": time.time(), **payload}
    AUDIT_LOG.append(rec)
    _append_jsonl(AUDIT_PATH, rec)


def check_clinician_auth(headers: Dict[str, str]) -> Dict[str, Any]:
    """Implement SSO/clinician auth here; returns clinician metadata if valid."""
    auth = headers.get("authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Clinician auth required")
    # Validate token -> return clinician info (stub)
    # TODO: integrate with real IdP/SSO
    return {"clinician_id": "clinician-123", "name": "Dr Example"}


def fetch_med_info_from_db(drug_name: str) -> Dict[str, Any]:
    """Replace with RxNorm/DailyMed/local formulary. Non-prescriptive fields only."""
    return {
        "drug_name": drug_name,
        "concept_id": f"rxnorm:{drug_name.lower()}",
        "generic_name": drug_name,
        "indications": ["Indication example"],
        "warnings": ["Contraindication example"],
        "common_adverse_effects": ["nausea", "headache"],
        "source": "LOCAL_DB_PLACEHOLDER",
    }


def run_interaction_checks(patient_id: str, drug_name: str) -> Dict[str, Any]:
    """Stub: integrate allergy & interaction engines; classify severities."""
    return {"allergy_conflict": False, "interaction_conflicts": [], "contraindications": [], "notes": ""}


def _get_knowledge_versions() -> Dict[str, Any]:
    return {
        "llm_model": os.getenv("BAYMAX_GGUF_MODEL_NAME", "unknown"),
        "embed_model": os.getenv("BAYMAX_EMBEDDING_MODEL", "unknown"),
        "drug_db_version": os.getenv("DRUG_DB_VERSION", "placeholder"),
    }

# Lazy imports to avoid startup cost if unused
def _lazy_services():
    from backend.services.retriever import MedicalRetriever
    from backend.services.advanced_orchestrator import AdvancedMedicalOrchestrator
    retriever = getattr(_lazy_services, "_retriever", None)
    orchestrator = getattr(_lazy_services, "_orchestrator", None)
    if retriever is None:
        retriever = MedicalRetriever(index_dir="data")
        _lazy_services._retriever = retriever
    if orchestrator is None:
        orchestrator = AdvancedMedicalOrchestrator()
        _lazy_services._orchestrator = orchestrator
    return retriever, orchestrator

# --- endpoints ---
@app.post("/med_info")
async def med_info(req: MedInfoRequest):
    # Returns non-prescriptive structured medication information
    info = fetch_med_info_from_db(req.drug_name)
    if not info:
        raise HTTPException(status_code=404, detail="Medication not found")
    info["disclaimer"] = (
        "Decision support only. Not a prescription. Verify with an authorized clinician and authoritative drug databases."
    )
    info["knowledge_versions"] = _get_knowledge_versions()
    _audit("med_info", {"drug_name": req.drug_name})
    return info

@app.post("/draft_recommendation")
async def draft_recommendation(req: DraftRequest):
    """Generate a DRAFT medication recommendation using LLM as decision support."""
    retriever, orchestrator = _lazy_services()
    # Minimal retrieval for medication-related context
    passages = retriever.search_by_category(req.chief_complaint, category="Medications", k=3)
    if not passages:
        passages = retriever.retrieve(req.chief_complaint, k=3, min_score=0.2)
    patient_context = req.context_notes
    draft = orchestrator.generate_medication_draft(
        user_text=req.chief_complaint,
        retrieved_passages=passages,
        patient_context=patient_context,
    )
    # Attach checks placeholder for each candidate
    for c in draft.get("candidate_drugs", []):
        c["checks"] = run_interaction_checks(req.patient_id, c.get("drug_name", ""))
    draft_id = f"draft-{uuid.uuid4().hex}"
    draft_payload = {
        "draft_id": draft_id,
        "patient_id": req.patient_id,
        "status": "DRAFT",
        "created_at": time.time(),
        "generated_by": "baymax-llm",
        "draft": draft,
        "disclaimer": (
            "This is a draft for clinician review only. Not a prescription. Clinician authentication and signature required for issuance."
        ),
        "knowledge_versions": _get_knowledge_versions(),
    }
    _audit("draft_created", {"draft_id": draft_id, "patient_id": req.patient_id})
    return draft_payload

@app.post("/submit_prescription")
async def submit_prescription(sub: PrescriptionSubmission, request: Request):
    clinician = check_clinician_auth(request.headers)
    # TODO: validate draft existence/consistency
    prescription_id = f"rx-{uuid.uuid4().hex}"
    record = {
        "prescription_id": prescription_id,
        "patient_id": sub.patient_id,
        "draft_id": sub.draft_id,
        "clinician_id": clinician["clinician_id"],
        "clinician_name": clinician["name"],
        "clinician_signature": sub.clinician_signature,
        "confirmed_at": time.time(),
        "confirmation_notes": sub.confirmation_notes,
        "knowledge_versions": _get_knowledge_versions(),
    }
    PRESCRIPTION_DB.append(record)
    _append_jsonl(RX_PATH, record)
    _audit("prescription_submitted", {"prescription_id": prescription_id, "by": clinician})
    return {"status": "ok", "prescription_id": prescription_id}

# Safety banner endpoint (optional)
@app.get("/safety_principles")
async def safety_principles():
    return {
        "principles": [
            "No prescriptions without authenticated clinician approval",
            "LLM output is DRAFT/Decision-Support only",
            "Allergy/interaction/contraindication checks required",
            "Clinician signature and audit logging required",
        ]
    }

