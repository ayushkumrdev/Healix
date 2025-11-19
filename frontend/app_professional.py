#!/usr/bin/env python3
"""
Healix Professional UI - Modern Medical AI Interface
- Professional healthcare design with smooth animations
- Advanced styling and visual hierarchy
- Enhanced emergency detection UI
- Real-time typing indicators
- Session management and history
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import Any, Dict, Optional, List
import streamlit as st

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Avatars
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
USER_AVATAR = os.path.join(ASSETS_DIR, 'user_avatar.svg')
BOT_AVATAR = os.path.join(ASSETS_DIR, 'bot_avatar.svg')

# Import services
from services.retriever import MedicalRetriever
from services.advanced_orchestrator import AdvancedMedicalOrchestrator

# Optional helpers
try:
    from services.clinical_safety import run_all_checks as _run_checks
except Exception:
    _run_checks = None

try:
    from services.audit import audit as audit_log
except Exception:
    def audit_log(event, payload):
        return {"event": event, **payload}


@st.cache_resource
def _load_retriever():
    return MedicalRetriever(index_dir="data")


@st.cache_resource
def _load_orchestrator():
    orchestrator_kwargs = {}
    model_name = os.getenv("BAYMAX_GGUF_MODEL_NAME")
    model_dir = os.getenv("BAYMAX_GGUF_MODEL_DIR")
    if model_name:
        orchestrator_kwargs["model_name"] = model_name
    if model_dir:
        orchestrator_kwargs["model_path"] = model_dir
    return AdvancedMedicalOrchestrator(**orchestrator_kwargs)


def _extract_display_text(response_data: Dict[str, Any]) -> str:
    """Extract clean display text from response payload"""
    try:
        if not isinstance(response_data, dict):
            return str(response_data or "").strip()
        if response_data.get('error'):
            return response_data.get('message', '')
        if response_data.get('emergency'):
            return ''
        response_content = response_data.get('response', {})
        if isinstance(response_content, str):
            return response_content.strip()
        if isinstance(response_content, dict):
            if response_content.get('medical_assessment'):
                return str(response_content['medical_assessment']).strip()
            parts = []
            if response_content.get('summary'):
                parts.append(str(response_content['summary']))
            recs = response_content.get('recommendations')
            if isinstance(recs, list):
                for rec in recs[:3]:
                    if isinstance(rec, str):
                        parts.append(rec)
                    elif isinstance(rec, dict):
                        parts.append(str(rec.get('action', str(rec))))
            return " ".join([p for p in parts if p]).strip()
    except Exception:
        return ""
    return ""


def inject_professional_styles():
    """Inject minimal, high-contrast black/white theme; gray sidebar; no avatars."""
    st.markdown("""
        <style>
        :root {
          --bg: #000000;          /* app background */
          --surface: #000000;     /* chat cards */
          --surface-alt: #000000; /* assistant same as user */
          --sidebar: #1f2937;     /* gray sidebar */
          --border: #333333;      /* subtle borders */
          --text: #ffffff;        /* all text white */
          --muted: #d1d5db;       /* muted text */
        }

        html, body { background: var(--bg); }
        * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color: var(--text); }
        #MainMenu, footer { visibility: hidden; }
        .main .block-container { max-width: 920px; padding: 1.25rem 1rem; background: transparent; }

        /* Header */
        .healix-header { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; }
        .healix-title { color: var(--text); font-size: 1.1rem; font-weight: 600; margin: 0; }
        .healix-subtitle { color: var(--muted); font-size: 0.9rem; margin-top: 2px; }

        /* Messages */
        [data-testid=\"stChatMessage\"] { max-width: 820px; margin: 8px auto; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface); }
        [data-testid=\"stChatMessage\"][data-testid*=\"assistant\"] { border-left: 3px solid #444; }
        [data-testid=\"chatAvatar\"], .stChatMessageAvatar { display: none !important; }
        .role-label { font-weight: 600; margin-bottom: 4px; color: var(--text); }
        a { color: var(--text); text-decoration: underline; }
        ::selection { background: #444; color: var(--text); }

        /* Sidebar */
        [data-testid=\"stSidebar\"] { background: var(--sidebar) !important; }
        .sidebar-section { margin-bottom: 12px; }
        .sidebar-title { font-weight: 600; color: var(--text); margin-bottom: 6px; }
        .stat-item { display:flex; justify-content:space-between; padding:6px 0; border-bottom: 1px solid var(--border); }
        .stat-label { color: var(--muted); }
        .stat-value { color: var(--text); font-weight: 500; }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render minimal header"""
    st.markdown("""
        <div class="healix-header">
            <h1 class="healix-title">Healix</h1>
            <p class="healix-subtitle">Medical assistant (local, evidence-grounded)</p>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar_stats(orchestrator, retriever):
    """Render session statistics in sidebar (non-blocking, no heavy health checks)"""
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Session statistics</div>', unsafe_allow_html=True)
        
        # Calculate stats
        total_messages = len(st.session_state.get('messages', []))
        user_messages = sum(1 for m in st.session_state.get('messages', []) if m.get('role') == 'user')
        
        st.markdown(f"""
            <div class="stat-item">
                <span class="stat-label">Messages</span>
                <span class="stat-value">{total_messages}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Queries</span>
                <span class="stat-value">{user_messages}</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Model info (avoid heavy generation health check)
        try:
            model_name = getattr(orchestrator, 'model_name', 'Unknown')
            st.markdown(f"""
                <div class="stat-item">
                    <span class="stat-label">Model</span>
                    <span class="stat-value">{model_name}</span>
                </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass
        
        # Knowledge base info (no embedding model load)
        try:
            stats = retriever.get_stats()
            chunks = stats.get('total_chunks', 0)
            st.markdown(f"""
                <div class="stat-item">
                    <span class="stat-label">Knowledge Base</span>
                    <span class="stat-value">{chunks:,} docs</span>
                </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Quick actions</div>', unsafe_allow_html=True)
        
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("Export history", use_container_width=True):
            if st.session_state.get('messages'):
                # Create export data
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "messages": st.session_state.messages
                }
                st.download_button(
                    "Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"healix_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("No messages to export")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Settings
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)
        
        show_timings = st.checkbox("Show response timings", value=False)
        st.session_state.show_timings = show_timings
        
        show_sources = st.checkbox("Show knowledge sources", value=False)
        st.session_state.show_sources = show_sources

        show_reasoning = st.checkbox("Show reasoning summary (concise)", value=False)
        st.session_state.show_reasoning = show_reasoning
        try:
            os.environ["BAYMAX_INCLUDE_REASONING_SUMMARY"] = "1" if show_reasoning else "0"
        except Exception:
            pass

        show_progress = st.checkbox("Show progress graphs", value=False)
        st.session_state.show_progress = show_progress
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_emergency_alert(emergency_data: Dict):
    """Render enhanced emergency alert"""
    level = emergency_data.get('emergency_level', 'UNKNOWN')
    message = emergency_data.get('message', 'Medical emergency detected')
    categories = emergency_data.get('detected_categories', [])
    score = emergency_data.get('severity_score', 0)
    
    st.markdown(f"""
        <div class="emergency-alert">
            <div class="emergency-title">MEDICAL EMERGENCY DETECTED</div>
            <div class="emergency-message">
                <strong>Severity:</strong> {level} (Score: {score})<br/>
                <strong>Action Required:</strong> {message}<br/>
                {f'<strong>Categories:</strong> {", ".join(categories)}' if categories else ''}
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_medication_draft(candidates: List[Dict]):
    """Render medication draft with professional styling"""
    st.markdown("""
        <div class="medication-draft">
            <div class="draft-title">Medication draft — decision support only</div>
            <div class="draft-subtitle">This is not a prescription. Consult a healthcare professional.</div>
        </div>
    """, unsafe_allow_html=True)
    
    for candidate in candidates[:3]:
        drug_name = candidate.get('drug_name', 'Unknown')
        reason = candidate.get('reason', 'No reason provided')
        
        st.markdown(f"""
            <div class="med-candidate">
                <div class="med-name">{drug_name}</div>
                <div class="med-reason">{reason}</div>
            </div>
        """, unsafe_allow_html=True)


def _load_csv_safe(path: str, parse_date_col: str = "date"):
    try:
        import pandas as pd
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if parse_date_col in df.columns:
            try:
                df[parse_date_col] = pd.to_datetime(df[parse_date_col])
            except Exception:
                pass
        return df
    except Exception:
        return None


def render_progress_graphs():
    base_dir = os.path.join("data", "user_states")
    st.markdown("## Progress graphs")
    st.caption("Visualize patterns over time. Place CSVs in data/user_states/.")

    try:
        import altair as alt
        import pandas as pd
    except Exception:
        st.info("Charts require Altair. Install with: pip install altair")
        return

    # Energy over days
    with st.container():
        st.markdown("### Energy over days")
        p = os.path.join(base_dir, "energy.csv")
        df = _load_csv_safe(p)
        if df is None or not set(["date", "energy"]).issubset(df.columns):
            st.info("Missing data/user_states/energy.csv. Expected columns: date, energy. Example:\n\n" +
                    "date,energy\n2025-11-01,6\n2025-11-02,7\n2025-11-03,5")
        else:
            dfx = df.sort_values("date").copy()
            chart = alt.Chart(dfx).mark_line(point=True).encode(
                x="date:T", y=alt.Y("energy:Q", title="Energy (0–10)"), tooltip=["date:T", "energy:Q"]
            ).properties(height=220)
            st.altair_chart(chart, use_container_width=True)

    # Headache frequency trend
    with st.container():
        st.markdown("### Headache frequency trend")
        p = os.path.join(base_dir, "headache.csv")
        df = _load_csv_safe(p)
        if df is None:
            st.info("Missing data/user_states/headache.csv. Expected columns: date and either count or had_headache (0/1). Example:\n\n" +
                    "date,count\n2025-11-01,1\n2025-11-02,0\n2025-11-03,1")
        else:
            dfx = df.copy()
            if "count" in dfx.columns:
                dfx["freq"] = pd.to_numeric(dfx["count"], errors="coerce").fillna(0)
            elif "had_headache" in dfx.columns:
                dfx["freq"] = pd.to_numeric(dfx["had_headache"], errors="coerce").fillna(0)
            else:
                st.info("Expected 'count' or 'had_headache' column.")
                dfx = None
            if dfx is not None:
                dfx = dfx.sort_values("date").copy()
                try:
                    dfx["rolling7"] = dfx["freq"].rolling(window=7, min_periods=1).mean()
                except Exception:
                    dfx["rolling7"] = dfx["freq"]
                line_raw = alt.Chart(dfx).mark_line(color="#999").encode(x="date:T", y=alt.Y("freq:Q", title="Headache frequency"))
                line_roll = alt.Chart(dfx).mark_line(color="#fff").encode(x="date:T", y="rolling7:Q")
                st.altair_chart((line_raw + line_roll).properties(height=220), use_container_width=True)

    # Sleep vs symptoms
    with st.container():
        st.markdown("### Sleep vs symptoms")
        p = os.path.join(base_dir, "sleep_symptoms.csv")
        df = _load_csv_safe(p)
        if df is None or not set(["sleep_hours", "symptom_score"]).issubset(df.columns):
            st.info("Missing data/user_states/sleep_symptoms.csv. Expected columns: date, sleep_hours, symptom_score. Example:\n\n" +
                    "date,sleep_hours,symptom_score\n2025-11-01,6.0,7\n2025-11-02,7.5,5\n2025-11-03,8.0,4")
        else:
            dfx = df.copy()
            try:
                dfx["sleep_hours"] = pd.to_numeric(dfx["sleep_hours"], errors="coerce")
                dfx["symptom_score"] = pd.to_numeric(dfx["symptom_score"], errors="coerce")
            except Exception:
                pass
            chart = alt.Chart(dfx).mark_circle(size=80).encode(
                x=alt.X("sleep_hours:Q", title="Sleep (hours)"),
                y=alt.Y("symptom_score:Q", title="Symptoms (higher=worse)"),
                tooltip=["date:T", "sleep_hours:Q", "symptom_score:Q"],
            ).properties(height=240)
            st.altair_chart(chart, use_container_width=True)

    # Stress vs pain correlations
    with st.container():
        st.markdown("### Stress vs pain")
        p = os.path.join(base_dir, "stress_pain.csv")
        df = _load_csv_safe(p)
        if df is None or not set(["stress", "pain"]).issubset(df.columns):
            st.info("Missing data/user_states/stress_pain.csv. Expected columns: date, stress, pain. Example:\n\n" +
                    "date,stress,pain\n2025-11-01,6,5\n2025-11-02,7,6\n2025-11-03,5,4")
        else:
            dfx = df.copy()
            try:
                corr = pd.to_numeric(dfx["stress"], errors="coerce").corr(pd.to_numeric(dfx["pain"], errors="coerce"))
                st.caption(f"Correlation (stress vs pain): {corr:.2f}" if pd.notna(corr) else "Correlation unavailable")
            except Exception:
                st.caption("Correlation unavailable")
            chart = alt.Chart(dfx).mark_circle(size=80).encode(
                x=alt.X("stress:Q", title="Stress"), y=alt.Y("pain:Q", title="Pain"), tooltip=["date:T", "stress:Q", "pain:Q"]
            ).properties(height=240)
            st.altair_chart(chart, use_container_width=True)


def render_footer():
    """Render professional footer disclaimer"""
    st.markdown("""
        <div class="footer-disclaimer">
            <p class="disclaimer-text">
                <strong>Medical Disclaimer:</strong> Healix provides information for educational purposes only.
                It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek
                the advice of a qualified health provider with any questions regarding a medical condition. In case of emergency, call 911.
            </p>
        </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Healix - AI Medical Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject professional styles
    inject_professional_styles()
    
    # Render header
    render_header()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'show_timings' not in st.session_state:
        st.session_state.show_timings = False
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = False
    if 'services_loaded' not in st.session_state:
        st.session_state.services_loaded = False
    
    # Load services (only show spinner if not already loaded)
    retriever = None
    orchestrator = None
    
    if not st.session_state.services_loaded:
        with st.spinner("🔄 Loading AI services... This may take 10-15 seconds..."):
            try:
                retriever = _load_retriever()
                orchestrator = _load_orchestrator()
                st.session_state.services_loaded = True
            except Exception as e:
                st.error(f"❌ Service initialization failed: {e}")
                st.info("💡 Make sure you're running via start_system.bat with environment variables set")
                st.code(f"Error details: {str(e)}", language="text")
                st.stop()
    else:
        # Services already loaded (cached)
        try:
            retriever = _load_retriever()
            orchestrator = _load_orchestrator()
        except Exception as e:
            st.error(f"❌ Error accessing services: {e}")
            st.session_state.services_loaded = False
            st.rerun()
    
    # Render sidebar (after services loaded)
    try:
        render_sidebar_stats(orchestrator, retriever)
    except Exception as e:
        # Don't block UI if sidebar fails
        with st.sidebar:
            st.warning(f"Sidebar stats unavailable")
    
    # Optional progress graphs
    if st.session_state.get("show_progress"):
        render_progress_graphs()

    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        # Render existing messages
        for message in st.session_state.messages:
            role = message.get("role")
            content = message.get("content")
            
            if role == "user":
                with st.chat_message("user", avatar=None):
                    st.markdown(f"<div class='role-label'>User</div>", unsafe_allow_html=True)
                    st.markdown(str(content))
            else:
                with st.chat_message("assistant", avatar=None):
                    st.markdown(f"<div class='role-label'>Healix</div>", unsafe_allow_html=True)
                    # Handle emergency
                    if isinstance(content, dict) and content.get('emergency'):
                        render_emergency_alert(content)
                    # Handle medication draft
                    elif isinstance(content, dict) and content.get('candidate_drugs'):
                        render_medication_draft(content.get('candidate_drugs', []))
                    # Handle regular response
                    else:
                        text = _extract_display_text(content)
                        if text:
                            st.markdown(text)
                            
                            # Show timings if enabled
                            if st.session_state.show_timings and isinstance(content, dict):
                                timings = content.get('timings', {})
                                if timings:
                                    st.caption(f"Retrieval: {timings.get('retrieval_s', 0):.2f}s | Generation: {timings.get('generation_s', 0):.2f}s | Total: {timings.get('total_s', 0):.2f}s")
                            
                            # Show sources if enabled
                            if st.session_state.show_sources and isinstance(content, dict):
                                passages_used = content.get('passages_used', 0)
                                if passages_used > 0:
                                    st.caption(f"Sources: {passages_used} medical documents")
                        else:
                            st.info("_No response generated._")
    
    # Chat input
    user_input = st.chat_input("Ask a follow up")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=None):
            st.markdown(f"<div class='role-label'>User</div>", unsafe_allow_html=True)
            st.markdown(user_input)
        
        # Generate response
        try:
            with st.chat_message("assistant", avatar=None):
                st.markdown(f"<div class='role-label'>Healix</div>", unsafe_allow_html=True)
                import time as _t
                
                # Emergency check
                emergency_check = orchestrator._detect_advanced_emergency(user_input)
                if emergency_check["is_emergency"] and emergency_check["emergency_level"] in ["CRITICAL", "HIGH"]:
                    payload = {
                        "emergency": True,
                        "emergency_level": emergency_check["emergency_level"],
                        "severity_score": emergency_check["severity_score"],
                        "message": emergency_check["emergency_message"],
                        "detected_categories": emergency_check["categories_detected"],
                        "immediate_action_required": True,
                    }
                    render_emergency_alert(payload)
                    st.session_state.messages.append({"role": "assistant", "content": payload})
                else:
                    # Route intent
                    intent = orchestrator.route_intent(user_input)
                    
                    if intent == "medication":
                        # Medication draft
                        with st.spinner("Analyzing medication options..."):
                            resp = orchestrator.generate_response(user_input, conversation_context=st.session_state.messages)
                        
                        # Attach safety checks if available
                        try:
                            cands = resp.get("candidate_drugs") or resp.get("draft", {}).get("candidate_drugs") or []
                            if _run_checks and isinstance(cands, list):
                                for c in cands:
                                    dn = str(c.get("drug_name", "")).strip()
                                    c["checks"] = _run_checks("unknown", dn, {"current_meds": [], "demographics": {}})
                        except Exception:
                            pass
                        
                        # Audit
                        try:
                            audit_log("chat_draft", {"intent": intent, "candidates": len(cands) if isinstance(cands, list) else 0})
                        except Exception:
                            pass
                        
                        render_medication_draft(cands if isinstance(cands, list) else [])
                        st.session_state.messages.append({"role": "assistant", "content": resp})
                    else:
                        # Streaming generation
                        t0 = _t.time()
                        
                        # Classify complexity
                        complexity = orchestrator.classify_query_complexity(user_input)
                        
                        # Retrieval (skip for greetings)
                        if complexity == "greeting":
                            passages = []
                        else:
                            passages = retriever.retrieve(query=user_input, k=10, min_score=0.25)
                        t1 = _t.time()
                        # Prepare prompt (metadata is useful even if we short-circuit for sources)
                        full_prompt, query_type, specialties = orchestrator.prepare_stream_prompt(
                            user_text=user_input,
                            retrieved_passages=passages,
                            symptom_data=None,
                            conversation_context=st.session_state.messages,
                            user_mode="patient"
                        )


                        # Identity one-liner
                        _tlow = user_input.strip().lower()
                        if any(re.match(p, _tlow) for p in [r"^who\s+are\s+you\??$", r"^what\s+are\s+you\??$", r"^who\s+is\s+healix\??$", r"^what\s+is\s+healix\??$", r"^explain\s+yourself\.?$", r"^tell\s+me\s+about\s+(you|yourself)\.?$", r"^describe\s+yourself\.?$"]):
                            ident = "I'm Healix, a medical companion here to help you understand your health."
                            payload = {
                                "emergency": False,
                                "response": {"medical_assessment": ident, "confidence_level": 0.98},
                                "specialties": [],
                                "query_type": "identity",
                                "passages_used": 0,
                                "model_used": orchestrator.model_name,
                            }
                            st.session_state.messages.append({"role": "assistant", "content": payload})
                            st.rerun()

                        # If user explicitly asked to show sources, return a concise reference list (no URLs)
                        if any(kw in user_input.lower() for kw in ["show sources", "where is this from", "cite this", "references", "source list"]):
                            uniq = []
                            seen = set()
                            for p in passages:
                                name = p.get('source') or 'Unknown'
                                cat = p.get('category') or ''
                                key = f"{name}|{cat}"
                                if key in seen:
                                    continue
                                seen.add(key)
                                uniq.append(f"• {name}" + (f" — {cat}" if cat else ""))
                            src_text = "\n".join(uniq) if uniq else "No sources available."
                            payload = {
                                "emergency": False,
                                "response": {"medical_assessment": src_text, "confidence_level": 0.85},
                                "specialties": specialties,
                                "query_type": "sources",
                                "passages_used": len(passages),
                                "model_used": orchestrator.model_name,
                            }
                            st.session_state.messages.append({"role": "assistant", "content": payload})
                            st.rerun()
                        
                        # Token allocation: honor global max tokens for full answers
                        try:
                            max_tok = int(os.getenv("BAYMAX_GEN_MAX_TOKENS", "900") or 900)
                        except Exception:
                            max_tok = 900
                        
                        # Stream response
                        placeholder = st.empty()
                        buf = ""
                        
                        with st.spinner("Thinking..."):
                            try:
                                t2s = _t.time()
                                for tok in orchestrator.stream_generate(
                                    full_prompt,
                                    max_tokens=max_tok,
                                    temp=0.15,
                                    top_k=int(os.getenv("BAYMAX_GEN_TOP_K", "10") or 10),
                                    top_p=float(os.getenv("BAYMAX_GEN_TOP_P", "0.9") or 0.9),
                                ):
                                    buf += tok
                                    placeholder.markdown(buf + "▌")
                                t2e = _t.time()
                                
                                # Final clean output
                                placeholder.markdown(buf)
                            except Exception as e:
                                t2e = _t.time()
                                st.error(f"Generation error: {e}")
                        
                        cleaned = orchestrator._extract_conversational_text(buf) if buf else ""
                        if not cleaned and buf:
                            cleaned = buf.strip()
                        
                        # Build payload
                        payload = {
                            "emergency": False,
                            "response": {"medical_assessment": cleaned, "confidence_level": 0.85},
                            "specialties": specialties,
                            "query_type": query_type,
                            "passages_used": len(passages),
                            "model_used": orchestrator.model_name,
                            "timings": {
                                "retrieval_s": round(t1 - t0, 3),
                                "generation_s": round(t2e - t2s, 3),
                                "total_s": round((t1 - t0) + (t2e - t2s), 3)
                            }
                        }
                        
                        # Show timings/sources if enabled
                        if st.session_state.show_timings:
                            st.caption(f"Retrieval: {payload['timings']['retrieval_s']:.2f}s | Generation: {payload['timings']['generation_s']:.2f}s | Total: {payload['timings']['total_s']:.2f}s")
                        
                        if st.session_state.show_sources and len(passages) > 0:
                            st.caption(f"Sources: {len(passages)} medical documents")
                        
                        # Audit
                        try:
                            audit_log("chat_answer", {"intent": intent, **payload.get("timings", {})})
                        except Exception:
                            pass
                        
                        st.session_state.messages.append({"role": "assistant", "content": payload})
        except Exception as e:
            with st.chat_message("assistant", avatar=None):
                st.error(f"❌ Generation error: {e}")
    
    # Footer removed by request (no disclaimers)


if __name__ == "__main__":
    main()
