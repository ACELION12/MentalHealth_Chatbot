from typing import Dict, Any, List, Optional, Union
import threading
import logging
from datetime import datetime

# Import components
from .nlu import nlu_manager
from .llm_client import llm_client
from .rag import rag_system
from .clinic_api import clinic_manager
from .models import AgentState, UIState

logger = logging.getLogger("aura.state_machine")
logging.basicConfig(level=logging.INFO)

_session_lock = threading.Lock()
session_data_store: Dict[str, Dict[str, Any]] = {}

# --- GLOBAL ALERT LOG FOR DASHBOARD ---
# Stores high-risk incidents for the admin dashboard
risk_alert_log: List[Dict[str, Any]] = []

HIGH_RISK_THRESHOLD = 0.85
SUICIDE_KEYWORDS = [
    "kill myself", "want to die", "end my life", "suicide",
    "can't go on", "i'm done", "shouldn't be here"
]
EMERGENCY_PHONE = "+91-7893223741"

def get_session_data(session_id: str) -> Dict[str, Any]:
    with _session_lock:
        if session_id not in session_data_store:
            session_data_store[session_id] = {
                "chat_history": [],
                "last_triage_score": 0.0,
            }
        return session_data_store[session_id]

def update_chat_history(session_id: str, role: str, content: str, max_len: int = 10):
    session = get_session_data(session_id)
    with _session_lock:
        session["chat_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        session["chat_history"] = session["chat_history"][-max_len:]

def _is_high_risk(score: float, text: str) -> bool:
    t = text.lower()
    keyword_hit = any(k in t for k in SUICIDE_KEYWORDS)
    return score >= HIGH_RISK_THRESHOLD or keyword_hit

def process_message(user_message: str, current_state: Optional[Union[str, AgentState]], session_id: str) -> Dict[str, Any]:
    logger.info("Processing message for session=%s", session_id)
    session = get_session_data(session_id)
    user_message = (user_message or "").strip()

    if not user_message:
        return {
            "agent_response": "I didn’t quite catch that. Could you say it again?",
            "current_agent_state": AgentState.ASSESSING,
            "ui_state": UIState.CHAT
        }

    lower_msg = user_message.lower()

    # -----------------------------------------------------------
    # 🛑 1. THE "HARD STOP" (TERMINATE EMERGENCY FLOW)
    # -----------------------------------------------------------
    termination_triggers = [
        "i am safe now", 
        "please return to normal chat",
        "no, i'm okay",
        "no, thanks",
        "no thanks"
    ]
    
    if any(trigger in lower_msg for trigger in termination_triggers):
        logger.info("🛑 Safety Override: Terminating emergency flow. Showing backup message.")
        
        backup_message = "I understand. I'm still here with you. You can talk to me whenever you're ready."
        
        update_chat_history(session_id, "user", user_message)
        update_chat_history(session_id, "assistant", backup_message)
        
        return {
            "agent_response": backup_message,   
            "current_agent_state": AgentState.ASSESSING, 
            "ui_state": UIState.CHAT,           
            "debug_info": {"bypass": True, "reason": "user_terminated_emergency"}
        }

    # -----------------------------------------------------------
    # ⚡ 2. CONNECTION BYPASS (NEW ADDITION)
    # If user asks to connect, open Call Interface directly.
    # Skips Gemini completely.
    # -----------------------------------------------------------
    connection_triggers = [
        "connect me to a clinician", "connect to clinician",
        "call the doctor", "call support", "speak to human",
        "yes, connect me"
    ]
    if any(trigger in lower_msg for trigger in connection_triggers):
        logger.info("⚡ Connection Bypass: Triggering Call Interface directly.")
        
        static_reply = "I can connect you immediately. Please enter your number below."
        
        update_chat_history(session_id, "user", user_message)
        update_chat_history(session_id, "assistant", static_reply)

        return {
            "agent_response": static_reply,
            "modal_text": "We can connect you to our emergency support line immediately.",
            "current_agent_state": AgentState.ESCALATING,
            "ui_state": UIState.ESCALATION_MODAL, # This opens the Call Interface
            "debug_info": {"bypass": True, "reason": "direct_connection_request"}
        }
    # -----------------------------------------------------------

    # 3. NLU (The "Sensor")
    try:
        nlu = nlu_manager.run_nlu_pipeline(user_message)
        intent = nlu.get("intent", "unknown")
        logger.info(f"NLU Analysis: {intent}")
    except Exception as e:
        logger.error(f"NLU Error: {e}")
        nlu = {"risk_score": 0.0, "intent": "unknown"}
        intent = "unknown"

    risk_score = float(nlu.get("risk_score", 0.0))
    session["last_triage_score"] = risk_score

    # 4. EMERGENCY BYPASS (Dashboard Logging + Split Message)
    if _is_high_risk(risk_score, user_message):
        
        # --- A. Log to Dashboard ---
        alert_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "user_message": user_message,
            "risk_score": risk_score,
            "status": "UNRESOLVED"
        }
        risk_alert_log.insert(0, alert_entry) # Add to top of list
        # ---------------------------

        # --- B. Prepare Split Messages ---
        full_crisis_message = (
            f"I’m really sorry you’re feeling this overwhelmed. "
            f"Please contact this emergency number immediately: {EMERGENCY_PHONE}. "
            f"Are you safe right now?"
        )
        chat_placeholder = "⚠️ Emergency Alert Triggered. Please see the contact card above."

        update_chat_history(session_id, "user", user_message)
        update_chat_history(session_id, "assistant", chat_placeholder)
        
        return {
            "agent_response": chat_placeholder,
            "modal_text": full_crisis_message,
            "current_agent_state": AgentState.ESCALATING,
            "ui_state": UIState.ESCALATION_MODAL, 
            "debug_info": {"bypass": True}
        }

    # 5. BOOKING BYPASS
    if intent == "request_appointment" or "book" in lower_msg:
        static_reply = "I can help with that. I've opened the scheduler for you—please select a time that works best."
        update_chat_history(session_id, "user", user_message)
        update_chat_history(session_id, "assistant", static_reply)
        logger.info("⚡ BYPASS: Booking intent detected. Skipping Gemini.")
        
        return {
            "agent_response": static_reply,
            "current_agent_state": AgentState.BOOKING,
            "ui_state": UIState.BOOKING_MODAL,
            "debug_info": {"bypass": True, "intent": "booking"}
        }

    # 6. NORMAL CHAT (Gemini)
    rag_context = ""
    try:
        search_result = rag_system.retrieve_info(user_message)
        rag_context = search_result.get("summary", "")
    except Exception:
        pass

    base_system_prompt = (
        "You are Aura, a compassionate mental health support assistant. "
        "Listen carefully, validate feelings, and be supportive. "
        "Keep responses concise (under 3 sentences)."
    )
    
    if rag_context:
        base_system_prompt += f"\n\nRELEVANT INFO:\n{rag_context}"

    # --- SAFETY BLOCK FOR LLM ---
    try:
        reply = llm_client.generate_response(
            system_instruction=base_system_prompt,
            history=session["chat_history"][-6:],
            user_message=user_message
        )
    except Exception as e:
        logger.error(f"LLM Client Failed: {e}")
        # Fallback response if Gemini fails
        reply = "I'm listening, but I'm having a little trouble connecting to my server right now. Can you tell me more about how you're feeling?"

    update_chat_history(session_id, "user", user_message)
    update_chat_history(session_id, "assistant", reply)

    return {
        "agent_response": reply,
        "current_agent_state": AgentState.ASSESSING,
        "ui_state": UIState.CHAT,
        "debug_info": {"rag_used": bool(rag_context)}
    }