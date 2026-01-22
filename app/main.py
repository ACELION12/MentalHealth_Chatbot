# app/main.py
import uuid
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# --- IMPORTS ---
# We import risk_alert_log so the dashboard can see the alerts
from .state_machine import process_message, get_session_data, risk_alert_log 
from .clinic_api import clinic_manager
from .models import AgentState, UIState 
from .voice_client import voice_client 

load_dotenv()

logger = logging.getLogger("aura")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

app = FastAPI(title="Mental Health Agent API Backend", description="API for managing conversational flow.")

# ---------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_message: str
    current_agent_state: Optional[str] = None

class ChatResponse(BaseModel):
    agent_response: str
    session_id: str
    current_agent_state: str
    ui_state: str
    modal_text: Optional[str] = None 
    debug_info: Optional[Dict[str, Any]] = None

class CallRequest(BaseModel):
    phone_number: str

# ---------------------------------------------------------
# FRONTEND SETUP
# ---------------------------------------------------------
ROOT = Path(__file__).parent.parent
frontend_dir = ROOT / "frontend"

if frontend_dir.exists():
    static_dir = frontend_dir
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(str(index_path))
else:
    logger.warning("Frontend directory not found.")

# Add dashboard if it exists
dashboard_path = frontend_dir / "dashboard.html"
if dashboard_path.exists():
    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard():
        return FileResponse(str(dashboard_path))

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}

@app.get("/status")
async def read_root():
    return {"status": "Mental Health Agent API is running"}

# ---------------------------------------------------------
# MAIN CHAT ENDPOINT
# ---------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    try:
        logger.info("[%s] User message: '%s' (state=%s)", session_id, request.user_message, request.current_agent_state)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop:
            response_data = await loop.run_in_executor(
                executor, process_message, request.user_message, request.current_agent_state, session_id
            )
        else:
            response_data = process_message(request.user_message, request.current_agent_state, session_id)

        return ChatResponse(
            agent_response=response_data["agent_response"],
            session_id=session_id,
            current_agent_state=response_data.get("current_agent_state", "ASSESSING"),
            ui_state=response_data.get("ui_state", "CHAT"),
            modal_text=response_data.get("modal_text"),
            debug_info=response_data.get("debug_info", {})
        )
    except Exception as e:
        logger.exception("Error handling chat for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# 📞 EMERGENCY CALL ENDPOINT
# ---------------------------------------------------------
@app.post("/api/emergency/call")
async def api_trigger_call(req: CallRequest):
    """
    Triggers the Voice Client (Mock or Exotel).
    """
    if not req.phone_number:
        raise HTTPException(status_code=400, detail="Phone number required")
    
    logger.info(f"📞 Received call request for: {req.phone_number}")
    result = voice_client.trigger_call(req.phone_number)
    
    if result["status"] == "error":
        logger.error(f"Call failed: {result['message']}")
        raise HTTPException(status_code=500, detail=result["message"])
        
    return result

# ---------------------------------------------------------
# 📊 ADMIN DASHBOARD ENDPOINT
# ---------------------------------------------------------
@app.get("/api/admin/alerts")
async def get_admin_alerts():
    """
    Returns the live list of high-risk incidents for the dashboard.
    """
    return {"alerts": risk_alert_log}

# ---------------------------------------------------------
# 📅 CLINIC ENDPOINTS (Booking + Mock Email/Calendar)
# ---------------------------------------------------------
class BookRequest(BaseModel):
    date: str
    time: str
    doctor: str
    meta: Optional[dict] = None

@app.get("/api/slots")
async def api_get_slots(date: str = Query(..., description="Date in format YYYY-MM-DD")):
    slots = clinic_manager.get_available_slots(date)
    return {"date": date, "slots": slots}

@app.post("/api/book")
async def api_book(req: BookRequest):
    # This now calls the updated clinic_manager with mock email/calendar support
    booking = clinic_manager.book_appointment(req.date, req.time, req.doctor, meta=req.meta)
    if booking is None:
        raise HTTPException(status_code=400, detail="Unable to book the requested slot.")
    return {"booking": booking}

@app.get("/api/booking/{booking_id}")
async def api_get_booking(booking_id: str):
    booking = clinic_manager.get_booking(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return {"booking": booking}

@app.post("/api/booking/{booking_id}/cancel")
async def api_cancel_booking(booking_id: str):
    ok = clinic_manager.cancel_booking(booking_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Booking not found or already cancelled")
    return {"status": "cancelled", "booking_id": booking_id}

# ---------------------------------------------------------
# DEBUG ENDPOINT
# ---------------------------------------------------------
@app.get("/api/session/{session_id}")
async def api_get_session(session_id: str):
    try:
        session = get_session_data(session_id)
        chat_history = session.get("chat_history", [])
        sanitized = [{"role": m.get("role"), "content": m.get("content"), "timestamp": m.get("timestamp")} for m in chat_history]
        return {
            "session_id": session_id,
            "last_triage_score": session.get("last_triage_score"),
            "chat_history": sanitized
        }
    except Exception:
        logger.exception("Failed to read session %s", session_id)
        raise HTTPException(status_code=500, detail="Internal server error")