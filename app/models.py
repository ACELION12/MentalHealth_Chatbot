# app/models.py (patched)
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime
import re

class AgentState(str, Enum):
    ASSESSING = "ASSESSING"
    PROVIDING_SUPPORT = "PROVIDING_SUPPORT"
    TRIAGED_LOW = "TRIAGED_LOW"
    TRIAGED_MEDIUM = "TRIAGED_MEDIUM"
    TRIAGED_HIGH = "TRIAGED_HIGH"
    ESCALATING = "ESCALATING"
    BOOKING = "BOOKING"
    ENDED = "ENDED"

class UIState(str, Enum):
    CHAT = "CHAT"
    CONSENT = "CONSENT"
    ESCALATION_MODAL = "ESCALATION_MODAL"
    BOOKING_MODAL = "BOOKING_MODAL"
    FEEDBACK = "FEEDBACK"

def _recursive_scrub(value: Any) -> Any:
    """
    Recursively redact strings that look like emails or phone numbers.
    Returns a deep-copied structure with redactions applied.
    """
    email_phone_pattern = re.compile(r"(\+?\d{7,15}|[^\s]+@[^\s]+\.[^\s]+)")
    if value is None:
        return None
    if isinstance(value, str):
        if email_phone_pattern.search(value):
            return "[REDACTED]"
        return value
    if isinstance(value, dict):
        return {k: _recursive_scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_recursive_scrub(v) for v in value]
    return value

class Message(BaseModel):
    """
    Message model — accepts either:
    - {'sender': 'user'|'agent'|'system', 'text': '...'}
    or
    - {'role': 'user'|'assistant'|'system', 'content': '...'}
    and normalizes to sender/text.
    """
    sender: str  # "user" or "agent" or "system"
    text: str = Field(..., min_length=0, max_length=5000)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @root_validator(pre=True)
    def accept_multiple_shapes_and_normalize(cls, values):
        # if frontend or other code uses role/content, accept those keys
        if "role" in values and "content" in values and ("sender" not in values or "text" not in values):
            role = values.get("role")
            # normalize assistant -> agent if needed
            if role == "assistant":
                role = "agent"
            values["sender"] = role
            values["text"] = values.get("content")
        # also accept 'sender'/'text' as-is
        return values

    @validator("sender")
    def sender_must_be_user_agent_or_system(cls, v):
        if v not in ("user", "agent", "system"):
            # map some common synonyms
            if v == "assistant":
                return "agent"
            raise ValueError("sender must be 'user', 'agent' or 'system'")
        return v

    @validator("timestamp")
    def timestamp_must_be_iso(cls, v):
        try:
            datetime.fromisoformat(v)
        except Exception:
            raise ValueError("timestamp must be an ISO-formatted datetime string")
        return v

class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=0, max_length=5000)
    session_id: Optional[str] = None
    # accept either enum or plain string from frontend
    current_agent_state: Union[AgentState, str] = AgentState.ASSESSING
    conversation_history: List[Message] = Field(default_factory=list)

    @validator("session_id")
    def session_id_non_empty_if_provided(cls, v):
        if v is not None and (not isinstance(v, str) or v.strip() == ""):
            raise ValueError("session_id, if provided, must be a non-empty string")
        return v

    @validator("current_agent_state", pre=True, always=True)
    def coerce_current_state(cls, v):
        if v is None:
            return AgentState.ASSESSING
        # if already an AgentState, keep it
        if isinstance(v, AgentState):
            return v
        # try to coerce from string
        if isinstance(v, str):
            vv = v.strip().upper()
            try:
                return AgentState(vv)
            except Exception:
                # fallback safe default
                return AgentState.ASSESSING
        return AgentState.ASSESSING

class ChatResponse(BaseModel):
    agent_response: str = Field(..., min_length=0, max_length=20000)
    session_id: str
    current_agent_state: AgentState
    ui_state: UIState
    debug_info: Optional[Dict[str, Any]] = None

    class Config:
        extra = "ignore"

    @validator("debug_info", pre=True, always=True)
    def scrub_debug_info(cls, v):
        if not v:
            return None
        try:
            safe = _recursive_scrub(v)
            return safe
        except Exception:
            return None
