# app/clinic_api.py
import datetime
import random
import threading
import uuid
import logging
from typing import Dict, List, Optional, Any
import re
import copy

logger = logging.getLogger("aura.clinic")
logging.basicConfig(level=logging.INFO)

# Basic PII patterns to redact from meta
_PII_RE = re.compile(r"(\+?\d{7,15}|[^\s]+@[^\s]+\.[^\s]+)")

def _redact_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a shallow-copied meta with obvious PII redacted."""
    if not meta:
        return {}
    safe = {}
    for k, v in meta.items():
        try:
            s = str(v)
            if _PII_RE.search(s):
                safe[k] = "[REDACTED]"
            else:
                safe[k] = v
        except Exception:
            safe[k] = "[REDACTED]"
    return safe

class ClinicManager:
    def __init__(self):
        logger.info("Clinic Manager: Initializing...")
        self._lock = threading.Lock()
        # appointments: {date_str: {time_str: {doctor_name: booking_id_or_None}}}
        self.appointments: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
        # bookings: booking_id -> booking metadata
        self.bookings: Dict[str, Dict[str, Any]] = {}
        self.doctors = ["Dr. Smith", "Dr. Jones", "Dr. Lee", "Dr. Chen"]
        self.available_times = ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]
        logger.info("Clinic Manager: Ready.")

    def _generate_slots_for_day(self, date_str: str):
        """Generates random availability for a specific day (idempotent)."""
        with self._lock:
            if date_str in self.appointments:
                return
            # Deterministic per date during a run for reproducible behavior
            random.seed(hash(date_str) & 0xFFFFFFFF)
            self.appointments[date_str] = {}
            for time_slot in self.available_times:
                available_doctors = [doc for doc in self.doctors if random.random() > 0.3]  # ~70% free
                if available_doctors:
                    self.appointments[date_str][time_slot] = {doc: None for doc in available_doctors}
                    # Occasionally mark one as pre-booked to simulate load
                    if random.random() < 0.2:
                        chosen = random.choice(available_doctors)
                        self.appointments[date_str][time_slot][chosen] = str(uuid.uuid4())

    def get_available_slots(self, date_str: str) -> Dict[str, List[str]]:
        try:
            requested_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            if requested_date < datetime.date.today():
                return {}
        except ValueError:
            return {}

        self._generate_slots_for_day(date_str)

        with self._lock:
            day_slots = self.appointments.get(date_str, {})
            available: Dict[str, List[str]] = {}
            for time_slot in sorted(day_slots.keys()):
                doctors_map = day_slots[time_slot]
                free_docs = [doc for doc, booking in doctors_map.items() if booking is None]
                if free_docs:
                    available[time_slot] = sorted(free_docs)
            return available

    def book_appointment(self, date_str: str, time_str: str, doctor_name: str, meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            requested_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            if requested_date < datetime.date.today():
                return None
        except ValueError:
            return None

        with self._lock:
            if date_str not in self.appointments:
                self._generate_slots_for_day(date_str)

            day = self.appointments.get(date_str, {})
            if time_str not in day or doctor_name not in day[time_str]:
                return None # Slot or doctor invalid

            if day[time_str][doctor_name] is not None:
                return None  # already booked

            booking_id = str(uuid.uuid4())
            day[time_str][doctor_name] = booking_id
            
            safe_meta = _redact_meta(meta)
            booking = {
                "booking_id": booking_id,
                "date": date_str,
                "time": time_str,
                "doctor": doctor_name,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "meta": safe_meta
            }
            self.bookings[booking_id] = booking
            return copy.deepcopy(booking)

    def get_booking(self, booking_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            b = self.bookings.get(booking_id)
            return copy.deepcopy(b) if b else None

    def cancel_booking(self, booking_id: str) -> bool:
        with self._lock:
            booking = self.bookings.get(booking_id)
            if not booking:
                return False
            
            # Clear the appointment slot
            date_str = booking["date"]
            time_str = booking["time"]
            doctor = booking["doctor"]
            
            if date_str in self.appointments and time_str in self.appointments[date_str]:
                if self.appointments[date_str][time_str].get(doctor) == booking_id:
                    self.appointments[date_str][time_str][doctor] = None
            
            self.bookings.pop(booking_id, None)
            logger.info("Clinic Manager: Cancelled booking %s", booking_id)
            return True

clinic_manager = ClinicManager()