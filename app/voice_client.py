'''import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("aura.voice")

class ExotelClient:
    def __init__(self):
        # Load credentials from .env
        self.sid = os.getenv("EXOTEL_SID")
        self.api_key = os.getenv("EXOTEL_API_KEY")
        self.api_token = os.getenv("EXOTEL_TOKEN")
        self.exophone = os.getenv("EXOTEL_EXOPHONE")
        self.doctor_number = os.getenv("EMERGENCY_ADMIN_NUMBER") 

    def trigger_call(self, patient_number: str):
        """
        Initiates a bridge call: Exotel calls Patient -> Connects to Doctor.
        """
        if not all([self.sid, self.api_key, self.api_token, self.exophone, self.doctor_number]):
            logger.error("❌ Exotel credentials missing in .env")
            return {"status": "error", "message": "Server configuration error. Credentials missing."}

        # Exotel API URL
        url = f"https://api.exotel.com/v1/Accounts/{self.sid}/Calls/connect"

        payload = {
            "From": patient_number,       # 1. Call the Patient first
            "To": self.doctor_number,     # 2. Connect them to the Doctor
            "CallerId": self.exophone,    # The number displayed on phone
            "CallType": "trans"           # Transactional call
        }

        try:
            response = requests.post(url, data=payload, auth=(self.api_key, self.api_token))
            
            if response.status_code == 200:
                logger.info(f"✅ Call triggered to {patient_number}")
                return {"status": "success", "message": "Calling your phone now..."}
            else:
                logger.error(f"❌ Exotel Error: {response.text}")
                return {"status": "error", "message": "Failed to connect call."}

        except Exception as e:
            logger.exception("Exotel Connection Failed")
            return {"status": "error", "message": str(e)}

voice_client = ExotelClient()'''




import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("aura.voice")

class ExotelClient:
    def __init__(self):
        # Load credentials from .env
        self.sid = os.getenv("EXOTEL_SID")
        self.api_key = os.getenv("EXOTEL_API_KEY")
        self.api_token = os.getenv("EXOTEL_TOKEN")
        self.exophone = os.getenv("EXOTEL_EXOPHONE")
        self.support_number = os.getenv("EMERGENCY_ADMIN_NUMBER") 

    def trigger_call(self, patient_number: str):
        """
        MOCK MODE: Simulates a call for demo purposes because
        Exotel requires KYC verification for real calls.
        """
        logger.info(f"🔄 MOCK CALL INITIATED...")
        logger.info(f"📞 FROM: {patient_number}")
        logger.info(f"📞 TO (support): {self.support_number}")
        logger.info(f"✅ Exotel API: [Mock Success Response 200 OK]")
        
        # We return a fake "Success" so the Frontend thinks it worked
        return {"status": "success", "message": "Calling your phone now (Mock)..."}

        # --- REAL CODE (Commented out for Demo) ---
        # if not all([self.sid, self.api_key, self.api_token, self.exophone, self.doctor_number]):
        #     logger.error("❌ Exotel credentials missing in .env")
        #     return {"status": "error", "message": "Server configuration error."}

        # url = f"https://api.exotel.com/v1/Accounts/{self.sid}/Calls/connect"
        # payload = {
        #     "From": patient_number,
        #     "To": self.doctor_number,
        #     "CallerId": self.exophone,
        #     "CallType": "trans"
        # }

        # try:
        #     response = requests.post(url, data=payload, auth=(self.api_key, self.api_token))
        #     if response.status_code == 200:
        #         return {"status": "success", "message": "Calling your phone now..."}
        #     else:
        #         logger.error(f"❌ Exotel Error: {response.text}")
        #         return {"status": "error", "message": "Failed to connect call."}
        # except Exception as e:
        #     return {"status": "error", "message": str(e)}

voice_client = ExotelClient()