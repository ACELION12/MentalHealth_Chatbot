# app/nlu.py
'''from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading
import re
import logging
import time

logger = logging.getLogger("aura.nlu")
logging.basicConfig(level=logging.INFO)

# Configurable model names (swap if you have a lighter/heavier model)
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
RISK_MODEL_PATH = "app/data/my_custom_risk_model"

# Crisis / suicide keyword list
SUICIDE_KEYWORDS = [
    "kill myself", "i will kill myself", "i'm going to kill myself", "im going to kill myself",
    "want to die", "end my life", "i have a plan", "i have a plan to", "suicide",
    "die by suicide", "i want to die", "cant go on", "can't go on", "cannot go on",
    "want to end", "shouldn't be here", "shouldnt be here", "i'm done", "i am done"
]

# Thread-safe lazy loader for Hugging Face pipelines
class _LazyPipelines:
    _lock = threading.Lock()
    sentiment = None
    zero_shot = None
    _last_load_attempt = 0

    @classmethod
    def ensure_loaded(cls):
        # avoid repeated rapid attempts
        now = time.time()
        if cls.sentiment is not None and cls.zero_shot is not None:
            return
        with cls._lock:
            # another check inside lock
            if cls.sentiment is not None and cls.zero_shot is not None:
                return
            # throttle re-attempts to 30s
            if now - cls._last_load_attempt < 30:
                return
            cls._last_load_attempt = now
            try:
                from transformers import pipeline
                logger.info("NLU: Loading HF pipelines (this may take a moment)...")
                # you can pass device=0 for GPU if available
                cls.sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
                cls.zero_shot = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
                logger.info("NLU: Pipelines loaded.")
            except Exception as e:
                logger.exception("Failed to load HF pipelines: %s", e)
                cls.sentiment = None
                cls.zero_shot = None

# Simple LRU-ish cache for NLU outputs
_nlu_cache: Dict[str, Dict[str, Any]] = {}
_NLU_CACHE_MAX = 512
_cache_lock = threading.Lock()

def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    with _cache_lock:
        return _nlu_cache.get(key)

def _cache_set(key: str, value: Dict[str, Any]):
    with _cache_lock:
        if key in _nlu_cache:
            _nlu_cache.pop(key, None)
        _nlu_cache[key] = value
        if len(_nlu_cache) > _NLU_CACHE_MAX:
            first = next(iter(_nlu_cache))
            _nlu_cache.pop(first, None)

@dataclass
class NLUManager:
    def detect_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Use HF pipeline if available; otherwise a light rule-based fallback.
        """
        _LazyPipelines.ensure_loaded()
        try:
            if _LazyPipelines.sentiment is None:
                t = (text or "").lower()
                negative_words = ("sad", "depress", "hopeless", "hate", "angry", "upset", "hurt", "worthless", "alone", "pain")
                positive_words = ("good", "ok", "fine", "better", "happy", "relieved", "well")
                if any(w in t for w in negative_words):
                    return {"label": "NEGATIVE", "score": 0.8}
                if any(w in t for w in positive_words):
                    return {"label": "POSITIVE", "score": 0.7}
                return {"label": "NEUTRAL", "score": 0.5}
            
            result = _LazyPipelines.sentiment(text)[0]
            return {"label": result.get("label"), "score": float(result.get("score", 0.0))}
        except Exception as e:
            logger.exception("Sentiment detection failed: %s", e)
            return {"label": "NEUTRAL", "score": 0.5}

    def detect_intent(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """
        Use zero-shot classification pipeline if available. Fallback to keywords.
        """
        _LazyPipelines.ensure_loaded()
        try:
            if _LazyPipelines.zero_shot is None:
                t = (text or "").lower()
                # simple heuristics
                if any(w in t for w in ("book", "appointment", "schedule", "clinician")):
                    return {"intent": "request_appointment", "score": 0.9}
                if any(w in t for w in ("help", "feel", "die", "kill", "suicide", "hurt")):
                    return {"intent": "express_distress", "score": 0.9}
                if any(w in t for w in ("how to", "what is", "where can", "resource")):
                    return {"intent": "request_info", "score": 0.8}
                if any(w in t for w in ("yes", "yeah", "sure", "ok")):
                    return {"intent": "confirm", "score": 0.7}
                if any(w in t for w in ("no", "nah", "not now")):
                    return {"intent": "deny", "score": 0.7}
                return {"intent": "general_chat", "score": 0.4}
            
            result = _LazyPipelines.zero_shot(text, candidate_labels=candidate_labels)
            best_intent = result['labels'][0]
            best_score = float(result['scores'][0])
            return {"intent": best_intent, "score": best_score}
        except Exception as e:
            logger.exception("Intent detection failed: %s", e)
            return {"intent": "unknown", "score": 0.0}

    def detect_emotions(self, text: str, emotion_labels: List[str], threshold: float = 0.45) -> List[Dict[str, Any]]:
        """
        Detects emotions using zero-shot classification (multi-label).
        """
        _LazyPipelines.ensure_loaded()
        try:
            if _LazyPipelines.zero_shot is None:
                # Basic fallback
                t = (text or "").lower()
                detected = []
                if any(w in t for w in ("sad", "hopeless", "down", "cry")):
                    detected.append({"emotion": "sadness", "score": 0.9})
                if any(w in t for w in ("angry", "mad", "furious")):
                    detected.append({"emotion": "anger", "score": 0.8})
                if any(w in t for w in ("fear", "scared", "afraid", "anxious")):
                    detected.append({"emotion": "fear", "score": 0.8})
                if not detected:
                    return [{"emotion": "neutral", "score": 1.0}]
                return detected
            
            result = _LazyPipelines.zero_shot(text, candidate_labels=emotion_labels, multi_label=True)
            detected = []
            for label, score in zip(result['labels'], result['scores']):
                if score >= threshold:
                    detected.append({"emotion": label, "score": float(score)})
            return detected if detected else [{"emotion": "neutral", "score": 1.0}]
        except Exception as e:
            logger.exception("Emotion detection failed: %s", e)
            return [{"emotion": "neutral", "score": 1.0}]

    def _keyword_boost(self, text: str) -> float:
        t = (text or "").lower()
        for kw in SUICIDE_KEYWORDS:
            if kw in t:
                return 1.0
        return 0.0

    def compute_risk_score(self, text: str, intent_info: Dict[str, Any], emotions: List[Dict[str, Any]], sentiment: Dict[str, Any]) -> float:
        score = 0.0
        
        # Intent Boost
        if intent_info.get("intent") == "crisis_signal":
            score += 0.6 * (intent_info.get("score", 1.0))

        # Sentiment Boost
        sentiment_label = str(sentiment.get("label", "")).lower()
        if "neg" in sentiment_label:
            score += 0.15 * float(sentiment.get("score", 0.0))

        # Emotion Boost
        for e in emotions:
            emo = e.get("emotion", "").lower()
            s = float(e.get("score", 0.0))
            if emo in ("hopelessness", "anxiety", "fear", "sadness"):
                score += 0.25 * s

        # Explicit Keyword Override
        if self._keyword_boost(text) >= 1.0:
            score = max(score, 0.9)

        return float(max(0.0, min(1.0, score)))

    def run_nlu_pipeline(self, user_message: str) -> Dict[str, Any]:
        cache_key = f"nlu:{user_message}"
        cached = _cache_get(cache_key)
        if cached:
            return cached

        intent_labels = [
            "request_appointment", "express_distress", "request_info",
            "general_chat", "crisis_signal", "confirm", "deny"
        ]
        emotion_labels = [
            "sadness", "joy", "anger", "fear", "disgust", "surprise",
            "neutral", "anxiety", "hopelessness", "calm"
        ]

        sentiment = self.detect_sentiment(user_message)
        intent = self.detect_intent(user_message, candidate_labels=intent_labels)
        emotions = self.detect_emotions(user_message, emotion_labels)
        
        risk_score = self.compute_risk_score(user_message, intent, emotions, sentiment)

        out = {
            "sentiment": sentiment,
            "intent": intent, 
            "emotions": emotions,
            "risk_score": risk_score
        }

        _cache_set(cache_key, out)
        return out

nlu_manager = NLUManager()'''




from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading
import logging
import time
import os

logger = logging.getLogger("aura.nlu")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------
# TRI-SENSOR CONFIGURATION
# ---------------------------------------------------------

# 1. General Sentiment (Emotion)
GENERAL_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# 2. Risk Detection (YOUR CUSTOM MODEL)
# This points to the folder where you put the 6 files.
RISK_MODEL_PATH = "app/data/my_custom_risk_model" 

# Fallback (Only used if the local folder is empty/broken)
RISK_MODEL_FALLBACK = "rafalposwiata/deproberta-large-depression"

# 3. Intent & Granular Emotions
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

# Keyword Backup (Fail-Safe)
SUICIDE_KEYWORDS = [
    "kill myself", "i will kill myself", "want to die", "end my life", 
    "suicide", "die by suicide", "cant go on", "shouldn't be here",
    "i have a plan", "not worth it", "better off without me"
]

class _LazyPipelines:
    _lock = threading.Lock()
    general_sentiment = None
    risk_classifier = None
    zero_shot = None
    _last_load_attempt = 0

    @classmethod
    def ensure_loaded(cls):
        now = time.time()
        # If already loaded, skip
        if cls.general_sentiment and cls.risk_classifier and cls.zero_shot:
            return

        with cls._lock:
            if cls.general_sentiment and cls.risk_classifier and cls.zero_shot:
                return
            if now - cls._last_load_attempt < 30:
                return
            cls._last_load_attempt = now
            try:
                from transformers import pipeline
                logger.info("NLU: Loading HF pipelines...")
                
                # 1. General Sentiment
                cls.general_sentiment = pipeline("text-classification", model=GENERAL_SENTIMENT_MODEL)
                
                # 2. YOUR CUSTOM RISK MODEL
                # We force it to look at your local folder
                if os.path.exists(RISK_MODEL_PATH):
                    logger.info(f"✅ Loading YOUR custom fine-tuned model from: {RISK_MODEL_PATH}")
                    # We specify the task to ensure it loads correctly
                    cls.risk_classifier = pipeline("text-classification", model=RISK_MODEL_PATH)
                else:
                    logger.warning(f"⚠️ Path '{RISK_MODEL_PATH}' not found. Using fallback.")
                    cls.risk_classifier = pipeline("text-classification", model=RISK_MODEL_FALLBACK)
                
                # 3. Intent
                cls.zero_shot = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
                
                logger.info("NLU: All pipelines loaded.")
            except Exception as e:
                logger.exception("Failed to load pipelines: %s", e)

# Simple Cache
_nlu_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()

def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    with _cache_lock: return _nlu_cache.get(key)

def _cache_set(key: str, value: Dict[str, Any]):
    with _cache_lock:
        if key in _nlu_cache: _nlu_cache.pop(key, None)
        _nlu_cache[key] = value
        if len(_nlu_cache) > 512: _nlu_cache.pop(next(iter(_nlu_cache)), None)

@dataclass
class NLUManager:
    
    def detect_general_sentiment(self, text: str) -> Dict[str, Any]:
        _LazyPipelines.ensure_loaded()
        try:
            if not _LazyPipelines.general_sentiment: return {"label": "NEUTRAL", "score": 0.5}
            result = _LazyPipelines.general_sentiment(text, truncation=True, max_length=512)[0]
            return {"label": result["label"], "score": float(result["score"])}
        except Exception:
            return {"label": "NEUTRAL", "score": 0.5}

    def detect_risk_level(self, text: str) -> Dict[str, Any]:
        _LazyPipelines.ensure_loaded()
        try:
            if not _LazyPipelines.risk_classifier: return {"label": "SAFE", "score": 0.0}
            
            result = _LazyPipelines.risk_classifier(text, truncation=True, max_length=512)[0]
            label = result.get("label") 
            score = float(result.get("score", 0.0))

            # LOGIC FOR YOUR CUSTOM MODEL
            # In your config.json, "1" is LABEL_1.
            # Based on your training data, LABEL_1 = Risk.
            if label == "LABEL_1" or label == "RISK" or label == "1": 
                return {"label": "RISK", "score": score}
            else:
                return {"label": "SAFE", "score": score}
        except Exception as e:
            logger.error(f"Risk model error: {e}")
            return {"label": "SAFE", "score": 0.0}

    def detect_intent(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:
        _LazyPipelines.ensure_loaded()
        try:
            if not _LazyPipelines.zero_shot:
                if "book" in text.lower(): return {"intent": "request_appointment", "score": 0.9}
                return {"intent": "general_chat", "score": 0.5}
            result = _LazyPipelines.zero_shot(text, candidate_labels=candidate_labels)
            return {"intent": result['labels'][0], "score": float(result['scores'][0])}
        except Exception:
            return {"intent": "unknown", "score": 0.0}

    def detect_emotions(self, text: str, emotion_labels: List[str], threshold: float = 0.45) -> List[Dict[str, Any]]:
        _LazyPipelines.ensure_loaded()
        try:
            if not _LazyPipelines.zero_shot: return [{"emotion": "neutral", "score": 1.0}]
            result = _LazyPipelines.zero_shot(text, candidate_labels=emotion_labels, multi_label=True)
            detected = []
            for label, score in zip(result['labels'], result['scores']):
                if score >= threshold:
                    detected.append({"emotion": label, "score": float(score)})
            return detected if detected else [{"emotion": "neutral", "score": 1.0}]
        except Exception:
            return [{"emotion": "neutral", "score": 1.0}]

    def _keyword_boost(self, text: str) -> float:
        t = text.lower()
        for kw in SUICIDE_KEYWORDS:
            if kw in t: return 1.0
        return 0.0

    def compute_risk_score(self, text: str, intent_info: Dict[str, Any], emotions: List[Dict[str, Any]], sentiment: Dict[str, Any], risk_analysis: Dict[str, Any]) -> float:
        score = 0.0
        
        # 1. YOUR CUSTOM MODEL (The Authority)
        if risk_analysis["label"] == "RISK":
            # Since this is your custom fine-tuned model, we trust it highly.
            # If it says Risk with >80% confidence, we trigger immediate alert (0.9).
            if risk_analysis["score"] > 0.8:
                score = 0.9 
            else:
                score = max(score, 0.7)

        # 2. KEYWORD FAILSAFE
        if self._keyword_boost(text) >= 1.0: return 1.0

        # 3. INTENT CONTEXT
        if intent_info.get("intent") == "crisis_signal":
            score = max(score, 0.75)

        # 4. GENERAL SENTIMENT
        if sentiment["label"] == "NEGATIVE":
            score += 0.05

        return float(max(0.0, min(1.0, score)))

    def run_nlu_pipeline(self, user_message: str) -> Dict[str, Any]:
        cache_key = f"nlu:{user_message}"
        cached = _cache_get(cache_key)
        if cached: return cached

        intent_labels = ["request_appointment", "express_distress", "general_chat", "crisis_signal"]
        emotion_labels = ["sadness", "joy", "anger", "fear", "disgust", "neutral", "anxiety", "hopelessness"]

        # Run All Sensors
        general_sentiment = self.detect_general_sentiment(user_message)
        risk_analysis = self.detect_risk_level(user_message)
        intent = self.detect_intent(user_message, candidate_labels=intent_labels)
        emotions = self.detect_emotions(user_message, emotion_labels)
        
        # Compute Combined Score
        risk_score = self.compute_risk_score(
            text=user_message,
            intent_info=intent,
            emotions=emotions,
            sentiment=general_sentiment,
            risk_analysis=risk_analysis
        )

        out = {
            "sentiment": general_sentiment,
            "risk_analysis": risk_analysis, 
            "intent": intent, 
            "emotions": emotions,
            "risk_score": risk_score
        }

        _cache_set(cache_key, out)
        return out

nlu_manager = NLUManager()