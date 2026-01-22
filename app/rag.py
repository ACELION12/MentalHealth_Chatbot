# app/rag.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, Any, List
import threading
import logging
import re
import time

logger = logging.getLogger("aura.rag")
logging.basicConfig(level=logging.INFO)

CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "mental_health_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Conservative patterns that indicate actionable self-harm instructions
_HARM_INSTRUCTION_PATTERNS = [
    r"how to (kill|harm|suicide|end my life)",
    r"(ways|methods) to (kill|harm|suicide|end my life)",
    r"step(s)? to (kill|harm|suicide|end my life)",
    r"how can i (kill|harm) myself",
    r"best way to (kill|harm) myself"
]
_HARM_RE = re.compile("|".join(_HARM_INSTRUCTION_PATTERNS), re.IGNORECASE)

# Simple in-memory LRU-like cache
_rag_cache: Dict[str, Dict[str, Any]] = {}
_RAG_CACHE_MAX = 256
_cache_lock = threading.Lock()


class RAGSystem:
    def __init__(
        self,
        embed_model_name: str = EMBED_MODEL_NAME,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
    ):
        self._init_lock = threading.Lock()
        self.collection = None
        self.client = None
        self.model = None
        self.embed_model_name = embed_model_name
        self.db_path = db_path
        self.collection_name = collection_name
        logger.info("RAGSystem initialized (lazy load).")

    def _init_once(self):
        """Lazy initialization of embedding model and chroma collection."""
        with self._init_lock:
            if self.collection is not None:
                return
            try:
                logger.info("RAG: loading embedding model '%s'...", self.embed_model_name)
                self.model = SentenceTransformer(self.embed_model_name)
            except Exception as e:
                logger.exception("RAG: failed to load embedding model: %s", e)
                self.model = None
            try:
                logger.info("RAG: connecting to ChromaDB (path=%s)...", self.db_path)
                # Use PersistentClient when available for on-disk persistence; fallback to in-memory client
                try:
                    self.client = chromadb.PersistentClient(path=self.db_path)
                except Exception:
                    # older/newer chroma may not provide PersistentClient
                    try:
                        self.client = chromadb.Client()
                    except Exception:
                        self.client = None
                if self.client is not None:
                    # try get_collection; if not available, create
                    try:
                        self.collection = self.client.get_collection(name=self.collection_name)
                    except Exception:
                        try:
                            self.collection = self.client.create_collection(name=self.collection_name)
                        except Exception as e:
                            logger.exception("RAG: failed to get/create collection: %s", e)
                            self.collection = None
                logger.info("RAG: collection ready (name=%s).", self.collection_name if self.collection else "NONE")
            except Exception as e:
                logger.exception("RAG: failed to initialize ChromaDB client/collection: %s", e)
                self.collection = None

    def _sanitize_text(self, text: str, max_chars: int = 800) -> str:
        """Minimal sanitization and truncation for snippets to avoid prompt bloat."""
        if not text:
            return ""
        txt = str(text).strip()
        txt = " ".join(txt.split())  # collapse whitespace
        if len(txt) > max_chars:
            cut = txt[:max_chars]
            last_period = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
            if last_period > int(max_chars * 0.5):
                cut = cut[: last_period + 1]
            txt = cut + " [...]"
        return txt

    def _check_harm_instructions(self, text: str) -> bool:
        """Return True if text appears to contain actionable self-harm instructions."""
        if not text:
            return False
        return bool(_HARM_RE.search(text))

    def _update_cache(self, key: str, value: Dict[str, Any]):
        with _cache_lock:
            if key in _rag_cache:
                _rag_cache.pop(key, None)
            _rag_cache[key] = value
            # Trim oldest when exceeding capacity
            if len(_rag_cache) > _RAG_CACHE_MAX:
                first_key = next(iter(_rag_cache))
                _rag_cache.pop(first_key, None)

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        with _cache_lock:
            return _rag_cache.get(key)

    def retrieve_info(self, query: str, top_k: int = 3, use_cache: bool = True) -> Dict[str, Any]:
        """
        Retrieve top_k relevant snippets for `query`.

        Returns:
        {
            "results": [{"text": str, "score": float, "source": str, "flagged": bool}, ...],
            "summary": str   # short, safe aggregation of non-flagged snippets
        }

        IMPORTANT: flagged==True means snippet appears to contain actionable self-harm instructions
        and MUST NOT be sent verbatim to the LLM.
        """
        self._init_once()
        cache_key = f"q:{query}:{top_k}"
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached:
                logger.debug("RAG: cache hit for query.")
                return cached

        if not self.collection:
            logger.warning("RAG: collection unavailable.")
            out = {"results": [], "summary": "Knowledge base unavailable."}
            self._update_cache(cache_key, out)
            return out

        try:
            # Query vector DB
            res = self.collection.query(query_texts=[query], n_results=top_k)
            # Different chroma versions return slightly different shapes; handle defensively
            documents = res.get("documents", [[]])[0]
            distances = res.get("distances", [[]])[0] if res.get("distances") else []
            results: List[Dict[str, Any]] = []
            for i, doc in enumerate(documents):
                doc_text = doc if isinstance(doc, str) else str(doc)
                score = float(distances[i]) if i < len(distances) else 0.0
                snippet = self._sanitize_text(doc_text, max_chars=800)
                flagged = self._check_harm_instructions(snippet)
                results.append({"text": snippet, "score": score, "source": f"doc_{i}", "flagged": bool(flagged)})

            # Aggregate safe snippets into a short summary (only non-flagged)
            safe_texts = [r["text"] for r in results if not r["flagged"]]
            if safe_texts:
                summary = " ".join(safe_texts[:2])
                if len(summary) > 900:
                    summary = summary[:900] + " [...]"
            else:
                summary = "Relevant resources found, but they require clinician review before being shared. Please consult a professional."

            out = {"results": results, "summary": summary}
            self._update_cache(cache_key, out)
            return out
        except Exception as e:
            logger.exception("RAG: query failed: %s", e)
            out = {"results": [], "summary": "An error occurred while searching the knowledge base."}
            self._update_cache(cache_key, out)
            return out

    def add_documents(
        self,
        docs: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add documents to the collection. docs should be list of strings.
        ids and metadatas are optional lists matching docs length.
        This function upserts into the collection and clears cache.
        """
        self._init_once()
        if not self.collection:
            raise RuntimeError("RAG collection not available")

        sanitized = [self._sanitize_text(d, max_chars=2000) for d in docs]
        try:
            # Try bulk add (many chroma versions accept raw documents and embed internally)
            self.collection.add(documents=sanitized, ids=ids or None, metadatas=metadatas or None)
            # clear cache because index changed
            with _cache_lock:
                _rag_cache.clear()
            logger.info("RAG: added %d documents", len(sanitized))
        except Exception as e:
            logger.exception("RAG: failed to add documents - %s", e)
            # Fallback: try adding one-by-one
            try:
                for idx, text in enumerate(sanitized):
                    _id = ids[idx] if ids and idx < len(ids) else None
                    _meta = metadatas[idx] if metadatas and idx < len(metadatas) else None
                    self.collection.add(documents=[text], ids=[_id] if _id else None, metadatas=[_meta] if _meta else None)
                with _cache_lock:
                    _rag_cache.clear()
                logger.info("RAG: fallback add succeeded.")
            except Exception as ex:
                logger.exception("RAG: fallback add also failed: %s", ex)
                raise
            # global instance
rag_system = RAGSystem()
