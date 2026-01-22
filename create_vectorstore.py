#!/usr/bin/env python3
"""
create_vectorstore.py

Improved vectorstore builder for the project.

Features:
- Reads .txt files from data/rag_knowledge_base/
- Splits into sensible chunks
- Optionally computes embeddings locally (SentenceTransformer) and passes them to Chroma
- Supports batch add / upsert and recreating the collection
- Robust handling for different chromadb client APIs (get_collection, create_collection,
get_or_create_collection, PersistentClient vs Client)
- CLI flags: --recreate, --batch-size, --embed (compute embeddings locally)
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import uuid
import logging
import argparse
import math
import time
from typing import List, Tuple, Optional

try:
    import chromadb
except Exception as e:
    chromadb = None  # We'll handle later

# Optional local embedding model
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---- CONFIG (can also be overridden via CLI) ----
KB_DIR = Path("data/rag_knowledge_base")
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "mental_health_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunking config
MAX_CHUNK_WORDS = 250
MIN_CHUNK_WORDS = 30

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("create_vectorstore")


def chunk_text(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    for p in paragraphs:
        words = p.split()
        if len(words) <= MAX_CHUNK_WORDS:
            chunks.append(p)
        else:
            # split long paragraph into MAX_CHUNK_WORDS segments (word-boundary)
            current = []
            for w in words:
                current.append(w)
                if len(current) >= MAX_CHUNK_WORDS:
                    chunks.append(" ".join(current))
                    current = []
            if current:
                chunks.append(" ".join(current))

    # Remove too-small chunks
    chunks = [c for c in chunks if len(c.split()) >= MIN_CHUNK_WORDS]
    return chunks


def load_kb_documents(kb_dir: Path) -> Tuple[List[str], List[str], List[dict]]:
    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[dict] = []

    if not kb_dir.exists():
        logger.warning("KB directory does not exist: %s", kb_dir)
        return ids, docs, metadatas

    txt_files = sorted(kb_dir.glob("*.txt"))
    if not txt_files:
        logger.info("No .txt files found in: %s", kb_dir)
        return ids, docs, metadatas

    logger.info("Reading knowledge base documents from: %s", kb_dir)
    for file in txt_files:
        try:
            content = file.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read %s: %s", file.name, e)
            continue

        chunks = chunk_text(content)
        if not chunks:
            logger.info("Skipping empty/too-small file: %s", file.name)
            continue

        logger.info("  ➜ %s → %d chunks", file.name, len(chunks))
        for idx, chunk in enumerate(chunks):
            uid = str(uuid.uuid4())
            ids.append(uid)
            docs.append(chunk)
            metadatas.append({"source": file.name, "chunk_index": idx})
    logger.info("Total chunks prepared: %d", len(docs))
    return ids, docs, metadatas


def connect_chroma(chroma_path: str):
    if chromadb is None:
        raise RuntimeError("chromadb package is not installed. Install it with `pip install chromadb`.")

    # Try PersistentClient -> Client -> fallback
    client = None
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        logger.info("Connected to chroma PersistentClient at %s", chroma_path)
    except Exception:
        try:
            client = chromadb.Client()
            logger.info("Connected to chroma Client (in-memory or default).")
        except Exception as e:
            logger.exception("Failed to create chroma client: %s", e)
            raise
    return client


def get_or_create_collection(client, name: str):
    """
    Handle different chroma-python versions:
    - get_or_create_collection (preferred)
    - get_collection / create_collection
    - get_collection may raise if not found
    """
    collection = None
    try:
        # new API
        if hasattr(client, "get_or_create_collection"):
            collection = client.get_or_create_collection(name=name)
            logger.info("Using get_or_create_collection -> %s", name)
            return collection
    except Exception:
        logger.debug("get_or_create_collection not available / failed, falling back...")

    try:
        collection = client.get_collection(name=name)
        logger.info("Loaded existing collection -> %s", name)
        return collection
    except Exception:
        logger.debug("get_collection failed; will try create_collection")

    try:
        collection = client.create_collection(name=name)
        logger.info("Created new collection -> %s", name)
        return collection
    except Exception as e:
        logger.exception("Failed to create collection %s: %s", name, e)
        raise


def compute_embeddings(texts: List[str], model_name: str) -> List[List[float]]:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Install with `pip install sentence-transformers` to compute embeddings locally.")
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Computing embeddings for %d documents (batching)...", len(texts))
    # model.encode can accept list and return list of vectors
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Convert numpy arrays to lists (chromadb may require plain lists)
    return [v.tolist() for v in vectors]


def add_to_collection(collection, ids: List[str], docs: List[str], metadatas: List[dict], embeddings: Optional[List[List[float]]] = None, batch_size: int = 128):
    total = len(docs)
    logger.info("Adding %d items to collection (batch_size=%d)...", total, batch_size)
    for i in range(0, total, batch_size):
        j = min(i + batch_size, total)
        ids_batch = ids[i:j]
        docs_batch = docs[i:j]
        metas_batch = metadatas[i:j]
        embeds_batch = embeddings[i:j] if embeddings else None
        # Some chroma versions accept embeddings parameter; attempt to use it, else fallback to add without embeddings
        try:
            if embeds_batch is not None:
                collection.add(ids=ids_batch, documents=docs_batch, metadatas=metas_batch, embeddings=embeds_batch)
            else:
                collection.add(ids=ids_batch, documents=docs_batch, metadatas=metas_batch)
        except TypeError:
            # possibly different param names or older API - try without embeddings
            logger.debug("collection.add() rejected embeddings param; retrying without embeddings for this batch.")
            try:
                collection.add(ids=ids_batch, documents=docs_batch, metadatas=metas_batch)
            except Exception as e:
                logger.exception("Failed to add batch [%d:%d]: %s", i, j, e)
                raise
        except Exception as e:
            logger.exception("Failed to add batch [%d:%d]: %s", i, j, e)
            raise
        logger.info("  Added %d/%d", j, total)


def recreate_collection(client, name: str):
    # Try to delete existing collection if API supports it
    try:
        if hasattr(client, "delete_collection"):
            client.delete_collection(name)
            logger.info("Deleted existing collection: %s", name)
    except Exception:
        logger.debug("delete_collection not available or failed; continuing to create new collection if needed.")
    # create fresh
    return get_or_create_collection(client, name)


def parse_args():
    p = argparse.ArgumentParser(description="Create / update Chroma vectorstore from plain text files.")
    p.add_argument("--kb-dir", default=str(KB_DIR), help="Path to knowledge base text files (default: data/rag_knowledge_base)")
    p.add_argument("--chroma-path", default=CHROMA_DB_PATH, help="Chromadb persistent path (default: data/chroma_db)")
    p.add_argument("--collection", default=COLLECTION_NAME, help="Chromadb collection name")
    p.add_argument("--embed", action="store_true", help="Compute embeddings locally using sentence-transformers and pass to Chroma (requires sentence-transformers). If not set, let Chroma compute embeddings if supported.")
    p.add_argument("--embed-model", default=EMBED_MODEL, help="SentenceTransformer model name (default all-MiniLM-L6-v2)")
    p.add_argument("--recreate", action="store_true", help="Recreate collection (delete and create fresh)")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for adding/upserting documents")
    return p.parse_args()


def main():
    args = parse_args()
    kb_dir = Path(args.kb_dir)
    chroma_path = args.chroma_path
    collection_name = args.collection
    do_embed = args.embed
    embed_model = args.embed_model
    recreate = args.recreate
    batch_size = max(1, args.batch_size)

    ids, docs, metadatas = load_kb_documents(kb_dir)
    if not docs:
        logger.warning("No documents to index. Exiting.")
        return

    client = connect_chroma(chroma_path)
    # optionally recreate
    if recreate:
        collection = recreate_collection(client, collection_name)
    else:
        collection = get_or_create_collection(client, collection_name)

    embeddings = None
    if do_embed:
        # compute local embeddings
        embeddings = compute_embeddings(docs, embed_model)

    # Add to collection (in batches)
    start = time.time()
    try:
        add_to_collection(collection, ids, docs, metadatas, embeddings=embeddings, batch_size=batch_size)
    except Exception as e:
        logger.exception("Failed to add documents: %s", e)
        sys.exit(2)

    # report count if supported
    try:
        count = collection.count()
        logger.info("Vectorstore updated. Total vectors in collection: %s", count)
    except Exception:
        logger.info("Vectorstore update completed (collection.count() not supported).")

    logger.info("Finished in %.2fs", time.time() - start)


if __name__ == "__main__":
    main()
