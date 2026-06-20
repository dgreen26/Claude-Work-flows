"""
Lightweight demo embedder using a hash-based stub when sentence-transformers
is not installed. Swaps to real BAAI/bge embeddings when the package is available.

This lets the full pipeline run locally without the ~1.3GB model download.
"""

import hashlib
import logging
import struct
from typing import Any

import psycopg2
import psycopg2.extras

from meridian.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

EMBED_DIM = settings.embedding_dim  # 1024


def _hash_embed(text: str) -> list[float]:
    """
    Deterministic pseudo-embedding for demo/testing without model download.
    Uses int-to-float mapping via SHA256 bytes to guarantee finite values.
    """
    seed = text.encode()
    vec = []
    for i in range(EMBED_DIM):
        h = hashlib.sha256(seed + i.to_bytes(4, "little")).digest()
        # Map first 2 bytes to float in [-1, 1]
        val = (int.from_bytes(h[:2], "little") / 32767.5) - 1.0
        vec.append(val)
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def _get_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(settings.embedding_model)
    except ImportError:
        log.info("sentence-transformers not available — using hash embedder (demo mode)")
        return None


_model = None


def embed(texts: list[str]) -> list[list[float]]:
    global _model
    if _model is None:
        _model = _get_model()

    if _model is not None:
        vecs = _model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]
    else:
        return [_hash_embed(t) for t in texts]


def _get_conn():
    return psycopg2.connect(settings.database_url)


def _ensure_table(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{{}}',
            embedding vector({EMBED_DIM}),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(doc_id, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
    """)


def chunk_text(text: str, target_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    if not text or not text.strip():
        return []
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        est = int(len(sentence.split()) * 1.3)
        if current_len + est > target_tokens and current:
            chunks.append(". ".join(current) + ".")
            while current and current_len > overlap_tokens:
                removed = current.pop(0)
                current_len -= int(len(removed.split()) * 1.3)
        current.append(sentence)
        current_len += est
    if current:
        chunks.append(". ".join(current) + ".")
    return [c for c in chunks if len(c) > 50]


import hashlib as _hashlib


def _doc_id(url_or_id: str) -> str:
    return _hashlib.sha256(url_or_id.encode()).hexdigest()[:16]


def upsert_documents(documents: list[dict], source: str) -> int:
    conn = _get_conn()
    cur = conn.cursor()
    _ensure_table(cur)
    conn.commit()

    total = 0
    for doc in documents:
        url = doc.get("url") or doc.get("document_url") or doc.get("accession_number") or doc.get("id", "")
        did = _doc_id(url)
        text = doc.get("raw_text") or doc.get("title") or ""
        if not text:
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = embed(chunks)
        metadata = {
            "source": source,
            "url": url,
            "title": doc.get("title", ""),
            "published_at": doc.get("published_at") or doc.get("filed_at", ""),
            "tone": doc.get("tone"),
            "company_name": doc.get("company_name", ""),
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute("""
                INSERT INTO document_chunks (doc_id, chunk_index, source, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id, chunk_index) DO NOTHING
            """, (did, i, source, chunk, psycopg2.extras.Json(metadata), emb))
            total += cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    log.info("Upserted %d vectors from source=%s", total, source)
    return total


def chunk_and_embed_articles(articles) -> int:
    return upsert_documents([a.__dict__ for a in articles], source="gdelt")


def chunk_and_embed_filings(filings) -> int:
    return upsert_documents([f.__dict__ for f in filings], source="edgar")
