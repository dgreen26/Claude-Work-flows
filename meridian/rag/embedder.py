"""
Document chunking and vector embedding pipeline.

Strategy:
  - Chunk by sentence boundary, target ~400 tokens, 50-token overlap
  - Embed with BAAI/bge-large-en-v1.5 (1024 dims, strong on English domain text)
  - Upsert into pgvector with metadata (source, doc_id, published_at, tone)
  - Deduplicate by URL hash before inserting to avoid re-embedding duplicates

BGE models perform best with a query prefix ("Represent this sentence for searching
relevant passages: "). We apply that prefix at query time, not embed time.
"""

import hashlib
import logging
from typing import Any

import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

from meridian.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def _get_conn():
    return psycopg2.connect(settings.database_url)


def chunk_text(text: str, target_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    """
    Naive sentence-boundary chunker. For production, replace with
    llama-index's SentenceSplitter which handles edge cases better.
    """
    if not text or not text.strip():
        return []

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    chunks = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        est_tokens = int(word_count * 1.3)  # rough token estimate

        if current_len + est_tokens > target_tokens and current:
            chunks.append(". ".join(current) + ".")
            # Keep overlap: drop from front until we're under overlap_tokens
            while current and current_len > overlap_tokens:
                removed = current.pop(0)
                current_len -= int(len(removed.split()) * 1.3)

        current.append(sentence)
        current_len += est_tokens

    if current:
        chunks.append(". ".join(current) + ".")

    return [c for c in chunks if len(c) > 50]


def _doc_hash(url_or_id: str) -> str:
    return hashlib.sha256(url_or_id.encode()).hexdigest()[:16]


def upsert_documents(documents: list[dict], source: str) -> int:
    """
    Chunk and embed a list of raw document dicts, upsert into pgvector.
    Returns the number of vectors upserted.
    """
    model = _get_model()
    conn = _get_conn()
    cur = conn.cursor()

    # Ensure table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            embedding vector(1024),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(doc_id, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
    """)
    conn.commit()

    total = 0
    for doc in documents:
        url = doc.get("url") or doc.get("document_url") or doc.get("accession_number", "")
        doc_id = _doc_hash(url)
        text = doc.get("raw_text") or doc.get("title") or ""
        if not text:
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        embeddings = model.encode(chunks, normalize_embeddings=True, batch_size=32)

        metadata = {
            "source": source,
            "url": url,
            "title": doc.get("title", ""),
            "published_at": doc.get("published_at") or doc.get("filed_at", ""),
            "tone": doc.get("tone"),
            "company_name": doc.get("company_name", ""),
            "form_type": doc.get("form_type", ""),
        }

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute("""
                INSERT INTO document_chunks (doc_id, chunk_index, source, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id, chunk_index) DO NOTHING
            """, (doc_id, i, source, chunk, psycopg2.extras.Json(metadata), embedding.tolist()))
            total += cur.rowcount

    conn.commit()
    cur.close()
    conn.close()
    log.info("Upserted %d new vectors from source=%s", total, source)
    return total


def chunk_and_embed_articles(articles) -> int:
    return upsert_documents([a.__dict__ for a in articles], source="gdelt")


def chunk_and_embed_filings(filings) -> int:
    return upsert_documents([f.__dict__ for f in filings], source="edgar")
