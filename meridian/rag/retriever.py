"""
Hybrid retrieval: dense (pgvector cosine) + sparse (BM25 via tsvector).

Why hybrid:
  - Dense retrieval handles semantic similarity ("port bottleneck" ≈ "shipping congestion")
  - Sparse retrieval handles exact keyword matches (ticker symbols, country names, CVE IDs)
  - RRF (Reciprocal Rank Fusion) combines both without needing a trained ranker

After retrieval, we re-rank with a cross-encoder to maximize precision before
passing to the LLM. This is the single highest-leverage improvement in RAG.

Query expansion (HyDE): we optionally generate a hypothetical document first,
embed that, then retrieve. This closes the query-document vocabulary gap for
short, under-specified queries.
"""

import logging
from dataclasses import dataclass

import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer, CrossEncoder

from meridian.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        # ms-marco cross-encoder is well-calibrated for passage reranking
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_index: int
    content: str
    source: str
    metadata: dict
    dense_rank: int = 0
    sparse_rank: int = 0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


def _embed_query(query: str) -> list[float]:
    model = _get_embedder()
    prefixed = BGE_QUERY_PREFIX + query
    vec = model.encode(prefixed, normalize_embeddings=True)
    return vec.tolist()


def dense_retrieve(query_vec: list[float], top_k: int, conn) -> list[RetrievedChunk]:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT doc_id, chunk_index, content, source, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_vec, query_vec, top_k))
    rows = cur.fetchall()
    cur.close()
    return [
        RetrievedChunk(
            doc_id=r["doc_id"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            source=r["source"],
            metadata=dict(r["metadata"]),
        )
        for r in rows
    ]


def sparse_retrieve(query: str, top_k: int, conn) -> list[RetrievedChunk]:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT doc_id, chunk_index, content, source, metadata,
               ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) AS score
        FROM document_chunks
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
    """, (query, query, top_k))
    rows = cur.fetchall()
    cur.close()
    return [
        RetrievedChunk(
            doc_id=r["doc_id"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            source=r["source"],
            metadata=dict(r["metadata"]),
        )
        for r in rows
    ]


def reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    RRF score = sum(1 / (k + rank)) across retrieval lists.
    k=60 is the standard default (Cormack et al. 2009).
    """
    scores: dict[tuple, float] = {}
    chunks: dict[tuple, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense):
        key = (chunk.doc_id, chunk.chunk_index)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunk.dense_rank = rank
        chunks[key] = chunk

    for rank, chunk in enumerate(sparse):
        key = (chunk.doc_id, chunk.chunk_index)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunk.sparse_rank = rank
        if key not in chunks:
            chunks[key] = chunk

    ranked = sorted(chunks.keys(), key=lambda k: scores[k], reverse=True)
    result = []
    for key in ranked:
        c = chunks[key]
        c.rrf_score = scores[key]
        result.append(c)
    return result


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []
    reranker = _get_reranker()
    pairs = [(query, c.content) for c in chunks]
    scores = reranker.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)
    ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    return ranked[:top_k]


def retrieve(
    query: str,
    top_k: int | None = None,
    rerank_k: int | None = None,
    use_hyde: bool = False,
    hyde_llm_fn=None,
) -> list[RetrievedChunk]:
    """
    Full hybrid retrieval pipeline.

    Args:
        query: Natural language question.
        top_k: Candidates from each retrieval method before RRF.
        rerank_k: Final chunks after cross-encoder reranking.
        use_hyde: If True, generate a hypothetical document to embed instead of raw query.
        hyde_llm_fn: Callable(query) -> str that generates the hypothetical document.
    """
    top_k = top_k or settings.retrieval_top_k
    rerank_k = rerank_k or settings.rerank_top_k

    embed_query = query
    if use_hyde and hyde_llm_fn:
        hypothetical = hyde_llm_fn(query)
        log.debug("HyDE hypothetical doc: %s", hypothetical[:100])
        embed_query = hypothetical

    query_vec = _embed_query(embed_query)

    conn = psycopg2.connect(settings.database_url)
    try:
        dense = dense_retrieve(query_vec, top_k, conn)
        sparse = sparse_retrieve(query, top_k, conn)
    finally:
        conn.close()

    fused = reciprocal_rank_fusion(dense, sparse)
    reranked = rerank(query, fused[:top_k * 2], rerank_k)

    log.info(
        "Retrieval: dense=%d sparse=%d fused=%d reranked=%d",
        len(dense), len(sparse), len(fused), len(reranked),
    )
    return reranked
