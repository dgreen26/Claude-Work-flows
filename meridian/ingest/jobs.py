"""
Celery task definitions for async ingestion.

Tasks are lightweight: they call the ingestion modules, then hand documents
off to the RAG pipeline for chunking and embedding. All side effects (DB writes,
embedding calls) happen inside the task so the API stays non-blocking.

Queue layout:
  - ingestion   : GDELT + EDGAR fetch tasks (I/O bound, high concurrency OK)
  - embedding   : Chunking + vector upsert (CPU/GPU bound, limit concurrency)
  - default     : Everything else

Start workers:
  celery -A meridian.ingest.jobs worker -Q ingestion,embedding,default --loglevel=info
"""

import logging
from celery import Celery
from meridian.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

app = Celery(
    "meridian",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_routes={
        "meridian.ingest.jobs.ingest_gdelt": {"queue": "ingestion"},
        "meridian.ingest.jobs.ingest_edgar_company": {"queue": "ingestion"},
        "meridian.ingest.jobs.embed_documents": {"queue": "embedding"},
    },
    task_acks_late=True,           # re-queue on worker crash
    worker_prefetch_multiplier=1,  # fair dispatch for long tasks
)


@app.task(bind=True, max_retries=3, default_retry_delay=30, name="meridian.ingest.jobs.ingest_gdelt")
def ingest_gdelt(self, timespan: str = "6h"):
    """Fetch GDELT articles, chunk them, and enqueue for embedding."""
    try:
        from meridian.ingest.gdelt import ingest_batch
        from meridian.rag.embedder import chunk_and_embed_articles

        articles = ingest_batch(timespan=timespan)
        if not articles:
            log.warning("GDELT returned 0 articles for timespan=%s", timespan)
            return {"ingested": 0}

        embed_documents.delay([a.__dict__ for a in articles], source="gdelt")
        return {"ingested": len(articles)}

    except Exception as exc:
        log.exception("ingest_gdelt failed")
        raise self.retry(exc=exc)


@app.task(bind=True, max_retries=3, default_retry_delay=60, name="meridian.ingest.jobs.ingest_edgar_company")
def ingest_edgar_company(self, ticker: str):
    """Fetch SEC filings for a ticker and enqueue for embedding."""
    try:
        from meridian.ingest.edgar import ingest_company
        from meridian.rag.embedder import chunk_and_embed_filings

        filings = ingest_company(ticker)
        if not filings:
            log.warning("No filings found for ticker=%s", ticker)
            return {"ticker": ticker, "ingested": 0}

        embed_documents.delay([f.__dict__ for f in filings], source="edgar")
        return {"ticker": ticker, "ingested": len(filings)}

    except Exception as exc:
        log.exception("ingest_edgar_company failed for %s", ticker)
        raise self.retry(exc=exc)


@app.task(bind=True, max_retries=2, name="meridian.ingest.jobs.embed_documents")
def embed_documents(self, documents: list[dict], source: str):
    """Chunk documents and upsert vectors into pgvector."""
    try:
        from meridian.rag.embedder import upsert_documents
        count = upsert_documents(documents, source=source)
        return {"source": source, "vectors_upserted": count}
    except Exception as exc:
        log.exception("embed_documents failed for source=%s", source)
        raise self.retry(exc=exc)
