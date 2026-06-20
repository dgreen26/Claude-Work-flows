"""
Meridian FastAPI application.

Endpoints:
  POST /ingest/gdelt          Trigger GDELT ingestion job (async via Celery)
  POST /ingest/company        Trigger SEC EDGAR ingestion for a ticker (async)
  POST /assess/{supplier}     Run full risk assessment for a supplier
  POST /query                 RAG query against the knowledge base
  GET  /health                Liveness check

All long-running work is dispatched to Celery. API responses are always fast.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from meridian.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log.info("Meridian API starting")
    yield
    log.info("Meridian API shutting down")


app = FastAPI(
    title="Meridian Risk Intelligence API",
    version="0.1.0",
    description="Multi-modal supply chain risk intelligence powered by RAG + multi-agent AI.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response models ---

class GDELTIngestRequest(BaseModel):
    timespan: str = Field(default="6h", description="Look-back window: '15min', '1h', '6h', '1d'")
    fetch_bodies: bool = Field(default=False, description="Fetch full article text (slower)")


class CompanyIngestRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol e.g. AAPL, TSMC")


class AssessRequest(BaseModel):
    supplier: str = Field(..., description="Supplier name or ticker")
    timespan: str = Field(default="1d", description="GDELT look-back window for fresh signals")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5)
    use_hyde: bool = Field(default=False, description="Use HyDE query expansion")


class JobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str


class AssessResponse(BaseModel):
    supplier: str
    risk_score: float
    dominant_category: str
    alert: bool
    alert_message: str
    rag_summary: str
    document_count: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict]
    chunks_used: int
    model: str


# --- Routes ---

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest/gdelt", response_model=JobResponse)
async def trigger_gdelt_ingest(req: GDELTIngestRequest):
    """Dispatch a GDELT ingestion job to the Celery queue."""
    try:
        from meridian.ingest.jobs import ingest_gdelt
        task = ingest_gdelt.delay(timespan=req.timespan)
        return JobResponse(
            job_id=task.id,
            message=f"GDELT ingestion queued for timespan={req.timespan}",
        )
    except Exception as e:
        log.exception("Failed to queue GDELT job")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/company", response_model=JobResponse)
async def trigger_company_ingest(req: CompanyIngestRequest):
    """Dispatch SEC EDGAR ingestion for a company ticker."""
    try:
        from meridian.ingest.jobs import ingest_edgar_company
        task = ingest_edgar_company.delay(ticker=req.ticker.upper())
        return JobResponse(
            job_id=task.id,
            message=f"EDGAR ingestion queued for {req.ticker.upper()}",
        )
    except Exception as e:
        log.exception("Failed to queue EDGAR job")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assess/{supplier}", response_model=AssessResponse)
async def assess_supplier(supplier: str, req: AssessRequest):
    """
    Synchronous risk assessment for a supplier.
    Fetches fresh GDELT signals, runs the full agent graph, and returns a risk score.
    For production, move this to a Celery task and poll for results.
    """
    from meridian.ingest.gdelt import ingest_batch
    from meridian.agents.orchestrator import assess_supplier as run_assessment

    articles = ingest_batch(timespan=req.timespan)
    # Filter to articles mentioning the supplier (naive keyword match)
    supplier_lower = supplier.lower()
    relevant = [
        a.__dict__ for a in articles
        if supplier_lower in (a.title or "").lower()
        or supplier_lower in (a.raw_text or "").lower()
    ]

    if not relevant:
        # Fall back to all articles if no supplier-specific ones found
        relevant = [a.__dict__ for a in articles[:20]]

    state = run_assessment(supplier=supplier, documents=relevant)

    return AssessResponse(
        supplier=supplier,
        risk_score=state["risk_score"],
        dominant_category=state["dominant_category"],
        alert=state["alert"],
        alert_message=state["alert_message"],
        rag_summary=state["rag_summary"],
        document_count=len(relevant),
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """RAG query against the Meridian knowledge base."""
    from meridian.rag.query_engine import answer

    try:
        result = answer(req.question, use_hyde=req.use_hyde)
        return QueryResponse(
            question=req.question,
            answer=result.answer,
            citations=result.citations,
            chunks_used=result.chunks_used,
            model=result.model,
        )
    except Exception as e:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
