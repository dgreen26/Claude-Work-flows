# Meridian: Supply Chain Risk Assessment Workflow

## Objective
Produce a risk score and risk digest for a named supplier using real-time news (GDELT),
SEC filings (EDGAR), and the RAG knowledge base.

## Required Inputs
- Supplier name or ticker symbol
- Timespan for fresh news signals (default: last 6 hours)

## Tools Used (in order)
1. `meridian/ingest/gdelt.py` — fetch recent news articles
2. `meridian/ingest/edgar.py` — fetch SEC filings for the supplier
3. `meridian/agents/text_agent.py` — NER + zero-shot classification + summarization
4. `meridian/agents/orchestrator.py` — LangGraph graph: analyze → score → alert
5. `meridian/rag/query_engine.py` — RAG query for historical context
6. `meridian/api/main.py` — POST /assess/{supplier} wraps all of the above

## Expected Output
```json
{
  "supplier": "TSMC",
  "risk_score": 0.73,
  "dominant_category": "geopolitical_risk",
  "alert": true,
  "alert_message": "RISK ALERT: TSMC — Score 0.73 (Geopolitical Risk). Immediate review recommended.",
  "rag_summary": "Historical context: TSMC has disclosed cross-strait tension risks...",
  "document_count": 12
}
```

## Risk Score Interpretation
| Score | Level | Action |
|---|---|---|
| 0.0 – 0.35 | Low | Monitor normally |
| 0.35 – 0.65 | Medium | Flag for weekly review |
| 0.65 – 1.0 | High | Immediate procurement team review |

## Edge Cases
- **No GDELT articles found for supplier**: Fall back to most recent 20 articles
  from the risk query pool. Log this fallback.
- **EDGAR CIK not found**: Log a warning, continue with GDELT-only assessment.
- **RAG knowledge base empty**: rag_summary will be empty. This is expected on first
  run before any documents have been ingested. Run `/ingest/gdelt` first.
- **LLM API timeout**: The TextAgent summarization will fall back to truncated raw text.
  Risk score is still computed from the zero-shot classifier (local model, no API needed).

## Ingestion Schedule (recommended)
- GDELT: every 15 minutes via Celery beat
- EDGAR: daily at 06:00 UTC for all tracked companies
- Embedding: immediately after each ingestion batch (chained task)

## Cost Notes
- GDELT: free
- EDGAR: free
- Embedding (local): free
- LLM calls: ~$0.002 per supplier assessment (Haiku for classification/summarization)
- LLM RAG synthesis: ~$0.008 per query (Sonnet)
- Total per daily run of 50 suppliers: ~$0.50
