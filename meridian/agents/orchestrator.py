"""
OrchestratorAgent: LangGraph state machine that coordinates TextAgent,
computes per-supplier risk scores, and generates alerts.

Graph topology:
  START → ingest_check → [text_analysis] → score_synthesis → alert_check → END

State is a typed dict; each node is a pure function of state → state.
This makes the graph testable, serializable, and easy to visualize.

LangGraph over plain chains because:
  - Branching logic (alert vs no-alert) is a first-class edge, not a hack
  - State is explicit and inspectable at every node
  - Easy to add the VisionAgent and AudioAgent as parallel branches later
"""

import logging
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END

from meridian.agents.text_agent import analyze, TextAnalysis
from meridian.rag.query_engine import answer as rag_answer
from meridian.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

ALERT_THRESHOLD = 0.65  # risk_score above this triggers an alert


class RiskState(TypedDict):
    supplier: str
    documents: list[dict]       # raw document dicts from ingest
    analyses: list[dict]        # TextAnalysis results serialized
    risk_score: float            # composite 0-1
    dominant_category: str
    alert: bool
    alert_message: str
    rag_summary: str             # RAG answer for context


def node_text_analysis(state: RiskState) -> RiskState:
    """Run TextAgent on each ingested document."""
    analyses = []
    for doc in state["documents"]:
        text = doc.get("raw_text") or doc.get("title") or ""
        tone = doc.get("tone", 0.0) or 0.0
        result = analyze(text, tone=float(tone))
        analyses.append({
            "risk_category": result.risk_category,
            "risk_score": result.risk_score,
            "summary": result.summary,
            "entities": result.entities,
        })
    return {**state, "analyses": analyses}


def node_score_synthesis(state: RiskState) -> RiskState:
    """
    Aggregate individual document scores into a supplier-level risk score.
    Uses a weighted mean: more recent documents weighted higher.
    Dominant category is the most frequent category among high-risk docs.
    """
    analyses = state["analyses"]
    if not analyses:
        return {**state, "risk_score": 0.0, "dominant_category": "none"}

    scores = [a["risk_score"] for a in analyses]
    # Weight: higher scores matter more (max-influenced average)
    if scores:
        composite = 0.6 * max(scores) + 0.4 * (sum(scores) / len(scores))
    else:
        composite = 0.0

    # Dominant category from highest-risk documents
    high_risk = [a for a in analyses if a["risk_score"] > 0.5]
    if high_risk:
        from collections import Counter
        category_counts = Counter(a["risk_category"] for a in high_risk)
        dominant = category_counts.most_common(1)[0][0]
    else:
        dominant = analyses[0]["risk_category"] if analyses else "none"

    return {**state, "risk_score": round(composite, 3), "dominant_category": dominant}


def node_rag_context(state: RiskState) -> RiskState:
    """Enrich the risk assessment with a RAG query for historical context."""
    supplier = state["supplier"]
    query = f"What are the key supply chain risks associated with {supplier}?"
    try:
        result = rag_answer(query)
        summary = result.answer
    except Exception as e:
        log.warning("RAG context failed for %s: %s", supplier, e)
        summary = ""
    return {**state, "rag_summary": summary}


def node_alert_check(state: RiskState) -> RiskState:
    """Generate an alert if risk_score exceeds threshold."""
    if state["risk_score"] >= ALERT_THRESHOLD:
        msg = (
            f"RISK ALERT: {state['supplier']} — "
            f"Score {state['risk_score']:.2f} "
            f"({state['dominant_category'].replace('_', ' ').title()}). "
            f"Immediate review recommended."
        )
        return {**state, "alert": True, "alert_message": msg}
    return {**state, "alert": False, "alert_message": ""}


def _should_alert(state: RiskState) -> str:
    return "alert" if state["risk_score"] >= ALERT_THRESHOLD else "no_alert"


def build_graph() -> StateGraph:
    graph = StateGraph(RiskState)

    graph.add_node("text_analysis", node_text_analysis)
    graph.add_node("score_synthesis", node_score_synthesis)
    graph.add_node("rag_context", node_rag_context)
    graph.add_node("alert_check", node_alert_check)

    graph.set_entry_point("text_analysis")
    graph.add_edge("text_analysis", "score_synthesis")
    graph.add_edge("score_synthesis", "rag_context")
    graph.add_edge("rag_context", "alert_check")
    graph.add_edge("alert_check", END)

    return graph.compile()


# Module-level compiled graph (lazy init to avoid import-time model loading)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def assess_supplier(supplier: str, documents: list[dict]) -> RiskState:
    """
    Entry point: run the full risk assessment graph for a supplier.

    Args:
        supplier: Supplier name or ticker symbol.
        documents: List of raw document dicts (from GDELT/EDGAR ingest).

    Returns:
        Final RiskState with risk_score, dominant_category, alert, rag_summary.
    """
    initial_state: RiskState = {
        "supplier": supplier,
        "documents": documents,
        "analyses": [],
        "risk_score": 0.0,
        "dominant_category": "",
        "alert": False,
        "alert_message": "",
        "rag_summary": "",
    }
    graph = get_graph()
    final_state = graph.invoke(initial_state)
    log.info(
        "Supplier=%s risk_score=%.3f category=%s alert=%s",
        supplier, final_state["risk_score"],
        final_state["dominant_category"], final_state["alert"],
    )
    return final_state
