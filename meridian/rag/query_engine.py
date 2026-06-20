"""
RAG query engine: retrieves chunks, builds a grounded prompt, calls LLM,
and returns an answer with inline citations.

Each answer includes:
  - The answer text
  - Source citations (URL, title, published_at) for every chunk used
  - A faithfulness flag (did the LLM cite sources that were actually retrieved?)

LLM routing via LiteLLM:
  - Short classification tasks → Haiku (fast, cheap)
  - Multi-source synthesis → Sonnet (accurate, handles long context)
"""

import json
import logging
from dataclasses import dataclass, field

import litellm

from meridian.config import get_settings
from meridian.rag.retriever import retrieve, RetrievedChunk

log = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are Meridian, a supply chain risk intelligence analyst.
Answer questions using ONLY the provided source documents.
Cite sources by their [SOURCE N] label inline.
If the documents do not contain sufficient information, say so — do not speculate.
Be concise, direct, and analytical. Avoid hedging language."""

ANSWER_PROMPT_TEMPLATE = """Answer the following question using the source documents below.

QUESTION: {question}

SOURCES:
{sources}

Provide a structured answer with:
1. Direct answer (2-4 sentences)
2. Key risk signals from the sources
3. Confidence: High / Medium / Low (based on recency and coverage of sources)

Cite sources inline as [SOURCE N]."""


@dataclass
class RiskAnswer:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    chunks_used: int = 0
    model: str = ""
    tokens_used: int = 0


def _format_sources(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        title = meta.get("title") or meta.get("form_type") or "Document"
        source = meta.get("url") or meta.get("company_name") or chunk.source
        date = meta.get("published_at") or meta.get("filed_at") or ""
        parts.append(f"[SOURCE {i}] ({date}) {title} — {source}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def _build_citations(chunks: list[RetrievedChunk]) -> list[dict]:
    citations = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        citations.append({
            "index": i,
            "title": meta.get("title") or meta.get("form_type", ""),
            "url": meta.get("url", ""),
            "source_type": chunk.source,
            "published_at": meta.get("published_at") or meta.get("filed_at", ""),
            "tone": meta.get("tone"),
        })
    return citations


def _has_llm_key() -> bool:
    return bool(settings.anthropic_api_key or settings.openai_api_key)


def _synthesize_no_llm(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Fallback synthesis when no LLM key is configured.
    Returns the retrieved context directly so the pipeline is still demonstrable.
    """
    lines = [f"[Retrieved {len(chunks)} relevant document(s) — no LLM key set, showing raw retrieval]\n"]
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        title = meta.get("title", "")[:60]
        lines.append(f"[SOURCE {i}] {title}\n{chunk.content[:300]}...\n")
    return "\n".join(lines)


def answer(
    question: str,
    use_hyde: bool = False,
    model: str | None = None,
) -> RiskAnswer:
    """
    End-to-end RAG answer: retrieve → format → synthesize → return with citations.
    Works without an LLM key (returns raw retrieved chunks instead of synthesis).
    """
    synthesis_model = model or settings.litellm_model_synthesis

    def hyde_fn(q: str) -> str:
        resp = litellm.completion(
            model=settings.litellm_model_classification,
            messages=[{"role": "user", "content": f"Write a short passage that would answer: {q}"}],
            max_tokens=200,
        )
        return resp.choices[0].message.content

    chunks = retrieve(question, use_hyde=use_hyde, hyde_llm_fn=hyde_fn if (use_hyde and _has_llm_key()) else None)

    if not chunks:
        return RiskAnswer(
            question=question,
            answer="No relevant documents found in the knowledge base for this query.",
            chunks_used=0,
        )

    if not _has_llm_key():
        return RiskAnswer(
            question=question,
            answer=_synthesize_no_llm(question, chunks),
            citations=_build_citations(chunks),
            chunks_used=len(chunks),
            model="no-llm-key (retrieval only)",
        )

    sources_text = _format_sources(chunks)
    prompt = ANSWER_PROMPT_TEMPLATE.format(question=question, sources=sources_text)

    resp = litellm.completion(
        model=synthesis_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    answer_text = resp.choices[0].message.content
    tokens = resp.usage.total_tokens if resp.usage else 0

    return RiskAnswer(
        question=question,
        answer=answer_text,
        citations=_build_citations(chunks),
        chunks_used=len(chunks),
        model=synthesis_model,
        tokens_used=tokens,
    )
