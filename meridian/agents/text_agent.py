"""
TextAgent: NER + risk classification + summarization.
Falls back to lightweight heuristics when transformers not installed (demo mode).
"""

import logging
import re
from dataclasses import dataclass, field

import litellm
from meridian.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

RISK_CATEGORIES = [
    "logistics_disruption", "geopolitical_risk", "financial_distress",
    "natural_disaster", "regulatory_change", "labor_dispute",
    "cybersecurity", "commodity_shortage",
]

RISK_KEYWORDS = {
    "logistics_disruption": ["port", "shipping", "congestion", "delay", "logistics", "vessel", "container"],
    "geopolitical_risk": ["sanction", "tariff", "trade war", "taiwan", "china", "russia", "conflict", "geopolit"],
    "financial_distress": ["bankrupt", "default", "debt", "downgrade", "cash flow", "covenant", "loss"],
    "natural_disaster": ["earthquake", "flood", "hurricane", "typhoon", "disaster", "wildfire"],
    "regulatory_change": ["regulation", "compliance", "ban", "export control", "legislation", "policy"],
    "labor_dispute": ["strike", "union", "walkout", "labor", "workers", "protest"],
    "cybersecurity": ["ransomware", "cyber", "hack", "breach", "malware", "attack"],
    "commodity_shortage": ["shortage", "supply", "semiconductor", "chip", "rare earth", "lithium"],
}

SUMMARIZE_PROMPT = """Summarize the following supply chain risk document in 3 bullet points.
Each bullet should be one sentence. Focus on: what happened, who is affected, and the risk severity.
Do not use em dashes. Be direct and specific.

DOCUMENT:
{text}

SUMMARY (3 bullets):"""

ORG_PATTERNS = [
    r'\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Co|Group|Holdings|Technologies|Systems|Industries)\b',
    r'\b(?:TSMC|NVIDIA|Apple|Samsung|Toyota|Ford|GM|Boeing|Foxconn|Qualcomm)\b',
]
LOC_PATTERNS = [
    r'\b(?:China|Taiwan|Japan|South Korea|Vietnam|Thailand|India|Malaysia|Mexico|Germany)\b',
    r'\b(?:Los Angeles|Shanghai|Rotterdam|Singapore|Suez|Panama|Shenzhen)\b',
]


@dataclass
class TextAnalysis:
    entities: list[dict] = field(default_factory=list)
    risk_category: str = ""
    risk_score: float = 0.0
    summary: str = ""
    raw_text: str = ""


def extract_entities(text: str) -> list[dict]:
    entities = []
    for pattern in ORG_PATTERNS:
        for m in re.finditer(pattern, text):
            entities.append({"entity": "ORG", "word": m.group(), "score": 0.9})
    for pattern in LOC_PATTERNS:
        for m in re.finditer(pattern, text):
            entities.append({"entity": "LOC", "word": m.group(), "score": 0.9})
    # Deduplicate by word
    seen = set()
    unique = []
    for e in entities:
        if e["word"] not in seen:
            seen.add(e["word"])
            unique.append(e)
    return unique[:10]


def classify_risk(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    scores = {}
    for category, keywords in RISK_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = hits / len(keywords)
    best = max(scores, key=lambda k: scores[k])
    score = min(1.0, scores[best] * 3)  # scale up sparse keyword hits
    return best, score


def summarize(text: str) -> str:
    if not text or len(text) < 100:
        return text
    if not settings.anthropic_api_key and not settings.openai_api_key:
        # No LLM key — return first 200 chars as placeholder
        return text[:200] + "..."
    try:
        resp = litellm.completion(
            model=settings.litellm_model_classification,
            messages=[{"role": "user", "content": SUMMARIZE_PROMPT.format(text=text[:3000])}],
            max_tokens=256,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("Summarization failed: %s", e)
        return text[:300]


def analyze(text: str, tone: float = 0.0) -> TextAnalysis:
    entities = extract_entities(text)
    risk_category, classifier_score = classify_risk(text)
    summary = summarize(text)
    tone_contribution = max(0.0, min(1.0, (-tone) / 10.0)) if tone != 0.0 else 0.0
    risk_score = round(0.7 * classifier_score + 0.3 * tone_contribution, 3)
    return TextAnalysis(
        entities=entities,
        risk_category=risk_category,
        risk_score=risk_score,
        summary=summary,
        raw_text=text[:200],
    )
