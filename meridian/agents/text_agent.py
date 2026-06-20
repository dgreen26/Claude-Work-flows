"""
TextAgent: processes raw text documents from GDELT and EDGAR.

Responsibilities:
  - NER: extract supplier names, port names, commodity types, countries
  - Risk classification: tag each document with a risk category
  - Sentiment: derive risk severity from GDELT tone + LLM classification
  - Summarization: compress long documents into structured risk digests

Uses HuggingFace pipeline for NER and zero-shot classification (local, no API cost).
Uses LLM for summarization (better abstractive quality than local models at this scale).
"""

import logging
from dataclasses import dataclass, field

from transformers import pipeline

import litellm
from meridian.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

RISK_CATEGORIES = [
    "logistics_disruption",
    "geopolitical_risk",
    "financial_distress",
    "natural_disaster",
    "regulatory_change",
    "labor_dispute",
    "cybersecurity",
    "commodity_shortage",
]

SUMMARIZE_PROMPT = """Summarize the following supply chain risk document in 3 bullet points.
Each bullet should be one sentence. Focus on: what happened, who is affected, and the risk severity.
Do not use em dashes. Be direct and specific.

DOCUMENT:
{text}

SUMMARY (3 bullets):"""

_ner_pipeline = None
_classifier_pipeline = None


def _get_ner():
    global _ner_pipeline
    if _ner_pipeline is None:
        # dslim/bert-base-NER is fast and accurate for ORG/LOC/MISC entities
        _ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1,  # CPU; set to 0 for GPU
        )
    return _ner_pipeline


def _get_classifier():
    global _classifier_pipeline
    if _classifier_pipeline is None:
        _classifier_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
    return _classifier_pipeline


@dataclass
class TextAnalysis:
    entities: list[dict] = field(default_factory=list)
    risk_category: str = ""
    risk_score: float = 0.0  # 0.0 (no risk) to 1.0 (critical)
    summary: str = ""
    raw_text: str = ""


def extract_entities(text: str) -> list[dict]:
    """Run NER and return ORG, LOC, MISC entities with confidence scores."""
    if not text or len(text) < 20:
        return []
    ner = _get_ner()
    try:
        results = ner(text[:512])  # BERT max input length
        return [
            {
                "entity": r["entity_group"],
                "word": r["word"],
                "score": round(float(r["score"]), 3),
            }
            for r in results
            if r["score"] > 0.7 and r["entity_group"] in ("ORG", "LOC", "MISC")
        ]
    except Exception as e:
        log.warning("NER failed: %s", e)
        return []


def classify_risk(text: str) -> tuple[str, float]:
    """Zero-shot classify text into one of the RISK_CATEGORIES."""
    if not text or len(text) < 20:
        return "unknown", 0.0
    classifier = _get_classifier()
    try:
        result = classifier(text[:512], RISK_CATEGORIES, multi_label=False)
        category = result["labels"][0]
        score = float(result["scores"][0])
        return category, score
    except Exception as e:
        log.warning("Zero-shot classification failed: %s", e)
        return "unknown", 0.0


def summarize(text: str) -> str:
    """Abstractive summarization via LLM."""
    if not text or len(text) < 100:
        return text
    try:
        resp = litellm.completion(
            model=settings.litellm_model_classification,
            messages=[{
                "role": "user",
                "content": SUMMARIZE_PROMPT.format(text=text[:3000]),
            }],
            max_tokens=256,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("Summarization failed: %s", e)
        return text[:500]


def analyze(text: str, tone: float = 0.0) -> TextAnalysis:
    """
    Full text analysis pipeline for a single document.
    tone: GDELT tone score (negative = negative sentiment, used to boost risk_score).
    """
    entities = extract_entities(text)
    risk_category, classifier_score = classify_risk(text)
    summary = summarize(text)

    # Combine classifier score with GDELT tone
    # GDELT tone ranges roughly -10 to +10; normalize to 0-1 risk contribution
    tone_contribution = max(0.0, min(1.0, (-tone) / 10.0)) if tone != 0.0 else 0.0
    risk_score = round(0.7 * classifier_score + 0.3 * tone_contribution, 3)

    return TextAnalysis(
        entities=entities,
        risk_category=risk_category,
        risk_score=risk_score,
        summary=summary,
        raw_text=text[:200],
    )
