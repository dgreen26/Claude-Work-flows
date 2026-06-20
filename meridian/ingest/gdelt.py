"""
GDELT 2.0 ingestion — fetches news articles related to supply chain risk
and returns structured Article records ready for the embedding pipeline.

GDELT updates every 15 minutes. We use the DOC 2.0 API (free, no key needed).
Rate limit: ~1 req/sec. We respect this with a 1.1s sleep between pages.
"""

import time
import logging
from datetime import datetime
from dataclasses import dataclass, field

import httpx

from meridian.config import get_settings

log = logging.getLogger(__name__)

RISK_QUERY = (
    '("supply chain" OR "port congestion" OR "shipping delay" OR "factory closure" '
    'OR "sanctions" OR "trade war" OR "logistics disruption" OR "semiconductor shortage")'
)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass
class Article:
    url: str
    title: str
    domain: str
    language: str
    published_at: str
    source_country: str
    tone: float  # GDELT tone score: negative = negative sentiment
    themes: list[str] = field(default_factory=list)
    raw_text: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def fetch_recent(
    query: str = RISK_QUERY,
    max_records: int = 250,
    mode: str = "ArtList",
    timespan: str = "1d",
) -> list[Article]:
    """
    Pull recent articles from GDELT matching the query.

    Args:
        query: GDELT FullText query string.
        max_records: Max articles to return (GDELT caps at 250 per call).
        mode: GDELT mode — ArtList returns article metadata.
        timespan: Look-back window e.g. "1d", "6h", "15min".
    """
    settings = get_settings()
    params = {
        "query": query,
        "mode": mode,
        "maxrecords": min(max_records, 250),
        "timespan": timespan,
        "format": "json",
        "sort": "DateDesc",
    }

    try:
        resp = httpx.get(
            GDELT_DOC_API,
            params=params,
            timeout=30.0,
            headers={"User-Agent": "Meridian-RiskIntel/0.1"},
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        log.error("GDELT request failed: %s", e)
        return []

    data = resp.json()
    articles_raw = data.get("articles", [])

    articles = []
    for raw in articles_raw:
        try:
            articles.append(
                Article(
                    url=raw.get("url", ""),
                    title=raw.get("title", ""),
                    domain=raw.get("domain", ""),
                    language=raw.get("language", "English"),
                    published_at=raw.get("seendate", ""),
                    source_country=raw.get("sourcecountry", ""),
                    tone=float(raw.get("tone", 0.0)),
                    themes=raw.get("themes", "").split(",") if raw.get("themes") else [],
                )
            )
        except Exception as e:
            log.warning("Skipping malformed GDELT article: %s", e)
            continue

    log.info("GDELT: fetched %d articles", len(articles))
    return articles


def fetch_article_text(url: str, timeout: float = 10.0) -> str:
    """Fetch raw article body. Falls back to empty string on any error."""
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Meridian-RiskIntel/0.1"},
        )
        resp.raise_for_status()
        return resp.text[:8000]  # cap to avoid oversized chunks
    except Exception as e:
        log.debug("Could not fetch article body for %s: %s", url, e)
        return ""


def ingest_batch(timespan: str = "6h", fetch_bodies: bool = False) -> list[Article]:
    """
    High-level entry point: fetch articles and optionally enrich with body text.
    Body fetching adds latency (~1s per article) and is skipped by default.
    """
    articles = fetch_recent(timespan=timespan)
    if fetch_bodies:
        for i, article in enumerate(articles):
            article.raw_text = fetch_article_text(article.url)
            if i % 10 == 0:
                time.sleep(1.1)  # be polite to source domains
    return articles
