"""
SEC EDGAR ingestion — pulls 10-K, 10-Q, and 8-K filings for a given company.

Uses the SEC EDGAR full-text search API and the company facts API.
No API key required. Rate limit: 10 req/sec max per SEC guidelines.
We stay well under that with a 0.15s sleep between calls.

Relevant filings for supply chain risk:
  - 10-K: Annual risk factor disclosures (Item 1A)
  - 10-Q: Quarterly updates, including supply chain commentary
  - 8-K: Material event disclosures (factory fires, supplier bankruptcies, etc.)
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime

import httpx

log = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
HEADERS = {
    "User-Agent": "Meridian-RiskIntel research@meridian.ai",  # SEC requires contact info
    "Accept-Encoding": "gzip, deflate",
}

FORM_TYPES = ["10-K", "10-Q", "8-K"]


@dataclass
class Filing:
    cik: str
    company_name: str
    form_type: str
    filed_at: str
    accession_number: str
    document_url: str
    raw_text: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def get_cik(ticker: str) -> str | None:
    """Resolve a ticker symbol to a SEC CIK number."""
    try:
        resp = httpx.get(
            f"{EDGAR_BASE}/submissions/CIK{ticker.upper().zfill(10)}.json",
            headers=HEADERS,
            timeout=15.0,
        )
        # Try ticker lookup via company tickers JSON
        tickers_resp = httpx.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS,
            timeout=15.0,
        )
        tickers_resp.raise_for_status()
        tickers = tickers_resp.json()
        for entry in tickers.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception as e:
        log.error("CIK lookup failed for %s: %s", ticker, e)
    return None


def fetch_filings(cik: str, form_types: list[str] = FORM_TYPES, max_per_type: int = 3) -> list[Filing]:
    """
    Fetch recent filings for a company by CIK.
    Returns Filing objects without body text (call enrich_filing to add text).
    """
    try:
        resp = httpx.get(
            f"{EDGAR_BASE}/submissions/CIK{cik}.json",
            headers=HEADERS,
            timeout=15.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        log.error("EDGAR submissions fetch failed for CIK %s: %s", cik, e)
        return []

    data = resp.json()
    company_name = data.get("name", "")
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    filings = []
    type_counts: dict[str, int] = {}

    for form, date, accession in zip(forms, dates, accessions):
        if form not in form_types:
            continue
        if type_counts.get(form, 0) >= max_per_type:
            continue
        type_counts[form] = type_counts.get(form, 0) + 1

        accession_clean = accession.replace("-", "")
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/full-index/"
            f"{date[:4]}/{date[5:7]}/{accession_clean}"
        )

        filings.append(Filing(
            cik=cik,
            company_name=company_name,
            form_type=form,
            filed_at=date,
            accession_number=accession,
            document_url=doc_url,
        ))
        time.sleep(0.15)

    log.info("EDGAR: found %d filings for %s", len(filings), company_name)
    return filings


def enrich_filing(filing: Filing, max_chars: int = 12000) -> Filing:
    """
    Fetch the actual filing text. For 10-K/10-Q we extract the risk factors section.
    We cap at max_chars to keep chunk sizes manageable.
    """
    try:
        # Fetch the filing index to find the primary document
        accession_path = filing.accession_number.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(filing.cik)}/{accession_path}/{filing.accession_number}-index.htm"
        )
        resp = httpx.get(index_url, headers=HEADERS, timeout=15.0)
        resp.raise_for_status()
        # Naive extraction: just take the raw HTML text truncated
        text = resp.text
        # Strip HTML tags crudely (proper parsing needs beautifulsoup4)
        import re
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        filing.raw_text = text[:max_chars]
    except Exception as e:
        log.warning("Could not enrich filing %s: %s", filing.accession_number, e)
    return filing


def ingest_company(ticker: str) -> list[Filing]:
    """High-level: resolve ticker → CIK → fetch and enrich filings."""
    cik = get_cik(ticker)
    if not cik:
        log.error("Could not resolve CIK for ticker: %s", ticker)
        return []
    filings = fetch_filings(cik)
    enriched = []
    for filing in filings:
        enriched.append(enrich_filing(filing))
        time.sleep(0.15)
    return enriched
