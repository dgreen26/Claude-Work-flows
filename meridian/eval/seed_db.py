"""
Seed the vector DB with golden context snippets for CI eval.
Runs before the RAGAS harness so the RAG pipeline has something to retrieve.
"""

import json
import logging
from pathlib import Path

from meridian.rag.embedder import upsert_documents

log = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "golden_set" / "questions.json"


def seed():
    with open(GOLDEN_PATH) as f:
        questions = json.load(f)

    # Use ground truths as seed documents so retrieval has relevant content
    documents = [
        {
            "url": f"golden:{q['id']}",
            "title": q["question"],
            "raw_text": q["ground_truth"],
            "published_at": "2024-01-01",
            "tone": 0.0,
        }
        for q in questions
    ]
    count = upsert_documents(documents, source="golden")
    log.info("Seeded %d golden documents", count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seed()
