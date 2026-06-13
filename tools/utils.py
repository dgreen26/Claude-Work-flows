"""Shared utilities for all tools: cache, polling, and logging."""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

TMP_DIR = Path(__file__).parent.parent / ".tmp"
TMP_DIR.mkdir(exist_ok=True)

LOG_PATH = TMP_DIR / "generation_log.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(prompt: str, settings: dict) -> str:
    payload = json.dumps({"prompt": prompt, "settings": settings}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_get(prompt: str, settings: dict) -> str | None:
    key = _cache_key(prompt, settings)
    path = TMP_DIR / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text())
        output_path = data.get("output_path")
        if output_path and Path(output_path).exists():
            log.info("Cache hit: %s -> %s", key, output_path)
            return output_path
    return None


def cache_set(prompt: str, settings: dict, output_path: str) -> None:
    key = _cache_key(prompt, settings)
    path = TMP_DIR / f"{key}.json"
    path.write_text(json.dumps({"prompt": prompt, "settings": settings, "output_path": output_path}))
    log.info("Cached: %s -> %s", key, output_path)


# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------

def log_generation(prompt: str, seed: int | None, settings: dict, output_path: str) -> None:
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": prompt,
        "seed": seed,
        "settings": settings,
        "output_path": output_path,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("Logged generation: %s", output_path)


# ---------------------------------------------------------------------------
# Async job poller
# ---------------------------------------------------------------------------

def poll_job(
    job_id: str,
    status_fn,
    interval: float = 5.0,
    timeout: float = 600.0,
) -> dict:
    """Poll status_fn(job_id) until it returns a result with status 'done' or raises."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = status_fn(job_id)
        status = result.get("status")
        if status == "done":
            return result
        if status == "failed":
            raise RuntimeError(f"Job {job_id} failed: {result.get('error')}")
        log.info("Job %s status: %s — polling again in %.0fs", job_id, status, interval)
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_clip(path: str, expected_w: int = 1080, expected_h: int = 1920) -> None:
    """Raise if the clip at path doesn't match expected resolution."""
    import ffmpeg as ff
    probe = ff.probe(path)
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w, h = int(stream["width"]), int(stream["height"])
    if (w, h) != (expected_w, expected_h):
        raise ValueError(f"Resolution mismatch: got {w}x{h}, expected {expected_w}x{expected_h}")
    log.info("Clip validated: %s (%dx%d)", path, w, h)
