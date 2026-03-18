from __future__ import annotations

import hashlib
from urllib.parse import urlparse

from app.schemas import JobRecord


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    netloc = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{netloc}{path}"


def make_fingerprint(record: JobRecord) -> str:
    core = "|".join(
        [
            normalize_text(record.title),
            normalize_text(record.company),
            normalize_text(record.location),
            normalize_url(record.job_url),
        ]
    )
    return hashlib.sha1(core.encode("utf-8")).hexdigest()
