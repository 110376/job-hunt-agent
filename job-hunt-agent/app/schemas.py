from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class SearchCandidate(BaseModel):
    title: str
    job_url: str
    source: str
    snippet: str = ""


class JobRecord(BaseModel):
    title: str
    company: str = "unknown"
    location: str = "unknown"
    salary: str = "unknown"
    tech_tags: list[str] = Field(default_factory=list)
    requirements: str
    source: str
    job_url: str


class JobJudgeResult(BaseModel):
    is_ai_related: bool
    is_campus_or_intern: bool
    tech_tags: list[str] = Field(default_factory=list)
    requirements_summary: str = ""
    reason: str = ""
    confidence: float = 0.0


@dataclass
class RuntimeState:
    role_name: str
    target_count: int
    selected_sites: list[str]
    records: list[JobRecord] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    seen_fingerprints: set[str] = field(default_factory=set)
    tried_queries: set[str] = field(default_factory=set)
    source_counts: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)
    query_attempt_counts: dict[str, int] = field(default_factory=dict)
    query_hit_counts: dict[str, int] = field(default_factory=dict)
    site_fail_streak: dict[str, int] = field(default_factory=dict)
    site_pause_until_round: dict[str, int] = field(default_factory=dict)
    round_index: int = 0
    no_growth_rounds: int = 0
    last_agent_message: str = ""
    debug_events: list[dict[str, Any]] = field(default_factory=list)
