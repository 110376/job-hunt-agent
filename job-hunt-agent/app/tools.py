from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.tools import tool

from app.logging_utils import log
from app.schemas import JobRecord, RuntimeState
from app.services.classifier import JobClassifier
from app.services.dedup import make_fingerprint, normalize_url
from app.services.query_planner import AdaptiveQueryPlanner
from app.sites.base import SiteAdapter


@dataclass
class ToolBuildResult:
    tools: list
    toolbox: "JobToolbox"


class JobToolbox:
    def __init__(
        self,
        state: RuntimeState,
        adapters: dict[str, SiteAdapter],
        classifier: JobClassifier,
    ) -> None:
        self.state = state
        self.adapters = adapters
        self.classifier = classifier
        self.query_planner = AdaptiveQueryPlanner(max_per_round=16)
        self.site_fail_threshold = 2
        self.site_pause_rounds = 2

    def list_sites_impl(self) -> dict[str, Any]:
        sites = sorted(self.adapters.keys())
        payload = {"sites": sites, "count": len(sites)}
        log("tool-result", f"list_sites -> {payload}")
        return payload

    def get_progress_impl(self) -> dict[str, Any]:
        payload = {
            "target": self.state.target_count,
            "collected": len(self.state.records),
            "remaining": max(self.state.target_count - len(self.state.records), 0),
            "source_counts": dict(self.state.source_counts),
            "status_counts": dict(self.state.status_counts),
            "round_index": self.state.round_index,
            "no_growth_rounds": self.state.no_growth_rounds,
            "tried_queries": len(self.state.tried_queries),
            "query_attempt_counts": dict(self.state.query_attempt_counts),
            "query_hit_counts": dict(self.state.query_hit_counts),
            "site_fail_streak": dict(self.state.site_fail_streak),
            "site_pause_until_round": dict(self.state.site_pause_until_round),
        }
        log("tool-result", f"get_progress -> collected={payload['collected']} status={payload['status_counts']}")
        return payload

    def get_query_hints_impl(self) -> dict[str, Any]:
        hints = self.query_planner.build_hints(self.state)
        payload = {"hints": hints}
        log("tool-result", f"get_query_hints -> {len(hints)} hints")
        return payload

    def search_jobs_impl(self, site: str, query: str, page: int = 1, limit: int = 10) -> dict[str, Any]:
        site = site.strip().lower()
        query = " ".join(query.strip().split())
        if not query:
            self._bump_status("invalid_query")
            return {"ok": False, "error": "query cannot be empty"}
        adapter = self.adapters.get(site)
        if adapter is None:
            self._bump_status("unsupported_site")
            return {"ok": False, "error": f"unsupported site: {site}"}
        if self._is_site_paused(site):
            self._bump_status("site_paused")
            return {"ok": False, "error": "site paused"}

        page = max(1, page)
        limit = max(1, min(limit, 20))
        trace = f"{site}|{query}|p{page}"
        self.state.tried_queries.add(trace)
        self.state.query_attempt_counts[query] = self.state.query_attempt_counts.get(query, 0) + 1

        log("tool", f"search_jobs site={site} query={query} page={page} limit={limit}")
        try:
            candidates = adapter.search_jobs(query=query, page=page, limit=limit)
            self.state.site_fail_streak[site] = 0
            hit_count = len(candidates)
            self.state.query_hit_counts[query] = self.state.query_hit_counts.get(query, 0) + hit_count
            payload = {
                "ok": True,
                "count": hit_count,
                "items": [item.model_dump() for item in candidates],
            }
            if hit_count > 0:
                self._bump_status("search_hit")
            else:
                self._bump_status("search_empty")
            log("tool-result", f"search_jobs site={site} -> {hit_count} items")
            return payload
        except Exception as exc:
            streak = self.state.site_fail_streak.get(site, 0) + 1
            self.state.site_fail_streak[site] = streak
            if streak >= self.site_fail_threshold:
                self.state.site_pause_until_round[site] = self.state.round_index + self.site_pause_rounds
            self._bump_status("search_failed")
            log("warn", f"search_jobs site={site} failed: {exc}")
            return {"ok": False, "error": str(exc)}

    def collect_job_impl(self, site: str, job_url: str, title: str = "") -> dict[str, Any]:
        site = site.strip().lower()
        adapter = self.adapters.get(site)
        if adapter is None:
            self._bump_status("unsupported_site")
            return {"ok": False, "status": "unsupported_site", "error": f"unsupported site: {site}"}

        norm_url = normalize_url(job_url)
        if not norm_url:
            self._bump_status("invalid_url")
            return {"ok": False, "status": "invalid_url", "error": "invalid job_url"}
        if norm_url in self.state.seen_urls:
            self._bump_status("duplicate_url")
            return {"ok": True, "status": "duplicate_url", "job_url": norm_url}

        log("tool", f"collect_job site={site} url={job_url}")
        try:
            detail = adapter.fetch_job_detail(job_url)
        except Exception as exc:
            self._bump_status("fetch_failed")
            log("warn", f"fetch_job_detail failed: {exc}")
            return {"ok": False, "status": "fetch_failed", "error": str(exc)}

        merged_title = (title or detail.get("title") or "").strip()
        detail_text = (detail.get("text") or "").strip()
        if not detail_text:
            self._bump_status("empty_detail")
            return {"ok": False, "status": "empty_detail"}

        log("llm", f"classify_job title={merged_title[:60]}")
        judge = self.classifier.classify(
            role_name=self.state.role_name,
            title=merged_title,
            detail_text=detail_text,
        )

        if not judge.is_ai_related:
            self._bump_status("rejected_non_ai")
            return {
                "ok": True,
                "status": "rejected_non_ai",
                "reason": judge.reason,
                "confidence": judge.confidence,
            }
        if not judge.is_campus_or_intern:
            self._bump_status("rejected_non_campus")
            return {
                "ok": True,
                "status": "rejected_non_campus",
                "reason": judge.reason,
                "confidence": judge.confidence,
            }

        record = JobRecord(
            title=merged_title or "unknown role",
            company=(detail.get("company") or "unknown")[:80],
            location=(detail.get("location") or "unknown")[:40],
            salary=(detail.get("salary") or "unknown")[:40],
            tech_tags=judge.tech_tags,
            requirements=(judge.requirements_summary or detail_text[:200])[:220],
            source=site,
            job_url=norm_url,
        )

        fingerprint = make_fingerprint(record)
        if fingerprint in self.state.seen_fingerprints:
            self.state.seen_urls.add(norm_url)
            self._bump_status("duplicate_content")
            return {"ok": True, "status": "duplicate_content", "job_url": norm_url}

        self.state.records.append(record)
        self.state.seen_urls.add(norm_url)
        self.state.seen_fingerprints.add(fingerprint)
        self.state.source_counts[site] = self.state.source_counts.get(site, 0) + 1
        self._bump_status("accepted")

        log(
            "state",
            (
                f"accepted={len(self.state.records)}/{self.state.target_count} "
                f"source_counts={self.state.source_counts}"
            ),
        )
        return {
            "ok": True,
            "status": "accepted",
            "record": record.model_dump(),
            "confidence": judge.confidence,
        }

    def _bump_status(self, key: str) -> None:
        self.state.status_counts[key] = self.state.status_counts.get(key, 0) + 1

    def _is_site_paused(self, site: str) -> bool:
        pause_until = self.state.site_pause_until_round.get(site, 0)
        return self.state.round_index > 0 and self.state.round_index <= pause_until


def build_tools(
    state: RuntimeState,
    adapters: dict[str, SiteAdapter],
    classifier: JobClassifier,
) -> ToolBuildResult:
    toolbox = JobToolbox(state=state, adapters=adapters, classifier=classifier)

    @tool
    def list_sites() -> dict[str, Any]:
        """Return available job sites in this run."""

        return toolbox.list_sites_impl()

    @tool
    def get_query_hints() -> dict[str, Any]:
        """Return adaptive query hints based on progress and previous rounds."""

        return toolbox.get_query_hints_impl()

    @tool
    def get_progress() -> dict[str, Any]:
        """Return collection progress and runtime metrics."""

        return toolbox.get_progress_impl()

    @tool
    def search_jobs(site: str, query: str, page: int = 1, limit: int = 10) -> dict[str, Any]:
        """Search job listings by site and query."""

        return toolbox.search_jobs_impl(site=site, query=query, page=page, limit=limit)

    @tool
    def collect_job(site: str, job_url: str, title: str = "") -> dict[str, Any]:
        """Fetch and classify a job page; accepted records are stored automatically."""

        return toolbox.collect_job_impl(site=site, job_url=job_url, title=title)

    return ToolBuildResult(
        tools=[list_sites, get_query_hints, get_progress, search_jobs, collect_job],
        toolbox=toolbox,
    )
