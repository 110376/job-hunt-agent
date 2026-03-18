from __future__ import annotations

from app.schemas import JobJudgeResult, RuntimeState, SearchCandidate
from app.tools import JobToolbox


class _FakeAdapter:
    def search_jobs(self, query: str, page: int = 1, limit: int = 10):
        if "none" in query:
            return []
        return [
            SearchCandidate(
                title="AI Engineer Intern",
                job_url="https://example.com/job/1",
                source="boss",
                snippet="test",
            )
        ]

    def fetch_job_detail(self, _job_url: str):
        return {
            "title": "AI Engineer Intern",
            "text": "AI intern role, machine learning, llm, 校招 实习",
            "company": "ACME",
            "location": "Shanghai",
            "salary": "20k-30k",
        }


class _FakeClassifier:
    def classify(self, role_name: str, title: str, detail_text: str):
        return JobJudgeResult(
            is_ai_related=True,
            is_campus_or_intern=True,
            tech_tags=["llm", "ml"],
            requirements_summary="ML + LLM basics",
            reason="test",
            confidence=0.9,
        )


class _FlakyAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def search_jobs(self, query: str, page: int = 1, limit: int = 10):
        self.calls += 1
        raise RuntimeError("site timeout")


def test_toolbox_tracks_query_and_status_metrics() -> None:
    state = RuntimeState(
        role_name="AI Engineer",
        target_count=5,
        selected_sites=["boss"],
    )
    toolbox = JobToolbox(
        state=state,
        adapters={"boss": _FakeAdapter()},
        classifier=_FakeClassifier(),
    )

    search = toolbox.search_jobs_impl(site="boss", query="AI Engineer 校招", page=1, limit=5)
    assert search["ok"] is True
    assert search["count"] == 1
    assert state.query_attempt_counts["AI Engineer 校招"] == 1
    assert state.query_hit_counts["AI Engineer 校招"] == 1

    collected = toolbox.collect_job_impl(
        site="boss",
        job_url=search["items"][0]["job_url"],
        title=search["items"][0]["title"],
    )
    assert collected["ok"] is True
    assert collected["status"] == "accepted"
    assert state.status_counts["accepted"] == 1

    progress = toolbox.get_progress_impl()
    assert progress["collected"] == 1
    assert progress["status_counts"]["accepted"] == 1


def test_toolbox_pauses_site_after_repeated_search_failures() -> None:
    state = RuntimeState(
        role_name="AI Engineer",
        target_count=5,
        selected_sites=["boss"],
    )
    state.round_index = 1
    adapter = _FlakyAdapter()
    toolbox = JobToolbox(
        state=state,
        adapters={"boss": adapter},
        classifier=_FakeClassifier(),
    )

    first = toolbox.search_jobs_impl(site="boss", query="AI Engineer 校招", page=1, limit=5)
    second = toolbox.search_jobs_impl(site="boss", query="AI Engineer 校招", page=1, limit=5)
    third = toolbox.search_jobs_impl(site="boss", query="AI Engineer 校招", page=1, limit=5)

    assert first["ok"] is False
    assert second["ok"] is False
    assert third["ok"] is False
    assert third["error"] == "site paused"
    assert state.site_pause_until_round["boss"] == 3
    assert adapter.calls == 2
