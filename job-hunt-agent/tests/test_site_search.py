from __future__ import annotations

from app.schemas import SearchCandidate
from app.sites.base import SiteAdapter


def test_search_jobs_fallback_to_sogou_when_ddg_fails(monkeypatch) -> None:
    adapter = SiteAdapter(name="boss", domains=["zhipin.com"])

    def fake_ddg(*, query: str, page: int, limit: int):
        raise RuntimeError("ddg blocked")

    def fake_sogou(*, query: str, page: int, limit: int):
        return [
            SearchCandidate(
                title="AI Engineer",
                job_url="https://www.zhipin.com/job_detail/abc.html",
                source="boss",
                snippet="test",
            )
        ]

    monkeypatch.setattr(adapter, "_search_duckduckgo", fake_ddg)
    monkeypatch.setattr(adapter, "_search_sogou", fake_sogou)

    results = adapter.search_jobs(query="AI Engineer 校招", page=1, limit=5)

    assert len(results) == 1
    assert results[0].job_url == "https://www.zhipin.com/job_detail/abc.html"


def test_parse_sogou_data_url_result() -> None:
    adapter = SiteAdapter(name="boss", domains=["zhipin.com"])
    html_text = """
    <html><body>
      <div class="vrwrap">
        <h3><a href="/link?url=foo">AI Engineer</a></h3>
        <div class="r-sech" data-url="https://www.zhipin.com/job_detail/xyz.html"></div>
      </div>
    </body></html>
    """

    results = adapter._parse_sogou_results(html_text=html_text, limit=3)

    assert len(results) == 1
    assert results[0].job_url == "https://www.zhipin.com/job_detail/xyz.html"
    assert results[0].title == "AI Engineer"


def test_extract_location_from_detail() -> None:
    adapter = SiteAdapter(name="boss", domains=["zhipin.com"])
    text = "工作地点：杭州，负责模型训练与推理优化。"

    assert adapter._extract_location(text) == "杭州"


def test_extract_salary_with_k_range() -> None:
    adapter = SiteAdapter(name="boss", domains=["zhipin.com"])
    text = "薪资范围：25k-40k，14薪。"

    assert adapter._extract_salary(text).lower() == "25k-40k"


class _FakeLoginFetcher:
    def __init__(self, html_text: str) -> None:
        self.html_text = html_text
        self.calls = 0

    def fetch_html(self, url: str) -> str:
        self.calls += 1
        return self.html_text


def test_fetch_job_detail_fallback_to_login_fetcher_when_http_blocked(monkeypatch) -> None:
    fallback_html = """
    <html>
      <head><title>AI Engineer Intern - ACME</title></head>
      <body>
        公司：ACME
        工作地点：上海
        薪资范围：25k-35k
        LLM, NLP, Python
      </body>
    </html>
    """
    fetcher = _FakeLoginFetcher(fallback_html)
    adapter = SiteAdapter(name="boss", domains=["zhipin.com"], login_fetcher=fetcher)

    monkeypatch.setattr(
        adapter,
        "_fetch_detail_html_http",
        lambda _url: "<html><body>请完成验证后继续访问</body></html>",
    )
    monkeypatch.setattr(adapter, "_should_use_login_fetcher", lambda: True)

    detail = adapter.fetch_job_detail("https://www.zhipin.com/job_detail/blocked.html")

    assert fetcher.calls == 1
    assert "AI Engineer Intern" in detail["title"]
    assert detail["location"] == "上海"
