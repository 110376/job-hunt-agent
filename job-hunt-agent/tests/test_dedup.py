from app.schemas import JobRecord
from app.services.dedup import make_fingerprint, normalize_url


def test_normalize_url_removes_query_and_www() -> None:
    url = "https://www.example.com/job/123?from=search"
    assert normalize_url(url) == "https://example.com/job/123"


def test_make_fingerprint_same_record_is_stable() -> None:
    record = JobRecord(
        title="AI Engineer",
        company="Foo",
        location="北京",
        salary="20k-30k",
        tech_tags=["llm"],
        requirements="要求熟悉 Python",
        source="boss",
        job_url="https://example.com/job/123",
    )
    fp1 = make_fingerprint(record)
    fp2 = make_fingerprint(record)
    assert fp1 == fp2
