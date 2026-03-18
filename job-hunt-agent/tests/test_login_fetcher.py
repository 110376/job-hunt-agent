from __future__ import annotations


def test_get_login_fetcher_returns_none_when_disabled(monkeypatch) -> None:
    from app.services import login_fetcher

    login_fetcher.reset_login_fetcher_for_test()
    monkeypatch.delenv("JOB_AGENT_USE_LOGIN_FETCHER", raising=False)

    assert login_fetcher.get_login_fetcher() is None
