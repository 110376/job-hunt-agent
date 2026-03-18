from __future__ import annotations

from types import SimpleNamespace


class _DummyClassifier:
    def __init__(self, model_name: str | None, provider: str | None = None) -> None:
        self.model_name = model_name
        self.provider = provider


class _RateLimitedAgent:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, *_args, **_kwargs):
        self.calls += 1
        raise RuntimeError("Error code: 429 - rate limit")


class _Toolbox:
    def get_query_hints_impl(self):
        return {"hints": []}

    def search_jobs_impl(self, **_kwargs):
        return {"ok": True, "items": []}

    def collect_job_impl(self, **_kwargs):
        return {"ok": True}


class _CollectingToolbox:
    def __init__(self, state) -> None:
        self.state = state
        self.seed = 0

    def get_query_hints_impl(self):
        return {"hints": ["AI Engineer 校招"]}

    def search_jobs_impl(self, **_kwargs):
        self.seed += 1
        return {
            "ok": True,
            "items": [
                {
                    "job_url": f"https://example.com/job/{self.seed}",
                    "title": f"AI Intern {self.seed}",
                }
            ],
        }

    def collect_job_impl(self, **_kwargs):
        self.state.records.append(
            SimpleNamespace(
                title="AI Intern",
                source="boss",
            )
        )
        return {"ok": True, "status": "accepted"}


def test_runner_disables_agent_after_rate_limit(monkeypatch) -> None:
    from app import agent_runner as ar

    fake_agent = _RateLimitedAgent()

    monkeypatch.setattr(ar, "build_adapters", lambda _sites: {"boss": object()})
    monkeypatch.setattr(ar, "build_tools", lambda **_kwargs: SimpleNamespace(tools=[], toolbox=_Toolbox()))
    monkeypatch.setattr(ar, "build_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(ar, "create_agent", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(ar, "JobClassifier", _DummyClassifier)
    monkeypatch.setattr(ar, "export_records", lambda **_kwargs: ("x.json", "x.csv"))

    runner = ar.JobAgentRunner(
        role_name="AI Engineer",
        target_count=50,
        site_names=["boss"],
        output_dir=".",
        model_name="glm-4.7",
        provider="glm",
        max_rounds=2,
    )
    result = runner.run()

    assert fake_agent.calls == 1
    assert result.total_collected == 0


def test_runner_auto_expands_sites_when_growth_stalls(monkeypatch) -> None:
    from app import agent_runner as ar

    adapters_map = {
        "boss": object(),
        "liepin": object(),
        "lagou": object(),
    }

    def fake_build_adapters(site_names):
        return {name: adapters_map[name] for name in site_names if name in adapters_map}

    monkeypatch.setattr(ar, "build_adapters", fake_build_adapters)
    monkeypatch.setattr(ar, "supported_sites", lambda: ["boss", "liepin", "lagou"])
    monkeypatch.setattr(ar, "build_tools", lambda **kwargs: SimpleNamespace(tools=[], toolbox=_Toolbox()))
    monkeypatch.setattr(ar, "build_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(ar, "create_agent", lambda **_kwargs: _RateLimitedAgent())
    monkeypatch.setattr(ar, "JobClassifier", _DummyClassifier)
    monkeypatch.setattr(ar, "export_records", lambda **_kwargs: ("x.json", "x.csv"))

    runner = ar.JobAgentRunner(
        role_name="AI Engineer",
        target_count=50,
        site_names=["boss"],
        output_dir=".",
        model_name="glm-4.7",
        provider="glm",
        max_rounds=3,
    )
    runner.state.no_growth_rounds = 2
    changed = runner._maybe_expand_site_pool()

    assert changed is True
    assert "liepin" in runner.state.selected_sites


def test_runner_skips_agent_before_prefetch_threshold(monkeypatch) -> None:
    from app import agent_runner as ar

    fake_agent = _RateLimitedAgent()

    monkeypatch.setattr(ar, "build_adapters", lambda _sites: {"boss": object()})
    monkeypatch.setattr(ar, "build_tools", lambda **_kwargs: SimpleNamespace(tools=[], toolbox=_Toolbox()))
    monkeypatch.setattr(ar, "build_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(ar, "create_agent", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(ar, "JobClassifier", _DummyClassifier)
    monkeypatch.setattr(ar, "export_records", lambda **_kwargs: ("x.json", "x.csv"))
    monkeypatch.setenv("JOB_AGENT_AGENT_START_COLLECTED", "999")

    runner = ar.JobAgentRunner(
        role_name="AI Engineer",
        target_count=50,
        site_names=["boss"],
        output_dir=".",
        model_name="glm-4.7",
        provider="glm",
        max_rounds=2,
    )
    runner.run()

    assert fake_agent.calls == 0


def test_runner_enables_agent_after_prefetch_threshold(monkeypatch) -> None:
    from app import agent_runner as ar

    fake_agent = _RateLimitedAgent()

    monkeypatch.setattr(ar, "build_adapters", lambda _sites: {"boss": object()})
    monkeypatch.setattr(
        ar,
        "build_tools",
        lambda **kwargs: SimpleNamespace(tools=[], toolbox=_CollectingToolbox(kwargs["state"])),
    )
    monkeypatch.setattr(ar, "build_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(ar, "create_agent", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(ar, "JobClassifier", _DummyClassifier)
    monkeypatch.setattr(ar, "export_records", lambda **_kwargs: ("x.json", "x.csv"))
    monkeypatch.setenv("JOB_AGENT_AGENT_START_COLLECTED", "1")

    runner = ar.JobAgentRunner(
        role_name="AI Engineer",
        target_count=50,
        site_names=["boss"],
        output_dir=".",
        model_name="glm-4.7",
        provider="glm",
        max_rounds=2,
    )
    runner.run()

    assert fake_agent.calls == 1
