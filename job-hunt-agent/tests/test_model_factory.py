import os

import pytest


def test_resolve_provider_prefers_explicit() -> None:
    from app.model_factory import resolve_provider

    assert resolve_provider("dashscope") == "dashscope"
    assert resolve_provider("openai") == "openai"
    assert resolve_provider("zhipu") == "glm"
    assert resolve_provider("deepseek") == "deepseek"


def test_resolve_provider_defaults_to_glm_even_when_dashscope_key_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.model_factory import resolve_provider

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("JOB_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    assert resolve_provider(None) == "deepseek"


def test_resolve_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.model_factory import resolve_provider

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("JOB_AGENT_PROVIDER", "aliyun")
    assert resolve_provider(None) == "dashscope"


def test_resolve_provider_defaults_to_glm(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.model_factory import resolve_provider

    monkeypatch.delenv("JOB_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    assert resolve_provider(None) == "deepseek"


def test_resolve_provider_auto_deepseek_when_key_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.model_factory import resolve_provider

    monkeypatch.delenv("JOB_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    assert resolve_provider(None) == "deepseek"


def test_resolve_provider_auto_glm_when_key_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.model_factory import resolve_provider

    monkeypatch.delenv("JOB_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("GLM_API_KEY", "glm-key")
    assert resolve_provider(None) == "glm"


def test_default_model_for_provider() -> None:
    from app.model_factory import default_model_for_provider

    assert default_model_for_provider("dashscope") == "qwen-plus"
    assert default_model_for_provider("glm") == "openai:glm-4.7"
    assert default_model_for_provider("deepseek") == "openai:deepseek-chat"
    assert default_model_for_provider("openai") == "openai:gpt-4.1-mini"


def test_build_chat_model_dashscope(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import model_factory

    captured: dict[str, object] = {}

    def fake_init_chat_model(model: str, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return "fake-model"

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)

    built = model_factory.build_chat_model(model_name="qwen-plus", provider="dashscope", temperature=0)

    assert built == "fake-model"
    assert captured["model"] == "openai:qwen-plus"
    assert captured["api_key"] == "dash-key"
    assert captured["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_build_chat_model_openai_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import model_factory

    captured: dict[str, object] = {}

    def fake_init_chat_model(model: str, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return "fake-openai"

    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)

    built = model_factory.build_chat_model(
        model_name="openai:gpt-4.1-mini",
        provider="openai",
        temperature=0,
    )

    assert built == "fake-openai"
    assert captured["model"] == "openai:gpt-4.1-mini"
    assert "base_url" not in captured
    assert "api_key" not in captured


def test_build_chat_model_dashscope_model_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import model_factory

    captured: dict[str, object] = {}

    def fake_init_chat_model(model: str, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return "fake-dashscope-alias"

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)

    built = model_factory.build_chat_model(
        model_name="qwen3.5plus",
        provider="dashscope",
        temperature=0,
    )

    assert built == "fake-dashscope-alias"
    assert captured["model"] == "openai:qwen3.5-plus"


def test_build_chat_model_dashscope_model_name_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import model_factory

    captured: dict[str, object] = {}

    def fake_init_chat_model(model: str, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return "fake-dashscope-keep"

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)

    built = model_factory.build_chat_model(
        model_name="qwen3.5-plus",
        provider="dashscope",
        temperature=0,
    )

    assert built == "fake-dashscope-keep"
    assert captured["model"] == "openai:qwen3.5-plus"


def test_build_chat_model_dashscope_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import model_factory

    captured: dict[str, object] = {}

    def fake_init_chat_model(model: str, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return "fake-dashscope-key"

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("JOB_AGENT_API_KEY", "agent-key")
    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)

    built = model_factory.build_chat_model(
        model_name="qwen-plus",
        provider="dashscope",
        temperature=0,
    )

    assert built == "fake-dashscope-key"
    assert captured["api_key"] == "agent-key"


def test_build_chat_model_glm(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import model_factory

    captured: dict[str, object] = {}

    def fake_init_chat_model(model: str, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return "fake-glm"

    monkeypatch.setenv("GLM_API_KEY", "glm-key")
    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)

    built = model_factory.build_chat_model(model_name="glm-4.7", provider="glm", temperature=0)

    assert built == "fake-glm"
    assert captured["model"] == "openai:glm-4.7"
    assert captured["api_key"] == "glm-key"
    assert captured["base_url"] == "https://open.bigmodel.cn/api/paas/v4"


def test_build_chat_model_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import model_factory

    captured: dict[str, object] = {}

    def fake_init_chat_model(model: str, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return "fake-deepseek"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setattr(model_factory, "init_chat_model", fake_init_chat_model)

    built = model_factory.build_chat_model(
        model_name="deepseek-chat",
        provider="deepseek",
        temperature=0,
    )

    assert built == "fake-deepseek"
    assert captured["model"] == "openai:deepseek-chat"
    assert captured["api_key"] == "deepseek-key"
    assert captured["base_url"] == "https://api.deepseek.com/v1"
