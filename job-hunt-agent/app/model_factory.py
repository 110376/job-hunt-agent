from __future__ import annotations

import os
from dataclasses import dataclass

from langchain.chat_models import init_chat_model


ALIYUN_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
GLM_COMPAT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEEPSEEK_COMPAT_BASE_URL = "https://api.deepseek.com/v1"

_DASHSCOPE_MODEL_ALIAS = {
    "qwen3.5plus": "qwen3.5-plus",
    "qwen-3.5-plus": "qwen3.5-plus",
    "qwen35plus": "qwen3.5-plus",
}

_GLM_MODEL_ALIAS = {
    "glm47": "glm-4.7",
    "glm-47": "glm-4.7",
    "glm4.7": "glm-4.7",
}

_DEEPSEEK_MODEL_ALIAS = {
    "deepseekchat": "deepseek-chat",
    "deepseek-chat": "deepseek-chat",
    "deepseekreasoner": "deepseek-reasoner",
    "deepseek-reasoner": "deepseek-reasoner",
}


@dataclass
class ModelSettings:
    provider: str
    model_name: str


def normalize_provider(provider: str | None) -> str:
    if not provider:
        return ""
    value = provider.strip().lower()
    mapping = {
        "openai": "openai",
        "dashscope": "dashscope",
        "glm": "glm",
        "deepseek": "deepseek",
        "deep-seek": "deepseek",
        "zhipu": "glm",
        "bigmodel": "glm",
        "ali": "dashscope",
        "aliyun": "dashscope",
        "bailian": "dashscope",
        "alibaba": "dashscope",
    }
    return mapping.get(value, value)


def resolve_provider(explicit_provider: str | None) -> str:
    normalized_explicit = normalize_provider(explicit_provider)
    if normalized_explicit:
        return normalized_explicit

    env_provider = normalize_provider(os.getenv("JOB_AGENT_PROVIDER"))
    if env_provider:
        return env_provider

    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"

    if os.getenv("GLM_API_KEY"):
        return "glm"

    return "deepseek"


def default_model_for_provider(provider: str) -> str:
    provider = normalize_provider(provider)
    if provider == "deepseek":
        return "openai:deepseek-chat"
    if provider == "glm":
        return "openai:glm-4.7"
    if provider == "dashscope":
        return "qwen-plus"
    return "openai:gpt-4.1-mini"


def resolve_model_settings(
    explicit_model_name: str | None,
    explicit_provider: str | None = None,
) -> ModelSettings:
    provider = resolve_provider(explicit_provider)
    model_name = (explicit_model_name or "").strip() or default_model_for_provider(provider)
    return ModelSettings(provider=provider, model_name=model_name)


def _to_dashscope_openai_model(model_name: str) -> str:
    trimmed = model_name.strip()
    if not trimmed:
        return "openai:qwen-plus"

    normalized = trimmed.lower().replace("_", "-").replace(" ", "")
    if normalized in _DASHSCOPE_MODEL_ALIAS:
        return f"openai:{_DASHSCOPE_MODEL_ALIAS[normalized]}"

    if ":" in trimmed:
        prefix, rest = trimmed.split(":", 1)
        norm_prefix = normalize_provider(prefix)
        normalized_rest = rest.strip().lower().replace("_", "-").replace(" ", "")
        if normalized_rest in _DASHSCOPE_MODEL_ALIAS:
            rest = _DASHSCOPE_MODEL_ALIAS[normalized_rest]
        if norm_prefix == "dashscope":
            return f"openai:{rest}"
        if prefix == "openai":
            return trimmed
        return f"openai:{rest}"

    return f"openai:{trimmed}"


def _to_glm_openai_model(model_name: str) -> str:
    trimmed = model_name.strip()
    if not trimmed:
        return "openai:glm-4.7"

    normalized = trimmed.lower().replace("_", "-").replace(" ", "")
    if normalized in _GLM_MODEL_ALIAS:
        return f"openai:{_GLM_MODEL_ALIAS[normalized]}"

    if ":" in trimmed:
        prefix, rest = trimmed.split(":", 1)
        norm_prefix = normalize_provider(prefix)
        normalized_rest = rest.strip().lower().replace("_", "-").replace(" ", "")
        if normalized_rest in _GLM_MODEL_ALIAS:
            rest = _GLM_MODEL_ALIAS[normalized_rest]
        if norm_prefix == "glm":
            return f"openai:{rest}"
        if prefix == "openai":
            return trimmed
        return f"openai:{rest}"

    return f"openai:{trimmed}"


def _to_deepseek_openai_model(model_name: str) -> str:
    trimmed = model_name.strip()
    if not trimmed:
        return "openai:deepseek-chat"

    normalized = trimmed.lower().replace("_", "-").replace(" ", "")
    if normalized in _DEEPSEEK_MODEL_ALIAS:
        return f"openai:{_DEEPSEEK_MODEL_ALIAS[normalized]}"

    if ":" in trimmed:
        prefix, rest = trimmed.split(":", 1)
        norm_prefix = normalize_provider(prefix)
        normalized_rest = rest.strip().lower().replace("_", "-").replace(" ", "")
        if normalized_rest in _DEEPSEEK_MODEL_ALIAS:
            rest = _DEEPSEEK_MODEL_ALIAS[normalized_rest]
        if norm_prefix == "deepseek":
            return f"openai:{rest}"
        if prefix == "openai":
            return trimmed
        return f"openai:{rest}"

    return f"openai:{trimmed}"


def build_chat_model(
    model_name: str,
    provider: str | None = None,
    temperature: float = 0,
):
    resolved_provider = resolve_provider(provider)
    kwargs: dict[str, object] = {"temperature": temperature}
    target_model = model_name

    if resolved_provider == "dashscope":
        target_model = _to_dashscope_openai_model(model_name)
        api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("JOB_AGENT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key
        kwargs["base_url"] = os.getenv("JOB_AGENT_BASE_URL", ALIYUN_COMPAT_BASE_URL)
    elif resolved_provider == "glm":
        target_model = _to_glm_openai_model(model_name)
        api_key = (
            os.getenv("GLM_API_KEY")
            or os.getenv("JOB_AGENT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key
        kwargs["base_url"] = os.getenv("JOB_AGENT_BASE_URL", GLM_COMPAT_BASE_URL)
    elif resolved_provider == "deepseek":
        target_model = _to_deepseek_openai_model(model_name)
        api_key = (
            os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("JOB_AGENT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key
        kwargs["base_url"] = os.getenv("JOB_AGENT_BASE_URL", DEEPSEEK_COMPAT_BASE_URL)

    return init_chat_model(target_model, **kwargs)
