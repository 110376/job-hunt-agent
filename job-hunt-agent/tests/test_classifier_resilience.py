from __future__ import annotations

from app.schemas import JobJudgeResult


class _RateLimitedStructured:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _prompt: str):
        self.calls += 1
        raise RuntimeError("Error code: 429 - rate limit")


class _RateLimitedModel:
    def __init__(self) -> None:
        self.structured = _RateLimitedStructured()

    def with_structured_output(self, _schema):
        return self.structured


class _InvalidRequestStructured:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _prompt: str):
        self.calls += 1
        raise RuntimeError(
            "Error code: 400 - {'error': {'message': 'This response_format type is unavailable now', "
            "'type': 'invalid_request_error'}}"
        )


class _InvalidRequestModel:
    def __init__(self) -> None:
        self.structured = _InvalidRequestStructured()

    def with_structured_output(self, _schema):
        return self.structured


class _OkStructured:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _prompt: str):
        self.calls += 1
        return JobJudgeResult(
            is_ai_related=True,
            is_campus_or_intern=True,
            tech_tags=["llm"],
            requirements_summary="ok",
            reason="ok",
            confidence=0.9,
        )


class _OkModel:
    def __init__(self) -> None:
        self.structured = _OkStructured()

    def with_structured_output(self, _schema):
        return self.structured


def test_classifier_disables_llm_after_rate_limit(monkeypatch) -> None:
    from app.services import classifier as cls

    fake_model = _RateLimitedModel()
    monkeypatch.setattr(cls, "build_chat_model", lambda **_kwargs: fake_model)

    c = cls.JobClassifier(model_name="glm-4.7", provider="glm")

    result_1 = c.classify(
        role_name="AI Engineer",
        title="AI Engineer 实习",
        detail_text="这是一个 AI 实习岗位，要求熟悉 LLM 和 Python",
    )
    result_2 = c.classify(
        role_name="AI Engineer",
        title="AI Engineer 校招",
        detail_text="这是一个 AI 校招岗位，要求熟悉 NLP",
    )

    assert isinstance(result_1, JobJudgeResult)
    assert isinstance(result_2, JobJudgeResult)
    assert fake_model.structured.calls == 1
    assert c.model is None


def test_classifier_uses_cache_to_reduce_repeated_llm_calls(monkeypatch) -> None:
    from app.services import classifier as cls

    fake_model = _OkModel()
    monkeypatch.setattr(cls, "build_chat_model", lambda **_kwargs: fake_model)

    c = cls.JobClassifier(model_name="glm-4.7", provider="glm")
    r1 = c.classify("AI Engineer", "AI Engineer 实习", "AI 实习, LLM")
    r2 = c.classify("AI Engineer", "AI Engineer 实习", "AI 实习, LLM")

    assert isinstance(r1, JobJudgeResult)
    assert isinstance(r2, JobJudgeResult)
    assert fake_model.structured.calls == 1


def test_classifier_disables_llm_after_non_retryable_invalid_request(monkeypatch) -> None:
    from app.services import classifier as cls

    fake_model = _InvalidRequestModel()
    monkeypatch.setattr(cls, "build_chat_model", lambda **_kwargs: fake_model)

    c = cls.JobClassifier(model_name="deepseek-chat", provider="deepseek")
    r1 = c.classify("AI Engineer", "AI Engineer 实习", "AI 实习, 机器学习, 校招")
    r2 = c.classify("AI Engineer", "算法工程师 校招", "机器学习, 大模型, 校招")

    assert isinstance(r1, JobJudgeResult)
    assert isinstance(r2, JobJudgeResult)
    assert fake_model.structured.calls == 1
    assert c.model is None


def test_classifier_heuristic_supports_chinese_ai_campus_keywords() -> None:
    from app.services.classifier import JobClassifier

    classifier = JobClassifier(model_name=None, provider=None)
    result = classifier.classify(
        role_name="AI Engineer",
        title="算法工程师（校招）",
        detail_text="岗位职责：负责机器学习与大模型方向研发，面向2026届校招/实习同学。",
    )

    assert result.is_ai_related is True
    assert result.is_campus_or_intern is True
