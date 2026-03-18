from __future__ import annotations

import hashlib
import re

from app.logging_utils import log
from app.model_factory import build_chat_model
from app.schemas import JobJudgeResult


AI_KEYWORDS = [
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "llm",
    "nlp",
    "cv",
    "recommendation",
    "algorithm",
    "机器学习",
    "深度学习",
    "大模型",
    "推荐",
    "算法",
    "数据智能",
]

CAMPUS_KEYWORDS = [
    "校招",
    "应届",
    "实习",
    "intern",
    "graduate",
    "new grad",
]

TECH_TAG_MAP = {
    "llm": ["llm", "大模型", "rag", "agent"],
    "nlp": ["nlp", "自然语言"],
    "cv": ["cv", "计算机视觉", "图像"],
    "recommendation": ["推荐", "召回", "排序"],
    "ml": ["机器学习", "xgboost", "lightgbm"],
    "dl": ["深度学习", "pytorch", "tensorflow"],
    "data-mining": ["数据挖掘", "特征工程"],
}


class JobClassifier:
    def __init__(self, model_name: str | None, provider: str | None = None) -> None:
        self.model_name = model_name or ""
        self.provider = provider
        self.model = None
        self._llm_disabled = False
        self._cache: dict[str, JobJudgeResult] = {}
        if self.model_name:
            try:
                self.model = build_chat_model(
                    model_name=self.model_name,
                    provider=self.provider,
                    temperature=0,
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                log("warn", f"Classifier model init failed, fallback to heuristic mode: {exc}")
                self.model = None

    def classify(self, role_name: str, title: str, detail_text: str) -> JobJudgeResult:
        cache_key = self._cache_key(role_name=role_name, title=title, detail_text=detail_text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if self.model is None or self._llm_disabled:
            result = self._heuristic_classify(role_name=role_name, title=title, detail_text=detail_text)
            self._cache[cache_key] = result
            return result

        try:
            structured = self.model.with_structured_output(JobJudgeResult)
            prompt = self._build_prompt(role_name=role_name, title=title, detail_text=detail_text)
            result = structured.invoke(prompt)
            if isinstance(result, JobJudgeResult):
                self._cache[cache_key] = result
                return result
        except Exception as exc:  # pragma: no cover - runtime safeguard
            if self._is_rate_limit_error(exc):
                self._disable_llm(f"LLM rate-limited, downgrade to heuristic mode: {exc}")
            elif self._is_non_retryable_llm_error(exc):
                self._disable_llm(f"LLM unavailable for structured output, downgrade to heuristic mode: {exc}")
            else:
                log("warn", f"LLM structured classification failed, fallback to heuristic mode: {exc}")

        result = self._heuristic_classify(role_name=role_name, title=title, detail_text=detail_text)
        self._cache[cache_key] = result
        return result

    def _disable_llm(self, msg: str) -> None:
        self._llm_disabled = True
        self.model = None
        log("warn", msg)

    def _build_prompt(self, role_name: str, title: str, detail_text: str) -> str:
        excerpt = detail_text[:4500]
        return (
            "You are a job post evaluator.\n"
            "Determine whether this job is AI-related and whether it is campus/intern.\n"
            "Return strictly matching schema fields only.\n\n"
            f"Target role: {role_name}\n"
            f"Job title: {title}\n"
            f"Job detail: {excerpt}\n"
        )

    def _heuristic_classify(self, role_name: str, title: str, detail_text: str) -> JobJudgeResult:
        all_text = f"{title} {detail_text}".lower()
        ai_hits = sum(1 for word in AI_KEYWORDS if word in all_text)
        campus_hits = sum(1 for word in CAMPUS_KEYWORDS if word in all_text)

        title_hit = role_name.lower() in title.lower()
        is_ai_related = ai_hits >= 2 or (ai_hits >= 1 and title_hit)
        is_campus_or_intern = campus_hits >= 1

        tech_tags: list[str] = []
        for tag, keys in TECH_TAG_MAP.items():
            if any(key.lower() in all_text for key in keys):
                tech_tags.append(tag)

        requirements_summary = self._extract_requirements(detail_text)
        confidence = min(1.0, 0.35 + ai_hits * 0.12 + campus_hits * 0.18)
        reason = f"heuristic: ai_hits={ai_hits}, campus_hits={campus_hits}, title_hit={title_hit}"

        return JobJudgeResult(
            is_ai_related=is_ai_related,
            is_campus_or_intern=is_campus_or_intern,
            tech_tags=tech_tags,
            requirements_summary=requirements_summary,
            reason=reason,
            confidence=confidence,
        )

    def _extract_requirements(self, detail_text: str) -> str:
        text = re.sub(r"\s+", " ", detail_text).strip()
        if not text:
            return ""

        patterns = [
            r"(任职要求[:：\s].{40,220})",
            r"(岗位要求[:：\s].{40,220})",
            r"(岗位职责[:：\s].{40,220})",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(1)[:220]

        return text[:220]

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        markers = ("429", "rate limit", "too many requests", "速率限制", "1302")
        return any(token in text for token in markers)

    def _is_non_retryable_llm_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "invalid_request_error",
            "response_format",
            "unsupported",
            "unauthorized",
            "authentication",
            "api key",
            "401",
        )
        return any(token in text for token in markers)

    def _cache_key(self, role_name: str, title: str, detail_text: str) -> str:
        text_digest = hashlib.sha1(detail_text[:6000].encode("utf-8")).hexdigest()
        return f"{role_name.strip().lower()}|{title.strip().lower()}|{text_digest}"
