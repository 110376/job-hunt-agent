from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterable, Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from lxml import html

from app.logging_utils import log
from app.schemas import SearchCandidate


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

ANTI_BOT_MARKERS = (
    "请完成验证",
    "访问受限",
    "异常流量",
    "行为验证",
    "人机验证",
    "captcha",
    "robot check",
    "security check",
)


class HtmlFetcher(Protocol):
    def fetch_html(self, url: str) -> str: ...


@dataclass
class SiteAdapter:
    name: str
    domains: list[str]
    login_fetcher: HtmlFetcher | None = None

    def search_jobs(self, query: str, page: int = 1, limit: int = 10) -> list[SearchCandidate]:
        query = query.strip()
        page = max(page, 1)
        limit = max(1, min(limit, 20))

        engines = self._resolve_search_engines()
        searchers: dict[str, Callable[..., list[SearchCandidate]]] = {
            "ddg": self._search_duckduckgo,
            "sogou": self._search_sogou,
        }

        results: list[SearchCandidate] = []
        seen: set[str] = set()
        for engine in engines:
            searcher = searchers.get(engine)
            if searcher is None:
                continue
            try:
                engine_results = searcher(query=query, page=page, limit=limit)
            except Exception:
                continue

            for item in engine_results:
                norm_url = item.job_url.strip()
                if not norm_url or norm_url in seen:
                    continue
                seen.add(norm_url)
                results.append(item)
                if len(results) >= limit:
                    return results

        return results

    def _resolve_search_engines(self) -> list[str]:
        raw = (os.getenv("JOB_AGENT_SEARCH_ENGINES") or "ddg,sogou").strip()
        candidates = [p.strip().lower() for p in raw.split(",") if p.strip()]
        if not candidates:
            candidates = ["ddg", "sogou"]
        valid: list[str] = []
        for name in candidates:
            if name in {"ddg", "sogou"} and name not in valid:
                valid.append(name)
        return valid or ["ddg", "sogou"]

    def _search_duckduckgo(self, query: str, page: int, limit: int) -> list[SearchCandidate]:
        ddg_query = f"{query} site:{self.domains[0]}"
        offset = max(page - 1, 0) * 30
        url = f"https://duckduckgo.com/html/?q={quote_plus(ddg_query)}&s={offset}"
        html_text = self._http_get_text(url)
        return self._parse_duckduckgo_results(html_text=html_text, limit=limit)

    def _search_sogou(self, query: str, page: int, limit: int) -> list[SearchCandidate]:
        sogou_query = f"{query} site:{self.domains[0]}"
        url = f"https://www.sogou.com/web?query={quote_plus(sogou_query)}&page={max(1, page)}"
        html_text = self._http_get_text(url)
        return self._parse_sogou_results(html_text=html_text, limit=limit)

    def _http_get_text(self, url: str, timeout: float = 15.0) -> str:
        errors: list[Exception] = []
        for trust_env in (True, False):
            for attempt in range(2):
                try:
                    with httpx.Client(
                        timeout=timeout,
                        headers=DEFAULT_HEADERS,
                        follow_redirects=True,
                        trust_env=trust_env,
                    ) as client:
                        resp = client.get(url)
                        resp.raise_for_status()
                        return resp.text
                except Exception as exc:
                    errors.append(exc)
                    time.sleep(0.25 * (attempt + 1))
        raise errors[-1]

    def _parse_duckduckgo_results(self, html_text: str, limit: int) -> list[SearchCandidate]:
        doc = html.fromstring(html_text)
        anchors = doc.xpath("//a[contains(@class, 'result__a')]")
        if not anchors:
            anchors = doc.xpath("//h2//a")

        results: list[SearchCandidate] = []
        seen: set[str] = set()
        for anchor in anchors:
            href = (anchor.get("href") or "").strip()
            resolved_url = self._resolve_url(href)
            if not resolved_url:
                continue
            if not self._in_domains(resolved_url):
                continue
            if resolved_url in seen:
                continue
            seen.add(resolved_url)

            title = " ".join(anchor.itertext()).strip()
            snippet = self._extract_snippet(anchor)
            results.append(
                SearchCandidate(
                    title=title or "Unknown Position",
                    job_url=resolved_url,
                    source=self.name,
                    snippet=snippet,
                )
            )
            if len(results) >= limit:
                break

        return results

    def _parse_sogou_results(self, html_text: str, limit: int) -> list[SearchCandidate]:
        doc = html.fromstring(html_text)
        wraps = doc.xpath("//div[contains(@class, 'vrwrap')]")

        results: list[SearchCandidate] = []
        seen: set[str] = set()
        for wrap in wraps:
            title_node = wrap.xpath(".//h3//a[1]")
            if not title_node:
                continue

            title = " ".join(title_node[0].xpath(".//text()")).strip()
            data_urls = wrap.xpath(".//*[@data-url]/@data-url")
            job_url = (data_urls[0] if data_urls else "").strip()

            if not job_url:
                href = (title_node[0].get("href") or "").strip()
                job_url = self._resolve_url(href) or ""

            if not job_url:
                continue
            if not self._in_domains(job_url):
                continue
            if job_url in seen:
                continue
            seen.add(job_url)

            snippet = re.sub(r"\s+", " ", " ".join(wrap.xpath(".//text()"))).strip()
            results.append(
                SearchCandidate(
                    title=title or "Unknown Position",
                    job_url=job_url,
                    source=self.name,
                    snippet=snippet[:200],
                )
            )
            if len(results) >= limit:
                break

        return results

    def fetch_job_detail(self, job_url: str) -> dict[str, str]:
        html_text: str | None = None
        http_error: Exception | None = None

        if self._force_login_fetcher():
            html_text = self._fetch_detail_html_with_login(job_url)

        if not html_text:
            try:
                html_text = self._fetch_detail_html_http(job_url)
            except Exception as exc:
                http_error = exc
                html_text = None

        if (not html_text or self._looks_like_antibot(html_text)) and self._should_use_login_fetcher():
            fallback_html = self._fetch_detail_html_with_login(job_url)
            if fallback_html:
                html_text = fallback_html

        if not html_text:
            if http_error is not None:
                raise http_error
            raise RuntimeError(f"Failed to fetch detail html for url={job_url}")

        return self._parse_job_detail_html(html_text)

    def _fetch_detail_html_http(self, job_url: str) -> str:
        return self._http_get_text(job_url, timeout=20.0)

    def _fetch_detail_html_with_login(self, job_url: str) -> str | None:
        fetcher = self.login_fetcher or self._build_login_fetcher()
        if fetcher is None:
            return None
        try:
            return fetcher.fetch_html(job_url)
        except Exception as exc:
            log("warn", f"login fetcher failed for url={job_url}: {exc}")
            return None

    def _build_login_fetcher(self) -> HtmlFetcher | None:
        if not self._should_use_login_fetcher():
            return None
        try:
            from app.services.login_fetcher import get_login_fetcher

            return get_login_fetcher()
        except Exception as exc:
            log("warn", f"failed to initialize login fetcher: {exc}")
            return None

    def _should_use_login_fetcher(self) -> bool:
        return (os.getenv("JOB_AGENT_USE_LOGIN_FETCHER") or "").strip() == "1"

    def _force_login_fetcher(self) -> bool:
        return (os.getenv("JOB_AGENT_FORCE_LOGIN_FETCHER") or "").strip() == "1"

    def _looks_like_antibot(self, html_text: str) -> bool:
        text = html_text.lower()
        return any(marker.lower() in text for marker in ANTI_BOT_MARKERS)

    def _parse_job_detail_html(self, html_text: str) -> dict[str, str]:
        doc = html.fromstring(html_text)
        for node in doc.xpath("//script|//style|//noscript"):
            node.drop_tree()

        title = ""
        title_nodes = doc.xpath("//title/text()")
        if title_nodes:
            title = title_nodes[0].strip()

        texts = [t.strip() for t in doc.xpath("//body//text()") if t and t.strip()]
        merged = re.sub(r"\s+", " ", " ".join(texts)).strip()
        merged = merged[:12000]

        return {
            "title": title,
            "text": merged,
            "company": self._extract_company(title, merged),
            "location": self._extract_location(merged),
            "salary": self._extract_salary(merged),
        }

    def _in_domains(self, url: str) -> bool:
        netloc = urlparse(url).netloc.lower()
        return any(domain in netloc for domain in self.domains)

    def _resolve_url(self, href: str) -> str | None:
        if not href:
            return None
        parsed = urlparse(href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            query_map = parse_qs(parsed.query)
            uddg = query_map.get("uddg", [""])[0]
            if uddg:
                return unquote(uddg)
        if href.startswith("http://") or href.startswith("https://"):
            return href
        return None

    def _extract_snippet(self, anchor_node: Any) -> str:
        container = anchor_node.getparent()
        if container is None:
            return ""
        text = " ".join(container.xpath(".//text()"))
        text = re.sub(r"\s+", " ", text).strip()
        return text[:200]

    def _extract_company(self, title: str, text: str) -> str:
        parts = [p.strip() for p in re.split(r"[-_|]", title) if p.strip()]
        if len(parts) >= 2:
            return parts[1][:60]

        m = re.search(r"(?:公司|企业)[:：\s]*([\u4e00-\u9fffA-Za-z0-9()（）\-·]{2,40})", text)
        if m:
            return m.group(1)
        return "unknown"

    def _extract_salary(self, text: str) -> str:
        patterns: Iterable[str] = [
            r"(\d{1,2}(?:\.\d+)?\s*[kK]\s*[-~]\s*\d{1,2}(?:\.\d+)?\s*[kK])",
            r"(\d{1,3}\s*[-~]\s*\d{1,3}\s*万/年)",
            r"(\d{1,3}\s*[-~]\s*\d{1,3}\s*元/月)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(1).replace(" ", "")
        return "unknown"

    def _extract_location(self, text: str) -> str:
        m = re.search(r"(北京|上海|广州|深圳|杭州|成都|武汉|西安|苏州|南京|天津|重庆)", text)
        if m:
            return m.group(1)
        m = re.search(r"工作地点[:：\s]*([\u4e00-\u9fff]{2,8})", text)
        if m:
            return m.group(1)
        return "unknown"
