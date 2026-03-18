from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.logging_utils import log


_LOGIN_FETCHER: "PlaywrightLoginFetcher | None" = None


def resolve_login_state_path() -> str:
    raw = (os.getenv("JOB_AGENT_LOGIN_STATE") or ".job_agent_login_state.json").strip()
    return str(Path(raw).resolve())


@dataclass
class PlaywrightLoginFetcher:
    storage_state_path: str
    headless: bool = True
    timeout_ms: int = 30_000
    wait_after_load_ms: int = 1500
    browser_name: str = "chromium"
    _playwright: Any = None
    _browser: Any = None
    _context: Any = None

    def fetch_html(self, url: str) -> str:
        self._ensure_context()
        page = self._context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            if self.wait_after_load_ms > 0:
                page.wait_for_timeout(self.wait_after_load_ms)
            return page.content()
        finally:
            page.close()

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def _ensure_context(self) -> None:
        if self._context is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "playwright is required for login fetcher. "
                "Install with: python -m pip install playwright && python -m playwright install chromium"
            ) from exc

        self._playwright = sync_playwright().start()
        browser_type = getattr(self._playwright, self.browser_name, None) or self._playwright.chromium
        self._browser = browser_type.launch(headless=self.headless)
        context_kwargs: dict[str, Any] = {}
        if Path(self.storage_state_path).exists():
            context_kwargs["storage_state"] = self.storage_state_path
        self._context = self._browser.new_context(**context_kwargs)


def get_login_fetcher() -> PlaywrightLoginFetcher | None:
    global _LOGIN_FETCHER
    if (os.getenv("JOB_AGENT_USE_LOGIN_FETCHER") or "").strip() != "1":
        return None
    if _LOGIN_FETCHER is not None:
        return _LOGIN_FETCHER

    headless = (os.getenv("JOB_AGENT_LOGIN_HEADLESS") or "1").strip() == "1"
    wait_ms = int((os.getenv("JOB_AGENT_LOGIN_WAIT_MS") or "1500").strip())
    browser_name = (os.getenv("JOB_AGENT_LOGIN_BROWSER") or "chromium").strip()
    fetcher = PlaywrightLoginFetcher(
        storage_state_path=resolve_login_state_path(),
        headless=headless,
        wait_after_load_ms=wait_ms,
        browser_name=browser_name,
    )
    _LOGIN_FETCHER = fetcher
    atexit.register(_safe_close_login_fetcher)
    return _LOGIN_FETCHER


def prepare_login_session(urls: list[str]) -> str:
    state_path = resolve_login_state_path()
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "playwright is required to prepare login session. "
            "Install with: python -m pip install playwright && python -m playwright install chromium"
        ) from exc

    targets = urls or ["https://www.zhipin.com", "https://www.liepin.com"]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context_kwargs: dict[str, Any] = {}
        if Path(state_path).exists():
            context_kwargs["storage_state"] = state_path
        context = browser.new_context(**context_kwargs)
        page = context.new_page()
        for url in targets:
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        print("\nPlease complete login in the opened browser, then return to terminal.")
        input("Press Enter to save login state and continue...")
        context.storage_state(path=state_path)
        context.close()
        browser.close()
    log("done", f"saved login state to {state_path}")
    return state_path


def reset_login_fetcher_for_test() -> None:
    global _LOGIN_FETCHER
    _safe_close_login_fetcher()
    _LOGIN_FETCHER = None


def _safe_close_login_fetcher() -> None:
    global _LOGIN_FETCHER
    if _LOGIN_FETCHER is None:
        return
    try:
        _LOGIN_FETCHER.close()
    except Exception:
        pass
