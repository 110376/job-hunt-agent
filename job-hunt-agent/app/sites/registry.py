from __future__ import annotations

from app.services.login_fetcher import get_login_fetcher
from app.sites.base import SiteAdapter


SITE_DOMAIN_MAP: dict[str, list[str]] = {
    "boss": ["zhipin.com"],
    "liepin": ["liepin.com"],
    "lagou": ["lagou.com"],
    "51job": ["51job.com"],
}


def normalize_site_name(name: str) -> str:
    return name.strip().lower()


def build_adapters(site_names: list[str]) -> dict[str, SiteAdapter]:
    adapters: dict[str, SiteAdapter] = {}
    login_fetcher = get_login_fetcher()
    for raw_name in site_names:
        site_name = normalize_site_name(raw_name)
        domains = SITE_DOMAIN_MAP.get(site_name)
        if not domains:
            continue
        adapters[site_name] = SiteAdapter(name=site_name, domains=domains, login_fetcher=login_fetcher)
    return adapters


def supported_sites() -> list[str]:
    return sorted(SITE_DOMAIN_MAP.keys())
