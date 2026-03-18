from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.agent_runner import JobAgentRunner
from app.logging_utils import log
from app.model_factory import resolve_model_settings
from app.sites.registry import build_adapters, supported_sites


def _input_role() -> str:
    role = input("Please input role name (default: AI Engineer): ").strip()
    return role or "AI Engineer"


def _input_target_count() -> int:
    raw = input("Please input target count (default: 50): ").strip()
    if not raw:
        return 50
    try:
        count = int(raw)
    except ValueError:
        print("Invalid number, fallback to default 50")
        return 50
    return max(1, count)


def _input_sites() -> list[str]:
    all_sites = supported_sites()
    print(f"Supported sites: {', '.join(all_sites)}")
    raw = input("Please input sites separated by comma (default: boss,liepin): ").strip()
    return _parse_sites(raw=raw, all_sites=all_sites, fallback=["boss", "liepin"])


def _parse_sites(raw: str, all_sites: list[str], fallback: list[str]) -> list[str]:
    raw = raw.strip()
    if not raw:
        chosen = fallback
    else:
        chosen = [x.strip().lower() for x in raw.split(",") if x.strip()]

    valid = [s for s in chosen if s in all_sites]
    invalid = [s for s in chosen if s not in all_sites]
    if invalid:
        log("warn", f"Ignore unsupported sites: {invalid}")
    if not valid:
        raise ValueError("No valid site selected, please retry")
    return sorted(set(valid))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Job Hunt Agent")
    parser.add_argument("--role", help="target role name")
    parser.add_argument("--target", type=int, help="target job count")
    parser.add_argument("--sites", help="comma-separated site names, e.g. boss,liepin")
    parser.add_argument("--provider", help="model provider, e.g. deepseek/glm/openai/dashscope")
    parser.add_argument("--model", help="model name, e.g. deepseek-chat")
    parser.add_argument("--max-rounds", type=int, default=8, help="max iterative rounds")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="do not prompt for missing inputs, use env/defaults directly",
    )
    parser.add_argument(
        "--prepare-login",
        action="store_true",
        help="open browser for manual login and save reusable session state",
    )
    parser.add_argument(
        "--login-sites",
        default="boss,liepin",
        help="sites to open during login preparation, e.g. boss,liepin",
    )
    return parser


def _resolve_inputs(args: argparse.Namespace) -> tuple[str, int, list[str]]:
    all_sites = supported_sites()

    role_name = args.role or os.getenv("JOB_AGENT_ROLE", "").strip()
    if not role_name and not args.non_interactive:
        role_name = _input_role()
    role_name = role_name or "AI Engineer"

    target_raw = args.target
    if target_raw is None:
        env_target = os.getenv("JOB_AGENT_TARGET", "").strip()
        if env_target:
            try:
                target_raw = int(env_target)
            except ValueError:
                log("warn", f"Invalid JOB_AGENT_TARGET={env_target}, fallback later")
    if target_raw is None and not args.non_interactive:
        target_count = _input_target_count()
    else:
        target_count = max(1, int(target_raw or 50))

    raw_sites = (args.sites or os.getenv("JOB_AGENT_SITES", "")).strip()
    if not raw_sites and not args.non_interactive:
        site_names = _input_sites()
    else:
        site_names = _parse_sites(raw=raw_sites, all_sites=all_sites, fallback=["boss", "liepin"])

    return role_name, target_count, site_names


def _validate_provider_key(provider: str) -> None:
    if provider == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("JOB_AGENT_API_KEY")
        if not key:
            log("warn", "Missing DEEPSEEK_API_KEY/JOB_AGENT_API_KEY; DeepSeek calls will fail")
        elif key.strip().lower().startswith(("http://", "https://")):
            log("warn", "Model key looks like a URL, not an API key; requests will likely fail")
        return

    if provider == "dashscope":
        key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("JOB_AGENT_API_KEY")
        if not key:
            log("warn", "Missing DASHSCOPE_API_KEY/JOB_AGENT_API_KEY; dashscope calls will fail")
        elif key.strip().lower().startswith(("http://", "https://")):
            log("warn", "Model key looks like a URL, not an API key; requests will likely fail")
        return

    if provider == "glm":
        key = os.getenv("GLM_API_KEY") or os.getenv("JOB_AGENT_API_KEY")
        if not key:
            log("warn", "Missing GLM_API_KEY/JOB_AGENT_API_KEY; GLM calls will fail")
        elif key.strip().lower().startswith(("http://", "https://")):
            log("warn", "Model key looks like a URL, not an API key; requests will likely fail")


def main() -> int:
    load_dotenv()
    args = _build_parser().parse_args()

    if args.prepare_login:
        return _prepare_login_state(args.login_sites)

    settings = resolve_model_settings(
        explicit_model_name=args.model or os.getenv("JOB_AGENT_MODEL"),
        explicit_provider=args.provider or os.getenv("JOB_AGENT_PROVIDER"),
    )
    model_name = settings.model_name
    provider = settings.provider
    output_dir = str((Path(__file__).resolve().parent / "outputs").resolve())
    _validate_provider_key(provider)

    try:
        role_name, target_count, site_names = _resolve_inputs(args)
    except Exception as exc:
        log("warn", f"Input stage failed: {exc}")
        return 1

    log(
        "input",
        (
            f"role={role_name} target_count={target_count} sites={site_names} "
            f"provider={provider} model={model_name}"
        ),
    )

    try:
        runner = JobAgentRunner(
            role_name=role_name,
            target_count=target_count,
            site_names=site_names,
            output_dir=output_dir,
            model_name=model_name,
            provider=provider,
            max_rounds=max(1, args.max_rounds),
        )
    except Exception as exc:
        log("warn", f"Initialization failed: {exc}")
        return 1

    try:
        result = runner.run()
    except Exception as exc:
        log("warn", f"Run failed: {exc}")
        return 2

    print("\nTask completed")
    print(f"- Collected: {result.total_collected}")
    print(f"- Source counts: {result.source_counts}")
    print(f"- JSON: {result.json_path}")
    print(f"- CSV : {result.csv_path}")
    return 0


def _prepare_login_state(raw_sites: str) -> int:
    try:
        all_sites = supported_sites()
        site_names = _parse_sites(raw=raw_sites, all_sites=all_sites, fallback=["boss", "liepin"])
        adapters = build_adapters(site_names)
        urls = [f"https://www.{adapter.domains[0]}" for adapter in adapters.values() if adapter.domains]

        from app.services.login_fetcher import prepare_login_session

        state_path = prepare_login_session(urls=urls)
        print("\nLogin session prepared.")
        print(f"- State file: {state_path}")
        print("- Set JOB_AGENT_USE_LOGIN_FETCHER=1 to enable login-mode fetching.")
        print(f"- Optional: set JOB_AGENT_LOGIN_STATE={state_path}")
        return 0
    except Exception as exc:
        log("warn", f"Prepare login failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
