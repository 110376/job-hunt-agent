from __future__ import annotations

from dataclasses import dataclass
import os

from langchain.agents import create_agent

from app.logging_utils import log
from app.model_factory import build_chat_model
from app.schemas import RuntimeState
from app.services.classifier import JobClassifier
from app.services.exporter import export_records
from app.sites.registry import build_adapters, supported_sites
from app.tools import build_tools


SYSTEM_PROMPT = """
You are an AI job collection agent.

Goal:
1) Collect enough AI-related campus/intern job posts.
2) Cover at least two different sources.
3) Complete actions through tools and avoid fabrication.

Suggested workflow:
- call list_sites and get_query_hints first
- use search_jobs to get candidates
- use collect_job one-by-one to validate and store
- call get_progress after each batch
- if growth is weak, change query and site

Constraints:
- do real tool calls in each round
- page starts from 1
- search limit per call <= 10
""".strip()


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = ("429", "rate limit", "too many requests", "速率限制", "1302")
    return any(token in text for token in markers)


@dataclass
class RunResult:
    json_path: str
    csv_path: str
    total_collected: int
    source_counts: dict[str, int]


class JobAgentRunner:
    def __init__(
        self,
        role_name: str,
        target_count: int,
        site_names: list[str],
        output_dir: str,
        model_name: str,
        provider: str | None = None,
        max_rounds: int = 8,
    ) -> None:
        self.output_dir = output_dir
        self.model_name = model_name
        self.provider = provider
        self.max_rounds = max_rounds
        self.max_query_attempts = int(os.getenv("JOB_AGENT_MAX_QUERY_ATTEMPTS", "120"))
        self.auto_expand_sites = (os.getenv("JOB_AGENT_AUTO_EXPAND_SITES") or "1").strip() == "1"
        self.site_expand_no_growth_threshold = int(
            os.getenv("JOB_AGENT_SITE_EXPAND_NO_GROWTH_THRESHOLD", "2")
        )
        self.agent_start_collected = int(os.getenv("JOB_AGENT_AGENT_START_COLLECTED", "0"))
        self.agent = None
        self._agent_disabled_by_rate_limit = False

        adapters = build_adapters(site_names)
        if not adapters:
            raise ValueError("No available site adapter, please select at least one supported site")

        self.state = RuntimeState(
            role_name=role_name,
            target_count=target_count,
            selected_sites=sorted(adapters.keys()),
        )
        self.adapters = adapters

        classifier = JobClassifier(model_name=model_name, provider=provider)
        tool_build = build_tools(state=self.state, adapters=adapters, classifier=classifier)
        self.toolbox = tool_build.toolbox
        try:
            agent_model = build_chat_model(
                model_name=model_name,
                provider=provider,
                temperature=0,
            )
            self.agent = create_agent(
                model=agent_model,
                tools=tool_build.tools,
                system_prompt=SYSTEM_PROMPT,
                debug=False,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            log("warn", f"Agent init failed, fallback to deterministic mode: {exc}")
            self.agent = None

    def run(self) -> RunResult:
        log(
            "agent",
            (
                f"Start Agent: provider={self.provider or 'auto'} "
                f"model={self.model_name} sites={self.state.selected_sites}"
            ),
        )

        messages: list[dict[str, str]] = [{"role": "user", "content": self._build_goal_message()}]

        best_count = len(self.state.records)
        for idx in range(1, self.max_rounds + 1):
            if self._query_budget_exhausted():
                log("warn", "Query budget exhausted, stop early to avoid infinite loops")
                break

            self.state.round_index = idx
            before = len(self.state.records)
            log("agent", f"round={idx} collected={before}/{self.state.target_count}")

            messages.append({"role": "user", "content": self._build_round_message(idx)})
            boosted_in_this_round = False

            if self._should_invoke_agent():
                try:
                    result = self.agent.invoke(
                        {"messages": messages},
                        config={"recursion_limit": 80},
                    )
                    messages = result.get("messages", messages)
                except Exception as exc:
                    log("warn", f"Agent invoke failed, switch to deterministic boost: {exc}")
                    if _is_rate_limit_error(exc):
                        self._agent_disabled_by_rate_limit = True
                        self.agent = None
                        log("warn", "Rate-limited by model provider, disable Agent LLM for remaining rounds")
                    self._deterministic_boost(round_index=idx)
                    boosted_in_this_round = True
            else:
                self._deterministic_boost(round_index=idx)
                boosted_in_this_round = True

            after = len(self.state.records)
            if after <= before and not boosted_in_this_round:
                log("warn", f"round={idx} no growth, start deterministic boost")
                self._deterministic_boost(round_index=idx)
                after = len(self.state.records)

            if after > before:
                self.state.no_growth_rounds = 0
                best_count = max(best_count, after)
            else:
                self.state.no_growth_rounds += 1
                self._maybe_expand_site_pool()

            if self._is_goal_reached():
                log("agent", "Goal reached, stop collection")
                break

            if self.state.no_growth_rounds >= 3 and len(self.state.records) <= best_count:
                log("warn", "No growth for multiple rounds, stop early and export partial result")
                break

        json_path, csv_path = export_records(
            records=self.state.records,
            output_dir=self.output_dir,
            role_name=self.state.role_name,
        )
        log(
            "done",
            (
                f"collected={len(self.state.records)} "
                f"json={json_path} csv={csv_path} source_counts={self.state.source_counts}"
            ),
        )
        return RunResult(
            json_path=json_path,
            csv_path=csv_path,
            total_collected=len(self.state.records),
            source_counts=dict(self.state.source_counts),
        )

    def _build_goal_message(self) -> str:
        sites = ", ".join(self.state.selected_sites)
        return (
            f"Goal: collect {self.state.target_count} {self.state.role_name} campus/intern jobs. "
            f"Sites: {sites}. Use tools only."
        )

    def _build_round_message(self, round_index: int) -> str:
        return (
            f"Round {round_index}. "
            f"Current collected={len(self.state.records)}/{self.state.target_count}, "
            f"source_counts={self.state.source_counts}, "
            f"tried_queries={len(self.state.tried_queries)}, "
            f"no_growth_rounds={self.state.no_growth_rounds}. Continue tool calls."
        )

    def _is_goal_reached(self) -> bool:
        enough_count = len(self.state.records) >= self.state.target_count
        enough_source = len([k for k, v in self.state.source_counts.items() if v > 0]) >= 2
        return enough_count and enough_source

    def _need_more_work(self) -> bool:
        return not self._is_goal_reached()

    def _deterministic_boost(self, round_index: int) -> None:
        if not self._need_more_work():
            return

        if self._query_budget_exhausted():
            return

        hints = self.toolbox.get_query_hints_impl().get("hints", [])
        if not hints:
            return

        query_window = min(10, 4 + self.state.no_growth_rounds * 2)
        search_limit = 10 if self.state.no_growth_rounds >= 2 else 6
        sites = self._ordered_sites_by_round(round_index)

        for site in sites:
            for query in hints[:query_window]:
                if not self._need_more_work():
                    return
                if self._query_budget_exhausted():
                    return
                response = self.toolbox.search_jobs_impl(
                    site=site,
                    query=query,
                    page=min(round_index + self.state.no_growth_rounds, 5),
                    limit=search_limit,
                )
                if not response.get("ok"):
                    continue
                for item in response.get("items", []):
                    if not self._need_more_work():
                        return
                    self.toolbox.collect_job_impl(
                        site=site,
                        job_url=item.get("job_url", ""),
                        title=item.get("title", ""),
                    )

    def _ordered_sites_by_round(self, round_index: int) -> list[str]:
        sites = list(self.state.selected_sites)
        if not sites:
            return sites
        offset = (round_index - 1) % len(sites)
        ordered = sites[offset:] + sites[:offset]
        active_sites = [site for site in ordered if not self._is_site_paused(site, round_index)]
        return active_sites or ordered

    def _query_budget_exhausted(self) -> bool:
        return len(self.state.tried_queries) >= self.max_query_attempts

    def _is_site_paused(self, site: str, round_index: int) -> bool:
        pause_until = self.state.site_pause_until_round.get(site, 0)
        return round_index <= pause_until

    def _maybe_expand_site_pool(self) -> bool:
        if not self.auto_expand_sites:
            return False
        if self.state.no_growth_rounds < self.site_expand_no_growth_threshold:
            return False

        existing = set(self.state.selected_sites)
        candidates = [name for name in supported_sites() if name not in existing]
        for site_name in candidates:
            extra_adapters = build_adapters([site_name])
            adapter = extra_adapters.get(site_name)
            if adapter is None:
                continue
            # Mutate shared adapter map so toolbox can see new site immediately.
            self.adapters[site_name] = adapter
            self.state.selected_sites.append(site_name)
            log("agent", f"Low growth detected, expand site pool with '{site_name}'")
            return True
        return False

    def _should_invoke_agent(self) -> bool:
        if self.agent is None or self._agent_disabled_by_rate_limit:
            return False
        # Prefetch jobs by deterministic tools first to reduce LLM/API pressure.
        if len(self.state.records) < self.agent_start_collected:
            return False
        return True
