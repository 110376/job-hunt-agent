from __future__ import annotations

from dataclasses import dataclass

from app.schemas import RuntimeState


PRECISION_ROLE_QUERIES = [
    "AI Engineer",
    "机器学习工程师",
    "算法工程师",
    "大模型工程师",
    "LLM Engineer",
    "NLP 算法工程师",
    "CV 算法工程师",
    "推荐算法工程师",
    "AIGC Engineer",
]

ENTRY_LEVEL_TERMS = ["校招", "实习", "应届"]
CITY_TERMS = ["北京", "上海", "深圳", "杭州", "广州", "成都", "南京", "武汉"]

RELATED_AI_QUERIES = [
    "AI 应用工程师",
    "AIGC 算法工程师",
    "多模态算法工程师",
    "推荐系统工程师",
    "NLP 实习",
    "CV 实习",
]

BROAD_ROLE_QUERIES = [
    "python工程师",
    "后端工程师",
    "软件开发工程师",
    "数据开发工程师",
    "数据工程师",
]

AI_CONTEXT_TERMS = ["AI", "机器学习", "大模型", "LLM", "NLP", "CV", "推荐算法"]


@dataclass
class AdaptiveQueryPlanner:
    max_per_round: int = 14

    def build_hints(self, state: RuntimeState) -> list[str]:
        active_tier = self._resolve_tier(state)
        tiered_candidates = self._build_tiered_candidates(state, active_tier)
        tier_quota = self._tier_quota(active_tier)

        hints: list[str] = []
        for tier in range(1, active_tier + 1):
            ranked = sorted(
                tiered_candidates[tier],
                key=lambda query: self._query_rank(state, query, tier),
            )
            for query in ranked:
                if len(hints) >= self.max_per_round:
                    return hints
                if len([q for q in hints if q in tiered_candidates[tier]]) >= tier_quota[tier]:
                    break
                if self._skip_query(state, query, hints):
                    continue
                hints.append(query)

        # Backfill from all active tiers when a tier has too few candidates.
        merged: list[tuple[int, str]] = []
        for tier in range(1, active_tier + 1):
            for query in tiered_candidates[tier]:
                merged.append((tier, query))
        merged = sorted(merged, key=lambda it: self._query_rank(state, it[1], it[0]))
        for tier, query in merged:
            if len(hints) >= self.max_per_round:
                break
            if self._skip_query(state, query, hints):
                continue
            hints.append(query)

        return hints

    def _resolve_tier(self, state: RuntimeState) -> int:
        collected = len(state.records)
        target = max(1, state.target_count)
        progress = collected / target

        if state.no_growth_rounds >= 3 or state.round_index >= 6:
            return 3
        if state.no_growth_rounds >= 2 or (state.round_index >= 3 and progress < 0.8):
            return 2
        return 1

    def _build_tiered_candidates(self, state: RuntimeState, active_tier: int) -> dict[int, list[str]]:
        role = state.role_name.strip() or "AI Engineer"
        tiered: dict[int, list[str]] = {1: [], 2: [], 3: []}

        # Tier 1: precise role + campus intent
        for term in ENTRY_LEVEL_TERMS:
            tiered[1].append(f"{role} {term}")
        for base in PRECISION_ROLE_QUERIES:
            for term in ENTRY_LEVEL_TERMS:
                tiered[1].append(f"{base} {term}")

        if active_tier >= 2:
            # Tier 2: broaden to city and adjacent AI roles
            for city in CITY_TERMS:
                tiered[2].append(f"{role} 校招 {city}")
                tiered[2].append(f"{role} 实习 {city}")
            for query in RELATED_AI_QUERIES:
                for term in ENTRY_LEVEL_TERMS:
                    tiered[2].append(f"{query} {term}")

        if active_tier >= 3:
            # Tier 3: broad roles + AI context guard
            for broad in BROAD_ROLE_QUERIES:
                for term in ENTRY_LEVEL_TERMS:
                    tiered[3].append(f"{broad} AI {term}")
                    tiered[3].append(f"{broad} 机器学习 {term}")
            for broad in BROAD_ROLE_QUERIES:
                for ai_term in AI_CONTEXT_TERMS:
                    tiered[3].append(f"{broad} {ai_term} 实习")

        for tier in tiered:
            cleaned: list[str] = []
            for query in tiered[tier]:
                normalized = " ".join(query.split())
                if normalized and normalized not in cleaned:
                    cleaned.append(normalized)
            tiered[tier] = cleaned

        return tiered

    def _tier_quota(self, active_tier: int) -> dict[int, int]:
        if active_tier == 1:
            return {1: self.max_per_round, 2: 0, 3: 0}
        if active_tier == 2:
            tier1 = min(8, self.max_per_round)
            return {1: tier1, 2: self.max_per_round - tier1, 3: 0}
        # Tier 3: guarantee fallback queries can enter the queue.
        tier1 = min(6, self.max_per_round)
        tier2 = min(4, max(self.max_per_round - tier1, 0))
        tier3 = max(self.max_per_round - tier1 - tier2, 0)
        return {1: tier1, 2: tier2, 3: tier3}

    def _query_rank(self, state: RuntimeState, query: str, tier: int) -> tuple[int, int, int, int]:
        attempts = state.query_attempt_counts.get(query, 0)
        hits = state.query_hit_counts.get(query, 0)
        role_bonus = 0 if state.role_name.lower() in query.lower() else 1
        return (-hits, attempts, tier, role_bonus)

    def _skip_query(self, state: RuntimeState, query: str, chosen: list[str]) -> bool:
        if query in chosen:
            return True
        return state.query_attempt_counts.get(query, 0) >= 3
