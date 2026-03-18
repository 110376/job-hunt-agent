from __future__ import annotations

from app.schemas import RuntimeState
from app.services.query_planner import AdaptiveQueryPlanner


def test_query_planner_generates_tier1_precision_hints() -> None:
    state = RuntimeState(
        role_name="AI Engineer",
        target_count=50,
        selected_sites=["boss", "liepin"],
    )
    planner = AdaptiveQueryPlanner(max_per_round=20)

    hints = planner.build_hints(state)

    assert len(hints) > 0
    assert any("AI Engineer 校招" in q for q in hints)
    assert any("AI Engineer 实习" in q for q in hints)


def test_query_planner_avoids_overtried_queries() -> None:
    state = RuntimeState(
        role_name="AI Engineer",
        target_count=50,
        selected_sites=["boss"],
    )
    state.no_growth_rounds = 2
    state.query_attempt_counts["AI Engineer 校招"] = 3

    planner = AdaptiveQueryPlanner(max_per_round=30)
    hints = planner.build_hints(state)

    assert "AI Engineer 校招" not in hints
    assert any("北京" in q or "上海" in q for q in hints)


def test_query_planner_enables_tier3_broad_fallback_queries() -> None:
    state = RuntimeState(
        role_name="AI Engineer",
        target_count=50,
        selected_sites=["boss", "liepin"],
    )
    state.no_growth_rounds = 3
    state.round_index = 6

    planner = AdaptiveQueryPlanner(max_per_round=60)
    hints = planner.build_hints(state)

    assert any("python工程师" in q for q in hints)
    assert any(("后端工程师" in q or "软件开发工程师" in q) and "AI" in q for q in hints)
