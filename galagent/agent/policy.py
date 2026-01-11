# galagent/agent/policy.py
from __future__ import annotations

from galagent.common.schemas import Decision, Observation


class DummyPolicy:
    """
    MVP: always choose the first option.
    Replace with LLM policy later (keep same interface).
    """

    def decide(self, obs: Observation, retrieved_hits: list[str]) -> Decision:
        return Decision(
            choice_index=0,
            rationale=f"MVP策略：先选第0个选项推进；检索命中={len(retrieved_hits)}。",
        )
