# galagent/agent/runner.py AgentLoop
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from galagent.common.schemas import Observation
from galagent.env.kb_env import KBEnv
from galagent.memory.store import MemoryStore
from galagent.memory.retriever import KeywordRetriever
from galagent.agent.policy import DummyPolicy


@dataclass
class AgentConfig:
    max_steps: int = 50
    retrieve_top_k: int = 3
    verbose: bool = True


class GalgameAgent:
    def __init__(
        self,
        env: KBEnv,
        store: MemoryStore,
        retriever: KeywordRetriever,
        policy: DummyPolicy,
        config: AgentConfig,
    ):
        self.env = env
        self.store = store
        self.retriever = retriever
        self.policy = policy
        self.config = config

    def run(self) -> None:
        for step in range(self.config.max_steps):
            obs = self.env.observe()

            # ✅ STRICT: only is_ending == True ends loop
            if obs.is_ending is True:
                print("\n" + "=" * 70)
                print(f"[END] node_id={obs.node_id} | name={obs.name}")
                print(obs.text)
                return

            # store current scene text into memory
            self.store.add(obs.text, meta={"node_id": obs.node_id, "name": obs.name})

            # retrieve
            hits = self.retriever.search(obs.text, top_k=self.config.retrieve_top_k)

            # decide
            decision = self.policy.decide(obs, hits)

            # log
            if self.config.verbose:
                print("\n" + "=" * 70)
                print(f"[STEP {step}] node={obs.node_id} | name={obs.name}")
                print("TEXT:", obs.text)
                print("CHOICES:")
                for c in obs.choices:
                    print(f"  ({c.index}) {c.text}")
                if hits:
                    print("RETRIEVED:")
                    for h in hits:
                        print("  -", h)
                print("RATIONALE:", decision.rationale)

            # act
            self.env.choose(decision.choice_index)

        raise RuntimeError("Max steps reached without reaching an ending (is_ending: true).")
