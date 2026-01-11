# galAgent.py
from __future__ import annotations

from pathlib import Path

from galagent.env.kb_env import KBEnv, KBEnvConfig
from galagent.memory.store import MemoryStore
from galagent.memory.retriever import KeywordRetriever
from galagent.agent.policy import DummyPolicy
from galagent.agent.runner import AgentConfig, GalgameAgent
from galagent.agent.llm_policy import LLMPolicy,load_llm_config


def main():
    ROOT = Path(__file__).resolve().parent
    scenes_path = ROOT / "dataset" / "scenes.json"
    llm_config_path = ROOT / "config.yaml" 
    llm_config = load_llm_config(llm_config_path)
    
    env = KBEnv(KBEnvConfig(scenes_path=scenes_path, start_node_id="start"))
    store = MemoryStore()
    retriever = KeywordRetriever(store)
    
    #policy = DummyPolicy() 
    policy = LLMPolicy(llm_config)
    
    agent = GalgameAgent(
        env=env,
        store=store,
        retriever=retriever,
        policy=policy,
        config=AgentConfig(max_steps=50, retrieve_top_k=3, verbose=True),
    )

    agent.run()


if __name__ == "__main__":
    main()
