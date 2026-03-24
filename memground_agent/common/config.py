# memground_agent/common/config.py
'''Loads all configuration sections'''
from __future__ import annotations

import yaml
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_output_tokens: int
    goal_instruction: str


@dataclass
class EmbeddingConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    dim: int
    use_real: bool


@dataclass
class MemAgentConfig:
    """Memory agent configuration (supports Mem0, A-mem, etc.)"""
    mem_name: str = "mem_0"  # "mem_0" or "a_mem"
    use_mem: bool = False  # Whether to use a memory agent for memory management
    clear_on_start: bool = True  # Whether to clear memory on startup

    # Mem0 configuration
    mem0_api_key: str = ""

    # A-mem configuration
    amem_embedding_model: str = "all-MiniLM-L6-v2"
    amem_llm_backend: str = "openai"  # openai or ollama
    amem_llm_model: str = "gpt-4o-mini"


@dataclass
class AgentConfig:
    """Agent runtime configuration"""
    max_steps: int = 50
    retrieve_top_k: int = 3
    verbose: bool = True
    retriever_type: str = "vector"  # keyword or vector
    max_context_tokens: int = 4000  # Maximum context token count
    enable_compression: bool = True  # Whether to enable memory compression
    compression_threshold: int = 20  # Trigger compression when conversation rounds exceed this value
    compression_count: int = 10  # Number of earliest conversation rounds to compress each time


@dataclass
class CheckpointConfig:
    """Checkpoint configuration"""
    enabled: bool = False  # Whether to enable checkpoints
    interval: int = 100  # Save interval in steps
    dir: str = "checkpoints"  # Directory where checkpoints are saved
    resume_from: Optional[str] = None  # Resume from a specific checkpoint (file path)


@dataclass
class EnvConfig:
    """Environment configuration"""
    game_type: str = "type_help"  # type_help, no_case_should_remain_unsolved, trpg
    scenes_path: str = "dataset/scenes.json"
    start_node_id: str = "start"
    test_language: str = "en"  # en (English prompts)
    enable_hint: bool = False  # Whether to enable the failure-count hint feature (type_help only)
    hint_failure_threshold: int = 15  # Consecutive failure threshold to trigger a hint (type_help only)
    provide_naming_rules: bool = False  # Whether to explicitly state file naming rules in the prompt (type_help only)
    show_order_judgements_history: bool = True  # dust only: show all past ordering judgements in prompt
    # TRPG specific fields
    story_name: str = "Terror_on_the_Orient_Express"
    data_dir: str = "dataset/trpg_en/data"
    qa_dir: str = "dataset/trpg_en/qa"

class ConfigLoader:
    """Configuration loader for loading all configuration sections"""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path.resolve()}")

        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def load_llm_config(self) -> LLMConfig:
        """Load LLM configuration"""
        llm = self._data.get("llm")
        if not llm:
            raise ValueError("LLM config not found in config file")

        required = ["provider", "api_key", "base_url", "model", "temperature", "max_output_tokens", "goal_instruction"]
        for k in required:
            if k not in llm:
                raise ValueError(f"Missing LLM config field: {k}")

        return LLMConfig(
            provider=str(llm["provider"]),
            api_key=str(llm["api_key"]),
            base_url=str(llm["base_url"]),
            model=str(llm["model"]),
            temperature=float(llm["temperature"]),
            max_output_tokens=int(llm["max_output_tokens"]),
            goal_instruction=str(llm["goal_instruction"]),
        )

    def load_embedding_config(self) -> EmbeddingConfig:
        """Load Embedding configuration"""
        embedding = self._data.get("embedding")
        if not embedding:
            raise ValueError("Embedding config not found in config file")

        required = ["provider", "api_key", "base_url", "model", "dim", "use_real"]
        for k in required:
            if k not in embedding:
                raise ValueError(f"Missing Embedding config field: {k}")

        return EmbeddingConfig(
            provider=str(embedding["provider"]),
            api_key=str(embedding["api_key"]),
            base_url=str(embedding["base_url"]),
            model=str(embedding["model"]),
            dim=int(embedding["dim"]),
            use_real=bool(embedding["use_real"]),
        )

    def load_agent_config(self) -> AgentConfig:
        """Load Agent configuration"""
        agent = self._data.get("agent", {})

        return AgentConfig(
            max_steps=int(agent.get("max_steps", 50)),
            retrieve_top_k=int(agent.get("retrieve_top_k", 3)),
            verbose=bool(agent.get("verbose", True)),
            retriever_type=str(agent.get("retriever_type", "vector")),
            max_context_tokens=int(agent.get("max_context_tokens", 4000)),
            enable_compression=bool(agent.get("enable_compression", True)),
            compression_threshold=int(agent.get("compression_threshold", 20)),
            compression_count=int(agent.get("compression_count", 10)),
        )

    def load_checkpoint_config(self) -> CheckpointConfig:
        """Load Checkpoint configuration"""
        checkpoint = self._data.get("checkpoint", {})

        return CheckpointConfig(
            enabled=bool(checkpoint.get("enabled", False)),
            interval=int(checkpoint.get("interval", 100)),
            dir=str(checkpoint.get("dir", "checkpoints")),
            resume_from=checkpoint.get("resume_from"),  # can be None
        )

    def load_mem_agent_config(self) -> MemAgentConfig:
        """Load memory agent configuration"""
        mem_agent = self._data.get("mem_agent", {})

        return MemAgentConfig(
            mem_name=str(mem_agent.get("mem_name", "mem_0")),
            use_mem=bool(mem_agent.get("use_mem", False)),
            clear_on_start=bool(mem_agent.get("clear_on_start", True)),
            # Mem0 configuration
            mem0_api_key=str(mem_agent.get("mem0_api_key", "")),
            # A-mem configuration
            amem_embedding_model=str(mem_agent.get("amem_embedding_model", "all-MiniLM-L6-v2")),
            amem_llm_backend=str(mem_agent.get("amem_llm_backend", "openai")),
            amem_llm_model=str(mem_agent.get("amem_llm_model", "gpt-4o-mini")),
        )
    
    def load_judge_llm_config(self) -> Optional[LLMConfig]:
        """Load judge-specific LLM configuration; returns None if not configured (falls back to the answer model)"""
        j = self._data.get("judge_llm")
        if not j:
            return None
        llm = self._data.get("llm", {})
        return LLMConfig(
            provider=str(j.get("provider", llm.get("provider", "openai"))),
            api_key=str(j["api_key"]),
            base_url=str(j["base_url"]),
            model=str(j["model"]),
            temperature=float(j.get("temperature", 0.0)),
            max_output_tokens=int(j.get("max_output_tokens", llm.get("max_output_tokens", 1024))),
            goal_instruction="",
        )

    def load_env_config(self) -> EnvConfig:
        """Load environment configuration"""
        env = self._data.get("env", {})
        game_type = str(env.get("game_type", "type_help"))
        sub = env.get(game_type, {}) or {}

        def get(key, default):
            return sub.get(key, env.get(key, default))

        return EnvConfig(
            game_type=game_type,
            scenes_path=str(get("scenes_path", "dataset/scenes.json")),
            start_node_id=str(get("start_node_id", "start")),
            test_language=str(get("test_language", "en")),
            enable_hint=bool(get("enable_hint", False)),
            hint_failure_threshold=int(get("hint_failure_threshold", 15)),
            provide_naming_rules=bool(get("provide_naming_rules", False)),
            show_order_judgements_history=bool(get("show_order_judgements_history", True)),
            story_name=str(get("story_name", "Terror_on_the_Orient_Express")),
            data_dir=str(get("data_dir", "dataset/trpg_en/data")),
            qa_dir=str(get("qa_dir", "dataset/trpg_en/qa")),
        )


def load_llm_config(path: Path) -> LLMConfig:
    """Load LLM configuration (maintains backward compatibility)"""
    return ConfigLoader(path).load_llm_config()


def load_embedding_config(path: Path) -> EmbeddingConfig:
    """Load Embedding configuration"""
    return ConfigLoader(path).load_embedding_config()


def load_agent_config(path: Path) -> AgentConfig:
    """Load Agent configuration"""
    return ConfigLoader(path).load_agent_config()


def load_checkpoint_config(path: Path) -> CheckpointConfig:
    """Load Checkpoint configuration"""
    return ConfigLoader(path).load_checkpoint_config()


def load_env_config(path: Path) -> EnvConfig:
    """Load environment configuration"""
    return ConfigLoader(path).load_env_config()


def load_mem_agent_config(path: Path) -> MemAgentConfig:
    """Load memory agent configuration"""
    return ConfigLoader(path).load_mem_agent_config()

def load_judge_llm_config(path: Path) -> Optional[LLMConfig]:
    """Load judge LLM configuration"""
    return ConfigLoader(path).load_judge_llm_config()