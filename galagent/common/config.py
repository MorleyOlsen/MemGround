# galagent/common/config.py
'''用于加载各个config'''
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
class AgentConfig:
    """Agent运行配置"""
    max_steps: int = 50
    retrieve_top_k: int = 3
    verbose: bool = True
    retriever_type: str = "vector"  # keyword 或 vector
    max_memory: int = 40  # 记忆滑动窗口大小


@dataclass
class EnvConfig:
    """环境配置"""
    game_type: str = "kb"  # kb, type_help, etc.
    scenes_path: str = "dataset/scenes.json"
    start_node_id: str = "start"


class ConfigLoader:
    """配置加载器，用于加载所有配置"""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path.resolve()}")

        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def load_llm_config(self) -> LLMConfig:
        """加载LLM配置"""
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
        """加载Embedding配置"""
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
        """加载Agent配置"""
        agent = self._data.get("agent", {})

        return AgentConfig(
            max_steps=int(agent.get("max_steps", 50)),
            retrieve_top_k=int(agent.get("retrieve_top_k", 3)),
            verbose=bool(agent.get("verbose", True)),
            retriever_type=str(agent.get("retriever_type", "vector")),
            max_memory=int(agent.get("max_memory", 40)),
        )

    def load_env_config(self) -> EnvConfig:
        """加载环境配置"""
        env = self._data.get("env", {})

        return EnvConfig(
            game_type=str(env.get("game_type", "kb")),
            scenes_path=str(env.get("scenes_path", "dataset/scenes.json")),
            start_node_id=str(env.get("start_node_id", "start")),
        )


def load_llm_config(path: Path) -> LLMConfig:
    """加载LLM配置（保持向后兼容）"""
    return ConfigLoader(path).load_llm_config()


def load_embedding_config(path: Path) -> EmbeddingConfig:
    """加载Embedding配置"""
    return ConfigLoader(path).load_embedding_config()


def load_agent_config(path: Path) -> AgentConfig:
    """加载Agent配置"""
    return ConfigLoader(path).load_agent_config()


def load_env_config(path: Path) -> EnvConfig:
    """加载环境配置"""
    return ConfigLoader(path).load_env_config()
