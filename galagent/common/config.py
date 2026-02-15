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
    max_context_tokens: int = 4000  # 最大上下文token数
    enable_compression: bool = True  # 是否启用记忆压缩
    compression_threshold: int = 20  # 对话轮次超过此值时触发压缩
    compression_count: int = 10  # 每次压缩最早的n轮对话


@dataclass
class CheckpointConfig:
    """Checkpoint配置"""
    enabled: bool = False  # 是否启用checkpoint
    interval: int = 100  # 每隔多少步保存一次
    dir: str = "checkpoints"  # checkpoint保存目录
    resume_from: Optional[str] = None  # 从指定checkpoint恢复（文件路径）


@dataclass
class EnvConfig:
    """环境配置"""
    game_type: str = "kb"  # kb, type_help, etc.
    scenes_path: str = "dataset/scenes.json"
    start_node_id: str = "start"
    test_language: str = "ch"  # ch or en (Chinese or English prompts)
    enable_hint: bool = False  # 是否启用失败次数提示功能 (仅type_help)
    hint_failure_threshold: int = 15  # 触发提示的连续失败次数阈值 (仅type_help)


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
            max_context_tokens=int(agent.get("max_context_tokens", 4000)),
            enable_compression=bool(agent.get("enable_compression", True)),
            compression_threshold=int(agent.get("compression_threshold", 20)),
            compression_count=int(agent.get("compression_count", 10)),
        )

    def load_checkpoint_config(self) -> CheckpointConfig:
        """加载Checkpoint配置"""
        checkpoint = self._data.get("checkpoint", {})

        return CheckpointConfig(
            enabled=bool(checkpoint.get("enabled", False)),
            interval=int(checkpoint.get("interval", 100)),
            dir=str(checkpoint.get("dir", "checkpoints")),
            resume_from=checkpoint.get("resume_from"),  # 可以是None
        )

    def load_env_config(self) -> EnvConfig:
        """加载环境配置"""
        env = self._data.get("env", {})

        return EnvConfig(
            game_type=str(env.get("game_type", "kb")),
            scenes_path=str(env.get("scenes_path", "dataset/scenes.json")),
            start_node_id=str(env.get("start_node_id", "start")),
            test_language=str(env.get("test_language", "ch")),
            enable_hint=bool(env.get("enable_hint", False)),
            hint_failure_threshold=int(env.get("hint_failure_threshold", 15)),
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


def load_checkpoint_config(path: Path) -> CheckpointConfig:
    """加载Checkpoint配置"""
    return ConfigLoader(path).load_checkpoint_config()


def load_env_config(path: Path) -> EnvConfig:
    """加载环境配置"""
    return ConfigLoader(path).load_env_config()
