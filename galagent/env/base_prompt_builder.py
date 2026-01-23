# galagent/env/base_prompt_builder.py
"""游戏特定的Prompt构建器基类"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from galagent.common.schemas import Observation


class BasePromptBuilder(ABC):
    """Prompt构建器基类

    每个游戏可以实现自己的PromptBuilder来定制：
    - System prompt
    - 场景信息格式化
    - 游戏特定的上下文信息
    - 输出格式要求
    """

    @abstractmethod
    def build_system_prompt(self) -> str:
        """构建系统提示词

        Returns:
            系统提示词字符串
        """
        pass

    @abstractmethod
    def build_user_prompt(
        self,
        obs: Observation,
        retrieved_hits: List[str],
        game_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """构建用户提示词

        Args:
            obs: 当前观察
            retrieved_hits: 检索到的记忆
            game_context: 游戏特定的上下文信息

        Returns:
            用户提示词字符串
        """
        pass

    def format_choices(self, obs: Observation) -> str:
        """格式化选项列表（通用实现）

        Args:
            obs: 当前观察

        Returns:
            格式化后的选项字符串
        """
        choices_lines = []
        for c in obs.choices:
            choices_lines.append(f"{c.index}: {c.text}")
        return "\n".join(choices_lines) if choices_lines else "(no choices)"

    def format_characters(self, obs: Observation) -> str:
        """格式化角色信息（通用实现）

        Args:
            obs: 当前观察

        Returns:
            格式化后的角色信息字符串
        """
        chars = obs.memory.characters or []
        if chars:
            return "\n".join(
                [f"- {x.name} | role={x.role} | desc={x.description}" for x in chars]
            )
        return "(none)"

    def format_retrieved_memory(self, retrieved_hits: List[str]) -> str:
        """格式化检索到的记忆（通用实现）

        Args:
            retrieved_hits: 检索到的记忆列表

        Returns:
            格式化后的记忆字符串
        """
        return "\n".join([f"- {t}" for t in retrieved_hits]) if retrieved_hits else "(none)"

    def build_retrieval_decision_prompt(self, obs: Observation) -> str:
        """构建检索决策提示词（通用实现，子类可覆盖）

        Args:
            obs: 当前观察

        Returns:
            检索决策提示词字符串
        """
        return f"""
CURRENT SCENE:
{obs.text}

TASK:
Decide if you need to retrieve past memories to help make a decision for this scene.
If yes, write a concise search query to find relevant past information.

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "need_retrieval": true/false,
  "query": "<your search query if need_retrieval is true, otherwise empty string>",
  "reason": "<brief reason for your decision>"
}}

GUIDELINES:
- Set need_retrieval to true if past information would help understand the current situation
- The query should be concise and focus on key information you need
- Set need_retrieval to false if the current scene provides enough information
""".strip()