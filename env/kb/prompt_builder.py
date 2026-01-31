# env/kb/prompt_builder.py
"""KB游戏的Prompt构建器"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from galagent.common.schemas import Observation
from galagent.env.base_prompt_builder import BasePromptBuilder


class KBPromptBuilder(BasePromptBuilder):
    """KB游戏的Prompt构建器

    这是标准的视觉小说/Galgame类型游戏
    - 注重角色关系
    - 注重剧情连贯性
    - 需要记忆之前的选择
    """

    def __init__(self, goal_instruction: str = "Reach the best ending by making optimal choices."):
        self.goal_instruction = goal_instruction

    def build_system_prompt(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建KB游戏的系统提示词

        Args:
            game_context: 游戏特定的上下文信息（KB游戏暂不使用）

        Returns:
            系统提示词字符串
        """
        return "You are a careful game-playing agent. Output strict JSON only."

    def build_user_prompt(
        self,
        obs: Observation,
        retrieved_hits: List[str],
        game_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """构建KB游戏的用户提示词

        Args:
            obs: 当前观察
            retrieved_hits: 检索到的记忆
            game_context: 游戏特定的上下文信息（KB游戏暂不使用）

        Returns:
            完整的用户提示词
        """
        # 格式化基础信息
        choices_str = self.format_choices(obs)
        chars_str = self.format_characters(obs)
        retrieved_str = self.format_retrieved_memory(retrieved_hits)

        # 构建完整prompt
        prompt = f"""
        GOAL:
        {self.goal_instruction}

        CURRENT SCENE:
        text: {obs.text}

        CONTEXT (memory fields):
        location: {obs.memory.location}
        time: {obs.memory.time}
        key_info:
        {chr(10).join(['- '+k for k in (obs.memory.key_info or [])]) if (obs.memory.key_info or []) else '(none)'}
        characters:
        {chars_str}

        RETRIEVED PAST MEMORY (may help consistency):
        {retrieved_str}

        AVAILABLE CHOICES (choose one index):
        {choices_str}

        OUTPUT FORMAT (STRICT JSON ONLY, no extra text):
        {{
        "choice_index": <int>,
        "reason": "<brief reason>"
        }}

        RULES:
        - choice_index must be one of the available indices.
        - reason must be concise and explain why this choice is best for the goal.
        - If information is insufficient, choose the safest/most informative option.
        - Pay attention to character relationships and story consistency.
        """.strip()

        return prompt