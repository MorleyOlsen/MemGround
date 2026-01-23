# env/type_help/prompt_builder.py
"""Type Help游戏的Prompt构建器"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from galagent.common.schemas import Observation
from galagent.env.base_prompt_builder import BasePromptBuilder


class TypeHelpPromptBuilder(BasePromptBuilder):
    """Type Help解谜游戏的Prompt构建器

    这个游戏的特点：
    - 文件命名规则推理
    - 文件解锁追踪
    - 需要根据已解锁文件推断新文件名
    """

    def __init__(self, goal_instruction: str = "Reach the best ending by making optimal choices."):
        self.goal_instruction = goal_instruction

    # 总的提示
    def build_system_prompt(self) -> str:
        """构建Type Help游戏的系统提示词"""
        return """你是一名游戏型智能体，专门擅长解谜类游戏。本游戏要求你通过输入**文件名**来获取信息，而这些文件名遵循某种特定规则。你需要根据已有信息**推断正确的文件名**以解锁相应文件，并利用获得的信息**重构整个故事**。

            **策略：**
            1. 优先级最高：首先尝试打开**已经解锁但尚未查看过的文件**，以获取更多信息。
            2. 如果你已经尝试过某个文件名且**失败了**（会显示在 FAILED ATTEMPTS 中），**不要再次尝试同一个文件名**——要从失败中学习。
            3. 在猜测文件名时，仔细分析已解锁文件中的命名模式 
            """

    # 每一个节点的提示
    def build_user_prompt(
        self,
        obs: Observation,
        retrieved_hits: List[str],
        game_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """构建Type Help游戏的用户提示词

        Args:
            obs: 当前观察
            retrieved_hits: 检索到的记忆
            game_context: 包含 file_tracker_info 的游戏上下文

        Returns:
            完整的用户提示词
        """
        # 格式化基础信息
        choices_str = self.format_choices(obs)
        chars_str = self.format_characters(obs)
        # TODO:检查一下retrieved的内容，还要修改一下retrieve的顺序逻辑
        retrieved_str = self.format_retrieved_memory(retrieved_hits)

        # 格式化文件追踪信息
        file_info_str = self._format_file_tracker_info(game_context)

        # 构建完整prompt
        prompt = f"""
        当前节点信息:
        内容: {obs.text}

        过去的记忆:
        {retrieved_str}
        {file_info_str}

        输出格式 (严格按照以下json格式输出):
        {{
        "choice_text": <文件名>,
        "reason": "<简单的原因>"
        }}
        要求原因必须简洁，并说明为什么这个选择最有利于达成目标
        
        """.strip()

        return prompt

    def _format_file_tracker_info(self, game_context: Optional[Dict[str, Any]]) -> str:
        """格式化文件追踪信息

        Args:
            game_context: 包含 file_tracker_info 的字典

        Returns:
            格式化后的文件追踪信息字符串
        """
        if not game_context:
            return ""

        file_tracker_info = game_context.get("file_tracker_info")
        if not file_tracker_info:
            return ""

        file_info_str = ""
        unlocked = file_tracker_info.get("unlocked_files", [])
        attempted = file_tracker_info.get("attempted_files", [])
        patterns = file_tracker_info.get("patterns", [])

        if unlocked:
            file_info_str += f"\n\n已解锁的文件 (你可以尝试打开阅读):\n"
            file_info_str += "\n".join([f"- {f}" for f in unlocked])

        if attempted:
            recent_attempts = attempted[-5:]  # 只显示最近5次尝试
            file_info_str += f"\n\n失败的尝试 (这些文件不存在):\n"
            file_info_str += "\n".join([f"- {f} 失败" for f in recent_attempts])

        return file_info_str
