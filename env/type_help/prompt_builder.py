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
        return """你是一名游戏型智能体，专门擅长解谜类游戏。本游戏要求你通过输入**文件名**来获取信息，而这些文件名遵循某种特定规则。你需要根据已有信息**推断正确的文件名**以解锁相应文件，并利用从文件中获得的信息**重构整个故事**。

            策略：
            1. 首先尝试打开**已经解锁但尚未查看过的文件**，以获取更多信息。
            2. 如果你已经尝试过某个文件名且**失败了**，**不要再次尝试同一个文件名**。
            3. 在猜测文件名时，仔细分析已解锁文件中的命名模式 
            """

    def build_retrieval_prompt(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建文件检索提示词（Type Help 游戏专用）

        让 LLM 决定要打开哪些已读过的文件来获取信息

        Args:
            obs: 当前观察
            game_context: 游戏上下文

        Returns:
            文件检索提示词字符串
        """
        # 获取已读文件列表
        read_files_text = ""
        conversation_history=""
        if game_context:
            read_files_text = game_context.get('read_files_text', '')
            conversation_history = game_context['conversation_history']
        
        return f"""
            当前节点信息:{obs.text}
            历史记忆：{conversation_history}

            {read_files_text}

            任务:
            决定是否需要查看之前已读过的文件来帮助做出决策。
            如果需要，请列出你想要查看的文件名（从已阅读的文件列表中选择，最多选择3个最相关的文件）。

            输出格式 (严格按照以下json格式输出):
            {{
            "need_retrieval": true/false,
            "filenames": ["文件名1", "文件名2", ...],
            "reason": "<简要说明为什么需要查看这些文件>"
            }}

            """.strip()

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
            retrieved_hits: 检索到的记忆（已废弃，现在从game_context中获取）
            game_context: 包含 conversation_history记忆, file_tracker_info 和 read_files_text 的游戏上下文

        Returns:
            完整的用户提示词
        """
        # 格式化文件追踪信息
        file_info_str = self._format_file_tracker_info(game_context)

        if game_context and 'conversation_history' in game_context:
            conversation_history = game_context['conversation_history']
        
        # 构建完整prompt
        prompt = f"""
        当前节点信息:{obs.text}
    
        历史记忆：{conversation_history}

        文件操作记录：
        {file_info_str}

        输出格式 (严格按照以下json格式输出):
        {{
        "choice_text": <文件名>,
        "reason": "<简单的原因，用于说明为什么这个选择有利于达成目标>"
        }}

        """.strip()

        return prompt

    def _format_file_tracker_info(self, game_context: Optional[Dict[str, Any]]) -> str:
        """格式化文件追踪信息

        Args:
            game_context: 包含 file_tracker_info 的字典

        Returns:
            格式化后的文件追踪信息字符串（包括已解锁、已读、失败）
        """
        if not game_context:
            return ""

        file_tracker_info = game_context.get("file_tracker_info")
        if not file_tracker_info:
            return ""

        file_info_str = ""
        unlocked = file_tracker_info.get("unlocked_files", [])
        failed = file_tracker_info.get("failed_files", [])
        readed = game_context.get('read_files_text', '')

        if unlocked:
            file_info_str += f"\n\n已解锁的文件 (你可以尝试打开阅读):\n"
            file_info_str += "\n".join([f"- {f}" for f in unlocked])

        if failed:
            recent_failed = failed[-10:]  # 只显示最近10次失败
            file_info_str += f"\n\n失败的尝试 (这些文件不存在，**不要再次尝试失败的文件名**):\n"
            file_info_str += "\n".join([f"- {f} 失败" for f in recent_failed])

        if readed:
            file_info_str += readed

        return file_info_str
