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

    def __init__(self, goal_instruction, test_language: str = "ch"):
        self.goal_instruction = goal_instruction
        self.test_language = test_language  # ch or en

    # 总的提示
    def build_system_prompt(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建Type Help游戏的系统提示词

        Args:
            game_context: 游戏上下文，包含已解锁文件列表

        Returns:
            系统提示词
        """
        if self.test_language == "en":
            return self._build_system_prompt_en(game_context)
        else:
            return self._build_system_prompt_ch(game_context)

    def _build_system_prompt_ch(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建中文系统提示词"""
        base_prompt = """你是一名游戏型智能体，专门擅长解谜类游戏。本游戏要求你通过输入**文件名**来获取信息，而这些文件名遵循某种特定规则。你需要根据已有信息合理猜测文件命名规则，并**推断可能正确的文件名**以解锁更多文件，并利用从文件中获得的信息**重构整个故事**。

        策略：
        1. 优先打开已经解锁但尚未查看过的文件可以帮助你获取更多信息。
        2. 文件名中包含问号的部分是需要你来进行猜测的
        2. 对于已经失败的文件，**不要再次尝试同一个已经失败的文件名**，要尝试其他的组合方式。
        3. 在猜测文件名时，仔细分析已解锁文件中的命名模式，例如数字的含义，数字大小的排序关系，字母的含义等等
        4. 对于角色的编号，不要猜测文本信息中未出现的数字过大的编号
        """

        # 添加已解锁文件列表到系统提示
        if game_context:
            file_tracker_info = game_context.get("file_tracker_info")
            if file_tracker_info:
                unlocked = file_tracker_info.get("unlocked_files", [])
                if unlocked:
                    base_prompt += f"\n\n**目前已解锁文件列表**（你可以通过输入这些文件名来查看内容）:\n"
                    base_prompt += ", ".join([f'"{f}"' for f in unlocked])

        return base_prompt

    def _build_system_prompt_en(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建英文系统提示词"""
        base_prompt = """You are a game agent specialized in puzzle-solving games. In this game, you need to obtain information by entering **filenames** that follow certain specific rules. You need to reasonably infer the file naming rules based on available information, **deduce possible correct filenames** to unlock more files, and use the information obtained from files to **reconstruct the entire story**.

        Strategy:
        1. Prioritize opening unlocked but not yet viewed files to help you gain more information.
        2. Question marks in filenames indicate parts you need to guess.
        3. For failed files, **do not try the same failed filename again**; try other combinations.
        4. When guessing filenames, carefully analyze the naming patterns in unlocked files, such as the meaning of numbers, the ordering relationship of number sizes, the meaning of letters, etc.
        5. For character numbers, do not guess numbers that are too large and haven't appeared in the text information.
        """

        # Add unlocked file list to system prompt
        if game_context:
            file_tracker_info = game_context.get("file_tracker_info")
            if file_tracker_info:
                unlocked = file_tracker_info.get("unlocked_files", [])
                if unlocked:
                    base_prompt += f"\n\n**Currently Unlocked Files** (you can view their content by entering these filenames):\n"
                    base_prompt += ", ".join([f'"{f}"' for f in unlocked])

        return base_prompt

    def build_retrieval_prompt(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建文件检索提示词（Type Help 游戏专用）

        让 LLM 决定要打开哪些已读过的文件来获取信息

        Args:
            obs: 当前观察
            game_context: 游戏上下文

        Returns:
            文件检索提示词字符串
        """
        if self.test_language == "en":
            return self._build_retrieval_prompt_en(obs, game_context)
        else:
            return self._build_retrieval_prompt_ch(obs, game_context)

    def _build_retrieval_prompt_ch(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建中文文件检索提示词"""
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
            禁止选择当前记忆中已经存在的文件。

            输出格式 (严格按照以下json格式输出):
            {{
            "need_retrieval": true/false,
            "filenames": ["文件名1", "文件名2", ...],
            "reason": "<简要说明为什么需要查看这些文件>"
            }}

            """.strip()

    def _build_retrieval_prompt_en(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建英文文件检索提示词"""
        # 获取已读文件列表
        read_files_text = ""
        conversation_history=""
        if game_context:
            read_files_text = game_context.get('read_files_text', '')
            conversation_history = game_context['conversation_history']

        return f"""
            Current node information: {obs.text}
            Historical memory: {conversation_history}

            {read_files_text}

            Task:
            Decide whether you need to review previously read files to help make a decision.
            If needed, list the filenames you want to review (select from the read files list, maximum 3 most relevant files).
            Do not select files that already exist in current memory.

            Output format (strictly follow this JSON format):
            {{
            "need_retrieval": true/false,
            "filenames": ["filename1", "filename2", ...],
            "reason": "<brief explanation of why you need to view these files>"
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
            retrieved_hits: 检索到的文件内容（字符串格式）
            game_context: 包含 conversation_history记忆, file_tracker_info, read_files_text 和 character_names_text 的游戏上下文

        Returns:
            完整的用户提示词
        """
        if self.test_language == "en":
            return self._build_user_prompt_en(obs, retrieved_hits, game_context)
        else:
            return self._build_user_prompt_ch(obs, retrieved_hits, game_context)

    def _build_user_prompt_ch(self, obs: Observation, retrieved_hits: List[str], game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建中文用户提示词"""
        # 格式化文件追踪信息
        file_info_str = self._format_file_tracker_info(game_context)

        if game_context and 'conversation_history' in game_context:
            conversation_history = game_context['conversation_history']

        # 获取角色名称信息
        character_names_text = ""
        if game_context and 'character_names_text' in game_context:
            character_names_text = game_context['character_names_text']

        # 构建完整prompt
        prompt = f"""
        当前节点信息:{obs.text}；
        历史记忆：{conversation_history}；
        """

        # 添加角色名称信息
        if character_names_text:
            prompt += f"""
        {character_names_text}；
        """

        if retrieved_hits:
            prompt += f"""
            检索到的相关文件内容：{retrieved_hits}；
            """
        # 如果有检索结果，添加到prompt中

        prompt += f"""
        文件操作记录：{file_info_str}；"""

        prompt += """
        输出格式 (严格按照以下json格式输出):
        {
        "choice_text": <文件名>,
        "reason": "<简单的原因，用于说明为什么这个选择有利于达成目标>"
        "recall": <文件名的列表，用于列出你认为和这个选择有关联的文件名称，如果没有则输出空列表>
        }
        """

        return prompt.strip()

    def _build_user_prompt_en(self, obs: Observation, retrieved_hits: List[str], game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建英文用户提示词"""
        # 格式化文件追踪信息
        file_info_str = self._format_file_tracker_info(game_context, is_english=True)

        if game_context and 'conversation_history' in game_context:
            conversation_history = game_context['conversation_history']

        # 获取角色名称信息
        character_names_text = ""
        if game_context and 'character_names_text' in game_context:
            character_names_text = self._translate_character_names_to_en(game_context['character_names_text'])

        # 构建完整prompt
        prompt = f"""
        Current node information: {obs.text};
        Historical memory: {conversation_history};
        """

        # 添加角色名称信息
        if character_names_text:
            prompt += f"""
        {character_names_text};
        """

        if retrieved_hits:
            prompt += f"""
            Retrieved relevant file content: {retrieved_hits};
            """

        prompt += f"""
        File operation records: {file_info_str};"""

        prompt += """
        Output format (strictly follow this JSON format):
        {
        "choice_text": <filename>,
        "reason": "<brief reason explaining why this choice helps achieve the goal>"
        "recall": <list of filenames that you think are related to this choice, or empty list if none>
        }
        """

        return prompt.strip()

    def _format_file_tracker_info(self, game_context: Optional[Dict[str, Any]], is_english: bool = False) -> str:
        """格式化文件追踪信息（仅包含已读和失败记录）

        Args:
            game_context: 包含 file_tracker_info 的字典
            is_english: 是否使用英文格式

        Returns:
            格式化后的文件追踪信息字符串（已读、失败）
        """
        if not game_context:
            return ""

        file_tracker_info = game_context.get("file_tracker_info")
        if not file_tracker_info:
            return ""

        file_info_str = ""
        failed = file_tracker_info.get("failed_files", [])
        readed = game_context.get('read_files_text', '')

        if failed:
            if is_english:
                file_info_str += f"\nFailed attempts (these files do not exist, **do not try failed filenames again**):\n"
            else:
                file_info_str += f"\n失败的尝试 (这些文件不存在，**不要再次尝试失败的文件名**):\n"
            file_info_str += ", ".join([f'"{f}"' for f in failed])

        if readed:
            file_info_str += f"\n{readed}"

        return file_info_str

    def _translate_character_names_to_en(self, character_names_text: str) -> str:
        """将角色名称信息翻译为英文标题

        Args:
            character_names_text: 中文角色名称信息

        Returns:
            英文格式的角色名称信息
        """
        if not character_names_text:
            return ""

        # 简单替换标题为英文
        result = character_names_text.replace("**角色编号与称呼对照表**：", "**Character Number and Name Reference**:")
        result = result.replace("编号", "Number ")
        return result

    def build_compression_prompt(self, conversations: List[str]) -> str:
        """构建记忆压缩的prompt

        Args:
            conversations: 需要压缩的对话列表

        Returns:
            压缩prompt
        """
        conversations_text = "\n\n".join(conversations)

        prompt = f"""将以下内容进行压缩总结，只保留关键信息（例如重要的文件名、文件主要内容、角色、事件顺序等等）。
                对话内容：{conversations_text}
                """

        return prompt
