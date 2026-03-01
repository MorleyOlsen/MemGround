# env/dust/prompt_builder.py
"""Dust 推理游戏的 Prompt 构建器"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from galagent.common.schemas import Observation
from galagent.env.base_prompt_builder import BasePromptBuilder
from env.dust.utils.scoring import build_trace_structure, describe_trace


def _format_judgements_history(order_judgements, lang="ch"):
    """将历史排序判定按角色分组格式化为可读文本"""
    if lang == "ch":
        result_map = {
            "correct": "顺序正确且连续（已得分）",
            "correct_not_consecutive": "顺序正确但不连续（未得分）",
            "incorrect": "顺序错误",
            "unknown": "无法判断，存在事件不属于该角色",
            "ignored": "已计分（跳过）"
        }
    else:
        result_map = {
            "correct": "correct and consecutive (scored)",
            "correct_not_consecutive": "correct order but not consecutive (not scored)",
            "incorrect": "wrong order",
            "unknown": "cannot judge, some events don't belong to this character",
            "ignored": "already scored (skipped)"
        }

    # 按角色分组，规范化角色名（去除首尾空白），并对相同 (earlier, later) 对只保留最新结果
    by_char = {}
    for j in order_judgements:
        char = j["character"].strip()
        by_char.setdefault(char, {})
        pair_key = (j["earlier"], j["later"])
        by_char[char][pair_key] = j["result"]  # 覆盖旧结果，保留最新

    lines = []
    for char, pair_map in by_char.items():
        lines.append(f"{char}:")
        for (earlier, later), result in pair_map.items():
            label = result_map.get(result, result)
            if earlier == later:
                lines.append(f"  {earlier}: {label}")
            else:
                lines.append(f"  {earlier} -> {later}: {label}")
        lines.append("")
    return "\n".join(lines)


class DustPromptBuilder(BasePromptBuilder):
    """Dust 推理游戏的 Prompt 构建器

    游戏特点：
    - 关键词发现与事件解锁
    - 事件阅读与信息收集
    - 人物视角下的事件排序
    - 锁机制（问答解锁、钥匙解锁）
    """

    def __init__(self, goal_instruction: str = "", test_language: str = "ch", show_order_judgements_history: bool = True):
        super().__init__(goal_instruction)
        self.test_language = test_language  # ch or en
        self.show_order_judgements_history = show_order_judgements_history
        if test_language == "en":
            self.goal_instruction = "Unlock all events and reconstruct the complete story with as few attempts as possible through reasoning and ordering"
        else:
            self.goal_instruction = "通过推理和排序，用尽可能少的次数解锁所有事件并重构完整故事"

    def build_system_prompt(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建 Dust 游戏的系统提示词

        Args:
            game_context: 游戏上下文，包含游戏状态信息

        Returns:
            系统提示词
        """
        if self.test_language == "en":
            return self._build_system_prompt_en(game_context)
        else:
            return self._build_system_prompt_ch(game_context)

    def _build_system_prompt_ch(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建中文系统提示词"""
        base_prompt = f"""你是一名推理游戏智能体，正在玩一个名为Dust的推理解谜游戏。

            游戏目标：{self.goal_instruction}

            游戏机制：
            1. 关键词发现：首次阅读事件文本时，会自动发现其中隐含的关键词，这些关键词会全部添加到你的关键词池中。
            2. 事件解锁：使用关键词可以解锁与该关键词关联的新事件。解锁后你会知道事件的名称，但需要主动阅读才能获得完整内容。
            3. 事件阅读：
               - 未阅读事件：从未阅读事件列表中选择事件进行首次阅读，阅读后获得完整内容并自动发现关键词
               - 已阅读事件：可以重新阅读已阅读事件列表中的任何事件以回顾内容
            4. 人物事件排序：每个事件涉及多个角色。你需要推断事件在各角色视角下的发生顺序。提交正确的排序可以获得积分。
            5. 计分与钥匙：每正确排序一对事件（某角色视角下的"较早-较晚"且连续发生的关系），获得 1 分，累积一定分数后自动获得钥匙，已经计分的事件对不会重复计分
            6. 锁机制：
            - 粉色锁 (pink) 和 紫色锁 (purple)：通过回答问题解锁
            - 黄色锁 (yellow)：消耗 1 把钥匙解锁

            策略建议（按优先级排序）：
            1. 优先使用关键词解锁新事件：当有可用关键词时，优先使用关键词解锁新事件，以扩展可探索的内容
            2. 优先阅读未阅读事件：阅读未阅读事件列表中的事件，从中提取关键词和角色信息
            3. 当无未阅读事件且无关键词时，尝试提交排序：
               - 排序策略：先确定每个事件属于哪个角色，确认事件归属正确后，再在该角色的所有事件中推断时间顺序
               - 即使信息不完整也可以尝试，正确的排序会获得分数和钥匙
            4. 当有钥匙时，解锁黄色锁事件：使用钥匙解锁重要的黄色锁事件
            5. 当有足够信息时，回答问题解锁粉色/紫色锁：根据已读事件推断答案
            6. 如需回顾细节，可以从已阅读事件列表中选择事件重新阅读

            注意事项：
            - 开头为"对话-"的节点（如"对话-1"、"对话-2"等）不包含任何重要推理信息，不参与人物事件的排序，无需反复阅读，也不要因为对话节点中的内容影响你对事件归属或顺序的判断
            - 提示：伊甸幼儿园角色下只有一个事件
            - 不要过早解锁大量锁定事件，优先探索无锁事件
            """

        # 添加当前游戏状态信息
        if game_context and 'dust_state' in game_context:
            state = game_context['dust_state']
            base_prompt += f"\n\n当前游戏状态："
            base_prompt += f"\n- 钥匙: {state.get('keys', 0)}"
            base_prompt += f"\n- 可使用的关键词 ({len(state.get('keyword_pool', []))}): {sorted(state.get('keyword_pool', []))}"
            base_prompt += f"\n- 未阅读事件 ({len(state.get('event_pool', []))}): {state.get('event_pool', [])}"
            base_prompt += f"\n- 已阅读事件 ({len(state.get('read_events', []))}): {sorted(state.get('read_events', []))}"

            # 锁定事件信息
            locked_events = state.get('locked_events', {})
            if any(locked_events.values()):
                base_prompt += f"\n- 锁定事件:"

                # 获取锁信息（用于显示问题）
                lock_info_list = game_context.get('lock_info', [])

                for lock_type in ['pink', 'purple', 'yellow']:
                    events = locked_events.get(lock_type, [])
                    if events:
                        if lock_type in ['pink', 'purple']:
                            # 对于粉色和紫色锁，显示事件名和对应的问题
                            base_prompt += f"\n  - {lock_type} (需要回答问题):"
                            for event_name in sorted(events):
                                # 查找该事件的锁信息
                                question = ""
                                for lock_info in lock_info_list:
                                    if lock_info.get('sub_name') == event_name or lock_info.get('name') == event_name:
                                        question = lock_info.get('question', '')
                                        break

                                if question:
                                    base_prompt += f"\n    * {event_name}: {question}"
                                else:
                                    base_prompt += f"\n    * {event_name}"
                        else:
                            # 黄色锁只显示事件名
                            base_prompt += f"\n  - {lock_type} (需要钥匙): {sorted(events)}"

            # 当前排序情况和判断结果
            character_orders = state.get('character_orders', {})
            order_gt = game_context.get('order_gt', [])

            if character_orders and order_gt:
                # 使用 trace 结构生成详细的排序分析
                trace = build_trace_structure(character_orders, order_gt)
                trace_description = describe_trace(trace)

                base_prompt += f"\n\n**当前已提交的排序分析**：\n{trace_description}"

            if self.show_order_judgements_history:
                order_judgements = state.get('order_judgements', [])
                if order_judgements:
                    base_prompt += f"\n\n**历史排序判定记录（所有提交）**：\n"
                    base_prompt += _format_judgements_history(order_judgements, lang="ch")

        print("system_prompt:",base_prompt)
        return base_prompt

    def _build_system_prompt_en(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建英文系统提示词"""
        base_prompt = f"""You are a reasoning game agent playing a puzzle game called Dust.

            Game Objective: {self.goal_instruction}

            Game Mechanics:
            1. Keyword Discovery: When reading event text for the first time, keywords hidden within will be automatically discovered and added to your keyword pool.
            2. Event Unlocking: Use keywords to unlock new events associated with that keyword. After unlocking, you'll know the event name but need to actively read it to get the full content.
            3. Event Reading:
               - Unread Events: Select events from the unread events list for first-time reading to get full content and discover keywords
               - Read Events: You can re-read any event from the read events list to review their content
            4. Character Event Ordering: Each event involves multiple characters. You need to infer the chronological order of events from each character's perspective. Submitting correct orderings earns points.
            5. Scoring and Keys: For each correctly ordered event pair (an "earlier-later" relationship from a character's perspective that are consecutive), you earn 1 point. Accumulating a certain score automatically gives you a key. Already scored event pairs won't be scored again.
            6. Lock Mechanism:
            - Pink and Purple locks: Unlock by answering questions
            - Yellow lock: Unlock by consuming 1 key

            Strategy Suggestions (in priority order):
            1. Prioritize using keywords to unlock new events: When keywords are available, use them to unlock new events and expand explorable content
            2. Prioritize reading unread events: Read events from the unread events list to extract keywords and character information
            3. When no unread events and no keywords, try submitting orderings:
               - Ordering strategy: First determine which character each event belongs to, then after confirming correct event attribution, infer the chronological order within that character's events
               - Even if incomplete information, you can try - correct orderings earn points and keys
            4. When you have keys, unlock yellow-locked events: Use keys to unlock important yellow-locked events
            5. When you have sufficient information, answer questions to unlock pink/purple locks: Infer answers based on read events
            6. You can select events from the read events list to re-read if you need to review details

            Important Notes:
            - Nodes starting with "对话-" (e.g. "对话-1", "对话-2") contain no important reasoning information, do not participate in character event ordering, do not need to be repeatedly read, and must not influence your judgment on event attribution or ordering
            - Hint: There is only one event under the 伊甸幼儿园 (Eden Kindergarten) character
            - Don't unlock too many locked events too early, prioritize exploring unlocked events
            """

        # 添加当前游戏状态信息
        if game_context and 'dust_state' in game_context:
            state = game_context['dust_state']
            base_prompt += f"\n\nCurrent Game State:"
            base_prompt += f"\n- Keys: {state.get('keys', 0)}"
            base_prompt += f"\n- Available Keywords ({len(state.get('keyword_pool', []))}): {sorted(state.get('keyword_pool', []))}"
            base_prompt += f"\n- Unread Events ({len(state.get('event_pool', []))}): {state.get('event_pool', [])}"
            base_prompt += f"\n- Read Events ({len(state.get('read_events', []))}): {sorted(state.get('read_events', []))}"

            # 锁定事件信息
            locked_events = state.get('locked_events', {})
            if any(locked_events.values()):
                base_prompt += f"\n- Locked Events:"

                # 获取锁信息（用于显示问题）
                lock_info_list = game_context.get('lock_info', [])

                for lock_type in ['pink', 'purple', 'yellow']:
                    events = locked_events.get(lock_type, [])
                    if events:
                        if lock_type in ['pink', 'purple']:
                            # 对于粉色和紫色锁，显示事件名和对应的问题
                            base_prompt += f"\n  - {lock_type} (requires answering question):"
                            for event_name in sorted(events):
                                # 查找该事件的锁信息
                                question = ""
                                for lock_info in lock_info_list:
                                    if lock_info.get('sub_name') == event_name or lock_info.get('name') == event_name:
                                        question = lock_info.get('question', '')
                                        break

                                if question:
                                    base_prompt += f"\n    * {event_name}: {question}"
                                else:
                                    base_prompt += f"\n    * {event_name}"
                        else:
                            # 黄色锁只显示事件名
                            base_prompt += f"\n  - {lock_type} (requires key): {sorted(events)}"

            # 当前排序情况和判断结果
            character_orders = state.get('character_orders', {})
            order_gt = game_context.get('order_gt', [])

            if character_orders and order_gt:
                # 使用 trace 结构生成详细的排序分析
                trace = build_trace_structure(character_orders, order_gt)
                trace_description = describe_trace(trace)

                base_prompt += f"\n\n**Current Submitted Ordering Analysis**:\n{trace_description}"

            if self.show_order_judgements_history:
                order_judgements = state.get('order_judgements', [])
                if order_judgements:
                    base_prompt += f"\n\n**Historical Ordering Judgements (all submissions)**:\n"
                    base_prompt += _format_judgements_history(order_judgements, lang="en")

        print("system_prompt:",base_prompt)
        return base_prompt

    def build_user_prompt(
        self,
        obs: Observation,
        retrieved_hits: List[str],
        game_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """构建 Dust 游戏的用户提示词

        Args:
            obs: 当前观察
            retrieved_hits: 检索到的记忆（暂时不用）
            game_context: 游戏上下文

        Returns:
            完整的用户提示词
        """
        if self.test_language == "en":
            return self._build_user_prompt_en(obs, retrieved_hits, game_context)
        else:
            return self._build_user_prompt_ch(obs, retrieved_hits, game_context)

    def _build_user_prompt_ch(self, obs: Observation, retrieved_hits: List[str], game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建中文用户提示词"""
        # 获取对话历史
        conversation_history = ""
        if game_context and 'conversation_history' in game_context:
            conversation_history = game_context['conversation_history']

        # 构建完整 prompt
        prompt = f"""
        历史记忆:
        {conversation_history if conversation_history else "（无）"}
        """

        # 如果有检索到的记忆，添加到 prompt
        if retrieved_hits:
            prompt += f"\n检索到的相关信息:\n"
            prompt += "\n".join([f"{hit}" for hit in retrieved_hits])

        # 输出格式说明
        prompt += """

        请根据当前游戏状态选择下一步动作。输出格式 (严格按照以下 JSON 格式):
        {
        "action_type": <动作类型编号: 0-4 的整数>,
        "action_params": {
            // 根据动作类型填写不同参数：
            // 0 (unlock_keyword 使用关键词解锁事件): {"keyword": "关键词"}
            // 1 (read_event 阅读事件): {"event_name": "事件名"}
            // 2 (submit_orders 提交人物事件排序): {"orders": {"角色1": ["事件1", "事件2", ...], "角色2": [...]}}
            // 3 (unlock_with_key 用钥匙解锁黄色锁): {"event_name": "事件名"}
            // 4 (answer_lock 回答问题解锁粉色/紫色锁): {"event_name": "事件名", "answer": "答案"}
        },
        "rationale": "<详细说明你的推理过程和选择该动作的原因>"
        }

        注意事项：
        - submit_orders 的 orders 参数格式：每个角色对应一个事件列表，列表中的事件按该角色视角下的时间顺序排列（从早到晚）
        - 只提交你有充分信息可以确定顺序的事件对，不确定的不要提交
        - answer_lock 的答案需要根据已读事件中的信息推断
        """

        return prompt.strip()

    def _build_user_prompt_en(self, obs: Observation, retrieved_hits: List[str], game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建英文用户提示词"""
        # 获取对话历史
        conversation_history = ""
        if game_context and 'conversation_history' in game_context:
            conversation_history = game_context['conversation_history']

        # 构建完整 prompt
        prompt = f"""
        Historical Memory:
        {conversation_history if conversation_history else "(None)"}
        """

        # 如果有检索到的记忆，添加到 prompt
        if retrieved_hits:
            prompt += f"\nRetrieved Relevant Information:\n"
            prompt += "\n".join([f"{hit}" for hit in retrieved_hits])

        # 输出格式说明
        prompt += """

        Please select the next action based on the current game state. Output format (strictly follow this JSON format):
        {
        "action_type": <action type number: integer from 0-4>,
        "action_params": {
            // Fill in different parameters based on action type:
            // 0 (unlock_keyword - use keyword to unlock events): {"keyword": "keyword"}
            // 1 (read_event - read an event): {"event_name": "event name"}
            // 2 (submit_orders - submit character event ordering): {"orders": {"character1": ["event1", "event2", ...], "character2": [...]}}
            // 3 (unlock_with_key - unlock yellow lock with key): {"event_name": "event name"}
            // 4 (answer_lock - answer question to unlock pink/purple lock): {"event_name": "event name", "answer": "answer"}
        },
        "rationale": "<detailed explanation of your reasoning process and why you chose this action>"
        }

        Notes:
        - submit_orders orders parameter format: Each character corresponds to an event list, events in the list are arranged in chronological order from that character's perspective (earliest to latest)
        - Only submit event pairs where you have sufficient information to determine the order; don't submit uncertain ones
        - answer_lock answers need to be inferred from information in the read events
        """

        return prompt.strip()

    def build_retrieval_prompt(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建检索提示词，让模型决定是否需要查看已读事件的详细内容

        Args:
            obs: 当前观察
            game_context: 游戏上下文

        Returns:
            检索提示词
        """
        if self.test_language == "en":
            return self._build_retrieval_prompt_en(obs, game_context)
        else:
            return self._build_retrieval_prompt_ch(obs, game_context)

    def _build_retrieval_prompt_ch(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建中文检索提示词"""
        # 获取已读事件列表
        read_events = []
        conversation_history=""
        if game_context and 'dust_state' in game_context:
            read_events = game_context['dust_state'].get('read_events', [])
            conversation_history = game_context['conversation_history']

        if not read_events:
            return ""

        prompt = f"""
        历史记忆：{conversation_history}
        你已经阅读过以下事件: {sorted(read_events)}

        如果你需要回顾某些事件的详细内容来帮助推理，可以请求检索这些事件。

        输出格式 (严格按照以下 JSON 格式输出):
        {{
        "need_retrieval": true/false,
        "filenames": ["事件名1", "事件名2", ...],
        "reason": "<简要说明为什么需要查看这些事件>"
        }}

        注意：
        - 如果不需要检索，设置 need_retrieval 为 false，filenames 为空列表
        - filenames 中的事件名必须来自已读事件列表，已经存在在记忆中的事件不要再次检索
        - 尽可能减少检索内容，只检索对当前推理有帮助的事件，不要检索所有事件
        """

        return prompt.strip()

    def _build_retrieval_prompt_en(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        """构建英文检索提示词"""
        # 获取已读事件列表
        read_events = []
        conversation_history=""
        if game_context and 'dust_state' in game_context:
            read_events = game_context['dust_state'].get('read_events', [])
            conversation_history = game_context['conversation_history']

        if not read_events:
            return ""

        prompt = f"""
        Historical Memory: {conversation_history}
        You have already read the following events: {sorted(read_events)}

        If you need to review the detailed content of certain events to help with reasoning, you can request to retrieve these events.

        Output format (strictly follow this JSON format):
        {{
        "need_retrieval": true/false,
        "filenames": ["event name1", "event name2", ...],
        "reason": "<brief explanation of why you need to view these events>"
        }}

        Notes:
        - If retrieval is not needed, set need_retrieval to false and filenames to an empty list
        - Event names in filenames must come from the read events list, don't retrieve events that already exist in memory
        - Minimize retrieval content, only retrieve events helpful for current reasoning, don't retrieve all events
        """

        return prompt.strip()

    def build_compression_prompt(self, conversations: List[str]) -> str:
        """构建记忆压缩的prompt

        Args:
            conversations: 需要压缩的对话列表

        Returns:
            压缩prompt
        """
        conversations_text = "\n\n".join(conversations)

        prompt = f"""
                你是一名推理游戏智能体，正在玩一个名为Dust的推理解谜游戏。
                游戏机制：
                1. 关键词发现：阅读事件文本时，会自动发现其中隐含的关键词（tags），这些关键词会被添加到你的关键词池中。
                2. **事件解锁**：使用关键词可以解锁与该关键词关联的新事件。解锁后你会知道事件的名称，但需要主动阅读才能获得完整内容。
                3. **事件阅读**：从可阅读事件池中选择一个事件进行阅读，阅读后获得该事件的完整叙述、关键信息等。
                4. **人物事件排序**：每个事件涉及多个角色。你需要推断事件在各角色视角下的发生顺序。提交正确的排序可以获得积分。
                5. **计分与钥匙**：每正确排序一对事件（某角色视角下的"较早-较晚"关系），获得 1 分，累积一定分数后自动获得钥匙，已经计分的事件对不会重复计分
                6. **锁机制**：
                - **粉色锁 (pink)** 和 **紫色锁 (purple)**：通过回答问题解锁
                - **黄色锁 (yellow)**：消耗 1 把钥匙解锁
                将以下对话内容进行压缩总结，只保留你认为有助于推理人物与事件对应及顺序关系等关键信息。
                对话内容：{conversations_text}
                """

        return prompt
