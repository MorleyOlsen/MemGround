# galagent/agent/policy.py
from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI

from galagent.common.schemas import Decision, Observation
from galagent.common.config import LLMConfig
from galagent.env.base_prompt_builder import BasePromptBuilder

class DummyPolicy:
    """
    MVP: always choose the first option.
    Replace with LLM policy later (keep same interface).
    """

    def decide(self, obs: Observation, retrieved_hits: list[str]) -> Decision:
        return Decision(
            choice_index=0,
            rationale=f"MVP策略：先选第0个选项推进；检索命中={len(retrieved_hits)}。",
        )

class LLMPolicy:
    """
    LLM-driven decision policy.
    It asks the model to output strict JSON: {"choice_index": int, "reason": str}

    使用游戏特定的PromptBuilder来构建提示词，实现游戏逻辑解耦
    """

    def __init__(self, config: LLMConfig, prompt_builder: BasePromptBuilder, memory_store=None, game_utils=None, agent_config=None):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=120.0)
        self.prompt_builder = prompt_builder
        self.game_utils = game_utils
        self.agent_config = agent_config

        # 将 memory_store 传递给 prompt_builder
        if memory_store:
            self.prompt_builder.set_memory_store(memory_store)

    def _call_llm(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """调用LLM接口的统一方法

        Args:
            messages: 消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}]
            model: 模型名称，默认使用config中的model
            temperature: 温度参数，默认使用config中的temperature

        Returns:
            LLM返回的文本内容
        """
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=model or self.config.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.config.temperature,
                )
                msg = resp.choices[0].message
                return msg.content or ""
            except Exception as e:
                print(f"[LLM] 调用失败 (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(5)
        print("[LLM] 连续失败3次，跳过本步骤")
        return ""

    def decide_retrieval(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """决定是否需要检索记忆，并生成检索目标

        Args:
            obs: 当前观察
            game_context: 游戏特定的上下文信息

        Returns:
            包含 need_retrieval, query/filenames, reason 的字典
            - 对于KB游戏: {"need_retrieval": bool, "query": str, "reason": str}
            - 对于Type Help游戏: {"need_retrieval": bool, "filenames": list, "reason": str}
        """
        # 使用prompt builder构建检索决策提示词
        user_prompt = self.prompt_builder.build_retrieval_prompt(obs, game_context)

        # 如果没有文件检索prompt，回退到通用检索决策
        if not user_prompt:
            user_prompt = self.prompt_builder.build_retrieval_decision_prompt(obs)

        messages = [
            {"role": "user", "content": user_prompt},
        ]

        # 在调用LLM前进行记忆管理
        if self.game_utils and self.agent_config:
            full_prompt = user_prompt
            self.game_utils.manage_memory(
                self.agent_config,
                full_prompt=full_prompt,
                llm_client=self.client,
                llm_config=self.config,
                prompt_builder=self.prompt_builder
            )

        text = self._call_llm(messages)
        
        # 提取检索决策信息（兼容两种格式）
        parsed = self._parse_json(text)
        need_retrieval = parsed.get("need_retrieval", False)
        reason = parsed.get("reason", "")

        # 构建返回结果
        result = {
            "need_retrieval": need_retrieval,
            "reason": reason
        }

        # 根据返回的字段判断游戏类型
        if "filenames" in parsed:
            # Type Help游戏：返回文件名列表
            result["filenames"] = parsed.get("filenames", [])
        else:
            # KB游戏：返回query
            result["query"] = parsed.get("query", obs.text)

        return result

    def decide(self, obs: Observation, retrieved_hits: List[str], game_context: Optional[Dict[str, Any]] = None) -> Decision:
        """做出决策

        Args:
            obs: 当前观察
            retrieved_hits: 检索到的记忆（已解析为字符串列表）
            game_context: 游戏特定的上下文信息（如file_tracker_info）

        Returns:
            决策对象
        """
        # 使用游戏特定的prompt builder构建提示词
        system_prompt = self.prompt_builder.build_system_prompt(game_context)
        user_prompt = self.prompt_builder.build_user_prompt(obs, retrieved_hits, game_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 在调用LLM前进行记忆管理
        if self.game_utils and self.agent_config:
            full_prompt = system_prompt + user_prompt
            self.game_utils.manage_memory(
                self.agent_config,
                full_prompt=full_prompt,
                llm_client=self.client,
                llm_config=self.config,
                prompt_builder=self.prompt_builder
            )

        text = self._call_llm(messages)
        print("text:", text)

        parsed = self._parse_json(text)


        # 检查是否为 Dust 游戏的动作格式
        if "action_type" in parsed:
            # Dust 游戏：返回动作类型和参数
            action_type = parsed.get("action_type", 0)
            action_params = parsed.get("action_params", {})
            rationale = parsed.get("rationale", "")

            # 确保 action_type 是整数
            if isinstance(action_type, str):
                try:
                    action_type = int(action_type)
                except ValueError:
                    action_type = 0

            # 将整个动作信息序列化为 choice_text
            choice_text = json.dumps({
                "action_params": action_params
            }, ensure_ascii=False)

            if not isinstance(rationale, str) or not rationale.strip():
                rationale = "No reason provided."

            return Decision(
                choice_index=action_type,
                rationale=rationale.strip(),
                choice_text=choice_text,
                recall=[]
            )
        # 检查是否为文字输入模式（eg: Type Help游戏）
        elif "choice_text" in parsed:
            # Type Help游戏：返回文件名
            filename = parsed.get("choice_text", "")
            reason = parsed.get("reason", "")
            recall = parsed.get("recall", [])

            if not isinstance(reason, str) or not reason.strip():
                reason = "No reason provided."

            # 确保 recall 是列表
            if not isinstance(recall, list):
                recall = []

            return Decision(
                choice_index=0,  # Type Help游戏固定为0
                rationale=reason.strip(),
                choice_text=filename,
                recall=recall
            )
        else:
            # 传统选择模式（KB游戏等）
            choice_index = parsed.get("choice_index", 0)
            reason = parsed.get("reason", "")

            # 校验 choice_index 是否有效
            valid_indices = {c.index for c in obs.choices}

            if choice_index not in valid_indices:
                choice_index = 0  # 回退到第0个选项

            if not isinstance(reason, str) or not reason.strip():
                reason = "No reason provided."

            return Decision(
                choice_index=choice_index,
                rationale=reason.strip(),
                choice_text=""
            )

    def _parse_json(self, s: str) -> Optional[Dict[str, Any]]:
        """解析LLM返回的JSON字符串

        Args:
            s: LLM返回的字符串

        Returns:
            解析后的字典，如果解析失败返回空字典
        """
        s = (s or "").strip()
        if not s:
            return {}
        # 常见情况：模型输出被 ```json 包裹
        if s.startswith("```"):
            s = s.strip("`")
            # 可能包含 json\n{...}
            s = s[s.find("{"):] if "{" in s else s
        # 截取第一个 {...} 块
        if "{" in s and "}" in s:
            s2 = s[s.find("{"): s.rfind("}") + 1]
        else:
            s2 = s
        try:
            return json.loads(s2)
        except Exception:
            return {}

    def generate_story_summary(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """生成故事情节总结和推理

        Args:
            game_context: 游戏特定的上下文信息

        Returns:
            故事总结文本
        """
        is_english = getattr(self.prompt_builder, 'test_language', 'ch') == 'en'

        if is_english:
            system_prompt = """You are a professional story analyst. Based on all the events, dialogues, and clues you experienced in the game, complete the following tasks:
        1. Story Summary: Summarize the main plot of the entire story in 2-3 paragraphs
        2. Character Analysis: Analyze the motivations and relationships of the main characters
        3. Reasoning Conclusion: Based on all the information, deduce the truth or core secret of the story

        Please present your analysis in a clear and organized manner."""
        else:
            system_prompt = """你是一个专业的故事分析师。请根据你在游戏中经历的所有事件、对话和线索，完成以下任务：
        1. 故事梗概：用2-3段话总结整个故事的主要情节
        2. 角色分析：分析主要角色的动机和关系
        3. 推理结论：基于所有信息，推理出故事的真相或核心秘密

        请以清晰、有条理的方式输出你的分析。"""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if game_context and 'conversation_history' in game_context:
            conversation_history = game_context['conversation_history']
            if is_english:
                user_prompt = f"The following are all dialogues and event records from the game:\n\n{conversation_history}\n\nPlease complete the story analysis based on the above information."
            else:
                user_prompt = f"以下是游戏中的所有对话和事件记录：\n\n{conversation_history}\n\n请基于以上信息，完成故事分析。"
            messages.append({"role": "user", "content": user_prompt})

        summary = self._call_llm(messages, temperature=0.7)

        return summary
