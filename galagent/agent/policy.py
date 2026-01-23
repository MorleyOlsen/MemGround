# galagent/agent/policy.py
from __future__ import annotations
import json
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
    # TODO:这里要改一下 没有choice_index，每一步要自己决定
    LLM-driven decision policy.
    It asks the model to output strict JSON: {"choice_index": int, "reason": str}

    使用游戏特定的PromptBuilder来构建提示词，实现游戏逻辑解耦
    """

    def __init__(self, config: LLMConfig, prompt_builder: BasePromptBuilder):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.prompt_builder = prompt_builder

    def decide_retrieval(self, obs: Observation) -> Dict[str, Any]:
        """决定是否需要检索记忆，并生成检索query

        Args:
            obs: 当前观察

        Returns:
            包含 need_retrieval, query, reason 的字典
        """
        # 使用prompt builder构建检索决策提示词
        user_prompt = self.prompt_builder.build_retrieval_decision_prompt(obs)

        messages = [
            {"role": "system", "content": "You are a careful game-playing agent. Decide if you need to retrieve past memories."},
            {"role": "user", "content": user_prompt},
        ]

        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )

        msg = resp.choices[0].message
        text = msg.content or ""

        print("Retrieval decision:", text)

        parsed = self._parse_json(text)

        # 提取检索决策信息
        need_retrieval = parsed.get("need_retrieval", True)  # 默认为True以保持向后兼容
        query = parsed.get("query", obs.text)  # 默认使用场景文本
        reason = parsed.get("reason", "")

        return {
            "need_retrieval": need_retrieval,
            "query": query,
            "reason": reason
        }

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
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(obs, retrieved_hits, game_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        
        msg = resp.choices[0].message
        text = msg.content or ""
        
        print("text:",text)

        parsed = self._parse_json(text)
        
        
        # 检查是否为文字输入模式（eg: Type Help游戏）
        if "choice_text" in parsed:
            # Type Help游戏：返回文件名
            filename = parsed.get("choice_text", "")
            reason = parsed.get("reason", "")

            if not isinstance(reason, str) or not reason.strip():
                reason = "No reason provided."

            return Decision(
                choice_index=0,  # Type Help游戏固定为0
                rationale=reason.strip(),
                choice_text=filename
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
