# galagent/agent/llm_policy.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import yaml
from openai import OpenAI

from galagent.common.schemas import Decision, Observation


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_output_tokens: int
    goal_instruction: str


def load_llm_config(path: Path) -> LLMConfig:
    if not path.exists():
        raise FileNotFoundError(f"LLM config not found: {path.resolve()}")

    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    
    llm = data["llm"]
    required = ["provider", "api_key", "base_url", "model", "temperature", "max_output_tokens", "goal_instruction"]
    for k in required:
        if k not in llm:
            raise ValueError(f"Missing llm config field: {k}")

    return LLMConfig(
        provider=str(llm["provider"]),
        api_key=str(llm["api_key"]),
        base_url=str(llm["base_url"]),
        model=str(llm["model"]),
        temperature=float(llm["temperature"]),
        max_output_tokens=int(llm["max_output_tokens"]),
        goal_instruction=str(llm["goal_instruction"]),
    )


class LLMPolicy:
    """
    LLM-driven decision policy.
    It asks the model to output strict JSON: {"choice_index": int, "reason": str}
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def _build_prompt(self, obs: Observation, retrieved_hits: List[str]) -> str:
        # 将 choices 格式化
        choices_lines = []
        for c in obs.choices:
            choices_lines.append(f"{c.index}: {c.text}")
        choices_str = "\n".join(choices_lines) if choices_lines else "(no choices)"

        # 角色信息可增强一致性
        chars = obs.memory.characters or []
        if chars:
            chars_str = "\n".join(
                [f"- {x.name} | role={x.role} | desc={x.description}" for x in chars]
            )
        else:
            chars_str = "(none)"

        retrieved_str = "\n".join([f"- {t}" for t in retrieved_hits]) if retrieved_hits else "(none)"

        # 这里用“单轮输入文本”即可：Responses API 支持 instructions + input
        prompt = f"""
                GOAL:
                {self.config.goal_instruction}

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
                """.strip()
        return prompt

    def _parse_json(self, s: str) -> Optional[Dict[str, Any]]:
        s = (s or "").strip()
        if not s:
            return None
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
            return None

    def decide(self, obs: Observation, retrieved_hits: List[str]) -> Decision:
        # 若没有选项，仍然让模型决定没意义；这种情况一般说明数据问题（非 ending 却 links 为空）
        # 如果换成从网站上读取数据，要记得把agent改成有选项时再调用选择策略
        # if not obs.choices:
        #     return Decision(
        #         choice_index=self.config.fallback_choice_index,
        #         rationale="No available choices; fallback to 0 (dataset likely missing links).",
        #     )

        prompt = self._build_prompt(obs, retrieved_hits)
        messages = [
            {"role": "system", "content": "You are a careful game-playing agent. Output strict JSON only."},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.chat.completions.create(
            model=self.config.model,              # deepseek-chat / deepseek-reasoner :contentReference[oaicite:9]{index=9}
            messages=messages,
            temperature=self.config.temperature,
        )
        msg = resp.choices[0].message
        text = msg.content or ""
        # 可选：如果你用 deepseek-reasoner，它会给 reasoning_content（可用于显示思维链）:contentReference[oaicite:10]{index=10}
        reasoning = getattr(msg, "reasoning_content", None)
        
        # Responses API：官方推荐的新方式 :contentReference[oaicite:1]{index=1}
        # resp = self.client.responses.create(
        #     model=self.config.model,
        #     instructions="You are a careful game-playing agent. Follow the user's GOAL. Output strict JSON only.",
        #     input=prompt,
        #     temperature=self.config.temperature,
        #     max_output_tokens=self.config.max_output_tokens,
        # )

        # text = getattr(resp, "output_text", "")  # openai-python 提供 output_text :contentReference[oaicite:2]{index=2}
        # parsed = self._parse_json(text)

        parsed = self._parse_json(text)
        choice_index = parsed.get("choice_index", 0)
        reason = parsed.get("reason", "")

        # 校验 choice_index 是否有效
        valid_indices = {c.index for c in obs.choices}
        
        if choice_index not in valid_indices:
            choice_index = 0  # 回退到第0个选项

        if not isinstance(reason, str) or not reason.strip():
            reason = "No reason provided."

        # 如果想把 deepseek-reasoner 的 CoT 也打印出来，可以把它拼到 rationale 里（注意会很长）
        # if reasoning:
        #     # 这里不强制输出，给一个短摘要，避免刷屏
        #     reason = f"{reason.strip()} (CoT_len={len(str(reasoning))})"

        return Decision(choice_index=choice_index, rationale=reason.strip())
