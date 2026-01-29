# env/kb/utils/game_utils.py
"""KB游戏的工具类"""
from __future__ import annotations

from typing import Any, Dict, Optional

from galagent.common.schemas import Observation
from galagent.env.base_game_utils import BaseGameUtils


class KBGameUtils(BaseGameUtils):
    """KB游戏的工具类

    处理KB游戏特定的：
    - 标准选择模式（choice_index）
    - 日志数据格式化
    """

    def __init__(self):
        super().__init__()

    def get_game_context(self, env: Any) -> Dict[str, Any]:
        """获取KB游戏的上下文信息

        Args:
            env: KB游戏环境实例

        Returns:
            包含记忆的上下文字典
        """
        game_context = {}

        # 获取当前记忆
        if self.memory_store:
            game_context.update(self.memory_store.get_memory_context())

        return game_context

    def format_log_data(
        self,
        step: int,
        obs: Observation,
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """格式化KB游戏的日志数据

        Args:
            step: 当前步数
            obs: 当前观察
            decision: 决策对象
            game_context: 游戏上下文
            retrieval_decision: 检索决策信息

        Returns:
            完整的日志数据字典
        """
        # 格式化choices为新格式：{"text": 选择文本, "decision_rationale": 原因}
        # KB游戏使用choice_index，需要从obs.choices中获取对应的文本
        choice_text = ""
        if hasattr(decision, 'choice_index'):
            for choice in obs.choices:
                if choice.index == decision.choice_index:
                    choice_text = choice.text
                    break

        choices = {
            "text": choice_text,
            "decision_rationale": decision.rationale
        }

        # 构建日志数据（KB游戏没有文件追踪字段）
        log_data = {
            "step": step,
            "node_id": obs.node_id,
            "node_name": obs.name,
            "scene_text": obs.text,
            "choices": choices,
            "file_retrieval": None,
            "unlocked_files": None,
            "attempted_files": None,
            "success_files": None,
            "failed_files": None
        }

        # 添加检索决策信息
        if retrieval_decision:
            log_data["retrieval_decision"] = {
                "need_retrieval": retrieval_decision.get("need_retrieval", True),
                "query": retrieval_decision.get("query", ""),
                "reason": retrieval_decision.get("reason", "")
            }

        return log_data

    def execute_action(self, env: Any, decision: Any) -> None:
        """执行KB游戏的动作

        Args:
            env: KB游戏环境实例
            decision: 决策对象
        """
        # KB游戏使用标准的choice_index选择
        env.choose(decision.choice_index)
