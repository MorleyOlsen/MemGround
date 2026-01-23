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

    def get_game_context(self, env: Any) -> Dict[str, Any]:
        """获取KB游戏的上下文信息

        Args:
            env: KB游戏环境实例

        Returns:
            空字典（KB游戏暂无特殊上下文）
        """
        return {}

    def format_log_data(
        self,
        step: int,
        obs: Observation,
        search_results: str,
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """格式化KB游戏的日志数据

        Args:
            step: 当前步数
            obs: 当前观察
            search_results: 检索结果JSON字符串
            decision: 决策对象
            game_context: 游戏上下文
            retrieval_decision: 检索决策信息

        Returns:
            完整的日志数据字典
        """
        import json

        # 解析检索到的记忆
        retrieved_memory = []
        if search_results:
            data = json.loads(search_results)
            retrieved_memory = [r["text"] for r in data.get("results", [])]

        # 格式化选项
        choices_for_log = [{"index": c.index, "text": c.text} for c in obs.choices]

        # 构建日志数据（KB游戏没有特殊字段）
        log_data = {
            "step": step,
            "node_id": obs.node_id,
            "node_name": obs.name,
            "scene_text": obs.text,
            "choices": choices_for_log,
            "retrieved_memory": retrieved_memory,
            "decision_index": decision.choice_index,
            "decision_rationale": decision.rationale,
            "unlocked_files": None,
            "attempted_files": None
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
