# env/type_help/utils/game_utils.py
"""Type Help游戏的工具类"""
from __future__ import annotations

from typing import Any, Dict, Optional

from galagent.common.schemas import Observation
from galagent.env.base_game_utils import BaseGameUtils


class TypeHelpGameUtils(BaseGameUtils):
    """Type Help游戏的工具类

    处理Type Help游戏特定的：
    - 文件追踪信息获取
    - 日志数据格式化
    """

    def get_game_context(self, env: Any) -> Dict[str, Any]:
        """获取Type Help游戏的上下文信息

        Args:
            env: Type Help游戏环境实例

        Returns:
            包含file_tracker_info的上下文字典
        """
        game_context = {}

        # 获取文件追踪信息
        if hasattr(env, 'get_file_tracker_info'):
            game_context['file_tracker_info'] = env.get_file_tracker_info()

        return game_context

    def format_log_data(
        self,
        step: int,
        obs: Observation,
        search_results: str,
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """格式化Type Help游戏的日志数据

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

        # 获取文件追踪信息
        file_tracker_info = game_context.get('file_tracker_info')
        unlocked_files = None
        attempted_files = None
        if file_tracker_info:
            unlocked_files = file_tracker_info.get("unlocked_files", [])
            attempted_files = file_tracker_info.get("attempted_files", [])

        # 构建日志数据
        log_data = {
            "step": step,
            "node_id": obs.node_id,
            "node_name": obs.name,
            "scene_text": obs.text,
            "choices": choices_for_log,
            "retrieved_memory": retrieved_memory,
            "decision_index": decision.choice_index,
            "decision_rationale": decision.rationale,
            "unlocked_files": unlocked_files,
            "attempted_files": attempted_files
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
        """执行Type Help游戏的动作

        Args:
            env: Type Help游戏环境实例
            decision: 决策对象
        """
        # Type Help游戏使用文件名选择
        if decision.choice_text:
            if hasattr(env, 'choose_by_filename'):
                env.choose_by_filename(decision.choice_text)
            else:
                raise RuntimeError("Environment does not support choose_by_filename")
        else:
            raise ValueError("Type Help game requires choice_text in decision")

    def get_console_log_info(
        self,
        obs: Observation,
        search_results: str,
        decision: Any
    ) -> Optional[str]:
        """获取Type Help游戏的控制台日志信息

        Args:
            obs: 当前观察
            search_results: 检索结果
            decision: 决策对象

        Returns:
            格式化的控制台日志字符串
        """
        import json

        lines = []
        lines.append(f"Node: {obs.node_id} ({obs.name})")
        lines.append(f"Text: {obs.text[:100]}...")

        # 显示LLM的决策
        lines.append(f"Decision: {decision.choice_text if hasattr(decision, 'choice_text') else 'N/A'}")
        lines.append(f"Rationale: {decision.rationale}")

        # 显示检索结果
        if search_results:
            data = json.loads(search_results)
            results = data.get("results", [])
            if results:
                lines.append(f"Retrieved Memory ({len(results)} items):")
                for r in results[:3]:  # 只显示前3条
                    lines.append(f"  - {r['text'][:80]}...")

        return "\n".join(lines)