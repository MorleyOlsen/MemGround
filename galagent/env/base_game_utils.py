# galagent/env/base_game_utils.py
"""游戏特定工具的基类"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from galagent.common.schemas import Observation


class BaseGameUtils(ABC):
    """游戏特定工具的基类

    每个游戏可以实现自己的工具类来处理：
    - 游戏上下文信息的获取
    - 日志信息的格式化
    - 其他游戏特定的辅助功能
    """

    @abstractmethod
    def get_game_context(self, env: Any) -> Dict[str, Any]:
        """获取游戏特定的上下文信息

        Args:
            env: 游戏环境实例

        Returns:
            游戏上下文字典
        """
        pass

    @abstractmethod
    def format_log_data(
        self,
        step: int,
        obs: Observation,
        search_results: str,
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """格式化日志数据

        Args:
            step: 当前步数
            obs: 当前观察
            search_results: 检索结果JSON字符串
            decision: 决策对象
            game_context: 游戏上下文
            retrieval_decision: 检索决策信息（包含need_retrieval, query, reason）

        Returns:
            完整的日志数据字典，包含所有需要记录的字段
        """
        pass

    @abstractmethod
    def execute_action(self, env: Any, decision: Any) -> None:
        """执行游戏动作

        Args:
            env: 游戏环境实例
            decision: 决策对象
        """
        pass

    def get_console_log_info(
        self,
        obs: Observation,
        search_results: str,
        decision: Any
    ) -> Optional[str]:
        """获取控制台日志信息（可选实现）

        Args:
            obs: 当前观察
            search_results: 检索结果
            decision: 决策对象

        Returns:
            格式化的控制台日志字符串，如果不需要则返回None
        """
        return None
