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
    - 记忆管理（包括游戏特定的记忆保护）
    - 其他游戏特定的辅助功能
    """

    def __init__(self):
        self.memory_store = None

    def set_memory_store(self, store):
        """设置记忆存储引用

        Args:
            store: MemoryStore实例
        """
        self.memory_store = store

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
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """格式化日志数据

        Args:
            step: 当前步数
            obs: 当前观察
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

    def post_action_hook(self, obs: Observation, decision: Any, action_success: bool = True, step: int = 0) -> None:
        """动作执行后的钩子函数（可选实现）

        用于在动作执行后进行游戏特定的处理，例如：
        - 记录已访问的节点
        - 更新游戏状态
        - 收集统计信息
        - 处理动作失败的情况

        Args:
            obs: 当前观察
            decision: 执行的决策
            action_success: 动作是否执行成功（默认True）
            step: 当前步数
        """
        pass

    def retrieve_information(self, retrieval_decision: Dict[str, Any], config: Any) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """检索信息（游戏特定实现）

        Args:
            retrieval_decision: policy.decide_retrieval()返回的检索决策
                - 对于KB游戏: {"need_retrieval": bool, "query": str, "reason": str}
                - 对于Type Help游戏: {"need_retrieval": bool, "filenames": list, "reason": str}
            config: Agent配置对象

        Returns:
            (格式化的检索结果文本, 检索决策信息字典)
            如果不需要检索则返回 (None, retrieval_decision)
        """
        # 默认实现：不进行检索
        return None, retrieval_decision

    def manage_memory(self, max_context_tokens: int, config: Any) -> None:
        """管理记忆存储

        统一的记忆管理接口，子类可以重写以实现游戏特定的记忆保护逻辑

        Args:
            max_context_tokens: 最大上下文token数
            config: Agent配置对象
        """
        if not self.memory_store:
            return

        # 默认实现：简单的滑动窗口
        current_tokens = self.memory_store.get_total_tokens_estimate()
        if current_tokens > max_context_tokens:
            delete_count = (current_tokens - max_context_tokens) // 50 + 1
            deleted = self.memory_store.delete_oldest(delete_count)
            if config.verbose:
                print(f"[上下文管理] 删除了 {deleted} 条最早的记忆，当前token: {self.memory_store.get_total_tokens_estimate()}")

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

    def observation(self, obs: Observation) -> str:
        """简化观察信息

        Args:
            obs: 原始观察对象

        Returns:
            简化后的观察文本，默认返回原始text
        """
        # 默认实现：返回原始观察文本
        return obs.text
