# galagent/memory/base_mem_agent.py
"""记忆代理基类，定义统一接口"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseMemAgent(ABC):
    """记忆代理基类

    所有记忆代理（Mem0、A-mem等）都应该继承此类并实现这些方法
    """

    def __init__(self, game_name: str = "game_agent", verbose: bool = False):
        """初始化记忆代理

        Args:
            game_name: 游戏名称，用作 user_id 进行数据隔离
            verbose: 是否输出详细日志
        """
        self.game_name = game_name
        self.verbose = verbose

    @abstractmethod
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """添加记忆

        Args:
            content: 记忆内容
            metadata: 元数据（如 role, step, source 等）

        Returns:
            包含 success 和 memory_id 的字典
        """
        pass

    @abstractmethod
    def search_memories(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索记忆

        Args:
            query: 搜索查询
            top_k: 返回结果数量
            filters: 过滤条件，支持简化格式如 {"role": "user"} 或 {"step": "2"}

        Returns:
            记忆列表，每个记忆包含 id, text, score, metadata 等字段
        """
        pass

    @abstractmethod
    def get_all_memories(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """获取所有记忆

        Args:
            filters: 可选的过滤条件，支持简化格式

        Returns:
            记忆列表
        """
        pass

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """更新记忆

        Args:
            memory_id: 记忆ID
            content: 新的内容
            metadata: 新的元数据

        Returns:
            包含 success 的字典
        """
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """删除单个记忆

        Args:
            memory_id: 记忆ID

        Returns:
            包含 success 的字典
        """
        pass

    @abstractmethod
    def delete_all_memories(self) -> Dict[str, Any]:
        """删除该游戏的所有记忆

        Returns:
            包含 success 的字典
        """
        pass

    def get_memory_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的记忆历史（只有mem0实现了）

        Args:
            limit: 返回数量

        Returns:
            记忆列表（按时间倒序）
        """
        all_memories = self.get_all_memories()
        # 按创建时间倒序排序
        sorted_memories = sorted(
            all_memories,
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        return sorted_memories[:limit]

    def format_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """格式化记忆用于prompt（可选实现）

        Args:
            memories: 记忆列表

        Returns:
            格式化的文本
        """
        if not memories:
            return "No relevant memories found."

        lines = ["Relevant memories:"]
        for i, mem in enumerate(memories, 1):
            text = mem.get("text", "")
            metadata = mem.get("metadata", {})
            role = metadata.get("role", "unknown")
            lines.append(f"{i}. [{role}] {text}")

        return "\n".join(lines)

    def get_state(self) -> Dict[str, Any]:
        """获取当前状态（用于checkpoint）

        Returns:
            状态字典
        """
        return {
            "game_name": self.game_name,
            "verbose": self.verbose
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """恢复状态（从checkpoint）

        Args:
            state: 状态字典
        """
        self.game_name = state.get("game_name", self.game_name)
        self.verbose = state.get("verbose", self.verbose)

        if self.verbose:
            print(f"[MemAgent] 状态已恢复: game_name={self.game_name}")
