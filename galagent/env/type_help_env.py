# galagent/env/type_help_env.py
"""Type Help 解谜游戏环境"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path

from galagent.common.schemas import Observation, Choice, Memory, Character
from galagent.env.base_env import BaseGameEnv, GameConfig
from env.type_help.utils.file_tracker import FileTracker


@dataclass
class TypeHelpConfig(GameConfig):
    """Type Help 游戏配置"""
    data_path: Path = Path("dataset/type_help")
    game_type: str = "type_help"
    start_node_id: str = "Start"


class TypeHelpEnv(BaseGameEnv):
    """Type Help 游戏环境"""

    def __init__(self, config: TypeHelpConfig):
        super().__init__(config)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.file_tracker = FileTracker()
        self.load_game_data()

    def load_game_data(self) -> None:
        """加载游戏数据"""
        data_file = self.config.data_path / "nodes.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Game data not found: {data_file}")

        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 解析节点数据
        for node in data.get("game", {}).get("nodes", []):
            node_name = node.get("name")
            if node_name:
                self.nodes[node_name] = node

        # 初始化：解锁起始节点中提到的文件
        self._initialize_unlocked_files()

    def _initialize_unlocked_files(self) -> None:
        """初始化已解锁的文件列表"""
        # 自动解锁背景节点（非文件类型但用于获取故事背景）
        background_nodes = ["Background","message","00-readme"]
        for node_name in background_nodes:
            if node_name in self.nodes:
                self.file_tracker.unlock_file(node_name)

        # 从 message 节点获取初始文件列表
        # if "message" in self.nodes:
        #     memory = self.nodes["message"].get("memory", {})
        #     files = memory.get("files", [])
        #     for file in files:
        #         # 只解锁完整的文件名（不包含 ???? 的）
        #         if "?" not in file:
        #             self.file_tracker.unlock_file(file)

    # 只能观察到文本信息，不能观察到下一步跳转信息
    def observe(self) -> Observation:
        """获取当前观察"""
        if self.current_node_id not in self.nodes:
            raise ValueError(f"Node not found: {self.current_node_id}")

        node = self.nodes[self.current_node_id]
        node_name = node.get("name", "")

        # 只有一个选择：让LLM输入文件名
        choices = [Choice(index=0, text="输入文件名")]

        # 使用FileRetriever统一获取和格式化文件信息
        from env.type_help.utils.file_retriever import TypeHelpFileRetriever
        retriever = TypeHelpFileRetriever(self)

        # 获取文件信息（不包含links）
        file_results = retriever.retrieve_files([node_name])

        if not file_results or not file_results[0].get("exists", False):
            raise ValueError(f"Failed to retrieve node data: {node_name}")

        file_info = file_results[0]

        # 格式化为文本
        text = retriever.format_single_file(file_info).lstrip('\n')

        # 构建Memory对象（从file_info中提取）
        memory_data = node.get("memory", {})
        characters = []
        for char_data in memory_data.get("characters", []):
            characters.append(Character(
                name=char_data.get("name", ""),
                role=char_data.get("role", ""),
                number=char_data.get("number", 0),
                description=char_data.get("description", "")
            ))

        memory = Memory(
            description=memory_data.get("description", ""),
            key_info=memory_data.get("key_info", []),
            location=memory_data.get("location", ""),
            characters=characters
        )

        # 检查是否为结局节点
        is_ending = node_name == "00-final-note"

        return Observation(
            node_id=self.current_node_id,
            name=node_name,
            text=text,
            choices=choices,
            memory=memory,
            is_ending=is_ending
        )

    def choose(self, choice_index: int) -> None:
        """执行选择（保留用于兼容性，Type Help游戏不使用）"""
        raise NotImplementedError("Type Help game uses choose_by_filename instead")

    def choose_by_filename(self, filename: str) -> bool:
        """通过文件名进行选择（Type Help游戏专用）

        Args:
            filename: 要打开的文件名

        Returns:
            True if file exists and was opened, False otherwise
        """
        # 检查文件是否存在
        if filename not in self.nodes:
            # 记录尝试（失败）
            self.file_tracker.attempt_file(filename, success=False)
            print(f"[Type Help] File not found: {filename}")
            return False

        # 记录尝试（成功）
        self.file_tracker.attempt_file(filename, success=True)

        # 记录历史
        self.history.append(self.current_node_id)

        # 解锁并跳转到文件
        self.file_tracker.unlock_file(filename)
        self.current_node_id = filename

        # 解锁新节点中提到的文件
        self._unlock_files_in_node(filename)

        print(f"[Type Help] Successfully opened file: {filename}")
        return True

    def _unlock_files_in_node(self, node_id: str) -> None:
        """如果memory中files字段有内容，就解锁节点中提到的新文件"""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        memory = node.get("memory", {})
        files = memory.get("files", [])

        for file in files:
            if "?" not in file:
                self.file_tracker.unlock_file(file)

    def get_file_tracker_info(self) -> Dict[str, Any]:
        """获取文件追踪器信息"""
        return {
            "unlocked_files": self.file_tracker.get_unlocked_files(),
            "attempted_files": self.file_tracker.get_attempted_files(),
            "success_files": self.file_tracker.get_success_files(),
            "failed_files": self.file_tracker.get_failed_files(),
            "patterns": self.file_tracker.file_naming_patterns
        }

    def reset(self) -> None:
        """重置环境"""
        self.current_node_id = self.config.start_node_id
        self.history.clear()
        self.file_tracker = FileTracker()
        self._initialize_unlocked_files()

    def get_state(self) -> Dict[str, Any]:
        """获取环境状态用于checkpoint

        Returns:
            包含环境完整状态的字典
        """
        return {
            "current_node_id": self.current_node_id,
            "history": self.history.copy(),
            "file_tracker": {
                "unlocked_files": list(self.file_tracker.unlocked_files),
                "attempted_files": self.file_tracker.attempted_files.copy(),
                "success_files": self.file_tracker.success_files.copy(),
                "failed_files": self.file_tracker.failed_files.copy(),
                "file_naming_patterns": self.file_tracker.file_naming_patterns.copy()
            }
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """从checkpoint恢复环境状态

        Args:
            state: 环境状态字典
        """
        self.current_node_id = state["current_node_id"]
        self.history = state["history"].copy()

        # 恢复文件追踪器状态
        tracker_state = state["file_tracker"]
        self.file_tracker.unlocked_files = set(tracker_state["unlocked_files"])
        self.file_tracker.attempted_files = tracker_state["attempted_files"].copy()
        self.file_tracker.success_files = tracker_state["success_files"].copy()
        self.file_tracker.failed_files = tracker_state["failed_files"].copy()
        self.file_tracker.file_naming_patterns = tracker_state["file_naming_patterns"].copy()

        print(f"[Env] 已恢复状态: 当前节点={self.current_node_id}, "
              f"已解锁文件={len(self.file_tracker.unlocked_files)}个")

