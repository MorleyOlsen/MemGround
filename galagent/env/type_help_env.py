# galagent/env/type_help_env.py
"""Type Help 解谜游戏环境"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from galagent.common.schemas import Observation, Choice, Memory, Character
from galagent.env.base_env import BaseGameEnv, GameConfig


@dataclass
class TypeHelpConfig(GameConfig):
    """Type Help 游戏配置"""
    data_path: Path = Path("env/type_help")
    game_type: str = "type_help"
    start_node_id: str = "Start"


class FileTracker:
    """文件解锁追踪器"""

    def __init__(self):
        self.unlocked_files: Set[str] = set()  # 已解锁的文件名
        self.attempted_files: List[str] = []  # 尝试打开的文件名历史（所有尝试）
        self.success_files: List[str] = []  # 成功打开的文件
        self.failed_files: List[str] = []  # 失败的尝试（文件不存在）
        self.file_naming_patterns: List[str] = []  # 发现的命名规则

    def unlock_file(self, filename: str) -> bool:
        """解锁文件"""
        if filename not in self.unlocked_files:
            self.unlocked_files.add(filename)
            return True
        return False

    def attempt_file(self, filename: str, success: bool = True) -> None:
        """记录尝试打开文件

        Args:
            filename: 文件名
            success: 是否成功打开（True=成功，False=失败）
        """
        self.attempted_files.append(filename)
        if success:
            self.success_files.append(filename)
        else:
            self.failed_files.append(filename)

    # def add_pattern(self, pattern: str) -> None:
    #     """添加发现的命名规则"""
    #     if pattern not in self.file_naming_patterns:
    #         self.file_naming_patterns.append(pattern)

    def get_unlocked_files(self) -> List[str]:
        """获取已解锁文件列表"""
        return sorted(list(self.unlocked_files))

    def get_attempted_files(self) -> List[str]:
        """获取尝试历史（所有尝试）"""
        return self.attempted_files.copy()

    def get_success_files(self) -> List[str]:
        """获取成功打开的文件"""
        return self.success_files.copy()

    def get_failed_files(self) -> List[str]:
        """获取失败的尝试（文件不存在）"""
        return self.failed_files.copy()

    def is_unlocked(self, filename: str) -> bool:
        """检查文件是否已解锁"""
        return filename in self.unlocked_files


class TypeHelpEnv(BaseGameEnv):
    """Type Help 游戏环境

    这是一个解谜游戏，玩家需要通过输入文件名来探索故事。
    文件名遵循特定的命名规则，玩家需要推理出规则来解锁新文件。
    """

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
        node_name = node.get("name", "")  # 当前节点的文件名称
        memory_data = node.get("memory", {})

        # 只有一个选择：让LLM输入文件名
        choices = [Choice(index=0, text="输入文件名")]

        # 构建记忆对象
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

        # 构建观察文本text:agent可以看到文件名+关键信息+事件发生的地点+人物名称与编号+已解锁的文件
        # 后续可以再改写成提供更进一步的拆分信息，观察模型游戏进度是否有变化
        
        text = f"当前文件是：{node_name}\n"
        
        # 添加关键信息
        key_info = memory_data.get("key_info", [])
        if key_info:
            key_info_text = "\n[关键信息]:\n" + "\n".join([f"- {info}" for info in key_info])
            text += key_info_text

        # 添加地点信息
        location=memory_data.get("location", "")
        if location:
            text+=f"\n该文件内的事情发生在{location}"

        # 添加人物信息
        if memory.characters:
            characters_info = "\n[出现的人物]:\n" + "\n".join([f"- {char.name}，编号为{char.number}" for char in memory.characters])
            text += characters_info

        # 添加已解锁文件信息
        # unlocked_files = self.file_tracker.get_unlocked_files()
        # if unlocked_files:
        #     files_info = "\n[已解锁的文件]: " + ", ".join(unlocked_files)
        #     text += files_info

        # 检查是否为结局节点,目前设置当开启到00-final-note文件时作为结束
        is_ending = node_name == "00-final-note"

        return Observation(
            node_id=self.current_node_id,
            name=node.get("name", ""),
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

    def try_open_file(self, filename: str) -> bool:
        """尝试打开文件（供Agent调用）

        Returns:
            True if file exists and was unlocked, False otherwise
        """
        self.file_tracker.attempt_file(filename)

        if filename in self.nodes:
            self.file_tracker.unlock_file(filename)
            self.current_node_id = filename
            self.history.append(filename)
            return True
        return False

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

