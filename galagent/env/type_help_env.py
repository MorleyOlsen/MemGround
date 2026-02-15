# galagent/env/type_help_env.py
"""Type Help 解谜游戏环境"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List
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
    test_language: str = "ch"  # ch or en (Chinese or English prompts)
    enable_hint: bool = False  # 是否启用失败次数提示功能
    hint_failure_threshold: int = 15  # 触发提示的失败次数阈值


class TypeHelpEnv(BaseGameEnv):
    """Type Help 游戏环境"""

    def __init__(self, config: TypeHelpConfig):
        super().__init__(config)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.file_tracker = FileTracker()
        self.character_names = []  # 存储角色名称和编号信息
        self.node_links: Dict[str, List[str]] = {}  # 存储节点链接关系: {from: [target1, target2, ...]}
        self.consecutive_failures = 0  # 连续失败次数计数器
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

        # 加载角色名称数据
        self._load_character_names()

        # 加载节点链接关系
        self._load_node_links()

        # 初始化：解锁起始节点中提到的文件
        self._initialize_unlocked_files()

    def _load_character_names(self) -> None:
        """从name.json文件中加载角色名称和编号信息"""
        name_file = self.config.data_path / "name.json"
        if not name_file.exists():
            print(f"[Type Help] Warning: name.json not found at {name_file}")
            self.character_names = []
            return

        try:
            with open(name_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.character_names = data.get("name", [])
        except json.JSONDecodeError as e:
            print(f"[Type Help] Warning: Failed to parse name.json: {e}")
            self.character_names = []

    def _load_node_links(self) -> None:
        """从all_links_with_recall.json文件中加载节点链接关系"""
        links_file = self.config.data_path / "all_links_with_recall.json"
        if not links_file.exists():
            print(f"[Type Help] Warning: all_links_with_recall.json not found at {links_file}")
            self.node_links = {}
            return

        try:
            with open(links_file, 'r', encoding='utf-8') as f:
                # 文件是JSONL格式，每行一个JSON对象
                for line in f:
                    if line.strip():
                        link = json.loads(line)
                        from_node = link.get("from")
                        target_node = link.get("target")

                        if from_node and target_node:
                            if from_node not in self.node_links:
                                self.node_links[from_node] = []
                            self.node_links[from_node].append(target_node)
        except Exception as e:
            print(f"[Type Help] Warning: Failed to parse all_links_with_recall.json: {e}")
            self.node_links = {}

    def get_character_names_text(self) -> str:
        """获取格式化的角色名称信息文本

        Returns:
            格式化后的角色信息字符串
        """
        if not self.character_names:
            return ""

        lines = ["角色编号与称呼对照表："]
        for character in self.character_names:
            number = character.get("number")
            names = character.get("name", [])

            if number is not None:
                # 有编号的角色
                names_str = "、".join(names)
                lines.append(f"  编号{number}: {names_str}")
            else:
                # 无编号的角色（如K）
                names_str = "、".join(names)
                lines.append(f"  {names_str}")

        return "\n".join(lines)

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
            # 增加连续失败计数
            self.consecutive_failures += 1
            print(f"[Type Help] File not found: {filename} (连续失败: {self.consecutive_failures}次)")

            # 检查是否触发自动解锁提示
            self._auto_unlock_hint()

            return False

        # 记录尝试（成功）并添加到已读文件列表
        self.file_tracker.attempt_file(filename, success=True)

        # 重置连续失败计数（成功打开文件）
        self.consecutive_failures = 0

        # 解锁并跳转到文件
        self.file_tracker.unlock_file(filename)
        self.current_node_id = filename

        # 特判：如果打开了 "04-ST-1-5-8"，删除 "04-ST-?????"
        if filename == "04-ST-1-5-8":
            self.file_tracker.remove_unlocked_file("04-ST-?????")

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

    def _get_next_unlocked_target(self, current_node: str) -> str:
        """获取当前节点的下一个未解锁的目标节点

        Args:
            current_node: 当前节点名称

        Returns:
            下一个未解锁的目标节点名称，如果没有则返回空字符串
        """
        # 获取当前节点的所有目标节点
        targets = self.node_links.get(current_node, [])

        # 找到第一个未解锁的目标节点
        for target in targets:
            if not self.file_tracker.is_unlocked(target):
                return target

        return ""

    def _auto_unlock_hint(self) -> None:
        """自动解锁提示：当连续失败次数达到阈值时，解锁下一个目标节点"""
        if not self.config.enable_hint:
            return

        if self.consecutive_failures >= self.config.hint_failure_threshold:
            # 获取下一个未解锁的目标节点
            next_target = self._get_next_unlocked_target(self.current_node_id)

            if next_target:
                # 解锁该节点
                self.file_tracker.unlock_file(next_target)
                print(f"[Type Help Hint] 连续失败 {self.consecutive_failures} 次，自动解锁提示文件: {next_target}")
                # 重置连续失败计数器
                self.consecutive_failures = 0
            else:
                print(f"[Type Help Hint] 连续失败 {self.consecutive_failures} 次，但当前节点没有更多未解锁的目标节点")

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
        self.file_tracker = FileTracker()
        self.consecutive_failures = 0  # 重置连续失败计数器
        self._initialize_unlocked_files()

    def get_state(self) -> Dict[str, Any]:
        """获取环境状态用于checkpoint

        Returns:
            包含环境完整状态的字典
        """
        return {
            "current_node_id": self.current_node_id,
            "consecutive_failures": self.consecutive_failures,
            "file_tracker": {
                "unlocked_files": list(self.file_tracker.unlocked_files),
                "attempted_files": self.file_tracker.attempted_files.copy(),
                "success_files": self.file_tracker.success_files.copy(),
                "failed_files": self.file_tracker.failed_files.copy(),
                "file_naming_patterns": self.file_tracker.file_naming_patterns.copy(),
                "read_files": self.file_tracker.read_files.copy()
            }
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """从checkpoint恢复环境状态

        Args:
            state: 环境状态字典
        """
        self.current_node_id = state["current_node_id"]
        self.consecutive_failures = state.get("consecutive_failures", 0)

        # 恢复文件追踪器状态
        tracker_state = state["file_tracker"]
        self.file_tracker.unlocked_files = set(tracker_state["unlocked_files"])
        self.file_tracker.attempted_files = tracker_state["attempted_files"].copy()
        self.file_tracker.success_files = tracker_state["success_files"].copy()
        self.file_tracker.failed_files = tracker_state["failed_files"].copy()
        self.file_tracker.file_naming_patterns = tracker_state["file_naming_patterns"].copy()
        self.file_tracker.read_files = tracker_state.get("read_files", []).copy()

        print(f"[Env] 已恢复状态: 当前节点={self.current_node_id}, "
              f"已解锁文件={len(self.file_tracker.unlocked_files)}个, "
              f"连续失败={self.consecutive_failures}次")

