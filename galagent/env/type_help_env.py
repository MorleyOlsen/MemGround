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
        self.attempted_files: List[str] = []  # 尝试打开的文件名历史
        self.file_naming_patterns: List[str] = []  # 发现的命名规则

    def unlock_file(self, filename: str) -> bool:
        """解锁文件"""
        if filename not in self.unlocked_files:
            self.unlocked_files.add(filename)
            return True
        return False

    def attempt_file(self, filename: str) -> None:
        """记录尝试打开文件"""
        self.attempted_files.append(filename)

    def add_pattern(self, pattern: str) -> None:
        """添加发现的命名规则"""
        if pattern not in self.file_naming_patterns:
            self.file_naming_patterns.append(pattern)

    def get_unlocked_files(self) -> List[str]:
        """获取已解锁文件列表"""
        return sorted(list(self.unlocked_files))

    def get_attempted_files(self) -> List[str]:
        """获取尝试历史"""
        return self.attempted_files.copy()

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
        background_nodes = ["Background", "message", "00-readme"]
        for node_name in background_nodes:
            if node_name in self.nodes:
                self.file_tracker.unlock_file(node_name)

        # 从 message 节点获取初始文件列表
        if "message" in self.nodes:
            memory = self.nodes["message"].get("memory", {})
            files = memory.get("files", [])
            for file in files:
                # 只解锁完整的文件名（不包含 ???? 的）
                if "?" not in file:
                    self.file_tracker.unlock_file(file)

    # 只能观察到文本信息和当前解锁的信息，不能观察到下一步跳转信息
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
        unlocked_files = self.file_tracker.get_unlocked_files()
        if unlocked_files:
            files_info = "\n[已解锁的文件]: " + ", ".join(unlocked_files)
            text += files_info

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
        # 记录尝试
        self.file_tracker.attempt_file(filename)

        # 检查文件是否存在
        if filename not in self.nodes:
            print(f"[Type Help] File not found: {filename}")
            return False

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
            "patterns": self.file_tracker.file_naming_patterns
        }

    def reset(self) -> None:
        """重置环境"""
        self.current_node_id = self.config.start_node_id
        self.history.clear()
        self.file_tracker = FileTracker()
        self._initialize_unlocked_files()


def main():
    """测试Type Help环境的observe输出"""
    from pathlib import Path

    # 设置游戏数据路径
    game_root = Path(__file__).resolve().parent.parent.parent / "env" / "type_help"

    # 创建配置
    config = TypeHelpConfig(
        game_type="type_help",
        data_path=game_root,
        start_node_id="Start"
    )

    # 初始化环境
    env = TypeHelpEnv(config)

    print("=" * 80)
    print("Type Help Environment Test")
    print("=" * 80)
    print(f"Game data path: {game_root}")
    print(f"Total nodes loaded: {len(env.nodes)}")
    print(f"Starting node: {env.current_node_id}")
    print()

    # 交互式测试
    step = 0
    while True:
        print("\n" + "=" * 80)
        print(f"STEP {step}")
        print("=" * 80)

        # 获取当前观察
        obs = env.observe()

        print(f"\nNode ID: {obs.node_id}")
        print(f"Node Name: {obs.name}")
        print(f"Is Ending: {obs.is_ending}")
        print()

        print("TEXT:")
        print("-" * 80)
        print(obs.text)
        print("-" * 80)
        print()

        print("MEMORY INFO:")
        print(f"  Location: {obs.memory.location}")
        print(f"  Time: {obs.memory.time}")
        print(f"  Key Info: {obs.memory.key_info}")
        print(f"  Characters: {[c.name for c in obs.memory.characters]}")
        print()

        # 显示文件追踪信息
        file_info = env.get_file_tracker_info()
        print("FILE TRACKER:")
        print(f"  Unlocked: {file_info['unlocked_files']}")
        print(f"  Attempted: {file_info['attempted_files']}")
        print(f"  Patterns: {file_info['patterns']}")
        print()

        # 如果是结局节点，退出
        if obs.is_ending:
            print("\n[GAME ENDED]")
            break

        # 显示选择
        print("CHOICES:")
        for choice in obs.choices:
            print(f"  [{choice.index}] {choice.text}")
        print()

        # 用户输入
        try:
            user_input = input("Enter choice index (or 'q' to quit): ").strip()

            if user_input.lower() == 'q':
                print("\nExiting test...")
                break

            choice_idx = int(user_input)

            # 执行选择
            env.choose(choice_idx)
            step += 1

        except ValueError:
            print("\n[ERROR] Invalid input. Please enter a number or 'q'.")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            break

    print("\n" + "=" * 80)
    print("Test completed!")
    print(f"Total steps: {step}")
    print(f"History: {env.history}")
    print("=" * 80)


if __name__ == "__main__":
    main()
