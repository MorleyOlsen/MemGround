# galagent/logger/game_logger.py
"""游戏运行日志记录系统"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ActionLog:
    """单次行动记录"""
    step: int
    timestamp: str
    node_id: str
    node_name: str
    scene_text: str
    choices: List[Dict[str, Any]]
    retrieved_memory: List[str]
    decision_index: int
    decision_rationale: str
    unlocked_files: List[str] = None  # type_help游戏专用
    attempted_files: List[str] = None  # type_help游戏专用
    retrieval_decision: Dict[str, Any] = None  # 检索决策信息


@dataclass
class GameSession:
    """游戏会话记录"""
    session_id: str
    game_type: str
    start_time: str
    end_time: Optional[str] = None
    total_steps: int = 0
    reached_ending: bool = False
    ending_node: Optional[str] = None
    actions: List[ActionLog] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []


class GameLogger:
    """游戏日志记录器"""

    def __init__(self, log_dir: Path, game_type: str):
        self.log_dir = log_dir
        self.game_type = game_type
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建会话ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{game_type}_{timestamp}"

        # 初始化会话
        self.session = GameSession(
            session_id=self.session_id,
            game_type=game_type,
            start_time=datetime.now().isoformat(),
            actions=[]
        )

        # 日志文件路径
        self.session_file = self.log_dir / f"{self.session_id}.json"
        self.summary_file = self.log_dir / f"{self.session_id}_summary.txt"

    def log_action(
        self,
        step: int,
        node_id: str,
        node_name: str,
        scene_text: str,
        choices: List[Dict[str, Any]],
        retrieved_memory: List[str],
        decision_index: int,
        decision_rationale: str,
        unlocked_files: Optional[List[str]] = None,
        attempted_files: Optional[List[str]] = None,
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录一次行动"""
        action = ActionLog(
            step=step,
            timestamp=datetime.now().isoformat(),
            node_id=node_id,
            node_name=node_name,
            scene_text=scene_text,
            choices=choices,
            retrieved_memory=retrieved_memory,
            decision_index=decision_index,
            decision_rationale=decision_rationale,
            unlocked_files=unlocked_files,
            attempted_files=attempted_files,
            retrieval_decision=retrieval_decision
        )

        self.session.actions.append(action)
        self.session.total_steps = step + 1

        # 实时保存
        self._save_session()

    def log_ending(self, ending_node: str, reached_ending: bool = True) -> None:
        """记录游戏结束"""
        self.session.end_time = datetime.now().isoformat()
        self.session.reached_ending = reached_ending
        self.session.ending_node = ending_node

        self._save_session()
        self._generate_summary()

    def _save_session(self) -> None:
        """保存会话到JSON文件"""
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.session), f, ensure_ascii=False, indent=2)

    def _generate_summary(self) -> None:
        """生成可读的摘要文件"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"Game Session Summary: {self.session_id}")
        lines.append("=" * 80)
        lines.append(f"Game Type: {self.session.game_type}")
        lines.append(f"Start Time: {self.session.start_time}")
        lines.append(f"End Time: {self.session.end_time}")
        lines.append(f"Total Steps: {self.session.total_steps}")
        lines.append(f"Reached Ending: {self.session.reached_ending}")
        lines.append(f"Ending Node: {self.session.ending_node}")
        lines.append("")

        # 统计信息
        if self.game_type == "type_help":
            all_unlocked = set()
            all_attempted = set()
            for action in self.session.actions:
                if action.unlocked_files:
                    all_unlocked.update(action.unlocked_files)
                if action.attempted_files:
                    all_attempted.update(action.attempted_files)

            lines.append("File Statistics:")
            lines.append(f"  Total Unlocked Files: {len(all_unlocked)}")
            lines.append(f"  Unlocked: {sorted(all_unlocked)}")
            lines.append(f"  Total Attempted Files: {len(all_attempted)}")
            lines.append(f"  Attempted: {sorted(all_attempted)}")
            lines.append("")

        # 行动轨迹
        lines.append("=" * 80)
        lines.append("Action Trajectory:")
        lines.append("=" * 80)

        for action in self.session.actions:
            lines.append(f"\n[Step {action.step}] {action.timestamp}")
            lines.append(f"Node: {action.node_id} ({action.node_name})")
            lines.append(f"Scene: {action.scene_text[:100]}...")
            lines.append(f"Choices:")
            for choice in action.choices:
                lines.append(f"  [{choice['index']}] {choice['text']}")
            lines.append(f"Decision: {action.decision_index} - {action.decision_rationale}")

            if action.unlocked_files:
                lines.append(f"Unlocked Files: {action.unlocked_files}")

            if action.retrieved_memory:
                lines.append(f"Retrieved Memory ({len(action.retrieved_memory)} items):")
                for mem in action.retrieved_memory[:3]:  # 只显示前3条
                    lines.append(f"  - {mem[:80]}...")

            lines.append("-" * 80)

        # 写入文件
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n[LOG] Session saved:")
        print(f"  - JSON: {self.session_file}")
        print(f"  - Summary: {self.summary_file}")

    def get_session_path(self) -> Path:
        """获取会话文件路径"""
        return self.session_file

    def get_summary_path(self) -> Path:
        """获取摘要文件路径"""
        return self.summary_file
