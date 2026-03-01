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
    choices: Dict[str, str]  # 修改为字典：{"text": 文件名, "decision_rationale": 原因}
    file_retrieval: Dict[str, Any] = None  # 文件检索信息（need_retrieval, opened_files, reason）
    unlocked_files: List[str] = None  # type_help游戏专用
    attempted_files: List[str] = None  # type_help游戏专用：所有尝试
    success_files: List[str] = None  # type_help游戏专用：成功打开的文件
    failed_files: List[str] = None  # type_help游戏专用：失败的尝试
    hint_unlocked_files: List[str] = None  # type_help游戏专用：hint自动解锁的文件（累计）
    consecutive_failures: int = None  # type_help游戏专用：当前连续失败次数
    # Dust 游戏专用字段
    action_type: Any = None  # Dust 游戏：动作类型
    action_params: Dict[str, Any] = None  # Dust 游戏：动作参数
    current_node_id: str = None  # Dust 游戏：当前节点ID
    keyword_pool: List[str] = None  # Dust 游戏：关键词池
    known_events: List[str] = None  # Dust 游戏：已知事件
    event_pool: List[str] = None  # Dust 游戏：可阅读事件池
    read_events: List[str] = None  # Dust 游戏：已阅读事件
    locked_events: Dict[str, List[str]] = None  # Dust 游戏：锁定事件
    score: int = None  # Dust 游戏：得分
    keys: int = None  # Dust 游戏：钥匙数
    character_orders: Dict[str, List[str]] = None  # Dust 游戏：角色事件排序
    order_judgements: List[Dict] = None  # Dust 游戏：排序判断结果
    awarded_pairs: List[List] = None  # Dust 游戏：已计分事件对


@dataclass
class GameSession:
    """游戏会话记录"""
    session_id: str
    game_type: str
    start_time: str
    model: Optional[str] = None  # 使用的LLM模型
    end_time: Optional[str] = None
    total_steps: int = 0
    reached_ending: bool = False
    ending_node: Optional[str] = None
    story_summary: Optional[str] = None  # 故事总结与推理
    actions: List[ActionLog] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []


class GameLogger:
    """游戏日志记录器"""

    def __init__(self, log_dir: Path, game_type: str, session_id: Optional[str] = None, resume: bool = False, model: Optional[str] = None, truncate_after_step: Optional[int] = None):
        """初始化日志记录器

        Args:
            log_dir: 日志目录
            game_type: 游戏类型
            session_id: 会话ID（如果要恢复现有session）
            resume: 是否从现有session恢复
            model: 使用的LLM模型名称
        """
        self.log_dir = log_dir
        self.game_type = game_type
        self.model = model
        self.truncate_after_step = truncate_after_step
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 如果是恢复模式且提供了session_id，则加载现有session
        if resume and session_id:
            self.session_id = session_id
            self.session_file = self.log_dir / f"{self.session_id}.json"
            self.summary_file = self.log_dir / f"{self.session_id}_summary.txt"

            # 尝试加载现有session
            if self.session_file.exists():
                self._load_existing_session()
                print(f"[Logger] 已加载现有session: {self.session_id}, 当前有{len(self.session.actions)}条记录")
            else:
                print(f"[Logger] 警告: 未找到现有session文件，将创建新session")
                self._create_new_session()
        else:
            # 创建新的会话ID
            if session_id:
                self.session_id = session_id
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.session_id = f"{game_type}_{timestamp}"

            self.session_file = self.log_dir / f"{self.session_id}.json"
            self.summary_file = self.log_dir / f"{self.session_id}_summary.txt"
            self._create_new_session()

    def _create_new_session(self):
        """创建新的session"""
        self.session = GameSession(
            session_id=self.session_id,
            game_type=self.game_type,
            start_time=datetime.now().isoformat(),
            model=self.model,
            actions=[]
        )

    def _load_existing_session(self):
        """加载现有的session"""
        with open(self.session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        # 重建session对象
        actions = [
            ActionLog(
                step=action['step'],
                timestamp=action['timestamp'],
                node_id=action['node_id'],
                node_name=action['node_name'],
                scene_text=action['scene_text'],
                choices=action['choices'],
                file_retrieval=action.get('file_retrieval'),
                unlocked_files=action.get('unlocked_files'),
                attempted_files=action.get('attempted_files'),
                success_files=action.get('success_files'),
                failed_files=action.get('failed_files'),
                hint_unlocked_files=action.get('hint_unlocked_files'),
                consecutive_failures=action.get('consecutive_failures'),
                action_type=action.get('action_type'),
                action_params=action.get('action_params'),
                current_node_id=action.get('current_node_id'),
                keyword_pool=action.get('keyword_pool'),
                known_events=action.get('known_events'),
                event_pool=action.get('event_pool'),
                read_events=action.get('read_events'),
                locked_events=action.get('locked_events'),
                score=action.get('score'),
                keys=action.get('keys'),
                character_orders=action.get('character_orders'),
                order_judgements=action.get('order_judgements'),
                awarded_pairs=action.get('awarded_pairs'),
            )
            for action in session_data['actions']
        ]

        # 不截断已有记录，直接保留全部历史，新步骤追加在后面
        # if self.truncate_after_step is not None:
        #     actions = [a for a in actions if a.step <= self.truncate_after_step]

        self.session = GameSession(
            session_id=session_data['session_id'],
            game_type=session_data['game_type'],
            start_time=session_data['start_time'],
            model=session_data.get('model'),
            end_time=session_data.get('end_time'),
            total_steps=session_data.get('total_steps', 0),
            reached_ending=session_data.get('reached_ending', False),
            ending_node=session_data.get('ending_node'),
            actions=actions
        )

    def log_action(
        self,
        step: int,
        node_id: str,
        node_name: str,
        scene_text: str,
        choices: Dict[str, str],
        file_retrieval: Optional[Dict[str, Any]] = None,
        unlocked_files: Optional[List[str]] = None,
        attempted_files: Optional[List[str]] = None,
        success_files: Optional[List[str]] = None,
        failed_files: Optional[List[str]] = None,
        hint_unlocked_files: Optional[List[str]] = None,
        consecutive_failures: Optional[int] = None,
        # Dust 游戏专用参数
        action_type: Any = None,
        action_params: Optional[Dict[str, Any]] = None,
        current_node_id: Optional[str] = None,
        keyword_pool: Optional[List[str]] = None,
        known_events: Optional[List[str]] = None,
        event_pool: Optional[List[str]] = None,
        read_events: Optional[List[str]] = None,
        locked_events: Optional[Dict[str, List[str]]] = None,
        score: Optional[int] = None,
        keys: Optional[int] = None,
        character_orders: Optional[Dict[str, List[str]]] = None,
        order_judgements: Optional[List[Dict]] = None,
        awarded_pairs: Optional[List[List]] = None
    ) -> None:
        """记录一次行动"""
        action = ActionLog(
            step=step,
            timestamp=datetime.now().isoformat(),
            node_id=node_id,
            node_name=node_name,
            scene_text=scene_text,
            choices=choices,
            file_retrieval=file_retrieval,
            unlocked_files=unlocked_files,
            attempted_files=None,  # 不记录
            success_files=None,  # 不记录
            failed_files=failed_files,
            hint_unlocked_files=hint_unlocked_files,
            consecutive_failures=consecutive_failures,
            # Dust 游戏字段
            action_type=action_type,
            action_params=action_params,
            current_node_id=current_node_id,
            keyword_pool=keyword_pool,
            known_events=known_events,
            event_pool=event_pool,
            read_events=read_events,
            locked_events=locked_events,
            score=score,
            keys=keys,
            character_orders=character_orders,
            order_judgements=order_judgements,
            awarded_pairs=awarded_pairs
        )

        self.session.actions.append(action)
        self.session.total_steps = step + 1

        # 实时保存
        self._save_session()

    def log_ending(self, ending_node: str, reached_ending: bool = True, story_summary: Optional[str] = None) -> None:
        """记录游戏结束

        Args:
            ending_node: 结束节点ID
            reached_ending: 是否达到结局
            story_summary: 故事总结与推理（可选）
        """
        self.session.end_time = datetime.now().isoformat()
        self.session.reached_ending = reached_ending
        self.session.ending_node = ending_node
        self.session.story_summary = story_summary

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

        # 故事总结
        if self.session.story_summary:
            lines.append("=" * 80)
            lines.append("Story Summary & Analysis:")
            lines.append("=" * 80)
            lines.append(self.session.story_summary)
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
            lines.append(f"\n{'=' * 80}")
            lines.append(f"[Step {action.step}] {action.timestamp}")
            lines.append(f"{'=' * 80}")

            # 1. 观察信息
            lines.append(f"\n[观察]")
            lines.append(f"  节点: {action.node_id} ({action.node_name})")
            lines.append(f"  场景: {action.scene_text[:150]}...")

            # 2. LLM决策
            lines.append(f"\n[LLM决策]")
            lines.append(f"  选择: {action.choices.get('text', 'N/A')}")
            lines.append(f"  理由: {action.choices.get('decision_rationale', 'N/A')}")

            # 3. 文件检索决策（如果有）
            if action.file_retrieval:
                lines.append(f"\n[文件检索决策]")
                need_retrieval = action.file_retrieval.get("need_retrieval", False)
                if need_retrieval:
                    opened_files = action.file_retrieval.get("opened_files", [])
                    reason = action.file_retrieval.get("reason", "")
                    lines.append(f"  需要检索: 是")
                    lines.append(f"  打开的文件: {', '.join(opened_files) if opened_files else '无'}")
                    if reason:
                        lines.append(f"  原因: {reason}")
                else:
                    lines.append(f"  需要检索: 否")

            # 4. 文件追踪信息（Type Help游戏专用）
            if action.unlocked_files:
                lines.append(f"\n[解锁的文件]")
                lines.append(f"  {', '.join(action.unlocked_files)}")

            # if action.attempted_files:
            #     recent_attempts = action.attempted_files[-3:]  # 只显示最近3次
            #     lines.append(f"\n[失败的尝试]")
            #     lines.append(f"  {', '.join(recent_attempts)}")

            lines.append(f"\n{'-' * 80}")

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
