# galagent/logger/game_logger.py
"""Game run logging system"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ActionLog:
    """Single action record"""
    step: int
    timestamp: str
    node_id: str
    node_name: str
    scene_text: str
    choices: Dict[str, str]  # Modified to dict: {"text": filename, "decision_rationale": reason}
    file_retrieval: Dict[str, Any] = None  # File retrieval info (need_retrieval, opened_files, reason)
    unlocked_files: List[str] = None  # type_help game only
    attempted_files: List[str] = None  # type_help game only: all attempts
    success_files: List[str] = None  # type_help game only: successfully opened files
    failed_files: List[str] = None  # type_help game only: failed attempts
    hint_unlocked_files: List[str] = None  # type_help game only: files auto-unlocked by hint (cumulative)
    consecutive_failures: int = None  # type_help game only: current consecutive failure count
    # Dust game specific fields
    action_type: Any = None  # Dust game: action type
    action_params: Dict[str, Any] = None  # Dust game: action parameters
    current_node_id: str = None  # Dust game: current node ID
    keyword_pool: List[str] = None  # Dust game: keyword pool
    known_events: List[str] = None  # Dust game: known events
    event_pool: List[str] = None  # Dust game: readable event pool
    read_events: List[str] = None  # Dust game: read events
    locked_events: Dict[str, List[str]] = None  # Dust game: locked events
    score: int = None  # Dust game: score
    keys: int = None  # Dust game: key count
    character_orders: Dict[str, List[str]] = None  # Dust game: character event ordering
    order_judgements: List[Dict] = None  # Dust game: ordering judgement results
    awarded_pairs: List[List] = None  # Dust game: scored event pairs


@dataclass
class GameSession:
    """Game session record"""
    session_id: str
    game_type: str
    start_time: str
    model: Optional[str] = None  # LLM model used
    end_time: Optional[str] = None
    total_steps: int = 0
    reached_ending: bool = False
    ending_node: Optional[str] = None
    story_summary: Optional[str] = None  # Story summary and reasoning
    actions: List[ActionLog] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []


class GameLogger:
    """Game logger"""

    def __init__(self, log_dir: Path, game_type: str, session_id: Optional[str] = None, resume: bool = False, model: Optional[str] = None, truncate_after_step: Optional[int] = None):
        """Initialize the logger

        Args:
            log_dir: Log directory
            game_type: Game type
            session_id: Session ID (if resuming an existing session)
            resume: Whether to resume from an existing session
            model: LLM model name used
        """
        self.log_dir = log_dir
        self.game_type = game_type
        self.model = model
        self.truncate_after_step = truncate_after_step
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # If in resume mode and session_id is provided, load the existing session
        if resume and session_id:
            self.session_id = session_id
            self.session_file = self.log_dir / "game_log.json"
            self.summary_file = self.log_dir / "game_log_summary.txt"

            # Attempt to load existing session
            if self.session_file.exists():
                self._load_existing_session()
                print(f"[Logger] Loaded existing session: {self.session_id}, {len(self.session.actions)} records")
            else:
                print(f"[Logger] Warning: existing session file not found, creating new session")
                self._create_new_session()
        else:
            # Create a new session ID
            if session_id:
                self.session_id = session_id
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.session_id = f"{game_type}_{timestamp}"

            self.session_file = self.log_dir / "game_log.json"
            self.summary_file = self.log_dir / "game_log_summary.txt"
            self._create_new_session()

    def _create_new_session(self):
        """Create a new session"""
        self.session = GameSession(
            session_id=self.session_id,
            game_type=self.game_type,
            start_time=datetime.now().isoformat(),
            model=self.model,
            actions=[]
        )

    def _load_existing_session(self):
        """Load an existing session"""
        with open(self.session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        # Rebuild the session object
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

        # Do not truncate existing records; keep full history and append new steps after
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
        # Dust game specific parameters
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
        """Record a single action"""
        action = ActionLog(
            step=step,
            timestamp=datetime.now().isoformat(),
            node_id=node_id,
            node_name=node_name,
            scene_text=scene_text,
            choices=choices,
            file_retrieval=file_retrieval,
            unlocked_files=unlocked_files,
            attempted_files=None,  # not recorded
            success_files=None,  # not recorded
            failed_files=failed_files,
            hint_unlocked_files=hint_unlocked_files,
            consecutive_failures=consecutive_failures,
            # Dust game fields
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

        # Save in real time
        self._save_session()

    def log_ending(self, ending_node: str, reached_ending: bool = True, story_summary: Optional[str] = None) -> None:
        """Record game ending

        Args:
            ending_node: Ending node ID
            reached_ending: Whether an ending was reached
            story_summary: Story summary and reasoning (optional)
        """
        self.session.end_time = datetime.now().isoformat()
        self.session.reached_ending = reached_ending
        self.session.ending_node = ending_node
        self.session.story_summary = story_summary

        self._save_session()
        self._generate_summary()

    def _save_session(self) -> None:
        """Save session to JSON file"""
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.session), f, ensure_ascii=False, indent=2)

    def _generate_summary(self) -> None:
        """Generate a human-readable summary file"""
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

        # Story summary
        if self.session.story_summary:
            lines.append("=" * 80)
            lines.append("Story Summary & Analysis:")
            lines.append("=" * 80)
            lines.append(self.session.story_summary)
            lines.append("")

        # Statistics
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

        # Action trajectory
        lines.append("=" * 80)
        lines.append("Action Trajectory:")
        lines.append("=" * 80)

        for action in self.session.actions:
            lines.append(f"\n{'=' * 80}")
            lines.append(f"[Step {action.step}] {action.timestamp}")
            lines.append(f"{'=' * 80}")

            # 1. Observation info
            lines.append(f"\n[Observation]")
            lines.append(f"  Node: {action.node_id} ({action.node_name})")
            lines.append(f"  Scene: {action.scene_text[:150]}...")

            # 2. LLM decision
            lines.append(f"\n[LLM Decision]")
            lines.append(f"  Choice: {action.choices.get('text', 'N/A')}")
            lines.append(f"  Rationale: {action.choices.get('decision_rationale', 'N/A')}")

            # 3. File retrieval decision (if any)
            if action.file_retrieval:
                lines.append(f"\n[File Retrieval Decision]")
                need_retrieval = action.file_retrieval.get("need_retrieval", False)
                if need_retrieval:
                    opened_files = action.file_retrieval.get("opened_files", [])
                    reason = action.file_retrieval.get("reason", "")
                    lines.append(f"  Need retrieval: Yes")
                    lines.append(f"  Opened files: {', '.join(opened_files) if opened_files else 'None'}")
                    if reason:
                        lines.append(f"  Reason: {reason}")
                else:
                    lines.append(f"  Need retrieval: No")

            # 4. File tracking info (Type Help game only)
            if action.unlocked_files:
                lines.append(f"\n[Unlocked Files]")
                lines.append(f"  {', '.join(action.unlocked_files)}")

            # if action.attempted_files:
            #     recent_attempts = action.attempted_files[-3:]  # show only last 3
            #     lines.append(f"\n[Failed Attempts]")
            #     lines.append(f"  {', '.join(recent_attempts)}")

            lines.append(f"\n{'-' * 80}")

        # Write to file
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n[LOG] Session saved:")
        print(f"  - JSON: {self.session_file}")
        print(f"  - Summary: {self.summary_file}")

    def get_session_path(self) -> Path:
        """Get the session file path"""
        return self.session_file

    def get_summary_path(self) -> Path:
        """Get the summary file path"""
        return self.summary_file
