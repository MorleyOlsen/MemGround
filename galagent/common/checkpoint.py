# galagent/common/checkpoint.py
"""Checkpoint manager for saving and restoring game state"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class CheckpointManager:
    """Manages saving and restoring game state"""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize the checkpoint manager

        Args:
            checkpoint_dir: Directory where checkpoints are saved
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        step: int,
        env_state: Dict[str, Any],
        memory_state: Dict[str, Any],
        game_utils_state: Dict[str, Any],
        session_name: Optional[str] = None,
        logger_session_id: Optional[str] = None
    ) -> Path:
        """Save a checkpoint

        Args:
            step: Current step number
            env_state: Environment state
            memory_state: Memory state
            game_utils_state: Game utilities state
            session_name: Session name (optional)
            logger_session_id: Logger session_id (used to continue writing to the same log file on resume)

        Returns:
            Path to the saved checkpoint file
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "env_state": env_state,
            "memory_state": memory_state,
            "game_utils_state": game_utils_state,
            "logger_session_id": logger_session_id  # Save the logger's session_id
        }

        checkpoint_file = self.checkpoint_dir / f"step_{step:06d}.json"

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

        print(f"[Checkpoint] Saved to: {checkpoint_file}")
        return checkpoint_file

    def load_checkpoint(self, checkpoint_file: Path) -> Dict[str, Any]:
        """Load a checkpoint

        Args:
            checkpoint_file: Path to the checkpoint file

        Returns:
            Checkpoint data dictionary
        """
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        print(f"[Checkpoint] Loaded: {checkpoint_file}")
        print(f"  - Step: {checkpoint_data['step']}")
        print(f"  - Timestamp: {checkpoint_data['timestamp']}")

        return checkpoint_data

    def list_checkpoints(self, session_name: Optional[str] = None) -> list[Path]:
        """List all checkpoint files

        Args:
            session_name: Session name (optional); if provided, only lists checkpoints for that session

        Returns:
            List of checkpoint file paths
        """
        if session_name:
            pattern = f"step_*.json"
        else:
            pattern = "step_*.json"

        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        return checkpoints

    def get_latest_checkpoint(self, session_name: Optional[str] = None) -> Optional[Path]:
        """Get the most recent checkpoint

        Args:
            session_name: Session name (optional)

        Returns:
            Path to the latest checkpoint file, or None if none exist
        """
        checkpoints = self.list_checkpoints(session_name)
        return checkpoints[-1] if checkpoints else None
