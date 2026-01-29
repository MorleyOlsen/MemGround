# galagent/common/checkpoint.py
"""Checkpoint管理器，用于保存和恢复游戏状态"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class CheckpointManager:
    """管理游戏状态的保存和恢复"""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """初始化checkpoint管理器

        Args:
            checkpoint_dir: checkpoint保存目录
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
        """保存checkpoint

        Args:
            step: 当前步数
            env_state: 环境状态
            memory_state: 记忆状态
            game_utils_state: 游戏工具状态
            session_name: 会话名称（可选）
            logger_session_id: 日志记录器的session_id（用于恢复时继续写入同一个log文件）

        Returns:
            保存的checkpoint文件路径
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "env_state": env_state,
            "memory_state": memory_state,
            "game_utils_state": game_utils_state,
            "logger_session_id": logger_session_id  # 保存logger的session_id
        }

        checkpoint_file = self.checkpoint_dir / f"checkpoint_{session_name}_step_{step}.json"

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

        print(f"[Checkpoint] 已保存到: {checkpoint_file}")
        return checkpoint_file

    def load_checkpoint(self, checkpoint_file: Path) -> Dict[str, Any]:
        """加载checkpoint

        Args:
            checkpoint_file: checkpoint文件路径

        Returns:
            checkpoint数据字典
        """
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        print(f"[Checkpoint] 已加载: {checkpoint_file}")
        print(f"  - Step: {checkpoint_data['step']}")
        print(f"  - Timestamp: {checkpoint_data['timestamp']}")

        return checkpoint_data

    def list_checkpoints(self, session_name: Optional[str] = None) -> list[Path]:
        """列出所有checkpoint文件

        Args:
            session_name: 会话名称（可选），如果提供则只列出该会话的checkpoints

        Returns:
            checkpoint文件路径列表
        """
        if session_name:
            pattern = f"checkpoint_{session_name}_step_*.json"
        else:
            pattern = "checkpoint_*.json"

        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        return checkpoints

    def get_latest_checkpoint(self, session_name: Optional[str] = None) -> Optional[Path]:
        """获取最新的checkpoint

        Args:
            session_name: 会话名称（可选）

        Returns:
            最新的checkpoint文件路径，如果没有则返回None
        """
        checkpoints = self.list_checkpoints(session_name)
        return checkpoints[-1] if checkpoints else None
