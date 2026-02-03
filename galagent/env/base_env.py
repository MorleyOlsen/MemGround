# galagent/env/base_env.py
"""游戏环境基类"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path

from galagent.common.schemas import Observation


@dataclass
class GameConfig:
    """游戏配置基类"""
    game_type: str
    data_path: Path
    start_node_id: str = "start"


class BaseGameEnv(ABC):
    """游戏环境基类"""

    def __init__(self, config: GameConfig):
        self.config = config
        self.current_node_id = config.start_node_id

    @abstractmethod
    def load_game_data(self) -> None:
        """加载游戏数据"""
        pass

    @abstractmethod
    def observe(self) -> Observation:
        """获取当前观察"""
        pass

    @abstractmethod
    def choose(self, choice_index: int) -> None:
        """执行选择"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置环境"""
        pass
