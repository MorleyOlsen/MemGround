# galagent/env/base_env.py
"""Game environment base class"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path

from galagent.common.schemas import Observation


@dataclass
class GameConfig:
    """Game configuration base class"""
    game_type: str
    data_path: Path
    start_node_id: str = "start"


class BaseGameEnv(ABC):
    """Game environment base class"""

    def __init__(self, config: GameConfig):
        self.config = config
        self.current_node_id = config.start_node_id

    @abstractmethod
    def load_game_data(self) -> None:
        """Load game data"""
        pass

    @abstractmethod
    def observe(self) -> Observation:
        """Get the current observation"""
        pass

    @abstractmethod
    def choose(self, choice_index: int) -> None:
        """Execute a choice"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment"""
        pass
