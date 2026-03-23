# galagent/env/trpg_env.py
"""TRPG evaluation environment"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from galagent.env.base_env import BaseGameEnv, GameConfig
from galagent.common.schemas import Choice, Memory, Observation


@dataclass
class TRPGConfig(GameConfig):
    """TRPG environment configuration"""
    game_type: str = "trpg"
    data_path: Path = Path("dataset/trpg_en/stories")
    qa_path: Path = Path("dataset/trpg_en/qa")
    story_name: str = "Terror_on_the_Orient_Express"
    test_language: str = "en"  # en (English)


class TRPGEnv(BaseGameEnv):
    """
    TRPG evaluation environment (lightweight data container).
    Loads story sections and QA data; does not manage LLM conversations.
    The actual two-phase loop is driven by TRPGRunner.
    """

    def __init__(self, config: TRPGConfig):
        super().__init__(config)
        self.config: TRPGConfig = config
        self.sections: List[Dict[str, Any]] = []   # Ordered by section
        self.qa_list: List[Dict[str, Any]] = []    # Loaded from qa.json
        self.results: List[Dict[str, Any]] = []    # QA results (filled by runner)
        self.load_game_data()

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_game_data(self) -> None:
        """Load story sections and QA"""
        self._load_sections()
        self._load_qa()

    def _load_sections(self) -> None:
        story_path = self.config.data_path / self.config.story_name
        if not story_path.exists():
            raise FileNotFoundError(f"Story not found: {story_path}")
        json_files = sorted(story_path.glob("*_conversation.json"))
        self.sections = []
        for fn in json_files:
            with open(fn, encoding="utf-8") as f:
                data = json.load(f)
            self.sections.append(data)
        print(f"[TRPGEnv] Loaded story '{self.config.story_name}': {len(self.sections)} sections")

    def _load_qa(self) -> None:
        qa_file = self.config.qa_path / f"{self.config.story_name}_qa.json"
        if not qa_file.exists():
            print(f"[TRPGEnv] QA file not found: {qa_file} (QA phase will be skipped)")
            self.qa_list = []
            return
        with open(qa_file, encoding="utf-8") as f:
            data = json.load(f)
        self.qa_list = data[0].get("qa", []) if data else []
        print(f"[TRPGEnv] Loaded QA: {len(self.qa_list)} questions")

    # ── BaseGameEnv interface (TRPG does not use the choose loop; state queries only) ──

    def observe(self) -> Observation:
        """Return a summary of the current environment (for logging/status display)"""
        summary = (
            f"Story: {self.config.story_name} | "
            f"Sections: {len(self.sections)} | "
            f"QA: {len(self.qa_list)} questions"
        )
        return Observation(
            node_id="trpg_env",
            name=self.config.story_name,
            text=summary,
            choices=[],
            is_ending=False,
            memory=Memory(description=summary),
        )

    def choose(self, choice_index: int) -> None:
        """TRPG mode does not use the choose interface"""
        pass

    def reset(self) -> None:
        """Reset results"""
        self.results = []

    # ── Statistics ────────────────────────────────────────────────────────────

    def total_turns(self) -> int:
        """Total number of conversation turns across all sections"""
        return sum(
            len(s.get("conversation", []))
            for s in self.sections
        )
