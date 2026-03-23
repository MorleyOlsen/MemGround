# env/trpg/utils/game_utils.py
"""Utility class for the TRPG game"""
from __future__ import annotations

from typing import Any, Dict, Optional

from galagent.common.schemas import Observation
from galagent.env.base_game_utils import BaseGameUtils


class TRPGGameUtils(BaseGameUtils):
    """
    GameUtils for TRPG mode.
    TRPG does not use a choose/execute_action loop;
    this class mainly provides context information and log formatting interfaces.
    """

    def get_game_context(self, env: Any) -> Dict[str, Any]:
        return {
            "story":     env.config.story_name,
            "sections":  len(env.sections),
            "qa_count":  len(env.qa_list),
            "results":   len(env.results),
        }

    def format_log_data(
        self,
        step: int,
        obs: Observation,
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "step":         step,
            "obs_text":     obs.text if obs else "",
            "game_context": game_context,
        }

    def execute_action(self, env: Any, decision: Any) -> bool:
        """TRPG mode does not use this interface"""
        return True
