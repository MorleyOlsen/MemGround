# galagent/env/base_game_utils.py
"""Base class for game-specific utilities"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from galagent.common.schemas import Observation


class BaseGameUtils(ABC):
    """Base class for game-specific utilities

    Each game can implement its own utility class to handle:
    - Retrieving game context information
    - Formatting log information
    - Memory management (including game-specific memory protection)
    - Other game-specific helper functionality
    """

    def __init__(self):
        self.memory_store = None

    def set_memory_store(self, store):
        """Set the memory store reference

        Args:
            store: MemoryStore instance
        """
        self.memory_store = store

    @abstractmethod
    def get_game_context(self, env: Any) -> Dict[str, Any]:
        """Get game-specific context information

        Args:
            env: Game environment instance

        Returns:
            Game context dictionary
        """
        pass

    @abstractmethod
    def format_log_data(
        self,
        step: int,
        obs: Observation,
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format log data

        Args:
            step: Current step number
            obs: Current observation
            decision: Decision object
            game_context: Game context
            retrieval_decision: Retrieval decision info (contains need_retrieval, query, reason)

        Returns:
            Complete log data dictionary with all fields to be recorded
        """
        pass

    @abstractmethod
    def execute_action(self, env: Any, decision: Any) -> None:
        """Execute a game action

        Args:
            env: Game environment instance
            decision: Decision object
        """
        pass

    def post_action_hook(self, obs: Observation, decision: Any, action_success: bool = True, step: int = 0) -> None:
        """Hook function called after an action is executed (optional implementation)

        Used for game-specific processing after an action, for example:
        - Recording visited nodes
        - Updating game state
        - Collecting statistics
        - Handling action failures

        Args:
            obs: Current observation
            decision: Executed decision
            action_success: Whether the action succeeded (default True)
            step: Current step number
        """
        pass

    def retrieve_information(self, retrieval_decision: Dict[str, Any], config: Any) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Retrieve information (game-specific implementation)

        Args:
            retrieval_decision: Retrieval decision returned by policy.decide_retrieval()
                - For KB game: {"need_retrieval": bool, "query": str, "reason": str}
                - For Type Help game: {"need_retrieval": bool, "filenames": list, "reason": str}
            config: Agent configuration object

        Returns:
            (Formatted retrieval result text, retrieval decision info dict)
            Returns (None, retrieval_decision) if no retrieval is needed
        """
        # Default implementation: no retrieval
        return None, retrieval_decision

    def manage_memory(self, config: Any) -> None:
        """Manage the memory store

        Unified memory management interface; subclasses may override to implement
        game-specific memory protection logic

        Args:
            config: Agent configuration object
        """
        if not self.memory_store:
            return

        # Default implementation: sliding window that prioritizes deleting assistant messages
        current_tokens = self.memory_store.get_total_tokens_estimate()
        max_context_tokens = config.max_context_tokens
        if current_tokens > max_context_tokens:
            delete_count = (current_tokens - max_context_tokens) // 50 + 1
            deleted = self.memory_store.delete_by_priority(delete_count)
            if config.verbose:
                print(f"[Context management] Deleted {deleted} memories (assistant first), current tokens: {self.memory_store.get_total_tokens_estimate()}")

    def get_console_log_info(
        self,
        obs: Observation,
        search_results: str,
        decision: Any
    ) -> Optional[str]:
        """Get console log information (optional implementation)

        Args:
            obs: Current observation
            search_results: Search results
            decision: Decision object

        Returns:
            Formatted console log string, or None if not needed
        """
        return None
