"""Base class for game-specific prompt builders"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from galagent.common.schemas import Observation


class BasePromptBuilder(ABC):
    """Prompt builder base class

    Each game can implement its own PromptBuilder to customize:
    - System prompt
    - Scene information formatting
    - Game-specific context information
    - Output format requirements
    """

    def __init__(self, goal_instruction: str = ""):
        self.goal_instruction = goal_instruction
        self.memory_store = None  # Will be set in the policy

    def set_memory_store(self, store):
        """Set the memory store for retrieving conversation history"""
        self.memory_store = store

    @abstractmethod
    def build_system_prompt(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """Build the system prompt

        Args:
            game_context: Game-specific context information (optional)

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def build_user_prompt(
        self,
        obs: Observation,
        retrieved_hits: List[str],
        game_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the user prompt

        Args:
            obs: Current observation
            retrieved_hits: Retrieved memories
            game_context: Game-specific context information

        Returns:
            User prompt string
        """
        pass

    def format_conversation_history(self, max_turns: int = 10) -> str:
        """Format conversation history as text

        Args:
            max_turns: Maximum number of turns to display (default 10)

        Returns:
            Formatted conversation history string
        """
        if not self.memory_store:
            return "(no conversation history)"

        # Retrieve the most recent conversation history
        recent_messages = self.memory_store.recent(k=max_turns * 2)  # Each turn includes user and assistant

        if not recent_messages:
            return "(no conversation history)"

        history_lines = []
        for item in recent_messages:
            role = item.meta.get("role", "unknown")
            content = item.text

            # Format each message
            if role == "user":
                history_lines.append(f"[Observation] {content}")
            elif role == "assistant":
                history_lines.append(f"[Decision] {content}")
            elif role == "system":
                history_lines.append(f"[Retrieved Memory] {content}")
            else:
                history_lines.append(f"[{role}] {content}")

        return "\n".join(history_lines)

    def get_conversation_messages(self) -> List[Dict[str, str]]:
        """Get the conversation history as a message list (for passing directly to an LLM)

        Returns:
            Message list in the format [{"role": "user/assistant", "content": "..."}]
        """
        if not self.memory_store:
            return []

        return self.memory_store.to_chat_messages()

    def format_choices(self, obs: Observation) -> str:
        """Format the choices list (common implementation)

        Args:
            obs: Current observation

        Returns:
            Formatted choices string
        """
        choices_lines = []
        for c in obs.choices:
            choices_lines.append(f"{c.index}: {c.text}")
        return "\n".join(choices_lines) if choices_lines else "(no choices)"

    def format_characters(self, obs: Observation) -> str:
        """Format character information (common implementation)

        Args:
            obs: Current observation

        Returns:
            Formatted character information string
        """
        chars = obs.memory.characters or []
        if chars:
            return "\n".join(
                [f"- {x.name} | role={x.role} | desc={x.description}" for x in chars]
            )
        return "(none)"

    def format_retrieved_memory(self, retrieved_hits: List[str]) -> str:
        """Format retrieved memories (common implementation)

        Args:
            retrieved_hits: List of retrieved memories

        Returns:
            Formatted memory string
        """
        return "\n".join([f"- {t}" for t in retrieved_hits]) if retrieved_hits else "(none)"

    def build_retrieval_decision_prompt(self, obs: Observation) -> str:
        """Build the retrieval decision prompt (common implementation; subclasses may override; not yet used)

        Args:
            obs: Current observation

        Returns:
            Retrieval decision prompt string
        """
        return f"""
        CURRENT SCENE:
        {obs.text}

        TASK:
        Decide if you need to retrieve past memories to help make a decision for this scene.
        If yes, write a concise search query to find relevant past information.

        OUTPUT FORMAT (STRICT JSON ONLY):
        {{
        "need_retrieval": true/false,
        "query": "<your search query if need_retrieval is true, otherwise empty string>",
        "reason": "<brief reason for your decision>"
        }}

        GUIDELINES:
        - Set need_retrieval to true if past information would help understand the current situation
        - The query should be concise and focus on key information you need
        - Set need_retrieval to false if the current scene provides enough information
        """.strip()

    def build_retrieval_prompt(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        """Build the file retrieval prompt (Type Help game specific; subclasses may override)

        Args:
            obs: Current observation
            game_context: Game context

        Returns:
            File retrieval prompt string
        """
        # Default implementation returns empty string; subclasses can override
        return ""