# galagent/memory/base_mem_agent.py
"""Memory agent base class, defines the unified interface"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseMemAgent(ABC):
    """Memory agent base class

    All memory agents (Mem0, A-mem, etc.) should inherit from this class and implement these methods
    """

    def __init__(self, game_name: str = "game_agent", verbose: bool = False):
        """Initialize the memory agent

        Args:
            game_name: Game name, used as user_id for data isolation
            verbose: Whether to print detailed logs
        """
        self.game_name = game_name
        self.verbose = verbose

    @abstractmethod
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add a memory

        Args:
            content: Memory content
            metadata: Metadata (e.g. role, step, source, etc.)

        Returns:
            Dictionary containing success and memory_id
        """
        pass

    @abstractmethod
    def search_memories(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search memories

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Filter conditions, supports simplified format e.g. {"role": "user"} or {"step": "2"}

        Returns:
            List of memories, each containing id, text, score, metadata, etc.
        """
        pass

    @abstractmethod
    def get_all_memories(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get all memories

        Args:
            filters: Optional filter conditions, supports simplified format

        Returns:
            List of memories
        """
        pass

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update a memory

        Args:
            memory_id: Memory ID
            content: New content
            metadata: New metadata

        Returns:
            Dictionary containing success
        """
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a single memory

        Args:
            memory_id: Memory ID

        Returns:
            Dictionary containing success
        """
        pass

    @abstractmethod
    def delete_all_memories(self) -> Dict[str, Any]:
        """Delete all memories for this game

        Returns:
            Dictionary containing success
        """
        pass

    def get_memory_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memory history (only mem0 has implemented this)

        Args:
            limit: Number of items to return

        Returns:
            Memory list (sorted in reverse chronological order)
        """
        all_memories = self.get_all_memories()
        # Sort in reverse order by creation time
        sorted_memories = sorted(
            all_memories,
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        return sorted_memories[:limit]

    def format_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for prompt (optional implementation)

        Args:
            memories: List of memories

        Returns:
            Formatted text
        """
        if not memories:
            return "No relevant memories found."

        lines = ["Relevant memories:"]
        for i, mem in enumerate(memories, 1):
            text = mem.get("text", "")
            metadata = mem.get("metadata", {})
            role = metadata.get("role", "unknown")
            lines.append(f"{i}. [{role}] {text}")

        return "\n".join(lines)

    def get_state(self) -> Dict[str, Any]:
        """Get current state (for checkpoint)

        Returns:
            State dictionary
        """
        return {
            "game_name": self.game_name,
            "verbose": self.verbose
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state (from checkpoint)

        Args:
            state: State dictionary
        """
        self.game_name = state.get("game_name", self.game_name)
        self.verbose = state.get("verbose", self.verbose)

        if self.verbose:
            print(f"[MemAgent] State restored: game_name={self.game_name}")
