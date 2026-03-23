# galagent/memory/mem0.py
"""Mem0 memory management wrapper for game agent"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
# from mem0 import MemoryClient

from .base_mem_agent import BaseMemAgent


class Mem0Agent(BaseMemAgent):
    """Wrapper for Mem0 memory management system"""

    def __init__(self, api_key: str, game_name: str, model_name: str = "", verbose: bool = False):
        """Initialize Mem0 client

        Args:
            api_key: Mem0 API key
            game_name: Game name (used as part of user_id for data isolation)
            model_name: LLM model name (used as part of user_id for data isolation)
            verbose: Whether to print debug information
        """
        from mem0 import MemoryClient   # Deferred import
        super().__init__(game_name, verbose)
        self.client = MemoryClient(api_key=api_key)
        self.model_name = model_name

        # Construct combined user_id: game_name_model_name
        if model_name:
            self.user_id = f"{game_name}_{model_name}"
        else:
            self.user_id = game_name

        if self.verbose:
            print(f"[Mem0] Initialized for game: {game_name}, user_id: {self.user_id}")

    def _normalize_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize filters to Mem0 API format with user_id

        Mem0 requires filters in the format:
        {
            "AND": [
                {"user_id": "game_name_model_name"},
                {"metadata": {"key": "value"}}
            ]
        }

        Args:
            filters: Input filters (e.g., {"metadata": {"role": "user"}} or {"role": "user"})

        Returns:
            Normalized filters with user_id and metadata combined using AND
        """
        # If filters already contain logical operators, assume it's properly formatted
        if "AND" in filters or "OR" in filters or "NOT" in filters:
            return filters

        # Extract metadata filter part
        metadata_filter = None

        if "metadata" in filters and isinstance(filters["metadata"], dict):
            # Already in {"metadata": {...}} format
            metadata_filter = filters
        else:
            # Simple format {"key": "value"}, wrap in metadata
            metadata_filter = {"metadata": filters}

        # Combine with user_id using AND operator
        return {
            "AND": [
                {"user_id": self.user_id},
                metadata_filter
            ]
        }

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add a memory to Mem0

        Args:
            content: The text content to store
            metadata: Optional metadata dictionary (e.g., role, step, node_id)

        Returns:
            Dictionary with memory ID and status
        """
        try:
            # Prepare messages for Mem0
            messages = [{"role": "user", "content": content}]

            # Ensure metadata always has a source tag for filtering
            if metadata is None:
                metadata = {}
            metadata["source"] = "game_agent"  # Add default filter tag

            # Add memory using the user_id (game_name_model_name)
            result = self.client.add(
                messages=messages,
                user_id=self.user_id,
                metadata=metadata
            )

            if self.verbose:
                print(f"[Mem0] Added memory: {content[:50]}...")
                print(f"[Mem0] Memory ID: {result.get('id', 'unknown')}")

            return {
                "success": True,
                "memory_id": result.get("id"),  # Keep backward compatibility
                "id": result.get("id"),          # New unified field
                "result": result
            }
        except Exception as e:
            print(f"[Mem0] Error adding memory: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def search_memories(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for relevant memories

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"metadata": {"role": "user"}} or {"role": "user"})
                    If not provided, uses default filter with user_id and source tag

        Returns:
            List of memory results with content and metadata
        """
        try:
            # Mem0 API requires filters with AND operator combining user_id and metadata
            # Format: {"AND": [{"user_id": "game_name_model_name"}, {"metadata": {"key": "value"}}]}
            if not filters:
                filters = {
                    "AND": [
                        {"user_id": self.user_id},
                        {"metadata": {"source": "game_agent"}}
                    ]
                }
            else:
                # Auto-normalize to correct format if needed
                filters = self._normalize_filters(filters)

            # Search memories (user_id is already in filters, no need to pass separately)
            response = self.client.search(
                query=query,
                top_k=top_k,
                filters=filters
            )

            # Handle API response format: {'results': [...]}
            if not response:
                return []

            # Extract results array from response
            if isinstance(response, dict):
                results = response.get('results', [])
            elif isinstance(response, list):
                # Fallback: if API returns list directly
                results = response
            else:
                return []

            if not results:
                return []

            # Format results for consistency with existing memory system
            formatted_results = []
            for result in results:
                # Skip non-dict results
                if not isinstance(result, dict):
                    continue

                formatted_results.append({
                    "id": result.get("id", ""),              # Unified field name
                    "text": result.get("memory", ""),
                    "score": result.get("score", 0.0),
                    "metadata": result.get("metadata", {}),
                    "created_at": result.get("created_at", ""),
                    "updated_at": result.get("updated_at", "")
                })

            if self.verbose and formatted_results:
                print(f"[Mem0] Search query: {query[:50]}...")
                print(f"[Mem0] Found {len(formatted_results)} results")

            return formatted_results
        except Exception as e:
            print(f"[Mem0] Error searching memories: {e}")
            return []

    def get_all_memories(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get all memories for this agent

        Args:
            filters: Optional metadata filters (e.g., {"metadata": {"role": "user"}} or {"role": "user"})
                    If not provided, uses default filter with user_id and source tag

        Returns:
            List of all memories with content and metadata
        """
        try:
            # Mem0 API requires filters with AND operator combining user_id and metadata
            # Format: {"AND": [{"user_id": "game_name_model_name"}, {"metadata": {"key": "value"}}]}
            if not filters:
                filters = {
                    "AND": [
                        {"user_id": self.user_id},
                        {"metadata": {"source": "game_agent"}}
                    ]
                }
            else:
                # Auto-normalize to correct format if needed
                filters = self._normalize_filters(filters)

            # Get all memories (user_id is already in filters, no need to pass separately)
            response = self.client.get_all(
                filters=filters
            )

            # Handle API response format: {'results': [...]}
            if not response:
                return []

            # Extract results array from response
            if isinstance(response, dict):
                results = response.get('results', [])
            elif isinstance(response, list):
                # Fallback: if API returns list directly
                results = response
            else:
                return []

            if not results:
                return []

            # Format results
            formatted_results = []
            for result in results:
                # Skip non-dict results
                if not isinstance(result, dict):
                    continue

                formatted_results.append({
                    "id": result.get("id", ""),              # Unified field name
                    "text": result.get("memory", ""),
                    "metadata": result.get("metadata", {}),
                    "created_at": result.get("created_at", ""),
                    "updated_at": result.get("updated_at", "")
                })

            if self.verbose and formatted_results:
                print(f"[Mem0] Retrieved {len(formatted_results)} total memories")

            return formatted_results
        except Exception as e:
            print(f"[Mem0] Error getting all memories: {e}")
            return []



    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a specific memory

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            Dictionary with success status
        """
        try:
            result = self.client.delete(memory_id=memory_id)

            if self.verbose:
                print(f"[Mem0] Deleted memory: {memory_id}")

            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            print(f"[Mem0] Error deleting memory: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def delete_all_memories(self) -> Dict[str, Any]:
        """Delete all memories for this game

        Returns:
            Dictionary with success status
        """
        try:
            result = self.client.delete_all(user_id=self.user_id)

            if self.verbose:
                print(f"[Mem0] Deleted all memories for user_id: {self.user_id}")

            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            print(f"[Mem0] Error deleting all memories: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update an existing memory

        Args:
            memory_id: The ID of the memory to update
            content: New content
            metadata: New metadata

        Returns:
            Dictionary with success status
        """
        try:
            result = self.client.update(
                memory_id=memory_id,
                text=content,
                metadata=metadata or {}
            )

            if self.verbose:
                print(f"[Mem0] Updated memory: {memory_id}")

            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            print(f"[Mem0] Error updating memory: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_memory_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent memory history

        Args:
            limit: Maximum number of memories to retrieve

        Returns:
            List of recent memories sorted by creation time
        """
        try:
            memories = self.get_all_memories()

            # Sort by created_at timestamp (most recent first)
            sorted_memories = sorted(
                memories,
                key=lambda x: x.get("created_at", ""),
                reverse=True
            )

            return sorted_memories[:limit]
        except Exception as e:
            print(f"[Mem0] Error getting memory history: {e}")
            return []

    def format_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for inclusion in LLM prompt

        Args:
            memories: List of memory dictionaries

        Returns:
            Formatted string for prompt inclusion
        """
        if not memories:
            return ""

        formatted_lines = []
        for i, mem in enumerate(memories, 1):
            text = mem.get("text", "")
            metadata = mem.get("metadata", {})

            # Include metadata if available
            meta_str = ""
            if metadata:
                meta_parts = []
                if "step" in metadata:
                    meta_parts.append(f"Step {metadata['step']}")
                if "node_id" in metadata:
                    meta_parts.append(f"Node: {metadata['node_id']}")
                if "role" in metadata:
                    meta_parts.append(f"Role: {metadata['role']}")

                if meta_parts:
                    meta_str = f" [{', '.join(meta_parts)}]"

            formatted_lines.append(f"{i}. {text}{meta_str}")

        return "\n".join(formatted_lines)

    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpoint

        Returns:
            Dictionary containing game_name, model_name and configuration
        """
        return {
            "game_name": self.game_name,
            "model_name": self.model_name,
            "verbose": self.verbose
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint

        Args:
            state: State dictionary from checkpoint
        """
        self.game_name = state.get("game_name", self.game_name)
        self.model_name = state.get("model_name", "")
        self.verbose = state.get("verbose", self.verbose)

        # Rebuild user_id
        if self.model_name:
            self.user_id = f"{self.game_name}_{self.model_name}"
        else:
            self.user_id = self.game_name

        if self.verbose:
            print(f"[Mem0] Restored state, user_id: {self.user_id}")