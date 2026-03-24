# memground_agent/memory/amem.py
"""A-mem local memory management agent"""
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

from .base_mem_agent import BaseMemAgent
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'A-mem'))
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote


class AMemAgent(BaseMemAgent):
    """A-mem memory management agent

    Uses A-mem's AgenticMemorySystem for memory storage, retrieval, and management
    """

    def __init__(
        self,
        game_name: str = "game_agent",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        verbose: bool = False,
        api_key: Optional[str] = None,
        evo_threshold: int = 100
    ):
        """Initialize the A-mem agent

        Args:
            game_name: Game name, used as the data isolation identifier
            embedding_model: Embedding model name
            llm_backend: LLM backend (openai or ollama)
            llm_model: LLM model name
            verbose: Whether to print detailed logs
            api_key: LLM API key
            evo_threshold: Memory evolution threshold
        """
        super().__init__(game_name, verbose)

        self.embedding_model = embedding_model
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.api_key = api_key
        self.evo_threshold = evo_threshold

        if self.verbose:
            print(f"[A-mem] Initializing AgenticMemorySystem, game: {game_name}")

        # Initialize AgenticMemorySystem
        try:
            self.memory_system = AgenticMemorySystem(
                model_name=embedding_model,
                llm_backend=llm_backend,
                llm_model=llm_model,
                evo_threshold=evo_threshold,
                api_key=api_key
            )

            if self.verbose:
                print(f"[A-mem] AgenticMemorySystem initialized successfully")

        except ImportError as e:
            raise ImportError(f"Please install A-mem dependencies: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AgenticMemorySystem: {e}")

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add a memory to local storage

        Args:
            content: Memory content
            metadata: Metadata (e.g. role, step, source, category, tags, etc.)

        Returns:
            Dictionary containing success and memory_id
        """
        try:
            # Prepare parameters
            kwargs = {}
            extra_tags = []  # Used to store additional metadata fields

            if metadata:
                # Map metadata to AgenticMemorySystem parameters
                if "category" in metadata:
                    kwargs["category"] = metadata["category"]
                if "context" in metadata:
                    kwargs["context"] = metadata["context"]
                if "keywords" in metadata:
                    kwargs["keywords"] = metadata["keywords"]

                # Handle tags: preserve any tags already in metadata
                if "tags" in metadata:
                    if isinstance(metadata["tags"], list):
                        extra_tags.extend(metadata["tags"])
                    else:
                        extra_tags.append(str(metadata["tags"]))

                # Store other fields (e.g. role, step, node_id, name, etc.) as tags
                # Format: ["role:user", "step:1", "node_id:xxx"]
                reserved_fields = {"category", "tags", "context", "keywords"}
                for key, value in metadata.items():
                    if key not in reserved_fields:
                        extra_tags.append(f"{key}:{value}")

            # Set tags
            if extra_tags:
                kwargs["tags"] = extra_tags

            # Generate timestamp (YYYYMMDDHHmm format)
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            kwargs["timestamp"] = timestamp

            # Call AgenticMemorySystem.add_note
            memory_id = self.memory_system.add_note(
                content=content,
                time=timestamp,
                **kwargs
            )

            if self.verbose:
                print(f"[A-mem] Memory added successfully: {content[:50]}..., ID: {memory_id}")

            return {"success": True, "memory_id": memory_id}

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] Failed to add memory: {e}")
            return {"success": False, "error": str(e)}

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
            filters: Filter conditions (not yet used; kept for interface compatibility)

        Returns:
            List of memories, each containing id, text, score, metadata, etc.
        """
        try:
            # Use AgenticMemorySystem.search_agentic for searching
            search_results = self.memory_system.search_agentic(query, k=top_k)

            if self.verbose:
                print(f"[A-mem] Search query: {query}, found {len(search_results)} results")

            # Format results into a unified format
            formatted_results = []
            for result in search_results:
                # Parse additional fields from tags
                tags = result.get("tags", [])
                parsed_metadata = {}
                remaining_tags = []

                for tag in tags:
                    if isinstance(tag, str) and ":" in tag:
                        # Parse "key:value" format tags
                        key, value = tag.split(":", 1)
                        parsed_metadata[key] = value
                    else:
                        # Keep non-key-value format tags
                        remaining_tags.append(tag)

                # Build metadata
                metadata = {
                    "context": result.get("context", ""),
                    "keywords": result.get("keywords", []),
                    "tags": remaining_tags,  # Keep only non-key-value tags
                    "category": result.get("category", "Uncategorized"),
                    "timestamp": result.get("timestamp", ""),
                    "is_neighbor": result.get("is_neighbor", False),
                    **parsed_metadata  # Add extra fields parsed from tags (e.g. role, step, etc.)
                }

                formatted_results.append({
                    "id": result.get("id", ""),
                    "text": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "metadata": metadata,
                    "created_at": result.get("timestamp", ""),
                    "updated_at": result.get("timestamp", "")
                })

            return formatted_results

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] Failed to search memories: {e}")
            return []

    def get_all_memories(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get all memories

        Args:
            filters: Optional filter conditions

        Returns:
            List of memories
        """
        try:
            # Get all memories from AgenticMemorySystem
            all_memories = list(self.memory_system.memories.values())

            # Format results
            formatted_results = []
            for memory in all_memories:
                # Parse additional fields from tags
                tags = memory.tags
                parsed_metadata = {}
                remaining_tags = []

                for tag in tags:
                    if isinstance(tag, str) and ":" in tag:
                        # Parse "key:value" format tags
                        key, value = tag.split(":", 1)
                        parsed_metadata[key] = value
                    else:
                        # Keep non-key-value format tags
                        remaining_tags.append(tag)

                # Build complete result
                result = {
                    "id": memory.id,
                    "text": memory.content,
                    "metadata": {
                        "context": memory.context,
                        "keywords": memory.keywords,
                        "tags": remaining_tags,  # Keep only non-key-value tags
                        "category": memory.category,
                        "timestamp": memory.timestamp,
                        "last_accessed": memory.last_accessed,
                        "retrieval_count": memory.retrieval_count,
                        "links": memory.links,
                        **parsed_metadata  # Add extra fields parsed from tags (e.g. role, step, etc.)
                    },
                    "created_at": memory.timestamp,
                    "updated_at": memory.last_accessed
                }

                # Apply filters
                if filters:
                    match = True
                    for key, value in filters.items():
                        if key not in result["metadata"] or result["metadata"][key] != value:
                            match = False
                            break
                    if match:
                        formatted_results.append(result)
                else:
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] Failed to get all memories: {e}")
            return []

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
        try:
            # Prepare update parameters
            kwargs = {"content": content}
            extra_tags = []

            if metadata:
                # Map metadata to MemoryNote attributes
                if "category" in metadata:
                    kwargs["category"] = metadata["category"]
                if "context" in metadata:
                    kwargs["context"] = metadata["context"]
                if "keywords" in metadata:
                    kwargs["keywords"] = metadata["keywords"]

                # Handle tags: preserve any tags already in metadata
                if "tags" in metadata:
                    if isinstance(metadata["tags"], list):
                        extra_tags.extend(metadata["tags"])
                    else:
                        extra_tags.append(str(metadata["tags"]))

                # Store other fields (e.g. role, step, node_id, name, etc.) as tags
                reserved_fields = {"category", "tags", "context", "keywords"}
                for key, value in metadata.items():
                    if key not in reserved_fields:
                        extra_tags.append(f"{key}:{value}")

            # Set tags
            if extra_tags:
                kwargs["tags"] = extra_tags

            # Call AgenticMemorySystem.update
            success = self.memory_system.update(memory_id, **kwargs)

            if success:
                if self.verbose:
                    print(f"[A-mem] Memory updated successfully: {memory_id}")
                return {"success": True}
            else:
                return {"success": False, "error": "Memory not found"}

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] Failed to update memory: {e}")
            return {"success": False, "error": str(e)}

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a single memory

        Args:
            memory_id: Memory ID

        Returns:
            Dictionary containing success
        """
        try:
            # Call AgenticMemorySystem.delete
            success = self.memory_system.delete(memory_id)

            if success:
                if self.verbose:
                    print(f"[A-mem] Memory deleted successfully: {memory_id}")
                return {"success": True}
            else:
                return {"success": False, "error": "Memory not found"}

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] Failed to delete memory: {e}")
            return {"success": False, "error": str(e)}

    def delete_all_memories(self) -> Dict[str, Any]:
        """Delete all memories for this game

        Returns:
            Dictionary containing success
        """
        try:
            # Get all memory IDs
            memory_ids = list(self.memory_system.memories.keys())

            # Delete all memories
            for memory_id in memory_ids:
                self.memory_system.delete(memory_id)

            if self.verbose:
                print(f"[A-mem] All memories deleted successfully, total deleted: {len(memory_ids)}")

            return {"success": True}

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] Failed to delete all memories: {e}")
            return {"success": False, "error": str(e)}

    def get_state(self) -> Dict[str, Any]:
        """Get current memory state for persistence

        Returns:
            State dictionary containing all memory data
        """
        try:
            # Serialize all memories
            memories_state = []
            for _, memory in self.memory_system.memories.items():
                memories_state.append({
                    "id": memory.id,
                    "content": memory.content,
                    "keywords": memory.keywords,
                    "links": memory.links,
                    "retrieval_count": memory.retrieval_count,
                    "timestamp": memory.timestamp,
                    "last_accessed": memory.last_accessed,
                    "context": memory.context,
                    "evolution_history": memory.evolution_history,
                    "category": memory.category,
                    "tags": memory.tags
                })

            return {
                "game_name": self.game_name,
                "embedding_model": self.embedding_model,
                "llm_backend": self.llm_backend,
                "llm_model": self.llm_model,
                "evo_threshold": self.evo_threshold,
                "evo_cnt": self.memory_system.evo_cnt,
                "memories": memories_state,
                "total_memories": len(memories_state)
            }

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] Failed to get state: {e}")
            return {}

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore memories from a persisted state

        Args:
            state: State dictionary (from get_state)
        """
        try:
            # Restore base configuration
            self.game_name = state.get("game_name", self.game_name)
            self.embedding_model = state.get("embedding_model", self.embedding_model)
            self.llm_backend = state.get("llm_backend", self.llm_backend)
            self.llm_model = state.get("llm_model", self.llm_model)
            self.evo_threshold = state.get("evo_threshold", self.evo_threshold)

            # Restore evolution counter
            self.memory_system.evo_cnt = state.get("evo_cnt", 0)

            # Restore all memories
            memories_state = state.get("memories", [])
            for mem_data in memories_state:
                # Rebuild MemoryNote objects
                memory_note = MemoryNote(
                    content=mem_data["content"],
                    id=mem_data["id"],
                    keywords=mem_data.get("keywords", []),
                    links=mem_data.get("links", []),
                    retrieval_count=mem_data.get("retrieval_count", 0),
                    timestamp=mem_data.get("timestamp"),
                    last_accessed=mem_data.get("last_accessed"),
                    context=mem_data.get("context", "General"),
                    evolution_history=mem_data.get("evolution_history", []),
                    category=mem_data.get("category", "Uncategorized"),
                    tags=mem_data.get("tags", [])
                )

                # Add to the memory system
                self.memory_system.memories[memory_note.id] = memory_note

                # Rebuild the ChromaDB index
                metadata = {
                    "id": memory_note.id,
                    "content": memory_note.content,
                    "keywords": memory_note.keywords,
                    "links": memory_note.links,
                    "retrieval_count": memory_note.retrieval_count,
                    "timestamp": memory_note.timestamp,
                    "last_accessed": memory_note.last_accessed,
                    "context": memory_note.context,
                    "evolution_history": memory_note.evolution_history,
                    "category": memory_note.category,
                    "tags": memory_note.tags
                }
                self.memory_system.retriever.add_document(
                    memory_note.content,
                    metadata,
                    memory_note.id
                )

            if self.verbose:
                print(f"[A-mem] State restored successfully, {len(memories_state)} memories restored")

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] Failed to restore state: {e}")
            raise
