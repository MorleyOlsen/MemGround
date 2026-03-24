# env/type_help/utils/game_utils.py
"""Utility class for the Type Help game"""
from __future__ import annotations

from typing import Any, Dict, Optional

from memground_agent.common.schemas import Observation
from memground_agent.env.base_game_utils import BaseGameUtils
from env.type_help.utils.file_retriever import TypeHelpFileRetriever


class TypeHelpGameUtils(BaseGameUtils):
    """Utility class for the Type Help game

    Handles Type Help game-specific:
    - File tracker information retrieval
    - Log data formatting
    - Read file list management (stores filenames and IDs only)
    """

    def __init__(self):
        super().__init__()
        self.memory_store = None  # Memory store reference

    def set_memory_store(self, store):
        """Set the memory store reference"""
        self.memory_store = store

    def set_env(self, env):
        """Set the environment reference"""
        self.env = env

    def get_read_files_text(self) -> str:
        """Get the text representation of the read files list (for inclusion in prompts)

        Returns:
            Formatted read files list string
        """
        if hasattr(self, 'env') and hasattr(self.env, 'file_tracker'):
            return self.env.file_tracker.get_read_files_text()
        return "No files have been read yet"

    def get_read_files_token_estimate(self, chars_per_token: float = 2.5) -> int:
        """Estimate the token count for the read files list

        Args:
            chars_per_token: Average characters per token

        Returns:
            Estimated token count
        """
        text = self.get_read_files_text()
        return int(len(text) / chars_per_token)

    def retrieve_information(self, retrieval_decision: Dict[str, Any], config: Any) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Retrieve information (Type Help game: open files based on LLM decision)

        Args:
            retrieval_decision: Retrieval decision returned by policy.decide_retrieval()
                Contains: need_retrieval, filenames, reason, filters (optional)
            config: Agent configuration object

        Returns:
            (Formatted retrieval result text, retrieval decision info dict)
            Retrieval decision info contains: need_retrieval, filenames, reason
        """
        # Extract retrieval decision fields
        need_retrieval = retrieval_decision.get("need_retrieval", False)
        filenames = retrieval_decision.get("filenames", [])
        filters = retrieval_decision.get("filters", {})

        if not need_retrieval or not filenames:
            return None

        # Ensure filenames is a list
        if not isinstance(filenames, list):
            return None

        # If using a Mem agent, use its search function to retrieve
        if self.memory_store and hasattr(self.memory_store, 'use_mem') and self.memory_store.use_mem:
            if hasattr(self.memory_store, 'mem_agent') and self.memory_store.mem_agent:
                return self._retrieve_from_mem_agent(filenames, filters, config)

        # Otherwise use the file retriever to fetch file content from the environment
        if not hasattr(self, 'env'):
            return None

        retriever = TypeHelpFileRetriever(self.env)
        file_results = retriever.retrieve_files(filenames)
        formatted_result = retriever.format_retrieved_files(file_results)

        return formatted_result

    def _retrieve_from_mem_agent(self, filenames: list, filters: Dict[str, Any], config: Any) -> Optional[str]:
        """Retrieve file content from the memory agent

        Args:
            filenames: List of filenames to retrieve
            filters: Optional filter conditions
            config: Agent configuration object

        Returns:
            Formatted retrieval result text
        """
        mem_agent = self.memory_store.mem_agent

        # Combine all filenames into a single query for a single search call
        if not filenames:
            return None

        # Build a query containing all filenames
        files_str = " ".join(filenames)
        query = f"files {files_str} content"

        # Use filters if provided; otherwise filter only for user role messages (file content is usually stored as user messages)
        search_filters = filters if filters else {"role": "user"}

        # Retrieve from the memory agent with extra top_k to cover all relevant files
        top_k = min(len(filenames) * 3, 20)  # At most 3 results per file, capped at 20
        results = mem_agent.search_memories(query, top_k=top_k, filters=search_filters)

        if not results:
            return None

        # Format all results directly
        all_results = []
        for result in results:
            text = result.get("text", "")
            all_results.append(text)

        # Concatenate all results
        if all_results:
            return "\n\n".join(all_results)
        else:
            return None


    def post_action_hook(self, decision: Any, action_success: bool = True, step: int = 0) -> None:
        """Hook called after action execution

        Args:
            decision: Executed decision
            action_success: Whether the action succeeded
            step: Current step number
        """
        # All logic has been moved to choose_by_filename in type_help_env.py
        # This method is kept for interface consistency
        pass

    def _compress_memory(self, count: int, llm_client, llm_config, prompt_builder) -> Optional[str]:
        """Compress the earliest n conversation turns

        Args:
            count: Number of conversation turns to compress
            llm_client: LLM client
            llm_config: LLM configuration
            prompt_builder: Prompt builder

        Returns:
            Compressed text, or None if compression fails
        """
        if not self.memory_store or len(self.memory_store._items) < count:
            return None

        # Get the earliest n memories
        items_to_compress = self.memory_store._items[:count]

        # Build the conversation text list
        conversations = []
        for item in items_to_compress:
            role = item.meta.get("role", "user")
            conversations.append(f"[{role}] {item.text}")

        # Use prompt_builder to build the compression prompt
        compression_prompt = prompt_builder.build_compression_prompt(conversations)

        try:
            # Call the LLM to compress
            response = llm_client.chat.completions.create(
                model=llm_config.model,
                messages=[{"role": "user", "content": compression_prompt}],
                temperature=0.3  # Use low temperature for accuracy
            )

            compressed_text = response.choices[0].message.content or ""

            # Delete the earliest n memories
            for _ in range(count):
                if len(self.memory_store._items) > 0:
                    self.memory_store._items.pop(0)

            # Add the compressed content to memory (using add_message)
            self.memory_store.add_message(
                content=f"[compress memory] {compressed_text}",
                role="system",
                compressed=True
            )

            # Move the newly added memory to the beginning
            if len(self.memory_store._items) > 0:
                compressed_item = self.memory_store._items.pop()
                self.memory_store._items.insert(0, compressed_item)

            return compressed_text
        except Exception as e:
            print(f"[Memory compression] Compression failed: {e}")
            return None

    def manage_memory(self, config: Any, full_prompt: str = "", llm_client=None, llm_config=None, prompt_builder=None) -> None:
        """Manage memory to ensure the complete prompt does not exceed the token limit

        Args:
            config: Agent configuration object
            full_prompt: Full prompt text (optional; includes system_prompt + user_prompt etc.)
            llm_client: LLM client (used for memory compression)
            llm_config: LLM configuration (used for memory compression)
            prompt_builder: Prompt builder (used for building compression prompts)
        """
        if not self.memory_store:
            return

        # If using a Mem agent, skip local memory management (Mem agent handles cloud memory automatically)
        if hasattr(self.memory_store, 'use_mem') and self.memory_store.use_mem:
            return

        # Check if compression is enabled and the conversation exceeds the threshold
        if (hasattr(config, 'enable_compression') and config.enable_compression and
            hasattr(config, 'compression_threshold') and hasattr(config, 'compression_count')):

            current_turns = len(self.memory_store._items)

            if current_turns > config.compression_threshold:
                if config.verbose:
                    print(f"[Memory compression] Conversation turns {current_turns} exceed threshold {config.compression_threshold}, starting compression...")

                # Compress the earliest n conversation turns
                if llm_client and llm_config and prompt_builder:
                    compressed_text = self._compress_memory(config.compression_count, llm_client, llm_config, prompt_builder)

                    if compressed_text and config.verbose:
                        print(f"[Memory compression] Successfully compressed {config.compression_count} turns")
                        print(f"[Memory compression] Compressed content length: {len(compressed_text)} characters")
                else:
                    if config.verbose:
                        print(f"[Memory compression] Warning: no LLM client or prompt builder provided, skipping compression")

        # If a full prompt is provided, compute token count directly
        if full_prompt:
            total_tokens = len(full_prompt) / 2.5  # ~2.5 chars/token for mixed text
        else:
            # Otherwise use estimation (backward compatibility)
            read_files_tokens = self.get_read_files_token_estimate()
            conversation_history_tokens = self.memory_store.get_total_tokens_estimate()
            total_tokens = read_files_tokens + conversation_history_tokens

        # If token limit exceeded, delete memories
        max_context_tokens = config.max_context_tokens
        if total_tokens > max_context_tokens:
            excess_tokens = total_tokens - max_context_tokens
            delete_count = max(1, excess_tokens // 50)
            deleted = self.memory_store.delete_by_priority(delete_count)

            if deleted > 0 and config.verbose:
                print(f"[Context management] Deleted {deleted} memories (assistant first)")
                print(f"  Prompt token estimate: {int(total_tokens)} (limit: {max_context_tokens})")

    def get_game_context(self, env: Any) -> Dict[str, Any]:
        """Get context information for the Type Help game

        Args:
            env: Type Help game environment instance

        Returns:
            Context dictionary containing memory, file tracker info, read file list, and character name info
        """
        game_context = {}

        # Get current memory
        if self.memory_store:
            game_context.update(self.memory_store.get_memory_context())

        # Get file tracker information
        if hasattr(env, 'get_file_tracker_info'):
            game_context['file_tracker_info'] = env.get_file_tracker_info()

        # Add text representation of the read files list
        game_context['read_files_text'] = self.get_read_files_text()

        # Add character name information
        if hasattr(env, 'get_character_names_text'):
            game_context['character_names_text'] = env.get_character_names_text()

        return game_context

    def format_log_data(
        self,
        step: int,
        obs: Observation,
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format log data for the Type Help game

        Args:
            step: Current step number
            obs: Current observation
            decision: Decision object
            game_context: Game context
            retrieval_decision: Retrieval decision info (contains need_retrieval, filenames, reason)

        Returns:
            Complete log data dictionary
        """
        # Format choices into the new format: {"text": filename, "decision_rationale": reason}
        choices = {
            "text": decision.choice_text if hasattr(decision, 'choice_text') else "",
            "decision_rationale": decision.rationale,
            "recall": decision.recall if hasattr(decision, 'recall') else []
        }

        # Get file tracker information
        file_tracker_info = game_context.get('file_tracker_info')
        unlocked_files = None
        attempted_files = None
        success_files = None
        failed_files = None
        hint_unlocked_files = None
        consecutive_failures = None
        if file_tracker_info:
            unlocked_files = file_tracker_info.get("unlocked_files", [])
            attempted_files = file_tracker_info.get("attempted_files", [])
            success_files = file_tracker_info.get("success_files", [])
            failed_files = file_tracker_info.get("failed_files", [])
            hint_unlocked_files = file_tracker_info.get("hint_unlocked_files", [])
            consecutive_failures = file_tracker_info.get("consecutive_failures", 0)

        # Build log data
        log_data = {
            "step": step,
            "node_id": obs.node_id,
            "node_name": obs.name,
            "scene_text": obs.text,
            "choices": choices,
            "file_retrieval": None,
            "unlocked_files": unlocked_files,
            "attempted_files": attempted_files,
            "success_files": success_files,
            "failed_files": failed_files,
            "hint_unlocked_files": hint_unlocked_files,
            "consecutive_failures": consecutive_failures,
        }

        # Add file retrieval decision information
        if retrieval_decision:
            log_data["file_retrieval"] = {
                "need_retrieval": retrieval_decision.get("need_retrieval", False),
                "opened_files": retrieval_decision.get("filenames", []),
                "reason": retrieval_decision.get("reason", "")
            }

        return log_data

    def execute_action(self, env: Any, decision: Any) -> bool:
        """Execute an action in the Type Help game

        Args:
            env: Type Help game environment instance
            decision: Decision object

        Returns:
            bool: Whether the action was executed successfully
        """
        # Type Help game uses filename selection
        if decision.choice_text:
            if hasattr(env, 'choose_by_filename'):
                return env.choose_by_filename(decision.choice_text)
            else:
                raise RuntimeError("Environment does not support choose_by_filename")
        else:
            raise ValueError("Type Help game requires choice_text in decision")

    def get_console_log_info(
        self,
        obs: Observation,
        search_results: str,
        decision: Any
    ) -> Optional[str]:
        """Get console log information for the Type Help game

        Args:
            obs: Current observation
            search_results: Retrieval results
            decision: Decision object

        Returns:
            Formatted console log string
        """
        import json

        lines = []
        lines.append(f"Node: {obs.node_id} ({obs.name})")
        lines.append(f"Text: {obs.text[:100]}...")

        # Display the LLM decision
        lines.append(f"Decision: {decision.choice_text if hasattr(decision, 'choice_text') else 'N/A'}")
        lines.append(f"Rationale: {decision.rationale}")

        # Display retrieval results
        if search_results:
            data = json.loads(search_results)
            results = data.get("results", [])
            if results:
                lines.append(f"Retrieved Memory ({len(results)} items):")
                for r in results[:3]:  # Only show the first 3
                    lines.append(f"  - {r['text'][:80]}...")

        return "\n".join(lines)

    def get_state(self) -> Dict[str, Any]:
        """Get game utils state for checkpoint

        Returns:
            State dictionary containing the read files list
        """
        read_files = []
        if hasattr(self, 'env') and hasattr(self.env, 'file_tracker'):
            read_files = self.env.file_tracker.get_read_files()
        return {
            "read_files": read_files
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore game utils state from checkpoint

        Args:
            state: Game utils state dictionary
        """
        if hasattr(self, 'env') and hasattr(self.env, 'file_tracker'):
            self.env.file_tracker.read_files = state["read_files"].copy()
            print(f"[GameUtils] State restored: {len(state['read_files'])} read files")
