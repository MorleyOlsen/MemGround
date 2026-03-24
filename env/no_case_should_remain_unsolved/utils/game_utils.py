# env/no_case_should_remain_unsolved/utils/game_utils.py
"""Utility class for the No Case Should Remain Unsolved game"""
from __future__ import annotations

from typing import Any, Dict, Optional

from memground_agent.common.schemas import Observation
from memground_agent.env.base_game_utils import BaseGameUtils


class NoCaseGameUtils(BaseGameUtils):
    """Utility class for the No Case Should Remain Unsolved reasoning game

    Handles game-specific:
    - Game state context retrieval
    - Log data formatting
    - Action execution (keyword unlock, event reading, ordering submission, lock unlock)
    """

    def __init__(self):
        super().__init__()
        self.memory_store = None  # Memory store reference

    def set_memory_store(self, store):
        """Set memory store reference"""
        self.memory_store = store

    def set_env(self, env):
        """Set environment reference"""
        self.env = env

    def get_game_context(self, env: Any) -> Dict[str, Any]:
        """Get context information for the No Case Should Remain Unsolved game

        Args:
            env: NoCaseEnv game environment instance

        Returns:
            Context dictionary containing game state
        """
        game_context = {}

        # Get current memory
        if self.memory_store:
            game_context.update(self.memory_store.get_memory_context())

        # Get game state info
        if hasattr(env, 'get_state'):
            state = env.get_state()
            game_context['dust_state'] = {
                'current_node_id': state.get('current_node_id', ''),
                'keyword_pool': state.get('keyword_pool', []),
                'known_events': state.get('known_events', []),
                'event_pool': state.get('event_pool', []),
                'read_events': state.get('read_events', []),
                'locked_events': state.get('locked_events', {}),
                'score': state.get('score_points', 0),
                'keys': state.get('keys', 0),
                'character_orders': state.get('character_orders', {}),
                'order_judgements': state.get('order_judgements', []),
                'awarded_pairs': [list(pair) for pair in state.get('awarded_pairs', [])],
                'multi_lock_progress': state.get('multi_lock_progress', {}),
            }

            # Add order_gt for generating trace in prompt_builder
            if hasattr(env, 'order_gt'):
                game_context['order_gt'] = env.order_gt

            # Add lock_info for displaying lock questions in prompt_builder
            if hasattr(env, 'lock_info'):
                game_context['lock_info'] = env.lock_info

        return game_context

    def format_log_data(
        self,
        step: int,
        obs: Observation,
        decision: Any,
        game_context: Dict[str, Any],
        retrieval_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format log data for the No Case Should Remain Unsolved game

        Args:
            step: Current step number
            obs: Current observation
            decision: Decision object
            game_context: Game context
            retrieval_decision: Retrieval decision info (optional)

        Returns:
            Complete log data dictionary
        """
        # Extract decision info
        action_type = getattr(decision, 'choice_index', 0)  # choice_index stores action_type (0-4)

        # Parse action_params from choice_text
        import json
        action_params = {}
        try:
            choice_data = json.loads(decision.choice_text)
            action_params = choice_data.get("action_params", {})
        except (json.JSONDecodeError, AttributeError):
            pass

        rationale = getattr(decision, 'rationale', '')

        # Get game state
        dust_state = game_context.get('dust_state', {})

        # Build choices dict (for logging)
        choices = {
            "text": f"action_type={action_type}, params={action_params}",
            "decision_rationale": rationale
        }

        # Build log data (must include required fields for log_action)
        log_data = {
            # Standard required fields
            "step": step,
            "node_id": obs.meta.get('current_node_id', dust_state.get('current_node_id', '')),
            "node_name": obs.meta.get('current_node_id', dust_state.get('current_node_id', '')),
            "scene_text": obs.text if isinstance(obs.text, str) else str(obs.text),
            "choices": choices,
            "file_retrieval": retrieval_decision,

            # NoCaseEnv game-specific fields
            "action_type": action_type,
            "action_params": action_params,
            "current_node_id": dust_state.get('current_node_id', ''),
            "keyword_pool": dust_state.get('keyword_pool', []),
            "known_events": dust_state.get('known_events', []),
            "event_pool": dust_state.get('event_pool', []),
            "read_events": dust_state.get('read_events', []),
            "locked_events": dust_state.get('locked_events', {}),
            "score": dust_state.get('score', 0),
            "keys": dust_state.get('keys', 0),
            "character_orders": dust_state.get('character_orders', {}),
            "order_judgements": dust_state.get('order_judgements', []),
            "awarded_pairs": dust_state.get('awarded_pairs', []),
        }

        return log_data
   
    def execute_action(self, env: Any, decision: Any) -> bool:
        """Execute an action in the No Case Should Remain Unsolved game

        Args:
            env: NoCaseEnv game environment instance
            decision: Decision object containing choice_index (action type 0-4) and choice_text (action_params JSON)

        Returns:
            bool: Whether the action was executed successfully
        """
        import json

        action_type = decision.choice_index

        # Parse action_params from choice_text
        try:
            action_data = json.loads(decision.choice_text)
            action_params = action_data.get("action_params", {})
        except (json.JSONDecodeError, AttributeError):
            msg = f"Failed to parse action parameters: {decision.choice_text}"
            print(msg)
            if self.memory_store:
                self.memory_store.add_message(msg, role="system")
            return False

        try:
            action_success = False

            if action_type == 0:
                # unlock_keyword - unlock events using a keyword
                keyword = action_params.get('keyword', '')
                if not keyword:
                    return False
                newly_unlocked = env.apply_keyword_unlock(keyword)
                msg = f"Used keyword '{keyword}' to unlock {len(newly_unlocked)} event(s): {newly_unlocked}"
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                action_success = True

            elif action_type == 1:
                # read_event - read an event
                event_name = action_params.get('event_name', '')
                if not event_name:
                    return False
                env.read_event(event_name)
                msg = f"Read event '{event_name}'"
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                action_success = True

            elif action_type == 2:
                # submit_orders - submit character event ordering
                orders = action_params.get('orders', {})
                if not orders:
                    return False
                result = env.apply_orders(orders)

                # Collect occurrence times of scored event pairs and add to revealed set
                time_lines = []
                time_lookup = getattr(env, 'dialogue_time_lookup', {})
                time_revealed = getattr(env, 'time_revealed_events', None)
                for j in result.get('judgements', []):
                    if j.get('result') == 'correct':
                        earlier = j['earlier']
                        later = j['later']
                        # Add scored events to revealed set
                        if time_revealed is not None:
                            time_revealed.add(earlier)
                            time_revealed.add(later)
                        if earlier == later:
                            # Single event character
                            t = time_lookup.get(earlier, "")
                            time_lines.append(f"  {j['character']}: '{earlier}'" + (f" ({t})" if t else ""))
                        else:
                            t_e = time_lookup.get(earlier, "")
                            t_l = time_lookup.get(later, "")
                            time_lines.append(
                                f"  {j['character']}: '{earlier}'" + (f" ({t_e})" if t_e else "") +
                                f" → '{later}'" + (f" ({t_l})" if t_l else "")
                            )

                msg = f"Submitted orders: +{result['new_points']} point(s), earned {result['keys_earned']} key(s)"
                if time_lines:
                    msg += "\nTimestamps of scored event pairs:\n" + "\n".join(time_lines)
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                action_success = True

            elif action_type == 3:
                # unlock_with_key - unlock yellow lock using a key
                event_name = action_params.get('event_name', '')
                if not event_name:
                    return False
                success = env.unlock_by_key(event_name)
                msg = (f"Successfully unlocked event '{event_name}' with a key. Keys remaining: {env.keys}" if success
                       else f"Unlock failed: '{event_name}' (insufficient keys or not a yellow lock, current keys: {env.keys})")
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                action_success = success

            elif action_type == 4:
                # answer_lock - answer a question to unlock pink/purple lock
                event_name = action_params.get('event_name', '')
                answer = action_params.get('answer', '')
                if not event_name or not answer:
                    return False
                result = env.answer_lock(event_name, answer)

                if result["is_multi"]:
                    # Multi-event lock: submit one event name at a time
                    if result["already_submitted"]:
                        msg = (f"Dialogue name '{answer}' has already been submitted correctly"
                               f" ({result['progress']}/{result['total']}). Please submit another dialogue name.")
                        action_success = False
                    elif result["answer_correct"]:
                        if result["unlocked"]:
                            msg = f"All {result['total']} dialogue name(s) correct. Event '{event_name}' unlocked."
                        else:
                            msg = (f"Dialogue name '{answer}' is correct! "
                                   f"Found {result['progress']}/{result['total']}. "
                                   f"Please continue submitting the remaining dialogue names.")
                        action_success = True
                    else:
                        msg = (f"Dialogue name '{answer}' is incorrect "
                               f"(found {result['progress']}/{result['total']}). "
                               f"Please reconsider and submit another dialogue name.")
                        action_success = False
                else:
                    # Regular lock
                    if result["unlocked"]:
                        msg = f"Correct answer. Event '{event_name}' unlocked."
                        action_success = True
                    else:
                        msg = f"Incorrect answer. Failed to unlock event '{event_name}'."
                        action_success = False

                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")

            else:
                msg = f"Unknown action type: {action_type}"
                print(msg)
                if self.memory_store:
                    self.memory_store.add_message(msg, role="system")
                raise ValueError(msg)

            # Check if the current node has auto_link; if so, auto-navigate
            if action_success:
                self._handle_auto_link(env)

            return action_success

        except Exception as e:
            msg = f"Action execution failed: {e}"
            print(msg)
            if self.memory_store:
                self.memory_store.add_message(msg, role="system")
            return False

    def _handle_auto_link(self, env: Any) -> None:
        """Handle the auto-link logic for nodes

        Args:
            env: NoCaseEnv game environment instance
        """
        # Get current node
        current_node = env._get_node_by_name(env.current_node_id)
        if not current_node:
            return

        # Check for auto_link
        auto_link = current_node.get("auto_link", "")
        if not auto_link:
            return

        # Get target node
        target_node = env._get_node_by_name(auto_link)
        if not target_node:
            print(f"[Warning] Auto-link target node not found: {auto_link}")
            return

        # Get target node's sub_name; skip if None (nodes like talk-* without sub_name are not added to event pool)
        target_sub_name = target_node.get("sub_name") or None
        if not target_sub_name:
            return

        # Add to known events
        if target_sub_name not in env.known_events:
            env.known_events.add(target_sub_name)

        # Extract and add keywords
        emphasize = target_node.get("emphasize", [])
        if emphasize:
            for keyword in emphasize:
                # Check if keyword has already been used
                if keyword not in env.used_keywords:
                    env.keyword_pool.add(keyword)

        # Determine lock type
        lock_info = env._get_lock_info(auto_link)
        lock_type = lock_info.get("type", "none")

        # Handle according to lock type
        if lock_type == "none" or not lock_type:
            # No lock: add to readable event pool (skip if already read)
            if target_sub_name not in env.event_pool and target_sub_name not in env.read_events:
                env.event_pool.append(target_sub_name)

        else:
            # Has lock: add to corresponding lock pool
            lock_color = None
            if "yellow" in lock_type.lower():
                lock_color = "yellow"
            elif "pink" in lock_type.lower():
                lock_color = "pink"
            elif "purple" in lock_type.lower():
                lock_color = "purple"

            if lock_color and target_sub_name not in env.locked_events[lock_color]:
                env.locked_events[lock_color].add(target_sub_name)
         
                

    def retrieve_information(self, retrieval_decision: Dict[str, Any], config: Any = None) -> Optional[str]:
        """Retrieve information (NoCaseEnv: retrieve event key_info based on LLM decision)

        Args:
            retrieval_decision: Retrieval decision returned by policy.decide_retrieval()
                Contains: need_retrieval, filenames, reason, filters (optional)
            config: Agent configuration object (kept for interface consistency)

        Returns:
            Formatted retrieval result text
        """
        # Extract retrieval decision info
        need_retrieval = retrieval_decision.get("need_retrieval", False)
        filenames = retrieval_decision.get("filenames", [])
        filters = retrieval_decision.get("filters", {})

        if not need_retrieval or not filenames:
            return None

        # Ensure filenames is a list
        if not isinstance(filenames, list):
            return None

        # If using Mem agent, use its search functionality for retrieval
        if self.memory_store and hasattr(self.memory_store, 'use_mem') and self.memory_store.use_mem:
            if hasattr(self.memory_store, 'mem_agent') and self.memory_store.mem_agent:
                return self._retrieve_from_mem_agent(filenames, filters, config)

        # Otherwise, get event key_info from environment
        results = []
        for event_name in filenames:
            # Use environment's _get_node_by_name to get node
            node = self.env._get_node_by_name(event_name)
            if node:
                key_info = node.get("key_info", [])
                if key_info:
                    # Format output - reduce newlines
                    if isinstance(key_info, list):
                        # Join list elements with space instead of newlines
                        key_info_text = " ".join(key_info)
                    else:
                        key_info_text = str(key_info)
                    results.append(f"Event: {event_name} === {key_info_text}")
            else:
                results.append(f"Event: {event_name} === (Event not found)")

        # Join all results - use semicolons and spaces instead of newlines
        if results:
            return "; ".join(results)
        else:
            return None

    def _retrieve_from_mem_agent(self, event_names: list, filters: Dict[str, Any], config: Any) -> Optional[str]:
        """Retrieve event content from memory agent

        Args:
            event_names: List of event names to retrieve
            filters: Optional filter conditions
            config: Agent configuration object

        Returns:
            Formatted retrieval result text
        """
        mem_agent = self.memory_store.mem_agent

        # Combine all event names into a single query for one search call
        if not event_names:
            return None

        # Build query containing all event names
        events_str = " ".join(event_names)
        query = f"events {events_str} key_info"

        # Use filters if provided; otherwise filter by user role only (event info is usually stored as user messages)
        search_filters = filters if filters else {"role": "user"}

        # Retrieve from memory agent with more top_k to get all relevant events
        top_k = min(len(event_names) * 3, 20)  # Up to 3 results per event, max 20
        mem_results = mem_agent.search_memories(query, top_k=top_k, filters=search_filters)

        if not mem_results:
            return None

        # Format all results directly
        results = []
        for result in mem_results:
            text = result.get("text", "")
            results.append(text)

        # Join all results with semicolons and spaces
        if results:
            return "; ".join(results)
        else:
            return None

    def post_action_hook(self, decision: Any, action_success: bool = True, step: int = 0) -> None:
        """Hook called after action execution

        Args:
            decision: The executed decision
            action_success: Whether the action was executed successfully
            step: Current step number
        """
        # NoCaseEnv game currently needs no additional post-processing logic
        # All state updates are completed in NoCaseEnv methods
        pass

    def get_console_log_info(
        self,
        obs: Observation,
        search_results: str,
        decision: Any
    ) -> Optional[str]:
        """Get console log info for the No Case Should Remain Unsolved game

        Args:
            obs: Current observation
            search_results: Retrieval results
            decision: Decision object

        Returns:
            Formatted console log string
        """
        lines = []

        lines.append(f"=== No Case Should Remain Unsolved Reasoning Game - Step {obs.meta.get('step', 0)} ===")
        meta = obs.meta
        lines.append(f"Score: {meta.get('score', 0)} | Keys: {meta.get('keys', 0)}")
        lines.append(f"Known Events: {len(meta.get('known_events', []))} | "
                    f"Readable: {len(meta.get('event_pool', []))} | "
                    f"Read: {len(meta.get('read_events', []))}")
        action_type = getattr(decision, 'action_type', 'unknown')
        action_params = getattr(decision, 'action_params', {})
        lines.append(f"Action: {action_type}")
        lines.append(f"Params: {action_params}")
        lines.append(f"Rationale: {getattr(decision, 'rationale', 'N/A')}")

        return "\n".join(lines)

    def _compress_memory(self, count: int, llm_client, llm_config, prompt_builder) -> Optional[str]:
        """Compress the earliest n dialogue turns

        Args:
            count: Number of dialogue turns to compress
            llm_client: LLM client
            llm_config: LLM configuration
            prompt_builder: Prompt builder

        Returns:
            Compressed text, or None if failed
        """
        if not self.memory_store or len(self.memory_store._items) < count:
            return None

        # Get the earliest n memory items
        items_to_compress = self.memory_store._items[:count]

        # Build dialogue text list
        conversations = []
        for item in items_to_compress:
            role = item.meta.get("role", "user")
            conversations.append(f"[{role}] {item.text}")

        # Build compression prompt using prompt_builder
        compression_prompt = prompt_builder.build_compression_prompt(conversations)

        try:
            # Call LLM for compression
            response = llm_client.chat.completions.create(
                model=llm_config.model,
                messages=[{"role": "user", "content": compression_prompt}],
                temperature=0.3  # Use lower temperature for accuracy
            )

            compressed_text = response.choices[0].message.content or ""

            # Delete the earliest n memory items
            for _ in range(count):
                if len(self.memory_store._items) > 0:
                    self.memory_store._items.pop(0)

            # Add compressed content to memory (using add_message)
            self.memory_store.add_message(
                content=f"[Compressed Memory] {compressed_text}",
                role="system",
                compressed=True
            )

            # Move the newly added memory to the front
            if len(self.memory_store._items) > 0:
                compressed_item = self.memory_store._items.pop()
                self.memory_store._items.insert(0, compressed_item)

            return compressed_text
        except Exception as e:
            print(f"[Memory Compression] Compression failed: {e}")
            return None

    def manage_memory(self, config: Any, full_prompt: str = "", llm_client=None, llm_config=None, prompt_builder=None) -> None:
        """Manage memory to ensure the full prompt does not exceed the token limit

        Args:
            config: Agent configuration object
            full_prompt: Complete prompt text (optional; includes system_prompt + user_prompt + all other content)
            llm_client: LLM client (used for memory compression)
            llm_config: LLM configuration (used for memory compression)
            prompt_builder: Prompt builder (used to build the compression prompt)
        """
        if not self.memory_store:
            return

        # If using Mem agent, skip local memory management (Mem agent manages cloud memory automatically)
        if hasattr(self.memory_store, 'use_mem') and self.memory_store.use_mem:
            return

        # Check if compression is enabled and dialogue turns exceed threshold
        if (hasattr(config, 'enable_compression') and config.enable_compression and
            hasattr(config, 'compression_threshold') and hasattr(config, 'compression_count')):

            current_turns = len(self.memory_store._items)

            if current_turns > config.compression_threshold:
                if config.verbose:
                    print(f"[Memory Compression] Turn count {current_turns} exceeds threshold {config.compression_threshold}, starting compression...")

                # Compress the earliest n dialogue turns
                if llm_client and llm_config and prompt_builder:
                    compressed_text = self._compress_memory(config.compression_count, llm_client, llm_config, prompt_builder)

                    if compressed_text and config.verbose:
                        print(f"[Memory Compression] Successfully compressed {config.compression_count} turn(s)")
                        print(f"[Memory Compression] Compressed content length: {len(compressed_text)} chars")
                else:
                    if config.verbose:
                        print(f"[Memory Compression] Warning: LLM client or prompt builder not provided, skipping compression")

        # If full prompt is provided, calculate token count directly
        if full_prompt:
            total_tokens = len(full_prompt) / 2.5  # approx 2.5 chars/token for mixed text
        else:
            # Otherwise use estimation (backward-compatible)
            total_tokens = self.memory_store.get_total_tokens_estimate()

        # If exceeding token limit, delete memory items
        max_context_tokens = config.max_context_tokens
        if total_tokens > max_context_tokens:
            excess_tokens = total_tokens - max_context_tokens
            delete_count = max(1, excess_tokens // 50)
            deleted = self.memory_store.delete_by_priority(delete_count)

            if deleted > 0 and config.verbose:
                print(f"[Context Management] Deleted {deleted} memory item(s) (assistant-priority)")
                print(f"  Prompt token estimate: {int(total_tokens)} (limit: {max_context_tokens})")

    def get_state(self) -> Dict[str, Any]:
        """Get game utility state for checkpoint

        Returns:
            State dictionary (NoCaseEnv GameUtils currently has no additional state)
        """
        return {}

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore game utility state from a checkpoint

        Args:
            state: Game utility state dictionary
        """
        # NoCaseEnv GameUtils currently has no additional state to restore
        pass
