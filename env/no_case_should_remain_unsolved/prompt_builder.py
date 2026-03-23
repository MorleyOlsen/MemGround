# env/no_case_should_remain_unsolved/prompt_builder.py
"""No Case Should Remain Unsolved reasoning game Prompt builder"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from galagent.common.schemas import Observation
from galagent.env.base_prompt_builder import BasePromptBuilder
from env.no_case_should_remain_unsolved.utils.scoring import build_trace_structure, describe_trace


def _format_awarded_pairs(awarded_pairs):
    """Format already-scored ordering pairs grouped by character into readable text"""
    if not awarded_pairs:
        return ""

    by_char = {}
    for pair in awarded_pairs:
        char, earlier, later = pair[0], pair[1], pair[2]
        by_char.setdefault(char, []).append((earlier, later))

    lines = []
    for char, pairs in by_char.items():
        lines.append(f"{char}:")
        for earlier, later in pairs:
            if earlier == later:
                lines.append(f"  - {earlier} (single event, scored)")
            else:
                lines.append(f"  - {earlier} -> {later}")
        lines.append("")
    return "\n".join(lines)


def _format_judgements_history(order_judgements):
    """Format historical ordering judgements grouped by character into readable text"""
    result_map = {
        "correct": "correct and consecutive (scored)",
        "correct_not_consecutive": "correct order but not consecutive (not scored)",
        "incorrect": "wrong order",
        "unknown": "cannot judge, some events don't belong to this character",
        "ignored": "already scored (skipped)"
    }

    by_char = {}
    for j in order_judgements:
        char = j["character"].strip()
        by_char.setdefault(char, {})
        pair_key = (j["earlier"], j["later"])
        by_char[char][pair_key] = j["result"]

    lines = []
    for char, pair_map in by_char.items():
        lines.append(f"{char}:")
        for (earlier, later), result in pair_map.items():
            label = result_map.get(result, result)
            if earlier == later:
                lines.append(f"  {earlier}: {label}")
            else:
                lines.append(f"  {earlier} -> {later}: {label}")
        lines.append("")
    return "\n".join(lines)


class NoCasePromptBuilder(BasePromptBuilder):
    """No Case Should Remain Unsolved reasoning game Prompt builder"""

    def __init__(self, goal_instruction: str = "", test_language: str = "en", show_order_judgements_history: bool = True):
        super().__init__(goal_instruction)
        self.test_language = test_language
        self.show_order_judgements_history = show_order_judgements_history
        self.goal_instruction = "Unlock all events and reconstruct the complete story with as few attempts as possible through reasoning and ordering"

    def build_system_prompt(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        return self._build_system_prompt_en(game_context)

    def _build_system_prompt_en(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        base_prompt = f"""You are a reasoning game agent playing a puzzle game called Dust.

            Game Objective: {self.goal_instruction}

            Game Mechanics:
            1. Keyword Discovery: When reading event text for the first time, keywords hidden within will be automatically discovered and added to your keyword pool.
            2. Event Unlocking: Use keywords to unlock new events associated with that keyword. After unlocking, you'll know the event name but need to actively read it to get the full content.
            3. Event Reading:
               - Unread Events: Select events from the unread events list for first-time reading to get full content and discover keywords.
               - Read Events: You can re-read any event from the read events list to review their content.
            4. Character Event Ordering: Each event involves multiple characters. You need to infer the chronological order of events from each character's perspective. Submitting correct orderings earns points.
            5. Scoring and Keys: For each correctly ordered event pair (an "earlier-later" relationship from a character's perspective that are consecutive), you earn 1 point. Accumulating a certain score automatically gives you a key. Already scored event pairs won't be scored again.
            6. Lock Mechanism:
            - Pink and Purple locks: Unlock by answering questions. Pink locks fall into two types:
              - Single-answer lock: Answer according to what the question asks.
              - Multi-event lock: The question specifies that multiple event names are required and states the count (e.g. the "Do you really think it was" event requires submitting 3 dialogue names). Use action_type==4 to submit **one** event name at a time as the answer; the system will tell you if it's correct. The lock only opens after all correct names are submitted.
            - Yellow lock: Unlock by consuming 1 key. Finally unlock the "You're Not Reflecting" node.

            Strategy Suggestions (in priority order):
            1. Prioritize using keywords to unlock new events: When keywords are available, use them to unlock new events and expand explorable content.
            2. Prioritize reading unread events: Read events from the unread events list to extract keywords and character information.
            3. Only try to unlock important yellow-locked events when you have keys.
            4. When you have sufficient information, answer questions to unlock pink/purple locks: Infer answers based on read events.
            5. Character Event Ordering: Each event involves multiple characters. You need to infer the chronological order of events from each character's perspective. Submitting correct orderings earns points:
               - Result "cannot judge" means the event does not belong to that character.
               - Result "wrong order" means the order is incorrect.
               - Result "correct but not consecutive" means there are events in between.
            6. You can select events from the read events list to re-read if you need to review details.

            Important Notes:
            - Nodes starting with "talk-" (e.g. "talk-1", "talk-2") contain no important reasoning information, do not participate in character event ordering, and it's not recommended to read them repeatedly. They should not influence your judgment on event attribution or ordering. The content of dialogue nodes must not be used as a basis for answering pink or purple lock questions — answers must come from formal event nodes. If the current node is a dialogue node, you can choose other actions.
            - Hint: There is only one event under the Eden Kindergarten character.
            """

        if game_context and 'dust_state' in game_context:
            state = game_context['dust_state']
            base_prompt += f"\n\nCurrent Game State:"
            base_prompt += f"\n- Keys: {state.get('keys', 0)}"
            base_prompt += f"\n- Available Keywords ({len(state.get('keyword_pool', []))}): {sorted(state.get('keyword_pool', []))}"
            base_prompt += f"\n- Unread Events ({len(state.get('event_pool', []))}): {state.get('event_pool', [])}"
            base_prompt += f"\n- Read Events ({len(state.get('read_events', []))}): {sorted(state.get('read_events', []))}"

            locked_events = state.get('locked_events', {})
            if any(locked_events.values()):
                base_prompt += f"\n- Locked Events:"

                lock_info_list = game_context.get('lock_info', [])
                multi_lock_progress = state.get('multi_lock_progress', {})

                for lock_type in ['pink', 'purple', 'yellow']:
                    events = locked_events.get(lock_type, [])
                    if events:
                        if lock_type in ['pink', 'purple']:
                            base_prompt += f"\n  - {lock_type} (requires answering question):"
                            for event_name in sorted(events):
                                question = ""
                                answer_field = ""
                                for lock_info in lock_info_list:
                                    if lock_info.get('sub_name') == event_name or lock_info.get('name') == event_name:
                                        question = lock_info.get('question', '')
                                        answer_field = lock_info.get('answer', '')
                                        break

                                line = f"\n    * {event_name}"
                                if question:
                                    line += f": {question}"
                                if isinstance(answer_field, list):
                                    found = list(multi_lock_progress.get(event_name, []))
                                    total = len(answer_field)
                                    line += f" (found {len(found)}/{total}"
                                    if found:
                                        line += f", correct so far: {found}"
                                    line += ")"
                                base_prompt += line
                        else:
                            base_prompt += f"\n  - {lock_type} (requires key): {sorted(events)}"

            character_orders = state.get('character_orders', {})
            order_gt = game_context.get('order_gt', [])

            if character_orders and order_gt:
                trace = build_trace_structure(character_orders, order_gt)
                trace_description = describe_trace(trace, lang="en")
                base_prompt += f"\n\n**Current Submitted Ordering Analysis**:\n{trace_description}"

            if self.show_order_judgements_history:
                order_judgements = state.get('order_judgements', [])
                if order_judgements:
                    base_prompt += f"\n\n**Historical Ordering Judgements (all submissions)**:\n"
                    base_prompt += _format_judgements_history(order_judgements)

            awarded_pairs = state.get('awarded_pairs', [])
            if awarded_pairs:
                base_prompt += f"\n\n**Correctly Scored Ordering Pairs (by character)**:\n"
                base_prompt += _format_awarded_pairs(awarded_pairs)

        return base_prompt

    def build_user_prompt(
        self,
        obs: Observation,
        retrieved_hits: List[str],
        game_context: Optional[Dict[str, Any]] = None
    ) -> str:
        return self._build_user_prompt_en(obs, retrieved_hits, game_context)

    def _build_user_prompt_en(self, obs: Observation, retrieved_hits: List[str], game_context: Optional[Dict[str, Any]] = None) -> str:
        conversation_history = ""
        if game_context and 'conversation_history' in game_context:
            conversation_history = game_context['conversation_history']

        prompt = f"""
        Historical Memory:
        {conversation_history if conversation_history else "(None)"}
        """

        if retrieved_hits:
            prompt += f"\nRetrieved Relevant Information:\n"
            prompt += "\n".join([f"{hit}" for hit in retrieved_hits])

        prompt += """

        Please select the next action based on the current game state. Output format (strictly follow this JSON format):
        {
        "action_type": <action type number: integer from 0-4>,
        "action_params": {
            // Fill in different parameters based on action type:
            // 0 (unlock_keyword - use keyword to unlock events): {"keyword": "keyword"}
            // 1 (read_event - read an event): {"event_name": "event name"}
            // 2 (submit_orders - submit character event ordering): {"orders": {"character1": ["event1", "event2", ...], "character2": [...]}}
            // 3 (unlock_with_key - unlock yellow lock with key): {"event_name": "event name"}
            // 4 (answer_lock - answer question to unlock pink/purple lock): {"event_name": "event name", "answer": "answer"}
        },
        "rationale": "<detailed explanation of your reasoning process and why you chose this action>"
        }

        Notes:
        - submit_orders orders parameter format: Each character corresponds to an event list, events in the list are arranged in chronological order from that character's perspective (earliest to latest)
        - Only submit event pairs where you have sufficient information to determine the order; don't submit uncertain ones
        - answer_lock answers need to be inferred from information in the read events
        """

        return prompt.strip()

    def build_retrieval_prompt(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        return self._build_retrieval_prompt_en(obs, game_context)

    def _build_retrieval_prompt_en(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        read_events = []
        conversation_history=""
        if game_context and 'dust_state' in game_context:
            read_events = game_context['dust_state'].get('read_events', [])
            conversation_history = game_context['conversation_history']

        if not read_events:
            return ""

        prompt = f"""
        Historical Memory: {conversation_history}
        You have already read the following events: {sorted(read_events)}

        If you need to review the detailed content of certain events to help with reasoning, you can request to retrieve these events.

        Output format (strictly follow this JSON format):
        {{
        "need_retrieval": true/false,
        "filenames": ["event name1", "event name2", ...],
        "filters": {{"role": "user"}},
        "reason": "<brief explanation of why you need to view these events>"
        }}

        Notes:
        - If retrieval is not needed, set need_retrieval to false and filenames to an empty list
        - Event names in filenames must come from the read events list, don't retrieve events that already exist in memory
        - Minimize retrieval content, only retrieve events helpful for current reasoning, don't retrieve all events
        - filters is optional, use simple key-value format like {{"role": "user"}} or {{"step": "2"}}
        - System will automatically combine your filter with user_id
        - If no specific filtering is needed, omit the filters field or set it to {{}}
        """

        return prompt.strip()

    def build_compression_prompt(self, conversations: List[str]) -> str:
        return self._build_compression_prompt_en(conversations)

    def _build_compression_prompt_en(self, conversations: List[str]) -> str:
        conversations_text = "\n\n".join(conversations)

        prompt = f"""
                You are a reasoning game agent playing a mystery puzzle game called Dust.
                Game mechanics:
                1. Keyword discovery: When reading event texts, hidden keywords (tags) are automatically discovered and added to your keyword pool.
                2. **Event unlocking**: Using keywords can unlock new events associated with that keyword. After unlocking, you will know the event name but must actively read it to get the full content.
                3. **Event reading**: Select an event from the readable event pool to read, gaining the full narrative and key information of that event.
                4. **Character event ordering**: Each event involves multiple characters. You need to infer the order in which events occurred from each character's perspective. Submitting correct orderings earns points.
                5. **Scoring and keys**: Each correctly ordered pair of events (an "earlier-later" relationship from a character's perspective) earns 1 point. Accumulating enough points automatically grants a key. Already-scored event pairs will not be scored again.
                6. **Lock mechanics**:
                - **Pink lock** and **Purple lock**: Unlocked by answering questions
                - **Yellow lock**: Costs 1 key to unlock
                Compress and summarize the following conversation history, retaining only the key information you consider helpful for reasoning about character-event associations and their chronological order.
                Conversation content: {conversations_text}
                """

        return prompt
