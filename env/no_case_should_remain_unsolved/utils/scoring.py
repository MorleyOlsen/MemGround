# env/no_case_should_remain_unsolved/utils/scoring.py
"""Ordering judgement and scoring system for the No Case Should Remain Unsolved game"""
from typing import Dict, List, Set, Tuple


def judge_character_orders(
    character_orders: Dict[str, List[str]],  # Model output ordering: character -> [event_name, ...]
    order_gt: List[Dict],  # Character order ground truth
    awarded_pairs: Set[Tuple[str, str, str]],  # Already-scored (character, earlier, later) pairs
) -> Tuple[List[Dict], int, Set[Tuple[str, str, str]]]:
    """Batch-judge the correctness of character event orderings and compute scores

    Args:
        character_orders: Model output event ordering for each character
        order_gt: Character order ground truth list
        awarded_pairs: Set of already-scored event pairs (to prevent duplicate scoring)
                      For single-event characters, use (character, event, event) format

    Returns:
        (order_judgements, new_points, updated_awarded_pairs)
        - order_judgements: List of judgement results; each element contains {character, earlier, later, result}
        - new_points: Newly earned points
        - updated_awarded_pairs: Updated set of already-scored event pairs
    """
    order_judgements = []
    new_points = 0
    current_awarded_pairs = awarded_pairs.copy()

    # Use build_trace_structure to build the trace data structure
    trace = build_trace_structure(character_orders, order_gt)

    for character, data in trace.items():
        events = data["events"]
        gt_numbers = data["gt_numbers"]

        # Special case: only score when character is "Eden Kindergarten" with exactly one submitted event
        if character == "Eden Kindergarten" and len(events) == 1:
            event = events[0]
            gt_num = gt_numbers[0]
            single_key = (character, event, event)

            # Check if already scored
            if single_key in current_awarded_pairs:
                order_judgements.append({
                    "character": character,
                    "earlier": event,
                    "later": event,
                    "result": "ignored"
                })
                continue

            # Check if the event belongs to this character (gt_num != -1)
            if gt_num != -1:
                # Event belongs to this character; award points
                order_judgements.append({
                    "character": character,
                    "earlier": event,
                    "later": event,
                    "result": "correct"
                })
                new_points += 1
                current_awarded_pairs.add(single_key)
            else:
                # Event does not belong to this character
                order_judgements.append({
                    "character": character,
                    "earlier": event,
                    "later": event,
                    "result": "incorrect"
                })
            continue

        # For other characters with only one event, no scoring (skip)
        if len(events) == 1:
            continue

        # For each character's event sequence, judge all adjacent event pairs
        for i in range(len(events) - 1):
            earlier_event = events[i]
            later_event = events[i + 1]
            earlier_num = gt_numbers[i]
            later_num = gt_numbers[i + 1]

            # Check if already scored
            pair_key = (character, earlier_event, later_event)
            if pair_key in current_awarded_pairs:
                # Already scored; mark as ignored
                order_judgements.append({
                    "character": character,
                    "earlier": earlier_event,
                    "later": later_event,
                    "result": "ignored"
                })
                continue

            # Judge ordering correctness
            if earlier_num == -1 or later_num == -1:
                result = "unknown"  # Some event does not belong to this character; cannot judge
            elif later_num - earlier_num == 1:
                result = "correct"  # Correct order and consecutive
            elif later_num > earlier_num:
                result = "correct_not_consecutive"  # Correct order but not consecutive
            else:
                result = "incorrect"  # Wrong order

            order_judgements.append({
                "character": character,
                "earlier": earlier_event,
                "later": later_event,
                "result": result
            })

            # If correct and consecutive, score and record
            if result == "correct":
                new_points += 1
                current_awarded_pairs.add(pair_key)

    return order_judgements, new_points, current_awarded_pairs


def calculate_keys_earned(score: int, key_threshold: int) -> int:
    """Calculate the number of keys earned based on score

    Args:
        score: Current score
        key_threshold: Score required to earn one key

    Returns:
        Total keys that should be earned
    """
    if key_threshold <= 0:
        return 0
    return score // key_threshold


def can_unlock_with_key(event_name: str, keys: int, lock_type_map: Dict[str, str]) -> bool:
    """Check whether a specified event can be unlocked with a key

    Args:
        event_name: Event name
        keys: Current number of keys held
        lock_type_map: Mapping of event name to lock type

    Returns:
        True if the event can be unlocked (yellow lock and has keys), False otherwise
    """
    lock_type = lock_type_map.get(event_name, "none")
    return lock_type == "yellow" and keys > 0


def build_trace_structure(
    character_orders: Dict[str, List[str]],
    order_gt: List[Dict]
) -> Dict[str, Dict[str, List]]:
    """Build an event trace data structure recording whether each event belongs to a character and its GT position

    Args:
        character_orders: Model output event ordering for each character {character_name: [event1, event2, ...]}
        order_gt: Character order ground truth list

    Returns:
        Trace data structure in the format:
        {
            "character_name": {
                "events": ["event1", "event2", ...],
                "gt_numbers": [number1, number2, ...]  # -1 means does not belong to this character
            }
        }
    """
    # Build mapping from character name to character data
    character_gt_map = {}
    for char_data in order_gt:
        char_name = char_data.get("name", "")
        if char_name:
            character_gt_map[char_name] = char_data

    trace = {}

    for character, event_sequence in character_orders.items():
        # Get this character's ground truth data
        char_data = character_gt_map.get(character)

        # Build mapping from event name to order number
        event_order_map = {}
        if char_data:
            dialogues = char_data.get("dialogue", [])
            for dialogue in dialogues:
                event_name = dialogue.get("name", "")
                event_number = dialogue.get("number")
                if event_name and event_number is not None:
                    event_order_map[event_name] = event_number

        # Get GT number for each event in this character's sequence (-1 if not belonging)
        gt_numbers = []
        for event_name in event_sequence:
            gt_number = event_order_map.get(event_name, -1)
            gt_numbers.append(gt_number)

        trace[character] = {
            "events": event_sequence,
            "gt_numbers": gt_numbers
        }

    return trace


def describe_trace(trace: Dict[str, Dict[str, List]], lang: str = "en") -> str:
    """Translate a trace data structure into readable text

    Args:
        trace: Trace data structure returned by build_trace_structure
        lang: Language, "ch" or "en"

    Returns:
        Formatted text description
    """
    if lang == "en":
        label_belongs = "belongs to this character"
        label_not_belongs = "does not belong to this character"
        label_unknown = "cannot determine, event does not belong to this character"
        label_correct = "correct order and consecutive"
        label_correct_nc = "correct order but not consecutive"
        label_incorrect = "incorrect order"
    else:
        label_belongs = "belongs to this character"
        label_not_belongs = "does not belong to this character"
        label_unknown = "cannot determine, event does not belong to this character"
        label_correct = "correct order and consecutive"
        label_correct_nc = "correct order but not consecutive"
        label_incorrect = "incorrect order"

    lines = []

    for character, data in trace.items():
        events = data["events"]
        nums = data["gt_numbers"]

        lines.append(f"{character}:")

        # List each event and its number
        for ev, n in zip(events, nums):
            if n == -1:
                lines.append(f"  {ev}: {label_not_belongs}")
            else:
                lines.append(f"  {ev}: {label_belongs}")  # Outputting n would tell the model which event number this is

        # Judge adjacent pairs
        for i in range(len(nums) - 1):
            a, b = nums[i], nums[i+1]

            if a == -1 or b == -1:
                status = label_unknown
            elif b - a == 1:
                status = label_correct
            elif b > a:
                status = label_correct_nc
            else:
                status = label_incorrect

            lines.append(f"  {events[i]} -> {events[i+1]} : {status}")

        lines.append("")  # Empty line to separate different characters

    return "\n".join(lines)
