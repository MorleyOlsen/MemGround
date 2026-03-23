# galagent/env/no_case_should_remain_unsolved_env.py
"""No Case Should Remain Unsolved reasoning game environment"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple
from pathlib import Path
import base64

from galagent.common.schemas import Observation, Choice, Memory, Character
from galagent.env.base_env import BaseGameEnv, GameConfig
from env.no_case_should_remain_unsolved.utils.node_index import build_node_indices, get_events_by_tag, get_node_characters
from env.no_case_should_remain_unsolved.utils.scoring import judge_character_orders, calculate_keys_earned, can_unlock_with_key


@dataclass
class NoCaseConfig(GameConfig):
    """No Case Should Remain Unsolved game configuration"""
    data_path: Path = Path("dataset/no_case_should_remain_unsolved-en/data")
    game_type: str = "no_case_should_remain_unsolved"
    start_node_id: str = "start"
    key_threshold: int = 5  # Score required to earn one key
    ocr_appcode: str = "244a6973290f4311b061fc1b4969aa4c"  # Alibaba Cloud OCR APPCODE TODO: move to config.yaml later
    test_language: str = "en"  # en (English prompts)


class NoCaseEnv(BaseGameEnv):
    """No Case Should Remain Unsolved reasoning game environment

    Core mechanics:
    1. Keyword discovery: Identify keywords from text to form a keyword pool
    2. Event unlocking: Use keywords to unlock event names (without full content)
    3. Event reading: Select unlocked events to read and get full content
    4. Character event ordering: Determine the order of events from each character's perspective
    5. Scoring and keys: Earn points for correct ordering; accumulate points to earn keys
    6. Lock mechanism: pink/purple locks require answering questions; yellow locks require spending keys
    """

    def __init__(self, config: NoCaseConfig):
        super().__init__(config)

        # Raw data
        self.raw_nodes: List[Dict] = []  # Raw complete node data

        # Data indices (from nodes.json)
        self.nodes: List[Dict] = []  # Node list: [{"name": "step_1","sub_name":"xxxxxx", "emphasize": [...], "type": "...", "auto_link": "...", "key_info": [...]}]
        self.tag_index: Dict[str, List[str]] = {}  # keyword/tag -> event_name list
        self.lock_info: List[Dict] = []  # Lock info list: [{"name": "A", "type": "...", "question": "...", "answer": "..."}]

        # Character order ground truth (from order_gt.json)
        self.order_gt: List[Dict] = []  # Character list: [{"name": "character name", "tag": "character tag", "dialogue": [{"name": "event name", "number": order}]}]
        self.character_names: List[str] = []  # List of all character names
        self.character_tags: Dict[str, str] = {}  # Character name -> character tag mapping

        # Game state (needs to be persisted)
        self.keyword_pool: Set[str] = set()  # Discovered keywords
        self.used_keywords: Set[str] = set()  # Keywords already used (for deduplication)
        self.known_events: Set[str] = set()  # Known event names (includes unlocked, read, and locked pool)
        self.event_pool: List[str] = []  # Readable event name pool
        self.read_events: Set[str] = set()  # Read event names
        self.locked_events: Dict[str, Set[str]] = {
            "pink": set(),
            "purple": set(),
            "yellow": set()
        }  # Locked event name pools (by bucket)

        self.character_orders: Dict[str, List[str]] = {}  # Event order list for each character
        self.order_judgements: List[Dict] = []  # List of ordering judgement results
        self.score_points: int = 0  # Current score
        self.keys: int = 0  # Current key count
        self.awarded_pairs: Set[Tuple[str, str, str]] = set()  # Already-scored (character, earlier, later) pairs
        self.submitted_pairs: Set[Tuple[str, str, str]] = set()  # All submitted (character, earlier, later) pairs (to prevent duplicate records)

        # dialogue.json text lookup table (event name -> full text)
        self.dialogue_lookup: Dict[str, str] = {}
        # dialogue.json time lookup table (event name -> event time)
        self.dialogue_time_lookup: Dict[str, str] = {}
        # Set of events whose times have been revealed through ordering scores
        self.time_revealed_events: Set[str] = set()
        # Submission progress for multi-event locks (lock event name -> set of correctly submitted event names)
        self.multi_lock_progress: Dict[str, Set[str]] = {}

        # Load game data
        self.load_game_data()

    def load_game_data(self) -> None:
        """Load game data and build indices"""
        data_file = self.config.data_path / "nodes.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Game data not found: {data_file}")

        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Save raw node data
        self.raw_nodes = data.get("game", {}).get("nodes", [])

        # Build indices
        self.nodes, self.tag_index, self.lock_info = build_node_indices(self.raw_nodes)

        # Load character order ground truth (English version)
        order_gt_file = self.config.data_path / "order_gt-en.json"
        if order_gt_file.exists():
            with open(order_gt_file, 'r', encoding='utf-8') as f:
                order_gt_data = json.load(f)
            self.order_gt = order_gt_data.get("text", [])

            # Extract character names and tags
            for character in self.order_gt:
                char_name = character.get("name", "")
                char_tag = character.get("tag", "")
                if char_name:
                    self.character_names.append(char_name)
                    if isinstance(char_tag, str):
                        self.character_tags[char_name] = char_tag
                    elif isinstance(char_tag, list):
                        # If tag is a list, take the first one or join all
                        self.character_tags[char_name] = ", ".join(char_tag)

            # Initialize empty ordering lists for all characters
            for char_name in self.character_names:
                self.character_orders[char_name] = []
            print(f"[NoCaseEnv] Loaded {order_gt_file.name}: {len(self.character_names)} characters")
        else:
            print(f"[Warning] Character order file not found: {order_gt_file}")

        # Load dialogue.json text lookup table (English version)
        dialogue_file = self.config.data_path / "dialogue-en.json"
        if dialogue_file.exists():
            with open(dialogue_file, 'r', encoding='utf-8') as f:
                dialogue_data = json.load(f)
            # Normalize name keys with .strip() to prevent lookup failures from trailing spaces
            self.dialogue_lookup = {
                entry["name"].strip(): entry["text"]
                for entry in dialogue_data.get("text", [])
                if "name" in entry and "text" in entry
            }
            self.dialogue_time_lookup = {
                entry["name"].strip(): entry["time"]
                for entry in dialogue_data.get("text", [])
                if "name" in entry and "time" in entry
            }
            print(f"[NoCaseEnv] Loaded {dialogue_file.name}: {len(self.dialogue_lookup)} entries")
        else:
            print(f"[Warning] Dialogue file not found: {dialogue_file}")

        # Initialize start events
        self._initialize_start_events()

    def _initialize_start_events(self) -> None:
        """Initialize starting readable events"""
        # Get the start node
        start_node_name = self.config.start_node_id
        start_node = self._get_node_by_name(start_node_name)

        if not start_node:
            print(f"[Warning] Start node not found: {start_node_name}")
            return

        # Check if the start node has an auto_link
        auto_link = start_node.get("auto_link", "")

        if auto_link:
            # If auto_link exists, initialize the jump target node
            print(f"[Init] Start node '{start_node_name}' auto-links to '{auto_link}'")

            # Get the target node
            target_node = self._get_node_by_name(auto_link)
            if not target_node:
                print(f"[Warning] Auto-link target node not found: {auto_link}")
                return

            # Add target node to the readable event pool
            target_sub_name = target_node.get("sub_name", auto_link)
            self.event_pool.append(target_sub_name)
            self.known_events.add(target_sub_name)

            # Extract target node keywords (emphasize) and add to keyword pool
            emphasize = target_node.get("emphasize", [])
            for keyword in emphasize:
                # Check if keyword has already been used
                if keyword not in self.used_keywords:
                    self.keyword_pool.add(keyword)

            print(f"[Init] Added event '{target_sub_name}' to readable pool")
            print(f"[Init] Added keywords: {emphasize}")
        else:
            # If no auto_link, use the start node itself (if needed)
            # But per the comment, the start node only provides story background and
            # does not participate in character event ordering
            # So we don't add the start node to the event pool here
            print(f"[Init] Start node '{start_node_name}' has no auto-link")

    def _get_node_by_name(self, node_name: str) -> Dict[str, Any]:
        """Get node data by node name

        Args:
            node_name: Node name (can be name or sub_name)

        Returns:
            Node data dictionary, or empty dict if not found
        """
        for node in self.nodes:
            # Match name field first
            if node.get("name") == node_name:
                return node
            # Then match sub_name field (if it exists)
            if node.get("sub_name") == node_name:
                return node
        return {}

    def _is_dialogue_node(self, node_name: str) -> bool:
        """Determine if a node is a dialogue entry (dialogue-number or talk-number format)"""
        return bool(re.match(r'^\s*talk-\d+\s*$', node_name, re.IGNORECASE))

    def _get_lock_info(self, event_name: str) -> Dict[str, Any]:
        """Get lock info by event name

        Args:
            event_name: Event name

        Returns:
            Lock info dictionary, or {"name": event_name, "type": "none", "question": "", "answer": ""} if not found
        """
        for lock in self.lock_info:
            if lock["sub_name"] == event_name:
                return lock
        return {"name": event_name, "type": "none", "question": "", "answer": ""}


    def observe(self) -> Observation:
        """Get current environment observation

        Returns:
            Observation object containing the current game state; for the NoCaseEnv game, only obs.text is used
        """

        # Get current node info
        current_node = self._get_node_by_name(self.current_node_id)
        node_name = current_node.get("sub_name", self.current_node_id)

        # If node has already been read, use key_info directly without re-reading
        image_urls = []
        if node_name in self.read_events:
            # Already-read node: retrieve cached key_info directly
            node_key_info = current_node.get("key_info", []) if current_node else []
        else:
            # Unread node: determine reading method based on node type
            if self._is_dialogue_node(node_name):
                # Dialogue entry: use key_info directly, OCR disabled
                node_key_info = current_node.get("key_info", []) if current_node else []
            else:
                # Event node: prefer reading from dialogue.json; fall back to key_info if not found
                if node_name in self.dialogue_lookup:
                    dialogue_text = self.dialogue_lookup[node_name]
                    node_key_info = [dialogue_text]
                    # Cache in node object to avoid repeated lookups
                    if current_node:
                        current_node["key_info"] = node_key_info
                    print(f"[NoCaseEnv] Event '{node_name}' loaded from dialogue.json")
                else:
                    # Fallback: use existing text from key_info, OCR not triggered
                    node_key_info = current_node.get("key_info", []) if current_node else []
                    print(f"[NoCaseEnv] Event '{node_name}' not found in dialogue.json, falling back to key_info")

            # Mark the node as read
            if node_name and node_name not in self.read_events:
                self.read_events.add(node_name)
                print(f"[NoCaseEnv] Node '{node_name}' added to read events")

        # Build choices (action space for the NoCaseEnv game)
        choices = [
            Choice(index=0, text="Use keyword to unlock event"),
            Choice(index=1, text="Read event"),
            Choice(index=2, text="Submit character event ordering"),
            Choice(index=3, text="Use key to unlock yellow-lock event"),
            Choice(index=4, text="Answer question to unlock pink/purple lock event"),
        ]

        memory = Memory(
            description="No Case Should Remain Unsolved reasoning game state",
            key_info=node_key_info
        )

        # Check if game is over: score reaches 44 (all orderings correct = 44 points, with some margin)
        ENDING_SCORE_THRESHOLD = 40
        is_ending = self.score_points >= ENDING_SCORE_THRESHOLD

        # Convert node_key_info list to formatted text
        if isinstance(node_key_info, list):
            text = "\n".join(node_key_info).lstrip('\n')
        else:
            text = str(node_key_info).lstrip('\n')

        # If this event's time has been revealed through ordering scores, prepend it to the text
        if node_name in self.time_revealed_events:
            event_time = self.dialogue_time_lookup.get(node_name, "")
            if event_time and text:
                text = f"[Event Occurrence Time: {event_time}]\n{text}"

        # Calculate unread events (event_pool excluding already-read events)
        unread_events = [e for e in self.event_pool if e not in self.read_events]

        return Observation(
            node_id=self.current_node_id,
            name=node_name,
            text=text,
            choices=choices,
            memory=memory,
            is_ending=is_ending,
            meta={
                "keyword_pool": list(self.keyword_pool),
                "known_events": list(self.known_events),
                "event_pool": unread_events,  # Changed to unread events
                "read_events": list(self.read_events),
                "locked_events": {k: list(v) for k, v in self.locked_events.items()},
                "score": self.score_points,
                "keys": self.keys,
                "image_urls": image_urls  # Add image URL list
            }
        )

    def choose(self, choice_index: int) -> None:
        """Execute a choice (retained for compatibility; NoCaseEnv uses specialized methods for actions)"""
        raise NotImplementedError("NoCaseEnv game uses specialized methods for actions")

    def apply_keyword_unlock(self, keyword: str) -> List[str]:
        """Unlock events using a keyword

        Args:
            keyword: The keyword to use

        Returns:
            List of newly unlocked event names
        """
        # Check if the keyword is in the keyword pool
        if keyword not in self.keyword_pool:
            print(f"[NoCaseEnv] Keyword '{keyword}' not in keyword pool")
            return []

        # Remove the keyword from the pool (each keyword can only be used once)
        self.keyword_pool.remove(keyword)

        # Add keyword to the used list
        self.used_keywords.add(keyword)

        # Look up related events via tag_index
        related_events = get_events_by_tag(keyword, self.tag_index)

        newly_unlocked = []
        for event_name in related_events:
            if event_name not in self.known_events:
                # Check lock type
                lock_data = self._get_lock_info(event_name)
                lock_type = lock_data.get("type", "none")

                # Normalize lock type
                if "yellow" in lock_type.lower():
                    # Add to yellow lock pool
                    self.locked_events["yellow"].add(event_name)
                elif "pink" in lock_type.lower():
                    # Add to pink lock pool
                    self.locked_events["pink"].add(event_name)
                elif "purple" in lock_type.lower():
                    # Add to purple lock pool
                    self.locked_events["purple"].add(event_name)
                else:
                    # No lock, add directly to readable pool
                    self.event_pool.append(event_name)

                # Mark as known event
                self.known_events.add(event_name)
                newly_unlocked.append(event_name)

        return newly_unlocked

    def read_event(self, event_name: str) -> Dict[str, Any]:
        """Read an event to get its full content

        Args:
            event_name: Event name

        Returns:
            Full event data (including description, key_info, characters, tags, etc.)
        """
        if event_name not in self.event_pool:
            # Allow re-reading already-read events
            if event_name not in self.read_events:
                raise ValueError(f"Event '{event_name}' is not available for reading")

        # Don't add to read events here; do it in observe() after OCR execution
        # self.read_events.add(event_name) # moved to observe()

        # Get event node data
        node = self._get_node_by_name(event_name)

        # After reading, remove from readable event list (regardless of whether it's a dialogue node)
        if event_name in self.event_pool:
            self.event_pool.remove(event_name)
            print(f"[NoCaseEnv] Event '{event_name}' removed from readable pool")

        # Update current node ID so next observe() can retrieve this node's info
        self.current_node_id = event_name

        # Extract new keywords from the event (automatically add to keyword pool)
        tags = node.get("emphasize", [])
        for tag in tags:
            # Check if keyword has already been used
            if tag not in self.used_keywords:
                self.keyword_pool.add(tag)

        return node

    def apply_orders(self, orders_dict: Dict[str, List[str]]) -> Dict[str, Any]:
        """Apply character event ordering and score

        Args:
            orders_dict: Mapping of character name -> event sequence

        Returns:
            Dictionary containing judgement results, new points, and new keys
        """
        # Call scoring.py for batch judgement
        judgements, new_points, updated_awarded_pairs = judge_character_orders(
            orders_dict,
            self.order_gt,
            self.awarded_pairs
        )

        # Update state
        self.character_orders.update(orders_dict)

        # Filter duplicates: ignored pairs are skipped (tracked by awarded_pairs); other pairs are only recorded on first submission
        for j in judgements:
            if j["result"] == "ignored":
                continue
            pair_key = (j["character"], j["earlier"], j["later"])
            if pair_key not in self.submitted_pairs:
                self.order_judgements.append(j)
                self.submitted_pairs.add(pair_key)

        self.awarded_pairs = updated_awarded_pairs

        # Update score
        old_score = self.score_points
        self.score_points += new_points

        # Calculate new keys earned
        old_keys = calculate_keys_earned(old_score, self.config.key_threshold)
        new_keys = calculate_keys_earned(self.score_points, self.config.key_threshold)
        keys_earned = new_keys - old_keys
        self.keys += keys_earned

        return {
            "judgements": judgements,
            "new_points": new_points,
            "keys_earned": keys_earned,
            "total_score": self.score_points,
            "total_keys": self.keys
        }

    def unlock_by_key(self, event_name: str) -> bool:
        """Unlock a yellow-lock event using a key

        Args:
            event_name: Event name

        Returns:
            True if unlock was successful, False otherwise
        """
        # Check if it's a yellow lock
        if event_name not in self.locked_events["yellow"]:
            return False

        # Check key count
        if self.keys <= 0:
            return False

        # Consume a key
        self.keys -= 1

        # Remove from lock pool and add to readable pool
        self.locked_events["yellow"].remove(event_name)
        self.event_pool.append(event_name)

        return True

    def answer_lock(self, event_name: str, answer: str) -> Dict[str, Any]:
        """Unlock a pink/purple lock event by answering a question

        For a regular lock (answer is a string), behavior is the same as before and returns a dict with unlocked info.
        For a multi-event lock (answer is a list), each submission is one event name; progress is tracked and the lock opens when all are correct.

        Args:
            event_name: Event name
            answer: Player's provided answer (string for regular lock; single event name for multi-event lock)

        Returns:
            Dictionary with the following fields:
            - unlocked (bool): Whether the lock was fully unlocked
            - is_multi (bool): Whether this is a multi-event lock
            - answer_correct (bool): Whether this submission was correct
            - already_submitted (bool): Whether this answer was already correctly submitted (multi-event lock only)
            - progress (int): Number of correctly submitted items so far (multi-event lock only)
            - total (int): Total items required (multi-event lock only)
        """
        # Check if event is in a lock pool
        lock_type = None
        for lt in ["pink", "purple"]:
            if event_name in self.locked_events[lt]:
                lock_type = lt
                break

        if lock_type is None:
            return {"unlocked": False, "is_multi": False, "answer_correct": False,
                    "already_submitted": False, "progress": 0, "total": 0}

        # Get the correct answer from lock_info
        lock_data = self._get_lock_info(event_name)
        correct_answer = lock_data.get("answer", "")

        if isinstance(correct_answer, list):
            # --- Multi-event lock: submit one event name at a time ---
            if event_name not in self.multi_lock_progress:
                self.multi_lock_progress[event_name] = set()
            progress_set = self.multi_lock_progress[event_name]
            total = len(correct_answer)
            normalized_correct = [a.strip() for a in correct_answer]
            normalized_input = answer.strip()

            # Check if already correctly submitted
            if normalized_input in progress_set:
                return {"unlocked": False, "is_multi": True, "answer_correct": True,
                        "already_submitted": True, "progress": len(progress_set), "total": total}

            if normalized_input in normalized_correct:
                progress_set.add(normalized_input)
                if len(progress_set) == total:
                    # All correct, officially unlock
                    self.locked_events[lock_type].remove(event_name)
                    self.event_pool.append(event_name)
                    return {"unlocked": True, "is_multi": True, "answer_correct": True,
                            "already_submitted": False, "progress": total, "total": total}
                else:
                    return {"unlocked": False, "is_multi": True, "answer_correct": True,
                            "already_submitted": False, "progress": len(progress_set), "total": total}
            else:
                return {"unlocked": False, "is_multi": True, "answer_correct": False,
                        "already_submitted": False, "progress": len(progress_set), "total": total}
        else:
            # --- Regular single-answer lock ---
            is_correct = answer.strip().lower() == correct_answer.strip().lower()
            if is_correct:
                self.locked_events[lock_type].remove(event_name)
                self.event_pool.append(event_name)
            return {"unlocked": is_correct, "is_multi": False, "answer_correct": is_correct,
                    "already_submitted": False, "progress": 0, "total": 0}

    def reset(self) -> None:
        """Reset the environment"""
        self.current_node_id = self.config.start_node_id

        # Reset game state
        self.keyword_pool.clear()
        self.used_keywords.clear()
        self.known_events.clear()
        self.event_pool.clear()
        self.read_events.clear()
        self.locked_events = {
            "pink": set(),
            "purple": set(),
            "yellow": set()
        }

        self.character_orders.clear()
        self.order_judgements.clear()
        self.score_points = 0
        self.keys = 0
        self.awarded_pairs.clear()
        self.submitted_pairs.clear()
        self.time_revealed_events.clear()
        self.multi_lock_progress.clear()

        # Re-initialize start events
        self._initialize_start_events()


    def get_state(self) -> Dict[str, Any]:
        """Get environment state for checkpoint

        Returns:
            Dictionary containing the complete environment state
        """
        # Save key_info for already-read nodes (to avoid re-OCR)
        ocr_cache = {}
        for event_name in self.read_events:
            node = self._get_node_by_name(event_name)
            if node and "key_info" in node:
                ocr_cache[event_name] = node["key_info"]

        return {
            "checkpoint_version": 2,  # Add version number; version 2 includes OCR cache
            "current_node_id": self.current_node_id,
            "keyword_pool": list(self.keyword_pool),
            "used_keywords": list(self.used_keywords),
            "known_events": list(self.known_events),
            "event_pool": self.event_pool.copy(),
            "read_events": list(self.read_events),
            "locked_events": {k: list(v) for k, v in self.locked_events.items()},
            "character_orders": self.character_orders.copy(),
            "order_judgements": self.order_judgements.copy(),
            "score_points": self.score_points,
            "keys": self.keys,
            "awarded_pairs": [list(pair) for pair in self.awarded_pairs],
            "submitted_pairs": [list(pair) for pair in self.submitted_pairs],
            "time_revealed_events": list(self.time_revealed_events),
            "multi_lock_progress": {k: list(v) for k, v in self.multi_lock_progress.items()},
            "ocr_cache": ocr_cache  # Save OCR cache
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore environment state from a checkpoint

        Args:
            state: Environment state dictionary

        Raises:
            ValueError: If the checkpoint version is incompatible
        """
        # Check checkpoint version
        checkpoint_version = state.get("checkpoint_version", 1)
        if checkpoint_version < 2:
            raise ValueError(
                f"Incompatible checkpoint version {checkpoint_version}. "
                f"This version requires checkpoint version >= 2 (with OCR cache). "
                f"Please re-run the game to generate a new checkpoint."
            )

        self.current_node_id = state.get("current_node_id", self.config.start_node_id)
        self.keyword_pool = set(state.get("keyword_pool", []))
        self.used_keywords = set(state.get("used_keywords", []))
        self.known_events = set(state.get("known_events", []))
        self.event_pool = state.get("event_pool", []).copy() if isinstance(state.get("event_pool"), list) else []
        self.read_events = set(state.get("read_events", []))

        # Restore locked events
        locked_events_data = state.get("locked_events", {})
        self.locked_events = {
            "pink": set(locked_events_data.get("pink", [])),
            "purple": set(locked_events_data.get("purple", [])),
            "yellow": set(locked_events_data.get("yellow", []))
        }

        self.character_orders = state.get("character_orders", {}).copy()
        self.order_judgements = state.get("order_judgements", []).copy()
        self.score_points = state.get("score_points", 0)
        self.keys = state.get("keys", 0)

        # Restore OCR cache into nodes objects
        ocr_cache = state.get("ocr_cache", {})
        for event_name, key_info in ocr_cache.items():
            node = self._get_node_by_name(event_name)
            if node:
                node["key_info"] = key_info
                print(f"[Checkpoint] Restored text cache for event '{event_name}'")

        # Restore awarded event pairs
        awarded_pairs_data = state.get("awarded_pairs", [])
        self.awarded_pairs = set(tuple(pair) for pair in awarded_pairs_data)

        # Restore submitted event pairs
        submitted_pairs_data = state.get("submitted_pairs", [])
        self.submitted_pairs = set(tuple(pair) for pair in submitted_pairs_data)
        # Compatibility with old checkpoints: rebuild submitted_pairs from order_judgements
        if not self.submitted_pairs and self.order_judgements:
            for j in self.order_judgements:
                if j.get("result") != "ignored":
                    self.submitted_pairs.add((j["character"], j["earlier"], j["later"]))

        # Restore set of events whose times have been revealed
        self.time_revealed_events = set(state.get("time_revealed_events", []))

        # Restore multi-event lock progress
        self.multi_lock_progress = {
            k: set(v) for k, v in state.get("multi_lock_progress", {}).items()
        }

        print(f"[NoCaseEnv] State restored: score={self.score_points}, keys={self.keys}, "
              f"known_events={len(self.known_events)}")

    def file_to_data_url(path: Path) -> str:
        ext = path.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"