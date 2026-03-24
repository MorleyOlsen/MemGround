# memground_agent/env/type_help_env.py
"""Type Help puzzle game environment"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path

from memground_agent.common.schemas import Observation, Choice, Memory, Character
from memground_agent.env.base_env import BaseGameEnv, GameConfig
from env.type_help.utils.file_tracker import FileTracker


@dataclass
class TypeHelpConfig(GameConfig):
    """Type Help game configuration"""
    data_path: Path = Path("dataset/type_help")
    game_type: str = "type_help"
    start_node_id: str = "Start"
    test_language: str = "en"  # en (English prompts)
    enable_hint: bool = False  # Whether to enable the hint feature on consecutive failures
    hint_failure_threshold: int = 15  # Number of consecutive failures that trigger a hint


class TypeHelpEnv(BaseGameEnv):
    """Type Help game environment"""

    def __init__(self, config: TypeHelpConfig):
        super().__init__(config)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.file_tracker = FileTracker()
        self.character_names = []  # Stores character names and number information
        self.node_links: Dict[str, List[str]] = {}  # Stores node link relationships: {from: [target1, target2, ...]}
        self.node_predecessors: Dict[str, List[str]] = {}  # Reverse mapping: {target: [from1, from2, ...]}
        self.consecutive_failures = 0  # Counter for consecutive failures
        self.hint_unlocked_files: List[str] = []  # List of files auto-unlocked via hint
        self.load_game_data()

    def load_game_data(self) -> None:
        """Load game data"""
        data_file = self.config.data_path / "nodes.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Game data not found: {data_file}")

        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse node data
        for node in data.get("game", {}).get("nodes", []):
            node_name = node.get("name")
            if node_name:
                self.nodes[node_name] = node

        # Load character name data
        self._load_character_names()

        # Load node link relationships
        self._load_node_links()

        # Override node_predecessors with in_degree from nodes.json to ensure hint conditions
        # use the complete predecessor list
        self._build_predecessors_from_indegree()

        # Initialize: unlock files mentioned in the starting node
        self._initialize_unlocked_files()

    def _load_character_names(self) -> None:
        """Load character names and number information from name.json"""
        name_file = self.config.data_path / "name.json"
        if not name_file.exists():
            print(f"[Type Help] Warning: name.json not found at {name_file}")
            self.character_names = []
            return

        try:
            with open(name_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.character_names = data.get("name", [])
        except json.JSONDecodeError as e:
            print(f"[Type Help] Warning: Failed to parse name.json: {e}")
            self.character_names = []

    def _load_node_links(self) -> None:
        """Load node link relationships from all_links_with_recall.json"""
        links_file = self.config.data_path / "all_links_with_recall.json"
        if not links_file.exists():
            print(f"[Type Help] Warning: all_links_with_recall.json not found at {links_file}")
            self.node_links = {}
            return

        try:
            with open(links_file, 'r', encoding='utf-8') as f:
                # File is in JSONL format, one JSON object per line
                for line in f:
                    if line.strip():
                        link = json.loads(line)
                        from_node = link.get("from")
                        target_node = link.get("target")

                        if from_node and target_node:
                            if from_node not in self.node_links:
                                self.node_links[from_node] = []
                            self.node_links[from_node].append(target_node)
                            if target_node not in self.node_predecessors:
                                self.node_predecessors[target_node] = []
                            self.node_predecessors[target_node].append(from_node)
        except Exception as e:
            print(f"[Type Help] Warning: Failed to parse all_links_with_recall.json: {e}")
            self.node_links = {}
            self.node_predecessors = {}

    def _build_predecessors_from_indegree(self) -> None:
        """Override node_predecessors using the in_degree field from each node in nodes.json.

        all_links_with_recall.json records only a single most-direct predecessor edge per node,
        whereas in_degree contains the complete predecessor list. The hint trigger condition
        requires all predecessors to be unlocked, so in_degree must be used as the data source
        for node_predecessors. Non-existent node references and duplicates are filtered out
        to prevent the hint from never triggering.
        """
        for node_name, node_data in self.nodes.items():
            in_degree = node_data.get("in_degree", [])
            # Deduplicate and filter non-existent nodes, preserving original order
            seen = set()
            valid = []
            for p in in_degree:
                if p in self.nodes and p not in seen:
                    valid.append(p)
                    seen.add(p)
                elif p not in self.nodes:
                    print(f"[Type Help] Warning: in_degree of '{node_name}' references non-existent node '{p}', skipping.")
            if valid:
                self.node_predecessors[node_name] = valid
            elif node_name in self.node_predecessors:
                # If in_degree is empty, clear any old entry (possibly from all_links)
                del self.node_predecessors[node_name]

    def get_character_names_text(self) -> str:
        """Get formatted character name information text

        Returns:
            Formatted character information string
        """
        if not self.character_names:
            return ""

        lines = ["Character Number and Name Reference:"]
        for character in self.character_names:
            number = character.get("number")
            names = character.get("name", [])

            if number is not None:
                # Characters with a number
                names_str = ", ".join(names)
                lines.append(f"Number {number}: {names_str}")
            else:
                # Characters without a number (e.g. K)
                names_str = ", ".join(names)
                lines.append(f"  {names_str}")

        return "\n".join(lines)

    def _initialize_unlocked_files(self) -> None:
        """Initialize the list of unlocked files"""
        # Automatically unlock background nodes (non-file type, used for story background)
        background_nodes = ["Background","message","00-readme"]
        for node_name in background_nodes:
            if node_name in self.nodes:
                self.file_tracker.unlock_file(node_name)

        # Get initial file list from the message node
        # if "message" in self.nodes:
        #     memory = self.nodes["message"].get("memory", {})
        #     files = memory.get("files", [])
        #     for file in files:
        #         # Only unlock complete filenames (those without ????)
        #         if "?" not in file:
        #             self.file_tracker.unlock_file(file)

    # Can only observe text information, not next-step navigation information
    def observe(self) -> Observation:
        """Get the current observation"""
        if self.current_node_id not in self.nodes:
            raise ValueError(f"Node not found: {self.current_node_id}")

        node = self.nodes[self.current_node_id]
        node_name = node.get("name", "")

        # Only one choice: have the LLM enter a filename
        choices = [Choice(index=0, text="Enter filename")]

        # Use FileRetriever to uniformly retrieve and format file information
        from env.type_help.utils.file_retriever import TypeHelpFileRetriever
        retriever = TypeHelpFileRetriever(self)

        # Retrieve file information (without links)
        file_results = retriever.retrieve_files([node_name])

        if not file_results or not file_results[0].get("exists", False):
            raise ValueError(f"Failed to retrieve node data: {node_name}")

        file_info = file_results[0]

        # Format as text
        text = retriever.format_single_file(file_info).lstrip('\n')

        # Build Memory object (extracted from file_info)
        memory_data = node.get("memory", {})
        characters = []
        for char_data in memory_data.get("characters", []):
            characters.append(Character(
                name=char_data.get("name", ""),
                role=char_data.get("role", ""),
                number=char_data.get("number", 0),
                description=char_data.get("description", "")
            ))

        memory = Memory(
            description=memory_data.get("description", ""),
            key_info=memory_data.get("key_info", []),
            location=memory_data.get("location", ""),
            characters=characters
        )

        # Check if this is an ending node
        is_ending = node_name == "00-final-note"

        return Observation(
            node_id=self.current_node_id,
            name=node_name,
            text=text,
            choices=choices,
            memory=memory,
            is_ending=is_ending
        )

    def choose(self, choice_index: int) -> None:
        """Execute a choice (kept for compatibility; not used by Type Help game)"""
        raise NotImplementedError("Type Help game uses choose_by_filename instead")

    def choose_by_filename(self, filename: str) -> bool:
        """Select by filename (Type Help game specific)

        Args:
            filename: The filename to open

        Returns:
            True if file exists and was opened, False otherwise
        """
        # Check if the file exists
        if filename not in self.nodes:
            # Record the attempt (failure)
            self.file_tracker.attempt_file(filename, success=False)
            # Increment consecutive failure counter
            self.consecutive_failures += 1
            print(f"[Type Help] File not found: {filename} (consecutive failures: {self.consecutive_failures})")

            # Check whether to trigger auto-unlock hint
            self._auto_unlock_hint()

            return False

        # Record the attempt (success) and add to opened files list
        self.file_tracker.attempt_file(filename, success=True)

        # Only reset consecutive failure counter when successfully opening a new previously-unlocked file
        is_new_unlock = not self.file_tracker.is_unlocked(filename)
        if is_new_unlock:
            self.consecutive_failures = 0

        # Unlock and navigate to the file
        self.file_tracker.unlock_file(filename)
        self.current_node_id = filename

        # Special case: if "04-ST-1-5-8" is opened, remove "04-ST-?????"
        if filename == "04-ST-1-5-8":
            self.file_tracker.remove_unlocked_file("04-ST-?????")

        # Unlock new files mentioned in the newly opened node
        self._unlock_files_in_node(filename)

        print(f"[Type Help] Successfully opened file: {filename}")
        return True

    def _unlock_files_in_node(self, node_id: str) -> None:
        """If the memory's files field has content, unlock the new files mentioned in the node"""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        memory = node.get("memory", {})
        files = memory.get("files", [])

        for file in files:
            if "?" not in file:
                self.file_tracker.unlock_file(file)

    def _get_next_unlocked_target(self) -> str:
        """Traverse all unlocked nodes and find the unlocked target with the lowest time id
        among those whose all predecessor nodes are already unlocked"""
        candidates = []
        for node in self.file_tracker.unlocked_files:
            for target in self.node_links.get(node, []):
                if not self.file_tracker.is_unlocked(target):
                    # Check whether all predecessors of this target are already unlocked
                    predecessors = self.node_predecessors.get(target, [])
                    if all(self.file_tracker.is_unlocked(p) for p in predecessors):
                        candidates.append(target)
        if not candidates:
            return ""

        def _time_id(name: str) -> int:
            try:
                return int(name.split("-")[0])
            except (ValueError, IndexError):
                return 9999

        return min(candidates, key=_time_id)

    def _auto_unlock_hint(self) -> None:
        """Auto-unlock hint: when consecutive failures reach the threshold, unlock the next target node"""
        if not self.config.enable_hint:
            return

        if self.consecutive_failures >= self.config.hint_failure_threshold:
            # Find the next unlocked target node
            next_target = self._get_next_unlocked_target()

            if next_target:
                # Unlock the node
                self.file_tracker.unlock_file(next_target)
                self.hint_unlocked_files.append(next_target)
                print(f"[Type Help Hint] {self.consecutive_failures} consecutive failures, auto-unlocked hint file: {next_target}")
                # Reset the consecutive failure counter
                self.consecutive_failures = 0
            else:
                print(f"[Type Help Hint] {self.consecutive_failures} consecutive failures, but no more unlockable target nodes")

    def get_file_tracker_info(self) -> Dict[str, Any]:
        """Get file tracker information"""
        return {
            "unlocked_files": self.file_tracker.get_unlocked_files(),
            "attempted_files": self.file_tracker.get_attempted_files(),
            "success_files": self.file_tracker.get_success_files(),
            "failed_files": self.file_tracker.get_failed_files(),
            "patterns": self.file_tracker.file_naming_patterns,
            "hint_unlocked_files": self.hint_unlocked_files.copy(),
            "consecutive_failures": self.consecutive_failures,
        }

    def reset(self) -> None:
        """Reset the environment"""
        self.current_node_id = self.config.start_node_id
        self.file_tracker = FileTracker()
        self.consecutive_failures = 0  # Reset consecutive failure counter
        self.hint_unlocked_files = []
        self._initialize_unlocked_files()

    def get_state(self) -> Dict[str, Any]:
        """Get environment state for checkpoint

        Returns:
            Dictionary containing the complete environment state
        """
        return {
            "current_node_id": self.current_node_id,
            "consecutive_failures": self.consecutive_failures,
            "hint_unlocked_files": self.hint_unlocked_files.copy(),
            "file_tracker": {
                "unlocked_files": list(self.file_tracker.unlocked_files),
                "attempted_files": self.file_tracker.attempted_files.copy(),
                "success_files": self.file_tracker.success_files.copy(),
                "failed_files": self.file_tracker.failed_files.copy(),
                "file_naming_patterns": self.file_tracker.file_naming_patterns.copy(),
                "read_files": self.file_tracker.read_files.copy()
            }
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore environment state from checkpoint

        Args:
            state: Environment state dictionary
        """
        self.current_node_id = state["current_node_id"]
        self.consecutive_failures = state.get("consecutive_failures", 0)
        self.hint_unlocked_files = state.get("hint_unlocked_files", []).copy()

        # Restore file tracker state
        tracker_state = state["file_tracker"]
        self.file_tracker.unlocked_files = set(tracker_state["unlocked_files"])
        self.file_tracker.attempted_files = tracker_state["attempted_files"].copy()
        self.file_tracker.success_files = tracker_state["success_files"].copy()
        self.file_tracker.failed_files = tracker_state["failed_files"].copy()
        self.file_tracker.file_naming_patterns = tracker_state["file_naming_patterns"].copy()
        self.file_tracker.read_files = tracker_state.get("read_files", []).copy()

        print(f"[Env] State restored: current_node={self.current_node_id}, "
              f"unlocked_files={len(self.file_tracker.unlocked_files)}, "
              f"consecutive_failures={self.consecutive_failures}")

