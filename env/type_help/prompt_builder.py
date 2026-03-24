# env/type_help/prompt_builder.py
"""Prompt builder for the Type Help game"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from memground_agent.common.schemas import Observation
from memground_agent.env.base_prompt_builder import BasePromptBuilder


class TypeHelpPromptBuilder(BasePromptBuilder):
    """Prompt builder for the Type Help puzzle game

    Features of this game:
    - File naming rule inference
    - File unlock tracking
    - Deduce new filenames based on already-unlocked files
    """

    def __init__(self, goal_instruction, test_language: str = "en", provide_naming_rules: bool = False):
        self.goal_instruction = goal_instruction
        self.test_language = test_language
        self.provide_naming_rules = provide_naming_rules

    # Overall prompt
    def build_system_prompt(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """Build the system prompt for the Type Help game"""
        return self._build_system_prompt_en(game_context)

    def _build_system_prompt_en(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        base_prompt = """You are a game agent specialized in puzzle-solving games. In this game, you need to obtain information by entering **filenames** that follow certain specific rules. You need to reasonably infer the file naming rules based on available information, **deduce possible correct filenames** to unlock more files, and use the information obtained from files to **reconstruct the entire story**.

        Strategy:
        1. Prioritize opening unlocked but not yet viewed files to help you gain more information.
        2. Question marks in filenames indicate parts you need to guess.
        3. For failed files, **do not try the same failed filename again**; try other combinations.
        4. When guessing filenames, carefully analyze the naming patterns in unlocked files, such as the meaning of numbers, the ordering relationship of number sizes, the meaning of letters, etc.
        5. For character numbers, do not guess numbers that are too large and haven't appeared in the text information.
        6. **Focus on character movement information**: Carefully read dialogues and scene descriptions to extract these clues for deducing the next filename:
           - A character says "I'm going to [location]", "Go find him at [location]", or is summoned to a location — this character will appear in the next time slot's file for that location
           - A character leaves the current scene — they will no longer appear in subsequent files of the current location, but will appear in files of the new location
           - New characters enter the scene — their numbers should be added to the current location's subsequent file name
           - Multiple characters heading to the same location — all their numbers should appear together in the target location's filename
        7. **Pay special attention to the beginning and end of each node's text**: Character movement clues tend to concentrate there — the opening describes who enters the scene or where they came from, while the ending describes who leaves, where they are going, or what action is planned next.
        """

        # Naming rules (switch-controlled)
        if self.provide_naming_rules:
            base_prompt += """

        **File Naming Rules**:
        The filename format is: `[two-digit time number]-[two-letter location code]-[character number 1]-[character number 2]-...`
        - **Time number**: Two digits indicating chronological order (e.g. 01, 02, 03 ...), larger numbers mean later in time
        - **Location code**: Two uppercase letters representing the scene location (e.g. LI, KI, DI, EN, QU, etc.)
        - **Character numbers**: Numbers of characters present at that time and location, separated by `-`, **listed in ascending order**
        - Example: `03-LI-1-4-5` means time slot 03, location LI, characters 1, 4, and 5 are present
        - Special files (e.g. `00-readme`, `00-invitation`) start with `00` and are background/reference files that do not follow the above pattern
        """

        # Add unlocked file list to system prompt
        if game_context:
            file_tracker_info = game_context.get("file_tracker_info")
            if file_tracker_info:
                unlocked = file_tracker_info.get("unlocked_files", [])
                if unlocked:
                    base_prompt += f"\n\n**Currently Unlocked Files** (you can view their content by entering these filenames):\n"
                    base_prompt += ", ".join([f'"{f}"' for f in unlocked])

        return base_prompt

    def build_retrieval_prompt(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        """Build the file retrieval prompt (Type Help game specific)"""
        return self._build_retrieval_prompt_en(obs, game_context)

    def _build_retrieval_prompt_en(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> str:
        read_files_text = ""
        conversation_history=""
        if game_context:
            read_files_text = game_context.get('read_files_text', '')
            conversation_history = game_context['conversation_history']

        return f"""
            Current node information: {obs.text}
            Historical memory: {conversation_history}

            {read_files_text}

            Task:
            Decide whether you need to review previously read files to help make a decision.
            If needed, list the filenames you want to review (select from the read files list, maximum 3 most relevant files).
            Do not select files that already exist in current memory.

            Output format (strictly follow this JSON format, all text values must be in English):
            {{
            "need_retrieval": true/false,
            "filenames": ["filename1", "filename2", ...],
            "filters": {{"role": "user"}},
            "reason": "<brief explanation in English of why you need to view these files>"
            }}

            Notes:
            - filters is optional, use simple key-value format like {{"role": "user"}} or {{"step": "2"}}
            - System will automatically combine your filter with user_id
            - If no specific filtering is needed, omit the filters field or set it to {{}}

            """.strip()

    # Per-node prompt
    def build_user_prompt(
        self,
        obs: Observation,
        retrieved_hits: List[str],
        game_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the user prompt for the Type Help game"""
        return self._build_user_prompt_en(obs, retrieved_hits, game_context)

    def _build_user_prompt_en(self, obs: Observation, retrieved_hits: List[str], game_context: Optional[Dict[str, Any]] = None) -> str:
        # Format file tracker information
        file_info_str = self._format_file_tracker_info(game_context)

        if game_context and 'conversation_history' in game_context:
            conversation_history = game_context['conversation_history']

        # Get character name information
        character_names_text = ""
        if game_context and 'character_names_text' in game_context:
            character_names_text = self._translate_character_names_to_en(game_context['character_names_text'])

        # Build the complete prompt
        prompt = f"""
        Current node information: {obs.text};
        Historical memory: {conversation_history};
        """

        if character_names_text:
            prompt += f"""
        {character_names_text};
        """

        if retrieved_hits:
            prompt += f"""
            Retrieved relevant file content: {retrieved_hits};
            """

        prompt += f"""
        File operation records: {file_info_str};"""

        prompt += """
        (Internal reasoning only — do NOT output the analysis) Before deciding, think through the following internally:
        [Character Movement Analysis] From the current node and historical memory, analyze (pay special attention to the **beginning** and **end** of each node's text — that is where character movement clues are most concentrated):
          - Which characters mentioned going to (or were directed to) a location? → Target location + their numbers
          - Which characters left the current scene? → They will appear in files of the new location
          - Which new characters entered the scene? → Their numbers should appear in subsequent filenames
        Based on this analysis and the file naming rules, output ONLY the following JSON in English, nothing else. Do NOT output any analysis, explanation, or text outside the JSON.

        Output format (strictly follow this JSON format, all text values must be in English):
        {
        "choice_text": <filename>,
        "reason": "<one sentence in English, max 20 words, explaining why this choice helps achieve the goal>"
        "recall": <list of filenames that you think are related to this choice, or empty list if none>
        }
        """

        return prompt.strip()

    def _format_file_tracker_info(self, game_context: Optional[Dict[str, Any]]) -> str:
        """Format file tracker information (only read and failed records)"""
        if not game_context:
            return ""

        file_tracker_info = game_context.get("file_tracker_info")
        if not file_tracker_info:
            return ""

        file_info_str = ""
        failed = file_tracker_info.get("failed_files", [])
        readed = game_context.get('read_files_text', '')

        if failed:
            file_info_str += f"\nFailed attempts (these files do not exist, **do not try failed filenames again**):\n"
            file_info_str += ", ".join([f'"{f}"' for f in failed])

        if readed:
            file_info_str += f"\n{readed}"

        return file_info_str

    def _translate_character_names_to_en(self, character_names_text: str) -> str:
        """Translate character name information to an English heading"""
        if not character_names_text:
            return ""

        # Replace legacy Chinese-labeled headings if present (English-only mode: these are no-ops)
        result = character_names_text.replace("**Character Number and Name Reference**:", "**Character Number and Name Reference**:")
        result = result.replace("Number ", "Number ")
        return result

    def build_compression_prompt(self, conversations: List[str]) -> str:
        """Build the prompt for memory compression"""
        conversations_text = "\n\n".join(conversations)
        prompt = f"""Summarize the following content, keeping only the key information (e.g. important filenames, main content of files, characters, sequence of events, etc.).
            Conversation content: {conversations_text}
            """
        return prompt
