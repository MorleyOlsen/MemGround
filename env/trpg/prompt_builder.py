# env/trpg/prompt_builder.py
"""TRPG evaluation prompt builder and prompt templates"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from galagent.common.schemas import Observation
from galagent.env.base_prompt_builder import BasePromptBuilder


# ── QA prompt templates ──────────────────────────────────────────────────────

QA_SYSTEM_EN = """\
You are an expert in deep reading comprehension of TRPG narrative records.
You have read a TRPG session log in order (earlier content may have been summarized due to context limits).
Based on your memory and current context, respond using the following fixed format (do NOT omit either label):

Reasoning: [2-4 sentences of analytical reasoning, using logical connectives such as "because... therefore...", "this suggests...", "taken together...", to show your reasoning chain]
Answer: [1-2 sentence concise conclusion that directly answers the question]

Answer in English. All content must be grounded in the story.\
"""


# ── Compression summary prompt templates ────────────────────────────────────

SUMMARY_SYSTEM_EN = """\
Summarize the following TRPG session log into a single cohesive new summary.

**INPUT STRUCTURE**: The content may contain two types of material:
- Paragraphs starting with [Story Summary]: previously compressed summaries, already highly condensed — every sentence contains critical, irreplaceable information.
- Raw session lines ([playerX] / [GM] format): original TRPG records with lower information density; redundant content can be trimmed.

**COMPRESSION TARGET**: Merge both types into one coherent narrative. Aim to compress the total input to 50%–60% of its original length; use your own judgment on how to weight each part.

**HARD CHARACTER LIMIT** — Output MUST be between {min_chars} and {max_chars} characters. This is a non-negotiable constraint:
- Do NOT write fewer than {min_chars} characters — falling short means over-compression; you must add back detail.
- Do NOT exceed {max_chars} characters — exceeding means insufficient compression; you must cut further.

Preserve all information valuable for understanding the story, including but not limited to:

- Character actions and movements: what each character did, where they went, who they interacted with, and the outcomes
- Inner monologue and emotions: characters' true thoughts, feelings, and reactions (even brief asides should be kept)
- Contradictions between words and actions: what characters said versus what they actually thought
- Skill checks and dice rolls: skill names, results (success/failure/hard success/critical success/critical failure) and their narrative impact
- Sanity and HP changes: specific numerical changes and their triggers
- NPC behavior: important NPCs' words, attitudes, information revealed or concealed
- Clues and foreshadowing: investigation findings, anomalies, unresolved mysteries
- Event consequences: how actions or events changed the situation, relationships, or available information
- Scene information: time, location, environmental atmosphere, and other contextual details

**FINAL CHECK**: Before outputting, confirm that the character count is between {min_chars} and {max_chars}. Adjust if needed.
Output in English. Do not add any prefix or explanation.\
"""


# ── Retry prompt templates ───────────────────────────────────────────────────

RETRY_HIGH_EN = """\
Your summary has {actual_chars} characters, exceeding the limit of {max_chars}. Please condense the content, remove redundancies, and ensure the output is between {min_chars} and {max_chars} characters.\
"""

RETRY_LOW_EN = """\
Your summary has only {actual_chars} characters, below the minimum of {min_chars}. Please add key details to ensure the output is between {min_chars} and {max_chars} characters.\
"""


class TRPGPromptBuilder(BasePromptBuilder):
    """
    PromptBuilder for TRPG mode.
    Primarily used in the QA phase; the reading phase requires no LLM calls and thus
    no system/user prompt.
    """

    def build_system_prompt(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        story = (game_context or {}).get("story", "TRPG Session")
        return (
            f"You are an expert in deep reading comprehension of TRPG narrative records, reading the story '{story}'.\n"
            "Please read the dialogue carefully and remember its content; you will need to answer questions about the story afterwards."
        )

    def build_user_prompt(
        self,
        obs: Observation,
        retrieved_hits: List[str],
        game_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        return obs.text

    def build_qa_user_message(self, question: str) -> str:
        """Build the user message for the QA phase (includes system instruction + question)"""
        return (
            "You are an expert in deep reading comprehension of TRPG narrative records.\n"
            "You have read a TRPG session log in order "
            "(earlier content may have been summarized due to context limits).\n"
            "Please answer the following question in 2-4 sentences based on your memory and current context, "
            "including reasoning (because... therefore... / this suggests... / taken together...).\n"
            "Answer in English. The answer must be grounded in the story content.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
