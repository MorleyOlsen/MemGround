# memground_agent/agent/policy.py
from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI

from memground_agent.common.schemas import Decision, Observation
from memground_agent.common.config import LLMConfig
from memground_agent.env.base_prompt_builder import BasePromptBuilder


class LLMPolicy:
    """
    LLM-driven decision policy.
    It asks the model to output strict JSON: {"choice_index": int, "reason": str}

    Uses game-specific PromptBuilder to construct prompts, decoupling game logic
    """

    def __init__(self, config: LLMConfig, prompt_builder: BasePromptBuilder, memory_store=None, game_utils=None, agent_config=None):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=120.0)
        self.prompt_builder = prompt_builder
        self.game_utils = game_utils
        self.agent_config = agent_config

        # Pass memory_store to prompt_builder
        if memory_store:
            self.prompt_builder.set_memory_store(memory_store)

    def _call_llm(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """Unified method for calling the LLM API

        Args:
            messages: Message list, format: [{"role": "system/user/assistant", "content": "..."}]
            model: Model name; defaults to the model in config
            temperature: Temperature parameter; defaults to the temperature in config

        Returns:
            Text content returned by the LLM
        """
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=model or self.config.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.config.temperature,
                )
                msg = resp.choices[0].message
                return msg.content or ""
            except Exception as e:
                # Print detailed error information
                error_type = type(e).__name__
                import json
                with open("./test", "w", encoding="utf-8") as fw:
                    fw.write(json.dumps(messages, ensure_ascii=False, indent=2))
                print(f"[LLM] Call failed (attempt {attempt + 1}/3)")
                print(f"  Error type: {error_type}")
                print(f"  Error message: {str(e)}")

                # If it's an OpenAI API error, print more details
                if hasattr(e, 'status_code'):
                    print(f"  HTTP status: {e.status_code}")
                if hasattr(e, 'response'):
                    try:
                        response_json = e.response.json() if hasattr(e.response, 'json') else None
                        if response_json:
                            print(f"  API response: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
                    except:
                        pass
                if hasattr(e, 'code'):
                    print(f"  Error code: {e.code}")
                if hasattr(e, 'param'):
                    print(f"  Error param: {e.param}")

                # Print full traceback (only on the first failure)
                if attempt == 0:
                    import traceback
                    print(f"  Traceback:")
                    traceback.print_exc()

                if attempt < 2:
                    time.sleep(5)
        print("[LLM] 3 consecutive failures, skipping this step")
        return ""

    def decide_retrieval(self, obs: Observation, game_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Decide whether memory retrieval is needed and generate retrieval targets

        Args:
            obs: Current observation
            game_context: Game-specific context information

        Returns:
            Dict containing need_retrieval, query/filenames, reason
            - For KB game: {"need_retrieval": bool, "query": str, "reason": str}
            - For Type Help game: {"need_retrieval": bool, "filenames": list, "reason": str}
        """
        # Use prompt builder to construct the retrieval decision prompt
        user_prompt = self.prompt_builder.build_retrieval_prompt(obs, game_context)

        # If no file retrieval prompt exists, fall back to generic retrieval decision
        if not user_prompt:
            user_prompt = self.prompt_builder.build_retrieval_decision_prompt(obs)

        messages = [
            {"role": "user", "content": user_prompt},
        ]

        # Perform memory management before calling the LLM
        if self.game_utils and self.agent_config:
            full_prompt = user_prompt
            self.game_utils.manage_memory(
                self.agent_config,
                full_prompt=full_prompt,
                llm_client=self.client,
                llm_config=self.config,
                prompt_builder=self.prompt_builder
            )

        text = self._call_llm(messages)

        # Extract retrieval decision info (compatible with both formats)
        parsed = self._parse_json(text)
        need_retrieval = parsed.get("need_retrieval", False)
        reason = parsed.get("reason", "")
        filters = parsed.get("filters", None)  # Extract filters field (if present)

        # Build return result
        result = {
            "need_retrieval": need_retrieval,
            "reason": reason
        }

        # Add filters (if present and non-empty)
        if filters and isinstance(filters, dict):
            result["filters"] = filters

        # Determine game type based on the returned fields
        if "filenames" in parsed:
            # Type Help game: return list of filenames
            result["filenames"] = parsed.get("filenames", [])
        else:
            # KB game: return query
            result["query"] = parsed.get("query", obs.text)

        return result

    def decide(self, obs: Observation, retrieved_hits: List[str], game_context: Optional[Dict[str, Any]] = None) -> Decision:
        """Make a decision

        Args:
            obs: Current observation
            retrieved_hits: Retrieved memory (already parsed as a list of strings)
            game_context: Game-specific context information (e.g. file_tracker_info)

        Returns:
            Decision object
        """
        # Use the game-specific prompt builder to construct prompts
        system_prompt = self.prompt_builder.build_system_prompt(game_context)
        user_prompt = self.prompt_builder.build_user_prompt(obs, retrieved_hits, game_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Perform memory management before calling the LLM
        if self.game_utils and self.agent_config:
            full_prompt = system_prompt + user_prompt
            self.game_utils.manage_memory(
                self.agent_config,
                full_prompt=full_prompt,
                llm_client=self.client,
                llm_config=self.config,
                prompt_builder=self.prompt_builder
            )

        text = self._call_llm(messages)
        print("text:", text)

        parsed = self._parse_json(text)


        # Check if this is a Dust game action format
        if "action_type" in parsed:
            # Dust game: return action type and parameters
            action_type = parsed.get("action_type", 0)
            action_params = parsed.get("action_params", {})
            rationale = parsed.get("rationale", "")

            # Ensure action_type is an integer
            if isinstance(action_type, str):
                try:
                    action_type = int(action_type)
                except ValueError:
                    action_type = 0

            # Serialize the entire action info as choice_text
            choice_text = json.dumps({
                "action_params": action_params
            }, ensure_ascii=False)

            if not isinstance(rationale, str) or not rationale.strip():
                rationale = "No reason provided."

            return Decision(
                choice_index=action_type,
                rationale=rationale.strip(),
                choice_text=choice_text,
                recall=[]
            )
        # Check if this is text-input mode (e.g. Type Help game)
        elif "choice_text" in parsed:
            # Type Help game: return filename
            filename = parsed.get("choice_text", "")
            reason = parsed.get("reason", "")
            recall = parsed.get("recall", [])

            if not isinstance(reason, str) or not reason.strip():
                reason = "No reason provided."

            # Ensure recall is a list
            if not isinstance(recall, list):
                recall = []

            return Decision(
                choice_index=0,  # Fixed at 0 for Type Help game
                rationale=reason.strip(),
                choice_text=filename,
                recall=recall
            )
        else:
            # Traditional choice mode (KB game etc.)
            choice_index = parsed.get("choice_index", 0)
            reason = parsed.get("reason", "")

            # Validate whether choice_index is valid
            valid_indices = {c.index for c in obs.choices}

            if choice_index not in valid_indices:
                choice_index = 0  # Fall back to option 0

            if not isinstance(reason, str) or not reason.strip():
                reason = "No reason provided."

            return Decision(
                choice_index=choice_index,
                rationale=reason.strip(),
                choice_text=""
            )

    def _parse_json(self, s: str) -> Optional[Dict[str, Any]]:
        """Parse the JSON string returned by the LLM

        Args:
            s: String returned by the LLM

        Returns:
            Parsed dictionary, or an empty dict if parsing fails
        """
        s = (s or "").strip()
        if not s:
            return {}
        # Common case: model output wrapped in ```json
        if s.startswith("```"):
            s = s.strip("`")
            # May contain json\n{...}
            s = s[s.find("{"):] if "{" in s else s
        # Extract the first {...} block
        if "{" in s and "}" in s:
            s2 = s[s.find("{"): s.rfind("}") + 1]
        else:
            s2 = s
        try:
            return json.loads(s2)
        except Exception:
            return {}

    def generate_story_summary(self, game_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a story plot summary and reasoning

        Args:
            game_context: Game-specific context information

        Returns:
            Story summary text
        """
        is_english = getattr(self.prompt_builder, 'test_language', 'en') == 'en'

        if is_english:
            system_prompt = """You are a professional story analyst. Based on all the events, dialogues, and clues you experienced in the game, complete the following tasks:
        1. Story Summary: Summarize the main plot of the entire story in 2-3 paragraphs
        2. Character Analysis: Analyze the motivations and relationships of the main characters
        3. Reasoning Conclusion: Based on all the information, deduce the truth or core secret of the story

        Please present your analysis in a clear and organized manner."""
        else:
            system_prompt = """You are a professional story analyst. Based on all the events, dialogues, and clues you experienced in the game, complete the following tasks:
        1. Story Summary: Summarize the main plot of the entire story in 2-3 paragraphs
        2. Character Analysis: Analyze the motivations and relationships of the main characters
        3. Reasoning Conclusion: Based on all the information, deduce the truth or core secret of the story

        Please present your analysis in a clear and organized manner."""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if game_context and 'conversation_history' in game_context:
            conversation_history = game_context['conversation_history']
            if is_english:
                user_prompt = f"The following are all dialogues and event records from the game:\n\n{conversation_history}\n\nPlease complete the story analysis based on the above information."
            else:
                user_prompt = f"The following are all dialogues and event records from the game:\n\n{conversation_history}\n\nPlease complete the story analysis based on the above information."
            messages.append({"role": "user", "content": user_prompt})

        summary = self._call_llm(messages, temperature=0.7)

        return summary
