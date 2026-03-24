# memground_agent/agent/runner.py AgentLoop
# TODO: investigate repeated file opens - 1.unlock list not updating 2.initial node needs more info 3.language 4.check LLM input
from __future__ import annotations

from typing import List, Union, Optional
import json

from memground_agent.common.schemas import Observation
from memground_agent.common.openai_harmony import Message, TextContent, Author, Role
from memground_agent.common.config import AgentConfig
from memground_agent.common.checkpoint import CheckpointManager
from memground_agent.env.base_env import BaseGameEnv
from memground_agent.env.base_game_utils import BaseGameUtils
from memground_agent.memory.store import MemoryStore
from memground_agent.memory.retriever import KeywordRetrieverTool, VectorRetriever
from memground_agent.agent.policy import DummyPolicy, LLMPolicy
from memground_agent.logger import GameLogger


class GalgameAgent:
    def __init__(
        self,
        env: BaseGameEnv,
        store: MemoryStore,
        retriever: Union[KeywordRetrieverTool, VectorRetriever],
        policy: LLMPolicy,
        config: AgentConfig,
        game_utils: BaseGameUtils,
        logger: Optional[GameLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        checkpoint_interval: int = 100,
        session_name: Optional[str] = None,
    ):
        self.env = env
        self.store = store
        self.retriever = retriever
        self.policy = policy
        self.config = config
        self.game_utils = game_utils
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_interval = checkpoint_interval
        self.session_name = session_name
        self.start_step = 0  # Starting step, used to keep step continuity when resuming from a checkpoint

        # Pass memory_store to game_utils (if supported)
        if hasattr(self.game_utils, 'set_memory_store'):
            self.game_utils.set_memory_store(self.store)

        # Pass env to game_utils (if supported)
        if hasattr(self.game_utils, 'set_env'):
            self.game_utils.set_env(self.env)

    async def run(self) -> None:
        # Initialize: observe the starting node and add to memory (first run only; on resume, memory is already restored from checkpoint, so skip)
        if self.start_step == 0:
            initial_obs = self.env.observe()
            self.store.add_message(initial_obs.text, role="user", step=0, node_id=initial_obs.node_id, name=initial_obs.name)

        # Early stop: terminate if no new node has been self-discovered in N consecutive steps
        _EARLY_STOP_STEPS = 200
        steps_no_new_self_unlock = 0
        if hasattr(self.env, 'get_file_tracker_info'):
            _ft = self.env.get_file_tracker_info()
            _prev_self_unlock_count = len(
                set(_ft.get('unlocked_files', [])) - set(_ft.get('hint_unlocked_files', []))
            )
        else:
            _prev_self_unlock_count = 0

        for step in range(self.start_step, self.start_step + self.config.max_steps):
            # Get game context (used for retrieval decision)
            game_context = self.game_utils.get_game_context(self.env)

            # Get current observation (used for decision, but not added to memory)
            current_obs = self.env.observe()

            # Check whether an ending has been reached
            if current_obs.is_ending is True:
                # Generate story summary
                story_summary = self.policy.generate_story_summary(game_context)

                print("\n" + "=" * 70)
                print("Story Summary & Reasoning")
                print("=" * 70)
                print(story_summary)
                print("=" * 70 + "\n")

                # Log ending
                if self.logger:
                    self.logger.log_ending(
                        ending_node=current_obs.node_id,
                        reached_ending=True,
                        story_summary=story_summary
                    )
                return

            # 1. Let policy decide whether retrieval is needed and what to retrieve
            retrieval_decision = self.policy.decide_retrieval(current_obs, game_context)
            # 2. Use game utilities to perform the actual retrieval operation
            retrieval_result = self.game_utils.retrieve_information(retrieval_decision, self.config)

            # Retrieval results are not added to memory; they are passed directly to decide() as part of the prompt
            # if retrieval_result and self.config.verbose:
            #     print(f"[Retrieval] Retrieved {len(retrieval_decision.get('filenames', []))} files")

            # decide (retrieval result is part of the prompt but not added to memory)
            decision = self.policy.decide(current_obs, retrieval_result or "", game_context)

            # Add decision as an assistant message to memory (conversation history) TODO: extract into a separate function
            decision_text = f"{decision.choice_text if decision.choice_text else f'Action {decision.choice_index}'}: {decision.rationale}"
            if decision.recall:
                recall_text = ", ".join(decision.recall)
                decision_text += f"{recall_text}"
            self.store.add_message(decision_text, role="assistant", step=step)

            # act (catch single-step execution exceptions to avoid one failure terminating the entire run)
            try:
                action_result = self.game_utils.execute_action(self.env, decision)
            except Exception as e:
                print(f"[Runner] Step {step} execute_action failed, skipping: {e}")
                action_result = False

            # Post-action hook (game-specific handling, e.g. recording read files, handling failures)
            self.game_utils.post_action_hook(decision, action_success=action_result, step=step)

            # Early stop check: stop early if no new node has been self-discovered in N consecutive steps
            if hasattr(self.env, 'get_file_tracker_info'):
                _ft = self.env.get_file_tracker_info()
                _cur = len(set(_ft.get('unlocked_files', [])) - set(_ft.get('hint_unlocked_files', [])))
                if _cur > _prev_self_unlock_count:
                    steps_no_new_self_unlock = 0
                else:
                    steps_no_new_self_unlock += 1
                _prev_self_unlock_count = _cur

                if steps_no_new_self_unlock >= _EARLY_STOP_STEPS:
                    print(f"\n[Early Stop] No new nodes discovered in {_EARLY_STOP_STEPS} consecutive steps, stopping early.")
                    if self.logger:
                        self.logger.log_ending(
                            ending_node=current_obs.node_id,
                            reached_ending=False,
                            story_summary=f"Early stop: no new nodes self-unlocked in {_EARLY_STOP_STEPS} consecutive steps."
                        )
                    return

            # Only observe new node and add to memory when the action succeeds
            if action_result:
                new_obs = self.env.observe()
                self.store.add_message(new_obs.text, role="user", step=step+1, node_id=new_obs.node_id, name=new_obs.name)
            else:
                new_obs = None

            # log action
            if self.logger:
                # Re-fetch game context to include the latest file tracking info (updated after execute_action)
                updated_game_context = self.game_utils.get_game_context(self.env)

                # Use the observation after action execution for logging (use pre-execution observation if action failed)
                log_obs = new_obs if action_result else current_obs

                # Use game utilities to format all log data
                log_data = self.game_utils.format_log_data(
                    step=step,
                    obs=log_obs,
                    decision=decision,
                    game_context=updated_game_context,
                    retrieval_decision=retrieval_decision
                )

                # Directly log using the formatted data
                self.logger.log_action(**log_data)

            # Save checkpoint (if enabled and interval reached)
            if self.checkpoint_manager and (step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(step)

        # Generate summary when max steps are reached
        game_context = self.game_utils.get_game_context(self.env)
        story_summary = self.policy.generate_story_summary(game_context)

        print("\n" + "=" * 70)
        print("Story Summary & Reasoning")
        print("=" * 70)
        print(story_summary)
        print("=" * 70 + "\n")

        if self.logger:
            self.logger.log_ending(
                ending_node=self.env.current_node_id,
                reached_ending=False,
                story_summary=story_summary
            )

        raise RuntimeError("Max steps reached without reaching an ending (is_ending: true).")

    def save_checkpoint(self, step: int) -> None:
        """Save current state to a checkpoint

        Args:
            step: Current step number
        """
        if not self.checkpoint_manager:
            return

        # Collect state from each component
        env_state = self.env.get_state() if hasattr(self.env, 'get_state') else {}
        memory_state = self.store.get_state() if hasattr(self.store, 'get_state') else {}
        game_utils_state = self.game_utils.get_state() if hasattr(self.game_utils, 'get_state') else {}

        # Get logger's session_id (if logger exists)
        logger_session_id = self.logger.session_id if self.logger else None

        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            step=step,
            env_state=env_state,
            memory_state=memory_state,
            game_utils_state=game_utils_state,
            session_name=self.session_name,
            logger_session_id=logger_session_id
        )

    def load_checkpoint(self, checkpoint_file: str) -> tuple[int, Optional[str]]:
        """Restore state from a checkpoint

        Args:
            checkpoint_file: Path to the checkpoint file

        Returns:
            (restored step number, logger_session_id)
        """
        if not self.checkpoint_manager:
            raise RuntimeError("CheckpointManager not initialized")

        from pathlib import Path
        checkpoint_data = self.checkpoint_manager.load_checkpoint(Path(checkpoint_file))

        # Restore state of each component
        if hasattr(self.env, 'restore_state'):
            self.env.restore_state(checkpoint_data['env_state'])

        if hasattr(self.store, 'restore_state'):
            self.store.restore_state(checkpoint_data['memory_state'])

        if hasattr(self.game_utils, 'restore_state'):
            self.game_utils.restore_state(checkpoint_data['game_utils_state'])

        # Set starting step to the step after the checkpoint
        self.start_step = checkpoint_data['step'] + 1

        # Get logger_session_id
        logger_session_id = checkpoint_data.get('logger_session_id')

        print(f"[Agent] Restored from checkpoint, resuming from step {self.start_step}")
        if logger_session_id:
            print(f"[Agent] Logger session ID: {logger_session_id}")

        return checkpoint_data['step'], logger_session_id
