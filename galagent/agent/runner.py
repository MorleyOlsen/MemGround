# galagent/agent/runner.py AgentLoop
# TODO：排查为什么反复打开一个文件 1.unlock文件列表没有更新 2.初始的node节点要给的更多一点 3.语言改一下 4.提供给llm的信息看一下
from __future__ import annotations

from typing import List, Union, Optional
import json

from galagent.common.schemas import Observation
from galagent.common.openai_harmony import Message, TextContent, Author, Role
from galagent.common.config import AgentConfig
from galagent.common.checkpoint import CheckpointManager
from galagent.env.kb_env import KBEnv
from galagent.env.base_game_utils import BaseGameUtils
from galagent.memory.store import MemoryStore
from galagent.memory.retriever import KeywordRetrieverTool, VectorRetriever
from galagent.agent.policy import DummyPolicy, LLMPolicy
from galagent.logger import GameLogger


class GalgameAgent:
    def __init__(
        self,
        env: KBEnv,
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
        self.start_step = 0  # 起始步数，用于从checkpoint恢复时保持step连续性

        # 将 memory_store 传递给 game_utils（如果支持）
        if hasattr(self.game_utils, 'set_memory_store'):
            self.game_utils.set_memory_store(self.store)

        # 将 env 传递给 game_utils（如果支持）
        if hasattr(self.game_utils, 'set_env'):
            self.game_utils.set_env(self.env)

    async def run(self) -> None:
        # 初始化：observe起始节点并添加到记忆
        initial_obs = self.env.observe()
        self.store.add_message(initial_obs.text, role="user", step=0, node_id=initial_obs.node_id, name=initial_obs.name)

        # 早停：连续 N 步无自主新节点则终止
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
            # 获取游戏上下文（用于检索决策）
            game_context = self.game_utils.get_game_context(self.env)

            # 获取当前观察（用于决策，但不添加到记忆）
            current_obs = self.env.observe()

            # 检查是否到达结局
            if current_obs.is_ending is True:
                # 生成故事总结
                story_summary = self.policy.generate_story_summary(game_context)

                print("\n" + "=" * 70)
                is_en = getattr(getattr(self.policy, "prompt_builder", None), "test_language", "ch") == "en"
                print("Story Summary & Reasoning" if is_en else "故事总结与推理")
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

            # 1. 让 policy 决定是否需要检索以及检索什么
            retrieval_decision = self.policy.decide_retrieval(current_obs, game_context)
            # 2. 使用游戏工具执行具体的检索操作
            retrieval_result = self.game_utils.retrieve_information(retrieval_decision, self.config)

            # 检索结果不添加到记忆，而是直接传递给decide方法作为prompt的一部分
            if retrieval_result and self.config.verbose:
                print(f"[检索] 检索到 {len(retrieval_decision.get('filenames', []))} 个文件")

            # decide（检索结果作为prompt的一部分但是不添加到记忆）
            decision = self.policy.decide(current_obs, retrieval_result or "", game_context)

            # 将决策作为助手消息添加到记忆（对话历史） TODO:单独写一个函数出来
            decision_text = f"{decision.choice_text if decision.choice_text else f'选择 {decision.choice_index}'}: {decision.rationale}"
            if decision.recall:
                recall_text = ", ".join(decision.recall)
                decision_text += f"{recall_text}"
            self.store.add_message(decision_text, role="assistant", step=step)

            # act（捕获单步执行异常，避免一次失败终止整个运行）
            try:
                action_result = self.game_utils.execute_action(self.env, decision)
            except Exception as e:
                print(f"[Runner] Step {step} execute_action 失败，跳过本步: {e}")
                action_result = False

            # 动作执行后的钩子（游戏特定处理，如记录已读文件、处理失败情况）
            self.game_utils.post_action_hook(decision, action_success=action_result, step=step)

            # 早停检查：连续 N 步未自主发现新节点则提前终止
            if hasattr(self.env, 'get_file_tracker_info'):
                _ft = self.env.get_file_tracker_info()
                _cur = len(set(_ft.get('unlocked_files', [])) - set(_ft.get('hint_unlocked_files', [])))
                if _cur > _prev_self_unlock_count:
                    steps_no_new_self_unlock = 0
                else:
                    steps_no_new_self_unlock += 1
                _prev_self_unlock_count = _cur

                if steps_no_new_self_unlock >= _EARLY_STOP_STEPS:
                    print(f"\n[早停] 连续 {_EARLY_STOP_STEPS} 步未自主发现新节点，提前终止。")
                    if self.logger:
                        self.logger.log_ending(
                            ending_node=current_obs.node_id,
                            reached_ending=False,
                            story_summary=f"早停：连续 {_EARLY_STOP_STEPS} 步未自主解锁新节点。"
                        )
                    return

            # 只有在动作成功时，才observe新节点并添加到记忆
            if action_result:
                new_obs = self.env.observe()
                self.store.add_message(new_obs.text, role="user", step=step+1, node_id=new_obs.node_id, name=new_obs.name)
            else:
                new_obs = None

            # log action
            if self.logger:
                # 重新获取游戏上下文以包含最新的文件追踪信息（execute_action后的更新）
                updated_game_context = self.game_utils.get_game_context(self.env)

                # 使用动作执行后的观察结果记录日志（如果动作失败则使用执行前的观察）
                log_obs = new_obs if action_result else current_obs

                # 使用游戏工具格式化所有日志数据
                log_data = self.game_utils.format_log_data(
                    step=step,
                    obs=log_obs,
                    decision=decision,
                    game_context=updated_game_context,
                    retrieval_decision=retrieval_decision
                )

                # 直接使用格式化后的数据记录日志
                self.logger.log_action(**log_data)

            # 保存checkpoint（如果启用且到达间隔）
            if self.checkpoint_manager and (step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(step)

        # 达到最大步数时生成总结
        game_context = self.game_utils.get_game_context(self.env)
        story_summary = self.policy.generate_story_summary(game_context)

        print("\n" + "=" * 70)
        print("故事总结与推理")
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
        """保存当前状态到checkpoint

        Args:
            step: 当前步数
        """
        if not self.checkpoint_manager:
            return

        # 收集各组件的状态
        env_state = self.env.get_state() if hasattr(self.env, 'get_state') else {}
        memory_state = self.store.get_state() if hasattr(self.store, 'get_state') else {}
        game_utils_state = self.game_utils.get_state() if hasattr(self.game_utils, 'get_state') else {}

        # 获取logger的session_id（如果有logger）
        logger_session_id = self.logger.session_id if self.logger else None

        # 保存checkpoint
        self.checkpoint_manager.save_checkpoint(
            step=step,
            env_state=env_state,
            memory_state=memory_state,
            game_utils_state=game_utils_state,
            session_name=self.session_name,
            logger_session_id=logger_session_id
        )

    def load_checkpoint(self, checkpoint_file: str) -> tuple[int, Optional[str]]:
        """从checkpoint恢复状态

        Args:
            checkpoint_file: checkpoint文件路径

        Returns:
            (恢复的步数, logger_session_id)
        """
        if not self.checkpoint_manager:
            raise RuntimeError("CheckpointManager not initialized")

        from pathlib import Path
        checkpoint_data = self.checkpoint_manager.load_checkpoint(Path(checkpoint_file))

        # 恢复各组件的状态
        if hasattr(self.env, 'restore_state'):
            self.env.restore_state(checkpoint_data['env_state'])

        if hasattr(self.store, 'restore_state'):
            self.store.restore_state(checkpoint_data['memory_state'])

        if hasattr(self.game_utils, 'restore_state'):
            self.game_utils.restore_state(checkpoint_data['game_utils_state'])

        # 设置起始步数为checkpoint的下一步
        self.start_step = checkpoint_data['step'] + 1

        # 获取logger_session_id
        logger_session_id = checkpoint_data.get('logger_session_id')

        print(f"[Agent] 已从checkpoint恢复，将从step {self.start_step}继续")
        if logger_session_id:
            print(f"[Agent] Logger session ID: {logger_session_id}")

        return checkpoint_data['step'], logger_session_id
