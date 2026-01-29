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
        for step in range(self.start_step, self.start_step + self.config.max_steps):
            # 获取当前节点的环境信息
            obs = self.env.observe()

            if obs.is_ending is True:
                # 生成故事总结
                game_context = self.game_utils.get_game_context(self.env)
                story_summary = self.policy.generate_story_summary(game_context)

                print("\n" + "=" * 70)
                print("故事总结与推理")
                print("=" * 70)
                print(story_summary)
                print("=" * 70 + "\n")

                # Log ending
                if self.logger:
                    self.logger.log_ending(
                        ending_node=obs.node_id,
                        reached_ending=True,
                        story_summary=story_summary
                    )
                return

            # 简化观察信息并存储到记忆（只保留关键信息）
            simplified_obs = self.game_utils.observation(obs)
            self.store.add_message(simplified_obs, role="user", step=step, node_id=obs.node_id, name=obs.name)

            # 获取游戏上下文（用于检索决策）
            game_context = self.game_utils.get_game_context(self.env)

            # 1. 让 policy 决定是否需要检索以及检索什么
            retrieval_decision = self.policy.decide_retrieval(obs, game_context)

            # 2. 使用游戏工具执行具体的检索操作
            retrieval_result, retrieval_info = self.game_utils.retrieve_information(retrieval_decision, self.config)

            # 如果有检索结果，添加到记忆中
            if retrieval_result:
                self.store.add_message(retrieval_result, role="system", step=step)

            # decide
            # 不再传递 search_results，让 LLM 从对话历史中获取
            decision = self.policy.decide(obs, "", game_context)

            # 将决策作为助手消息添加到记忆（对话历史）
            decision_text = f"{decision.choice_text if decision.choice_text else f'选择 {decision.choice_index}'}: {decision.rationale}"
            self.store.add_message(decision_text, role="assistant", step=step)

            # act
            action_result = self.game_utils.execute_action(self.env, decision)

            # 动作执行后的钩子（游戏特定处理，如记录已读文件、处理失败情况）
            self.game_utils.post_action_hook(obs, decision, action_success=action_result, step=step)

            # 统一的记忆管理（游戏特定逻辑在子类中实现）
            self.game_utils.manage_memory(self.config.max_context_tokens, self.config)

            # log action
            if self.logger:
                # 重新获取游戏上下文以包含最新的文件追踪信息（execute_action后的更新）
                updated_game_context = self.game_utils.get_game_context(self.env)

                # 使用游戏工具格式化所有日志数据
                log_data = self.game_utils.format_log_data(
                    step=step,
                    obs=obs,
                    decision=decision,
                    game_context=updated_game_context,
                    retrieval_decision=retrieval_info  # 传递检索决策信息
                )

                # 直接使用格式化后的数据记录日志
                self.logger.log_action(**log_data)

            # 保存checkpoint（如果启用且到达间隔）
            if self.checkpoint_manager and (step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(step)

        # Max steps reached without ending
        print("\n" + "=" * 70)
        print("达到最大步数！正在生成故事总结...")
        print("=" * 70 + "\n")

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
