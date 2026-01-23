# galagent/agent/runner.py AgentLoop
# TODO：排查为什么反复打开一个文件 1.unlock文件列表没有更新 2.初始的node节点要给的更多一点 3.语言改一下 4.提供给llm的信息看一下
from __future__ import annotations

from typing import List, Union, Optional
import json

from galagent.common.schemas import Observation
from galagent.common.openai_harmony import Message, TextContent, Author, Role
from galagent.common.config import AgentConfig
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
    ):
        self.env = env
        self.store = store
        self.retriever = retriever
        self.policy = policy
        self.config = config
        self.game_utils = game_utils
        self.logger = logger

    async def run(self) -> None:
        for step in range(self.config.max_steps):
            # 获取当前节点的环境信息
            obs = self.env.observe()

            if obs.is_ending is True:
                # Log ending
                if self.logger:
                    self.logger.log_ending(ending_node=obs.node_id, reached_ending=True)
                return

            # store current scene text into memory
            self.store.add(obs.text, meta={"node_id": obs.node_id, "name": obs.name})

            # 让LLM决定是否需要检索记忆
            retrieval_decision = self.policy.decide_retrieval(obs)
            need_retrieval = retrieval_decision["need_retrieval"]
            retrieval_query = retrieval_decision["query"]
            retrieval_reason = retrieval_decision["reason"]

            # 根据LLM的决策进行检索
            search_results = ""
            if need_retrieval and retrieval_query:
                # retrieve
                if self.config.retriever_type == "vector":
                    # VectorRetriever使用search方法直接返回结果列表
                    results = self.retriever.search(
                        retrieval_query,
                        top_k=self.config.retrieve_top_k
                    )
                    # 格式化为与KeywordRetrieverTool相同的JSON格式
                    search_results = json.dumps({
                        "query": retrieval_query,
                        "top_k": self.config.retrieve_top_k,
                        "results": [{"text": r} for r in results],
                        "total": len(results)
                    }, ensure_ascii=False)
                else:
                    # KeywordRetrieverTool使用Tool接口
                    llm_generated_json = json.dumps({
                        "query": retrieval_query,
                        "top_k": self.config.retrieve_top_k,
                    }, ensure_ascii=False)

                    llm_message = Message(
                        author=Author(role=Role.ASSISTANT, name="gpt-4o"),
                        content=[TextContent(text=llm_generated_json)],
                    ).with_recipient("KeywordRetrieverTool")

                    async for tool_response in self.retriever._process(llm_message):
                        search_results = tool_response.content[0].text    
           
            # decide
            # 使用游戏工具获取游戏特定的上下文信息
            game_context = self.game_utils.get_game_context(self.env)

            decision = self.policy.decide(obs, search_results, game_context)

            # log action
            if self.logger:
                # 使用游戏工具格式化所有日志数据
                log_data = self.game_utils.format_log_data(
                    step=step,
                    obs=obs,
                    search_results=search_results,
                    decision=decision,
                    game_context=game_context,
                    retrieval_decision={
                        "need_retrieval": need_retrieval,
                        "query": retrieval_query,
                        "reason": retrieval_reason
                    }
                )

                # 直接使用格式化后的数据记录日志
                self.logger.log_action(**log_data)

            # act
            self.game_utils.execute_action(self.env, decision)

        # Max steps reached without ending
        if self.logger:
            self.logger.log_ending(ending_node=self.env.current_node_id, reached_ending=False)

        raise RuntimeError("Max steps reached without reaching an ending (is_ending: true).")
