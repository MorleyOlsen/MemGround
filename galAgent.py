# galAgent.py
"""
Galgame Agent主程序
使用LLM和向量检索来玩Galgame并达到最佳结局
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from galagent.env.env_factory import create_game_env, create_prompt_builder, create_game_utils, get_supported_game_types
from galagent.memory.store import MemoryStore
from galagent.memory.retriever import KeywordRetrieverTool, VectorRetriever
from galagent.agent.policy import LLMPolicy
from galagent.agent.runner import GalgameAgent
from galagent.common.config import ConfigLoader
from galagent.common.checkpoint import CheckpointManager
from galagent.logger import GameLogger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Galgame Agent - Play galgames using LLM with memory recall"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["keyword", "vector"],
        default=None,
        help="Override retriever type: keyword or vector (default: from config)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Override number of top results to retrieve (default: from config)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max steps for agent (default: from config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Enable verbose output (default: from config)"
    )
    return parser.parse_args()


async def main_async():
    """主程序入口"""
    args = parse_args()

    # 设置项目根目录
    ROOT = Path(__file__).resolve().parent
    config_path = ROOT / args.config

    print(f"Loading configuration from: {config_path}")

    # 加载所有配置
    config_loader = ConfigLoader(config_path)
    llm_config = config_loader.load_llm_config()
    embedding_config = config_loader.load_embedding_config()
    agent_config = config_loader.load_agent_config()
    env_config = config_loader.load_env_config()
    checkpoint_config = config_loader.load_checkpoint_config()

    # 命令行参数覆盖配置文件
    if args.retriever:
        agent_config.retriever_type = args.retriever
    if args.top_k is not None:
        agent_config.retrieve_top_k = args.top_k
    if args.max_steps is not None:
        agent_config.max_steps = args.max_steps
    if args.verbose is not None:
        agent_config.verbose = args.verbose

    print(f"[OK] LLM: {llm_config.provider}/{llm_config.model}")
    print(f"[OK] Embedding: {embedding_config.provider}/{embedding_config.model} (dim={embedding_config.dim})")
    print(f"[OK] Agent: retriever={agent_config.retriever_type}, top_k={agent_config.retrieve_top_k}, max_memory={agent_config.max_memory}")
    print(f"[OK] Game: {env_config.game_type}")
    print(f"[OK] Checkpoint: enabled={checkpoint_config.enabled}, interval={checkpoint_config.interval}")
    print()

    # 初始化游戏环境
    env = create_game_env(env_config, ROOT)
    print(f"[OK] Game environment initialized: {env_config.game_type}")

    # 初始化记忆存储
    store = MemoryStore(
        embedding_config=embedding_config,
        max_memory=agent_config.max_memory,
        use_faiss=True
    )

    # 初始化检索器
    if agent_config.retriever_type == "vector":
        retriever = VectorRetriever(store)
        print(f"[OK] Using VectorRetriever with Faiss")
    else:
        retriever = KeywordRetrieverTool(store)
        print(f"[OK] Using KeywordRetrieverTool")

    # 创建游戏特定的Prompt构建器
    prompt_builder = create_prompt_builder(env_config, llm_config.goal_instruction)
    print(f"[OK] PromptBuilder created for game: {env_config.game_type}")

    # 创建游戏特定的工具类
    game_utils = create_game_utils(env_config)
    print(f"[OK] GameUtils created for game: {env_config.game_type}")

    # 初始化LLM策略
    policy = LLMPolicy(llm_config, prompt_builder, memory_store=store)
    print(f"[OK] LLMPolicy initialized")

    # 初始化CheckpointManager（如果启用）
    checkpoint_manager = None
    if checkpoint_config.enabled:
        checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_config.dir)
        print(f"[OK] CheckpointManager initialized: {checkpoint_config.dir}")

    # 如果需要从checkpoint恢复，先加载以获取logger_session_id
    logger_session_id = None
    start_step = 0
    resume_from = checkpoint_config.resume_from

    if resume_from:
        checkpoint_data = checkpoint_manager.load_checkpoint(Path(resume_from))
        logger_session_id = checkpoint_data.get('logger_session_id')
        start_step = checkpoint_data['step']
        print(f"[OK] Resuming from checkpoint: step={start_step}, logger_session_id={logger_session_id}")

    # 初始化GameLogger
    log_dir = ROOT / "logs"
    if resume_from and logger_session_id:
        # 恢复模式：使用checkpoint中的logger_session_id
        logger = GameLogger(log_dir, env_config.game_type, session_id=logger_session_id, resume=True)
    else:
        # 新建模式
        logger = GameLogger(log_dir, env_config.game_type)
    print(f"[OK] GameLogger initialized: {logger.session_id}")

    # 初始化Agent
    agent = GalgameAgent(
        env=env,
        store=store,
        retriever=retriever,
        policy=policy,
        config=agent_config,
        game_utils=game_utils,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        checkpoint_interval=checkpoint_config.interval if checkpoint_config.enabled else 0,
        session_name=f"{env_config.game_type}_session"
    )

    # 如果提供了checkpoint，则从checkpoint恢复状态
    if resume_from:
        start_step, _ = agent.load_checkpoint(resume_from)
        print(f"[OK] State restored, continuing from step {start_step + 1}")

    print("\n" + "=" * 70)
    print("Starting Galgame Agent...")
    print("=" * 70 + "\n")

    # 运行Agent
    await agent.run()

    # 保存Faiss索引
    if store.use_faiss and store.faiss_manager:
        store.save_faiss_index()
        print("\n[OK] Faiss index saved")


def main():
    """程序入口"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nAgent interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


if __name__ == "__main__":
    main()
