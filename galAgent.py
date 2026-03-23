# galAgent.py
"""
Galgame Agent主程序
使用LLM和向量检索来玩Galgame并达到最佳结局
"""
from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime
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
    parser.add_argument(
        "--story",
        type=str,
        default=None,
        help="[TRPG] Override story_name in config.yaml"
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
    mem_agent_config = config_loader.load_mem_agent_config()
    judge_llm_config = config_loader.load_judge_llm_config()

    # 命令行参数覆盖配置文件
    if args.retriever:
        agent_config.retriever_type = args.retriever
    if args.top_k is not None:
        agent_config.retrieve_top_k = args.top_k
    if args.max_steps is not None:
        agent_config.max_steps = args.max_steps
    if args.verbose is not None:
        agent_config.verbose = args.verbose
    if args.story is not None:
        env_config.story_name = args.story

    print(f"[OK] LLM: {llm_config.provider}/{llm_config.model}")
    print(f"[OK] Embedding: {embedding_config.provider}/{embedding_config.model} (dim={embedding_config.dim})")
    print(f"[OK] Agent: retriever={agent_config.retriever_type}, top_k={agent_config.retrieve_top_k}, max_context_tokens={agent_config.max_context_tokens}")
    print(f"[OK] Game: {env_config.game_type}")
    print(f"[OK] Checkpoint: enabled={checkpoint_config.enabled}, interval={checkpoint_config.interval}")
    print(f"[OK] Mem Agent: use_mem={mem_agent_config.use_mem}, mem_name={mem_agent_config.mem_name}")
    print()

    # 初始化游戏环境
    env = create_game_env(env_config, ROOT)
    print(f"[OK] Game environment initialized: {env_config.game_type}")

    # 获取resume_from配置（需要在记忆代理初始化前使用）
    resume_from = checkpoint_config.resume_from

    # 初始化记忆代理（如果启用）
    mem_agent = None
    if mem_agent_config.use_mem:
        try:
            # TRPG 模式用 story_name 作为 game_name，以区分不同故事的记忆
            _mem_game_name = (
                env_config.story_name
                if env_config.game_type == "trpg"
                else env_config.game_type
            )
            # 根据 mem_name 选择不同的记忆代理
            if mem_agent_config.mem_name == "mem_0":
                # 使用 Mem0 云端记忆代理
                from galagent.memory.mem0 import Mem0Agent

                if not mem_agent_config.mem0_api_key:
                    raise ValueError("Mem0 API key is required when mem_name='mem_0'")

                mem_agent = Mem0Agent(
                    api_key=mem_agent_config.mem0_api_key,
                    game_name=_mem_game_name,
                    model_name=llm_config.model,
                    verbose=agent_config.verbose
                )
                print(f"[OK] Mem0Agent initialized, user_id: {_mem_game_name}_{llm_config.model}")

            elif mem_agent_config.mem_name == "a_mem":
                from galagent.memory.amem import AMemAgent
                # 使用 A-mem 本地记忆代理
                mem_agent = AMemAgent(
                    game_name=_mem_game_name,
                    embedding_model=mem_agent_config.amem_embedding_model,
                    llm_backend=mem_agent_config.amem_llm_backend,
                    llm_model=mem_agent_config.amem_llm_model,
                    verbose=agent_config.verbose,
                    api_key=llm_config.api_key
                )
                print(f"[OK] AMemAgent initialized for game: {_mem_game_name}")

            else:
                raise ValueError(f"Unknown mem_name: {mem_agent_config.mem_name}. Supported: 'mem_0', 'a_mem'")

            # 根据配置决定是否清空记忆
            # 如果是从checkpoint恢复，不清空；如果是新开始且配置了clear_on_start，则清空
            if mem_agent_config.clear_on_start and not resume_from:
                print(f"[MemAgent] Clearing existing memories for fresh start...")
                result = mem_agent.delete_all_memories()
                if result.get("success"):
                    # 如果是 Mem0，等待云端完成异步删除操作
                    if mem_agent_config.mem_name == "mem_0":
                        time.sleep(3)

                    # 验证删除是否成功
                    remaining = mem_agent.get_all_memories()
                    print(f"[MemAgent] Verified: {len(remaining)} memories remaining")
                else:
                    print(f"[WARNING] Failed to clear memories: {result.get('error', 'unknown error')}")
            elif resume_from:
                print(f"[MemAgent] Resuming from checkpoint, keeping existing memories")
            else:
                print(f"[MemAgent] Using existing memories (clear_on_start=false)")

        except Exception as e:
            print(f"[WARNING] Failed to initialize MemAgent: {e}")
            print(f"[WARNING] Falling back to traditional memory system")
            mem_agent_config.use_mem = False
            mem_agent = None

    # 初始化记忆存储
    store = MemoryStore(
        embedding_config=embedding_config,
        use_faiss=not mem_agent_config.use_mem,  # disable Faiss when using mem_agent
        mem_agent=mem_agent,
        use_mem=mem_agent_config.use_mem
    )

    # ── TRPG 模式：使用专用 runner ──────────────────────────────────────────
    if env_config.game_type == "trpg":
        from datetime import datetime
        from galagent.agent.trpg_runner import run_trpg

        _mem_tag = mem_agent_config.mem_name if mem_agent_config.use_mem else "nomem"
        _model_tag = llm_config.model.replace("/", "_")
        _story_tag = getattr(env_config, "story_name", "trpg")
        _run_tag = f"{_story_tag}__{_model_tag}__{_mem_tag}"

        ckpt_resume_from = checkpoint_config.resume_from or None
        if ckpt_resume_from:
            try:
                import json as _json
                _ckpt_data = _json.loads(Path(ckpt_resume_from).read_text(encoding="utf-8"))
                session_id = _ckpt_data.get("session_id") or _run_tag
            except Exception as e:
                print(f"[Warning] Failed to read checkpoint: {e}")
                session_id = _run_tag
        else:
            session_id = _run_tag

        log_dir = ROOT / "logs" / "trpg" / session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        store.verbose = agent_config.verbose
        store.memory_log_file = str(log_dir / "memory.jsonl")

        try:
            run_trpg(
                env=env,
                store=store,
                config=agent_config,
                llm_config=llm_config,
                judge_llm_config=judge_llm_config,
                log_dir=log_dir,
                session_id=session_id,
                ckpt_resume_from=ckpt_resume_from,
                mem_agent=mem_agent,
            )
        except KeyboardInterrupt:
            print("\n[Interrupted] TRPG evaluation interrupted by user")
        return

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

    # 初始化LLM策略（传入game_utils和agent_config用于记忆管理）
    policy = LLMPolicy(llm_config, prompt_builder, memory_store=store, game_utils=game_utils, agent_config=agent_config)
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

    # 初始化GameLogger（每个 session 独立子目录）
    if resume_from and logger_session_id:
        # 恢复模式：使用 checkpoint 中记录的 session_id 对应的目录
        log_dir = ROOT / "logs" / env_config.game_type / logger_session_id
        logger = GameLogger(log_dir, env_config.game_type, session_id=logger_session_id,
                            resume=True, model=llm_config.model, truncate_after_step=start_step)
    else:
        # 新建模式：预生成 session_id 以便确定目录
        _sid = f"{env_config.game_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = ROOT / "logs" / env_config.game_type / _sid
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = GameLogger(log_dir, env_config.game_type, session_id=_sid, model=llm_config.model)
    print(f"[OK] GameLogger initialized: {logger.session_id}")

    # 记忆日志文件放在 session 目录内
    memory_log_file = log_dir / "memory.jsonl"
    store.verbose = agent_config.verbose
    store.memory_log_file = str(memory_log_file)
    print(f"[OK] Memory logging enabled: {memory_log_file}")

    # CheckpointManager 目录更新为 session 子目录
    if checkpoint_config.enabled:
        ckpt_dir = log_dir / "checkpoints"
        checkpoint_manager = CheckpointManager(checkpoint_dir=ckpt_dir)
        print(f"[OK] CheckpointManager initialized: {ckpt_dir}")

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
    try:
        await agent.run()
    except RuntimeError as e:
        print(f"\n[Agent] Game ended: {e}")
        # Continue to auto QA even when max_steps reached

    # Auto QA evaluation for type_help / dust
    if env_config.game_type in ("type_help", "dust"):
        print("\n" + "=" * 70)
        print("Starting automatic QA evaluation...")
        print("=" * 70 + "\n")
        try:
            import sys as _sys
            _scripts_dir = str(ROOT / "scripts")
            if _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)
            from run_qa import run_qa_from_store, GAME_CONFIG as _QA_GAME_CONFIG
            if env_config.game_type in _QA_GAME_CONFIG:
                _qa_log_dir = ROOT / "logs" / env_config.game_type / logger.session_id
                run_qa_from_store(
                    game_type=env_config.game_type,
                    store=store,
                    memory_log_file=str(memory_log_file),
                    log_dir=_qa_log_dir,
                    llm_config=llm_config,
                    judge_llm_config=judge_llm_config,
                    session_id=logger.session_id,
                )
            else:
                print(f"[QA] No QA data configured for game_type={env_config.game_type!r}, skipping")
        except Exception as _e:
            print(f"[QA] Auto QA failed: {_e}")
            raise

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
