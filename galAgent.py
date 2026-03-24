# galAgent.py
"""
Galgame Agent main program
Uses LLM and vector retrieval to play Galgame and reach the best ending
"""
from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime
from pathlib import Path

from memground_agent.env.env_factory import create_game_env, create_prompt_builder, create_game_utils, get_supported_game_types
from memground_agent.memory.store import MemoryStore
from memground_agent.memory.retriever import KeywordRetrieverTool, VectorRetriever
from memground_agent.agent.policy import LLMPolicy
from memground_agent.agent.runner import GalgameAgent
from memground_agent.common.config import ConfigLoader
from memground_agent.common.checkpoint import CheckpointManager
from memground_agent.logger import GameLogger


def parse_args():
    """Parse command-line arguments"""
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
    """Main program entry point"""
    args = parse_args()

    # Set project root directory
    ROOT = Path(__file__).resolve().parent
    config_path = ROOT / args.config

    print(f"Loading configuration from: {config_path}")

    # Load all configurations
    config_loader = ConfigLoader(config_path)
    llm_config = config_loader.load_llm_config()
    embedding_config = config_loader.load_embedding_config()
    agent_config = config_loader.load_agent_config()
    env_config = config_loader.load_env_config()
    checkpoint_config = config_loader.load_checkpoint_config()
    mem_agent_config = config_loader.load_mem_agent_config()
    judge_llm_config = config_loader.load_judge_llm_config()

    # Command-line arguments override config file
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

    # Initialize game environment
    env = create_game_env(env_config, ROOT)
    print(f"[OK] Game environment initialized: {env_config.game_type}")

    # Get resume_from config (needed before memory agent initialization)
    resume_from = checkpoint_config.resume_from

    # Initialize memory agent (if enabled)
    mem_agent = None
    if mem_agent_config.use_mem:
        try:
            # TRPG mode uses story_name as game_name to distinguish memories for different stories
            _mem_game_name = (
                env_config.story_name
                if env_config.game_type == "trpg"
                else env_config.game_type
            )
            # Select memory agent based on mem_name
            if mem_agent_config.mem_name == "mem_0":
                # Use Mem0 cloud memory agent
                from memground_agent.memory.mem0 import Mem0Agent

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
                from memground_agent.memory.amem import AMemAgent
                # Use A-mem local memory agent
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

            # Decide whether to clear memory based on config
            # If resuming from checkpoint, do not clear; if starting fresh and clear_on_start is set, clear
            if mem_agent_config.clear_on_start and not resume_from:
                print(f"[MemAgent] Clearing existing memories for fresh start...")
                result = mem_agent.delete_all_memories()
                if result.get("success"):
                    # If Mem0, wait for async cloud deletion to complete
                    if mem_agent_config.mem_name == "mem_0":
                        time.sleep(3)

                    # Verify deletion succeeded
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

    # Initialize memory store
    store = MemoryStore(
        embedding_config=embedding_config,
        use_faiss=not mem_agent_config.use_mem,  # disable Faiss when using mem_agent
        mem_agent=mem_agent,
        use_mem=mem_agent_config.use_mem
    )

    # ── TRPG mode: use dedicated runner ─────────────────────────────────────────
    if env_config.game_type == "trpg":
        from datetime import datetime
        from memground_agent.agent.trpg_runner import run_trpg

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

    # Initialize retriever
    if agent_config.retriever_type == "vector":
        retriever = VectorRetriever(store)
        print(f"[OK] Using VectorRetriever with Faiss")
    else:
        retriever = KeywordRetrieverTool(store)
        print(f"[OK] Using KeywordRetrieverTool")

    # Create game-specific Prompt builder
    prompt_builder = create_prompt_builder(env_config, llm_config.goal_instruction)
    print(f"[OK] PromptBuilder created for game: {env_config.game_type}")

    # Create game-specific utility class
    game_utils = create_game_utils(env_config)
    print(f"[OK] GameUtils created for game: {env_config.game_type}")

    # Initialize LLM policy (pass game_utils and agent_config for memory management)
    policy = LLMPolicy(llm_config, prompt_builder, memory_store=store, game_utils=game_utils, agent_config=agent_config)
    print(f"[OK] LLMPolicy initialized")

    # Initialize CheckpointManager (if enabled)
    checkpoint_manager = None
    if checkpoint_config.enabled:
        checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_config.dir)
        print(f"[OK] CheckpointManager initialized: {checkpoint_config.dir}")

    # If resuming from checkpoint, load first to obtain logger_session_id
    logger_session_id = None
    start_step = 0
    resume_from = checkpoint_config.resume_from

    if resume_from:
        checkpoint_data = checkpoint_manager.load_checkpoint(Path(resume_from))
        logger_session_id = checkpoint_data.get('logger_session_id')
        start_step = checkpoint_data['step']
        print(f"[OK] Resuming from checkpoint: step={start_step}, logger_session_id={logger_session_id}")

    # Initialize GameLogger (each session has its own subdirectory)
    if resume_from and logger_session_id:
        # Resume mode: use the session directory recorded in the checkpoint
        log_dir = ROOT / "logs" / env_config.game_type / logger_session_id
        logger = GameLogger(log_dir, env_config.game_type, session_id=logger_session_id,
                            resume=True, model=llm_config.model, truncate_after_step=start_step)
    else:
        # New session: pre-generate session_id to determine directory
        _sid = f"{env_config.game_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = ROOT / "logs" / env_config.game_type / _sid
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = GameLogger(log_dir, env_config.game_type, session_id=_sid, model=llm_config.model)
    print(f"[OK] GameLogger initialized: {logger.session_id}")

    # Memory log file placed inside the session directory
    memory_log_file = log_dir / "memory.jsonl"
    store.verbose = agent_config.verbose
    store.memory_log_file = str(memory_log_file)
    print(f"[OK] Memory logging enabled: {memory_log_file}")

    # Update CheckpointManager directory to session subdirectory
    if checkpoint_config.enabled:
        ckpt_dir = log_dir / "checkpoints"
        checkpoint_manager = CheckpointManager(checkpoint_dir=ckpt_dir)
        print(f"[OK] CheckpointManager initialized: {ckpt_dir}")

    # Initialize Agent
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

    # If checkpoint provided, restore state from checkpoint
    if resume_from:
        start_step, _ = agent.load_checkpoint(resume_from)
        print(f"[OK] State restored, continuing from step {start_step + 1}")

    print("\n" + "=" * 70)
    print("Starting Galgame Agent...")
    print("=" * 70 + "\n")

    # Run Agent
    try:
        await agent.run()
    except RuntimeError as e:
        print(f"\n[Agent] Game ended: {e}")
        # Continue to auto QA even when max_steps reached

    # Auto QA evaluation for type_help / no_case_should_remain_unsolved
    if env_config.game_type in ("type_help", "no_case_should_remain_unsolved"):
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

    # Save Faiss index
    if store.use_faiss and store.faiss_manager:
        store.save_faiss_index()
        print("\n[OK] Faiss index saved")


def main():
    """Program entry point"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nAgent interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


if __name__ == "__main__":
    main()
