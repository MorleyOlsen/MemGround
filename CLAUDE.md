# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GameAI is an LLM-based Galgame (visual novel) agent that plays text-based games using memory retrieval and decision-making capabilities. The system supports multiple game types with extensible architecture.

## Running the Project

### Basic Usage
```bash
# Run with default configuration
python galAgent.py

# Override specific settings
python galAgent.py --retriever vector --top_k 5 --max_steps 100 --verbose

# Resume from checkpoint
python galAgent.py  # Set resume_from in config.yaml first
```

### Configuration
- Main config: `config.yaml`
- Key sections: `llm`, `embedding`, `agent`, `checkpoint`, `env`
- Command-line args override config file settings

## Architecture

### Core Components

**Entry Point**: `galAgent.py`
- Loads configuration from `config.yaml`
- Initializes environment, memory store, retriever, policy, and logger
- Orchestrates the agent's game loop

**Agent Loop** (`galagent/agent/runner.py`):
1. Observe current game state
2. Decide if retrieval is needed (via policy)
3. Execute retrieval using game_utils
4. Make action decision based on observations and retrieved context
5. Log decision and take action
6. Save checkpoint at intervals
7. Generate story summary when ending is reached

**Game Environment System** (`galagent/env/`):
- `base_env.py`: BaseGameEnv interface (observe, choose, is_done)
- `env_factory.py`: Factory to create game-specific environments
- Specific envs: `kb_env.py`, `type_help_env.py`, `dust_env.py`
- Each game type has its own config dataclass

**Memory System** (`galagent/memory/`):
- `store.py`: MemoryStore manages conversation history with compression
- `retriever.py`: VectorRetriever (Faiss) and KeywordRetrieverTool
- `FaissManager.py`: Manages Faiss vector index
- Memory compression triggers when conversation exceeds threshold

**Decision Policy** (`galagent/agent/policy.py`):
- LLMPolicy implements three-stage decision making:
  1. Retrieval decision: What information to retrieve
  2. Action decision: Which choice to make
  3. Story summary: Generate analysis at game end
- Uses game-specific prompt builders

**Prompt System** (`env/{game_type}/prompt_builder.py`):
- Each game implements BasePromptBuilder
- Methods: build_system_prompt, build_user_prompt, build_retrieval_decision_prompt
- Prompts are game-specific and localized (ch/en)

**Game Utils** (`env/{game_type}/utils/game_utils.py`):
- Implements BaseGameUtils interface
- Handles game-specific logic: retrieval, context building, file tracking
- Each game type has different utilities (e.g., FileTracker for type_help)

**Checkpoint System** (`galagent/common/checkpoint.py`):
- Saves: environment state, memory state, game_utils state, step number, logger session ID
- Auto-saves at configurable intervals
- Resume via `resume_from` in config.yaml

**Logging System** (`galagent/logger/game_logger.py`):
- Saves to `logs/{game_type}/{session_id}/`
- Logs observations, decisions, retrieval actions, file tracking
- JSON format for easy analysis

### Supported Game Types

**kb**: Knowledge-based dialogue game
- Choice-based navigation through conversation trees

**type_help**: File exploration puzzle game
- Player inputs filenames to unlock new files and progress
- FileTracker manages unlocked/read/attempted files
- Auto-unlocks files mentioned in current file content

**dust**: Mystery/investigation game
- Interrogation-based gameplay with dialogue screenshots
- OCR utilities for processing dialogue images

## Adding a New Game Type

1. **Create game folder**: `env/new_game/` with `prompt_builder.py`, `utils/game_utils.py`, `__init__.py`

2. **Implement environment**: `galagent/env/new_game_env.py`
   - Extend BaseGameEnv
   - Create NewGameConfig dataclass
   - Implement: observe(), choose(), is_done()

3. **Implement PromptBuilder**: `env/new_game/prompt_builder.py`
   - Extend BasePromptBuilder
   - Implement: build_system_prompt(), build_user_prompt(), build_retrieval_decision_prompt()

4. **Implement GameUtils**: `env/new_game/utils/game_utils.py`
   - Extend BaseGameUtils
   - Implement: retrieve_information(), get_game_context()

5. **Register in factory**: Update `galagent/env/env_factory.py`
   - Add cases in create_game_env(), create_prompt_builder(), create_game_utils()
   - Add to get_supported_game_types()

6. **Update config**: Set `env.game_type: "new_game"` in config.yaml

## Key Patterns

### Memory Management
- Agent maintains sliding conversation history
- Compression occurs when exceeding `compression_threshold` turns
- Retrieval results are passed to decision prompt but NOT stored in conversation history
- Only user observations and assistant decisions are stored

### Decision Flow
```
Current Observation → Retrieval Decision (policy)
                   → Execute Retrieval (game_utils)
                   → Action Decision (policy, with retrieval context)
                   → Add to Memory & Execute Action (runner)
```

### Data Structures
- `Observation`: Current game state (text, choices, node_id, name, is_ending, ...)
- `Decision`: Agent's choice (choice_index, rationale, choice_text, recall)
- `Message`: Conversation history (role, content, step, node_id, name)

### Configuration Override
Command-line args > config.yaml values. Use CLI args for quick experiments without editing config.

## Development Notes

- API keys in `config.yaml` should be replaced with environment variables or secure storage
- Faiss is optional; system falls back to keyword retrieval
- `verbose` flag controls console output across memory, retrieval, and logging
- Type Help game: `read_files` (deduplicated) vs `attempted_files` (includes duplicates with success/failure)
- LLM reasoning is captured when using models like deepseek-reasoner
