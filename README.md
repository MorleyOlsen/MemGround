# MemGround

**MemGround** is an LLM agent benchmark built around interactive narrative puzzle games. It evaluates how well a language model can play story-driven games that require **long-term memory**, **logical reasoning**, and **language understanding** — abilities that are difficult to assess with static QA datasets alone.

The agent reads scenes, makes decisions, retrieves relevant memories, and is tested with comprehension questions after (or during) play. All three games push different cognitive limits of the model under evaluation.

---

## Games at a Glance

| Game | Type | Core Challenge | Auto QA |
|------|------|----------------|---------|
| **Type Help** | File-system puzzle | Decode scene clues → type the correct filename to unlock each stage | ✅ runs after game |
| **No Case Should Remain Unsolved** | Mystery investigation | Read fragmented events → reconstruct chronological order | ✅ runs after game |
| **TRPG** | Tabletop RPG comprehension | Follow a multi-chapter story → answer factual questions about characters & plot | ✅ built-in |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Game Types](#game-types)
- [Output Structure](#output-structure)
- [Evaluation](#evaluation)
- [Resuming a Run](#resuming-a-run)
- [Memory and Retrieval](#memory-and-retrieval)
- [Analysis Tools](#analysis-tools)
- [Dataset Structure](#dataset-structure)
- [Project Structure](#project-structure)
- [License](#license)

---

## Installation

**Requirements:** Python 3.9+

```bash
https://github.com/MorleyOlsen/MemGround.git
cd MemGround
pip install -r requirements.txt
```

For **vector-based memory retrieval** (recommended for long runs), also install FAISS:

```bash
pip install faiss-cpu
```

<details>
<summary>Full <code>requirements.txt</code></summary>

```
openai>=1.0.0
pyyaml>=6.0
numpy>=1.24.0
requests>=2.28.0

# Optional – for vector retrieval:
# faiss-cpu>=1.7.0
```

</details>

---

## Quick Start

### Step 1 — Configure API keys

Open `config.yaml` and fill in your credentials:

```yaml
llm:
  api_key: "YOUR_API_KEY_HERE"
  base_url: "https://api.openai.com/v1"   # or any OpenAI-compatible endpoint
  model: "gpt-4o"                         # model to evaluate
  temperature: 0.2
  max_output_tokens: 600

judge_llm:
  api_key: "YOUR_JUDGE_API_KEY_HERE"      # can be the same key
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"
  temperature: 0.0
  max_output_tokens: 1024

embedding:
  api_key: "YOUR_EMBEDDING_API_KEY_HERE"
  base_url: "https://api.openai.com/v1"
  model: "text-embedding-3-small"
  dim: 1536
  use_real: true   # set false to skip embedding (keyword retrieval only)
```

> **Security note:** `config.yaml` contains secrets. Add it to `.gitignore` or use environment variables before committing.

### Step 2 — Select a game

Edit `config.yaml` and uncomment **exactly one** `env:` block (see [Game Types](#game-types)). For example, to play Type Help:

```yaml
env:
  game_type: "type_help"
  scenes_path: "dataset/type_help-en/nodes.json"
  start_node_id: "Background"
  test_language: "en"
  enable_hint: true
  hint_failure_threshold: 50
  provide_naming_rules: false
```

### Step 3 — Run the agent

```bash
python memground_agent.py
```

The agent plays the selected game, logs every step, saves checkpoints, and automatically runs QA evaluation when the game ends.

**CLI overrides** (all are optional):

```bash
python memground_agent.py \
  --config config.yaml \          # path to config file (default: config.yaml)
  --retriever vector \            # override retriever: "keyword" or "vector"
  --top_k 5 \                     # override number of memories retrieved per step
  --max_steps 200 \               # override max game steps
  --verbose \                     # enable verbose output
  --story "Cold_Wind_Howling"     # TRPG only: override story name
```

---

## Configuration Reference

All settings live in `config.yaml`. The table below documents every supported field.

### `llm` — Agent LLM

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | string | `"openai"` | API provider (currently only `"openai"`-compatible) |
| `api_key` | string | — | API key |
| `base_url` | string | — | Endpoint URL (e.g. `https://api.openai.com/v1`) |
| `model` | string | — | Model ID to evaluate (e.g. `gpt-4o`, `qwen3-32b`) |
| `temperature` | float | `0.2` | Sampling temperature for the agent |
| `max_output_tokens` | int | `600` | Max tokens per LLM response |
| `goal_instruction` | string | *(built-in)* | System-level instruction prepended to every prompt |

### `judge_llm` — Scoring LLM

Same fields as `llm`. Used exclusively during QA evaluation to score the agent's answers. Can point to a different (stronger) model than the agent LLM.

### `embedding` — Vector Memory

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | string | `"openai"` | Embedding provider |
| `api_key` | string | — | API key |
| `base_url` | string | — | Endpoint URL |
| `model` | string | — | Embedding model name |
| `dim` | int | `1024` | Embedding dimension |
| `use_real` | bool | `false` | `true` = call API in real-time; `false` = use cached embeddings |

### `agent` — Agent Behavior

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_steps` | int | `500` | Maximum game steps before forced termination |
| `retrieve_top_k` | int | `3` | Number of memories retrieved per decision step |
| `retriever_type` | string | `"vector"` | `"keyword"` (BM25, no extra deps) or `"vector"` (FAISS) |
| `verbose` | bool | `false` | Print detailed per-step output |
| `max_context_tokens` | int | `65000` | Token budget for the conversation window |
| `enable_compression` | bool | `true` | Compress oldest messages when context exceeds budget |
| `compression_threshold` | int | `100` | Trigger compression after N turns in the context |
| `compression_count` | int | `80` | Number of oldest turns to compress each time |

### `mem_agent` — External Memory (Optional)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_mem` | bool | `false` | Enable external memory backend |
| `mem_name` | string | — | Backend: `"a_mem"` (local A-mem) or `"mem0"` (cloud Mem0) |
| `clear_on_start` | bool | `false` | Clear memory store at session start |

### `checkpoint` — Fault Tolerance

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable checkpoint saving |
| `interval` | int | `20` | Save a checkpoint every N steps |
| `dir` | string | `"checkpoints"` | *(legacy, now ignored)* Checkpoints are saved to the session log directory |
| `resume_from` | string\|null | `null` | Path to a checkpoint file to resume from (see [Resuming a Run](#resuming-a-run)) |

### `env` — Game Environment

Only **one** `env:` block should be active at a time.

See [Game Types](#game-types) for per-game field documentation.

---

## Game Types

### Type Help

A file-system puzzle game. The agent navigates branching scenes and must infer the correct filename to type at each stage. Unlocking a file may reveal further clues or trigger new branches.

```yaml
env:
  game_type: "type_help"
  scenes_path: "dataset/type_help-en/nodes.json"
  start_node_id: "Background"
  test_language: "en"
  enable_hint: true               # After hint_failure_threshold consecutive failures,
  hint_failure_threshold: 50      #   the game auto-hints the correct filename
  provide_naming_rules: false     # Set true to expose file-naming conventions in the prompt
```

**Evaluation:** After the game ends, QA runs automatically. The agent answers questions about the story using its accumulated memory. Results are saved to `logs/type_help/<session_id>/`.

---

### No Case Should Remain Unsolved

A mystery investigation game. The agent encounters event fragments out of order and must reconstruct the correct chronological sequence to solve the case. Scoring rewards both correct ordering and correct causal inference.

```yaml
env:
  game_type: "no_case_should_remain_unsolved"
  scenes_path: "dataset/no_case_should_remain_unsolved-en/data/nodes.json"
  start_node_id: "start"
  test_language: "en"
  show_order_judgements_history: false   # Set true to include all past ordering judgements in the prompt
```

**Evaluation:** QA runs automatically at game end. Results are saved to `logs/no_case_should_remain_unsolved/<session_id>/`.

---

### TRPG

A tabletop RPG comprehension task. The agent reads through a multi-chapter story and then answers factual questions about characters, events, and plot details. Evaluation is built into the runner and covers six dimensions.

```yaml
env:
  game_type: "trpg"
  trpg:
    story_name: "Terror_on_the_Orient_Express"  # see available stories below
    data_dir: "dataset/trpg_en/data"
    qa_dir: "dataset/trpg_en/qa"
    test_language: "en"
```

**Available stories:**

| Story | Genre |
|-------|-------|
| `Terror_on_the_Orient_Express` | Thriller |
| `Cold_Wind_Howling` | Suspense |
| `Spring_Snow_Incident` | Drama |

**Evaluation:** QA runs automatically as the final phase of the game. Results are saved to `logs/trpg/<session_id>/`.

---

## Output Structure

All output goes under `logs/`. Each game session gets its own subdirectory:

```
logs/
├── type_help/
│   └── type_help_20260224_210044/      ← one directory per session
│       ├── game_log.json               ← full step-by-step action log
│       ├── game_log_summary.txt        ← human-readable play transcript
│       ├── memory.jsonl                ← memory compression history
│       ├── checkpoints/
│       │   ├── step_000020.json        ← checkpoint at step 20
│       │   ├── step_000040.json
│       │   └── ...
│       ├── results.json                ← QA evaluation scores
│       └── summary.txt                 ← QA evaluation summary
│
├── no_case_should_remain_unsolved/
│   └── no_case_should_remain_unsolved_20260301_143012/
│       └── ...                         ← same structure as type_help
│
└── trpg/
    └── TotOE__gpt4o__amem_20260302/
        ├── checkpoints/
        │   ├── trpg_ckpt_reading_c001.json
        │   └── trpg_ckpt_qa_010.json
        ├── compressions.jsonl          ← story compression log
        ├── results.json
        └── summary.txt
```

### `game_log.json` schema (type_help / no_case_should_remain_unsolved)

```jsonc
{
  "session_id": "type_help_20260224_210044",
  "game_type": "type_help",
  "model": "gpt-4o",
  "start_time": "2026-02-24T21:00:44",
  "end_time": "2026-03-01T02:29:07",
  "total_steps": 471,
  "reached_ending": false,
  "ending_node": "03-KI-10-11",
  "actions": [
    {
      "step": 0,
      "timestamp": "...",
      "node_id": "00-readme",
      "scene_text": "...",
      "choices": { "text": "...", "decision_rationale": "..." },
      "file_retrieval": { "need_retrieval": true, "opened_files": [...], "reason": "..." },
      "unlocked_files": [...],   // type_help only
      "failed_files": [...],     // type_help only
      // no_case_should_remain_unsolved-specific fields: action_type, action_params, score, keys, ...
    }
  ]
}
```

### `results.json` schema (all games)

```jsonc
{
  "session_id": "...",
  "game_type": "type_help",
  "model": "gpt-4o",
  "results": [
    {
      "question": "...",
      "answer": "...",
      "judge_scores": {
        "Acc": 3,     // Accuracy (0–3)
        "Cit": 1,     // Citation grounding (0–1)
        "Inst": 1,    // Instruction following (0–1)
        "Read": 2     // Reading coverage (0–2)
      }
    }
  ],
  "aggregate": { ... }
}
```

---

## Evaluation

### Automatic QA (all games)

All three games run QA automatically when they end — no manual step required:

- **Type Help / No Case Should Remain Unsolved**: QA fires at the end of `memground_agent.py` using the live memory store. No checkpoint reload needed.
- **TRPG**: QA is the final phase of `trpg_runner.py`.

Results always appear in `logs/<game_type>/<session_id>/results.json`.

---

### Standalone QA (`scripts/run_qa.py`)

If you want to re-run QA against a previously saved checkpoint (e.g. after the run was interrupted before QA completed):

```bash
# 1. Edit GAME_TYPE and CHECKPOINT_PATH at the top of scripts/run_qa.py:
#
#   GAME_TYPE = "type_help"
#   CHECKPOINT_PATH = {
#       "type_help":                       "logs/type_help/type_help_20260224_210044/checkpoints/step_000199.json",
#       "no_case_should_remain_unsolved":  "logs/no_case_should_remain_unsolved/no_case_should_remain_unsolved_20260301_143012/checkpoints/step_000599.json",
#   }

python scripts/run_qa.py
```

This loads the agent's memory from the checkpoint, answers all QA questions, and saves `results.json` + `summary.txt` to the session directory.

---

### Re-judge with a Different Model (`scripts/rejudge.py`)

To re-score an existing `results.json` using a different judge LLM without re-playing the game:

```bash
# Edit _TARGET at the top of scripts/rejudge.py to point to your results file, then:
python scripts/rejudge.py
```

This overwrites the judge scores in-place and regenerates `summary.txt`.

---

### Six-Dimensional Summary (`scripts/summarize.py`)

Generate a scoring table from one or more `results.json` files:

```bash
# Single run
python scripts/summarize.py logs/trpg/TotOE__gpt4o__amem_20260302/results.json

# Compare multiple runs side-by-side
python scripts/summarize.py logs/trpg/*/results.json
```

**Scoring dimensions and weights:**

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Acc** | 3.0 | Answer accuracy |
| **Read.** | 3.0 | Evidence / reading coverage rate |
| **Comp.** | 1.5 | Comparative reasoning accuracy |
| **Depth** | 1.5 | Average of D1 (Surface) / D2 (Character) / D3 (Cross-section) |
| **Inst.** | 0.5 | Format / instruction-following pass rate |
| **Cit.** | 0.5 | Evidence grounding (HIGH citation fraction) |

Overall score = weighted average (max 100).

**Grade scale:** A+ (≥ 90) · A (≥ 80) · B+ (≥ 70) · B (≥ 60) · C+ (≥ 50) · C (≥ 40) · D (≥ 30) · F (< 30)

Output is printed to the terminal and saved as `summary.txt` next to the input file.

---

### Export to Excel (`scripts/export_results_excel.py`)

Export all results to color-coded Excel workbooks for comparison:

```bash
python scripts/export_results_excel.py
# Outputs: logs/type_help_results.xlsx
#          logs/no_case_should_remain_unsolved_results.xlsx
```

---

## Resuming a Run

If a run is interrupted, resume from the last saved checkpoint:

```yaml
# config.yaml
checkpoint:
  enabled: true
  resume_from: "logs/type_help/type_help_20260224_210044/checkpoints/step_000200.json"
```

The agent will restore the environment, memory store, and logger state and continue from step 200. The existing `game_log.json` is extended in-place.

**Finding your latest checkpoint:**

```bash
ls -t logs/type_help/<session_id>/checkpoints/ | head -1
```

---

## Memory and Retrieval

The agent maintains a rolling conversation memory. At each step it retrieves the most relevant past observations to include in the prompt. Two backends are supported:

| Mode | Config value | Dependencies | Notes |
|------|--------------|-------------|-------|
| Keyword (BM25-style) | `retriever_type: "keyword"` | None | Fast, no API calls; works offline |
| Vector (dense) | `retriever_type: "vector"` | `faiss-cpu` + embedding API | Higher recall; requires embedding key |

Set `embedding.use_real: false` to skip embedding API calls (e.g. during development). The agent falls back to keyword retrieval automatically.

**Memory compression** keeps the context window manageable during long runs. When the conversation exceeds `compression_threshold` turns, the oldest `compression_count` turns are summarised and removed. The raw history is preserved in `memory.jsonl` for offline analysis.

**Optional external memory backends** (experimental):

- **A-mem** (local, no cloud dependency): set `mem_agent.mem_name: "a_mem"` and configure `amem_*` fields.
- **Mem0** (cloud): set `mem_agent.mem_name: "mem0"` and provide a Mem0 API key.

---

## Analysis Tools

`scripts/analyze_logs.py` provides post-hoc analysis of completed game logs. It is primarily designed for **Type Help** runs.

```bash
python scripts/analyze_logs.py
```

An interactive menu offers four operations:

| # | Operation | Output |
|---|-----------|--------|
| 1 | **Extract step data** | Structured JSON of per-step file unlock sequences from raw game logs |
| 2 | **Gantt chart** | Timeline chart of file unlock events across steps |
| 3 | **Recall adjacency matrix** | Directed graph of `file A → unlocks file B` relationships observed during play |
| 4 | **DAG similarity** | Compares the model's recall graph against the human-annotated reference graph in `dataset/type_help-en/human_adj.csv`; reports precision, recall, F1 |

**Reference data** used by operations 3 and 4:

| File | Description |
|------|-------------|
| `dataset/type_help-en/qa/human_adj.csv` | Human-annotated ground-truth adjacency matrix |
| `dataset/type_help-en/qa/node_id_map.csv` | Integer node ID ↔ node name mapping |

All output is written to `analysis/`.

---

## Dataset Structure

```
dataset/
├── type_help-en/
│   ├── data/
│   │   ├── nodes.json                    # Scene graph (main game data)
│   │   ├── name.json                     # Character name aliases
│   │   └── all_links_with_recall.json    # Ground-truth file unlock relationships
│   ├── qa/
│   │   ├── type_help_qa_eval.json        # QA evaluation questions
│   │   ├── human_adj.csv                 # Human-annotated adjacency matrix
│   │   ├── node_id_map.csv               # Node ID ↔ name mapping (for analysis)
│   │   └── stories/type_help_en/         # Story text assets (used by QA)
│
├── no_case_should_remain_unsolved-en/
│   ├── data/
│   │   ├── nodes.json                    # Scene graph
│   │   ├── dialogue-en.json              # NPC dialogue texts
│   │   └── order_gt-en.json              # Ground-truth event ordering
│   ├── qa/
│   │   ├── no_case_should_remain_unsolved_qa_eval.json  # QA evaluation questions
│   │   └── stories/no_case_should_remain_unsolved_en/  # Story text assets (used by QA)
│
└── trpg_en/
    ├── data/
    │   └── <story_name>/                 # One subdirectory per story
    │       └── *.json                    # Story scene nodes
    └── qa/
        └── <story_name>_qa.json          # QA evaluation pairs per story
```

All data files use JSON format. No preprocessing is required.

---

## Project Structure

```
MemGround/
├── memground_agent.py            # Main entry point (type_help / no_case_should_remain_unsolved / trpg)
├── config.yaml                   # All configuration (API keys, game, agent settings)
├── requirements.txt
│
├── memground_agent/              # Core agent library
│   ├── agent/
│   │   ├── runner.py             # MemGroundAgent — main step loop (type_help / no_case)
│   │   ├── policy.py             # LLMPolicy — prompt construction + LLM call
│   │   └── trpg_runner.py        # TRPGRunner — TRPG-specific two-phase runner
│   ├── env/
│   │   ├── env_factory.py        # Creates the correct env/utils/prompt_builder
│   │   ├── base_env.py           # Abstract game environment interface
│   │   ├── type_help_env.py      # Type Help environment
│   │   ├── no_case_should_remain_unsolved_env.py  # No Case environment
│   │   └── trpg_env.py           # TRPG environment
│   ├── memory/
│   │   ├── store.py              # MemoryStore — conversation history management
│   │   ├── retriever.py          # KeywordRetriever + VectorRetriever
│   │   └── FaissManager.py       # FAISS vector index wrapper
│   ├── common/
│   │   ├── config.py             # Typed config dataclasses + ConfigLoader
│   │   └── checkpoint.py         # CheckpointManager — save / load state
│   └── logger/
│       └── game_logger.py        # GameLogger — structured per-step logging
│
├── env/                          # Game-specific prompt builders and utilities
│   ├── type_help/
│   ├── no_case_should_remain_unsolved/
│   └── trpg/
│
├── scripts/                      # Evaluation and analysis scripts
│   ├── run_qa.py                 # QA pipeline (standalone + inline mode)
│   ├── rejudge.py                # Re-score results with a different judge LLM
│   ├── summarize.py              # Six-dimensional summary table
│   ├── export_results_excel.py   # Export results to Excel
│   └── analyze_logs.py           # Post-hoc log analysis (Type Help)
│
├── dataset/                      # Game datasets (see Dataset Structure)
├── logs/                         # Runtime output (git-ignored)
└── faiss_data/                   # FAISS index cache (git-ignored)
```

---

## License

This project is released under the [MIT License](LICENSE).
