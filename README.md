# GameAI

基于 LLM 的 Galgame 智能代理，支持记忆检索和多游戏类型。

## 🚀 快速开始

```bash
# 使用默认配置运行
python galAgent.py

# 使用命令行参数覆盖配置
python galAgent.py --retriever vector --top_k 5 --max_steps 100

# 查看所有选项
python galAgent.py --help
```

## 📊 系统架构

```
GameAI/
├── galAgent.py                 # 主程序入口
├── config.yaml                 # 全局配置文件
├── env/                        # 游戏数据目录
│   ├── kb/                     # KB游戏数据
│   └── type_help/              # Type Help游戏数据
│       └── nodes.json          # 游戏节点定义
├── galagent/                   # 核心代码包
│   ├── agent/                  # 智能体模块
│   │   ├── policy.py           # LLM决策策略
│   │   └── runner.py           # Agent运行器
│   ├── env/                    # 游戏环境模块
│   │   ├── base_env.py         # 游戏环境基类
│   │   ├── base_prompt_builder.py  # Prompt构建器基类
│   │   ├── base_game_utils.py  # 游戏工具基类
│   │   ├── env_factory.py      # 环境工厂（注册游戏）
│   │   ├── dataset_loader.py   # 数据集加载器
│   │   ├── kb_env.py           # KB游戏环境
│   │   └── type_help_env.py    # Type Help游戏环境
│   ├── memory/                 # 记忆系统模块
│   │   ├── store.py            # 记忆存储（集成Faiss）
│   │   ├── retriever.py        # 检索器（关键词/向量）
│   │   ├── FaissManager.py     # Faiss索引管理器
│   │   └── db_tool.py          # 数据库工具
│   ├── common/                 # 公共模块
│   │   ├── schemas.py          # 数据结构定义
│   │   ├── config.py           # 配置加载器
│   │   └── openai_harmony.py   # OpenAI兼容层
│   ├── logger/                 # 日志模块
│   │   └── game_logger.py      # 游戏日志记录器
│   └── tool/                   # 工具模块
│       └── tool.py             # 游戏工具
├── logs/                       # 日志输出目录
└── faiss_data/                 # Faiss索引持久化目录
```

## 🎯 核心特性

### 1. Type Help 游戏环境

**文件追踪系统 (FileTracker)**
```python
file_tracker.unlock_file("01-QU-1-11")           # 解锁文件
file_tracker.attempt_file("04-ST-2-3-5-8")       # 记录尝试
file_tracker.get_unlocked_files()                # 获取已解锁列表
file_tracker.add_pattern("XX-YY-Z")              # 添加命名规则
```

**自动解锁背景节点**
- `Background` - 背景信息
- `message` - 初始消息
- `00-readme` - 游戏说明

**观察信息包含**
- 当前文件名
- 关键信息 (key_info)
- 事件发生地点 (location)
- 出现的人物及编号 (characters)
- 已解锁的文件列表

### 2. 记忆系统

**MemoryStore 集成 FaissManager (store.py)**
- 使用 `ThreadSafeFaissManager` 管理向量索引
- 支持余弦相似度（Inner Product）检索
- 自动归一化 embedding 向量
- 索引持久化到磁盘 (`faiss_data/memory.index`)

**滑动窗口机制**
- 默认保留最近 40 条记忆 (可配置 `max_memory`)
- 超过限制时自动删除最旧记忆
- 同步删除 Faiss 索引中对应的向量

**检索方式 (retriever.py)**
- `VectorRetriever`: 使用 Faiss 进行高效向量检索
- `KeywordRetrieverTool`: 基于关键词的检索
- Faiss 不可用时自动回退到本地计算
- 支持调试日志输出

### 3. LLM 决策策略 (policy.py)

**两阶段决策**
1. **检索决策**: 判断是否需要检索历史记忆
2. **行动决策**: 基于当前观察和检索结果做出选择

**游戏特定 Prompt 构建**
- 使用 `BasePromptBuilder` 接口
- 每个游戏实现自己的 prompt 逻辑
- 支持系统提示词和用户提示词定制

### 4. 配置系统 (config.yaml)

```yaml
llm:
  provider: openai
  api_key: "your-api-key"
 ...

embedding:
  provider: qwen
  api_key: "your-api-key"
 ...
agent:
  max_steps: 20
  retrieve_top_k: 3
 ...
env:
  game_type: "type_help"  # kb or type_help
  scenes_path: "env/type_help/nodes.json"
  start_node_id: "Start"
```

### 5. 游戏日志 (GameLogger)

- 自动记录每个游戏会话
- 保存到 `logs/{game_type}/{session_id}/` 目录
- 包含完整的观察、决策和行动历史
- 支持 JSON 格式导出


#### 6.Checkpoint
# 从最新checkpoint恢复:
`python galagent.py --resume`
这会：
- 自动查找最新的checkpoint文件
- 恢复所有状态
- 从保存的步数继续运行

# 从指定checkpoint恢复
`python galagent.py --resume checkpoints/checkpoint_type_help_long_run_step_500.json`

或者直接通过在代码里修改来实现从指定checkpoint恢复

## 🔧 添加新游戏

只需 3 步即可添加新游戏类型：

### 步骤 1：创建游戏文件夹

```
env/
└── new_game/
    ├── prompt_builder.py      # 实现 BasePromptBuilder
    ├── game_utils.py          # 实现 BaseGameUtils（可选）
    ├── game_data.json         # 游戏数据
    └── __init__.py
```

### 步骤 2：实现游戏环境

```python
# galagent/env/new_game_env.py
from galagent.env.base_env import BaseGameEnv, GameConfig

@dataclass
class NewGameConfig(GameConfig):
    data_path: Path = Path("env/new_game")
    game_type: str = "new_game"
    start_node_id: str = "start"

class NewGameEnv(BaseGameEnv):
    def __init__(self, config: NewGameConfig):
        super().__init__(config)
        self.load_game_data()

    def observe(self) -> Observation:
        # 返回当前游戏状态
        pass

    def choose(self, choice_index: int) -> None:
        # 执行玩家选择
        pass

    def is_done(self) -> bool:
        # 判断游戏是否结束
        pass
```

### 步骤 3：实现 PromptBuilder

```python
# env/new_game/prompt_builder.py
from galagent.env.base_prompt_builder import BasePromptBuilder

class NewGamePromptBuilder(BasePromptBuilder):
    def build_system_prompt(self) -> str:
        return "Your game-specific system prompt"

    def build_user_prompt(self, obs, retrieved_hits, game_context) -> str:
        # 构建游戏特定的 prompt
        return f"""
Current situation: {obs.text}
Retrieved memories: {retrieved_hits}
What do you do next?
"""

    def build_retrieval_decision_prompt(self, obs) -> str:
        # 构建检索决策 prompt
        return f"Should I retrieve memories for: {obs.text}"
```

### 步骤 4：在 env_factory.py 中注册

```python
# galagent/env/env_factory.py

def create_game_env(env_config, root_path):
    if env_config.game_type == "new_game":
        from galagent.env.new_game_env import NewGameEnv, NewGameConfig
        config = NewGameConfig(
            data_path=root_path / "env" / "new_game",
            start_node_id=env_config.start_node_id
        )
        return NewGameEnv(config)
    # ... 其他游戏类型

def create_prompt_builder(env_config, goal_instruction):
    if env_config.game_type == "new_game":
        from env.new_game.prompt_builder import NewGamePromptBuilder
        return NewGamePromptBuilder(goal_instruction)
    # ... 其他游戏类型
```

### 步骤 5：更新配置文件

```yaml
# config.yaml
env:
  game_type: "new_game"
  scenes_path: "env/new_game/game_data.json"
  start_node_id: "start"
```

## 📝 开发说明

### 支持的游戏类型

- **type_help**: 文件解谜游戏，通过输入文件名探索故事
- **kb**: 知识库游戏，基于选择的对话式游戏

### 关键接口

**BaseGameEnv**
- `observe() -> Observation`: 获取当前游戏状态
- `choose(choice_index: int) -> None`: 执行选择
- `is_done() -> bool`: 判断游戏是否结束

**BasePromptBuilder**
- `build_system_prompt() -> str`: 构建系统提示词
- `build_user_prompt(obs, retrieved_hits, game_context) -> str`: 构建用户提示词
- `build_retrieval_decision_prompt(obs) -> str`: 构建检索决策提示词

**BaseGameUtils** (可选)
- 提供游戏特定的工具函数
- 例如：文件名验证、规则检查等

## 🛠️ 技术栈

- **LLM**: OpenAI 兼容接口 (DeepSeek, GPT, etc.)
- **Embedding**: Qwen text-embedding-v4
- **向量检索**: Faiss (Facebook AI Similarity Search)
- **配置管理**: YAML
- **日志系统**: 自定义 GameLogger

