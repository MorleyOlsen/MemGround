# Env Utils - 游戏环境工具

## 📁 目录结构

```
env/
├── utils/
│   ├── file_manager.py    # 文件存储管理工具
│   └── README.md          # 本文档
├── type_help/             # Type Help游戏数据
│   ├── nodes.json         # 节点数据
│   ├── chunks/            # 节点分块（可选）
│   ├── saves/             # 游戏存档
│   └── assets/            # 游戏资源
└── kb/                    # 知识库游戏数据
    └── ...
```

## 🔧 GameFileManager 使用方法

### 基本用法

```python
from pathlib import Path
from env.utils.file_manager import GameFileManager

# 初始化文件管理器
manager = GameFileManager(Path("env/type_help"))

# 加载节点数据
nodes_data = manager.load_nodes("nodes.json")

# 保存节点数据
manager.save_nodes(nodes_data, "nodes_backup.json")
```

### 分块管理（适用于大型游戏）

```python
# 将大文件分割成多个小文件（每个50个节点）
from env.utils.file_manager import split_nodes_into_chunks

split_nodes_into_chunks(
    game_root=Path("env/type_help"),
    source_file="nodes.json",
    nodes_per_chunk=50
)

# 加载特定分块
chunk_data = manager.load_node_chunk("nodes_chunk_001")

# 列出所有分块
chunks = manager.list_chunks()
print(f"Available chunks: {chunks}")

# 合并分块
from env.utils.file_manager import merge_node_chunks

merge_node_chunks(
    game_root=Path("env/type_help"),
    output_file="nodes_merged.json"
)
```

### 游戏存档管理

```python
# 保存游戏状态
game_state = {
    "current_node": "Box0",
    "unlocked_files": ["01-QU-1-11", "02-EN-1-6-7-10"],
    "inventory": ["key", "note"]
}
manager.save_game_state(game_state, "save_001")

# 加载游戏状态
loaded_state = manager.load_game_state("save_001")

# 列出所有存档
saves = manager.list_saves()
print(f"Available saves: {saves}")
```

### 资源文件管理

```python
# 获取资源文件路径
image_path = manager.get_asset_path("images", "background.png")
audio_path = manager.get_asset_path("audio", "bgm.mp3")
data_path = manager.get_asset_path("data", "config.json")

# 确保所有必要目录存在
manager.ensure_directories()
```

## 📊 实用脚本示例

### 1. 分割大型节点文件

```python
# scripts/split_nodes.py
from pathlib import Path
from env.utils.file_manager import split_nodes_into_chunks

if __name__ == "__main__":
    split_nodes_into_chunks(
        game_root=Path("env/type_help"),
        source_file="nodes.json",
        nodes_per_chunk=50
    )
```

### 2. 合并分块文件

```python
# scripts/merge_chunks.py
from pathlib import Path
from env.utils.file_manager import merge_node_chunks

if __name__ == "__main__":
    merge_node_chunks(
        game_root=Path("env/type_help"),
        output_file="nodes_complete.json"
    )
```

### 3. 备份游戏数据

```python
# scripts/backup_game.py
from pathlib import Path
from env.utils.file_manager import GameFileManager
from datetime import datetime

manager = GameFileManager(Path("env/type_help"))

# 创建带时间戳的备份
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nodes = manager.load_nodes()
manager.save_nodes(nodes, f"nodes_backup_{timestamp}.json")
```

## 🎮 游戏数据组织建议

### 小型游戏（<100个节点）
- 使用单个 `nodes.json` 文件
- 直接加载和管理

### 中型游戏（100-500个节点）
- 考虑按章节分块
- 每个章节一个文件
- 使用 `chunks/` 目录管理

### 大型游戏（>500个节点）
- 强烈建议分块管理
- 每50-100个节点一个文件
- 使用懒加载（只加载当前需要的分块）

## 🔍 最佳实践

1. **版本控制**: 将原始 `nodes.json` 纳入版本控制
2. **备份**: 定期备份游戏数据
3. **分块命名**: 使用有意义的分块名称（如 `nodes_chapter1.json`）
4. **资源组织**: 将图片、音频等资源放在 `assets/` 目录下
5. **存档隔离**: 不同游戏的存档分开存储
