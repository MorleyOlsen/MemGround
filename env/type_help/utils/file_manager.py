# env/utils/file_manager.py
"""游戏文件存储和读取管理工具"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class GameFileManager:
    """游戏文件管理器

    统一管理游戏数据文件的读取和写入
    """

    def __init__(self, game_root: Path):
        """初始化文件管理器

        Args:
            game_root: 游戏根目录（例如 env/type_help）
        """
        self.game_root = Path(game_root)
        if not self.game_root.exists():
            raise FileNotFoundError(f"Game root not found: {self.game_root}")

    def load_nodes(self, filename: str = "nodes.json") -> Dict[str, Any]:
        """加载节点数据

        Args:
            filename: 节点文件名，默认为 nodes.json

        Returns:
            节点数据字典
        """
        node_file = self.game_root / filename
        if not node_file.exists():
            raise FileNotFoundError(f"Node file not found: {node_file}")

        with open(node_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_nodes(self, data: Dict[str, Any], filename: str = "nodes.json") -> None:
        """保存节点数据

        Args:
            data: 要保存的节点数据
            filename: 保存的文件名
        """
        node_file = self.game_root / filename
        with open(node_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_node_chunk(self, chunk_name: str) -> Dict[str, Any]:
        """加载节点分块文件

        对于大型游戏，可以将节点分成多个文件
        例如: nodes_chapter1.json, nodes_chapter2.json

        Args:
            chunk_name: 分块文件名（不含.json后缀）

        Returns:
            节点数据
        """
        chunk_file = self.game_root / "chunks" / f"{chunk_name}.json"
        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

        with open(chunk_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_chunks(self) -> List[str]:
        """列出所有可用的节点分块

        Returns:
            分块文件名列表（不含.json后缀）
        """
        chunks_dir = self.game_root / "chunks"
        if not chunks_dir.exists():
            return []

        return [
            f.stem for f in chunks_dir.glob("*.json")
        ]

    def save_game_state(self, state: Dict[str, Any], save_name: str = "autosave") -> None:
        """保存游戏状态

        Args:
            state: 游戏状态数据
            save_name: 存档名称
        """
        saves_dir = self.game_root / "saves"
        saves_dir.mkdir(exist_ok=True)

        save_file = saves_dir / f"{save_name}.json"
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_game_state(self, save_name: str = "autosave") -> Optional[Dict[str, Any]]:
        """加载游戏状态

        Args:
            save_name: 存档名称

        Returns:
            游戏状态数据，如果不存在返回None
        """
        save_file = self.game_root / "saves" / f"{save_name}.json"
        if not save_file.exists():
            return None

        with open(save_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_saves(self) -> List[str]:
        """列出所有存档

        Returns:
            存档名称列表
        """
        saves_dir = self.game_root / "saves"
        if not saves_dir.exists():
            return []

        return [f.stem for f in saves_dir.glob("*.json")]

    def get_asset_path(self, asset_type: str, asset_name: str) -> Path:
        """获取资源文件路径

        Args:
            asset_type: 资源类型（images, audio, data等）
            asset_name: 资源文件名

        Returns:
            资源文件的完整路径
        """
        asset_path = self.game_root / "assets" / asset_type / asset_name
        return asset_path

    def ensure_directories(self) -> None:
        """确保必要的目录存在"""
        directories = [
            self.game_root / "chunks",
            self.game_root / "saves",
            self.game_root / "assets" / "images",
            self.game_root / "assets" / "audio",
            self.game_root / "assets" / "data"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def merge_node_chunks(game_root: Path, output_file: str = "nodes_merged.json") -> None:
    """合并所有节点分块文件

    Args:
        game_root: 游戏根目录
        output_file: 输出文件名
    """
    manager = GameFileManager(game_root)
    chunks = manager.list_chunks()

    if not chunks:
        print(f"No chunks found in {game_root / 'chunks'}")
        return

    merged_nodes = []
    for chunk_name in sorted(chunks):
        chunk_data = manager.load_node_chunk(chunk_name)
        nodes = chunk_data.get("game", {}).get("nodes", [])
        merged_nodes.extend(nodes)
        print(f"Loaded {len(nodes)} nodes from {chunk_name}")

    # 保存合并后的数据
    merged_data = {"game": {"nodes": merged_nodes}}
    manager.save_nodes(merged_data, output_file)

    print(f"\nMerged {len(merged_nodes)} nodes into {output_file}")


def split_nodes_into_chunks(
    game_root: Path,
    source_file: str = "nodes.json",
    nodes_per_chunk: int = 50
) -> None:
    """将大型节点文件分割成多个小文件

    Args:
        game_root: 游戏根目录
        source_file: 源节点文件
        nodes_per_chunk: 每个分块的节点数量
    """
    manager = GameFileManager(game_root)
    data = manager.load_nodes(source_file)
    nodes = data.get("game", {}).get("nodes", [])

    chunks_dir = game_root / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    # 分割节点
    for i in range(0, len(nodes), nodes_per_chunk):
        chunk_nodes = nodes[i:i + nodes_per_chunk]
        chunk_num = i // nodes_per_chunk + 1
        chunk_name = f"nodes_chunk_{chunk_num:03d}"

        chunk_data = {"game": {"nodes": chunk_nodes}}
        chunk_file = chunks_dir / f"{chunk_name}.json"

        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        print(f"Created {chunk_name} with {len(chunk_nodes)} nodes")

    print(f"\nSplit {len(nodes)} nodes into {(len(nodes) + nodes_per_chunk - 1) // nodes_per_chunk} chunks")
