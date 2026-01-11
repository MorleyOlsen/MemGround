# galagent/env/kb_env.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from galagent.common.schemas import Choice, Observation, Node
from galagent.env.dataset_loader import load_nodes


@dataclass
class KBEnvConfig:
    scenes_path: Path # 场景数据文件路径
    start_node_id: str = "start" # 开始节点
    # 后续可以加入例如是否可以回溯等配置项


class KBEnv:
    """
    Knowledge Base based Galgame environment.
    TODO: implement a PlaywrightEnv with same interface.
    """

    def __init__(self, config: KBEnvConfig):
        self.config = config
        self.nodes: Dict[str, Node] = {}
        self.node_id: str = config.start_node_id
        self.reload()

    # 初始化节点信息（对于从数据库里获取信息来说可以一次性读到所有文字和选项，之后文本量上升后全存到nodes里感觉会存不下）
    def reload(self) -> None:
        self.nodes = load_nodes(self.config.scenes_path)
        # double check
        if self.node_id not in self.nodes:
            self.node_id = self.config.start_node_id if self.config.start_node_id in self.nodes else next(iter(self.nodes.keys()))

    def reset(self, node_id: Optional[str] = None) -> None:
        nid = node_id or self.config.start_node_id
        if nid not in self.nodes:
            raise KeyError(f"Unknown node_id: {nid}")
        self.node_id = nid

    # 获取当前节点的信息
    def observe(self) -> Observation:
        node = self.nodes[self.node_id]
        choices = [Choice(index=i, text=lk.condition) for i, lk in enumerate(node.links)]
        return Observation(
            node_id=node.id,
            name=node.name,
            text=node.memory.description,
            choices=choices,
            is_ending=node.is_ending,  # STRICT: only is_ending ends the agent loop
            memory=node.memory,
            meta={"scenes_path": str(self.config.scenes_path), "node_count": len(self.nodes)},
        )

    # 在agent做出选择，更新当前节点
    def choose(self, choice_index: int) -> str:
        node = self.nodes[self.node_id]

        if node.is_ending:
            raise RuntimeError(f"Already at ending node: {self.node_id}")

        if choice_index < 0 or choice_index >= len(node.links):
            raise IndexError(f"choice_index out of range: {choice_index}")

        target = node.links[choice_index].target
        if target not in self.nodes:
            raise KeyError(f"Target not found: {target}")

        self.node_id = target
        return self.node_id
