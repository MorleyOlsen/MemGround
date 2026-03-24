# memground_agent/env/dataset_loader.py
# This module is responsible for reading JSON data into the internal data structures used by the program
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from memground_agent.common.schemas import Character, Link, Memory, Node


def _character_from_obj(obj: Dict[str, Any]) -> Character:
    name = obj.get("name", "")
    role = obj.get("role", "") or ""
    desc = obj.get("description", "") or ""
    return Character(name=name, role=role, description=desc)


def _memory_from_obj(obj: Dict[str, Any]) -> Memory:
    if not isinstance(obj, dict):
        raise ValueError("Node.memory must be an object")

    desc = obj.get("description", "")
    key_info = obj.get("key_info", obj.get("key info", []))
    location = obj.get("location", "") or ""
    time = obj.get("time", "") or ""
    chars_obj = obj.get("characters", [])
    characters: List[Character] = []
    for c in chars_obj:
        characters.append(_character_from_obj(c))

    return Memory(
        description=desc,
        key_info=key_info,
        location=location,
        time=time,
        characters=characters,
    )


def _link_from_obj(obj: Dict[str, Any]) -> Link:
    target = obj.get("target", "")
    condition = obj.get("condition", "")
    return Link(target=target, condition=condition)


def _node_from_obj(obj: Dict[str, Any]) -> Node:
    node_id = obj.get("id", obj.get("node_id", obj.get("scene_id", "")))
    name = obj.get("name", "")
    is_ending = bool(obj.get("is_ending", False))
    memory = _memory_from_obj(obj.get("memory", {}))

    links_obj = obj.get("links", [])
    links: List[Link] = []
    for l in links_obj:
        links.append(_link_from_obj(l))

    return Node(id=node_id, name=name, memory=memory, links=links, is_ending=is_ending)


def load_nodes(path: Union[str, Path]) -> Dict[str, Node]:
    """
    Load nodes from dataset/scenes.json.

    Supports:
      - list: [ {id, name, memory, links, is_ending}, ... ]
      - dict: { "id": {name, memory, links, is_ending}, ... }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scenes.json not found: {p.resolve()}")

    data = json.loads(p.read_text(encoding="utf-8"))
    nodes: Dict[str, Node] = {}

    if isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                raise ValueError("list format requires dict entries")
            node = _node_from_obj(obj)
            nodes[node.id] = node
    elif isinstance(data, dict):
        for nid, obj in data.items():
            if not isinstance(nid, str) or not nid:
                raise ValueError("dict format keys must be non-empty strings")
            if not isinstance(obj, dict):
                raise ValueError(f"node {nid} must be object")
            obj2 = dict(obj)
            obj2.setdefault("id", nid)
            node = _node_from_obj(obj2)
            nodes[node.id] = node
    else:
        raise ValueError("scenes.json must be a list or a dict")

    if not nodes:
        raise ValueError("No nodes loaded")

    # validate link targets
    for src_id, node in nodes.items():
        for link in node.links:
            if link.target not in nodes:
                raise ValueError(f"Invalid link target: {src_id} -> {link.target} (not found)")

    return nodes
