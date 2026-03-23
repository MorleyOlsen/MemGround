# galagent/common/schemas.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# -------- Dataset Node schema --------
@dataclass(frozen=True)
class Character:
    name: str
    role: str = ""
    number: int = 0
    description: str = ""


@dataclass(frozen=True)
class Memory:
    description: str
    key_info: List[str] = field(default_factory=list)
    location: str = ""
    time: str = ""
    characters: List[Character] = field(default_factory=list)


@dataclass(frozen=True)
class Link:
    target: str
    condition: str


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    memory: Memory
    links: List[Link] = field(default_factory=list)
    is_ending: bool = False


# -------- Runtime schema --------
@dataclass(frozen=True)
class Choice:
    index: int
    text: str


@dataclass(frozen=True)
class Observation:
    node_id: str
    name: str
    text: str
    choices: List[Choice]
    is_ending: bool
    memory: Memory
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    choice_index: int
    rationale: str
    choice_text: str = ""
    recall: list = field(default_factory=list)  # List of filenames related to the decision
