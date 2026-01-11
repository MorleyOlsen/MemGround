# galagent/memory/store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    text: str
    meta: Dict[str, Any]


class MemoryStore:
    def __init__(self):
        self._items: List[MemoryItem] = []

    def reset(self) -> None:
        self._items.clear()

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._items.append(MemoryItem(text=text, meta=meta or {}))

    def recent(self, k: int = 10) -> List[MemoryItem]:
        k = max(0, min(int(k), 500))
        return self._items[-k:]

    @property
    def items(self) -> List[MemoryItem]:
        return self._items
