# galagent/memory/retriever.py
# TODO: 后续改成真正的向量retriever接口
from __future__ import annotations

import re
from typing import List

from galagent.memory.store import MemoryStore


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower().strip()
    parts = re.split(r"[^\w\u4e00-\u9fff]+", s)
    return [p for p in parts if p]

# 关键词检索 选取top3
class KeywordRetriever:
    def __init__(self, store: MemoryStore):
        self.store = store

    def search(self, query: str, top_k: int = 3) -> List[str]:
        top_k = int(top_k)
        if top_k <= 0:
            return []

        q_tokens = _tokenize(query)[:12]
        if not q_tokens:
            return []

        scored = []
        for idx, item in enumerate(self.store.items):
            text = item.text.lower()
            score = sum(1 for t in q_tokens if t and t in text)
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        hits = [self.store.items[idx].text for _, idx in scored[:top_k]]
        return hits
